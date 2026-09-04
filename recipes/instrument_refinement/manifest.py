"""学習用 manifest（CSV）の作成。

手元のデータセットは置き場所も命名規則もばらばらなので、JSON の設定ファイルで
「どこに何があるか」を宣言し、ここで 1 枚の CSV に正規化する。1 行 = 1 音源
（1 ステムの wav と、その採譜 MIDI のペア）で、楽器の正解ラベルが付く。

対応するデータセットの形は 2 種類:

- ``class_manifests`` … 楽器クラスごとにフォルダ・manifest が分かれている形。
  クラス名がフォルダ名から決まるので、ラベルは 1 つに確定する。
- ``flat`` … wav と MIDI が平置きされている形。ラベルは npz か MIDI から推定する。

重要な概念:

- ``split_group_id`` … train/validation/test を分ける単位。同じ曲のピッチ変更版などが
  train と test に分かれて「実質カンニング」になるのを防ぐため、曲単位でまとめる。
- ``atomic_group_id`` … 学習時に分割してはいけない単位。例えばリードボーカルとコーラスは
  必ず一緒に混ぜないと、片方だけ聴かせて「これは melody」と教えることになってしまう。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import soundfile as sf

from instrument_agnostic_amt.amt.inference.instruments import STEM_INSTRUMENT_CLASSES
from instrument_agnostic_amt.taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_instrument_class_id_by_name,
)
from instrument_agnostic_amt.instrument_refinement.data.labels import (
    family_ids_for_class_ids,
)
from instrument_agnostic_amt.instrument_refinement.data.midi import (
    load_refinement_note_table,
)

MANIFEST_FIELDS = (
    "source_id",
    "dataset",
    "song_id",
    "split_group_id",
    "atomic_group_id",
    "stem_name",
    "track_type",
    "audio_path",
    "midi_path",
    "duration_seconds",
    "target_class_names",
    "target_class_ids",
    "target_family_ids",
    "note_target_mode",
    "split",
)


@dataclass(frozen=True)
class ManifestSourceRecord:
    """manifest の 1 行（= 1 音源）。CSV の列と 1 対 1 で対応する。"""

    source_id: str
    dataset: str
    song_id: str
    split_group_id: str
    atomic_group_id: str
    stem_name: str
    track_type: str
    audio_path: Path
    midi_path: Path
    duration_seconds: float
    target_class_ids: tuple[int, ...]
    note_target_mode: str


def _stable_fraction(value: str, seed: int) -> float:
    """文字列から [0, 1) の値を決定的に作る。乱数の代わりに使い、再実行でも同じ結果にする。"""
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


_STEM_GROUP_BY_CLASS = {
    class_name: group
    for group, class_names in STEM_INSTRUMENT_CLASSES.items()
    if group != "drums"
    for class_name in class_names
}


def _stem_group_for_class_ids(class_ids: Iterable[int]) -> str | None:
    """その音源が推論時のどのステムに現れるかを判定する。またがる場合は None。

    None を返した音源は学習に使わない。例えばギターとストリングスが混在した音源は、
    分離器の guitar/other のどちらに出るか決められず、教師として使えないため。
    """
    groups = {
        _STEM_GROUP_BY_CLASS.get(INSTRUMENT_CLASSES[class_id])
        for class_id in class_ids
        if INSTRUMENT_CLASSES[class_id] != "drums"
    }
    groups.discard(None)
    return next(iter(groups)) if len(groups) == 1 else None


def _midi_class_ids(path: Path) -> tuple[int, ...]:
    """MIDI の program から、その音源に含まれる楽器クラスを拾う（ドラムは除く）。"""
    drum_id = get_instrument_class_id_by_name("drums")
    notes = load_refinement_note_table(path)
    return tuple(sorted(class_id for class_id in set(map(int, notes.prior_class_id.tolist())) if class_id != drum_id))


def _npz_class_ids(path: Path, remap: Mapping[int, int]) -> tuple[int, ...] | None:
    """前処理済み npz からラベルを拾う。MIDI の program より信頼できる場合に使う。"""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "note_instrument" not in data:
            return None
        values = set(map(int, np.unique(data["note_instrument"]).tolist()))
    drum_id = get_instrument_class_id_by_name("drums")
    return tuple(sorted({int(remap.get(class_id, class_id)) for class_id in values} - {drum_id}))


def _duration_lookup(manifest_path: Path | None) -> dict[str, float]:
    """既存 manifest から長さを引く表を作る。wav を開かずに済ませるための最適化。"""
    if manifest_path is None or not manifest_path.is_file():
        return {}
    result: dict[str, float] = {}
    with manifest_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as file:
        for row in csv.DictReader(file):
            raw_path = str(row.get("wav_path", "")).replace("\\", "/")
            duration_ms = row.get("duration_ms")
            if raw_path and duration_ms:
                result[Path(raw_path).stem] = float(duration_ms) / 1000.0
    return result


def _audio_duration(path: Path, lookup: Mapping[str, float]) -> float:
    duration = lookup.get(path.stem)
    if duration is not None and duration > 0.0:
        return float(duration)
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _split_augmentation(stem: str, spec: Mapping[str, Any]) -> tuple[str, str]:
    """ファイル名を「元の曲名」と「augmentation の種類」に分ける。

    例: "song_a_pitch_2" -> ("song_a", "pitch_2")。augmentation 違いを同じ曲として
    扱い、train/test をまたがせないために使う。
    """
    pattern = str(spec.get("augmentation_pattern", r"_(?:stretch|pitch)_"))
    match = re.search(pattern, stem)
    if match is None:
        return stem, "original"
    return stem[: match.start()], stem[match.start() :].lstrip("_")


def _apply_song_rules(value: str, spec: Mapping[str, Any]) -> str:
    """データセットごとの命名規則を吸収して、曲を識別するキーへ正規化する。

    同じ曲が別表記（全角/半角、接尾辞違いなど）で入っていると別曲と見なされ、
    train/test に分かれてしまうので、設定に応じて表記ゆれを潰す。
    """
    mode = str(spec.get("song_key_mode", "base"))
    if mode == "prefix_before_last_double_underscore":
        if "__" in value:
            value = value.rsplit("__", 1)[0]
    elif mode != "base":
        raise ValueError(f"Unknown song_key_mode: {mode}")
    for rule in spec.get("song_key_rules", []):
        value = re.sub(str(rule["pattern"]), str(rule.get("replacement", "")), value)
    if bool(spec.get("normalize_nfkc", False)):
        value = unicodedata.normalize("NFKC", value).casefold()
    if bool(spec.get("alnum_only", False)):
        value = "".join(character for character in value if character.isalnum())
    return value or "unknown"


def _record_identity(
    dataset: str,
    audio_stem: str,
    spec: Mapping[str, Any],
) -> tuple[str, str, str, str]:
    """1 音源に対する 4 種類の ID を作る。

    Returns:
        (source_id, song_id, split_group_id, atomic_group_id)。
        source_id は音源そのもの、song_id は augmentation 込みの曲、
        split_group_id は augmentation を無視した曲（分割の単位）、
        atomic_group_id は必ず一緒に扱う単位。
    """
    base_stem, augmentation = _split_augmentation(audio_stem, spec)
    song_key = _apply_song_rules(base_stem, spec)
    split_group_id = f"{dataset}:{song_key}"
    song_id = f"{split_group_id}:{augmentation}"
    source_id = f"{dataset}:{audio_stem}"
    atomic_mode = str(spec.get("atomic_mode", "source"))
    if atomic_mode == "source":
        atomic_group_id = source_id
    elif atomic_mode == "song_variant":
        atomic_group_id = song_id
    else:
        raise ValueError(f"Unknown atomic_mode: {atomic_mode}")
    return source_id, song_id, split_group_id, atomic_group_id


def _class_manifest_records(
    spec: Mapping[str, Any], config_root: Path
) -> tuple[list[ManifestSourceRecord], Counter[str]]:
    """楽器クラスごとに manifest が分かれているデータセットを読む。

    manifest のファイル名（例: ``piano_stem_manifest.csv``）からクラス名を取り出すので、
    ラベルは 1 つに確定する。除外できなかった音源の理由は Counter に積んで返す。
    """
    dataset = str(spec["name"])
    root = _resolve_path(str(spec["root"]), config_root)
    manifest_glob = str(spec.get("manifest_glob", "manifests/*.csv"))
    class_pattern = re.compile(str(spec["class_name_regex"]))
    audio_column = str(spec.get("audio_column", "wav_path"))
    stem_column = str(spec.get("stem_column", "stem_name"))
    midi_template = str(spec["midi_dir_template"])
    excluded = {str(value) for value in spec.get("exclude_classes", [])}
    excluded.add("drums")  # ドラムはこのモデルの対象外なので常に除外する。
    records: list[ManifestSourceRecord] = []
    skipped: Counter[str] = Counter()
    for manifest in sorted(root.glob(manifest_glob)):
        match = class_pattern.fullmatch(manifest.stem)
        if match is None or "class" not in match.groupdict():
            skipped["unmatched_manifest_name"] += 1
            continue
        class_name = str(match.group("class"))
        if class_name in excluded:
            skipped["excluded_class"] += 1
            continue
        class_id = get_instrument_class_id_by_name(class_name)
        stem_group = _stem_group_for_class_ids((class_id,))
        if stem_group is None:
            skipped["unsupported_class"] += 1
            continue
        midi_root = root / midi_template.format(class_name=class_name)
        with manifest.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                audio = (manifest.parent / row[audio_column]).resolve()
                midi = (midi_root / f"{audio.stem}.mid").resolve()
                if not audio.is_file() or not midi.is_file():
                    skipped["missing_audio_or_midi"] += 1
                    continue
                source_id, song_id, split_group_id, atomic_group_id = _record_identity(
                    dataset, str(row.get(stem_column) or audio.stem), spec
                )
                records.append(
                    ManifestSourceRecord(
                        source_id=source_id,
                        dataset=dataset,
                        song_id=song_id,
                        split_group_id=split_group_id,
                        atomic_group_id=atomic_group_id,
                        stem_name=stem_group,
                        track_type=class_name,
                        audio_path=audio,
                        midi_path=midi,
                        duration_seconds=(float(row.get("duration_ms") or 0.0) / 1000.0),
                        target_class_ids=(class_id,),
                        note_target_mode="source",
                    )
                )
    return records, skipped


def _limit_augmentation_variants(
    audio_paths: list[Path],
    *,
    dataset: str,
    root: Path,
    spec: Mapping[str, Any],
    skipped: Counter[str],
) -> list[Path]:
    """1 曲あたりの augmentation 本数を制限する（元データは必ず残す）。

    どれを残すかはハッシュ順で決めるので、実行ごとに変わらない。
    """
    raw_limit = spec.get("max_augmentation_variants_per_group")
    if raw_limit is None:
        return audio_paths
    limit = int(raw_limit)
    if limit < 0:
        raise ValueError("max_augmentation_variants_per_group must be non-negative")
    selection_seed = int(spec.get("augmentation_selection_seed", 0))
    groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for audio in audio_paths:
        base_stem, augmentation = _split_augmentation(audio.stem, spec)
        groups[base_stem].append((audio, augmentation))
    selected: list[Path] = []
    for members in groups.values():
        variants = [audio for audio, value in members if value != "original"]
        variants.sort(
            key=lambda audio: _stable_fraction(f"{dataset}:{audio.relative_to(root).as_posix()}", selection_seed)
        )
        selected.extend(audio for audio, value in members if value == "original")
        selected.extend(variants[:limit])
        skipped["augmentation_variant_limit"] += max(0, len(variants) - limit)
    return sorted(selected)


def _find_midi(audio: Path, midi_root: Path, spec: Mapping[str, Any]) -> Path | None:
    for suffix in spec.get("midi_extensions", [".mid", ".midi"]):
        candidate = midi_root / f"{audio.stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    return None


def _flat_class_ids(
    audio: Path,
    midi: Path,
    *,
    class_source: str,
    npz_root: Path | None,
    remap: Mapping[int, int],
) -> tuple[int, ...]:
    """平置きデータセットのラベルを npz か MIDI から推定する。

    npz が優先。無い / 読めない場合は MIDI の program にフォールバックする。
    """
    if class_source == "npz" and npz_root is not None:
        class_ids = _npz_class_ids(npz_root / f"{audio.stem}.npz", remap)
        if class_ids is not None:
            return class_ids
    elif class_source not in ("npz", "midi"):
        raise ValueError(f"Unknown class_source: {class_source}")
    return tuple(sorted({remap.get(value, value) for value in _midi_class_ids(midi)}))


def _flat_records(spec: Mapping[str, Any], config_root: Path) -> tuple[list[ManifestSourceRecord], Counter[str]]:
    """wav と MIDI が平置きされたデータセットを読む。

    ラベルはファイル名から分からないので、npz か MIDI の中身から推定する。
    推定が曖昧なもの（ドラムしかない、ステムをまたぐ、設定で禁止したクラス）は
    理由つきで捨てる。
    """
    dataset = str(spec["name"])
    root = _resolve_path(str(spec["root"]), config_root)
    midi_root = _resolve_path(str(spec["midi_dir"]), root)
    duration_manifest = spec.get("duration_manifest")
    duration_lookup = _duration_lookup(_resolve_path(str(duration_manifest), root) if duration_manifest else None)
    npz_dir = spec.get("npz_dir")
    npz_root = _resolve_path(str(npz_dir), root) if npz_dir else None
    remap = {
        get_instrument_class_id_by_name(str(source)): get_instrument_class_id_by_name(str(target))
        for source, target in dict(spec.get("class_remap", {})).items()
    }
    raw_allowed_classes = spec.get("allowed_classes")
    allowed_classes = (
        {get_instrument_class_id_by_name(str(class_name)) for class_name in raw_allowed_classes}
        if raw_allowed_classes is not None
        else None
    )
    allowed_groups = {str(value) for value in spec["allowed_stem_groups"]} if "allowed_stem_groups" in spec else None
    class_source = str(spec.get("class_source", "npz"))
    note_target_mode = str(spec.get("note_target_mode", "midi"))
    require_notes = bool(spec.get("require_midi_notes", False))

    records: list[ManifestSourceRecord] = []
    skipped: Counter[str] = Counter()
    class_cache: dict[str, tuple[int, ...]] = {}
    audio_paths = _limit_augmentation_variants(
        sorted(root.glob(str(spec["audio_glob"]))),
        dataset=dataset,
        root=root,
        spec=spec,
        skipped=skipped,
    )
    for audio in audio_paths:
        midi = _find_midi(audio, midi_root, spec)
        if midi is None:
            skipped["missing_midi"] += 1
            continue
        if require_notes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                if not load_refinement_note_table(midi).note_count:
                    skipped["no_midi_notes"] += 1
                    continue
        # 同じ曲の augmentation 違いはラベルも同じなので、元の曲名でキャッシュする。
        base_stem, _ = _split_augmentation(audio.stem, spec)
        if base_stem not in class_cache:
            class_cache[base_stem] = _flat_class_ids(
                audio, midi, class_source=class_source, npz_root=npz_root, remap=remap
            )
        class_ids = class_cache[base_stem]
        if not class_ids:
            skipped["no_non_drum_notes"] += 1
            continue
        if allowed_classes is not None and not set(class_ids).issubset(allowed_classes):
            skipped["disallowed_class"] += 1
            continue
        stem_group = _stem_group_for_class_ids(class_ids)
        if stem_group is None:
            skipped["cross_stem_group"] += 1
            continue
        if allowed_groups is not None and stem_group not in allowed_groups:
            skipped["disallowed_stem_group"] += 1
            continue
        source_id, song_id, split_group_id, atomic_group_id = _record_identity(dataset, audio.stem, spec)
        records.append(
            ManifestSourceRecord(
                source_id=source_id,
                dataset=dataset,
                song_id=song_id,
                split_group_id=split_group_id,
                atomic_group_id=atomic_group_id,
                stem_name=stem_group,
                track_type=base_stem,
                audio_path=audio.resolve(),
                midi_path=midi,
                duration_seconds=_audio_duration(audio, duration_lookup),
                target_class_ids=class_ids,
                note_target_mode=note_target_mode,
            )
        )
    return records, skipped


def _assign_splits(
    records: list[ManifestSourceRecord],
    *,
    seed: int,
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    """曲単位で train/validation/test を割り振る。

    割り振りは曲名のハッシュで決めるので、データが増えても既存の曲の所属は変わらない。
    ただしそれだけだと希少クラスが train に 1 曲も残らないことがあるため、
    最後に「train に 1 つも無いクラス」を救済して train へ移す。
    """
    groups: dict[str, set[int]] = defaultdict(set)
    for record in records:
        groups[record.split_group_id].update(record.target_class_ids)
    validation_end = train_fraction + validation_fraction
    result = {}
    for group_id in groups:
        value = _stable_fraction(group_id, seed)
        result[group_id] = "train" if value < train_fraction else "validation" if value < validation_end else "test"
    # 救済処理: train に 1 曲も無いクラスがあれば、そのクラスを含む曲を 1 つ train へ移す。
    all_classes = set().union(*groups.values()) if groups else set()
    for class_id in sorted(all_classes):
        candidates = [key for key, values in groups.items() if class_id in values]
        if candidates and not any(result[key] == "train" for key in candidates):
            selected = min(candidates, key=lambda key: _stable_fraction(key, seed + 1))
            result[selected] = "train"
    return result


def _write_manifest_csv(
    destination: Path, records: list[ManifestSourceRecord], split_by_group: Mapping[str, str]
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(records, key=lambda item: (item.dataset, item.song_id, item.source_id))
    with destination.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in sorted_records:
            class_names = tuple(INSTRUMENT_CLASSES[value] for value in record.target_class_ids)
            writer.writerow(
                {
                    "source_id": record.source_id,
                    "dataset": record.dataset,
                    "song_id": record.song_id,
                    "split_group_id": record.split_group_id,
                    "atomic_group_id": record.atomic_group_id,
                    "stem_name": record.stem_name,
                    "track_type": record.track_type,
                    "audio_path": str(record.audio_path),
                    "midi_path": str(record.midi_path),
                    "duration_seconds": f"{record.duration_seconds:.6f}",
                    "target_class_names": "|".join(class_names),
                    "target_class_ids": "|".join(map(str, record.target_class_ids)),
                    "target_family_ids": "|".join(map(str, family_ids_for_class_ids(record.target_class_ids))),
                    "note_target_mode": record.note_target_mode,
                    "split": split_by_group[record.split_group_id],
                }
            )


def _summarize(
    records: list[ManifestSourceRecord],
    *,
    config_file: Path,
    destination: Path,
    split_by_group: Mapping[str, str],
    skipped: Mapping[str, Counter[str]],
) -> dict[str, object]:
    """クラス分布の偏りと、除外された音源の理由をまとめる。

    「思ったよりデータが少ない」ときに最初に見る場所。
    """
    class_counts: Counter[str] = Counter()
    for record in records:
        for class_id in record.target_class_ids:
            class_counts[INSTRUMENT_CLASSES[class_id]] += 1
    atomic_members = Counter(record.atomic_group_id for record in records)
    return {
        "config": str(config_file),
        "manifest": str(destination),
        "source_count": len(records),
        "atomic_unit_count": len(atomic_members),
        "multi_source_atomic_unit_count": sum(count > 1 for count in atomic_members.values()),
        "dataset_counts": dict(sorted(Counter(record.dataset for record in records).items())),
        "split_counts": dict(
            sorted(Counter(split_by_group[record.split_group_id] for record in records).items())
        ),
        "class_counts": dict(sorted(class_counts.items())),
        "skipped": {name: dict(sorted(counts.items())) for name, counts in skipped.items()},
    }


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict) or not isinstance(config.get("datasets"), list):
        raise ValueError("Dataset config must be an object containing a datasets list")
    return config


def build_refinement_manifest(
    config_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, object]:
    """設定 JSON に書かれた各データセットを走査し、manifest CSV と要約を書き出す。

    Returns:
        件数・クラス分布・除外理由をまとめた要約。同じ内容を ``*.summary.json`` にも保存する。
        除外理由の内訳は「思ったよりデータが少ない」ときに最初に見る場所。
    """
    config_file = Path(config_path).resolve()
    config = _load_config(config_file)
    config_root = config_file.parent
    destination_value = output_path or config.get("output")
    if destination_value is None:
        raise ValueError("An output path is required in the config or function call")
    destination = _resolve_path(str(destination_value), config_root)
    seed = int(config.get("seed", 42))
    train_fraction = float(config.get("train_fraction", 0.8))
    validation_fraction = float(config.get("validation_fraction", 0.1))
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1)")
    if validation_fraction < 0.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test split")

    # 1. データセットごとに、形式に応じたアダプタで音源を集める。
    records: list[ManifestSourceRecord] = []
    skipped: dict[str, Counter[str]] = {}
    seen_names: set[str] = set()
    forced_splits: dict[str, str] = {}
    for raw_spec in config["datasets"]:
        if not isinstance(raw_spec, dict):
            raise ValueError("Each dataset entry must be an object")
        name = str(raw_spec["name"])
        if name in seen_names:
            raise ValueError(f"Duplicate dataset name: {name}")
        seen_names.add(name)
        adapter = str(raw_spec["type"])
        if adapter == "class_manifests":
            current, current_skipped = _class_manifest_records(raw_spec, config_root)
        elif adapter == "flat":
            current, current_skipped = _flat_records(raw_spec, config_root)
        else:
            raise ValueError(f"Unknown dataset adapter: {adapter}")
        records.extend(current)
        skipped[name] = current_skipped
        forced = raw_spec.get("force_split")
        if forced is not None:
            if forced not in ("train", "validation", "test"):
                raise ValueError(f"force_split must be train/validation/test, got {forced!r}")
            forced_splits[name] = str(forced)

    # 2. 曲単位で split を決め、CSV へ書き出す。
    split_by_group = _assign_splits(
        records,
        seed=seed,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )
    # 曲数が少ないデータセットはハッシュ分割のばらつきが大きく、貴重な音源が
    # validation/test へ流れてしまう（実録音 14 曲で train 7 / val 5 / test 2 になった）。
    # 別に評価用の音源を確保してある場合は、force_split で分割ごと固定する。
    for record in records:
        if record.dataset in forced_splits:
            split_by_group[record.split_group_id] = forced_splits[record.dataset]
    _write_manifest_csv(destination, records, split_by_group)

    # 3. 内訳を要約する。
    summary = _summarize(
        records,
        config_file=config_file,
        destination=destination,
        split_by_group=split_by_group,
        skipped=skipped,
    )
    with destination.with_suffix(".summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    return summary
