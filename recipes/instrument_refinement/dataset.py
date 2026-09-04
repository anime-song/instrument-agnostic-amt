"""学習用 Dataset とサンプラー。

manifest の 1 行（= 音源 1 本）をそのまま学習サンプルにすると、
「単一楽器のきれいな音」しか見たことがないモデルになってしまう。実際の推論では
分離器の 1 ステムに複数の楽器が混ざって入ってくるので、学習時にも同じ曲・同じステム
グループの音源を足し合わせた「混合サンプル」を作る（composite）。

サンプルの作り方:

1. atomic unit（分割してはいけない音源のまとまり）を 1 つ選ぶ。
2. 学習時は確率 composite_probability で、同じステムグループの別 unit を足す。
   相手は既定では同じ曲から選ぶが、cross_song_composite_ratio を上げると曲をまたいで
   選べる（同一曲に別楽器が揃っているデータが少ないため、組み合わせを増やす手段）。
3. 窓（既定 8 秒）を切り出し、その窓に発音があるノートを集める。
4. 混ぜた結果、どちらの音源のノートか区別できないもの（同音高・ほぼ同時刻）は
   1 つのクエリにまとめ、「どちらかである」という部分ラベルにする。

正解ラベルは「1 つに確定」ではなく「候補集合」で持つ。混合サンプルではノート単位の
正解が一意に決まらないため、loss 側で部分ラベルとして扱う（training/losses.py）。
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import math
import os
import random
import re
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from torch.utils.data import Dataset, Sampler

from instrument_agnostic_amt.taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    NUM_INSTRUMENT_CLASSES,
)
from instrument_agnostic_amt.instrument_refinement.data.labels import (
    FAMILY_NAMES,
    family_id_for_class_name,
    inference_stem_group,
    stem_context_id,
)
from instrument_agnostic_amt.instrument_refinement.data.midi import (
    RefinementNoteTable,
    load_refinement_note_table,
)

# pretty_midi は type 0/1 の規格から外れた MIDI を読むたびにこの警告を出す。学習
# コーパスの性質で直しようがなく、8 万件を読む間ずっと流れ続けるだけなので黙らせる。
# import 時に 1 度だけ登録する（DataLoader のワーカーにも効かせるため、かつ
# warnings.catch_warnings を毎回通ると全モジュールの重複抑制が壊れるため）。
warnings.filterwarnings(
    "ignore",
    message="Tempo, Key or Time signature change events found on non-zero tracks",
    category=RuntimeWarning,
    module="pretty_midi",
)


@dataclass(frozen=True)
class RefinementSourceRecord:
    """manifest から読み込んだ音源 1 本。

    note_target_mode が "midi" のときはノートごとの正解を MIDI の program から取り、
    それ以外は音源全体のラベル (target_class_ids) を全ノートに適用する。
    """

    source_id: str
    dataset: str
    song_id: str
    atomic_group_id: str
    stem_name: str
    track_type: str
    audio_path: Path
    midi_path: Path | None
    duration_seconds: float
    target_class_ids: tuple[int, ...]
    target_family_ids: tuple[int, ...]
    note_target_mode: str


@dataclass(frozen=True)
class RefinementWindowRecord:
    """sampling="all"（評価用の全走査）で使う、窓 1 つ分の位置。"""

    unit_index: int
    start_seconds: float


@dataclass(frozen=True)
class NoteQueries:
    """窓に含まれるノート列。全フィールドが同じ長さで並ぶ。

    音源をまたいで連結する・時刻順に並べ直す・区別できない組をまとめる、という操作を
    フィールドごとに書くと同じ行が 9 回並ぶので、まとめて 1 つの値として扱う。
    """

    start: torch.Tensor  # [N] 窓先頭を 0 秒とした発音開始
    end: torch.Tensor  # [N]
    pitch: torch.Tensor  # [N]
    class_target: torch.Tensor  # [N, クラス数] bool。正解の候補集合
    family_target: torch.Tensor  # [N, ファミリー数] bool
    primary: torch.Tensor  # [N] 候補が 1 つに定まるときだけクラス ID、他は -1
    source_hash: torch.Tensor  # [N] どの音源由来か
    weight: torch.Tensor  # [N]
    contrastive: torch.Tensor  # [N] bool。音色の対照学習に使ってよいか

    _FIELDS = (
        "start",
        "end",
        "pitch",
        "class_target",
        "family_target",
        "primary",
        "source_hash",
        "weight",
        "contrastive",
    )

    def __len__(self) -> int:
        return int(self.pitch.numel())

    @classmethod
    def empty(cls) -> NoteQueries:
        return cls(
            start=torch.empty(0, dtype=torch.float32),
            end=torch.empty(0, dtype=torch.float32),
            pitch=torch.empty(0, dtype=torch.long),
            class_target=torch.empty((0, NUM_INSTRUMENT_CLASSES), dtype=torch.bool),
            family_target=torch.empty((0, len(FAMILY_NAMES)), dtype=torch.bool),
            primary=torch.empty(0, dtype=torch.long),
            source_hash=torch.empty(0, dtype=torch.long),
            weight=torch.empty(0, dtype=torch.float32),
            contrastive=torch.empty(0, dtype=torch.bool),
        )

    @classmethod
    def concat(cls, parts: list[NoteQueries]) -> NoteQueries:
        non_empty = [part for part in parts if len(part)]
        if not non_empty:
            return cls.empty()
        return cls(**{name: torch.cat([getattr(part, name) for part in non_empty], dim=0) for name in cls._FIELDS})

    def select(self, order: torch.Tensor) -> NoteQueries:
        """与えた添字で並べ替える / 部分集合を取る。"""
        return NoteQueries(**{name: getattr(self, name)[order] for name in self._FIELDS})

    def sorted_by_time(self) -> NoteQueries:
        """時刻順に並べ直す。混ぜた音源の順序が結果に出ないようにするため。"""
        if not len(self):
            return self
        # テンソルへのスカラー添字アクセスは 1 回 1us 近くかかるので、先に Python の
        # 値へ落としてから並べ替える。
        keys = list(zip(self.start.tolist(), self.pitch.tolist(), self.source_hash.tolist()))
        order = sorted(range(len(self)), key=keys.__getitem__)
        return self.select(torch.tensor(order, dtype=torch.long))


class ClassBalancedSampler(Sampler[int]):
    """Choose a target class uniformly, then choose one atomic unit of that class.

    音源数はクラスごとに大きく偏る（ピアノは大量、ハーモニカは少数）。素直に一様抽出すると
    多数派しか学習されないので、まずクラスを一様に選び、その中から音源を選ぶ。
    結果として希少クラスの音源は何度も出てくる。

    クラスの中でどのデータセットから引くかは、既定では件数任せになる。これだと件数の
    少ないデータセットがほぼ出てこない。例えば orchestral_woodwind は cocochorales 62% /
    single_stems 35% に対し、実録音の real_instrument_sample は 3.4% しか引かれない。
    合成音に偏った学習になるので、``class_dataset_unit_indices`` と ``dataset_weights`` を
    渡すと「クラスを一様に選ぶ -> データセットを重みで選ぶ -> unit を一様に選ぶ」に変わる。
    AMT 側の dataset_config_*.yaml が weight でやっているのと同じ考え方。
    """

    def __init__(
        self,
        class_to_unit_indices: dict[int, tuple[int, ...]],
        *,
        num_samples: int,
        seed: int,
        class_dataset_unit_indices: Mapping[int, Mapping[str, tuple[int, ...]]] | None = None,
        dataset_weights: Mapping[str, float] | None = None,
    ) -> None:
        self.class_to_unit_indices = {
            int(class_id): tuple(indices) for class_id, indices in class_to_unit_indices.items() if indices
        }
        if not self.class_to_unit_indices:
            raise ValueError("Class-balanced sampling requires at least one class")
        self.class_ids = tuple(sorted(self.class_to_unit_indices))
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        # クラスごとに (データセット別 unit 一覧, 累積重み) を用意しておく。
        # 重み 0 のデータセットは候補から外す。全部 0 になったクラスは件数任せへ戻す。
        self.class_dataset_choices: dict[int, tuple[tuple[tuple[int, ...], ...], list[float]]] = {}
        if class_dataset_unit_indices is not None and dataset_weights is not None:
            for class_id, per_dataset in class_dataset_unit_indices.items():
                buckets: list[tuple[int, ...]] = []
                cumulative: list[float] = []
                total = 0.0
                for dataset in sorted(per_dataset):
                    indices = tuple(per_dataset[dataset])
                    weight = float(dataset_weights.get(dataset, 1.0))
                    if not indices or weight <= 0.0:
                        continue
                    total += weight
                    buckets.append(indices)
                    cumulative.append(total)
                if buckets:
                    self.class_dataset_choices[int(class_id)] = (tuple(buckets), cumulative)

    def __iter__(self) -> Iterator[int]:
        # epoch ごとに種を変えて、同じ順序を繰り返さないようにする。
        generator = random.Random(self.seed + self.epoch)
        self.epoch += 1
        for _ in range(self.num_samples):
            class_id = generator.choice(self.class_ids)
            choice = self.class_dataset_choices.get(class_id)
            if choice is None:
                yield generator.choice(self.class_to_unit_indices[class_id])
                continue
            buckets, cumulative = choice
            position = generator.uniform(0.0, cumulative[-1])
            yield generator.choice(buckets[bisect.bisect_left(cumulative, position)])

    def __len__(self) -> int:
        return self.num_samples


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _manifest_path(value: str, *, platform_name: str | None = None) -> Path:
    """Interpret Windows manifest paths from WSL without rewriting the CSV.

    Windows で作った manifest を WSL 側から学習に使えるよう、``D:\\x`` を
    ``/mnt/d/x`` に読み替える。CSV そのものは書き換えない。
    """
    raw = str(value).strip()
    platform = platform_name or os.name
    if platform != "nt" and _WINDOWS_ABSOLUTE_PATH.match(raw):
        windows_path = PureWindowsPath(raw)
        drive = windows_path.drive.rstrip(":").casefold()
        return Path("/mnt") / drive / Path(*windows_path.parts[1:])
    return Path(raw)


def _resolve_path(value: str, root: Path) -> Path | None:
    if not str(value).strip():
        return None
    path = _manifest_path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _source_hash(source_id: str) -> int:
    """音源 ID を安定した整数へ。対照学習で「同じ音源か」を比べる鍵に使う。"""
    digest = hashlib.sha256(source_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def _load_audio_window(
    path: Path,
    *,
    start_seconds: float,
    window_seconds: float,
    target_sample_rate: int,
) -> tuple[torch.Tensor, int]:
    """必要な区間だけを読み、2ch・指定サンプルレート・固定長にそろえる。

    Returns:
        (波形 [2, 窓長], 0 埋め前の有効サンプル数)。
    """
    info = sf.info(str(path))
    source_rate = int(info.samplerate)
    source_start = max(0, int(round(start_seconds * source_rate)))
    source_frames = max(1, int(round(window_seconds * source_rate)))
    audio, _ = sf.read(
        str(path),
        start=source_start,
        frames=source_frames,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(np.asarray(audio.T, dtype=np.float32))
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1).contiguous()
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    if source_rate != target_sample_rate and waveform.numel():
        waveform = audio_functional.resample(waveform, source_rate, int(target_sample_rate))
    target_frames = int(round(window_seconds * target_sample_rate))
    valid_frames = min(int(waveform.shape[-1]), target_frames)
    if int(waveform.shape[-1]) < target_frames:
        waveform = torch.nn.functional.pad(waveform, (0, target_frames - int(waveform.shape[-1])))
    else:
        waveform = waveform[:, :target_frames]
    return waveform.contiguous(), valid_frames


def _read_manifest_records(
    manifest_path: Path,
    *,
    split: str,
    require_midi: bool,
    max_sources: int | None,
) -> list[RefinementSourceRecord]:
    """manifest から、この split で実際に使える音源だけを読む。

    ファイルが無い / ラベルが空 / ドラム、はここで落とす。max_sources は動作確認用の
    件数制限で、atomic unit の途中で切らないよう unit 単位で足していく。
    """
    records: list[RefinementSourceRecord] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if split != "all" and row.get("split") != split:
                continue
            if str(row.get("stem_name", "")).strip().casefold() == "drums":
                continue  # ドラムは対象外。
            audio_path = _resolve_path(row.get("audio_path", ""), manifest_path.parent)
            midi_path = _resolve_path(row.get("midi_path", ""), manifest_path.parent)
            if audio_path is None or not audio_path.is_file():
                continue
            if require_midi and (midi_path is None or not midi_path.is_file()):
                continue
            class_ids = tuple(int(item) for item in str(row.get("target_class_ids", "")).split("|") if item != "")
            family_ids = tuple(int(item) for item in str(row.get("target_family_ids", "")).split("|") if item != "")
            if not class_ids:
                continue
            records.append(
                RefinementSourceRecord(
                    source_id=str(row["source_id"]),
                    dataset=str(row.get("dataset", "dataset")),
                    song_id=str(row.get("song_id", "")),
                    atomic_group_id=str(row.get("atomic_group_id") or row["source_id"]),
                    stem_name=str(row.get("stem_name", "unknown")),
                    track_type=str(row.get("track_type", "")),
                    audio_path=audio_path,
                    midi_path=midi_path,
                    duration_seconds=float(row.get("duration_seconds") or 0.0),
                    target_class_ids=class_ids,
                    target_family_ids=family_ids,
                    note_target_mode=str(row.get("note_target_mode") or "source"),
                )
            )
    if max_sources is None:
        return records
    grouped: dict[str, list[RefinementSourceRecord]] = {}
    for record in records:
        grouped.setdefault(record.atomic_group_id, []).append(record)
    limited: list[RefinementSourceRecord] = []
    for members in grouped.values():
        if limited and len(limited) + len(members) > int(max_sources):
            break
        limited.extend(members)
    return limited


def _group_atomic_units(sources: tuple[RefinementSourceRecord, ...]) -> tuple[tuple[int, ...], ...]:
    """同じ atomic_group_id の音源を 1 つの unit にまとめる。これが取り出しの単位。"""
    members: dict[str, list[int]] = {}
    for source_index, source in enumerate(sources):
        members.setdefault(source.atomic_group_id, []).append(source_index)
    units = tuple(tuple(indices) for indices in members.values())
    # unit の中で曲やステムグループが混ざっていると、混合の前提が崩れるので弾く。
    for unit in units:
        groups = {inference_stem_group(sources[index].stem_name) for index in unit}
        songs = {sources[index].song_id for index in unit}
        if len(groups) != 1 or len(songs) != 1:
            raise ValueError("An atomic source unit must stay in one song/stem group")
    return units


def _same_song_composite_candidates(
    unit_song_ids: tuple[str, ...], unit_stem_groups: tuple[str, ...]
) -> tuple[tuple[int, ...], ...]:
    """unit ごとの混合相手の候補。同じ曲・同じステムグループの別 unit。

    分離器の出力として実際に起こりうる組み合わせになる。
    """
    by_song_group: dict[tuple[str, str], list[int]] = {}
    for unit_index, key in enumerate(zip(unit_song_ids, unit_stem_groups)):
        by_song_group.setdefault(key, []).append(unit_index)
    return tuple(
        tuple(member for member in by_song_group[key] if member != unit_index)
        for unit_index, key in enumerate(zip(unit_song_ids, unit_stem_groups))
    )


def _class_indices(
    sources: tuple[RefinementSourceRecord, ...], units: tuple[tuple[int, ...], ...]
) -> tuple[dict[int, tuple[int, ...]], dict[int, dict[str, tuple[int, ...]]]]:
    """クラス -> そのクラスを含む unit の対応表。ClassBalancedSampler が使う。

    データセット別の内訳も返す。クラス内でどのデータセットから引くかを件数任せにせず
    重みで決められるようにするため。
    """
    class_to_units: dict[int, list[int]] = {}
    class_dataset_units: dict[int, dict[str, list[int]]] = {}
    for unit_index, unit in enumerate(units):
        class_ids = {class_id for source_index in unit for class_id in sources[source_index].target_class_ids}
        dataset = sources[unit[0]].dataset
        for class_id in class_ids:
            class_to_units.setdefault(class_id, []).append(unit_index)
            class_dataset_units.setdefault(class_id, {}).setdefault(dataset, []).append(unit_index)
    return (
        {class_id: tuple(indices) for class_id, indices in class_to_units.items()},
        {
            class_id: {dataset: tuple(indices) for dataset, indices in per_dataset.items()}
            for class_id, per_dataset in class_dataset_units.items()
        },
    )


def _all_window_records(
    sources: tuple[RefinementSourceRecord, ...],
    units: tuple[tuple[int, ...], ...],
    *,
    window_seconds: float,
    hop_seconds: float,
) -> tuple[RefinementWindowRecord, ...]:
    """sampling="all" 用に、全 unit の窓位置を並べる。末尾が余る場合は末尾の窓を足す。"""
    windows: list[RefinementWindowRecord] = []
    for unit_index, unit in enumerate(units):
        duration_seconds = max(sources[index].duration_seconds for index in unit)
        if duration_seconds <= window_seconds:
            starts = [0.0]
        else:
            count = int(math.floor((duration_seconds - window_seconds) / hop_seconds)) + 1
            starts = [index * hop_seconds for index in range(count)]
            last_start = max(0.0, duration_seconds - window_seconds)
            if not starts or abs(starts[-1] - last_start) > 1e-6:
                starts.append(last_start)
        windows.extend(RefinementWindowRecord(unit_index=unit_index, start_seconds=start) for start in starts)
    return tuple(windows)


def _ambiguous_note_groups(queries: NoteQueries, onset_tolerance_seconds: float) -> list[list[int]]:
    """区別できないノートの組を作る。

    同じ音高・発音時刻が tolerance 以内・音源が異なる、の 3 条件がそろうと、音を聴いても
    どちらの楽器のノートか決められない。先に見つかった組へ順に足していく貪欲法で、
    1 つの組には同じ音源のノートを 2 つ入れない。

    ノート数が数百になる窓があるので、テンソルへスカラー添字アクセスするのではなく
    先に Python の値へ落とし、走査対象も音高が一致する組だけに絞る。
    """
    pitches = queries.pitch.tolist()
    starts = queries.start.tolist()
    sources = queries.source_hash.tolist()

    groups: list[list[int]] = []
    group_sources: list[set[int]] = []
    # 音高の一致は必須条件なので、組を音高ごとに分けて持つ。バケットの中は生成順の
    # ままなので、「最初に一致した組へ入れる」という貪欲法の結果は変わらない。
    groups_by_pitch: dict[int, list[int]] = {}
    for note_index, (pitch, start, current_source) in enumerate(zip(pitches, starts, sources)):
        same_pitch_groups = groups_by_pitch.setdefault(pitch, [])
        for group_index in same_pitch_groups:
            representative = groups[group_index][0]
            if abs(starts[representative] - start) > onset_tolerance_seconds:
                continue
            if current_source in group_sources[group_index]:
                continue
            groups[group_index].append(note_index)
            group_sources[group_index].add(current_source)
            break
        else:
            same_pitch_groups.append(len(groups))
            groups.append([note_index])
            group_sources.append({current_source})
    return groups


def _merge_ambiguous_note_queries(queries: NoteQueries, *, onset_tolerance_seconds: float) -> NoteQueries:
    """Merge indistinguishable cross-source notes into partial-label queries.

    複数音源を混ぜたとき、区別できないノートを別々のクエリとして「これはピアノ」
    「これはギター」と教えると、モデルは矛盾した教師を受け取ることになる。そこで
    区別できない組を 1 つのクエリにまとめ、正解を「ピアノまたはギター」という
    候補集合（部分ラベル）にする。まとめたクエリは音色が特定できないので、
    対照学習からは外す。
    """
    if len(queries) < 2 or onset_tolerance_seconds < 0.0:
        return queries
    groups = _ambiguous_note_groups(queries, onset_tolerance_seconds)
    if all(len(members) == 1 for members in groups):
        return queries

    merged: dict[str, list[torch.Tensor]] = {name: [] for name in NoteQueries._FIELDS}
    for members in groups:
        indices = torch.tensor(members, dtype=torch.long)
        # 正解は OR で合成する（= 候補集合）。時刻は平均と最大でならす。
        class_target = queries.class_target[indices].any(dim=0)
        class_ids = torch.nonzero(class_target, as_tuple=False).flatten()
        merged["start"].append(queries.start[indices].mean())
        merged["end"].append(queries.end[indices].max())
        merged["pitch"].append(queries.pitch[members[0]])
        merged["class_target"].append(class_target)
        merged["family_target"].append(queries.family_target[indices].any(dim=0))
        # 候補が 1 つに絞れたときだけ「確定した正解」として扱う（精度計算に使う）。
        merged["primary"].append(class_ids[0] if int(class_ids.numel()) == 1 else torch.tensor(-1, dtype=torch.long))
        merged["source_hash"].append(queries.source_hash[members[0]])
        merged["weight"].append(queries.weight[indices].min())
        merged["contrastive"].append(torch.tensor(len(members) == 1, dtype=torch.bool))
    return NoteQueries(**{name: torch.stack(values) for name, values in merged.items()})


class ManifestInstrumentRefinementDataset(Dataset):
    """Exact-label manifest sources and deployment-stem mixtures.

    1 要素 = atomic unit 1 つ（sampling="all" のときは窓 1 つ）。学習時は
    ``{"views": [...]}`` の形で同じ音源から複数の窓を返す。これは同一音源の異なる窓を
    同じバッチに入れて対照学習・一貫性 loss を効かせるため（collate 側で平坦化される）。

    Args:
        sampling: "random"（学習: 窓位置をランダム）/"fixed"（検証: 音源ごとに固定）/
            "all"（評価: 全窓を走査）。
        composite_probability: 学習時に他 unit を混ぜる確率。
        cross_song_composite_ratio: 混ぜるとき、相手を別の曲から選ぶ確率。0 なら
            同じ曲の中だけで混ぜる（従来の挙動）。同一曲に複数楽器が揃っている
            データは少ないため、これを上げると学べる楽器の組み合わせが大きく増える。
        random_gain_db: 混合前に音源ごとの音量を振る幅。バランスの偏りに強くする。
        ambiguous_note_onset_tolerance_seconds: 区別不能なノートをまとめる時刻の許容差。
        augment_probability: 音源ごとに AudioAugmentor（EQ・歪み・リバーブ・ノイズ等）を
            掛ける確率。0 なら一切呼ばず、依存も読み込まない（従来の挙動）。本家 AMT と
            同じく混合の前に音源単位で掛けるので、混ぜた後の音は音源ごとに別々の録音条件を
            通ったものになる。
        ir_folder: リバーブ用インパルス応答のフォルダ。None なら畳み込みリバーブを使わない。
        noise_folder: 背景ノイズのフォルダ。None ならガウスノイズだけになる。
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        split: str = "train",
        sample_rate: int = 22_050,
        window_seconds: float = 8.0,
        hop_seconds: float = 4.0,
        sampling: str | None = None,
        windows_per_source: int = 2,
        require_midi: bool = True,
        random_gain_db: float = 3.0,
        composite_probability: float = 0.5,
        cross_song_composite_ratio: float = 0.0,
        max_mixture_sources: int | None = None,
        ambiguous_note_onset_tolerance_seconds: float = 0.03,
        max_sources: int | None = None,
        augment_probability: float = 0.0,
        ir_folder: str | Path | None = None,
        noise_folder: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path).resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        if sample_rate <= 0 or window_seconds <= 0.0 or hop_seconds <= 0.0:
            raise ValueError("sample rate, window and hop must be positive")
        if windows_per_source <= 0:
            raise ValueError("windows_per_source must be positive")
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.sampling = sampling or ("random" if split == "train" else "fixed")
        if self.sampling not in {"random", "fixed", "all"}:
            raise ValueError("sampling must be 'random', 'fixed', or 'all'")
        self.windows_per_source = int(windows_per_source)
        self.random_gain_db = float(random_gain_db)
        if not 0.0 <= composite_probability <= 1.0:
            raise ValueError("composite_probability must be in [0, 1]")
        self.composite_probability = float(composite_probability)
        if not 0.0 <= cross_song_composite_ratio <= 1.0:
            raise ValueError("cross_song_composite_ratio must be in [0, 1]")
        self.cross_song_composite_ratio = float(cross_song_composite_ratio)
        if max_mixture_sources is not None and max_mixture_sources < 2:
            raise ValueError("max_mixture_sources must be at least two")
        if ambiguous_note_onset_tolerance_seconds < 0.0:
            raise ValueError("ambiguous note onset tolerance must be nonnegative")
        self.max_mixture_sources = max_mixture_sources
        self.ambiguous_note_onset_tolerance_seconds = float(ambiguous_note_onset_tolerance_seconds)
        if not 0.0 <= augment_probability <= 1.0:
            raise ValueError("augment_probability must be in [0, 1]")
        self.augment_probability = float(augment_probability)
        self.ir_folder = str(ir_folder) if ir_folder is not None else None
        self.noise_folder = str(noise_folder) if noise_folder is not None else None
        self._note_cache: dict[Path, RefinementNoteTable] = {}
        # AudioAugmentor は audiomentations / pedalboard を抱えていて pickle できないので、
        # DataLoader のワーカーごとに最初の 1 回だけ組み立てる。
        self._augmentor: Any | None = None

        records = _read_manifest_records(
            self.manifest_path, split=split, require_midi=require_midi, max_sources=max_sources
        )
        if not records:
            raise ValueError(f"No usable {split!r} records in {self.manifest_path}")
        self.sources = tuple(records)
        self.units = _group_atomic_units(self.sources)
        self.unit_song_ids = tuple(self.sources[unit[0]].song_id for unit in self.units)
        self.unit_stem_groups = tuple(inference_stem_group(self.sources[unit[0]].stem_name) for unit in self.units)
        self.composite_candidates = _same_song_composite_candidates(self.unit_song_ids, self.unit_stem_groups)
        # 別曲混合用の一覧。同じ曲に別楽器が揃っているデータは少なく、例えば
        # 「アコギと歪みギターが同居する曲」は実データに 1 件も無い。曲をまたいで
        # 混ぜれば和声的には不自然になるが、音色を見分ける学習には十分で、
        # 組み合わせ数を桁違いに増やせる。
        # 候補が数万件になるステムもあるため、事前に全対を持たず抽選で引く。
        stem_group_units: dict[str, list[int]] = {}
        for unit_index, stem_group in enumerate(self.unit_stem_groups):
            stem_group_units.setdefault(stem_group, []).append(unit_index)
        self.stem_group_unit_indices = {group: tuple(members) for group, members in stem_group_units.items()}
        self.class_to_unit_indices, self.class_dataset_unit_indices = _class_indices(self.sources, self.units)
        # sampling="all" のときだけ、全 unit の窓位置を事前に並べておく。
        self.windows: tuple[RefinementWindowRecord, ...] = ()
        if self.sampling == "all":
            self.windows = _all_window_records(
                self.sources, self.units, window_seconds=self.window_seconds, hop_seconds=self.hop_seconds
            )

    def __getstate__(self) -> dict[str, Any]:
        # DataLoader のワーカーへ渡すときに、MIDI キャッシュや augmentor まで pickle しない。
        state = dict(self.__dict__)
        state["_note_cache"] = {}
        state["_augmentor"] = None
        return state

    def _get_augmentor(self) -> Any | None:
        """augment を使うときだけ AudioAugmentor を作る。

        audiomentations / pedalboard は import が重く、拡張を切って回す場合まで
        読み込ませたくないので、ここで初めて import する。
        """
        if self.augment_probability <= 0.0:
            return None
        if self._augmentor is None:
            from recipes.common.augmentation import AudioAugmentor

            self._augmentor = AudioAugmentor(
                sample_rate=self.sample_rate,
                ir_folder=self.ir_folder,
                noise_folder=self.noise_folder,
            )
        return self._augmentor

    def _augment_window(self, audio: torch.Tensor) -> torch.Tensor:
        """1 音源ぶんの窓に AudioAugmentor を掛ける。

        窓長はバッチをそろえるために固定でなければならない。今の effect は長さを
        変えない実装だが、将来 1 つでも長さを変えるものが入ると学習が落ちるので、
        ここで詰める・切るのどちらもできるようにしておく。
        """
        augmentor = self._get_augmentor()
        if augmentor is None:
            return audio
        target_frames = int(audio.shape[-1])
        augmented = augmentor(audio.numpy())
        result = torch.from_numpy(np.ascontiguousarray(augmented, dtype=np.float32))
        if result.ndim == 1:
            result = result.unsqueeze(0).expand(2, -1)
        if int(result.shape[0]) != int(audio.shape[0]):
            result = result[: audio.shape[0]]
        current_frames = int(result.shape[-1])
        if current_frames < target_frames:
            result = torch.nn.functional.pad(result, (0, target_frames - current_frames))
        elif current_frames > target_frames:
            result = result[:, :target_frames]
        return result.contiguous()

    def __len__(self) -> int:
        return len(self.windows) if self.sampling == "all" else len(self.units)

    def _notes(self, source: RefinementSourceRecord) -> RefinementNoteTable:
        if source.midi_path is None:
            return load_refinement_note_table(None)
        if source.midi_path not in self._note_cache:
            # 自動で直せた MIDI の通知は、学習中に出しても対処のしようがなく進捗表示を
            # 埋めるだけなので出さない。読めずに捨てた MIDI の警告はそのまま出る。
            self._note_cache[source.midi_path] = load_refinement_note_table(source.midi_path, warn_repairs=False)
        return self._note_cache[source.midi_path]

    def _sample_cross_song_candidates(
        self,
        unit_index: int,
        *,
        sample_size: int = 4,
        max_attempts: int = 16,
    ) -> tuple[int, ...]:
        """同じステムグループの、別の曲の unit を抽選で選ぶ。

        候補が最大で数万件になるため全対を保持せず、その都度引いて曲が違うものだけ残す。
        引けなければ空を返し、呼び出し側が同一曲の候補へ戻る。
        """
        group_units = self.stem_group_unit_indices.get(self.unit_stem_groups[unit_index], ())
        if len(group_units) < 2:
            return ()
        song_id = self.unit_song_ids[unit_index]
        selected: list[int] = []
        seen = {unit_index}
        for _ in range(max_attempts):
            if len(selected) >= sample_size:
                break
            candidate = random.choice(group_units)
            if candidate in seen or self.unit_song_ids[candidate] == song_id:
                continue
            seen.add(candidate)
            selected.append(candidate)
        return tuple(selected)

    def _composite_candidates_for(self, unit_index: int) -> tuple[int, ...]:
        """この unit に混ぜる相手の候補を返す。

        cross_song_composite_ratio の確率で別曲混合を選ぶ。同一曲の相手がいない unit
        （実データの大半がこれに当たる）は、別曲混合が有効なら常にそちらを使う。
        """
        same_song_candidates = self.composite_candidates[unit_index]
        if self.cross_song_composite_ratio <= 0.0:
            return same_song_candidates
        if not same_song_candidates or random.random() < self.cross_song_composite_ratio:
            cross_song_candidates = self._sample_cross_song_candidates(unit_index)
            if cross_song_candidates:
                return cross_song_candidates
        return same_song_candidates

    def _random_window_start_for_sources(self, sources: tuple[RefinementSourceRecord, ...]) -> float:
        """学習用の窓位置。完全なランダムではなく、必ずノートが入るように選ぶ。

        無音区間ばかり引くと学習が進まないので、まずノートを 1 つ選び、それが窓に
        収まる範囲から開始位置を引く。
        """
        duration_seconds = max(source.duration_seconds for source in sources)
        max_start = max(0.0, duration_seconds - self.window_seconds)
        if max_start <= 0.0:
            return 0.0
        active_tables = [self._notes(source) for source in sources]
        active_tables = [table for table in active_tables if table.note_count]
        if active_tables:
            notes = random.choice(active_tables)
            note_time = float(notes.start_seconds[random.randrange(notes.note_count)])
            lower = max(0.0, note_time - self.window_seconds + 1e-4)
            upper = min(note_time, max_start)
            if upper >= lower:
                return random.uniform(lower, upper)
        return random.uniform(0.0, max_start)

    def _fixed_window_start_for_sources(self, sources: tuple[RefinementSourceRecord, ...]) -> float:
        """検証用の窓位置。音源 ID のハッシュで決めるので毎エポック同じ場所になる。"""
        duration_seconds = max(source.duration_seconds for source in sources)
        max_start = max(0.0, duration_seconds - self.window_seconds)
        if max_start <= 0.0:
            return 0.0
        notes = [self._notes(source) for source in sources]
        note_times = (
            np.concatenate([table.start_seconds for table in notes if table.note_count])
            if any(table.note_count for table in notes)
            else np.zeros(0, dtype=np.float32)
        )
        identity = "+".join(sorted(source.source_id for source in sources))
        if note_times.size:
            note_index = _source_hash(identity) % int(note_times.size)
            note_time = float(np.sort(note_times)[note_index])
            return min(max(0.0, note_time - 0.5 * self.window_seconds), max_start)
        fraction = _source_hash(identity) / float(1 << 63)
        return fraction * max_start

    def _load_window_audio(self, source: RefinementSourceRecord, start_seconds: float) -> tuple[torch.Tensor, int]:
        audio, valid_frames = _load_audio_window(
            source.audio_path,
            start_seconds=start_seconds,
            window_seconds=self.window_seconds,
            target_sample_rate=self.sample_rate,
        )
        # 拡張は音源ごとに、混ぜる前に掛ける（本家 AMT の StemDataset と同じ順番）。
        # 混ぜた後に掛けると全音源が同じ EQ・同じ部屋を通ったことになり、
        # 「別々に録られた音が 1 つのミックスに同居している」状況を学べない。
        if self.sampling == "random" and random.random() < self.augment_probability:
            audio = self._augment_window(audio)
        if self.sampling == "random" and self.random_gain_db > 0.0:
            gain_db = random.uniform(-self.random_gain_db, self.random_gain_db)
            audio = audio * float(10.0 ** (gain_db / 20.0))
        return audio, valid_frames

    def _source_note_queries(self, source: RefinementSourceRecord, start_seconds: float) -> NoteQueries:
        """1 音源について、窓に発音が入るノートを取り出して正解を付ける。

        note_target_mode が "midi" ならノートごとに MIDI の program を正解にし
        （1 音源に複数楽器がある場合に効く）、それ以外は音源全体のラベルを全ノートへ配る。
        """
        notes = self._notes(source)
        end_seconds = start_seconds + self.window_seconds
        active = (notes.start_seconds >= start_seconds) & (notes.start_seconds < end_seconds)
        indices = np.flatnonzero(active)
        count = len(indices)
        class_target = torch.zeros(count, NUM_INSTRUMENT_CLASSES, dtype=torch.bool)
        family_target = torch.zeros(count, len(FAMILY_NAMES), dtype=torch.bool)
        if source.note_target_mode == "midi":
            primary = torch.from_numpy(notes.prior_class_id[indices].astype(np.int64))
            if count:
                class_target[torch.arange(count, dtype=torch.long), primary] = True
                for row, class_id in enumerate(primary.tolist()):
                    family_target[row, family_id_for_class_name(INSTRUMENT_CLASSES[class_id])] = True
            # ノートごとに楽器が違いうるので、対照学習の鍵も楽器ごとに分ける。
            source_hash = torch.tensor(
                [_source_hash(f"{source.source_id}:{class_id}") for class_id in primary.tolist()],
                dtype=torch.long,
            )
        else:
            if count:
                class_target[:, list(source.target_class_ids)] = True
                family_target[:, list(source.target_family_ids)] = True
            single = source.target_class_ids[0] if len(source.target_class_ids) == 1 else -1
            primary = torch.full((count,), single, dtype=torch.long)
            source_hash = torch.full((count,), _source_hash(source.source_id), dtype=torch.long)
        return NoteQueries(
            start=torch.from_numpy((notes.start_seconds[indices] - start_seconds).astype(np.float32)),
            end=torch.from_numpy((notes.end_seconds[indices] - start_seconds).astype(np.float32)),
            pitch=torch.from_numpy(notes.pitch[indices].astype(np.int64)),
            class_target=class_target,
            family_target=family_target,
            primary=primary,
            source_hash=source_hash,
            weight=torch.ones(count, dtype=torch.float32),
            contrastive=torch.ones(count, dtype=torch.bool),
        )

    def _build_sources_view(
        self,
        sources: tuple[RefinementSourceRecord, ...],
        start_seconds: float,
    ) -> dict[str, Any]:
        """音源を足し合わせて学習サンプル 1 つを組み立てる。

        sources が 1 本なら単独サンプル、複数なら混合サンプル（composite）になる。
        """
        stem_group = inference_stem_group(sources[0].stem_name)
        if any(inference_stem_group(source.stem_name) != stem_group for source in sources):
            raise ValueError("A training mixture cannot cross inference stem groups")

        audio_parts: list[torch.Tensor] = []
        query_parts: list[NoteQueries] = []
        valid_frames = 0
        for source in sources:
            audio, current_valid = self._load_window_audio(source, start_seconds)
            audio_parts.append(audio)
            valid_frames = max(valid_frames, current_valid)
            query_parts.append(self._source_note_queries(source, start_seconds))
        # 音源を足す。本数が増えても音量が上がりすぎないよう sqrt で割る。
        audio = torch.stack(audio_parts).sum(dim=0) / math.sqrt(len(audio_parts))
        queries = _merge_ambiguous_note_queries(
            NoteQueries.concat(query_parts).sorted_by_time(),
            onset_tolerance_seconds=self.ambiguous_note_onset_tolerance_seconds,
        )

        # 窓全体のラベルは、混ぜた音源すべてのラベルの和集合。
        class_target = torch.zeros(NUM_INSTRUMENT_CLASSES, dtype=torch.bool)
        family_target = torch.zeros(len(FAMILY_NAMES), dtype=torch.bool)
        for source in sources:
            class_target[list(source.target_class_ids)] = True
            family_target[list(source.target_family_ids)] = True
        # 単独音源かつラベルが 1 つのときだけ「窓の楽器はこれ」と確定できる（精度計算に使う）。
        single_source = len(sources) == 1 and len(sources[0].target_class_ids) == 1
        primary = sources[0].target_class_ids[0] if single_source else -1
        source_ids = tuple(sorted(source.source_id for source in sources))
        return {
            "audio": audio,
            "valid_audio_frames": int(valid_frames),
            "note_start_seconds": queries.start,
            "note_end_seconds": queries.end,
            "note_pitch": queries.pitch,
            # AMT が付けた楽器は渡さない。常に「未知」を意味する -1。
            "note_prior_class": torch.full_like(queries.pitch, -1),
            "note_confidence": torch.ones(len(queries), dtype=torch.float32),
            "note_class_target_mask": queries.class_target,
            "note_family_target_mask": queries.family_target,
            "note_primary_class_id": queries.primary,
            "note_source_hash": queries.source_hash,
            "note_sample_weight": queries.weight,
            "note_contrastive_mask": queries.contrastive,
            "class_target_mask": class_target,
            "family_target_mask": family_target,
            "primary_class_id": int(primary),
            "stem_context_id": stem_context_id(stem_group),
            "sample_weight": 1.0,
            # 混合サンプルの「窓全体の楽器」は候補が複数あって教師として弱いので、
            # 窓分類 loss からは重み 0 で外す（ノート単位の loss では使う）。
            "window_sample_weight": 1.0 if len(sources) == 1 else 0.0,
            "source_hash": _source_hash("+".join(source_ids)),
            "source_id": "+".join(source_ids),
            "song_id": sources[0].song_id,
            "stem_name": stem_group,
            "track_type": "+".join(source.track_type for source in sources),
            "is_composite": len(sources) > 1,
            "window_start_seconds": float(start_seconds),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        # 評価: 事前に並べた窓を 1 つ返す。
        if self.sampling == "all":
            window = self.windows[index]
            sources = tuple(self.sources[source_index] for source_index in self.units[window.unit_index])
            return self._build_sources_view(sources, window.start_seconds)
        base_unit = self.units[index]
        # 検証: 混合せず、固定位置の窓を 1 つ返す。
        if self.sampling == "fixed":
            sources = tuple(self.sources[source_index] for source_index in base_unit)
            return self._build_sources_view(sources, self._fixed_window_start_for_sources(sources))
        # 学習: 確率で他 unit を混ぜてから、同じ音源の窓を windows_per_source 個返す。
        unit_indices = [index]
        candidates = self._composite_candidates_for(index)
        if candidates and random.random() < self.composite_probability:
            shuffled = list(candidates)
            random.shuffle(shuffled)
            source_count = len(base_unit)
            eligible = []
            for unit_index in shuffled:
                unit_size = len(self.units[unit_index])
                if self.max_mixture_sources is None or source_count + unit_size <= self.max_mixture_sources:
                    eligible.append(unit_index)
                    source_count += unit_size
            if eligible:
                extra_count = random.randint(1, len(eligible))
                unit_indices.extend(eligible[:extra_count])
        source_indices = [source_index for unit_index in unit_indices for source_index in self.units[unit_index]]
        sources = tuple(self.sources[source_index] for source_index in source_indices)
        return {
            "views": [
                self._build_sources_view(sources, self._random_window_start_for_sources(sources))
                for _ in range(self.windows_per_source)
            ]
        }
