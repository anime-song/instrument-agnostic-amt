from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from instrument_agnostic_amt.velocity.data.midi import canonicalize_amt_midi
from .config import SyntheticDataConfig
from .midi import write_target_velocity_midi
from .sampling import derived_seed, sample_note_velocities, sample_stem_gains


RENDER_FIELDS = (
    "example_id",
    "song_id",
    "variation",
    "stem_name",
    "input_midi_path",
    "target_midi_path",
    "rendered_stem_path",
    "label_path",
    "note_count",
    "pseudo_valid_count",
    "base_relative_level_db",
    "stem_gain_db",
    "sample_rate",
    "render_synth_gain",
    "sample_seed",
)

EXAMPLE_FIELDS = (
    "example_id",
    "song_id",
    "variation",
    "stem_count",
    "mixture_path",
    "master_gain_db",
)


def _safe_component(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9._$-]+", value):
        return value
    cleaned = re.sub(r"[^A-Za-z0-9._$-]+", "_", value).strip("._") or "item"
    suffix = hashlib.blake2b(value.encode("utf-8"), digest_size=4).hexdigest()
    return f"{cleaned}_{suffix}"


def _resolve_input_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.expanduser().resolve()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _optional_float(value: object) -> float | None:
    text = str(value or "").strip()
    return float(text) if text else None


def _load_ready_rows(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        source_rows = list(csv.DictReader(file))
    ready: list[dict[str, str]] = []
    for row in source_rows:
        if str(row.get("error", "")).strip():
            continue
        if str(row.get("has_midi", "0")) != "1":
            continue
        midi_text = str(row.get("midi_path", "")).strip()
        pseudo_text = str(row.get("pseudo_label_path", "")).strip()
        if not midi_text or not pseudo_text:
            continue
        midi_path = _resolve_input_path(midi_text, manifest_path.parent)
        pseudo_path = _resolve_input_path(pseudo_text, manifest_path.parent)
        if not midi_path.is_file() or not pseudo_path.is_file():
            continue
        copied = dict(row)
        copied["midi_path"] = str(midi_path)
        copied["pseudo_label_path"] = str(pseudo_path)
        ready.append(copied)
    return ready


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_label_npz(
    path: Path,
    *,
    pseudo_path: Path,
    velocity_sample: Any,
    stem_gain_db: float,
    base_relative_level_db: float | None,
    sample_seed: int,
) -> tuple[int, int]:
    pseudo_stat = pseudo_path.stat()
    with np.load(pseudo_path, allow_pickle=False) as source:
        arrays = {
            "note_start_seconds": source["note_start_seconds"].astype(np.float64),
            "note_end_seconds": source["note_end_seconds"].astype(np.float64),
            "note_pitch": source["note_pitch"].astype(np.int16),
            "note_program": source["note_program"].astype(np.int16),
            "note_is_drum": source["note_is_drum"].astype(np.bool_),
            "note_track_index": source["note_track_index"].astype(np.int16),
            "source_pseudo_rank": source["pseudo_velocity_rank"].astype(np.float32),
            "source_pseudo_confidence": source["pseudo_confidence"].astype(np.float32),
            "source_pseudo_valid": source["pseudo_valid"].astype(np.bool_),
        }
    note_count = int(arrays["note_pitch"].size)
    valid_count = int(arrays["source_pseudo_valid"].sum())
    arrays.update(
        {
            "target_velocity": velocity_sample.target_velocity.astype(np.int16),
            "rank_used": velocity_sample.filled_rank.astype(np.float32),
            "rank_source": velocity_sample.rank_source.astype(np.int8),
            "independently_randomized": (
                velocity_sample.independently_randomized.astype(np.bool_)
            ),
            "velocity_style_center": np.asarray(
                velocity_sample.style_center,
                dtype=np.float32,
            ),
            "velocity_style_span": np.asarray(
                velocity_sample.style_span,
                dtype=np.float32,
            ),
            "stem_gain_db": np.asarray(stem_gain_db, dtype=np.float32),
            "base_relative_level_db": np.asarray(
                np.nan if base_relative_level_db is None else base_relative_level_db,
                dtype=np.float32,
            ),
            "sample_seed": np.asarray(sample_seed, dtype=np.uint64),
            "source_pseudo_size_bytes": np.asarray(
                pseudo_stat.st_size,
                dtype=np.int64,
            ),
            "source_pseudo_mtime_ns": np.asarray(
                pseudo_stat.st_mtime_ns,
                dtype=np.int64,
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(path)
    return note_count, valid_count


def prepare_synthetic_plan(
    pseudo_manifest_path: str | Path,
    output_root: str | Path,
    *,
    config: SyntheticDataConfig,
    variations: int,
    seed: int,
    min_stems: int = 2,
    limit_songs: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if variations < 1:
        raise ValueError("variations must be positive")
    if min_stems < 1:
        raise ValueError("min_stems must be positive")
    manifest_path = Path(pseudo_manifest_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    rows_by_song: dict[str, list[dict[str, str]]] = {}
    for row in _load_ready_rows(manifest_path):
        rows_by_song.setdefault(str(row["song_id"]), []).append(row)
    song_ids = sorted(
        (song_id for song_id, rows in rows_by_song.items() if len(rows) >= min_stems),
        key=str.casefold,
    )
    if limit_songs is not None:
        if limit_songs < 1:
            raise ValueError("limit_songs must be positive")
        song_ids = song_ids[:limit_songs]

    render_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for song_id in song_ids:
        song_rows = sorted(rows_by_song[song_id], key=lambda row: row["stem_name"])
        safe_song_id = _safe_component(song_id)
        for row in song_rows:
            input_midi_path = (
                destination / "input_midis" / safe_song_id / f"{row['stem_name']}.mid"
            )
            if overwrite or not input_midi_path.is_file():
                canonicalize_amt_midi(
                    row["midi_path"],
                    input_midi_path,
                    canonical_velocity=config.canonical_velocity,
                )

        base_levels = {
            row["stem_name"]: _optional_float(row.get("relative_active_level_db"))
            for row in song_rows
        }
        for variation in range(variations):
            example_id = f"{song_id}__v{variation:03d}"
            safe_variation = f"v{variation:03d}"
            example_seed = derived_seed(seed, song_id, variation, "example")
            example_rng = np.random.default_rng(example_seed)
            stem_gains = sample_stem_gains(
                base_levels,
                rng=example_rng,
                config=config,
            )
            master_gain_db = (
                float(
                    example_rng.uniform(
                        config.master_gain_min_db,
                        config.master_gain_max_db,
                    )
                )
                if config.use_gain_augmentation
                else 0.0
            )
            example_rows.append(
                {
                    "example_id": example_id,
                    "song_id": song_id,
                    "variation": variation,
                    "stem_count": len(song_rows),
                    "mixture_path": (
                        Path("mixtures") / safe_song_id / f"{safe_variation}.wav"
                    ).as_posix(),
                    "master_gain_db": master_gain_db,
                }
            )
            for row in song_rows:
                stem_name = str(row["stem_name"])
                sample_seed = derived_seed(seed, song_id, variation, stem_name)
                rng = np.random.default_rng(sample_seed)
                pseudo_path = Path(row["pseudo_label_path"])
                with np.load(pseudo_path, allow_pickle=False) as pseudo:
                    velocity_sample = sample_note_velocities(
                        pseudo_rank=pseudo["pseudo_velocity_rank"],
                        pseudo_valid=pseudo["pseudo_valid"],
                        track_index=pseudo["note_track_index"],
                        note_start_seconds=pseudo["note_start_seconds"],
                        rng=rng,
                        config=config,
                    )
                target_midi_path = (
                    destination
                    / "target_midis"
                    / safe_song_id
                    / safe_variation
                    / f"{stem_name}.mid"
                )
                label_path = (
                    destination
                    / "labels"
                    / safe_song_id
                    / safe_variation
                    / f"{stem_name}.npz"
                )
                rendered_stem_path = (
                    destination
                    / "rendered_stems"
                    / safe_song_id
                    / safe_variation
                    / f"{stem_name}.wav"
                )
                regenerate = (
                    overwrite
                    or not target_midi_path.is_file()
                    or not label_path.is_file()
                )
                if not regenerate:
                    pseudo_stat = pseudo_path.stat()
                    with np.load(label_path, allow_pickle=False) as labels:
                        fingerprint_fields = {
                            "sample_seed",
                            "source_pseudo_size_bytes",
                            "source_pseudo_mtime_ns",
                            "target_velocity",
                            "stem_gain_db",
                        }
                        regenerate = not fingerprint_fields.issubset(labels.files)
                        if not regenerate:
                            regenerate = (
                                int(labels["sample_seed"]) != sample_seed
                                or int(labels["source_pseudo_size_bytes"])
                                != pseudo_stat.st_size
                                or int(labels["source_pseudo_mtime_ns"])
                                != pseudo_stat.st_mtime_ns
                                or not np.array_equal(
                                    labels["target_velocity"],
                                    velocity_sample.target_velocity,
                                )
                                or not np.isclose(
                                    float(labels["stem_gain_db"]),
                                    stem_gains[stem_name],
                                )
                            )
                if regenerate:
                    write_target_velocity_midi(
                        row["midi_path"],
                        target_midi_path,
                        velocity_sample.target_velocity,
                    )
                    note_count, valid_count = _write_label_npz(
                        label_path,
                        pseudo_path=pseudo_path,
                        velocity_sample=velocity_sample,
                        stem_gain_db=stem_gains[stem_name],
                        base_relative_level_db=base_levels[stem_name],
                        sample_seed=sample_seed,
                    )
                else:
                    with np.load(label_path, allow_pickle=False) as labels:
                        note_count = int(labels["target_velocity"].size)
                        valid_count = int(labels["source_pseudo_valid"].sum())
                input_midi_path = (
                    destination / "input_midis" / safe_song_id / f"{stem_name}.mid"
                )
                render_rows.append(
                    {
                        "example_id": example_id,
                        "song_id": song_id,
                        "variation": variation,
                        "stem_name": stem_name,
                        "input_midi_path": _relative(input_midi_path, destination),
                        "target_midi_path": _relative(target_midi_path, destination),
                        "rendered_stem_path": _relative(
                            rendered_stem_path, destination
                        ),
                        "label_path": _relative(label_path, destination),
                        "note_count": note_count,
                        "pseudo_valid_count": valid_count,
                        "base_relative_level_db": (
                            ""
                            if base_levels[stem_name] is None
                            else base_levels[stem_name]
                        ),
                        "stem_gain_db": stem_gains[stem_name],
                        "sample_rate": config.render_sample_rate,
                        "render_synth_gain": config.render_synth_gain,
                        "sample_seed": sample_seed,
                    }
                )

    destination.mkdir(parents=True, exist_ok=True)
    _write_csv(destination / "render_manifest.csv", RENDER_FIELDS, render_rows)
    _write_csv(destination / "examples.csv", EXAMPLE_FIELDS, example_rows)
    summary = {
        "schema_version": 1,
        "seed": int(seed),
        "requested_variations": int(variations),
        "song_count": len(song_ids),
        "example_count": len(example_rows),
        "render_job_count": len(render_rows),
        "config": asdict(config),
    }
    (destination / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
