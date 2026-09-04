from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

from instrument_agnostic_amt.velocity.data.midi import (
    canonicalize_amt_midi,
    load_midi_note_table,
)

from .config import VelocityPipelineConfig, load_pipeline_config
from .data.index import VelocitySourceItem, discover_amt_cbnet_items
from .data.pseudo import (
    PseudoLabelSummary,
    load_pseudo_label_summary,
    prepare_pseudo_label_file,
)


LOGGER = logging.getLogger(__name__)
VELOCITY_ROOT = (
    Path(__file__).resolve().parents[2] / "instrument_agnostic_amt" / "velocity"
)
DEFAULT_CONFIG = VELOCITY_ROOT / "configs" / "monalisa_gm.json"
DEFAULT_OUTPUT_ROOT = VELOCITY_ROOT / "artifacts" / "amt_cbnet"

MANIFEST_FIELDS = (
    "song_id",
    "stem_name",
    "wav_path",
    "midi_path",
    "merged_midi_path",
    "has_midi",
    "canonical_midi_path",
    "pseudo_label_path",
    "note_count",
    "valid_note_count",
    "duration_seconds",
    "sample_rate",
    "audio_rms_dbfs",
    "audio_peak",
    "active_ratio",
    "active_level_dbfs",
    "active_rms_dbfs",
    "relative_active_level_db",
    "level_confidence",
    "error",
)


def _float_or_blank(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else ""


def _base_row(item: VelocitySourceItem) -> dict[str, Any]:
    return {
        "song_id": item.song_id,
        "stem_name": item.stem_name,
        "wav_path": str(item.wav_path),
        "midi_path": str(item.midi_path) if item.midi_path is not None else "",
        "merged_midi_path": (
            str(item.merged_midi_path) if item.merged_midi_path is not None else ""
        ),
        "has_midi": int(item.has_midi),
        "canonical_midi_path": "",
        "pseudo_label_path": "",
        "note_count": "",
        "valid_note_count": "",
        "duration_seconds": "",
        "sample_rate": "",
        "audio_rms_dbfs": "",
        "audio_peak": "",
        "active_ratio": "",
        "active_level_dbfs": "",
        "active_rms_dbfs": "",
        "relative_active_level_db": "",
        "level_confidence": "",
        "error": "",
    }


def _summary_fields(summary: PseudoLabelSummary) -> dict[str, Any]:
    return {
        "note_count": summary.note_count,
        "valid_note_count": summary.valid_note_count,
        "duration_seconds": round(summary.duration_seconds, 6),
        "sample_rate": summary.sample_rate,
        "audio_rms_dbfs": _float_or_blank(summary.audio_rms_dbfs),
        "audio_peak": summary.audio_peak,
        "active_ratio": summary.active_ratio,
        "active_level_dbfs": _float_or_blank(summary.active_level_dbfs),
        "active_rms_dbfs": _float_or_blank(summary.active_rms_dbfs),
        "level_confidence": summary.level_confidence,
    }


def _prepare_one(
    item: VelocitySourceItem,
    *,
    output_root: Path,
    pipeline_config: VelocityPipelineConfig,
    write_canonical_midi: bool,
    overwrite: bool,
) -> dict[str, Any]:
    row = _base_row(item)
    pseudo_path = output_root / "pseudo_labels" / item.song_id / f"{item.stem_name}.npz"
    canonical_path = (
        output_root / "canonical_midis" / item.song_id / f"{item.stem_name}.mid"
    )
    try:
        if item.midi_path is not None and write_canonical_midi:
            if overwrite or not canonical_path.is_file():
                canonicalize_amt_midi(
                    item.midi_path,
                    canonical_path,
                    canonical_velocity=pipeline_config.canonical_velocity,
                )
            row["canonical_midi_path"] = str(canonical_path.resolve())

        if pseudo_path.is_file() and not overwrite:
            summary = load_pseudo_label_summary(pseudo_path)
        else:
            note_table = load_midi_note_table(item.midi_path)
            summary = prepare_pseudo_label_file(
                item.wav_path,
                note_table,
                pseudo_path,
                config=pipeline_config.pseudo_labels,
            )
        row["pseudo_label_path"] = str(pseudo_path.resolve())
        row.update(_summary_fields(summary))
    except Exception as error:  # keep the full audit manifest even for broken files
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def _prepare_worker(arguments: tuple[Any, ...]) -> dict[str, Any]:
    item, output_root, config, write_canonical_midi, overwrite = arguments
    return _prepare_one(
        item,
        output_root=output_root,
        pipeline_config=config,
        write_canonical_midi=write_canonical_midi,
        overwrite=overwrite,
    )


def _add_relative_levels(rows: list[dict[str, Any]]) -> None:
    rows_by_song: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_song.setdefault(str(row["song_id"]), []).append(row)
    for song_rows in rows_by_song.values():
        usable = [
            float(row["active_level_dbfs"])
            for row in song_rows
            if row["active_level_dbfs"] != ""
            and float(row["level_confidence"] or 0.0) > 0.0
        ]
        if not usable:
            continue
        usable.sort()
        middle = len(usable) // 2
        center = (
            usable[middle]
            if len(usable) % 2
            else 0.5 * (usable[middle - 1] + usable[middle])
        )
        for row in song_rows:
            if row["active_level_dbfs"] != "":
                row["relative_active_level_db"] = (
                    float(row["active_level_dbfs"]) - center
                )


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _serializable_config(config: VelocityPipelineConfig) -> dict[str, Any]:
    return asdict(config)


def _write_summary(
    path: Path,
    *,
    source_root: Path,
    output_root: Path,
    rows: list[dict[str, Any]],
    config: VelocityPipelineConfig,
    mode: str,
) -> None:
    payload = {
        "schema_version": 1,
        "mode": mode,
        "source_root": str(source_root.resolve()),
        "output_root": str(output_root.resolve()),
        "item_count": len(rows),
        "song_count": len({str(row["song_id"]) for row in rows}),
        "missing_midi_count": sum(not int(row["has_midi"]) for row in rows),
        "error_count": sum(bool(row["error"]) for row in rows),
        "empty_midi_count": sum(row["note_count"] == 0 for row in rows),
        "valid_note_count": sum(int(row["valid_note_count"] or 0) for row in rows),
        "config": _serializable_config(config),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare AMT-CBNet stem/MIDI pairs for the independent velocity pipeline."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("index", "pseudo"), default="pseudo")
    parser.add_argument("--limit-songs", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, os.cpu_count() or 1)),
    )
    parser.add_argument("--write-canonical-midi", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    config = load_pipeline_config(args.config)
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    items = discover_amt_cbnet_items(
        source_root,
        limit_songs=args.limit_songs,
    )
    LOGGER.info(
        "Discovered %d stems across %d songs",
        len(items),
        len({item.song_id for item in items}),
    )

    if args.mode == "index":
        rows = [_base_row(item) for item in items]
    else:
        tasks = [
            (
                item,
                output_root,
                config,
                bool(args.write_canonical_midi),
                bool(args.overwrite),
            )
            for item in items
        ]
        if args.workers == 1:
            rows = [
                _prepare_worker(task)
                for task in tqdm(tasks, desc="velocity pseudo labels")
            ]
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                rows = list(
                    tqdm(
                        executor.map(_prepare_worker, tasks, chunksize=1),
                        total=len(tasks),
                        desc="velocity pseudo labels",
                    )
                )
        _add_relative_levels(rows)

    rows.sort(key=lambda row: (str(row["song_id"]).casefold(), row["stem_name"]))
    manifest_path = output_root / "manifest.csv"
    summary_path = output_root / "summary.json"
    _write_manifest(manifest_path, rows)
    _write_summary(
        summary_path,
        source_root=source_root,
        output_root=output_root,
        rows=rows,
        config=config,
        mode=args.mode,
    )
    LOGGER.info("Wrote %s", manifest_path)
    LOGGER.info("Wrote %s", summary_path)


if __name__ == "__main__":
    main()
