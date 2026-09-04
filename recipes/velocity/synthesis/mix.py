"""Mixing utilities for rendered velocity training stems."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional


DATASET_FIELDS = (
    "example_id",
    "song_id",
    "variation",
    "mixture_path",
    "stem_count",
    "duration_seconds",
    "sample_rate",
    "master_gain_db",
    "peak_limiter_gain_db",
    "final_peak_dbfs",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _resolve(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _db_to_amplitude(value_db: float) -> float:
    return float(10.0 ** (float(value_db) / 20.0))


def _amplitude_to_db(amplitude: float) -> float:
    return float(20.0 * math.log10(max(float(amplitude), 1e-12)))


def mix_rendered_stems(
    render_manifest_path: str | Path,
    examples_path: str | Path,
    *,
    peak_limit_dbfs: float,
    output_sample_rate: int | None = None,
    overwrite: bool = False,
    skip_missing: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if peak_limit_dbfs > 0.0:
        raise ValueError("peak_limit_dbfs must not exceed 0 dBFS")
    if output_sample_rate is not None and output_sample_rate <= 0:
        raise ValueError("output_sample_rate must be positive")
    render_manifest = Path(render_manifest_path).expanduser().resolve()
    example_manifest = Path(examples_path).expanduser().resolve()
    render_rows = _read_csv(render_manifest)
    example_rows = _read_csv(example_manifest)
    render_by_example: dict[str, list[dict[str, str]]] = {}
    for row in render_rows:
        render_by_example.setdefault(row["example_id"], []).append(row)

    dataset_rows: list[dict[str, Any]] = []
    missing_count = 0
    skipped_count = 0
    for example in example_rows:
        example_id = example["example_id"]
        stems = render_by_example.get(example_id, [])
        if len(stems) != int(example["stem_count"]):
            raise ValueError(f"Stem count mismatch for {example_id}")
        mixture_path = _resolve(example["mixture_path"], example_manifest.parent)
        mixture_exists = mixture_path.is_file()
        mixture_rate_matches = (
            not mixture_exists
            or output_sample_rate is None
            or int(sf.info(str(mixture_path)).samplerate) == int(output_sample_rate)
        )

        loaded: list[tuple[np.ndarray, int, float]] = []
        stem_paths: list[Path] = []
        missing = False
        for stem in stems:
            wav_path = _resolve(stem["rendered_stem_path"], render_manifest.parent)
            if not wav_path.is_file():
                missing_count += 1
                missing = True
                break
            stem_paths.append(wav_path)
            waveform, sample_rate = sf.read(
                str(wav_path),
                dtype="float32",
                always_2d=True,
            )
            loaded.append((waveform, int(sample_rate), float(stem["stem_gain_db"])))
        if missing:
            if skip_missing:
                continue
            raise FileNotFoundError(f"A rendered stem is missing for {example_id}")
        if not loaded:
            continue
        newest_input_mtime = max(
            [
                render_manifest.stat().st_mtime_ns,
                example_manifest.stat().st_mtime_ns,
                *(path.stat().st_mtime_ns for path in stem_paths),
            ]
        )
        write_mixture = (
            overwrite
            or not mixture_exists
            or not mixture_rate_matches
            or mixture_path.stat().st_mtime_ns < newest_input_mtime
        )
        if not write_mixture:
            skipped_count += 1
        sample_rates = {item[1] for item in loaded}
        if len(sample_rates) != 1:
            raise ValueError(f"Sample-rate mismatch for {example_id}")
        render_sample_rate = sample_rates.pop()
        channel_count = max(item[0].shape[1] for item in loaded)
        if any(item[0].shape[1] not in (1, channel_count) for item in loaded):
            raise ValueError(f"Channel-count mismatch for {example_id}")
        sample_count = max(item[0].shape[0] for item in loaded)
        mixture = np.zeros((sample_count, channel_count), dtype=np.float64)
        for waveform, _, gain_db in loaded:
            if waveform.shape[1] == 1 and channel_count > 1:
                waveform = np.repeat(waveform, channel_count, axis=1)
            mixture[: waveform.shape[0]] += waveform.astype(
                np.float64
            ) * _db_to_amplitude(gain_db)

        master_gain_db = float(example["master_gain_db"])
        mixture *= _db_to_amplitude(master_gain_db)
        sample_rate = (
            render_sample_rate
            if output_sample_rate is None
            else int(output_sample_rate)
        )
        if sample_rate != render_sample_rate:
            waveform = torch.from_numpy(mixture.astype(np.float32, copy=False).T.copy())
            with torch.no_grad():
                waveform = audio_functional.resample(
                    waveform,
                    render_sample_rate,
                    sample_rate,
                )
            mixture = waveform.T.numpy().astype(np.float64, copy=False)
        sample_count = int(mixture.shape[0])
        peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
        peak_limit = _db_to_amplitude(peak_limit_dbfs)
        limiter_gain_db = 0.0
        if peak > peak_limit and peak > 0.0:
            limiter = peak_limit / peak
            mixture *= limiter
            limiter_gain_db = _amplitude_to_db(limiter)
        final_peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
        if write_mixture:
            mixture_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                str(mixture_path),
                mixture.astype(np.float32),
                sample_rate,
                subtype="FLOAT",
            )
        dataset_rows.append(
            {
                "example_id": example_id,
                "song_id": example["song_id"],
                "variation": int(example["variation"]),
                "mixture_path": Path(example["mixture_path"]).as_posix(),
                "stem_count": len(stems),
                "duration_seconds": sample_count / float(sample_rate),
                "sample_rate": sample_rate,
                "master_gain_db": master_gain_db,
                "peak_limiter_gain_db": limiter_gain_db,
                "final_peak_dbfs": _amplitude_to_db(final_peak),
            }
        )
    return dataset_rows, {
        "mixed_example_count": len(dataset_rows),
        "skipped_existing_count": skipped_count,
        "missing_stem_count": missing_count,
    }


def write_dataset_manifest(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=DATASET_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
