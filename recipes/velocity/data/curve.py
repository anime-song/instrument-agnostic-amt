"""SoundFont velocity-curve analysis."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf


POWER_FLOOR = 1e-12


@dataclass(frozen=True)
class CalibrationAnalysisConfig:
    """Measurement settings for isolated velocity-sweep notes."""

    frame_ms: float = 10.0
    signal_start_ms: float = 5.0
    signal_end_ms: float = 350.0
    noise_start_ms: float = -140.0
    noise_end_ms: float = -20.0
    frame_percentile: float = 95.0
    min_signal_dbfs: float = -90.0
    low_snr_db: float = 3.0
    clipping_threshold: float = 0.999

    def __post_init__(self) -> None:
        if self.frame_ms <= 0.0:
            raise ValueError("frame_ms must be positive")
        if self.signal_end_ms <= self.signal_start_ms:
            raise ValueError("the signal window must have positive length")
        if self.noise_end_ms <= self.noise_start_ms:
            raise ValueError("the noise window must have positive length")
        if not 0.0 <= self.frame_percentile <= 100.0:
            raise ValueError("frame_percentile must be within 0..100")
        if not 0.0 < self.clipping_threshold <= 1.0:
            raise ValueError("clipping_threshold must be within (0, 1]")


@dataclass(frozen=True)
class SweepEventRecord:
    program: int
    is_drum: bool
    pitch: int
    velocity: int
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class RenderTarget:
    program: int
    is_drum: bool
    wav_path: Path


CURVE_FIELDS = (
    "program",
    "is_drum",
    "pitch",
    "velocity",
    "start_seconds",
    "end_seconds",
    "wav_file",
    "sample_rate",
    "raw_level_dbfs",
    "rms_dbfs",
    "peak_dbfs",
    "pre_note_level_dbfs",
    "snr_db",
    "valid",
    "clipped",
    "fit_observed",
    "tail_contaminated",
    "fitted_level_dbfs",
    "relative_level_db",
    "fit_adjustment_db",
    "curve_dynamic_range_db",
)


def _resolve_manifest_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.expanduser().resolve()


def load_sweep_events(path: str | Path) -> list[SweepEventRecord]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    events = [
        SweepEventRecord(
            program=int(row["program"]),
            is_drum=bool(int(row["is_drum"])),
            pitch=int(row["pitch"]),
            velocity=int(row["velocity"]),
            start_seconds=float(row["start_seconds"]),
            end_seconds=float(row["end_seconds"]),
        )
        for row in rows
    ]
    if not events:
        raise ValueError(f"No sweep events found in {source.name}")
    return events


def load_render_targets(path: str | Path) -> dict[tuple[int, bool], RenderTarget]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    targets: dict[tuple[int, bool], RenderTarget] = {}
    for row in rows:
        target = RenderTarget(
            program=int(row["program"]),
            is_drum=bool(int(row["is_drum"])),
            wav_path=_resolve_manifest_path(row["wav_path"], source.parent),
        )
        key = (target.program, target.is_drum)
        if key in targets:
            raise ValueError(
                f"Duplicate render target for program={key[0]}, drum={key[1]}"
            )
        targets[key] = target
    if not targets:
        raise ValueError(f"No render targets found in {source.name}")
    return targets


def isotonic_increasing(values: Iterable[float]) -> np.ndarray:
    """Least-squares nondecreasing fit using the pool-adjacent-violators algorithm."""

    observations = np.asarray(tuple(values), dtype=np.float64)
    if observations.ndim != 1 or observations.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(observations)):
        raise ValueError("values must be finite")

    means: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, value in enumerate(observations):
        means.append(float(value))
        weights.append(1.0)
        starts.append(index)
        ends.append(index + 1)
        while len(means) >= 2 and means[-2] > means[-1]:
            weight = weights[-2] + weights[-1]
            mean = (means[-2] * weights[-2] + means[-1] * weights[-1]) / weight
            means[-2:] = [mean]
            weights[-2:] = [weight]
            starts[-2:] = [starts[-2]]
            ends[-2:] = [ends[-1]]

    fitted = np.empty_like(observations)
    for mean, start, end in zip(means, starts, ends):
        fitted[start:end] = mean
    return fitted


def _power_to_db(power: float) -> float:
    return float(10.0 * math.log10(max(float(power), POWER_FLOOR)))


def _amplitude_to_db(amplitude: float) -> float:
    return float(20.0 * math.log10(max(float(amplitude), math.sqrt(POWER_FLOOR))))


def _window_samples(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    anchor_seconds: float,
    start_offset_ms: float,
    end_offset_ms: float,
) -> np.ndarray:
    left = int(math.floor((anchor_seconds + start_offset_ms / 1000.0) * sample_rate))
    right = int(math.ceil((anchor_seconds + end_offset_ms / 1000.0) * sample_rate))
    left = max(0, min(left, waveform.shape[0]))
    right = max(0, min(right, waveform.shape[0]))
    if right <= left:
        return np.zeros((0, waveform.shape[1]), dtype=np.float32)
    return waveform[left:right]


def _frame_power(segment: np.ndarray, sample_rate: int, frame_ms: float) -> np.ndarray:
    if segment.size == 0:
        return np.zeros(0, dtype=np.float64)
    sample_power = np.mean(
        np.square(segment.astype(np.float64, copy=False)),
        axis=1,
    )
    frame_samples = max(1, int(round(sample_rate * frame_ms / 1000.0)))
    return np.asarray(
        [
            sample_power[left : left + frame_samples].mean()
            for left in range(0, sample_power.size, frame_samples)
        ],
        dtype=np.float64,
    )


def _measure_event(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    event: SweepEventRecord,
    config: CalibrationAnalysisConfig,
) -> dict[str, Any]:
    note_ms = max(0.0, (event.end_seconds - event.start_seconds) * 1000.0)
    signal_end_ms = min(config.signal_end_ms, note_ms)
    signal = _window_samples(
        waveform,
        sample_rate=sample_rate,
        anchor_seconds=event.start_seconds,
        start_offset_ms=config.signal_start_ms,
        end_offset_ms=signal_end_ms,
    )
    noise = _window_samples(
        waveform,
        sample_rate=sample_rate,
        anchor_seconds=event.start_seconds,
        start_offset_ms=config.noise_start_ms,
        end_offset_ms=config.noise_end_ms,
    )
    signal_frames = _frame_power(signal, sample_rate, config.frame_ms)
    noise_frames = _frame_power(noise, sample_rate, config.frame_ms)
    if signal_frames.size:
        level_power = float(np.percentile(signal_frames, config.frame_percentile))
        rms_power = float(np.mean(np.square(signal.astype(np.float64, copy=False))))
        peak = float(np.max(np.abs(signal)))
        clipped = bool(np.any(np.abs(signal) >= config.clipping_threshold))
    else:
        level_power = POWER_FLOOR
        rms_power = POWER_FLOOR
        peak = 0.0
        clipped = False
    noise_power = (
        float(np.percentile(noise_frames, config.frame_percentile))
        if noise_frames.size
        else POWER_FLOOR
    )
    level_dbfs = _power_to_db(level_power)
    noise_dbfs = _power_to_db(noise_power)
    return {
        "raw_level_dbfs": level_dbfs,
        "rms_dbfs": _power_to_db(rms_power),
        "peak_dbfs": _amplitude_to_db(peak),
        "pre_note_level_dbfs": noise_dbfs,
        "snr_db": level_dbfs - noise_dbfs,
        "valid": int(bool(signal_frames.size) and level_dbfs >= config.min_signal_dbfs),
        "clipped": int(clipped),
    }


def _fit_curves(
    rows: list[dict[str, Any]],
    *,
    low_snr_db: float,
) -> dict[str, Any]:
    groups: dict[tuple[bool, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (bool(row["is_drum"]), int(row["program"]), int(row["pitch"]))
        groups.setdefault(key, []).append(row)

    violation_count = 0
    adjusted: list[float] = []
    complete_group_count = 0
    nonresponsive_group_count = 0
    tail_contaminated_count = 0
    extrapolated_count = 0
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: int(row["velocity"]))
        for row in group_rows:
            row["fit_observed"] = int(row["valid"])
            row["tail_contaminated"] = 0
        valid_raw = [row for row in group_rows if int(row["valid"])]
        raw = np.asarray(
            [float(row["raw_level_dbfs"]) for row in valid_raw],
            dtype=np.float64,
        )
        violation_count += int(np.sum(np.diff(raw) < 0.0)) if raw.size > 1 else 0
        if (
            len(group_rows) >= 2
            and int(group_rows[0]["valid"])
            and int(group_rows[1]["valid"])
            and float(group_rows[0]["snr_db"]) < low_snr_db
            and float(group_rows[0]["raw_level_dbfs"])
            > float(group_rows[1]["raw_level_dbfs"])
        ):
            group_rows[0]["fit_observed"] = 0
            group_rows[0]["tail_contaminated"] = 1
            tail_contaminated_count += 1

        observed_rows = [row for row in group_rows if int(row["fit_observed"])]
        for row in group_rows:
            row["fitted_level_dbfs"] = ""
            row["relative_level_db"] = ""
            row["fit_adjustment_db"] = ""
            row["curve_dynamic_range_db"] = ""
        if not observed_rows:
            continue
        observed_raw = np.asarray(
            [float(row["raw_level_dbfs"]) for row in observed_rows],
            dtype=np.float64,
        )
        observed_fitted = isotonic_increasing(observed_raw)
        all_fitted = np.full(len(group_rows), np.nan, dtype=np.float64)
        observed_indices = np.asarray(
            [group_rows.index(row) for row in observed_rows],
            dtype=np.int64,
        )
        all_fitted[observed_indices] = observed_fitted
        if len(observed_rows) >= 2:
            all_x = np.log(np.asarray([float(row["velocity"]) for row in group_rows]))
            observed_x = all_x[observed_indices]
            missing_indices = np.flatnonzero(~np.isfinite(all_fitted))
            all_fitted[missing_indices] = np.interp(
                all_x[missing_indices],
                observed_x,
                observed_fitted,
            )
            left = missing_indices[missing_indices < observed_indices[0]]
            if left.size:
                slope = (observed_fitted[1] - observed_fitted[0]) / (
                    observed_x[1] - observed_x[0]
                )
                all_fitted[left] = observed_fitted[0] + slope * (
                    all_x[left] - observed_x[0]
                )
            right = missing_indices[missing_indices > observed_indices[-1]]
            if right.size:
                slope = (observed_fitted[-1] - observed_fitted[-2]) / (
                    observed_x[-1] - observed_x[-2]
                )
                all_fitted[right] = observed_fitted[-1] + slope * (
                    all_x[right] - observed_x[-1]
                )
            all_fitted = np.maximum.accumulate(all_fitted)
        fitted_indices = np.flatnonzero(np.isfinite(all_fitted))
        reference = float(all_fitted[fitted_indices[-1]])
        dynamic_range = float(
            all_fitted[fitted_indices[-1]] - all_fitted[fitted_indices[0]]
        )
        if fitted_indices.size == len(group_rows):
            complete_group_count += 1
        if dynamic_range < 1.0:
            nonresponsive_group_count += 1
        for index in fitted_indices:
            row = group_rows[int(index)]
            value = float(all_fitted[int(index)])
            adjustment: float | str = ""
            if int(row["fit_observed"]):
                adjustment = float(value - float(row["raw_level_dbfs"]))
                adjusted.append(abs(adjustment))
            else:
                extrapolated_count += 1
            row["fitted_level_dbfs"] = float(value)
            row["relative_level_db"] = float(value - reference)
            row["fit_adjustment_db"] = adjustment
            row["curve_dynamic_range_db"] = dynamic_range

    return {
        "curve_group_count": len(groups),
        "complete_curve_group_count": complete_group_count,
        "nonresponsive_curve_group_count": nonresponsive_group_count,
        "raw_monotonic_violation_count": violation_count,
        "tail_contaminated_event_count": tail_contaminated_count,
        "interpolated_or_extrapolated_event_count": extrapolated_count,
        "isotonic_adjusted_event_count": sum(value > 1e-9 for value in adjusted),
        "mean_abs_fit_adjustment_db": float(np.mean(adjusted)) if adjusted else 0.0,
        "max_abs_fit_adjustment_db": float(np.max(adjusted)) if adjusted else 0.0,
    }


def analyze_sweep_files(
    events_path: str | Path,
    render_manifest_path: str | Path,
    *,
    config: CalibrationAnalysisConfig = CalibrationAnalysisConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Measure all rendered sweeps and fit a pitch-conditioned velocity curve."""

    events = load_sweep_events(events_path)
    targets = load_render_targets(render_manifest_path)
    events_by_target: dict[tuple[int, bool], list[SweepEventRecord]] = {}
    for event in events:
        events_by_target.setdefault((event.program, event.is_drum), []).append(event)

    missing_targets = sorted(set(events_by_target) - set(targets))
    if missing_targets:
        raise ValueError(f"Missing render targets for {len(missing_targets)} sweep(s)")

    rows: list[dict[str, Any]] = []
    sample_rates: set[int] = set()
    for key in sorted(events_by_target, key=lambda value: (value[1], value[0])):
        target = targets[key]
        if not target.wav_path.is_file():
            raise FileNotFoundError(f"Rendered WAV not found: {target.wav_path.name}")
        waveform, sample_rate = sf.read(
            str(target.wav_path),
            dtype="float32",
            always_2d=True,
        )
        sample_rate = int(sample_rate)
        sample_rates.add(sample_rate)
        for event in events_by_target[key]:
            row: dict[str, Any] = {
                "program": event.program,
                "is_drum": int(event.is_drum),
                "pitch": event.pitch,
                "velocity": event.velocity,
                "start_seconds": event.start_seconds,
                "end_seconds": event.end_seconds,
                "wav_file": target.wav_path.name,
                "sample_rate": sample_rate,
            }
            row.update(
                _measure_event(
                    waveform,
                    sample_rate=sample_rate,
                    event=event,
                    config=config,
                )
            )
            rows.append(row)

    fit_summary = _fit_curves(rows, low_snr_db=config.low_snr_db)
    valid_rows = [row for row in rows if int(row["valid"])]
    summary = {
        "schema_version": 1,
        "rendered_wav_count": len(events_by_target),
        "event_count": len(rows),
        "valid_event_count": len(valid_rows),
        "invalid_event_count": len(rows) - len(valid_rows),
        "low_snr_event_count": sum(
            float(row["snr_db"]) < config.low_snr_db for row in valid_rows
        ),
        "clipped_event_count": sum(int(row["clipped"]) for row in rows),
        "sample_rates": sorted(sample_rates),
        "analysis_config": {
            "frame_ms": config.frame_ms,
            "signal_start_ms": config.signal_start_ms,
            "signal_end_ms": config.signal_end_ms,
            "noise_start_ms": config.noise_start_ms,
            "noise_end_ms": config.noise_end_ms,
            "frame_percentile": config.frame_percentile,
            "min_signal_dbfs": config.min_signal_dbfs,
            "low_snr_db": config.low_snr_db,
            "clipping_threshold": config.clipping_threshold,
        },
        **fit_summary,
    }
    rows.sort(
        key=lambda row: (
            int(row["is_drum"]),
            int(row["program"]),
            int(row["pitch"]),
            int(row["velocity"]),
        )
    )
    return rows, summary


def write_curve_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_curve_npz(path: str | Path, rows: list[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    numeric_fields = [field for field in CURVE_FIELDS if field != "wav_file"]
    arrays: dict[str, np.ndarray] = {}
    integer_fields = {
        "program",
        "is_drum",
        "pitch",
        "velocity",
        "sample_rate",
        "valid",
        "clipped",
        "fit_observed",
        "tail_contaminated",
    }
    for field in numeric_fields:
        if field in integer_fields:
            arrays[field] = np.asarray(
                [int(row[field]) for row in rows], dtype=np.int32
            )
        else:
            arrays[field] = np.asarray(
                [float(row[field]) if row[field] != "" else np.nan for row in rows],
                dtype=np.float32,
            )
    arrays["wav_file"] = np.asarray([str(row["wav_file"]) for row in rows])
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(destination)
