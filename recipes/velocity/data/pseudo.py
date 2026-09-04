from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import soundfile as sf

from instrument_agnostic_amt.velocity.data.midi import MidiNoteTable

from ..config import PseudoLabelConfig


POWER_FLOOR = 1e-12


@dataclass(frozen=True)
class PseudoLabelSummary:
    note_count: int
    valid_note_count: int
    duration_seconds: float
    sample_rate: int
    audio_rms_dbfs: float
    audio_peak: float
    active_ratio: float
    active_level_dbfs: float
    active_rms_dbfs: float
    level_confidence: float


def _power_to_db(power: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(power, POWER_FLOOR))


def _frame_mean_square(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    frame_ms: float,
) -> np.ndarray:
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    if waveform.ndim != 2:
        raise ValueError("waveform must be [samples] or [samples, channels]")
    samples_per_frame = max(1, int(round(float(sample_rate) * frame_ms / 1000.0)))
    sample_power = np.einsum(
        "sc,sc->s",
        waveform.astype(np.float32, copy=False),
        waveform.astype(np.float32, copy=False),
        optimize=True,
    ) / max(1, int(waveform.shape[1]))
    frame_count = int(math.ceil(sample_power.size / samples_per_frame))
    if frame_count == 0:
        return np.zeros(0, dtype=np.float64)
    padded_size = frame_count * samples_per_frame
    if padded_size > sample_power.size:
        sample_power = np.pad(sample_power, (0, padded_size - sample_power.size))
    return sample_power.reshape(frame_count, samples_per_frame).mean(
        axis=1,
        dtype=np.float64,
    )


def _window_power(
    prefix_power: np.ndarray,
    *,
    start_seconds: np.ndarray,
    start_offset_ms: float,
    end_offset_ms: float,
    frame_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = max(0, int(prefix_power.size) - 1)
    start_ms = start_seconds * 1000.0
    left_raw = np.floor((start_ms + float(start_offset_ms)) / frame_ms).astype(np.int64)
    right_raw = np.ceil((start_ms + float(end_offset_ms)) / frame_ms).astype(np.int64)
    left = np.clip(left_raw, 0, frame_count)
    right = np.clip(right_raw, 0, frame_count)
    valid = right > left
    width = np.maximum(1, right - left)
    result = np.full(start_seconds.shape, POWER_FLOOR, dtype=np.float64)
    result[valid] = (prefix_power[right[valid]] - prefix_power[left[valid]]) / width[
        valid
    ]
    return result, valid


def percentile_ranks(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Return tie-aware ranks in [0, 1], leaving invalid entries as NaN."""

    values = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid, dtype=np.bool_)
    output = np.full(values.shape, np.nan, dtype=np.float32)
    indices = np.flatnonzero(valid & np.isfinite(values))
    if indices.size == 0:
        return output
    if indices.size == 1:
        output[indices[0]] = 0.5
        return output

    order = indices[np.argsort(values[indices], kind="mergesort")]
    sorted_values = values[order]
    position = 0
    denominator = float(indices.size - 1)
    while position < order.size:
        end = position + 1
        while end < order.size and sorted_values[end] == sorted_values[position]:
            end += 1
        average_rank = (float(position) + float(end - 1)) * 0.5 / denominator
        output[order[position:end]] = average_rank
        position = end
    return output


def _trackwise_ranks(
    values: np.ndarray,
    valid: np.ndarray,
    track_index: np.ndarray,
    *,
    min_rank_notes: int,
) -> np.ndarray:
    ranks = np.full(values.shape, np.nan, dtype=np.float32)
    for track in np.unique(track_index):
        track_mask = valid & (track_index == track)
        if int(track_mask.sum()) >= int(min_rank_notes):
            track_ranks = percentile_ranks(values, track_mask)
            ranks[track_mask] = track_ranks[track_mask]

    unresolved = valid & ~np.isfinite(ranks)
    if np.any(unresolved):
        global_ranks = percentile_ranks(values, valid)
        ranks[unresolved] = global_ranks[unresolved]
    return ranks


def _collision_counts(start_seconds: np.ndarray, window_ms: float) -> np.ndarray:
    if start_seconds.size == 0:
        return np.zeros(0, dtype=np.int16)
    window_seconds = float(window_ms) / 1000.0
    order = np.argsort(start_seconds, kind="mergesort")
    sorted_starts = start_seconds[order]
    left = np.searchsorted(
        sorted_starts,
        sorted_starts - window_seconds,
        side="left",
    )
    right = np.searchsorted(
        sorted_starts,
        sorted_starts + window_seconds,
        side="right",
    )
    sorted_counts = np.maximum(1, right - left).astype(np.int16)
    counts = np.empty_like(sorted_counts)
    counts[order] = sorted_counts
    return counts


def _active_frame_mask(
    note_table: MidiNoteTable,
    *,
    frame_count: int,
    frame_ms: float,
) -> np.ndarray:
    difference = np.zeros(frame_count + 1, dtype=np.int32)
    for start, end in zip(note_table.start_seconds, note_table.end_seconds):
        left = int(np.clip(math.floor(start * 1000.0 / frame_ms), 0, frame_count))
        right = int(np.clip(math.ceil(end * 1000.0 / frame_ms), 0, frame_count))
        if right <= left:
            continue
        difference[left] += 1
        difference[right] -= 1
    return np.cumsum(difference[:-1]) > 0


def build_pseudo_labels_from_audio(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    note_table: MidiNoteTable,
    config: PseudoLabelConfig,
) -> tuple[dict[str, np.ndarray], PseudoLabelSummary]:
    """
    Build weak note-strength labels from onset/pre-onset energy.

    These labels deliberately express within-track strength rank and confidence;
    they are not treated as physical ground-truth MIDI velocity.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if waveform.ndim == 1:
        waveform = waveform[:, None]
    frame_power = _frame_mean_square(
        waveform,
        sample_rate=int(sample_rate),
        frame_ms=float(config.frame_ms),
    )
    prefix_power = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(frame_power, dtype=np.float64)]
    )
    signal_power, signal_window_valid = _window_power(
        prefix_power,
        start_seconds=note_table.start_seconds,
        start_offset_ms=config.onset_signal_start_ms,
        end_offset_ms=config.onset_signal_end_ms,
        frame_ms=config.frame_ms,
    )
    noise_power, noise_window_valid = _window_power(
        prefix_power,
        start_seconds=note_table.start_seconds,
        start_offset_ms=config.onset_noise_start_ms,
        end_offset_ms=config.onset_noise_end_ms,
        frame_ms=config.frame_ms,
    )
    signal_dbfs = np.asarray(_power_to_db(signal_power), dtype=np.float32)
    noise_dbfs = np.asarray(_power_to_db(noise_power), dtype=np.float32)
    snr_db = signal_dbfs - noise_dbfs
    duration_ms = (note_table.end_seconds - note_table.start_seconds) * 1000.0
    audio_duration_seconds = float(waveform.shape[0]) / float(sample_rate)
    onset_inside_audio = (note_table.start_seconds >= 0.0) & (
        note_table.start_seconds < audio_duration_seconds
    )
    valid = (
        signal_window_valid
        & noise_window_valid
        & onset_inside_audio
        & (duration_ms >= float(config.min_note_duration_ms))
        & (signal_dbfs >= float(config.min_signal_dbfs))
        & (snr_db >= float(config.min_snr_db))
    )

    collision_count = _collision_counts(
        note_table.start_seconds,
        config.collision_window_ms,
    )
    snr_confidence = np.clip(
        (snr_db - float(config.min_snr_db))
        / (float(config.full_confidence_snr_db) - float(config.min_snr_db)),
        0.0,
        1.0,
    )
    collision_confidence = 1.0 / np.sqrt(
        np.maximum(1, collision_count).astype(np.float32)
    )
    confidence = (snr_confidence * collision_confidence).astype(np.float32)
    confidence[~valid] = 0.0

    ranks = _trackwise_ranks(
        signal_dbfs,
        valid,
        note_table.track_index,
        min_rank_notes=config.min_rank_notes,
    )
    pseudo_velocity = np.zeros(note_table.note_count, dtype=np.int16)
    ranked = valid & np.isfinite(ranks)
    pseudo_velocity[ranked] = np.rint(
        float(config.pseudo_velocity_min)
        + ranks[ranked] * float(config.pseudo_velocity_max - config.pseudo_velocity_min)
    ).astype(np.int16)

    active_mask = _active_frame_mask(
        note_table,
        frame_count=int(frame_power.size),
        frame_ms=config.frame_ms,
    )
    active_power = frame_power[active_mask]
    active_ratio = float(active_mask.mean()) if active_mask.size else 0.0
    if active_power.size:
        active_db = np.asarray(_power_to_db(active_power), dtype=np.float64)
        active_level_dbfs = float(np.median(active_db))
        active_rms_dbfs = float(_power_to_db(float(active_power.mean())))
    else:
        active_level_dbfs = float("nan")
        active_rms_dbfs = float("nan")

    if waveform.size:
        waveform_power = float(
            np.einsum(
                "sc,sc->",
                waveform.astype(np.float32, copy=False),
                waveform.astype(np.float32, copy=False),
                optimize=True,
            )
            / waveform.size
        )
        audio_peak = float(np.max(np.abs(waveform)))
    else:
        waveform_power = POWER_FLOOR
        audio_peak = 0.0
    audio_rms_dbfs = float(_power_to_db(waveform_power))
    valid_note_count = int(valid.sum())
    level_confidence = float(
        min(1.0, valid_note_count / 20.0) * min(1.0, active_ratio / 0.1)
    )

    arrays = {
        "note_start_seconds": note_table.start_seconds.astype(np.float64, copy=False),
        "note_end_seconds": note_table.end_seconds.astype(np.float64, copy=False),
        "note_pitch": note_table.pitch.astype(np.int16, copy=False),
        "input_velocity": note_table.input_velocity.astype(np.int16, copy=False),
        "note_program": note_table.program.astype(np.int16, copy=False),
        "note_is_drum": note_table.is_drum.astype(np.bool_, copy=False),
        "note_track_index": note_table.track_index.astype(np.int16, copy=False),
        "onset_signal_dbfs": signal_dbfs,
        "pre_onset_dbfs": noise_dbfs,
        "onset_snr_db": snr_db.astype(np.float32, copy=False),
        "onset_collision_count": collision_count,
        "pseudo_velocity_rank": ranks,
        "pseudo_velocity": pseudo_velocity,
        "pseudo_confidence": confidence,
        "pseudo_valid": valid.astype(np.bool_, copy=False),
        "frame_ms": np.asarray(config.frame_ms, dtype=np.float32),
        "active_level_dbfs": np.asarray(active_level_dbfs, dtype=np.float32),
        "active_rms_dbfs": np.asarray(active_rms_dbfs, dtype=np.float32),
        "active_ratio": np.asarray(active_ratio, dtype=np.float32),
        "audio_rms_dbfs": np.asarray(audio_rms_dbfs, dtype=np.float32),
        "audio_peak": np.asarray(audio_peak, dtype=np.float32),
        "duration_seconds": np.asarray(audio_duration_seconds, dtype=np.float64),
        "sample_rate": np.asarray(sample_rate, dtype=np.int32),
        "level_confidence": np.asarray(level_confidence, dtype=np.float32),
    }
    summary = PseudoLabelSummary(
        note_count=note_table.note_count,
        valid_note_count=valid_note_count,
        duration_seconds=audio_duration_seconds,
        sample_rate=int(sample_rate),
        audio_rms_dbfs=audio_rms_dbfs,
        audio_peak=audio_peak,
        active_ratio=active_ratio,
        active_level_dbfs=active_level_dbfs,
        active_rms_dbfs=active_rms_dbfs,
        level_confidence=level_confidence,
    )
    return arrays, summary


def save_pseudo_label_npz(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp")
    with temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)
    temporary.replace(destination)


def prepare_pseudo_label_file(
    wav_path: str | Path,
    note_table: MidiNoteTable,
    output_path: str | Path,
    *,
    config: PseudoLabelConfig,
) -> PseudoLabelSummary:
    waveform, sample_rate = sf.read(
        str(wav_path),
        dtype="float32",
        always_2d=True,
    )
    arrays, summary = build_pseudo_labels_from_audio(
        waveform,
        sample_rate=int(sample_rate),
        note_table=note_table,
        config=config,
    )
    save_pseudo_label_npz(output_path, arrays)
    return summary


def load_pseudo_label_summary(path: str | Path) -> PseudoLabelSummary:
    with np.load(path, allow_pickle=False) as data:
        return PseudoLabelSummary(
            note_count=int(data["note_pitch"].size),
            valid_note_count=int(data["pseudo_valid"].sum()),
            duration_seconds=float(data["duration_seconds"]),
            sample_rate=int(data["sample_rate"]),
            audio_rms_dbfs=float(data["audio_rms_dbfs"]),
            audio_peak=float(data["audio_peak"]),
            active_ratio=float(data["active_ratio"]),
            active_level_dbfs=float(data["active_level_dbfs"]),
            active_rms_dbfs=float(data["active_rms_dbfs"]),
            level_confidence=float(data["level_confidence"]),
        )
