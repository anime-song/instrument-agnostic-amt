from __future__ import annotations

import math

import numpy as np

from .beat_grid import MeterGridSegment


def detect_peaks(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    minimum_distance_frames: int = 5,
) -> list[int]:
    """Detect strict local maxima, preserving the original CLI behavior."""

    peaks: list[int] = []
    for index in range(1, len(probabilities) - 1):
        if (
            probabilities[index] > threshold
            and probabilities[index] > probabilities[index - 1]
            and probabilities[index] > probabilities[index + 1]
        ):
            if not peaks or index - peaks[-1] >= minimum_distance_frames:
                peaks.append(index)
    return peaks


def log_softmax_numpy(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    return shifted - np.log(np.exp(shifted).sum())


def build_grid_candidate(
    *,
    beat_probabilities: np.ndarray,
    start_frame: int,
    end_frame: int,
    beat_count: int,
    tolerance_frames: int,
) -> tuple[list[int], float, float]:
    bar_length = int(end_frame - start_frame)
    if bar_length <= 1:
        return [], 0.0, 0.0

    grid_frames: list[int] = []
    grid_mask = np.zeros(bar_length, dtype=bool)
    for beat_index in range(int(beat_count)):
        ideal_local = int(
            round(float(beat_index) * float(bar_length) / float(beat_count))
        )
        ideal_frame = start_frame + max(0, min(bar_length - 1, ideal_local))
        snap_start = max(start_frame, ideal_frame - tolerance_frames)
        snap_end = min(end_frame, ideal_frame + tolerance_frames + 1)
        local_probs = beat_probabilities[snap_start:snap_end]
        snapped_frame = ideal_frame
        if local_probs.size > 0:
            snapped_frame = int(snap_start + int(np.argmax(local_probs)))
        grid_frames.append(snapped_frame)
        mask_start = max(0, ideal_frame - tolerance_frames - start_frame)
        mask_end = min(bar_length, ideal_frame + tolerance_frames + 1 - start_frame)
        grid_mask[mask_start:mask_end] = True

    bar_probs = beat_probabilities[start_frame:end_frame]
    on_grid_score = float(np.mean([beat_probabilities[frame] for frame in grid_frames]))
    off_grid_score = float(bar_probs[~grid_mask].mean()) if np.any(~grid_mask) else 0.0
    return grid_frames, on_grid_score, off_grid_score


def decode_beats_with_meter_grid(
    *,
    beat_probabilities: np.ndarray,
    downbeat_frames: list[int],
    meter_logits: np.ndarray,
    meter_classes: list[tuple[int, int]],
    tolerance_frames: int,
    meter_score_weight: float,
    beat_grid_score_weight: float,
) -> tuple[list[int], list[MeterGridSegment]]:
    """Original interval-by-interval decoder retained for regression comparison."""

    if len(downbeat_frames) < 2:
        return [], []
    bar_lengths = np.diff(np.asarray(downbeat_frames, dtype=np.float64))
    typical_bar_frames = (
        float(np.quantile(bar_lengths, 0.25)) if bar_lengths.size >= 2 else None
    )
    decoded_beats: list[int] = []
    decoded_meters: list[MeterGridSegment] = []
    for start_frame, end_frame in zip(downbeat_frames[:-1], downbeat_frames[1:]):
        if end_frame <= start_frame + 1:
            continue
        meter_log_probs = log_softmax_numpy(
            meter_logits[start_frame:end_frame].mean(axis=0)
        )
        interval_probabilities = beat_probabilities[start_frame:end_frame]
        raw_peak_count = len(
            detect_peaks(
                interval_probabilities,
                threshold=0.3,
                minimum_distance_frames=max(2, tolerance_frames * 2 + 1),
            )
        )
        best: tuple[float, int, int, list[int]] | None = None
        for meter_index, (meter_num, _meter_den) in enumerate(meter_classes):
            candidate_bar_counts = {1}
            peak_bar_count = int(round((raw_peak_count + 1) / max(1, meter_num)))
            if peak_bar_count > 1:
                candidate_bar_counts.add(min(8, peak_bar_count))
            if typical_bar_frames:
                duration_bar_count = int(
                    round((end_frame - start_frame) / typical_bar_frames)
                )
                if duration_bar_count > 1:
                    candidate_bar_counts.add(min(8, duration_bar_count))
            for bar_count in sorted(candidate_bar_counts):
                grid, on_grid, off_grid = build_grid_candidate(
                    beat_probabilities=beat_probabilities,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    beat_count=meter_num * bar_count,
                    tolerance_frames=tolerance_frames,
                )
                score = meter_score_weight * float(meter_log_probs[meter_index])
                score += beat_grid_score_weight * (on_grid - off_grid)
                score -= 0.15 * float(bar_count - 1)
                expected_peaks = max(0, meter_num * bar_count - 1)
                score -= (
                    0.1 * abs(raw_peak_count - expected_peaks) / max(1, expected_peaks)
                )
                if typical_bar_frames:
                    candidate_bar_frames = (end_frame - start_frame) / bar_count
                    score -= 0.25 * abs(
                        math.log(candidate_bar_frames / typical_bar_frames)
                    )
                if best is None or score > best[0]:
                    best = (float(score), meter_index, bar_count, grid)
        if best is None:
            continue
        score, meter_index, bar_count, grid = best
        meter_num, meter_den = meter_classes[meter_index]
        mapped = tuple(grid[:: max(1, meter_num)])
        decoded_meters.append(
            MeterGridSegment(
                start_frame=start_frame,
                end_frame=end_frame,
                meter_index=meter_index,
                meter_num=meter_num,
                meter_den=meter_den,
                bar_count=bar_count,
                mapped_downbeat_frames=mapped,
                score=score,
            )
        )
        for frame in grid:
            if not decoded_beats or frame != decoded_beats[-1]:
                decoded_beats.append(frame)
    return decoded_beats, decoded_meters
