from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .meter_grouping import (
    group_boundary_log_odds_array,
    score_major_groupings,
)


_LOG_TWO = math.log(2.0)
_LOG_TEMPO_OCTAVE_WINDOW = math.log(1.15)


@dataclass(frozen=True)
class MeterGridSegment:
    """One meter/grid interval selected by a beat decoder."""

    start_frame: int
    end_frame: int
    meter_index: int
    meter_num: int
    meter_den: int
    bar_count: int
    mapped_downbeat_frames: tuple[int, ...]
    score: float
    quarter_note_bpm: float | None = None
    score_components: dict[str, float] | None = None
    confidence_margin: float | None = None
    meter_evidence_source: str = "direct"
    major_grouping: tuple[int, ...] | None = None


@dataclass(frozen=True)
class BeatGridDPConfig:
    """Configuration for the bar-lattice dynamic-programming decoder."""

    sample_rate: int
    hop_length: int
    tolerance_frames: int = 2
    downbeat_candidate_threshold: float = 0.15
    beat_candidate_threshold: float = 0.35
    max_bar_count: int = 4
    beam_size: int = 24
    max_boundary_hops: int = 12
    max_meter_candidates: int = 5
    max_beat_boundary_candidates_per_second: float = 0.5
    min_quarter_bpm: float = 30.0
    max_quarter_bpm: float = 300.0
    max_edge_seconds: float = 20.0
    max_leading_seconds: float = 8.0
    max_trailing_seconds: float = 8.0
    beat_score_weight: float = 1.0
    offbeat_peak_weight: float = 0.35
    downbeat_score_weight: float = 1.5
    false_downbeat_weight: float = 0.75
    meter_score_weight: float = 0.5
    group_boundary_score_weight: float = 0.5
    false_group_boundary_weight: float = 0.25
    additive_meter_penalty: float = 0.35
    collapse_additive_meters: bool = True
    minimum_additive_meter_pairs: int = 2
    additive_tempo_tolerance_ratio: float = 1.15
    additive_auxiliary_pass: bool = True
    additive_aux_meter_score_weight: float = 0.1
    additive_aux_tempo_transition_weight: float = 12.0
    additive_aux_meter_change_penalty: float = 3.5
    use_jit_grid: bool = False
    segment_penalty: float = 0.25
    missing_bar_penalty: float = 0.15
    tempo_transition_weight: float = 2.0
    meter_change_penalty: float = 12.0
    short_meter_run_penalty: float = 8.5
    minimum_meter_run_quarter_notes: float = 4.0
    octave_jump_penalty: float = 2.0
    snap_penalty: float = 0.25
    leading_trailing_penalty_per_second: float = 0.25
    tempo_free_ratio: float = 1.08
    tempo_huber_ratio: float = 1.20

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.tolerance_frames < 0:
            raise ValueError("tolerance_frames must be non-negative")
        for name, value in (
            ("downbeat_candidate_threshold", self.downbeat_candidate_threshold),
            ("beat_candidate_threshold", self.beat_candidate_threshold),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if self.max_bar_count <= 0:
            raise ValueError("max_bar_count must be positive")
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.max_boundary_hops <= 0:
            raise ValueError("max_boundary_hops must be positive")
        if self.max_meter_candidates <= 0:
            raise ValueError("max_meter_candidates must be positive")
        if self.minimum_meter_run_quarter_notes <= 0.0:
            raise ValueError("minimum_meter_run_quarter_notes must be positive")
        if self.minimum_additive_meter_pairs <= 0:
            raise ValueError("minimum_additive_meter_pairs must be positive")
        if self.group_boundary_score_weight < 0.0:
            raise ValueError("group_boundary_score_weight must be non-negative")
        if self.false_group_boundary_weight < 0.0:
            raise ValueError("false_group_boundary_weight must be non-negative")
        if self.additive_tempo_tolerance_ratio < 1.0:
            raise ValueError("additive_tempo_tolerance_ratio must be at least one")
        if self.max_beat_boundary_candidates_per_second < 0.0:
            raise ValueError(
                "max_beat_boundary_candidates_per_second must be non-negative"
            )
        if self.min_quarter_bpm <= 0.0:
            raise ValueError("min_quarter_bpm must be positive")
        if self.max_quarter_bpm <= self.min_quarter_bpm:
            raise ValueError("max_quarter_bpm must be greater than min_quarter_bpm")
        if self.max_edge_seconds <= 0.0:
            raise ValueError("max_edge_seconds must be positive")
        if self.max_leading_seconds < 0.0 or self.max_trailing_seconds < 0.0:
            raise ValueError("leading/trailing durations must be non-negative")
        if self.tempo_free_ratio < 1.0:
            raise ValueError("tempo_free_ratio must be at least one")
        if self.tempo_huber_ratio <= self.tempo_free_ratio:
            raise ValueError("tempo_huber_ratio must exceed tempo_free_ratio")

    @property
    def seconds_per_frame(self) -> float:
        return float(self.hop_length) / float(self.sample_rate)


@dataclass(frozen=True)
class BeatGridDecodeResult:
    beat_frames: tuple[int, ...]
    downbeat_frames: tuple[int, ...]
    meter_segments: tuple[MeterGridSegment, ...]
    raw_downbeat_candidates: tuple[int, ...]
    all_boundary_candidates: tuple[int, ...]
    rejected_downbeat_candidates: tuple[int, ...]
    inferred_downbeat_frames: tuple[int, ...]
    total_score: float
    confidence_margin: float | None


@dataclass(frozen=True)
class GridEdgeHypothesis:
    start_candidate_index: int
    end_candidate_index: int
    start_frame: int
    end_frame: int
    meter_index: int
    meter_num: int
    meter_den: int
    bar_count: int
    grid_frames: tuple[int, ...]
    mapped_downbeat_frames: tuple[int, ...]
    beat_period_frames: float
    quarter_period_frames: float
    quarter_note_bpm: float
    score: float
    score_components: dict[str, float]
    meter_mean_log_prob: float
    meter_evidence_source: str
    major_groupings: tuple[tuple[int, ...], ...] = ()


@dataclass(frozen=True)
class _PathState:
    total_score: float
    end_candidate_index: int
    last_quarter_period_frames: float | None
    last_meter_index: int | None
    last_meter_run_quarter_notes: float
    previous: _PathState | None
    edge: GridEdgeHypothesis | None


@dataclass(frozen=True)
class _DecodedBar:
    segment: MeterGridSegment
    beat_frames: tuple[int, ...]


def _validate_decoder_inputs(
    *,
    beat_probabilities: np.ndarray,
    downbeat_probabilities: np.ndarray,
    meter_logits: np.ndarray,
    meter_classes: list[tuple[int, int]],
    group_boundary_probabilities: np.ndarray | None = None,
) -> None:
    if beat_probabilities.ndim != 1:
        raise ValueError("beat_probabilities must be one-dimensional")
    if downbeat_probabilities.shape != beat_probabilities.shape:
        raise ValueError(
            "downbeat_probabilities must have the same shape as beat_probabilities"
        )
    if meter_logits.ndim != 2:
        raise ValueError("meter_logits must have shape [T, M]")
    if meter_logits.shape[0] != beat_probabilities.shape[0]:
        raise ValueError("meter logits and beat probabilities must share a time axis")
    if meter_logits.shape[1] != len(meter_classes):
        raise ValueError("meter class count must match meter_logits")
    if any(num <= 0 or den <= 0 for num, den in meter_classes):
        raise ValueError("meter class values must be positive")
    if (
        group_boundary_probabilities is not None
        and group_boundary_probabilities.shape != beat_probabilities.shape
    ):
        raise ValueError("group_boundary_probabilities must match beat probabilities")


def _logit(probability: float) -> float:
    clipped = min(1.0 - 1e-4, max(1e-4, float(probability)))
    return max(-6.0, min(6.0, math.log(clipped / (1.0 - clipped))))


def _log_softmax_matrix(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _log_odds_array(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities.astype(np.float64), 1e-4, 1.0 - 1e-4)
    return np.clip(np.log(clipped / (1.0 - clipped)), -6.0, 6.0)


def _ranked_peak_candidates(
    probabilities: np.ndarray,
    *,
    threshold: float,
    minimum_distance_frames: int,
    auxiliary_probabilities: np.ndarray | None = None,
) -> list[int]:
    """Find local maxima and run score-ordered non-maximum suppression."""

    if probabilities.size == 0:
        return []
    raw: list[int] = []
    for index in range(len(probabilities)):
        value = float(probabilities[index])
        if value < float(threshold):
            continue
        left = float(probabilities[index - 1]) if index > 0 else -float("inf")
        right = (
            float(probabilities[index + 1])
            if index + 1 < len(probabilities)
            else -float("inf")
        )
        if value >= left and value >= right and (value > left or value > right):
            raw.append(index)

    def rank_score(frame: int) -> float:
        auxiliary = (
            0.0
            if auxiliary_probabilities is None
            else 0.1 * float(auxiliary_probabilities[frame])
        )
        return float(probabilities[frame]) + auxiliary

    selected: list[int] = []
    radius = max(1, int(minimum_distance_frames))
    for frame in sorted(raw, key=lambda item: (-rank_score(item), item)):
        if all(abs(frame - existing) >= radius for existing in selected):
            selected.append(int(frame))
    return sorted(selected)


def _merge_boundary_candidates(
    *,
    raw_downbeats: list[int],
    beat_candidates: list[int],
    beat_probabilities: np.ndarray,
    downbeat_probabilities: np.ndarray,
    merge_radius: int,
) -> list[int]:
    raw = sorted(set([0, *raw_downbeats, *beat_candidates]))
    if not raw:
        return []
    groups: list[list[int]] = [[raw[0]]]
    for frame in raw[1:]:
        if frame - groups[-1][-1] <= int(merge_radius):
            groups[-1].append(frame)
        else:
            groups.append([frame])

    merged: list[int] = []
    raw_downbeat_set = set(raw_downbeats)
    for group in groups:
        merged.append(
            max(
                group,
                key=lambda frame: (
                    float(downbeat_probabilities[frame])
                    + 0.25 * float(beat_probabilities[frame])
                    + (0.05 if frame in raw_downbeat_set else 0.0),
                    -frame,
                ),
            )
        )
    if 0 not in merged:
        merged.insert(0, 0)
    return sorted(set(merged))


def _select_sparse_beat_boundary_candidates(
    *,
    beat_peak_frames: list[int],
    beat_probabilities: np.ndarray,
    downbeat_probabilities: np.ndarray,
    config: BeatGridDPConfig,
) -> list[int]:
    """Keep only ambiguous bar-boundary beats; multi-bar edges fill omissions."""

    if not beat_peak_frames:
        return []
    minimum_downbeat_probability = max(
        0.02, float(config.downbeat_candidate_threshold) * 0.5
    )
    eligible = [
        int(frame)
        for frame in beat_peak_frames
        if float(downbeat_probabilities[frame]) >= minimum_downbeat_probability
    ]
    duration_seconds = len(beat_probabilities) * config.seconds_per_frame
    budget = int(
        math.ceil(
            duration_seconds * float(config.max_beat_boundary_candidates_per_second)
        )
    )
    if budget <= 0:
        selected: list[int] = []
    else:
        ranked = sorted(
            eligible,
            key=lambda frame: (
                -float(downbeat_probabilities[frame]),
                -float(beat_probabilities[frame]),
                frame,
            ),
        )
        selected = ranked[:budget]

    # Preserve plausible first/last boundaries even when their downbeat head is weak.
    selected.extend((int(beat_peak_frames[0]), int(beat_peak_frames[-1])))
    return sorted(set(selected))


def _is_near_any(
    frame: int, references: tuple[int, ...] | list[int], radius: int
) -> bool:
    if not references:
        return False
    insertion = bisect_left(references, frame)
    for index in (insertion - 1, insertion):
        if 0 <= index < len(references):
            if abs(int(references[index]) - int(frame)) <= int(radius):
                return True
    return False


def _build_regularized_grid(
    *,
    beat_probabilities: np.ndarray,
    downbeat_probabilities: np.ndarray,
    beat_log_odds: np.ndarray,
    downbeat_log_odds: np.ndarray,
    start_frame: int,
    end_frame: int,
    meter_num: int,
    bar_count: int,
    tolerance_frames: int,
    snap_penalty: float,
) -> tuple[int, ...]:
    beat_count = int(meter_num) * int(bar_count)
    if beat_count <= 0 or end_frame <= start_frame:
        return ()
    period = float(end_frame - start_frame) / float(beat_count)
    grid: list[int] = []
    previous = start_frame - 1
    for beat_index in range(beat_count):
        ideal = float(start_frame) + float(beat_index) * period
        ideal_frame = max(
            start_frame,
            min(end_frame - 1, int(round(ideal))),
        )
        if beat_index == 0:
            snapped = int(start_frame)
        else:
            search_start = max(previous + 1, ideal_frame - tolerance_frames)
            search_end = min(end_frame, ideal_frame + tolerance_frames + 1)
            if search_end <= search_start:
                snapped = max(previous + 1, min(end_frame - 1, ideal_frame))
            else:
                best_score = -float("inf")
                snapped = ideal_frame
                for frame in range(search_start, search_end):
                    normalized_distance = (
                        0.0
                        if tolerance_frames <= 0
                        else (float(frame) - ideal) / float(tolerance_frames)
                    )
                    score = float(beat_log_odds[frame])
                    score -= float(snap_penalty) * normalized_distance**2
                    if beat_index % meter_num == 0:
                        score += 0.5 * float(downbeat_log_odds[frame])
                    if score > best_score:
                        best_score = score
                        snapped = int(frame)
        if snapped >= end_frame:
            return ()
        grid.append(snapped)
        previous = snapped
    return tuple(grid)


def _meter_candidates_for_interval(
    *,
    mean_meter_log_probs: np.ndarray,
    meter_classes: list[tuple[int, int]],
    observed_beat_count: int,
    max_bar_count: int,
    max_meter_candidates: int,
) -> tuple[int, ...]:
    count = min(int(max_meter_candidates), len(meter_classes))
    top_indices = np.argsort(mean_meter_log_probs)[-count:].tolist()
    selected = {int(index) for index in top_indices}

    for bar_count in range(1, int(max_bar_count) + 1):
        inferred_numerator = max(
            1,
            int(round(float(observed_beat_count) / float(bar_count))),
        )
        for meter_index, (meter_num, _meter_den) in enumerate(meter_classes):
            if int(meter_num) == inferred_numerator:
                selected.add(int(meter_index))
    return tuple(sorted(selected))


def _meter_evidence_for_edge(
    *,
    start_frame: int,
    end_frame: int,
    meter_index: int,
    meter_num: int,
    meter_den: int,
    bar_count: int,
    meter_prefix: np.ndarray,
    meter_index_by_class: dict[tuple[int, int], int],
    additive_meter_penalty: float,
) -> tuple[float, str]:
    """Score a meter directly or as an ordered additive-meter partition."""

    duration = int(end_frame - start_frame)
    direct = float(
        (meter_prefix[end_frame, meter_index] - meter_prefix[start_frame, meter_index])
        / max(1, duration)
    )
    best_score = direct
    best_source = "direct"
    if (int(meter_num), int(meter_den)) != (7, 4):
        return best_score, best_source

    for left_num in (3, 4):
        right_num = int(meter_num) - left_num
        left_index = meter_index_by_class.get((left_num, int(meter_den)))
        right_index = meter_index_by_class.get((right_num, int(meter_den)))
        if left_index is None or right_index is None:
            continue

        evidence_sum = 0.0
        valid = True
        for bar_index in range(int(bar_count)):
            bar_start = int(
                round(start_frame + duration * bar_index / float(bar_count))
            )
            bar_end = int(
                round(start_frame + duration * (bar_index + 1) / float(bar_count))
            )
            split = int(
                round(bar_start + (bar_end - bar_start) * left_num / float(meter_num))
            )
            if split <= bar_start or split >= bar_end:
                valid = False
                break
            evidence_sum += float(
                meter_prefix[split, left_index] - meter_prefix[bar_start, left_index]
            )
            evidence_sum += float(
                meter_prefix[bar_end, right_index] - meter_prefix[split, right_index]
            )
        if not valid:
            continue
        composite_score = evidence_sum / max(1, duration)
        composite_score -= float(additive_meter_penalty)
        if composite_score > best_score:
            best_score = float(composite_score)
            best_source = f"{left_num}/{meter_den}+{right_num}/{meter_den}"
    return best_score, best_source


def _score_grid_edge(
    *,
    start_candidate_index: int,
    end_candidate_index: int,
    candidates: list[int],
    beat_probabilities: np.ndarray,
    downbeat_probabilities: np.ndarray,
    group_boundary_probabilities: np.ndarray | None,
    group_boundary_log_odds: np.ndarray | None,
    beat_log_odds: np.ndarray,
    downbeat_log_odds: np.ndarray,
    meter_mean_log_prob: float,
    meter_evidence_source: str,
    beat_peak_frames: np.ndarray,
    meter_index: int,
    meter_num: int,
    meter_den: int,
    bar_count: int,
    config: BeatGridDPConfig,
    grid_builder: Any = _build_regularized_grid,
) -> GridEdgeHypothesis | None:
    start_frame = int(candidates[start_candidate_index])
    end_frame = int(candidates[end_candidate_index])
    beat_count = int(meter_num) * int(bar_count)
    if end_frame <= start_frame or beat_count <= 0:
        return None

    beat_period_frames = float(end_frame - start_frame) / float(beat_count)
    beat_period_seconds = beat_period_frames * config.seconds_per_frame
    quarter_note_bpm = 240.0 / (beat_period_seconds * float(meter_den))
    if not config.min_quarter_bpm <= quarter_note_bpm <= config.max_quarter_bpm:
        return None

    grid_frames = grid_builder(
        beat_probabilities=beat_probabilities,
        downbeat_probabilities=downbeat_probabilities,
        beat_log_odds=beat_log_odds,
        downbeat_log_odds=downbeat_log_odds,
        start_frame=start_frame,
        end_frame=end_frame,
        meter_num=int(meter_num),
        bar_count=int(bar_count),
        tolerance_frames=int(config.tolerance_frames),
        snap_penalty=float(config.snap_penalty),
    )
    if len(grid_frames) != beat_count:
        return None

    beat_score = float(config.beat_score_weight) * sum(
        float(beat_log_odds[frame]) for frame in grid_frames
    )
    interval_peaks = beat_peak_frames[
        int(beat_peak_frames.searchsorted(start_frame, side="left")) : int(
            beat_peak_frames.searchsorted(end_frame, side="left")
        )
    ]
    # interval_peaksとgrid_framesはともに昇順。peakごとのbisectを避け、1回の
    # two-pointer走査でgrid近傍にない余分なpeakだけを数える。
    extra_peak_log_odds = 0.0
    grid_index = 0
    tolerance_frames = int(config.tolerance_frames)
    for peak_frame_value in interval_peaks:
        peak_frame = int(peak_frame_value)
        while (
            grid_index < len(grid_frames)
            and int(grid_frames[grid_index]) < peak_frame - tolerance_frames
        ):
            grid_index += 1
        if (
            grid_index >= len(grid_frames)
            or abs(int(grid_frames[grid_index]) - peak_frame) > tolerance_frames
        ):
            extra_peak_log_odds += max(0.0, float(beat_log_odds[peak_frame]))
    extra_peak_score = -float(config.offbeat_peak_weight) * extra_peak_log_odds

    downbeat_score = 0.0
    false_downbeat_score = 0.0
    mapped_downbeats: list[int] = []
    for beat_index, frame in enumerate(grid_frames):
        downbeat_logit = float(downbeat_log_odds[frame])
        if beat_index % int(meter_num) == 0:
            mapped_downbeats.append(int(frame))
            downbeat_score += float(config.downbeat_score_weight) * downbeat_logit
        else:
            false_downbeat_score -= float(config.false_downbeat_weight) * max(
                0.0, downbeat_logit
            )

    meter_score = (
        float(config.meter_score_weight)
        * float(meter_mean_log_prob)
        * float(beat_count)
    )
    raw_grouping_score, major_groupings = score_major_groupings(
        grid_frames=grid_frames,
        meter_num=int(meter_num),
        meter_den=int(meter_den),
        bar_count=int(bar_count),
        group_boundary_probabilities=group_boundary_probabilities,
        group_boundary_log_odds=group_boundary_log_odds,
        false_boundary_weight=float(config.false_group_boundary_weight),
    )
    grouping_score = float(config.group_boundary_score_weight) * raw_grouping_score
    segment_score = -float(config.segment_penalty)
    missing_bar_score = -float(config.missing_bar_penalty) * float(bar_count - 1)
    score_components = {
        "beat_grid": float(beat_score),
        "offbeat_peaks": float(extra_peak_score),
        "downbeat_grid": float(downbeat_score),
        "false_downbeats": float(false_downbeat_score),
        "meter": float(meter_score),
        "major_grouping": float(grouping_score),
        "segment": float(segment_score),
        "missing_bars": float(missing_bar_score),
    }
    total_score = float(sum(score_components.values()))
    quarter_period_frames = beat_period_frames * float(meter_den) / 4.0
    return GridEdgeHypothesis(
        start_candidate_index=int(start_candidate_index),
        end_candidate_index=int(end_candidate_index),
        start_frame=start_frame,
        end_frame=end_frame,
        meter_index=int(meter_index),
        meter_num=int(meter_num),
        meter_den=int(meter_den),
        bar_count=int(bar_count),
        grid_frames=grid_frames,
        mapped_downbeat_frames=tuple(mapped_downbeats),
        beat_period_frames=float(beat_period_frames),
        quarter_period_frames=float(quarter_period_frames),
        quarter_note_bpm=float(quarter_note_bpm),
        score=total_score,
        score_components=score_components,
        meter_mean_log_prob=float(meter_mean_log_prob),
        meter_evidence_source=str(meter_evidence_source),
        major_groupings=major_groupings,
    )


def _with_meter_score_weight(
    edge: GridEdgeHypothesis,
    meter_score_weight: float,
) -> GridEdgeHypothesis:
    """Rescore a cached edge without rebuilding its regularized beat grid."""

    score_components = dict(edge.score_components)
    score_components["meter"] = (
        float(meter_score_weight)
        * float(edge.meter_mean_log_prob)
        * float(edge.meter_num * edge.bar_count)
    )
    return replace(
        edge,
        score=float(sum(score_components.values())),
        score_components=score_components,
    )


def _tempo_transition_penalty(
    previous_period: float,
    current_period: float,
    config: BeatGridDPConfig,
    free_delta: float | None = None,
    huber_width: float | None = None,
) -> float:
    if previous_period <= 0.0 or current_period <= 0.0:
        return 0.0
    delta = abs(math.log(float(current_period) / float(previous_period)))
    if free_delta is None:
        free_delta = math.log(float(config.tempo_free_ratio))
    if delta <= free_delta:
        base_penalty = 0.0
    else:
        if huber_width is None:
            huber_width = math.log(float(config.tempo_huber_ratio)) - free_delta
        normalized = (delta - free_delta) / max(1e-6, huber_width)
        huber = 0.5 * normalized**2 if normalized <= 1.0 else normalized - 0.5
        base_penalty = float(config.tempo_transition_weight) * huber

    octave_distance = abs(delta - _LOG_TWO)
    if octave_distance <= _LOG_TEMPO_OCTAVE_WINDOW:
        base_penalty += float(config.octave_jump_penalty) * (
            1.0 - octave_distance / _LOG_TEMPO_OCTAVE_WINDOW
        )
    return float(base_penalty)


def _transition_score(
    state: _PathState,
    edge: GridEdgeHypothesis,
    config: BeatGridDPConfig,
    tempo_penalty_cache: dict[tuple[float, float], float] | None = None,
    free_delta: float | None = None,
    huber_width: float | None = None,
) -> float:
    penalty = 0.0
    if state.last_quarter_period_frames is not None:
        cache_key = (
            float(state.last_quarter_period_frames),
            float(edge.quarter_period_frames),
        )
        tempo_penalty = (
            None if tempo_penalty_cache is None else tempo_penalty_cache.get(cache_key)
        )
        if tempo_penalty is None:
            tempo_penalty = _tempo_transition_penalty(
                state.last_quarter_period_frames,
                edge.quarter_period_frames,
                config,
                free_delta,
                huber_width,
            )
            if tempo_penalty_cache is not None:
                tempo_penalty_cache[cache_key] = tempo_penalty
        penalty += tempo_penalty
    if (
        state.last_meter_index is not None
        and state.last_meter_index != edge.meter_index
    ):
        penalty += float(config.meter_change_penalty)
        if (
            0.0
            < state.last_meter_run_quarter_notes
            < float(config.minimum_meter_run_quarter_notes)
        ):
            penalty += float(config.short_meter_run_penalty)
    return -float(penalty)


def _period_bin(period_frames: float | None) -> int:
    if period_frames is None or period_frames <= 0.0:
        return -1
    return int(round(math.log(period_frames) * 12.0))


def _prune_states(
    states: list[_PathState],
    *,
    beam_size: int,
    minimum_meter_run_quarter_notes: float,
) -> list[_PathState]:
    if len(states) <= beam_size:
        return sorted(states, key=lambda state: state.total_score, reverse=True)
    best_by_signature: dict[tuple[int | None, int, int], _PathState] = {}
    for state in states:
        signature = (
            state.last_meter_index,
            _period_bin(state.last_quarter_period_frames),
            int(
                state.last_meter_run_quarter_notes
                >= float(minimum_meter_run_quarter_notes)
            ),
        )
        previous = best_by_signature.get(signature)
        if previous is None or state.total_score > previous.total_score:
            best_by_signature[signature] = state
    return sorted(
        best_by_signature.values(),
        key=lambda state: state.total_score,
        reverse=True,
    )[: int(beam_size)]


def _reconstruct_edges(state: _PathState) -> list[GridEdgeHypothesis]:
    edges: list[GridEdgeHypothesis] = []
    cursor: _PathState | None = state
    while cursor is not None:
        if cursor.edge is not None:
            edges.append(cursor.edge)
        cursor = cursor.previous
    edges.reverse()
    return edges


def _expand_edges_to_bars(
    edges: list[GridEdgeHypothesis],
    confidence_margin: float | None,
) -> list[_DecodedBar]:
    bars: list[_DecodedBar] = []
    for edge in edges:
        meter_num = int(edge.meter_num)
        for bar_index in range(int(edge.bar_count)):
            beat_start = bar_index * meter_num
            beat_end = beat_start + meter_num
            bar_beats = tuple(edge.grid_frames[beat_start:beat_end])
            if len(bar_beats) != meter_num:
                continue
            bar_start = int(bar_beats[0])
            bar_end = (
                int(edge.mapped_downbeat_frames[bar_index + 1])
                if bar_index + 1 < len(edge.mapped_downbeat_frames)
                else int(edge.end_frame)
            )
            divided_components = {
                name: float(value) / float(edge.bar_count)
                for name, value in edge.score_components.items()
            }
            bars.append(
                _DecodedBar(
                    segment=MeterGridSegment(
                        start_frame=bar_start,
                        end_frame=bar_end,
                        meter_index=int(edge.meter_index),
                        meter_num=meter_num,
                        meter_den=int(edge.meter_den),
                        bar_count=1,
                        mapped_downbeat_frames=(bar_start,),
                        score=float(edge.score) / float(edge.bar_count),
                        quarter_note_bpm=float(edge.quarter_note_bpm),
                        score_components=divided_components,
                        confidence_margin=confidence_margin,
                        meter_evidence_source=str(edge.meter_evidence_source),
                        major_grouping=(
                            edge.major_groupings[bar_index]
                            if bar_index < len(edge.major_groupings)
                            else None
                        ),
                    ),
                    beat_frames=bar_beats,
                )
            )
    return bars


def _sum_score_components(bars: tuple[_DecodedBar, ...]) -> dict[str, float]:
    combined: dict[str, float] = {}
    for bar in bars:
        for name, value in (bar.segment.score_components or {}).items():
            combined[name] = combined.get(name, 0.0) + float(value)
    return combined


def _collapse_additive_meter_runs(
    bars: list[_DecodedBar],
    *,
    meter_classes: list[tuple[int, int]],
    config: BeatGridDPConfig,
) -> list[_DecodedBar]:
    """Collapse repeated 4+3/3+4-style bars into their additive meter."""

    if not config.collapse_additive_meters or len(bars) < 4:
        return bars
    meter_index_by_class = {
        (int(num), int(den)): int(index)
        for index, (num, den) in enumerate(meter_classes)
    }

    def pair_target(index: int) -> tuple[int, int] | None:
        if index + 1 >= len(bars):
            return None
        left = bars[index].segment
        right = bars[index + 1].segment
        if left.meter_den != right.meter_den:
            return None
        if {left.meter_num, right.meter_num} != {3, 4}:
            return None
        target = (left.meter_num + right.meter_num, left.meter_den)
        if target != (7, 4) or target not in meter_index_by_class:
            return None
        bpms = (left.quarter_note_bpm, right.quarter_note_bpm)
        if any(bpm is None or bpm <= 0.0 for bpm in bpms):
            return None
        ratio = max(float(bpms[0]), float(bpms[1])) / min(
            float(bpms[0]), float(bpms[1])
        )
        if ratio > float(config.additive_tempo_tolerance_ratio):
            return None
        return target

    collapsed: list[_DecodedBar] = []
    index = 0
    while index < len(bars):
        target = pair_target(index)
        if target is None:
            collapsed.append(bars[index])
            index += 1
            continue

        run_end = index
        run_bpms: list[float] = []
        while run_end + 1 < len(bars) and pair_target(run_end) == target:
            run_bpms.extend(
                [
                    float(bars[run_end].segment.quarter_note_bpm),
                    float(bars[run_end + 1].segment.quarter_note_bpm),
                ]
            )
            if max(run_bpms) / min(run_bpms) > float(
                config.additive_tempo_tolerance_ratio
            ):
                run_bpms = run_bpms[:-2]
                break
            run_end += 2

        pair_count = (run_end - index) // 2
        if pair_count < int(config.minimum_additive_meter_pairs):
            collapsed.append(bars[index])
            index += 1
            continue

        target_index = meter_index_by_class[target]
        for pair_start in range(index, run_end, 2):
            pair = (bars[pair_start], bars[pair_start + 1])
            first, second = pair
            pair_beats = tuple([*first.beat_frames, *second.beat_frames])
            source = (
                f"collapsed:{first.segment.meter_num}/{target[1]}+"
                f"{second.segment.meter_num}/{target[1]}"
            )
            collapsed.append(
                _DecodedBar(
                    segment=MeterGridSegment(
                        start_frame=int(first.segment.start_frame),
                        end_frame=int(second.segment.end_frame),
                        meter_index=int(target_index),
                        meter_num=int(target[0]),
                        meter_den=int(target[1]),
                        bar_count=1,
                        mapped_downbeat_frames=(int(first.segment.start_frame),),
                        score=float(first.segment.score + second.segment.score),
                        quarter_note_bpm=float(
                            (
                                float(first.segment.quarter_note_bpm)
                                * first.segment.meter_num
                                + float(second.segment.quarter_note_bpm)
                                * second.segment.meter_num
                            )
                            / float(target[0])
                        ),
                        score_components=_sum_score_components(pair),
                        confidence_margin=first.segment.confidence_margin,
                        meter_evidence_source=source,
                        major_grouping=(
                            int(first.segment.meter_num),
                            int(second.segment.meter_num),
                        ),
                    ),
                    beat_frames=pair_beats,
                )
            )
        index = run_end
    return collapsed


def _decode_beats_with_meter_grid_dp_single_pass(
    *,
    beat_probabilities: np.ndarray,
    downbeat_probabilities: np.ndarray,
    group_boundary_probabilities: np.ndarray | None,
    meter_logits: np.ndarray,
    meter_classes: list[tuple[int, int]],
    config: BeatGridDPConfig,
    edge_cache: (
        dict[tuple[int, int, int, int], GridEdgeHypothesis | None] | None
    ) = None,
) -> BeatGridDecodeResult:
    """Decode a globally coherent beat/downbeat/meter path over a bar lattice."""

    _validate_decoder_inputs(
        beat_probabilities=beat_probabilities,
        downbeat_probabilities=downbeat_probabilities,
        meter_logits=meter_logits,
        meter_classes=meter_classes,
        group_boundary_probabilities=group_boundary_probabilities,
    )
    frame_count = int(beat_probabilities.shape[0])
    if frame_count <= 1:
        return BeatGridDecodeResult((), (), (), (), (), (), (), 0.0, None)
    grid_builder = _build_regularized_grid
    if config.use_jit_grid:
        from .beat_grid_jit import build_regularized_grid_jit

        grid_builder = build_regularized_grid_jit
    beat_log_odds = _log_odds_array(beat_probabilities)
    downbeat_log_odds = _log_odds_array(downbeat_probabilities)
    group_boundary_log_odds = (
        None
        if group_boundary_probabilities is None
        else group_boundary_log_odds_array(group_boundary_probabilities)
    )

    raw_downbeat_candidates = _ranked_peak_candidates(
        downbeat_probabilities,
        threshold=float(config.downbeat_candidate_threshold),
        minimum_distance_frames=max(2, int(config.tolerance_frames) + 1),
        auxiliary_probabilities=beat_probabilities,
    )
    beat_peak_frames = _ranked_peak_candidates(
        beat_probabilities,
        threshold=float(config.beat_candidate_threshold),
        minimum_distance_frames=max(2, int(config.tolerance_frames) * 2 + 1),
        auxiliary_probabilities=downbeat_probabilities,
    )
    beat_peak_array = np.asarray(beat_peak_frames, dtype=np.int64)
    beat_boundary_candidates = _select_sparse_beat_boundary_candidates(
        beat_peak_frames=beat_peak_frames,
        beat_probabilities=beat_probabilities,
        downbeat_probabilities=downbeat_probabilities,
        config=config,
    )
    candidates = _merge_boundary_candidates(
        raw_downbeats=raw_downbeat_candidates,
        beat_candidates=beat_boundary_candidates,
        beat_probabilities=beat_probabilities,
        downbeat_probabilities=downbeat_probabilities,
        merge_radius=max(1, int(config.tolerance_frames)),
    )
    candidates = [frame for frame in candidates if 0 <= frame < frame_count]
    if len(candidates) < 2:
        return BeatGridDecodeResult(
            (),
            (),
            (),
            tuple(raw_downbeat_candidates),
            tuple(candidates),
            tuple(raw_downbeat_candidates),
            (),
            0.0,
            None,
        )

    meter_log_probs = _log_softmax_matrix(meter_logits)
    meter_prefix = np.vstack(
        [
            np.zeros((1, meter_log_probs.shape[1]), dtype=np.float64),
            np.cumsum(meter_log_probs, axis=0),
        ]
    )
    meter_index_by_class = {
        (int(meter_num), int(meter_den)): int(index)
        for index, (meter_num, meter_den) in enumerate(meter_classes)
    }
    states_by_candidate: list[list[_PathState]] = [[] for _candidate in candidates]
    maximum_leading_frames = int(
        round(float(config.max_leading_seconds) / config.seconds_per_frame)
    )
    for candidate_index, frame in enumerate(candidates):
        if frame > maximum_leading_frames:
            break
        leading_seconds = float(frame) * config.seconds_per_frame
        start_evidence = 0.5 * _logit(float(downbeat_probabilities[frame]))
        states_by_candidate[candidate_index].append(
            _PathState(
                total_score=float(
                    start_evidence
                    - config.leading_trailing_penalty_per_second * leading_seconds
                ),
                end_candidate_index=int(candidate_index),
                last_quarter_period_frames=None,
                last_meter_index=None,
                last_meter_run_quarter_notes=0.0,
                previous=None,
                edge=None,
            )
        )

    max_edge_frames = int(
        round(float(config.max_edge_seconds) / config.seconds_per_frame)
    )
    tempo_penalty_cache: dict[tuple[float, float], float] = {}
    tempo_free_delta = math.log(float(config.tempo_free_ratio))
    tempo_huber_width = math.log(float(config.tempo_huber_ratio)) - tempo_free_delta
    for start_index, start_frame in enumerate(candidates):
        if not states_by_candidate[start_index]:
            continue
        states_by_candidate[start_index] = _prune_states(
            states_by_candidate[start_index],
            beam_size=int(config.beam_size),
            minimum_meter_run_quarter_notes=(config.minimum_meter_run_quarter_notes),
        )
        predecessor_states = states_by_candidate[start_index]
        # 同じstart候補では、遷移の評価はedgeのquarter periodとmeterだけで決まる。
        # meter evidenceやbar scoreが異なるedge間で最良predecessorを再利用する。
        predecessor_choice_cache: dict[
            tuple[float, int], tuple[_PathState, float]
        ] = {}

        for end_index in range(start_index + 1, len(candidates)):
            if end_index - start_index > int(config.max_boundary_hops):
                break
            end_frame = int(candidates[end_index])
            if end_frame - int(start_frame) > max_edge_frames:
                break
            if end_frame <= int(start_frame) + 1:
                continue

            meter_mean_log_probs = (
                meter_prefix[end_frame] - meter_prefix[int(start_frame)]
            ) / float(end_frame - int(start_frame))
            observed_beat_count = max(
                1,
                int(
                    beat_peak_array.searchsorted(end_frame, side="left")
                    - beat_peak_array.searchsorted(int(start_frame), side="left")
                ),
            )
            meter_candidates = _meter_candidates_for_interval(
                mean_meter_log_probs=meter_mean_log_probs,
                meter_classes=meter_classes,
                observed_beat_count=observed_beat_count,
                max_bar_count=int(config.max_bar_count),
                max_meter_candidates=int(config.max_meter_candidates),
            )

            for meter_index in meter_candidates:
                meter_num, meter_den = meter_classes[meter_index]
                for bar_count in range(1, int(config.max_bar_count) + 1):
                    expected_beat_count = int(meter_num) * int(bar_count)
                    allowed_peak_error = max(
                        2, int(math.ceil(0.35 * expected_beat_count))
                    )
                    if (
                        abs(observed_beat_count - expected_beat_count)
                        > allowed_peak_error
                    ):
                        continue
                    edge_key = (
                        int(start_frame),
                        int(end_frame),
                        int(meter_index),
                        int(bar_count),
                    )
                    if edge_cache is not None and edge_key in edge_cache:
                        cached_edge = edge_cache[edge_key]
                        edge = (
                            None
                            if cached_edge is None
                            else _with_meter_score_weight(
                                cached_edge,
                                config.meter_score_weight,
                            )
                        )
                    else:
                        meter_evidence, meter_evidence_source = (
                            _meter_evidence_for_edge(
                                start_frame=int(start_frame),
                                end_frame=int(end_frame),
                                meter_index=int(meter_index),
                                meter_num=int(meter_num),
                                meter_den=int(meter_den),
                                bar_count=int(bar_count),
                                meter_prefix=meter_prefix,
                                meter_index_by_class=meter_index_by_class,
                                additive_meter_penalty=(config.additive_meter_penalty),
                            )
                        )
                        edge = _score_grid_edge(
                            start_candidate_index=start_index,
                            end_candidate_index=end_index,
                            candidates=candidates,
                            beat_probabilities=beat_probabilities,
                            downbeat_probabilities=downbeat_probabilities,
                            group_boundary_probabilities=(group_boundary_probabilities),
                            group_boundary_log_odds=group_boundary_log_odds,
                            beat_log_odds=beat_log_odds,
                            downbeat_log_odds=downbeat_log_odds,
                            meter_mean_log_prob=float(meter_evidence),
                            meter_evidence_source=meter_evidence_source,
                            beat_peak_frames=beat_peak_array,
                            meter_index=int(meter_index),
                            meter_num=int(meter_num),
                            meter_den=int(meter_den),
                            bar_count=int(bar_count),
                            config=config,
                            grid_builder=grid_builder,
                        )
                        if edge_cache is not None:
                            edge_cache[edge_key] = edge
                    if edge is None:
                        continue
                    predecessor_key = (
                        float(edge.quarter_period_frames),
                        int(edge.meter_index),
                    )
                    predecessor_choice = predecessor_choice_cache.get(
                        predecessor_key
                    )
                    if predecessor_choice is None:
                        best_predecessor: _PathState | None = None
                        best_predecessor_score = -float("inf")
                        for predecessor in predecessor_states:
                            predecessor_score = (
                                predecessor.total_score
                                + _transition_score(
                                    predecessor,
                                    edge,
                                    config,
                                    tempo_penalty_cache,
                                    tempo_free_delta,
                                    tempo_huber_width,
                                )
                            )
                            if predecessor_score > best_predecessor_score:
                                best_predecessor_score = float(predecessor_score)
                                best_predecessor = predecessor
                        if best_predecessor is None:
                            continue
                        predecessor_choice = (
                            best_predecessor,
                            best_predecessor_score,
                        )
                        predecessor_choice_cache[predecessor_key] = (
                            predecessor_choice
                        )
                    best_predecessor, best_predecessor_score = predecessor_choice
                    best_total = best_predecessor_score + edge.score
                    edge_quarter_notes = (
                        float(edge.meter_num)
                        * 4.0
                        / float(edge.meter_den)
                        * float(edge.bar_count)
                    )
                    if best_predecessor.last_meter_index == edge.meter_index:
                        meter_run_quarter_notes = (
                            best_predecessor.last_meter_run_quarter_notes
                            + edge_quarter_notes
                        )
                    else:
                        meter_run_quarter_notes = edge_quarter_notes
                    states_by_candidate[end_index].append(
                        _PathState(
                            total_score=float(best_total),
                            end_candidate_index=end_index,
                            last_quarter_period_frames=float(
                                edge.quarter_period_frames
                            ),
                            last_meter_index=int(edge.meter_index),
                            last_meter_run_quarter_notes=float(meter_run_quarter_notes),
                            previous=best_predecessor,
                            edge=edge,
                        )
                    )
                    if len(states_by_candidate[end_index]) > int(config.beam_size) * 4:
                        states_by_candidate[end_index] = _prune_states(
                            states_by_candidate[end_index],
                            beam_size=int(config.beam_size),
                            minimum_meter_run_quarter_notes=(
                                config.minimum_meter_run_quarter_notes
                            ),
                        )

    maximum_trailing_frames = int(
        round(float(config.max_trailing_seconds) / config.seconds_per_frame)
    )
    final_states: list[tuple[float, _PathState]] = []
    for candidate_index, frame in enumerate(candidates):
        trailing_frames = frame_count - int(frame)
        if trailing_frames < 0 or trailing_frames > maximum_trailing_frames:
            continue
        trailing_seconds = float(trailing_frames) * config.seconds_per_frame
        for state in _prune_states(
            states_by_candidate[candidate_index],
            beam_size=int(config.beam_size),
            minimum_meter_run_quarter_notes=(config.minimum_meter_run_quarter_notes),
        ):
            if state.edge is None:
                continue
            final_score = state.total_score - (
                float(config.leading_trailing_penalty_per_second) * trailing_seconds
            )
            if (
                0.0
                < state.last_meter_run_quarter_notes
                < float(config.minimum_meter_run_quarter_notes)
            ):
                final_score -= float(config.short_meter_run_penalty)
            final_states.append((float(final_score), state))

    if not final_states:
        return BeatGridDecodeResult(
            (),
            (),
            (),
            tuple(raw_downbeat_candidates),
            tuple(candidates),
            tuple(raw_downbeat_candidates),
            (),
            0.0,
            None,
        )
    final_states.sort(key=lambda item: item[0], reverse=True)
    best_score, best_state = final_states[0]
    confidence_margin = (
        float(best_score - final_states[1][0]) if len(final_states) > 1 else None
    )
    selected_edges = _reconstruct_edges(best_state)
    if not selected_edges:
        return BeatGridDecodeResult(
            (),
            (),
            (),
            tuple(raw_downbeat_candidates),
            tuple(candidates),
            tuple(raw_downbeat_candidates),
            (),
            float(best_score),
            confidence_margin,
        )

    decoded_bars = _expand_edges_to_bars(selected_edges, confidence_margin)
    decoded_bars = _collapse_additive_meter_runs(
        decoded_bars,
        meter_classes=meter_classes,
        config=config,
    )
    beat_frames: list[int] = []
    downbeat_frames: list[int] = []
    meter_segments = [bar.segment for bar in decoded_bars]
    for bar in decoded_bars:
        for frame in bar.beat_frames:
            if not beat_frames or frame != beat_frames[-1]:
                beat_frames.append(int(frame))
        start_frame = int(bar.segment.start_frame)
        if not downbeat_frames or start_frame != downbeat_frames[-1]:
            downbeat_frames.append(start_frame)
    final_boundary = int(selected_edges[-1].end_frame)
    if not downbeat_frames or final_boundary != downbeat_frames[-1]:
        downbeat_frames.append(final_boundary)

    selected_downbeats = tuple(sorted(set(downbeat_frames)))
    raw_downbeats = tuple(raw_downbeat_candidates)
    rejected = tuple(
        frame
        for frame in raw_downbeats
        if not _is_near_any(frame, selected_downbeats, config.tolerance_frames)
    )
    inferred = tuple(
        frame
        for frame in selected_downbeats
        if not _is_near_any(frame, raw_downbeats, config.tolerance_frames)
    )
    if confidence_margin is not None:
        meter_segments = [
            replace(segment, confidence_margin=float(confidence_margin))
            for segment in meter_segments
        ]
    return BeatGridDecodeResult(
        beat_frames=tuple(beat_frames),
        downbeat_frames=selected_downbeats,
        meter_segments=tuple(meter_segments),
        raw_downbeat_candidates=raw_downbeats,
        all_boundary_candidates=tuple(candidates),
        rejected_downbeat_candidates=rejected,
        inferred_downbeat_frames=inferred,
        total_score=float(best_score),
        confidence_margin=confidence_margin,
    )


def _overlay_auxiliary_additive_meters(
    primary: BeatGridDecodeResult,
    auxiliary: BeatGridDecodeResult,
    *,
    config: BeatGridDPConfig,
) -> BeatGridDecodeResult:
    targets = [
        segment
        for segment in auxiliary.meter_segments
        if (segment.meter_num, segment.meter_den) == (7, 4)
        and segment.meter_evidence_source.startswith("collapsed:")
    ]
    if not targets:
        return primary

    tolerance = int(config.tolerance_frames)

    def overlaps_target(segment: MeterGridSegment) -> bool:
        return any(
            min(segment.end_frame, target.end_frame)
            - max(segment.start_frame, target.start_frame)
            > tolerance
            for target in targets
        )

    def frame_in_target(frame: int) -> bool:
        return any(
            target.start_frame <= int(frame) < target.end_frame for target in targets
        )

    meter_segments = sorted(
        [segment for segment in primary.meter_segments if not overlaps_target(segment)]
        + targets,
        key=lambda segment: segment.start_frame,
    )
    beat_frames = sorted(
        set(
            [frame for frame in primary.beat_frames if not frame_in_target(frame)]
            + [frame for frame in auxiliary.beat_frames if frame_in_target(frame)]
        )
    )
    final_boundary = (
        int(primary.downbeat_frames[-1]) if primary.downbeat_frames else None
    )
    downbeat_frames = sorted(
        set(segment.start_frame for segment in meter_segments)
        | ({final_boundary} if final_boundary is not None else set())
    )
    raw_downbeats = tuple(primary.raw_downbeat_candidates)
    selected_downbeats = tuple(downbeat_frames)
    rejected = tuple(
        frame
        for frame in raw_downbeats
        if not _is_near_any(frame, selected_downbeats, tolerance)
    )
    inferred = tuple(
        frame
        for frame in selected_downbeats
        if not _is_near_any(frame, raw_downbeats, tolerance)
    )
    return replace(
        primary,
        beat_frames=tuple(beat_frames),
        downbeat_frames=selected_downbeats,
        meter_segments=tuple(meter_segments),
        rejected_downbeat_candidates=rejected,
        inferred_downbeat_frames=inferred,
    )


def decode_beats_with_meter_grid_dp(
    *,
    beat_probabilities: np.ndarray,
    downbeat_probabilities: np.ndarray,
    meter_logits: np.ndarray,
    meter_classes: list[tuple[int, int]],
    config: BeatGridDPConfig,
    group_boundary_probabilities: np.ndarray | None = None,
) -> BeatGridDecodeResult:
    """Decode a stable primary path and selectively overlay repeated 7/4 evidence."""

    edge_cache: dict[tuple[int, int, int, int], GridEdgeHypothesis | None] = {}
    primary = _decode_beats_with_meter_grid_dp_single_pass(
        beat_probabilities=beat_probabilities,
        downbeat_probabilities=downbeat_probabilities,
        group_boundary_probabilities=group_boundary_probabilities,
        meter_logits=meter_logits,
        meter_classes=meter_classes,
        config=config,
        edge_cache=edge_cache,
    )
    if not config.additive_auxiliary_pass or (7, 4) not in meter_classes:
        return primary
    auxiliary_config = replace(
        config,
        meter_score_weight=float(config.additive_aux_meter_score_weight),
        tempo_transition_weight=float(config.additive_aux_tempo_transition_weight),
        meter_change_penalty=float(config.additive_aux_meter_change_penalty),
        additive_auxiliary_pass=False,
    )
    auxiliary = _decode_beats_with_meter_grid_dp_single_pass(
        beat_probabilities=beat_probabilities,
        downbeat_probabilities=downbeat_probabilities,
        group_boundary_probabilities=group_boundary_probabilities,
        meter_logits=meter_logits,
        meter_classes=meter_classes,
        config=auxiliary_config,
        edge_cache=edge_cache,
    )
    return _overlay_auxiliary_additive_meters(
        primary,
        auxiliary,
        config=config,
    )


def result_to_diagnostics(result: BeatGridDecodeResult) -> dict[str, Any]:
    """Convert the non-segment DP diagnostics to a JSON-ready dictionary."""

    return {
        "total_score": float(result.total_score),
        "confidence_margin": (
            None
            if result.confidence_margin is None
            else float(result.confidence_margin)
        ),
        "raw_downbeat_candidates": list(result.raw_downbeat_candidates),
        "all_boundary_candidates": list(result.all_boundary_candidates),
        "rejected_downbeat_candidates": list(result.rejected_downbeat_candidates),
        "inferred_downbeat_frames": list(result.inferred_downbeat_frames),
        "major_groupings": [
            {
                "start_frame": int(segment.start_frame),
                "end_frame": int(segment.end_frame),
                "meter": f"{segment.meter_num}/{segment.meter_den}",
                "pattern": list(segment.major_grouping),
            }
            for segment in result.meter_segments
            if segment.major_grouping is not None
        ],
        "additive_meter_segments": [
            {
                "start_frame": int(segment.start_frame),
                "end_frame": int(segment.end_frame),
                "meter": f"{segment.meter_num}/{segment.meter_den}",
                "source": segment.meter_evidence_source,
            }
            for segment in result.meter_segments
            if segment.meter_evidence_source.startswith("collapsed:")
        ],
    }
