from __future__ import annotations

import math
import random
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass

from instrument_agnostic_amt.beat_chord.meter_grouping import grouping_spec_for_meter


@dataclass(frozen=True)
class MeterAwareCropConfig:
    """Controls complete-bar sampling for meters with major groups."""

    probability: float = 0.75
    rarity_power: float = 0.5
    boundary_margin_frames: int = 2

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between zero and one")
        if self.rarity_power < 0.0:
            raise ValueError("rarity_power must be non-negative")
        if self.boundary_margin_frames < 0:
            raise ValueError("boundary_margin_frames must be non-negative")


@dataclass(frozen=True)
class MeterAwareCropSelection:
    window_start_sec: float
    used_meter_aware: bool
    target_meter_index: int | None = None
    target_bar_start_sec: float | None = None
    target_bar_end_sec: float | None = None


@dataclass(frozen=True)
class _EligibleBar:
    start_sec: float
    end_sec: float
    meter_index: int
    minimum_window_start_sec: float
    maximum_window_start_sec: float


def _random_labeled_window_start(
    *,
    meter_intervals: Sequence[tuple[float, float, int]],
    duration_sec: float,
    source_window_sec: float,
    rng: random.Random,
) -> float:
    max_start = max(0.0, float(duration_sec) - float(source_window_sec))
    if max_start <= 0.0:
        return 0.0
    if meter_intervals:
        first_label = float(meter_intervals[0][0])
        last_label = float(meter_intervals[-1][1])
        min_start = max(0.0, first_label - float(source_window_sec))
        max_labeled_start = min(max_start, last_label)
        if max_labeled_start > min_start:
            return float(rng.uniform(min_start, max_labeled_start))
    return float(rng.uniform(0.0, max_start))


def _has_event_near(
    sorted_events: tuple[float, ...],
    value: float,
    tolerance_sec: float,
) -> bool:
    index = bisect_left(sorted_events, float(value))
    return any(
        0 <= candidate_index < len(sorted_events)
        and abs(sorted_events[candidate_index] - float(value)) <= tolerance_sec
        for candidate_index in (index - 1, index)
    )


def _beat_count_in_bar(
    sorted_beats: tuple[float, ...],
    start_sec: float,
    end_sec: float,
    tolerance_sec: float,
) -> int:
    left = bisect_left(sorted_beats, float(start_sec) - tolerance_sec)
    right = bisect_left(sorted_beats, float(end_sec) - tolerance_sec)
    return max(0, int(right - left))


def _eligible_grouping_bars(
    *,
    meter_intervals: Sequence[tuple[float, float, int]],
    beat_times: Sequence[float],
    downbeat_times: Sequence[float],
    meter_classes: Sequence[tuple[int, int]],
    duration_sec: float,
    source_window_sec: float,
    boundary_margin_sec: float,
    event_tolerance_sec: float,
) -> tuple[_EligibleBar, ...]:
    sorted_beats = tuple(sorted(float(value) for value in beat_times))
    sorted_downbeats = tuple(sorted(float(value) for value in downbeat_times))
    max_start = max(0.0, float(duration_sec) - float(source_window_sec))
    right_margin = max(float(boundary_margin_sec), float(event_tolerance_sec))
    eligible: list[_EligibleBar] = []

    for raw_start, raw_end, raw_meter_index in meter_intervals:
        start_sec = float(raw_start)
        end_sec = float(raw_end)
        meter_index = int(raw_meter_index)
        if end_sec <= start_sec:
            continue
        if meter_index < 0 or meter_index >= len(meter_classes):
            continue
        meter_num, meter_den = meter_classes[meter_index]
        if grouping_spec_for_meter(meter_num, meter_den) is None:
            continue
        if not _has_event_near(
            sorted_downbeats, start_sec, event_tolerance_sec
        ) or not _has_event_near(sorted_downbeats, end_sec, event_tolerance_sec):
            continue
        if _beat_count_in_bar(
            sorted_beats,
            start_sec,
            end_sec,
            event_tolerance_sec,
        ) != int(meter_num):
            continue

        minimum_start = max(
            0.0,
            end_sec + right_margin - float(source_window_sec),
        )
        left_margin = min(float(boundary_margin_sec), max(0.0, start_sec))
        maximum_start = min(max_start, start_sec - left_margin)
        if minimum_start > maximum_start + event_tolerance_sec:
            continue
        eligible.append(
            _EligibleBar(
                start_sec=start_sec,
                end_sec=end_sec,
                meter_index=meter_index,
                minimum_window_start_sec=max(0.0, minimum_start),
                maximum_window_start_sec=max(0.0, maximum_start),
            )
        )
    return tuple(eligible)


def _weighted_meter_index(
    *,
    candidates_by_meter: dict[int, list[_EligibleBar]],
    meter_class_counts: Sequence[float],
    rarity_power: float,
    rng: random.Random,
) -> int:
    meter_indices = tuple(candidates_by_meter)
    weights: list[float] = []
    for meter_index in meter_indices:
        count = (
            float(meter_class_counts[meter_index])
            if 0 <= meter_index < len(meter_class_counts)
            else 1.0
        )
        weights.append(math.pow(max(count, 1.0), -float(rarity_power)))

    weight_sum = float(sum(weights))
    if weight_sum <= 0.0 or not math.isfinite(weight_sum):
        return int(rng.choice(meter_indices))
    threshold = float(rng.random()) * weight_sum
    cumulative = 0.0
    for meter_index, weight in zip(meter_indices, weights):
        cumulative += float(weight)
        if threshold <= cumulative:
            return int(meter_index)
    return int(meter_indices[-1])


def choose_meter_aware_window_start(
    *,
    meter_intervals: Sequence[tuple[float, float, int]],
    beat_times: Sequence[float],
    downbeat_times: Sequence[float],
    meter_classes: Sequence[tuple[int, int]],
    meter_class_counts: Sequence[float],
    duration_sec: float,
    source_window_sec: float,
    sample_rate: int,
    hop_length: int,
    config: MeterAwareCropConfig,
    rng: random.Random,
) -> MeterAwareCropSelection:
    """Choose a crop that contains one fully supervised grouping bar."""

    if duration_sec <= 0.0:
        raise ValueError("duration_sec must be positive")
    if source_window_sec <= 0.0:
        raise ValueError("source_window_sec must be positive")
    if sample_rate <= 0 or hop_length <= 0:
        raise ValueError("sample_rate and hop_length must be positive")

    def fallback() -> MeterAwareCropSelection:
        return MeterAwareCropSelection(
            window_start_sec=_random_labeled_window_start(
                meter_intervals=meter_intervals,
                duration_sec=duration_sec,
                source_window_sec=source_window_sec,
                rng=rng,
            ),
            used_meter_aware=False,
        )

    if config.probability <= 0.0:
        return fallback()
    if config.probability < 1.0 and rng.random() >= config.probability:
        return fallback()

    frame_sec = float(hop_length) / float(sample_rate)
    candidates = _eligible_grouping_bars(
        meter_intervals=meter_intervals,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        meter_classes=meter_classes,
        duration_sec=duration_sec,
        source_window_sec=source_window_sec,
        boundary_margin_sec=(config.boundary_margin_frames * frame_sec),
        event_tolerance_sec=max(1e-5, 0.25 * frame_sec),
    )
    if not candidates:
        return fallback()

    candidates_by_meter: dict[int, list[_EligibleBar]] = {}
    for candidate in candidates:
        candidates_by_meter.setdefault(candidate.meter_index, []).append(candidate)
    target_meter_index = _weighted_meter_index(
        candidates_by_meter=candidates_by_meter,
        meter_class_counts=meter_class_counts,
        rarity_power=config.rarity_power,
        rng=rng,
    )
    target_bar = rng.choice(candidates_by_meter[target_meter_index])
    if target_bar.maximum_window_start_sec > target_bar.minimum_window_start_sec:
        window_start_sec = rng.uniform(
            target_bar.minimum_window_start_sec,
            target_bar.maximum_window_start_sec,
        )
    else:
        window_start_sec = target_bar.minimum_window_start_sec
    return MeterAwareCropSelection(
        window_start_sec=float(window_start_sec),
        used_meter_aware=True,
        target_meter_index=int(target_meter_index),
        target_bar_start_sec=float(target_bar.start_sec),
        target_bar_end_sec=float(target_bar.end_sec),
    )
