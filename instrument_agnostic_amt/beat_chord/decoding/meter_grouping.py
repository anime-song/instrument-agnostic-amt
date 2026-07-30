from __future__ import annotations

import math

import numpy as np

from ..meter_grouping import (
    GroupingPattern,
    grouping_boundary_offsets,
    grouping_spec_for_meter,
)


def _logit(probability: float) -> float:
    clipped = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


def score_major_groupings(
    *,
    grid_frames: tuple[int, ...],
    meter_num: int,
    meter_den: int,
    bar_count: int,
    group_boundary_probabilities: np.ndarray | None,
    false_boundary_weight: float,
) -> tuple[float, tuple[GroupingPattern, ...]]:
    """Score the best major-group pattern independently for each bar."""

    spec = grouping_spec_for_meter(meter_num, meter_den)
    if spec is None or group_boundary_probabilities is None:
        return 0.0, ()
    if false_boundary_weight < 0.0:
        raise ValueError("false_boundary_weight must be non-negative")

    total_score = 0.0
    selected_patterns: list[GroupingPattern] = []
    for bar_index in range(int(bar_count)):
        bar_start = bar_index * int(meter_num)
        bar_frames = grid_frames[bar_start : bar_start + int(meter_num)]
        if len(bar_frames) != int(meter_num):
            return 0.0, ()

        internal_frames = bar_frames[1:]
        internal_logits = [
            _logit(float(group_boundary_probabilities[int(frame)]))
            for frame in internal_frames
        ]
        best_pattern: GroupingPattern | None = None
        best_score = -float("inf")
        for pattern in spec.patterns:
            expected_offsets = set(grouping_boundary_offsets(pattern))
            pattern_score = 0.0
            for offset, boundary_logit in enumerate(internal_logits, start=1):
                if offset in expected_offsets:
                    pattern_score += boundary_logit
                else:
                    pattern_score -= float(false_boundary_weight) * max(
                        0.0,
                        boundary_logit,
                    )
            if pattern_score > best_score:
                best_score = float(pattern_score)
                best_pattern = pattern

        if best_pattern is None:
            return 0.0, ()
        total_score += best_score
        selected_patterns.append(best_pattern)
    return float(total_score), tuple(selected_patterns)
