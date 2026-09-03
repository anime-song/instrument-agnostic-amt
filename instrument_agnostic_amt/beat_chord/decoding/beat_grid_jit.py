from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without the optional extra
    njit = None


if njit is not None:

    @njit(cache=True, fastmath=False)
    def _build_regularized_grid_kernel(
        beat_log_odds: np.ndarray,
        downbeat_log_odds: np.ndarray,
        start_frame: int,
        end_frame: int,
        meter_num: int,
        bar_count: int,
        tolerance_frames: int,
        snap_penalty: float,
    ) -> np.ndarray:
        beat_count = int(meter_num) * int(bar_count)
        if beat_count <= 0 or end_frame <= start_frame:
            return np.empty(0, dtype=np.int64)

        output = np.empty(beat_count, dtype=np.int64)
        period = float(end_frame - start_frame) / float(beat_count)
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
                search_start = max(
                    previous + 1,
                    ideal_frame - tolerance_frames,
                )
                search_end = min(
                    end_frame,
                    ideal_frame + tolerance_frames + 1,
                )
                if search_end <= search_start:
                    snapped = max(
                        previous + 1,
                        min(end_frame - 1, ideal_frame),
                    )
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
                return np.empty(0, dtype=np.int64)
            output[beat_index] = snapped
            previous = snapped
        return output

else:
    _build_regularized_grid_kernel = None


@lru_cache(maxsize=1)
def is_jit_grid_available() -> bool:
    """Return whether the exact Numba grid kernel can be used.

    Numba fails at first call rather than at import, so the kernel is warmed up
    on a tiny input here. Auto-selection must not hand back a backend that only
    raises once decoding has started.
    """

    if _build_regularized_grid_kernel is None:
        return False
    probe = np.zeros(8, dtype=np.float64)
    try:
        _build_regularized_grid_kernel(probe, probe, 0, 4, 2, 2, 1, 1.0)
    except Exception:  # pragma: no cover - depends on the local Numba install
        return False
    return True


def build_regularized_grid_jit(
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
    """Build the exact grid with an optional cached CPU Numba kernel."""

    del beat_probabilities, downbeat_probabilities
    kernel: Any = _build_regularized_grid_kernel
    if kernel is None:
        raise RuntimeError("--grid_jit requires the optional 'numba' package")
    grid = kernel(
        beat_log_odds,
        downbeat_log_odds,
        int(start_frame),
        int(end_frame),
        int(meter_num),
        int(bar_count),
        int(tolerance_frames),
        float(snap_penalty),
    )
    # tolist() は int64 配列を Python int の list へ一括変換する。要素ごとに
    # numpy scalar を作る generator より速く、値は同じ。
    return tuple(grid.tolist())
