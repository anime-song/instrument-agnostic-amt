"""Post-processing that removes the two artefacts the bar lattice leaves behind.

The DP in :mod:`beat_grid` emits integer frame indices, and at 512 samples on a
22.05 kHz signal one frame is 23.2 ms.  A segment's tempo is read straight off
its endpoints -- ``(end - start) / beat_count`` -- so at 120 BPM, where a bar
spans about 86 frames, a single frame of boundary error moves the reported tempo
by 1.4 BPM and the decoder's own +/-2 frame snap tolerance can stack that to
roughly 5 BPM.  That quantisation, not the model, is the floor on tempo jitter.

Three passes address it, each usable on its own:

``snap_to_curve``
    Re-place every beat on a finer curve -- an audio pulse curve at the onset
    hop, or the beat posterior itself -- with parabolic interpolation, which
    takes placement below the frame grid entirely.

``fit_piecewise_constant_tempo``
    Fit beat time against beat index with straight-line segments and a fixed
    cost per break.  Constant tempo is then the cheapest description of an
    ordinary track, while a track that really does change tempo pays once and
    keeps the change.  This is the shape the export path wants: a handful of
    tempo events rather than one per bar.

``select_metrical_level``
    Re-score the whole grid at 2x, 1/2x and the triplet relatives against audio
    tempo evidence.  A uniformly double-time grid never triggers the decoder's
    octave-jump penalty, because it never jumps; only an absolute reference can
    catch it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .audio_tempo import TempoPrior

_EPSILON = 1e-12

# Relatives of the estimated pulse that beat trackers realistically confuse it
# with: the octaves, then the triplet/duplet pairs.
DEFAULT_LEVEL_RATIOS: tuple[float, ...] = (
    1.0,
    2.0,
    0.5,
    3.0,
    1.0 / 3.0,
    1.5,
    2.0 / 3.0,
)


def snap_to_curve(
    beat_seconds: np.ndarray,
    *,
    curve: np.ndarray,
    curve_hop_seconds: float,
    search_radius_seconds: float = 0.025,
) -> np.ndarray:
    """Move each beat to the nearby peak of a finer-resolution curve.

    ``curve`` is expected to be non-negative and sampled at
    ``curve_hop_seconds``.  A parabola through the winning sample and its two
    neighbours puts the returned time between samples, so the result is not
    quantised to either grid.
    """

    beats = np.asarray(beat_seconds, dtype=np.float64)
    values = np.asarray(curve, dtype=np.float64)
    if beats.size == 0 or values.size < 3 or curve_hop_seconds <= 0.0:
        return beats
    radius = max(1, int(round(float(search_radius_seconds) / float(curve_hop_seconds))))

    refined = np.empty_like(beats)
    for index, beat_time in enumerate(beats):
        center = int(round(beat_time / float(curve_hop_seconds)))
        low = max(0, center - radius)
        high = min(values.size, center + radius + 1)
        if high - low < 1:
            refined[index] = beat_time
            continue
        window = values[low:high]
        peak = low + int(np.argmax(window))
        if window.max() <= _EPSILON:
            refined[index] = beat_time
            continue
        if 0 < peak < values.size - 1:
            left = float(values[peak - 1])
            center_value = float(values[peak])
            right = float(values[peak + 1])
            denominator = left - 2.0 * center_value + right
            offset = (
                0.0
                if abs(denominator) < _EPSILON
                else 0.5 * (left - right) / denominator
            )
            # A parabola fitted to three samples can only place its vertex
            # inside the middle sample; anything else means the fit failed.
            offset = float(np.clip(offset, -0.5, 0.5))
        else:
            offset = 0.0
        refined[index] = (peak + offset) * float(curve_hop_seconds)

    # Snapping is per-beat, so a badly placed neighbour can invert the order.
    return np.maximum.accumulate(refined)


@dataclass(frozen=True)
class TempoSegment:
    """One constant-tempo run of the fitted grid."""

    start_beat: int
    end_beat: int  # exclusive
    start_seconds: float
    beat_period_seconds: float

    @property
    def bpm(self) -> float:
        if self.beat_period_seconds <= 0.0:
            return float("nan")
        return 60.0 / self.beat_period_seconds


def _regression_prefixes(times: np.ndarray) -> dict[str, np.ndarray]:
    count = times.size
    index = np.arange(count, dtype=np.float64)
    zero = np.zeros(1, dtype=np.float64)
    return {
        "n": np.concatenate([zero, np.cumsum(np.ones(count))]),
        "x": np.concatenate([zero, np.cumsum(index)]),
        "xx": np.concatenate([zero, np.cumsum(index * index)]),
        "y": np.concatenate([zero, np.cumsum(times)]),
        "xy": np.concatenate([zero, np.cumsum(index * times)]),
        "yy": np.concatenate([zero, np.cumsum(times * times)]),
    }


def _segment_fit(
    prefixes: dict[str, np.ndarray], start: int, end: int
) -> tuple[float, float, float]:
    """Least-squares line over beats ``[start, end)``: (sse, intercept, slope)."""

    count = prefixes["n"][end] - prefixes["n"][start]
    if count < 2.0:
        return 0.0, float(prefixes["y"][end] - prefixes["y"][start]), 0.0
    sum_x = prefixes["x"][end] - prefixes["x"][start]
    sum_xx = prefixes["xx"][end] - prefixes["xx"][start]
    sum_y = prefixes["y"][end] - prefixes["y"][start]
    sum_xy = prefixes["xy"][end] - prefixes["xy"][start]
    sum_yy = prefixes["yy"][end] - prefixes["yy"][start]
    centered_xx = sum_xx - sum_x * sum_x / count
    centered_xy = sum_xy - sum_x * sum_y / count
    centered_yy = sum_yy - sum_y * sum_y / count
    if centered_xx <= _EPSILON:
        return max(0.0, float(centered_yy)), float(sum_y / count), 0.0
    slope = centered_xy / centered_xx
    sse = max(0.0, float(centered_yy - centered_xy * slope))
    intercept = float((sum_y - slope * sum_x) / count)
    return sse, intercept, float(slope)


def fit_piecewise_constant_tempo(
    beat_seconds: np.ndarray,
    *,
    change_penalty_seconds: float = 0.08,
    min_segment_beats: int = 8,
) -> tuple[np.ndarray, tuple[TempoSegment, ...]]:
    """Replace the beat grid with the cheapest piecewise-constant-tempo fit.

    Beat time is regressed on beat index; a break costs
    ``change_penalty_seconds ** 2`` in the same units as the residual sum of
    squares.  Raising it drives the fit towards a single global tempo, lowering
    it lets the grid follow the input.  Returns the fitted times alongside the
    segments, so the export path can emit one tempo event per segment.
    """

    beats = np.asarray(beat_seconds, dtype=np.float64)
    count = beats.size
    minimum = max(2, int(min_segment_beats))
    if count < minimum:
        return beats, ()

    prefixes = _regression_prefixes(beats)
    penalty = float(change_penalty_seconds) ** 2

    # cost[j] is the best total for the first j beats; back[j] is where the
    # final segment of that solution starts.
    cost = np.full(count + 1, np.inf)
    back = np.zeros(count + 1, dtype=np.int64)
    cost[0] = 0.0
    for end in range(minimum, count + 1):
        for start in range(0, end - minimum + 1):
            if not np.isfinite(cost[start]):
                continue
            sse, _intercept, _slope = _segment_fit(prefixes, start, end)
            total = cost[start] + sse + (penalty if start > 0 else 0.0)
            if total < cost[end]:
                cost[end] = total
                back[end] = start
    if not np.isfinite(cost[count]):
        return beats, ()

    bounds: list[tuple[int, int]] = []
    cursor = count
    while cursor > 0:
        start = int(back[cursor])
        bounds.append((start, cursor))
        cursor = start
    bounds.reverse()

    fitted = np.empty_like(beats)
    segments: list[TempoSegment] = []
    for start, end in bounds:
        _sse, intercept, slope = _segment_fit(prefixes, start, end)
        indices = np.arange(start, end, dtype=np.float64)
        fitted[start:end] = intercept + slope * indices
        segments.append(
            TempoSegment(
                start_beat=start,
                end_beat=end,
                start_seconds=float(fitted[start]),
                beat_period_seconds=float(slope),
            )
        )
    return np.maximum.accumulate(fitted), tuple(segments)


def resample_metrical_level(beat_seconds: np.ndarray, ratio: float) -> np.ndarray:
    """Re-express a beat grid at ``ratio`` times its pulse rate.

    ``ratio > 1`` subdivides (2.0 gives eighth-note beats), ``ratio < 1``
    aggregates (0.5 keeps every second beat).  Positions are interpolated in
    beat-index space so a grid that speeds up stays consistent with itself.
    """

    beats = np.asarray(beat_seconds, dtype=np.float64)
    if beats.size < 2 or ratio <= 0.0:
        return beats
    if abs(ratio - 1.0) < 1e-9:
        return beats
    source_index = np.arange(beats.size, dtype=np.float64)
    target_count = int(math.floor((beats.size - 1) * ratio)) + 1
    if target_count < 2:
        return beats
    target_index = np.arange(target_count, dtype=np.float64) / ratio
    return np.interp(target_index, source_index, beats)


def beat_to_quarter_ratio(meter_denominator: int) -> float:
    """How many quarter notes one beat lasts under a meter denominator.

    A beat is one ``meter_denominator``-th note, so in 7/8 a beat is half a
    quarter and the quarter-note period is twice the beat period.  Every tempo
    figure the decoder and the prior exchange is quarter-note based, so beat
    spacing has to be converted before it is looked up.
    """

    return float(meter_denominator) / 4.0


_RESCALE_DENOMINATORS: frozenset[int] = frozenset({2, 4, 8})
_RESCALE_MAX_NUMERATOR = 12


def rescale_meter(
    meter_numerator: int, meter_denominator: int, ratio: float
) -> tuple[int, int] | None:
    """Restate a meter at ``ratio`` times the pulse rate, or None if it cannot be.

    Resampling the pulse does not change how long a bar lasts, only how many
    beats fill it, so both halves of the meter scale together: taking 8/8 down
    to half the pulse rate gives 4/4, the same bar counted in quarter notes.
    The quarter-note tempo is untouched by construction, which is what keeps the
    tempo map valid across the change.  Ratios that would produce a fractional
    numerator or denominator have no such restatement and return None.
    """

    scaled_numerator = float(meter_numerator) * float(ratio)
    scaled_denominator = float(meter_denominator) * float(ratio)
    numerator = int(round(scaled_numerator))
    denominator = int(round(scaled_denominator))
    if numerator < 1 or denominator < 1:
        return None
    if abs(scaled_numerator - numerator) > 1e-6:
        return None
    if abs(scaled_denominator - denominator) > 1e-6:
        return None
    # Refuse restatements no one writes. Doubling 7/8 is arithmetically 14/16,
    # which is a legal MIDI signature and a meter that does not exist; refusing
    # it leaves the decoder's own reading in place instead.
    if denominator not in _RESCALE_DENOMINATORS:
        return None
    if numerator > _RESCALE_MAX_NUMERATOR:
        return None
    return numerator, denominator


def score_beat_grid(
    beat_seconds: np.ndarray,
    *,
    tempo_prior: TempoPrior | None,
    beat_log_odds: np.ndarray | None,
    seconds_per_frame: float,
    tempo_prior_weight: float = 1.0,
    beat_evidence_weight: float = 1.0,
    beat_to_quarter: float = 1.0,
) -> float:
    """Per-beat evidence for a candidate grid, from audio tempo and the model."""

    beats = np.asarray(beat_seconds, dtype=np.float64)
    if beats.size < 2:
        return -float("inf")
    score = 0.0
    if tempo_prior is not None and tempo_prior_weight > 0.0:
        periods = (
            np.diff(beats) * float(beat_to_quarter) / float(seconds_per_frame)
        )
        centers = ((beats[:-1] + beats[1:]) * 0.5) / float(seconds_per_frame)
        total = 0.0
        for period, center in zip(periods, centers):
            if period <= 0.0:
                continue
            frame = int(round(center))
            total += tempo_prior.mean_log_prob(frame, frame + 1, float(period))
        score += float(tempo_prior_weight) * total / max(1, periods.size)
    if beat_log_odds is not None and beat_evidence_weight > 0.0:
        frames = np.clip(
            np.round(beats / float(seconds_per_frame)).astype(np.int64),
            0,
            beat_log_odds.size - 1,
        )
        score += float(beat_evidence_weight) * float(
            np.mean(beat_log_odds[frames])
        )
    return float(score)


def select_metrical_level(
    beat_seconds: np.ndarray,
    *,
    tempo_prior: TempoPrior | None,
    beat_log_odds: np.ndarray | None,
    seconds_per_frame: float,
    ratios: tuple[float, ...] = DEFAULT_LEVEL_RATIOS,
    tempo_prior_weight: float = 1.0,
    beat_evidence_weight: float = 1.0,
    beat_to_quarter: float = 1.0,
    margin: float = 0.0,
    min_quarter_bpm: float = 30.0,
    max_quarter_bpm: float = 300.0,
) -> tuple[np.ndarray, float]:
    """Pick the metrical level the audio supports best.

    ``margin`` is how much better an alternative must score before the decoder's
    own choice is overruled, so a near-tie leaves the DP result untouched.
    Returns the chosen grid and the ratio that produced it.
    """

    beats = np.asarray(beat_seconds, dtype=np.float64)
    if beats.size < 3 or (tempo_prior is None and beat_log_odds is None):
        return beats, 1.0

    def evaluate(candidate: np.ndarray) -> float:
        if candidate.size < 3:
            return -float("inf")
        median_period = float(np.median(np.diff(candidate)))
        if median_period <= 0.0:
            return -float("inf")
        quarter_bpm = 60.0 / (median_period * float(beat_to_quarter))
        if not min_quarter_bpm <= quarter_bpm <= max_quarter_bpm:
            return -float("inf")
        return score_beat_grid(
            candidate,
            tempo_prior=tempo_prior,
            beat_log_odds=beat_log_odds,
            seconds_per_frame=seconds_per_frame,
            tempo_prior_weight=tempo_prior_weight,
            beat_evidence_weight=beat_evidence_weight,
            beat_to_quarter=beat_to_quarter,
        )

    base_score = evaluate(beats)
    best_grid, best_ratio, best_score = beats, 1.0, base_score
    for ratio in ratios:
        if abs(ratio - 1.0) < 1e-9:
            continue
        candidate = resample_metrical_level(beats, ratio)
        candidate_score = evaluate(candidate)
        if candidate_score > best_score and candidate_score > base_score + float(margin):
            best_grid, best_ratio, best_score = candidate, float(ratio), candidate_score
    return best_grid, best_ratio
