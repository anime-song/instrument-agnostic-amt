"""Wire the audio tempo evidence into a single beat decoding entry point.

:mod:`audio_tempo` produces the evidence and :mod:`beat_refine` the individual
passes; this module is the order they run in and the bookkeeping that keeps the
rest of the pipeline consistent with the result.

That bookkeeping is the reason it exists.  The decoder reports beats as integer
frames and describes bars as frame spans, and the tempo-map export reads both.
Moving beats off the frame grid without moving the bar boundaries with them
leaves the export unable to match beats to bars, so it silently falls back to
dividing each bar uniformly and the refinement is thrown away.  Every time this
module returns is therefore passed through one shared monotonic warp.

Correcting the metrical level means restating the meter with the grid, not
just resampling the beats: halving a grid the decoder called 8/8 without also
calling it 4/4 leaves bars that no longer hold a whole number of beats.  Bar
boundaries and quarter-note tempo are invariant under the restatement, so it
costs nothing.  Meters with no integral restatement -- doubling 7/8 would give
14/16, which is not a meter anyone writes -- fall back to a second decode
steered by a prior peaked at the chosen period.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from .audio_tempo import (
    TempoPrior,
    TempoPriorConfig,
    compute_onset_envelopes,
    compute_pulse_curve,
    compute_tempo_prior,
    focused_tempo_prior,
)
from .beat_grid import (
    BeatGridDecodeResult,
    BeatGridDPConfig,
    MeterGridSegment,
    decode_beats_with_meter_grid_dp,
)
from .beat_refine import (
    DEFAULT_LEVEL_RATIOS,
    beat_to_quarter_ratio,
    fit_piecewise_constant_tempo,
    resample_metrical_level,
    rescale_meter,
    select_metrical_level,
    snap_to_curve,
)


def rescale_decode_to_level(decode: BeatGridDecodeResult, ratio: float) -> BeatGridDecodeResult | None:
    """Restate a decode at ``ratio`` times its pulse rate, meter included.

    Bar boundaries and quarter-note tempo are invariant under this change -- a
    bar lasts exactly as long, it is just counted in different note values -- so
    the mapped downbeats and each segment's ``quarter_note_bpm`` carry over
    untouched.  Only the beat grid and the meter are restated.  Returns None
    when any segment's meter has no integral restatement at this ratio, which is
    the caller's cue to fall back to decoding again.
    """

    rescaled: list[MeterGridSegment] = []
    for segment in decode.meter_segments:
        meter = rescale_meter(segment.meter_num, segment.meter_den, ratio)
        if meter is None:
            return None
        numerator, denominator = meter
        rescaled.append(replace(segment, meter_num=numerator, meter_den=denominator))

    beats = np.asarray(decode.beat_frames, dtype=np.float64)
    resampled = resample_metrical_level(beats, ratio)
    if resampled.size < 2:
        return None
    return replace(
        decode,
        beat_frames=tuple(int(round(frame)) for frame in resampled),
        meter_segments=tuple(rescaled),
    )


def dominant_meter_denominator(decode: BeatGridDecodeResult) -> int:
    """The denominator covering the most beats, or 4 when nothing was decoded.

    A beat is one denominator-th note, so this is what converts decoded beat
    spacing into the quarter-note period the tempo prior is indexed by. Taking
    the longest-running denominator keeps a stray bar of another meter from
    reinterpreting the whole track.
    """

    weights: dict[int, int] = {}
    for segment in decode.meter_segments:
        beats = int(segment.meter_num) * int(segment.bar_count)
        weights[int(segment.meter_den)] = weights.get(int(segment.meter_den), 0) + beats
    if not weights:
        return 4
    return max(weights.items(), key=lambda item: (item[1], -item[0]))[0]


@dataclass(frozen=True)
class AudioRefinementConfig:
    """Which audio-driven passes run, and how hard each one argues."""

    # Absolute tempo evidence inside the bar lattice. Zero by default: on GTZAN
    # it helps on its own (CMLt 0.459 -> 0.491) but overlaps with the metrical
    # level pass below, and running both was worse than the level pass alone.
    tempo_prior_weight: float = 0.0
    # Correct a grid that sits uniformly on the wrong metrical level. No
    # transition penalty can catch that, because a uniformly double-time grid
    # never jumps.
    select_metrical_level: bool = True
    # Largest meter denominator the level search is trusted on. At 4 it runs
    # only where a beat is a quarter note or longer; on an x/8 grid the pulse
    # is already a subdivision and every ratio it could pick is worse than
    # leaving the decoder alone.
    max_level_meter_denominator: int = 4
    # How much better an alternative level must score before the decoder's own
    # choice is overruled. Zero overrules on any improvement, which on material
    # the model already reads correctly is a net loss; 0.3 keeps the whole
    # held-out gain while halving that damage.
    level_margin: float = 0.3
    level_ratios: tuple[float, ...] = DEFAULT_LEVEL_RATIOS
    # Compare the decoded beat spacing to the prior directly. Converting it to a
    # quarter-note period first assumes the prior's peak is a quarter note,
    # which it is not in any x/8 meter -- see the module docstring of
    # audio_tempo for why the tempogram reports the pulse, not the notation.
    level_uses_quarter_period: bool = False
    # Restating the meter alongside the grid keeps bars and beats consistent and
    # needs no second decode. Ratios with no integral meter fall back to a
    # decode steered by a prior peaked at the chosen period.
    redecode_when_meter_cannot_scale: bool = True
    redecode_prior_weight: float = 4.0
    redecode_sigma_octaves: float = 0.06
    # Take beat placement off the 23 ms frame grid using the pulse curve.
    snap_to_pulse: bool = True
    snap_radius_seconds: float = 0.025
    # Collapse the grid onto piecewise-constant tempo. 0.5 s trades about 0.02
    # of beat F-measure for a two-thirds cut in tempo jitter; lower it towards
    # 0.08 to track the input more closely.
    piecewise_tempo: bool = True
    change_penalty_seconds: float = 0.50
    min_segment_beats: int = 8

    def __post_init__(self) -> None:
        if self.tempo_prior_weight < 0.0:
            raise ValueError("tempo_prior_weight must be non-negative")
        if self.redecode_prior_weight < 0.0:
            raise ValueError("redecode_prior_weight must be non-negative")
        if self.redecode_sigma_octaves <= 0.0:
            raise ValueError("redecode_sigma_octaves must be positive")
        if self.snap_radius_seconds < 0.0:
            raise ValueError("snap_radius_seconds must be non-negative")
        if self.change_penalty_seconds < 0.0:
            raise ValueError("change_penalty_seconds must be non-negative")
        if self.min_segment_beats < 2:
            raise ValueError("min_segment_beats must be at least two")
        if self.max_level_meter_denominator < 1:
            raise ValueError("max_level_meter_denominator must be positive")


@dataclass(frozen=True)
class AudioTempoEvidence:
    """Everything derived from the waveform, on the decoder's terms."""

    tempo_prior: TempoPrior
    pulse_curve: np.ndarray
    pulse_hop_seconds: float


@dataclass(frozen=True)
class RefinedBeatResult:
    """A decode plus the sub-frame times and the warp that produced them."""

    decode: BeatGridDecodeResult
    beat_seconds: tuple[float, ...]
    warp: Callable[[float], float]
    metrical_level_ratio: float
    tempo_segment_count: int

    def warp_all(self, seconds: Sequence[float]) -> list[float]:
        return [self.warp(float(value)) for value in seconds]


def analyze_audio(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    frame_count: int,
    config: TempoPriorConfig = TempoPriorConfig(),
    device: object | None = None,
) -> AudioTempoEvidence:
    """Derive the tempo prior and pulse curve for one track.

    Both passes read the same onset envelopes, so they are built once here and
    handed to each; ``device`` runs the transforms on an accelerator.
    """

    bands, mixed = compute_onset_envelopes(waveform, sample_rate=sample_rate, config=config)
    prior = compute_tempo_prior(
        sample_rate=sample_rate,
        target_hop_length=hop_length,
        target_frame_count=frame_count,
        config=config,
        onset_bands=bands,
        device=device,
    )
    pulse, pulse_hop_seconds = compute_pulse_curve(
        sample_rate=sample_rate,
        config=config,
        onset_envelope=mixed,
        device=device,
    )
    return AudioTempoEvidence(tempo_prior=prior, pulse_curve=pulse, pulse_hop_seconds=pulse_hop_seconds)


def build_time_warp(original_seconds: np.ndarray, refined_seconds: np.ndarray) -> Callable[[float], float]:
    """Monotonic map from pre-refinement times to post-refinement times.

    Anchored on the beats themselves and linear in between, so any other time in
    the result -- a bar boundary, a segment edge -- moves with the beats it sits
    among instead of drifting away from them.
    """

    source = np.asarray(original_seconds, dtype=np.float64)
    target = np.asarray(refined_seconds, dtype=np.float64)
    if source.size < 2 or source.size != target.size:
        return lambda value: float(value)

    order = np.argsort(source)
    source = source[order]
    target = target[order]
    # np.interp needs strictly increasing x; duplicated beat frames would make
    # the map ambiguous, so keep the first of each run.
    keep = np.concatenate([[True], np.diff(source) > 1e-9])
    source = source[keep]
    target = target[keep]
    if source.size < 2:
        return lambda value: float(value)

    head_slope = (target[1] - target[0]) / (source[1] - source[0])
    tail_slope = (target[-1] - target[-2]) / (source[-1] - source[-2])

    def warp(value: float) -> float:
        point = float(value)
        if point <= source[0]:
            return float(target[0] + head_slope * (point - source[0]))
        if point >= source[-1]:
            return float(target[-1] + tail_slope * (point - source[-1]))
        return float(np.interp(point, source, target))

    return warp


def _beat_log_odds(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def decode_beats_with_audio(
    *,
    beat_probabilities: np.ndarray,
    downbeat_probabilities: np.ndarray,
    meter_logits: np.ndarray,
    meter_classes: list[tuple[int, int]],
    config: BeatGridDPConfig,
    group_boundary_probabilities: np.ndarray | None = None,
    evidence: AudioTempoEvidence | None = None,
    refinement: AudioRefinementConfig = AudioRefinementConfig(),
) -> RefinedBeatResult:
    """Decode beats using audio tempo evidence, then refine their placement.

    With ``evidence`` omitted this is the MIDI-only decoder plus an identity
    warp, so callers can run the same path whether or not audio is available.
    """

    seconds_per_frame = config.seconds_per_frame
    prior = None if evidence is None else evidence.tempo_prior
    active_config = config
    if prior is not None and refinement.tempo_prior_weight > 0.0:
        active_config = replace(config, tempo_prior_weight=float(refinement.tempo_prior_weight))

    decode = decode_beats_with_meter_grid_dp(
        beat_probabilities=beat_probabilities,
        downbeat_probabilities=downbeat_probabilities,
        group_boundary_probabilities=group_boundary_probabilities,
        meter_logits=meter_logits,
        meter_classes=meter_classes,
        config=active_config,
        tempo_prior=prior,
    )
    beats = np.asarray(decode.beat_frames, dtype=np.float64) * seconds_per_frame
    if beats.size < 2:
        return RefinedBeatResult(
            decode=decode,
            beat_seconds=tuple(beats.tolist()),
            warp=lambda value: float(value),
            metrical_level_ratio=1.0,
            tempo_segment_count=0,
        )

    level_ratio = 1.0
    meter_denominator = dominant_meter_denominator(decode)
    if (
        prior is not None
        and refinement.select_metrical_level
        and meter_denominator <= refinement.max_level_meter_denominator
    ):
        beat_to_quarter = beat_to_quarter_ratio(meter_denominator) if refinement.level_uses_quarter_period else 1.0
        _candidate, level_ratio = select_metrical_level(
            beats,
            tempo_prior=prior,
            beat_log_odds=_beat_log_odds(beat_probabilities),
            seconds_per_frame=seconds_per_frame,
            ratios=refinement.level_ratios,
            beat_to_quarter=beat_to_quarter,
            margin=float(refinement.level_margin),
            min_quarter_bpm=float(config.min_quarter_bpm),
            max_quarter_bpm=float(config.max_quarter_bpm),
        )
        rescaled = None if abs(level_ratio - 1.0) <= 1e-9 else rescale_decode_to_level(decode, level_ratio)
        if rescaled is not None:
            # The rescaled decode stores integer frames for the meter segments,
            # but the beat grid itself keeps the unrounded resampled times: a
            # round trip through the 23 ms frame grid here would reintroduce the
            # quantisation the snap and fit passes exist to remove.
            decode = rescaled
            beats = resample_metrical_level(beats, level_ratio)
        elif (
            abs(level_ratio - 1.0) > 1e-9
            and refinement.redecode_when_meter_cannot_scale
            and refinement.redecode_prior_weight > 0.0
        ):
            median_period = float(np.median(np.diff(beats)))
            if median_period > 0.0:
                retried = decode_beats_with_meter_grid_dp(
                    beat_probabilities=beat_probabilities,
                    downbeat_probabilities=downbeat_probabilities,
                    group_boundary_probabilities=group_boundary_probabilities,
                    meter_logits=meter_logits,
                    meter_classes=meter_classes,
                    config=replace(
                        config,
                        tempo_prior_weight=float(refinement.redecode_prior_weight),
                    ),
                    tempo_prior=focused_tempo_prior(
                        prior,
                        # The prior is indexed by quarter-note period, so beat
                        # spacing is converted before it becomes a target: in
                        # 7/8 a beat is half a quarter note.
                        (median_period / level_ratio) * beat_to_quarter / seconds_per_frame,
                        sigma_octaves=float(refinement.redecode_sigma_octaves),
                    ),
                )
                retried_beats = np.asarray(retried.beat_frames, dtype=np.float64) * seconds_per_frame
                if retried_beats.size >= 2:
                    decode, beats = retried, retried_beats
                else:
                    level_ratio = 1.0
            else:
                level_ratio = 1.0

    original = beats.copy()
    if (
        refinement.snap_to_pulse
        and evidence is not None
        and evidence.pulse_curve.size > 2
        and evidence.pulse_hop_seconds > 0.0
    ):
        beats = snap_to_curve(
            beats,
            curve=evidence.pulse_curve,
            curve_hop_seconds=evidence.pulse_hop_seconds,
            search_radius_seconds=float(refinement.snap_radius_seconds),
        )

    segment_count = 0
    if refinement.piecewise_tempo:
        beats, segments = fit_piecewise_constant_tempo(
            beats,
            change_penalty_seconds=float(refinement.change_penalty_seconds),
            min_segment_beats=int(refinement.min_segment_beats),
        )
        segment_count = len(segments)

    return RefinedBeatResult(
        decode=decode,
        beat_seconds=tuple(beats.tolist()),
        warp=build_time_warp(original, beats),
        metrical_level_ratio=float(level_ratio),
        tempo_segment_count=segment_count,
    )
