from __future__ import annotations

import math
import tempfile
from bisect import bisect_right
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import mido


DEFAULT_TEMPO = 500_000
TARGET_MAX_DRIFT_SECONDS = 0.001
STABLE_TEMPO_MAX_RATIO = 1.03
STABLE_TEMPO_MAX_BOUNDARY_ERROR_SECONDS = 0.001
METER_GAP_MAX_BEATS = 4
METER_GAP_MAX_DURATION_ERROR_RATIO = 0.40
EXPRESSIVE_TREND_MIN_BEATS = 4
EXPRESSIVE_TREND_MIN_SPAN_RATIO = 0.15
EXPRESSIVE_TREND_MIN_CORRELATION = 0.90
EXPRESSIVE_TREND_MIN_STEP_AGREEMENT = 2.0 / 3.0


@dataclass(frozen=True)
class MeterSegmentSpec:
    start_seconds: float
    end_seconds: float
    numerator: int
    denominator: int
    bar_count: int = 1
    score: float = 0.0
    gap_filled_beats: int = 0


@dataclass(frozen=True)
class TempoMappedMidiExportResult:
    output_path: Path
    ticks_per_beat: int
    used_predicted_tempo: bool
    tempo_event_count: int
    time_signature_count: int
    chord_marker_count: int
    key_signature_count: int
    retimed_event_count: int
    max_event_drift_seconds: float
    max_note_drift_seconds: float
    regions: tuple[dict[str, Any], ...]

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["output_path"] = str(self.output_path)
        data["regions"] = [dict(region) for region in self.regions]
        return data


@dataclass(frozen=True)
class _TempoConverter:
    ticks_per_beat: int
    start_ticks: tuple[int, ...]
    start_seconds: tuple[float, ...]
    tempos: tuple[int, ...]

    @classmethod
    def from_events(
        cls,
        *,
        ticks_per_beat: int,
        tempo_events: Sequence[tuple[int, int]],
    ) -> "_TempoConverter":
        if ticks_per_beat <= 0:
            raise ValueError("ticks_per_beat must be positive")
        by_tick: dict[int, int] = {0: DEFAULT_TEMPO}
        for tick, tempo in tempo_events:
            if int(tick) >= 0 and int(tempo) > 0:
                by_tick[int(tick)] = int(tempo)
        ordered = sorted(by_tick.items())
        ticks: list[int] = []
        seconds: list[float] = []
        tempos: list[int] = []
        current_seconds = 0.0
        previous_tick, previous_tempo = ordered[0]
        for tick, tempo in ordered:
            current_seconds += (
                float(tick - previous_tick)
                * float(previous_tempo)
                / 1_000_000.0
                / float(ticks_per_beat)
            )
            ticks.append(int(tick))
            seconds.append(float(current_seconds))
            tempos.append(int(tempo))
            previous_tick, previous_tempo = int(tick), int(tempo)
        return cls(
            ticks_per_beat=int(ticks_per_beat),
            start_ticks=tuple(ticks),
            start_seconds=tuple(seconds),
            tempos=tuple(tempos),
        )

    def tick_to_seconds(self, tick: int | float) -> float:
        index = max(0, bisect_right(self.start_ticks, float(tick)) - 1)
        return float(self.start_seconds[index]) + (
            float(tick) - float(self.start_ticks[index])
        ) * float(self.tempos[index]) / 1_000_000.0 / float(self.ticks_per_beat)

    def seconds_to_tick(self, seconds: float) -> int:
        seconds = max(0.0, float(seconds))
        index = max(0, bisect_right(self.start_seconds, seconds) - 1)
        tick = float(self.start_ticks[index]) + (
            seconds - float(self.start_seconds[index])
        ) * 1_000_000.0 * float(self.ticks_per_beat) / float(self.tempos[index])
        return max(0, int(round(tick)))


@dataclass(frozen=True)
class _TimedMessage:
    seconds: float
    order: int
    message: mido.Message | mido.MetaMessage


@dataclass(frozen=True)
class _GridInterval:
    start_seconds: float
    end_seconds: float
    delta_ticks: int
    segment_index: int
    expressive: bool = False


def _source_tempo_events(midi: mido.MidiFile) -> tuple[tuple[int, int], ...]:
    raw: list[tuple[int, int, int]] = []
    serial = 0
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.type == "set_tempo":
                raw.append((tick, serial, int(message.tempo)))
            serial += 1
    by_tick: dict[int, tuple[int, int]] = {0: (-1, DEFAULT_TEMPO)}
    for tick, order, tempo in sorted(raw):
        by_tick[int(tick)] = (int(order), int(tempo))
    return tuple((tick, value[1]) for tick, value in sorted(by_tick.items()))


def _source_time_signature_events(
    midi: mido.MidiFile,
) -> tuple[tuple[int, int, int], ...]:
    raw: list[tuple[int, int, int, int]] = []
    serial = 0
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.type == "time_signature":
                raw.append(
                    (
                        tick,
                        serial,
                        int(message.numerator),
                        int(message.denominator),
                    )
                )
            serial += 1
    by_tick: dict[int, tuple[int, int, int]] = {}
    for tick, order, numerator, denominator in sorted(raw):
        by_tick[int(tick)] = (int(order), int(numerator), int(denominator))
    return tuple(
        (tick, values[1], values[2]) for tick, values in sorted(by_tick.items())
    )


def _timed_source_tracks(
    midi: mido.MidiFile,
    converter: _TempoConverter,
    *,
    exclude_key_signatures: bool = False,
) -> tuple[tuple[_TimedMessage, ...], ...]:
    excluded = {"set_tempo", "time_signature", "smpte_offset", "end_of_track"}
    if exclude_key_signatures:
        excluded.add("key_signature")
    tracks: list[tuple[_TimedMessage, ...]] = []
    for track in midi.tracks:
        tick = 0
        events: list[_TimedMessage] = []
        for order, message in enumerate(track):
            tick += int(message.time)
            if message.type in excluded:
                continue
            events.append(
                _TimedMessage(
                    seconds=converter.tick_to_seconds(tick),
                    order=int(order),
                    message=message.copy(time=0),
                )
            )
        tracks.append(tuple(events))
    return tuple(tracks)


def _meter_segments(
    meter_segments: Sequence[MeterSegmentSpec],
) -> tuple[MeterSegmentSpec, ...]:
    valid: list[MeterSegmentSpec] = []
    for segment in sorted(meter_segments, key=lambda item: item.start_seconds):
        if segment.end_seconds <= segment.start_seconds:
            continue
        if segment.numerator <= 0 or segment.denominator <= 0:
            continue
        valid.append(
            MeterSegmentSpec(
                start_seconds=max(0.0, float(segment.start_seconds)),
                end_seconds=float(segment.end_seconds),
                numerator=int(segment.numerator),
                denominator=int(segment.denominator),
                bar_count=max(1, int(segment.bar_count)),
                score=float(segment.score),
                gap_filled_beats=max(0, int(segment.gap_filled_beats)),
            )
        )
    return tuple(valid)


def _segment_grid_times(
    segment: MeterSegmentSpec,
    beat_times: Sequence[float],
) -> tuple[float, ...]:
    expected_intervals = int(segment.numerator) * int(segment.bar_count)
    epsilon = 1e-6
    interior = sorted(
        {
            float(time)
            for time in beat_times
            if segment.start_seconds + epsilon
            < float(time)
            < segment.end_seconds - epsilon
        }
    )
    points = [float(segment.start_seconds), *interior, float(segment.end_seconds)]
    if len(points) != expected_intervals + 1:
        duration = float(segment.end_seconds - segment.start_seconds)
        points = [
            float(segment.start_seconds) + duration * index / expected_intervals
            for index in range(expected_intervals + 1)
        ]
    return tuple(points)


def _meter_gap_beat_count(
    left: MeterSegmentSpec,
    right: MeterSegmentSpec,
    beat_times: Sequence[float],
) -> int:
    gap_seconds = float(right.start_seconds) - float(left.end_seconds)
    if gap_seconds <= 1e-6:
        return 0

    left_points = _segment_grid_times(left, beat_times)
    right_points = _segment_grid_times(right, beat_times)
    left_durations = [
        end - start for start, end in zip(left_points[:-1], left_points[1:])
    ]
    right_durations = [
        (end - start) * float(right.denominator) / float(left.denominator)
        for start, end in zip(right_points[:-1], right_points[1:])
    ]
    neighboring_durations = [*left_durations[-2:], *right_durations[:2]]
    if not neighboring_durations:
        return 0
    reference_duration = float(median(neighboring_durations))
    if reference_duration <= 0.0:
        return 0

    gap_beats = int(round(gap_seconds / reference_duration))
    if gap_beats <= 0 or gap_beats > METER_GAP_MAX_BEATS:
        return 0
    inferred_duration = gap_seconds / gap_beats
    duration_error_ratio = abs(inferred_duration - reference_duration) / (
        reference_duration
    )
    if duration_error_ratio > METER_GAP_MAX_DURATION_ERROR_RATIO:
        return 0
    return int(gap_beats)


def _fill_meter_segment_gaps(
    segments: Sequence[MeterSegmentSpec],
    beat_times: Sequence[float],
) -> tuple[MeterSegmentSpec, ...]:
    """Fold short uncovered beat spans into the preceding bar."""

    filled: list[MeterSegmentSpec] = []
    for index, segment in enumerate(segments):
        if index + 1 >= len(segments):
            filled.append(segment)
            continue
        next_segment = segments[index + 1]
        gap_beats = _meter_gap_beat_count(segment, next_segment, beat_times)
        if gap_beats <= 0:
            filled.append(segment)
            continue

        last_bar_start = float(segment.start_seconds)
        if segment.bar_count > 1:
            bar_seconds = (
                float(segment.end_seconds) - float(segment.start_seconds)
            ) / int(segment.bar_count)
            last_bar_start = float(segment.end_seconds) - bar_seconds
            filled.append(
                MeterSegmentSpec(
                    start_seconds=float(segment.start_seconds),
                    end_seconds=last_bar_start,
                    numerator=int(segment.numerator),
                    denominator=int(segment.denominator),
                    bar_count=int(segment.bar_count) - 1,
                    score=float(segment.score),
                    gap_filled_beats=int(segment.gap_filled_beats),
                )
            )
        filled.append(
            MeterSegmentSpec(
                start_seconds=last_bar_start,
                end_seconds=float(next_segment.start_seconds),
                numerator=int(segment.numerator) + gap_beats,
                denominator=int(segment.denominator),
                bar_count=1,
                score=float(segment.score),
                gap_filled_beats=int(segment.gap_filled_beats) + gap_beats,
            )
        )
    return tuple(filled)


def _smoothed_expressive_grid_times(
    points: Sequence[float],
) -> tuple[float, ...] | None:
    """Fit a monotonic beat-duration trend when a bar contains clear rubato."""

    durations = [
        float(end) - float(start) for start, end in zip(points[:-1], points[1:])
    ]
    beat_count = len(durations)
    if beat_count < EXPRESSIVE_TREND_MIN_BEATS or any(
        duration <= 0.0 for duration in durations
    ):
        return None

    mean_duration = sum(durations) / beat_count
    center = (beat_count - 1) / 2.0
    x_variance = sum((index - center) ** 2 for index in range(beat_count))
    y_variance = sum((duration - mean_duration) ** 2 for duration in durations)
    if x_variance <= 0.0 or y_variance <= 0.0:
        return None
    covariance = sum(
        (index - center) * (duration - mean_duration)
        for index, duration in enumerate(durations)
    )
    slope = covariance / x_variance
    correlation = covariance / math.sqrt(x_variance * y_variance)
    span_ratio = abs(slope) * (beat_count - 1) / mean_duration
    agreeing_steps = sum(
        1
        for left, right in zip(durations[:-1], durations[1:])
        if (right - left) * slope > 0.0
    )
    step_agreement = agreeing_steps / max(1, beat_count - 1)
    if (
        span_ratio < EXPRESSIVE_TREND_MIN_SPAN_RATIO
        or abs(correlation) < EXPRESSIVE_TREND_MIN_CORRELATION
        or step_agreement < EXPRESSIVE_TREND_MIN_STEP_AGREEMENT
    ):
        return None

    fitted_durations = [
        mean_duration + slope * (index - center) for index in range(beat_count)
    ]
    if any(duration <= 0.0 for duration in fitted_durations):
        return None
    fitted_points = [float(points[0])]
    for duration in fitted_durations:
        fitted_points.append(fitted_points[-1] + duration)
    # Centered least squares preserves the total duration mathematically. Pin the
    # final point anyway so floating-point noise cannot move the next barline.
    fitted_points[-1] = float(points[-1])
    return tuple(fitted_points)


def _segment_tempo_intervals(
    *,
    segment: MeterSegmentSpec,
    segment_index: int,
    beat_times: Sequence[float],
    beat_ticks: int,
) -> tuple[tuple[_GridInterval, ...], bool]:
    grid_times = _segment_grid_times(segment, beat_times)
    beats_per_bar = int(segment.numerator)
    intervals: list[_GridInterval] = []
    used_expressive_curve = False
    for bar_index in range(int(segment.bar_count)):
        point_start = bar_index * beats_per_bar
        bar_points = grid_times[point_start : point_start + beats_per_bar + 1]
        if len(bar_points) != beats_per_bar + 1:
            continue
        expressive_points = _smoothed_expressive_grid_times(bar_points)
        if expressive_points is None:
            intervals.append(
                _GridInterval(
                    start_seconds=float(bar_points[0]),
                    end_seconds=float(bar_points[-1]),
                    delta_ticks=int(beat_ticks) * beats_per_bar,
                    segment_index=int(segment_index),
                    expressive=False,
                )
            )
            continue

        used_expressive_curve = True
        for start, end in zip(expressive_points[:-1], expressive_points[1:]):
            intervals.append(
                _GridInterval(
                    start_seconds=float(start),
                    end_seconds=float(end),
                    delta_ticks=int(beat_ticks),
                    segment_index=int(segment_index),
                    expressive=True,
                )
            )
    return tuple(intervals), used_expressive_curve


def _interval_tempo(interval: _GridInterval, ticks_per_beat: int) -> float:
    return (
        (float(interval.end_seconds) - float(interval.start_seconds))
        * 1_000_000.0
        * float(ticks_per_beat)
        / float(interval.delta_ticks)
    )


def _stable_group_fits(
    intervals: Sequence[_GridInterval],
    *,
    segments: Sequence[MeterSegmentSpec],
    ticks_per_beat: int,
) -> bool:
    if not intervals or any(interval.expressive for interval in intervals):
        return False
    first_segment = segments[intervals[0].segment_index]
    first_meter = (first_segment.numerator, first_segment.denominator)
    if any(
        (
            segments[interval.segment_index].numerator,
            segments[interval.segment_index].denominator,
        )
        != first_meter
        for interval in intervals[1:]
    ):
        return False
    if any(
        abs(left.end_seconds - right.start_seconds) > 1e-6
        for left, right in zip(intervals[:-1], intervals[1:])
    ):
        return False

    tempos = [_interval_tempo(interval, ticks_per_beat) for interval in intervals]
    if max(tempos) / min(tempos) > STABLE_TEMPO_MAX_RATIO:
        return False

    total_ticks = sum(interval.delta_ticks for interval in intervals)
    total_seconds = intervals[-1].end_seconds - intervals[0].start_seconds
    seconds_per_tick = total_seconds / total_ticks
    elapsed_ticks = 0
    for interval in intervals[:-1]:
        elapsed_ticks += interval.delta_ticks
        expected_seconds = interval.end_seconds - intervals[0].start_seconds
        mapped_seconds = elapsed_ticks * seconds_per_tick
        if (
            abs(mapped_seconds - expected_seconds)
            > STABLE_TEMPO_MAX_BOUNDARY_ERROR_SECONDS
        ):
            return False
    return True


def _merge_stable_tempo_intervals(
    intervals: Sequence[_GridInterval],
    *,
    segments: Sequence[MeterSegmentSpec],
    ticks_per_beat: int,
) -> tuple[_GridInterval, ...]:
    merged: list[_GridInterval] = []
    stable_group: list[_GridInterval] = []

    def flush_stable_group() -> None:
        if not stable_group:
            return
        merged.append(
            _GridInterval(
                start_seconds=stable_group[0].start_seconds,
                end_seconds=stable_group[-1].end_seconds,
                delta_ticks=sum(interval.delta_ticks for interval in stable_group),
                segment_index=stable_group[0].segment_index,
                expressive=False,
            )
        )
        stable_group.clear()

    for interval in intervals:
        if interval.expressive:
            flush_stable_group()
            merged.append(interval)
            continue
        candidate = [*stable_group, interval]
        if stable_group and not _stable_group_fits(
            candidate,
            segments=segments,
            ticks_per_beat=ticks_per_beat,
        ):
            flush_stable_group()
        stable_group.append(interval)
    flush_stable_group()
    return tuple(merged)


def _build_predicted_map(
    *,
    beat_times: Sequence[float],
    meter_segments: Sequence[MeterSegmentSpec],
    ticks_per_beat: int,
    duration_seconds: float,
) -> (
    tuple[
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int, int], ...],
        tuple[dict[str, Any], ...],
    ]
    | None
):
    segments = _fill_meter_segment_gaps(
        _meter_segments(meter_segments),
        beat_times,
    )
    if not segments:
        return None

    raw_intervals: list[_GridInterval] = []
    regions: list[dict[str, Any]] = []
    for segment_index, segment in enumerate(segments):
        beat_ticks = int(round(ticks_per_beat * 4.0 / segment.denominator))
        if beat_ticks <= 0:
            continue
        segment_intervals, used_expressive_curve = _segment_tempo_intervals(
            segment=segment,
            segment_index=segment_index,
            beat_times=beat_times,
            beat_ticks=beat_ticks,
        )
        raw_intervals.extend(segment_intervals)
        regions.append(
            {
                "start": float(segment.start_seconds),
                "end": float(segment.end_seconds),
                "meter": f"{segment.numerator}/{segment.denominator}",
                "bar_count": int(segment.bar_count),
                "source": (
                    "gap_extended"
                    if segment.gap_filled_beats > 0
                    else "interpolated"
                    if segment.bar_count > 1
                    else "detected"
                ),
                "score": float(segment.score),
                "gap_filled_beats": int(segment.gap_filled_beats),
                "tempo_mode": (
                    "expressive_beat_curve" if used_expressive_curve else "smoothed_bar"
                ),
            }
        )
    if not raw_intervals:
        return None
    intervals = _merge_stable_tempo_intervals(
        raw_intervals,
        segments=segments,
        ticks_per_beat=ticks_per_beat,
    )

    first_interval = intervals[0]
    first_tempo = int(
        round(
            (first_interval.end_seconds - first_interval.start_seconds)
            * 1_000_000.0
            * ticks_per_beat
            / first_interval.delta_ticks
        )
    )
    first_tempo = max(1, min(16_777_215, first_tempo))
    first_segment = segments[first_interval.segment_index]
    first_bar_ticks = int(
        round(
            ticks_per_beat * first_segment.numerator * 4.0 / first_segment.denominator
        )
    )
    first_bar_seconds = (
        first_segment.end_seconds - first_segment.start_seconds
    ) / first_segment.bar_count
    if first_interval.start_seconds <= 0.0:
        current_tick = 0
    else:
        leading_bars = max(
            1, int(round(first_interval.start_seconds / max(first_bar_seconds, 1e-6)))
        )
        current_tick = leading_bars * first_bar_ticks
        first_tempo = int(
            round(
                first_interval.start_seconds
                * 1_000_000.0
                * ticks_per_beat
                / current_tick
            )
        )
        first_tempo = max(1, min(16_777_215, first_tempo))

    tempo_events: list[tuple[int, int]] = [(0, first_tempo)]
    first_meter = (first_segment.numerator, first_segment.denominator)
    # The leading region is extrapolated in the first decoded meter, so declare
    # that meter at tick 0 as well. Otherwise MIDI readers assume 4/4 until the
    # first detected downbeat. A non-4/4 signature inserted there can land
    # mid-measure in the implicit 4/4 grid and shift every displayed barline.
    signature_events: list[tuple[int, int, int]] = [(0, first_meter[0], first_meter[1])]
    previous_meter: tuple[int, int] | None = first_meter
    previous_segment_index = -1
    for interval in intervals:
        segment = segments[interval.segment_index]
        if interval.segment_index != previous_segment_index:
            meter = (segment.numerator, segment.denominator)
            if meter != previous_meter or previous_meter is None:
                signature_events.append((current_tick, meter[0], meter[1]))
                previous_meter = meter
            previous_segment_index = interval.segment_index
        tempo = int(
            round(
                (interval.end_seconds - interval.start_seconds)
                * 1_000_000.0
                * ticks_per_beat
                / interval.delta_ticks
            )
        )
        tempo_events.append((current_tick, max(1, min(16_777_215, int(tempo)))))
        current_tick += int(interval.delta_ticks)

    if intervals[0].start_seconds > 0.0:
        regions.insert(
            0,
            {
                "start": 0.0,
                "end": float(intervals[0].start_seconds),
                "meter": f"{first_segment.numerator}/{first_segment.denominator}",
                "bar_count": None,
                "source": "extrapolated",
                "score": None,
            },
        )
    if intervals[-1].end_seconds < duration_seconds:
        last_segment = segments[intervals[-1].segment_index]
        regions.append(
            {
                "start": float(intervals[-1].end_seconds),
                "end": float(duration_seconds),
                "meter": f"{last_segment.numerator}/{last_segment.denominator}",
                "bar_count": None,
                "source": "extrapolated",
                "score": None,
            }
        )

    by_tick: dict[int, int] = {}
    for tick, tempo in tempo_events:
        by_tick[int(tick)] = int(tempo)
    deduplicated_tempos: list[tuple[int, int]] = []
    for tick, tempo in sorted(by_tick.items()):
        if deduplicated_tempos and tempo == deduplicated_tempos[-1][1]:
            continue
        deduplicated_tempos.append((int(tick), int(tempo)))
    return (
        tuple(deduplicated_tempos),
        tuple(signature_events),
        tuple(regions),
    )


def _append_timed_messages(
    *,
    track: mido.MidiTrack,
    events: Sequence[tuple[int, int, mido.Message | mido.MetaMessage]],
) -> None:
    previous_tick = 0
    for tick, _order, message in sorted(events, key=lambda item: (item[0], item[1])):
        tick = max(previous_tick, int(tick))
        track.append(message.copy(time=tick - previous_tick))
        previous_tick = tick
    track.append(mido.MetaMessage("end_of_track", time=0))


def _resolve_ticks_per_beat(
    *,
    minimum_ticks_per_beat: int,
    tempo_events: Sequence[tuple[int, int]],
    target_max_drift_seconds: float,
) -> int:
    if minimum_ticks_per_beat <= 0 or minimum_ticks_per_beat > 32_767:
        raise ValueError("ticks_per_beat must be in the range 1..32767")
    if target_max_drift_seconds <= 0.0:
        raise ValueError("target_max_drift_seconds must be positive")
    maximum_tempo = max((int(tempo) for _tick, tempo in tempo_events), default=0)
    required = int(math.ceil(maximum_tempo / (2_000_000.0 * target_max_drift_seconds)))
    if required <= minimum_ticks_per_beat:
        return int(minimum_ticks_per_beat)
    multiplier = int(math.ceil(required / minimum_ticks_per_beat))
    resolved = int(minimum_ticks_per_beat) * multiplier
    return min(32_767, resolved)


def _verify_reloaded_note_timing(
    *,
    output_midi_path: Path,
    source_tracks: Sequence[Sequence[_TimedMessage]],
) -> float:
    reloaded = mido.MidiFile(str(output_midi_path), clip=True)
    converter = _TempoConverter.from_events(
        ticks_per_beat=int(reloaded.ticks_per_beat),
        tempo_events=_source_tempo_events(reloaded),
    )
    reloaded_tracks = _timed_source_tracks(reloaded, converter)
    performance_tracks = reloaded_tracks[2:]
    if len(performance_tracks) != len(source_tracks):
        raise RuntimeError("tempo-mapped MIDI changed the performance track count")

    max_drift = 0.0
    for source_track, output_track in zip(source_tracks, performance_tracks):
        source_notes = [
            event
            for event in source_track
            if event.message.type in {"note_on", "note_off"}
        ]
        output_notes = [
            event
            for event in output_track
            if event.message.type in {"note_on", "note_off"}
        ]
        if len(source_notes) != len(output_notes):
            raise RuntimeError("tempo-mapped MIDI changed the note event count")
        for source_event, output_event in zip(source_notes, output_notes):
            source_message = source_event.message
            output_message = output_event.message
            source_identity = (
                source_message.type,
                int(source_message.channel),
                int(source_message.note),
                int(source_message.velocity),
            )
            output_identity = (
                output_message.type,
                int(output_message.channel),
                int(output_message.note),
                int(output_message.velocity),
            )
            if source_identity != output_identity:
                raise RuntimeError("tempo-mapped MIDI changed note event ordering")
            max_drift = max(
                max_drift,
                abs(float(source_event.seconds) - float(output_event.seconds)),
            )
    return float(max_drift)


def export_tempo_mapped_midi(
    *,
    source_midi_path: Path,
    output_midi_path: Path,
    beat_times: Sequence[float],
    meter_segments: Sequence[MeterSegmentSpec],
    chord_segments: Sequence[dict[str, Any]],
    duration_seconds: float,
    key_segments: Sequence[dict[str, Any]] = (),
    ticks_per_beat: int = 960,
    target_max_drift_seconds: float = TARGET_MAX_DRIFT_SECONDS,
) -> TempoMappedMidiExportResult:
    """Write a tempo-mapped MIDI while preserving every source event's seconds."""

    source_midi_path = Path(source_midi_path)
    output_midi_path = Path(output_midi_path)
    if source_midi_path.resolve() == output_midi_path.resolve():
        raise ValueError("tempo-mapped MIDI output must differ from the source MIDI")
    if ticks_per_beat <= 0 or ticks_per_beat > 32_767:
        raise ValueError("ticks_per_beat must be in the range 1..32767")

    source_midi = mido.MidiFile(str(source_midi_path), clip=True)
    source_tempos = _source_tempo_events(source_midi)
    source_time_signatures = _source_time_signature_events(source_midi)
    source_converter = _TempoConverter.from_events(
        ticks_per_beat=int(source_midi.ticks_per_beat),
        tempo_events=source_tempos,
    )
    predicted_key_segments = tuple(
        segment for segment in key_segments if str(segment.get("key", "N")) != "N"
    )
    source_tracks = _timed_source_tracks(
        source_midi,
        source_converter,
        exclude_key_signatures=bool(predicted_key_segments),
    )

    predicted_map = _build_predicted_map(
        beat_times=beat_times,
        meter_segments=meter_segments,
        ticks_per_beat=int(ticks_per_beat),
        duration_seconds=float(duration_seconds),
    )
    if predicted_map is None:
        resolution_probe = source_tempos
    else:
        resolution_probe = predicted_map[0]
    resolved_ticks_per_beat = _resolve_ticks_per_beat(
        minimum_ticks_per_beat=int(ticks_per_beat),
        tempo_events=resolution_probe,
        target_max_drift_seconds=float(target_max_drift_seconds),
    )
    if resolved_ticks_per_beat != int(ticks_per_beat):
        predicted_map = _build_predicted_map(
            beat_times=beat_times,
            meter_segments=meter_segments,
            ticks_per_beat=resolved_ticks_per_beat,
            duration_seconds=float(duration_seconds),
        )

    if predicted_map is None:
        tempo_events = tuple(
            (
                int(round(tick * resolved_ticks_per_beat / source_midi.ticks_per_beat)),
                int(tempo),
            )
            for tick, tempo in source_tempos
        )
        signature_events = tuple(
            (
                int(round(tick * resolved_ticks_per_beat / source_midi.ticks_per_beat)),
                int(numerator),
                int(denominator),
            )
            for tick, numerator, denominator in source_time_signatures
        )
        regions = (
            {
                "start": 0.0,
                "end": float(duration_seconds),
                "meter": None,
                "bar_count": None,
                "source": "original_tempo_fallback",
                "score": None,
            },
        )
        used_predicted_tempo = False
    else:
        tempo_events, signature_events, regions = predicted_map
        used_predicted_tempo = True

    output_converter = _TempoConverter.from_events(
        ticks_per_beat=resolved_ticks_per_beat,
        tempo_events=tempo_events,
    )
    output_midi = mido.MidiFile(type=1, ticks_per_beat=resolved_ticks_per_beat)

    conductor = mido.MidiTrack()
    conductor_events: list[tuple[int, int, mido.MetaMessage]] = [
        (0, 0, mido.MetaMessage("track_name", name="Predicted Tempo Map"))
    ]
    for index, (tick, tempo) in enumerate(tempo_events):
        conductor_events.append(
            (int(tick), 10 + index, mido.MetaMessage("set_tempo", tempo=int(tempo)))
        )
    for index, (tick, numerator, denominator) in enumerate(signature_events):
        conductor_events.append(
            (
                int(tick),
                1_000_000 + index,
                mido.MetaMessage(
                    "time_signature",
                    numerator=int(numerator),
                    denominator=int(denominator),
                ),
            )
        )
    for index, segment in enumerate(predicted_key_segments):
        key_name = str(segment.get("key", "N"))
        start_seconds = max(0.0, float(segment.get("start", 0.0)))
        conductor_events.append(
            (
                output_converter.seconds_to_tick(start_seconds),
                2_000_000 + index,
                mido.MetaMessage("key_signature", key=key_name),
            )
        )
    _append_timed_messages(track=conductor, events=conductor_events)
    output_midi.tracks.append(conductor)

    chord_track = mido.MidiTrack()
    chord_events: list[tuple[int, int, mido.MetaMessage]] = [
        (0, 0, mido.MetaMessage("track_name", name="Predicted Chords"))
    ]
    for index, segment in enumerate(chord_segments):
        chord = str(segment.get("combined_label") or segment.get("chord", "N"))
        start_seconds = max(0.0, float(segment.get("start", 0.0)))
        chord_events.append(
            (
                output_converter.seconds_to_tick(start_seconds),
                index + 1,
                mido.MetaMessage("marker", text=chord),
            )
        )
    _append_timed_messages(track=chord_track, events=chord_events)
    output_midi.tracks.append(chord_track)

    max_event_drift = 0.0
    max_note_drift = 0.0
    retimed_event_count = 0
    for source_track in source_tracks:
        output_track = mido.MidiTrack()
        output_events: list[tuple[int, int, mido.Message | mido.MetaMessage]] = []
        for event in source_track:
            tick = output_converter.seconds_to_tick(event.seconds)
            drift = abs(output_converter.tick_to_seconds(tick) - event.seconds)
            max_event_drift = max(max_event_drift, drift)
            if event.message.type in {"note_on", "note_off"}:
                max_note_drift = max(max_note_drift, drift)
            output_events.append((tick, event.order, event.message))
            retimed_event_count += 1
        _append_timed_messages(track=output_track, events=output_events)
        output_midi.tracks.append(output_track)

    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_midi_path.stem}.",
        suffix=output_midi_path.suffix,
        dir=output_midi_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        output_midi.save(str(temporary_path))
        max_note_drift = _verify_reloaded_note_timing(
            output_midi_path=temporary_path,
            source_tracks=source_tracks,
        )
        if max_note_drift > float(target_max_drift_seconds) * 1.05:
            raise RuntimeError(
                "tempo-mapped MIDI note timing drift exceeded the requested limit: "
                f"{max_note_drift:.9f}s > {target_max_drift_seconds:.9f}s"
            )
        temporary_path.replace(output_midi_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return TempoMappedMidiExportResult(
        output_path=output_midi_path,
        ticks_per_beat=resolved_ticks_per_beat,
        used_predicted_tempo=bool(used_predicted_tempo),
        tempo_event_count=len(tempo_events),
        time_signature_count=len(signature_events),
        chord_marker_count=len(chord_segments),
        key_signature_count=len(predicted_key_segments),
        retimed_event_count=int(retimed_event_count),
        max_event_drift_seconds=float(max_event_drift),
        max_note_drift_seconds=float(max_note_drift),
        regions=tuple(regions),
    )
