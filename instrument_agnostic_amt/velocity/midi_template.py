from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict, deque
from dataclasses import dataclass

import mido

from .midi_events import remove_control_changes


@dataclass(slots=True)
class VelocityNoteRecord:
    start_seconds: float
    end_seconds: float
    pitch: int
    velocity: int
    program: int
    is_drum: bool
    stem_index: int
    stem_name: str
    track_index: int
    message_index: int


@dataclass(frozen=True, slots=True)
class _PendingNote:
    start_tick: int
    pitch: int
    velocity: int
    program: int
    is_drum: bool
    track_index: int
    message_index: int


@dataclass(frozen=True, slots=True)
class _TempoPoint:
    tick: int
    seconds: float
    tempo: int


def _tempo_points(midi: mido.MidiFile) -> tuple[_TempoPoint, ...]:
    events: list[tuple[int, int, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        for message_index, message in enumerate(track):
            absolute_tick += int(message.time)
            if message.type == "set_tempo":
                events.append(
                    (
                        absolute_tick,
                        track_index,
                        message_index,
                        int(message.tempo),
                    )
                )
    events.sort()

    current_tick = 0
    current_seconds = 0.0
    current_tempo = 500_000
    points = [_TempoPoint(0, 0.0, current_tempo)]
    for tick, _track_index, _message_index, tempo in events:
        current_seconds += mido.tick2second(
            tick - current_tick,
            midi.ticks_per_beat,
            current_tempo,
        )
        current_tick = tick
        current_tempo = tempo
        point = _TempoPoint(tick, current_seconds, current_tempo)
        if points[-1].tick == tick:
            points[-1] = point
        else:
            points.append(point)
    return tuple(points)


def _tick_to_seconds(
    tick: int,
    *,
    points: tuple[_TempoPoint, ...],
    ticks_per_beat: int,
) -> float:
    point_ticks = [point.tick for point in points]
    point = points[max(0, bisect_right(point_ticks, tick) - 1)]
    return point.seconds + mido.tick2second(
        tick - point.tick,
        ticks_per_beat,
        point.tempo,
    )


def note_records_from_mido(
    midi: mido.MidiFile,
    *,
    stem_name: str,
    stem_index: int,
) -> list[VelocityNoteRecord]:
    """Pair note events without losing overlaps or Note On-time programs."""

    if midi.type == 2:
        raise ValueError("MIDI type 2 is not supported for audio-aligned velocity")
    points = _tempo_points(midi)
    records: list[VelocityNoteRecord] = []
    dangling_notes: list[tuple[int, int, int]] = []

    for track_index, track in enumerate(midi.tracks):
        absolute_tick = 0
        program_by_channel: dict[int, int] = {}
        open_notes: dict[tuple[int, int], deque[_PendingNote]] = defaultdict(deque)
        for message_index, message in enumerate(track):
            absolute_tick += int(message.time)
            if message.type == "program_change":
                program_by_channel[int(message.channel)] = int(message.program)
                continue
            if message.type == "note_on" and int(message.velocity) > 0:
                channel = int(message.channel)
                pitch = int(message.note)
                open_notes[(channel, pitch)].append(
                    _PendingNote(
                        start_tick=absolute_tick,
                        pitch=pitch,
                        velocity=int(message.velocity),
                        program=int(program_by_channel.get(channel, 0)),
                        is_drum=channel == 9,
                        track_index=track_index,
                        message_index=message_index,
                    )
                )
                continue
            is_note_off = message.type == "note_off" or (
                message.type == "note_on" and int(message.velocity) == 0
            )
            if not is_note_off:
                continue
            key = (int(message.channel), int(message.note))
            if not open_notes[key]:
                continue
            pending = open_notes[key].popleft()
            if absolute_tick <= pending.start_tick:
                raise ValueError(
                    "MIDI contains a note with non-positive duration: "
                    f"track={track_index}, pitch={pending.pitch}"
                )
            records.append(
                VelocityNoteRecord(
                    start_seconds=_tick_to_seconds(
                        pending.start_tick,
                        points=points,
                        ticks_per_beat=midi.ticks_per_beat,
                    ),
                    end_seconds=_tick_to_seconds(
                        absolute_tick,
                        points=points,
                        ticks_per_beat=midi.ticks_per_beat,
                    ),
                    pitch=pending.pitch,
                    velocity=pending.velocity,
                    program=pending.program,
                    is_drum=pending.is_drum,
                    stem_index=int(stem_index),
                    stem_name=stem_name,
                    track_index=pending.track_index,
                    message_index=pending.message_index,
                )
            )

        for (channel, pitch), pending_notes in open_notes.items():
            dangling_notes.extend(
                (track_index, channel, pitch) for _pending in pending_notes
            )

    if dangling_notes:
        preview = ", ".join(
            f"track={track}, channel={channel}, pitch={pitch}"
            for track, channel, pitch in dangling_notes[:5]
        )
        raise ValueError(
            "MIDI Note On events could not be paired into complete notes. " + preview
        )
    records.sort(key=lambda record: (record.track_index, record.message_index))
    return records


def apply_velocities_to_template(
    *,
    template_midi: mido.MidiFile,
    note_records: list[VelocityNoteRecord],
    loudness_controls: str,
) -> None:
    """Update located positive Note On events, then apply the requested CC policy."""

    positive_note_on_count = sum(
        1
        for track in template_midi.tracks
        for message in track
        if message.type == "note_on" and int(message.velocity) > 0
    )
    if positive_note_on_count != len(note_records):
        raise ValueError(
            "Velocity prediction count does not match template Note On events: "
            f"note_ons={positive_note_on_count}, predictions={len(note_records)}"
        )
    used_locations: set[tuple[int, int]] = set()
    for record in note_records:
        location = (record.track_index, record.message_index)
        if location in used_locations:
            raise ValueError(f"Duplicate MIDI Note On locator: {location}")
        used_locations.add(location)
        try:
            message = template_midi.tracks[record.track_index][record.message_index]
        except IndexError as exc:
            raise ValueError(f"MIDI Note On locator is outside the template: {location}") from exc
        if (
            message.type != "note_on"
            or int(message.velocity) <= 0
            or int(message.note) != int(record.pitch)
        ):
            raise ValueError(f"MIDI Note On locator no longer matches: {location}")
        message.velocity = int(record.velocity)
    apply_loudness_controls(template_midi, loudness_controls)


def strip_loudness_controls(midi: mido.MidiFile) -> None:
    """Remove CC7/CC11 while retaining every remaining event's absolute tick."""

    for track in midi.tracks:
        track[:] = remove_control_changes(track, control_numbers=(7, 11))


def replace_fixed_loudness_controls(midi: mido.MidiFile) -> None:
    """Set CC7/CC11 to 127 before the first note on each note-bearing channel."""

    note_channels_by_track = [
        sorted(
            {
                int(message.channel)
                for message in track
                if message.type == "note_on" and int(message.velocity) > 0
            }
        )
        for track in midi.tracks
    ]
    strip_loudness_controls(midi)
    for track, channels in zip(midi.tracks, note_channels_by_track):
        if not channels:
            continue
        insert_at = 0
        for index, message in enumerate(track):
            if int(message.time) > 0 or message.type in {"note_on", "note_off"}:
                insert_at = index
                break
            insert_at = index + 1
        track[insert_at:insert_at] = [
            mido.Message(
                "control_change",
                channel=channel,
                control=control,
                value=127,
                time=0,
            )
            for channel in channels
            for control in (7, 11)
        ]


def apply_loudness_controls(midi: mido.MidiFile, policy: str) -> None:
    if policy == "preserve":
        return
    if policy == "strip":
        strip_loudness_controls(midi)
        return
    if policy == "velocity_only":
        replace_fixed_loudness_controls(midi)
        return
    raise ValueError(
        "loudness_controls must be one of: velocity_only, preserve, strip"
    )
