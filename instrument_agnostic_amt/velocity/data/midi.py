from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np
import pretty_midi


@dataclass(frozen=True)
class MidiNoteTable:
    start_seconds: np.ndarray
    end_seconds: np.ndarray
    pitch: np.ndarray
    input_velocity: np.ndarray
    program: np.ndarray
    is_drum: np.ndarray
    track_index: np.ndarray

    @property
    def note_count(self) -> int:
        return int(self.pitch.size)

    @classmethod
    def empty(cls) -> "MidiNoteTable":
        return cls(
            start_seconds=np.zeros(0, dtype=np.float64),
            end_seconds=np.zeros(0, dtype=np.float64),
            pitch=np.zeros(0, dtype=np.int16),
            input_velocity=np.zeros(0, dtype=np.int16),
            program=np.zeros(0, dtype=np.int16),
            is_drum=np.zeros(0, dtype=np.bool_),
            track_index=np.zeros(0, dtype=np.int16),
        )


def load_midi_note_table(path: str | Path | None) -> MidiNoteTable:
    """Read every positive-duration note while retaining its source track metadata."""

    if path is None:
        return MidiNoteTable.empty()
    midi = pretty_midi.PrettyMIDI(str(path))
    rows: list[tuple[float, float, int, int, int, bool, int]] = []
    for track_index, instrument in enumerate(midi.instruments):
        for note in instrument.notes:
            if float(note.end) <= float(note.start):
                continue
            rows.append(
                (
                    float(note.start),
                    float(note.end),
                    int(note.pitch),
                    int(note.velocity),
                    int(instrument.program),
                    bool(instrument.is_drum),
                    int(track_index),
                )
            )
    if not rows:
        return MidiNoteTable.empty()
    rows.sort(key=lambda row: (row[0], row[2], row[1], row[6]))
    values = np.asarray(rows, dtype=object)
    return MidiNoteTable(
        start_seconds=np.asarray(values[:, 0], dtype=np.float64),
        end_seconds=np.asarray(values[:, 1], dtype=np.float64),
        pitch=np.asarray(values[:, 2], dtype=np.int16),
        input_velocity=np.asarray(values[:, 3], dtype=np.int16),
        program=np.asarray(values[:, 4], dtype=np.int16),
        is_drum=np.asarray(values[:, 5], dtype=np.bool_),
        track_index=np.asarray(values[:, 6], dtype=np.int16),
    )


def canonicalize_amt_midi(
    source_path: str | Path,
    output_path: str | Path,
    *,
    canonical_velocity: int = 80,
    removed_control_numbers: tuple[int, ...] = (7, 11),
) -> None:
    """
    Preserve AMT note/timing errors but remove untrustworthy dynamics metadata.

    Note-On velocities are replaced by one canonical value. CC7 (Channel
    Volume) and CC11 (Expression) are removed by default; other controls such
    as sustain pedal remain available to a future renderer.
    """

    if not 1 <= int(canonical_velocity) <= 127:
        raise ValueError("canonical_velocity must be within MIDI 1..127")
    source = mido.MidiFile(str(source_path), clip=True)
    output = mido.MidiFile(
        type=source.type,
        ticks_per_beat=source.ticks_per_beat,
        charset=source.charset,
        clip=True,
    )
    removed = set(int(value) for value in removed_control_numbers)
    for source_track in source.tracks:
        target_track = mido.MidiTrack()
        carried_time = 0
        for message in source_track:
            carried_time += int(message.time)
            if message.type == "control_change" and int(message.control) in removed:
                continue
            copied = message.copy(time=carried_time)
            carried_time = 0
            if copied.type == "note_on" and int(copied.velocity) > 0:
                copied = copied.copy(velocity=int(canonical_velocity))
            target_track.append(copied)
        if carried_time:
            target_track.append(mido.MetaMessage("end_of_track", time=carried_time))
        output.tracks.append(target_track)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(str(destination))
