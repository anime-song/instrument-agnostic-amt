"""SoundFont calibration MIDI generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pretty_midi


@dataclass(frozen=True)
class SweepEvent:
    program: int
    is_drum: bool
    pitch: int
    velocity: int
    start_seconds: float
    end_seconds: float


def build_velocity_sweep_midi(
    *,
    program: int,
    is_drum: bool = False,
    pitches: tuple[int, ...],
    velocities: tuple[int, ...],
    note_seconds: float,
    gap_seconds: float,
    lead_in_seconds: float = 0.25,
) -> tuple[pretty_midi.PrettyMIDI, list[SweepEvent]]:
    """Create a deterministic GM-program sweep for one target SoundFont."""

    if not 0 <= int(program) <= 127:
        raise ValueError("program must be within General MIDI 0..127")
    if note_seconds <= 0.0 or gap_seconds < 0.0 or lead_in_seconds < 0.0:
        raise ValueError("invalid sweep timing")
    if not pitches or any(pitch < 0 or pitch > 127 for pitch in pitches):
        raise ValueError("pitches must contain MIDI values 0..127")
    if not velocities or any(velocity < 1 or velocity > 127 for velocity in velocities):
        raise ValueError("velocities must contain MIDI values 1..127")

    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0, resolution=960)
    instrument = pretty_midi.Instrument(
        program=int(program),
        is_drum=bool(is_drum),
        name=("GM drums" if is_drum else f"GM program {int(program):03d}"),
    )
    instrument.control_changes.extend(
        [
            pretty_midi.ControlChange(number=7, value=127, time=0.0),
            pretty_midi.ControlChange(number=11, value=127, time=0.0),
        ]
    )
    events: list[SweepEvent] = []
    cursor = float(lead_in_seconds)
    for pitch in pitches:
        for velocity in velocities:
            end = cursor + float(note_seconds)
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(pitch),
                    start=cursor,
                    end=end,
                )
            )
            events.append(
                SweepEvent(
                    program=int(program),
                    is_drum=bool(is_drum),
                    pitch=int(pitch),
                    velocity=int(velocity),
                    start_seconds=cursor,
                    end_seconds=end,
                )
            )
            cursor = end + float(gap_seconds)
    midi.instruments.append(instrument)
    return midi, events


def write_velocity_sweep_midi(
    output_path: str | Path,
    **kwargs: object,
) -> list[SweepEvent]:
    midi, events = build_velocity_sweep_midi(**kwargs)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(destination))
    return events
