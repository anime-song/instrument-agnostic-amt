"""MIDI generation for synthetic velocity training data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi


def _sorted_note_references(
    midi: pretty_midi.PrettyMIDI,
) -> list[tuple[int, pretty_midi.Note]]:
    rows: list[tuple[float, int, float, int, int, pretty_midi.Note]] = []
    for track_index, instrument in enumerate(midi.instruments):
        for note_index, note in enumerate(instrument.notes):
            if float(note.end) <= float(note.start):
                continue
            rows.append(
                (
                    float(note.start),
                    int(note.pitch),
                    float(note.end),
                    int(track_index),
                    int(note_index),
                    note,
                )
            )
    rows.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
    return [(int(row[3]), row[5]) for row in rows]


def write_target_velocity_midi(
    source_path: str | Path,
    output_path: str | Path,
    target_velocity: np.ndarray,
    *,
    render_volume: int = 127,
    render_expression: int = 127,
) -> None:
    """Write a render-only MIDI whose note velocities exactly match the labels."""

    if not 0 <= render_volume <= 127 or not 0 <= render_expression <= 127:
        raise ValueError("render controllers must be within MIDI 0..127")
    velocities = np.asarray(target_velocity, dtype=np.int64)
    if velocities.ndim != 1 or np.any((velocities < 1) | (velocities > 127)):
        raise ValueError("target_velocity must contain MIDI velocities 1..127")
    midi = pretty_midi.PrettyMIDI(str(source_path))
    references = _sorted_note_references(midi)
    if len(references) != velocities.size:
        raise ValueError(
            f"MIDI/label note count mismatch: {len(references)} != {velocities.size}"
        )
    for (_, note), velocity in zip(references, velocities):
        note.velocity = int(velocity)
    for instrument in midi.instruments:
        instrument.control_changes = [
            control
            for control in instrument.control_changes
            if int(control.number) not in (7, 11)
        ]
        instrument.control_changes.extend(
            [
                pretty_midi.ControlChange(number=7, value=render_volume, time=0.0),
                pretty_midi.ControlChange(
                    number=11,
                    value=render_expression,
                    time=0.0,
                ),
            ]
        )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(destination))
