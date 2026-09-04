from __future__ import annotations

from pathlib import Path

import pretty_midi
import pytest

from instrument_agnostic_amt.amt.inference.midi import build_midi
from instrument_agnostic_amt.amt.inference.types import PredictedNote


def test_build_midi_resolves_same_pitch_overlaps_across_slots(
    tmp_path: Path,
) -> None:
    notes = [
        PredictedNote(
            instrument_id=0,
            pitch=60,
            start_sample=0,
            end_sample=1_500,
            velocity=100,
            slot_index=0,
        ),
        PredictedNote(
            instrument_id=0,
            pitch=60,
            start_sample=500,
            end_sample=1_000,
            velocity=90,
            slot_index=1,
        ),
    ]

    midi = build_midi(
        notes,
        sample_rate=1_000,
        instrument_id=None,
        min_midi_note_ms=5.0,
    )
    instrument = midi.instruments[0]
    before_roundtrip = [
        (note.start, note.end, note.pitch) for note in instrument.notes
    ]

    assert before_roundtrip == pytest.approx(
        [
            (0.0, 0.495, 60),
            (0.5, 1.0, 60),
        ],
        abs=1e-9,
    )
    assert instrument.notes[0].end < instrument.notes[1].start

    midi_path = tmp_path / "pitch_slot_overlap.mid"
    midi.write(str(midi_path))
    reloaded = pretty_midi.PrettyMIDI(str(midi_path))
    after_roundtrip = [
        (note.start, note.end, note.pitch)
        for note in reloaded.instruments[0].notes
    ]

    assert len(after_roundtrip) == len(before_roundtrip)
    for actual, expected in zip(after_roundtrip, before_roundtrip):
        assert actual[:2] == pytest.approx(expected[:2], abs=0.001)
        assert actual[2] == expected[2]
    assert reloaded.instruments[0].notes[0].end < (
        reloaded.instruments[0].notes[1].start
    )
