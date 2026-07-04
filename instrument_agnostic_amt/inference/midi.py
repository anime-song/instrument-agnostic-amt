from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import pretty_midi

from ..taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_program_number_from_class_id,
)
from .types import PredictedNote


def _truncate_overlapping_notes(
    notes: list[PredictedNote],
    *,
    min_separation_samples: int,
) -> list[PredictedNote]:
    if not notes:
        return []

    ordered_notes = sorted(
        notes,
        key=lambda note: (note.pitch, note.start_sample, note.end_sample),
    )
    by_pitch: dict[int, list[PredictedNote]] = defaultdict(list)
    for note in ordered_notes:
        pitch_notes = by_pitch.setdefault(int(note.pitch), [])
        if pitch_notes:
            previous_note = pitch_notes[-1]
            separation_samples = max(1, int(min_separation_samples))
            if previous_note.end_sample > note.start_sample - separation_samples:
                new_end_sample = int(note.start_sample) - separation_samples
                if (
                    new_end_sample - int(previous_note.start_sample)
                    >= separation_samples
                ):
                    pitch_notes[-1] = replace(
                        previous_note,
                        end_sample=new_end_sample,
                        has_offset=True,
                    )
                else:
                    pitch_notes.pop()
        pitch_notes.append(note)

    valid_notes = [
        note
        for pitch_notes in by_pitch.values()
        for note in pitch_notes
        if int(note.start_sample) < int(note.end_sample)
    ]
    return sorted(valid_notes, key=lambda note: (note.start_sample, note.pitch))


def _enforce_minimum_note_duration(
    notes: list[PredictedNote],
    *,
    min_duration_samples: int,
    min_separation_samples: int,
) -> list[PredictedNote]:
    if not notes or min_duration_samples <= 1:
        return notes

    by_pitch: dict[int, list[PredictedNote]] = defaultdict(list)
    for note in sorted(notes, key=lambda item: (item.pitch, item.start_sample)):
        by_pitch[int(note.pitch)].append(note)

    adjusted_notes: list[PredictedNote] = []
    for pitch_notes in by_pitch.values():
        for note_index, note in enumerate(pitch_notes):
            target_end_sample = max(
                int(note.end_sample),
                int(note.start_sample) + int(min_duration_samples),
            )
            if note_index + 1 < len(pitch_notes):
                next_note = pitch_notes[note_index + 1]
                target_end_sample = min(
                    target_end_sample,
                    int(next_note.start_sample) - int(min_separation_samples),
                )
            if target_end_sample <= int(note.start_sample):
                adjusted_notes.append(note)
                continue
            adjusted_notes.append(replace(note, end_sample=int(target_end_sample)))

    return sorted(adjusted_notes, key=lambda note: (note.start_sample, note.pitch))


def build_midi(
    notes: list[PredictedNote],
    *,
    sample_rate: int,
    instrument_id: int,
    min_midi_note_ms: float,
) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(resolution=1920)
    class_name = (
        INSTRUMENT_CLASSES[instrument_id]
        if 0 <= instrument_id < len(INSTRUMENT_CLASSES)
        else "Piano"
    )
    instrument = pretty_midi.Instrument(
        program=get_program_number_from_class_id(instrument_id),
        is_drum=class_name.lower() == "drums",
        name=class_name,
    )
    min_midi_note_samples = max(
        0,
        int(round(float(min_midi_note_ms) * float(sample_rate) / 1000.0)),
    )
    min_separation_samples = max(1, int(round(float(sample_rate) * 0.005)))
    export_notes = _truncate_overlapping_notes(
        notes,
        min_separation_samples=min_separation_samples,
    )
    export_notes = _enforce_minimum_note_duration(
        export_notes,
        min_duration_samples=min_midi_note_samples,
        min_separation_samples=min_separation_samples,
    )
    for note in export_notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=float(note.start_sample) / float(sample_rate),
                end=float(note.end_sample) / float(sample_rate),
            )
        )
    midi.instruments.append(instrument)
    return midi
