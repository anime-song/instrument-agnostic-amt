from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import pretty_midi

from ..taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_program_number_from_class_id,
)
from .types import PredictedNote


def _note_group_key(note: PredictedNote) -> tuple[int, int, int]:
    return int(note.instrument_id), int(note.pitch), int(note.slot_index)


def _truncate_overlapping_notes(
    notes: list[PredictedNote],
    *,
    min_separation_samples: int,
) -> list[PredictedNote]:
    if not notes:
        return []

    ordered_notes = sorted(
        notes,
        key=lambda note: (
            int(note.instrument_id),
            int(note.pitch),
            int(note.slot_index),
            int(note.start_sample),
            int(note.end_sample),
        ),
    )
    by_track: dict[tuple[int, int, int], list[PredictedNote]] = defaultdict(list)
    for note in ordered_notes:
        track_notes = by_track.setdefault(_note_group_key(note), [])
        if track_notes:
            previous_note = track_notes[-1]
            separation_samples = max(1, int(min_separation_samples))
            if previous_note.end_sample > note.start_sample - separation_samples:
                new_end_sample = int(note.start_sample) - separation_samples
                if (
                    new_end_sample - int(previous_note.start_sample)
                    >= separation_samples
                ):
                    track_notes[-1] = replace(
                        previous_note,
                        end_sample=new_end_sample,
                        has_offset=True,
                    )
                else:
                    track_notes.pop()
        track_notes.append(note)

    valid_notes = [
        note
        for track_notes in by_track.values()
        for note in track_notes
        if int(note.start_sample) < int(note.end_sample)
    ]
    return sorted(
        valid_notes,
        key=lambda note: (
            int(note.start_sample),
            int(note.instrument_id),
            int(note.pitch),
        ),
    )


def _enforce_minimum_note_duration(
    notes: list[PredictedNote],
    *,
    min_duration_samples: int,
    min_separation_samples: int,
) -> list[PredictedNote]:
    if not notes or min_duration_samples <= 1:
        return notes

    by_track: dict[tuple[int, int, int], list[PredictedNote]] = defaultdict(list)
    for note in sorted(
        notes,
        key=lambda item: (
            int(item.instrument_id),
            int(item.pitch),
            int(item.slot_index),
            int(item.start_sample),
        ),
    ):
        by_track[_note_group_key(note)].append(note)

    adjusted_notes: list[PredictedNote] = []
    for track_notes in by_track.values():
        for note_index, note in enumerate(track_notes):
            target_end_sample = max(
                int(note.end_sample),
                int(note.start_sample) + int(min_duration_samples),
            )
            if note_index + 1 < len(track_notes):
                next_note = track_notes[note_index + 1]
                target_end_sample = min(
                    target_end_sample,
                    int(next_note.start_sample) - int(min_separation_samples),
                )
            if target_end_sample <= int(note.start_sample):
                adjusted_notes.append(note)
                continue
            adjusted_notes.append(replace(note, end_sample=int(target_end_sample)))

    return sorted(
        adjusted_notes,
        key=lambda note: (
            int(note.start_sample),
            int(note.instrument_id),
            int(note.pitch),
        ),
    )


def _instrument_name(instrument_id: int) -> str:
    if 0 <= int(instrument_id) < len(INSTRUMENT_CLASSES):
        return INSTRUMENT_CLASSES[int(instrument_id)]
    return "Piano"


def _build_instrument(instrument_id: int) -> pretty_midi.Instrument:
    class_name = _instrument_name(int(instrument_id))
    return pretty_midi.Instrument(
        program=get_program_number_from_class_id(int(instrument_id)),
        is_drum=class_name.lower() == "drums",
        name=class_name,
    )


def build_midi(
    notes: list[PredictedNote],
    *,
    sample_rate: int,
    instrument_id: int | None,
    min_midi_note_ms: float,
) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(resolution=1920)
    min_midi_note_samples = max(
        0,
        int(round(float(min_midi_note_ms) * float(sample_rate) / 1000.0)),
    )
    min_separation_samples = max(1, int(round(float(sample_rate) * 0.005)))

    export_notes = notes
    if instrument_id is not None:
        export_notes = [
            note
            for note in export_notes
            if int(note.instrument_id) == int(instrument_id)
        ]
    export_notes = _truncate_overlapping_notes(
        export_notes,
        min_separation_samples=min_separation_samples,
    )
    export_notes = _enforce_minimum_note_duration(
        export_notes,
        min_duration_samples=min_midi_note_samples,
        min_separation_samples=min_separation_samples,
    )

    grouped_notes: dict[int, list[PredictedNote]] = defaultdict(list)
    if instrument_id is not None:
        grouped_notes[int(instrument_id)] = export_notes
    else:
        for note in export_notes:
            grouped_notes[int(note.instrument_id)].append(note)

    for current_instrument_id in sorted(grouped_notes):
        instrument = _build_instrument(int(current_instrument_id))
        for note in sorted(
            grouped_notes[current_instrument_id],
            key=lambda item: (
                int(item.start_sample),
                int(item.pitch),
                int(item.end_sample),
            ),
        ):
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
