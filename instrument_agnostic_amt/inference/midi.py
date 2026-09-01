from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import replace

import pretty_midi

from ..taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_program_number_from_class_id,
)
from ..data.pitch_aliases import parse_pitch_aliases, resolve_pitch_alias
from .types import PredictedNote


def _midi_note_group_key(note: PredictedNote) -> tuple[int, int]:
    # Pitch slots are an internal model representation. MIDI note-off events
    # cannot identify which overlapping note instance on the same
    # channel/pitch should end, so all slots must share one export group.
    return int(note.instrument_id), int(note.pitch)


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
            int(note.start_sample),
            int(note.end_sample),
            int(note.slot_index),
        ),
    )
    by_track: dict[tuple[int, int], list[PredictedNote]] = defaultdict(list)
    for note in ordered_notes:
        track_notes = by_track.setdefault(_midi_note_group_key(note), [])
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
            int(item.start_sample),
            int(item.end_sample),
            int(item.slot_index),
        ),
    ):
        by_track[_midi_note_group_key(note)].append(note)

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


def _normalize_instrument_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _is_drum_instrument_id(instrument_id: int) -> bool:
    return _instrument_name(int(instrument_id)).lower() == "drums"


def _remap_drum_pitch_notes(
    notes: list[PredictedNote],
    aliases: Mapping[int, int] | None,
) -> tuple[list[PredictedNote], int]:
    if not notes or not aliases:
        return notes, 0

    remapped_notes: list[PredictedNote] = []
    remapped_note_count = 0
    for note in notes:
        if not _is_drum_instrument_id(int(note.instrument_id)):
            remapped_notes.append(note)
            continue
        remapped_pitch = resolve_pitch_alias(int(note.pitch), aliases)
        if remapped_pitch == int(note.pitch):
            remapped_notes.append(note)
            continue
        remapped_notes.append(replace(note, pitch=int(remapped_pitch)))
        remapped_note_count += 1
    return remapped_notes, remapped_note_count


def _midi_instrument_count(notes: list[PredictedNote]) -> int:
    return len({int(note.instrument_id) for note in notes})


def _remap_stats(
    *,
    before_count: int,
    after_count: int,
    remapped_instrument_count: int,
    remapped_note_count: int,
) -> dict[str, int]:
    return {
        "midi_instrument_count_before_remap": int(before_count),
        "midi_instrument_count_after_remap": int(after_count),
        "remapped_instrument_count": int(remapped_instrument_count),
        "remapped_note_count": int(remapped_note_count),
    }


def _remap_overflow_midi_instruments(
    notes: list[PredictedNote],
    *,
    max_melodic_instruments: int,
) -> tuple[list[PredictedNote], dict[str, int]]:
    before_count = _midi_instrument_count(notes)
    if not notes or int(max_melodic_instruments) <= 0:
        return notes, _remap_stats(
            before_count=before_count,
            after_count=before_count,
            remapped_instrument_count=0,
            remapped_note_count=0,
        )

    counts = Counter(int(note.instrument_id) for note in notes)
    melodic_ids = [
        instrument_id
        for instrument_id in counts
        if not _is_drum_instrument_id(int(instrument_id))
    ]
    if len(melodic_ids) <= int(max_melodic_instruments):
        return notes, _remap_stats(
            before_count=before_count,
            after_count=before_count,
            remapped_instrument_count=0,
            remapped_note_count=0,
        )

    kept_melodic_ids = set(
        sorted(melodic_ids, key=lambda inst_id: (-counts[int(inst_id)], int(inst_id)))[
            : int(max_melodic_instruments)
        ]
    )
    overflow_ids = set(melodic_ids) - kept_melodic_ids
    fallback_inst_id = sorted(
        kept_melodic_ids,
        key=lambda inst_id: (-counts[int(inst_id)], int(inst_id)),
    )[0]

    remapped_notes: list[PredictedNote] = []
    remapped_note_count = 0
    for note in notes:
        inst_id = int(note.instrument_id)
        if inst_id not in overflow_ids:
            remapped_notes.append(note)
            continue

        replacement_inst_id = None
        for candidate_id in note.instrument_candidates:
            candidate_id = int(candidate_id)
            if candidate_id in kept_melodic_ids:
                replacement_inst_id = candidate_id
                break
        if replacement_inst_id is None:
            replacement_inst_id = int(fallback_inst_id)

        remapped_notes.append(replace(note, instrument_id=int(replacement_inst_id)))
        remapped_note_count += 1

    remapped_notes = sorted(
        remapped_notes,
        key=lambda note: (
            int(note.start_sample),
            int(note.pitch),
            int(note.instrument_id),
            int(note.end_sample),
        ),
    )
    return remapped_notes, _remap_stats(
        before_count=before_count,
        after_count=_midi_instrument_count(remapped_notes),
        remapped_instrument_count=len(overflow_ids),
        remapped_note_count=remapped_note_count,
    )


def _build_instrument(
    instrument_id: int,
    *,
    instrument_volumes: dict[str, int] | None,
) -> pretty_midi.Instrument:
    class_name = _instrument_name(int(instrument_id))
    instrument = pretty_midi.Instrument(
        program=get_program_number_from_class_id(int(instrument_id)),
        is_drum=class_name.lower() == "drums",
        name=class_name,
    )
    if instrument_volumes is not None:
        class_key = _normalize_instrument_class_name(class_name)
        volume = instrument_volumes.get(class_key)
        if volume is not None:
            instrument.control_changes.append(
                pretty_midi.ControlChange(number=7, value=int(volume), time=0.0)
            )
    return instrument


def build_midi(
    notes: list[PredictedNote],
    *,
    sample_rate: int,
    instrument_id: int | None,
    min_midi_note_ms: float,
    max_midi_melodic_instruments: int = 0,
    instrument_volumes: dict[str, int] | None = None,
    drum_pitch_aliases: Mapping[int, int] | None = None,
    return_stats: bool = False,
) -> pretty_midi.PrettyMIDI | tuple[pretty_midi.PrettyMIDI, dict[str, int]]:
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
    normalized_drum_pitch_aliases = parse_pitch_aliases(drum_pitch_aliases)
    export_notes, remapped_drum_pitch_note_count = _remap_drum_pitch_notes(
        export_notes,
        normalized_drum_pitch_aliases,
    )
    export_notes, midi_stats = _remap_overflow_midi_instruments(
        export_notes,
        max_melodic_instruments=int(max_midi_melodic_instruments),
    )
    export_notes = _truncate_overlapping_notes(
        export_notes,
        min_separation_samples=min_separation_samples,
    )
    export_notes = _enforce_minimum_note_duration(
        export_notes,
        min_duration_samples=min_midi_note_samples,
        min_separation_samples=min_separation_samples,
    )
    midi_stats["midi_instrument_count_after_remap"] = _midi_instrument_count(
        export_notes
    )
    midi_stats["remapped_drum_pitch_note_count"] = int(
        remapped_drum_pitch_note_count
    )

    grouped_notes: dict[int, list[PredictedNote]] = defaultdict(list)
    if instrument_id is not None:
        grouped_notes[int(instrument_id)] = export_notes
    else:
        for note in export_notes:
            grouped_notes[int(note.instrument_id)].append(note)

    for current_instrument_id in sorted(grouped_notes):
        instrument = _build_instrument(
            int(current_instrument_id),
            instrument_volumes=instrument_volumes,
        )
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
    if return_stats:
        return midi, midi_stats
    return midi
