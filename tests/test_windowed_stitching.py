from instrument_agnostic_amt.amt.inference.types import PredictedNote
from instrument_agnostic_amt.amt.inference.windowed import WindowNoteStitcher


def _note(
    start: int,
    end: int,
    *,
    has_onset: bool,
    has_offset: bool,
) -> PredictedNote:
    return PredictedNote(
        instrument_id=0,
        pitch=41,
        start_sample=start,
        end_sample=end,
        velocity=100,
        slot_index=0,
        has_onset=has_onset,
        has_offset=has_offset,
        instrument_candidates=(21, 11, 23),
    )


def test_finalize_merges_window_continuations_before_offset_filtering() -> None:
    stitcher = WindowNoteStitcher(
        hop_length=512,
        total_audio_frames=10_000,
        velocity=100,
        merge_gap_samples=512,
        merge_onset_samples=1_102,
    )
    # Append order follows overlapping inference windows, so the second window
    # can report a slightly earlier onset than the first one.
    stitcher.notes_by_pair[20] = [
        _note(2_100, 4_000, has_onset=True, has_offset=False),
        _note(2_000, 6_000, has_onset=True, has_offset=False),
        _note(4_001, 8_000, has_onset=False, has_offset=False),
        _note(4_500, 9_000, has_onset=False, has_offset=False),
    ]

    notes = stitcher.finalize()

    assert len(notes) == 1
    assert notes[0].start_sample == 2_000
    assert notes[0].end_sample == 9_000
    assert notes[0].has_onset is True
    assert notes[0].has_offset is True


def test_finalize_keeps_distinct_repeated_onsets_separate() -> None:
    stitcher = WindowNoteStitcher(
        hop_length=512,
        total_audio_frames=10_000,
        velocity=100,
        merge_gap_samples=512,
        merge_onset_samples=100,
    )
    stitcher.notes_by_pair[20] = [
        _note(1_000, 2_000, has_onset=True, has_offset=True),
        _note(3_000, 4_000, has_onset=True, has_offset=True),
    ]

    notes = stitcher.finalize()

    assert [(note.start_sample, note.end_sample) for note in notes] == [
        (1_000, 2_000),
        (3_000, 4_000),
    ]
