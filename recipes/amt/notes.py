from __future__ import annotations

import numpy as np


class WindowNotes:
    """Window-local note arrays used by target builders."""

    def __init__(
        self,
        start_ms: np.ndarray,
        end_ms: np.ndarray,
        pitch: np.ndarray,
        velocity: np.ndarray,
        has_onset: np.ndarray,
        has_offset: np.ndarray,
        instrument: np.ndarray | None = None,
    ):
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.pitch = pitch
        self.velocity = velocity
        # Whether each note starts/ends inside the selected window.
        self.has_onset = has_onset
        self.has_offset = has_offset
        if instrument is None:
            self.instrument = np.zeros_like(pitch)
        else:
            self.instrument = instrument

    @classmethod
    def empty(cls) -> "WindowNotes":
        return cls(
            start_ms=np.zeros((0,), dtype=np.int64),
            end_ms=np.zeros((0,), dtype=np.int64),
            pitch=np.zeros((0,), dtype=np.int16),
            velocity=np.zeros((0,), dtype=np.int16),
            has_onset=np.zeros((0,), dtype=np.bool_),
            has_offset=np.zeros((0,), dtype=np.bool_),
            instrument=np.zeros((0,), dtype=np.int16),
        )


def split_window_notes(
    *,
    start_ms: np.ndarray,
    end_ms: np.ndarray,
    pitch: np.ndarray,
    velocity: np.ndarray,
    instrument: np.ndarray,
    window_start_ms: int,
    window_end_ms: int,
    clip_note_end_to_window: bool = True,
) -> tuple[WindowNotes, WindowNotes]:
    """
    Select notes that intersect a window.
    Returns carry-in notes that started before the window and body notes that
    start inside the window.
    """
    max_window_length_ms = int(window_end_ms) - int(window_start_ms)

    def select(note_mask: np.ndarray, *, start_at_zero: bool) -> WindowNotes:
        if not np.any(note_mask):
            return WindowNotes.empty()

        # Convert note times to offsets relative to the window start.
        rel_start = (
            np.zeros(int(note_mask.sum()), dtype=np.int64)
            if start_at_zero
            else start_ms[note_mask] - window_start_ms
        )
        rel_end = end_ms[note_mask] - window_start_ms

        if clip_note_end_to_window:
            rel_end = np.minimum(rel_end, max_window_length_ms)
        # Keep every clipped note at least 1 ms long.
        rel_end = np.maximum(rel_end, rel_start + 1)

        # Notes extending past the window do not have an offset inside it.
        tie_to_next_window = (end_ms[note_mask] > window_end_ms).astype(
            np.bool_, copy=False
        )

        return WindowNotes(
            start_ms=rel_start.astype(np.int64, copy=False),
            end_ms=rel_end.astype(np.int64, copy=False),
            pitch=pitch[note_mask].astype(np.int16, copy=False),
            velocity=velocity[note_mask].astype(np.int16, copy=False),
            instrument=instrument[note_mask].astype(np.int16, copy=False),
            has_onset=np.full(
                int(note_mask.sum()), fill_value=(not start_at_zero), dtype=np.bool_
            ),
            has_offset=np.logical_not(tie_to_next_window).astype(np.bool_, copy=False),
        )

    # carry_in: notes already active at the window start.
    carry_in_mask = (start_ms < window_start_ms) & (end_ms > window_start_ms)
    # body: notes whose onset is inside the window.
    body_mask = (start_ms >= window_start_ms) & (start_ms < window_end_ms)

    return select(carry_in_mask, start_at_zero=True), select(
        body_mask, start_at_zero=False
    )


def concat_window_notes(*note_groups: WindowNotes) -> WindowNotes:
    """Concatenate multiple WindowNotes objects."""
    non_empty_groups = [group for group in note_groups if group.start_ms.size > 0]
    if not non_empty_groups:
        return WindowNotes.empty()

    return WindowNotes(
        start_ms=np.concatenate([group.start_ms for group in non_empty_groups]).astype(
            np.int64, copy=False
        ),
        end_ms=np.concatenate([group.end_ms for group in non_empty_groups]).astype(
            np.int64, copy=False
        ),
        pitch=np.concatenate([group.pitch for group in non_empty_groups]).astype(
            np.int16, copy=False
        ),
        velocity=np.concatenate([group.velocity for group in non_empty_groups]).astype(
            np.int16, copy=False
        ),
        instrument=np.concatenate(
            [group.instrument for group in non_empty_groups]
        ).astype(np.int16, copy=False),
        has_onset=np.concatenate(
            [group.has_onset for group in non_empty_groups]
        ).astype(np.bool_, copy=False),
        has_offset=np.concatenate(
            [group.has_offset for group in non_empty_groups]
        ).astype(np.bool_, copy=False),
    )
