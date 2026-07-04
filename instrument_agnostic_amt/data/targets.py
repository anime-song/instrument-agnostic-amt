from __future__ import annotations

import numpy as np
import torch

from ..modeling.heads.interval_boundaries import PitchIntervalTargets
from ..taxonomy.instrument_classes import NUM_INSTRUMENT_CLASSES
from .constants import MAX_MIDI_PITCH, MIN_MIDI_PITCH, NUM_PITCHES


def _ms_to_sample_index(ms: np.ndarray, *, sample_rate: int) -> np.ndarray:
    """Convert milliseconds to sample indices."""
    return np.rint(
        ms.astype(np.float64, copy=False) * float(sample_rate) / 1000.0
    ).astype(np.int64, copy=False)


def _valid_model_pitch_mask(pitch: np.ndarray) -> np.ndarray:
    """Mask pitches inside the supported MIDI pitch range."""
    pitch_i64 = pitch.astype(np.int64, copy=False)
    return (pitch_i64 >= MIN_MIDI_PITCH) & (pitch_i64 <= MAX_MIDI_PITCH)


def _map_model_pitch_array(pitch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map MIDI pitches 21..108 to model indices 0..87."""
    valid_mask = _valid_model_pitch_mask(pitch)
    mapped_pitch = pitch.astype(np.int64, copy=False)[valid_mask] - MIN_MIDI_PITCH
    return mapped_pitch.astype(np.int64, copy=False), valid_mask


def build_frame_note_targets(
    *,
    active_start_ms: np.ndarray,
    active_end_ms: np.ndarray,
    active_pitch: np.ndarray,
    sample_rate: int,
    hop_length: int,
    num_frames: int,
) -> torch.Tensor:
    """Build frame-level pitch activation targets with shape [num_frames, 88]."""
    active_targets = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
    if num_frames <= 0 or active_start_ms.size == 0:
        return torch.from_numpy(active_targets)

    start_samples = _ms_to_sample_index(active_start_ms, sample_rate=sample_rate)
    end_samples = _ms_to_sample_index(active_end_ms, sample_rate=sample_rate)

    start_frames = np.clip(start_samples // int(hop_length), 0, num_frames - 1)
    end_frames = (np.maximum(end_samples - 1, 0) // int(hop_length)) + 1
    end_frames = np.clip(np.maximum(end_frames, start_frames + 1), 0, num_frames)

    active_pitches, valid_pitch_mask = _map_model_pitch_array(active_pitch)
    if not np.any(valid_pitch_mask):
        return torch.from_numpy(active_targets)

    start_frames = start_frames[valid_pitch_mask]
    end_frames = end_frames[valid_pitch_mask]

    for start_frame, end_frame, pitch_value in zip(
        start_frames.tolist(), end_frames.tolist(), active_pitches.tolist()
    ):
        if start_frame >= num_frames:
            continue
        active_targets[start_frame:end_frame, pitch_value] = 1.0

    return torch.from_numpy(active_targets)


def build_pitch_interval_targets(
    *,
    active_start_ms: np.ndarray,
    active_end_ms: np.ndarray,
    active_pitch: np.ndarray,
    active_instrument: np.ndarray,
    active_has_onset: np.ndarray,
    active_has_offset: np.ndarray,
    sample_rate: int,
    hop_length: int,
    num_frames: int,
    num_pitch_slots: int = 1,
) -> PitchIntervalTargets:
    """Build per-pitch Semi-CRF interval targets."""
    num_pitch_slots = max(1, int(num_pitch_slots))
    num_tracks = NUM_PITCHES * num_pitch_slots
    pitch_intervals: list[list[tuple[int, int]]] = [[] for _ in range(num_tracks)]
    has_onset_tracks: list[list[bool]] = [[] for _ in range(num_tracks)]
    has_offset_tracks: list[list[bool]] = [[] for _ in range(num_tracks)]
    onset_offsets_tracks: list[list[float]] = [[] for _ in range(num_tracks)]
    offset_offsets_tracks: list[list[float]] = [[] for _ in range(num_tracks)]
    instrument_sets_tracks: list[list[tuple[int, ...]]] = [
        [] for _ in range(num_tracks)
    ]

    if num_frames <= 0 or active_start_ms.size == 0:
        return PitchIntervalTargets(
            intervals=pitch_intervals,
            has_onset=has_onset_tracks,
            has_offset=has_offset_tracks,
            onset_offsets=onset_offsets_tracks,
            offset_offsets=offset_offsets_tracks,
            instrument_sets=instrument_sets_tracks,
        )

    # Keep frame-boundary math explicit so interval endpoints and overlap handling
    # stay consistent with the Semi-CRF loss and boundary head.
    start_samples = _ms_to_sample_index(active_start_ms, sample_rate=sample_rate)
    end_samples = _ms_to_sample_index(active_end_ms, sample_rate=sample_rate)

    real_start_frames = start_samples.astype(np.float64, copy=False) / float(hop_length)
    real_end_frames = end_samples.astype(np.float64, copy=False) / float(hop_length)

    raw_start_frames = start_samples // int(hop_length)
    raw_end_frames_exclusive = (np.maximum(end_samples - 1, 0) // int(hop_length)) + 1
    raw_end_frames_inclusive = raw_end_frames_exclusive - 1

    start_frames = np.clip(raw_start_frames, 0, num_frames - 1)
    end_frames_exclusive = np.clip(
        np.maximum(raw_end_frames_exclusive, start_frames + 1), 0, num_frames
    )

    # Fractional offsets are used by the boundary timing correction loss.
    onset_offsets = real_start_frames - raw_start_frames
    offset_offsets = real_end_frames - raw_end_frames_inclusive
    mapped_pitch, valid_pitch_mask = _map_model_pitch_array(active_pitch)

    if not np.any(valid_pitch_mask):
        return PitchIntervalTargets(
            intervals=pitch_intervals,
            has_onset=has_onset_tracks,
            has_offset=has_offset_tracks,
            onset_offsets=onset_offsets_tracks,
            offset_offsets=offset_offsets_tracks,
            instrument_sets=instrument_sets_tracks,
        )

    start_frames = start_frames[valid_pitch_mask]
    end_frames_exclusive = end_frames_exclusive[valid_pitch_mask]
    active_instrument = active_instrument[valid_pitch_mask]
    active_has_onset = active_has_onset[valid_pitch_mask]
    active_has_offset = active_has_offset[valid_pitch_mask]
    onset_offsets = onset_offsets[valid_pitch_mask]
    offset_offsets = offset_offsets[valid_pitch_mask]

    raw_by_pitch: list[list[tuple[int, int, int, bool, bool, float, float]]] = [
        [] for _ in range(NUM_PITCHES)
    ]
    for (
        start_frame,
        end_frame_exclusive,
        pitch_value,
        instrument_id,
        has_onset,
        has_offset,
        onset_off,
        offset_off,
    ) in zip(
        start_frames.tolist(),
        end_frames_exclusive.tolist(),
        mapped_pitch.tolist(),
        active_instrument.tolist(),
        active_has_onset.tolist(),
        active_has_offset.tolist(),
        onset_offsets.tolist(),
        offset_offsets.tolist(),
    ):
        if start_frame >= num_frames or end_frame_exclusive <= start_frame:
            continue
        raw_by_pitch[pitch_value].append(
            (
                int(start_frame),
                int(end_frame_exclusive - 1),
                int(instrument_id),
                bool(has_onset),
                bool(has_offset),
                float(onset_off),
                float(offset_off),
            )
        )

    if num_pitch_slots > 1:
        for pitch_value, intervals in enumerate(raw_by_pitch):
            if not intervals:
                continue
            intervals.sort(
                key=lambda item: (item[0], item[1], item[3], item[4], item[2])
            )
            slot_last_end_frames = [-1] * num_pitch_slots
            for (
                begin,
                end,
                instrument_id,
                has_onset,
                has_offset,
                onset_off,
                offset_off,
            ) in intervals:
                if int(begin) > int(end):
                    continue

                assigned_slot = None
                for slot_index, last_end in enumerate(slot_last_end_frames):
                    if int(begin) > int(last_end):
                        assigned_slot = slot_index
                        break
                if assigned_slot is None:
                    continue

                track_index = int(pitch_value) * num_pitch_slots + int(assigned_slot)
                pitch_intervals[track_index].append((int(begin), int(end)))
                has_onset_tracks[track_index].append(bool(has_onset))
                has_offset_tracks[track_index].append(bool(has_offset))
                onset_offsets_tracks[track_index].append(float(onset_off))
                offset_offsets_tracks[track_index].append(float(offset_off))
                if 0 <= int(instrument_id) < NUM_INSTRUMENT_CLASSES:
                    instrument_sets_tracks[track_index].append((int(instrument_id),))
                else:
                    instrument_sets_tracks[track_index].append(())
                slot_last_end_frames[int(assigned_slot)] = int(end)

        return PitchIntervalTargets(
            intervals=pitch_intervals,
            has_onset=has_onset_tracks,
            has_offset=has_offset_tracks,
            onset_offsets=onset_offsets_tracks,
            offset_offsets=offset_offsets_tracks,
            instrument_sets=instrument_sets_tracks,
        )

    # Sort intervals per pitch and merge overlaps into non-overlapping targets.
    for pitch_value, intervals in enumerate(raw_by_pitch):
        if not intervals:
            continue
        intervals.sort(key=lambda item: (item[0], item[1], item[3], item[4], item[2]))
        sanitized: list[list[int | bool | float]] = []
        for begin, end, _, has_onset, has_offset, onset_off, offset_off in intervals:
            # Trim or merge overlapping intervals before adding the next one.
            if sanitized and begin <= sanitized[-1][1]:
                prev_begin = int(sanitized[-1][0])
                if begin > prev_begin:
                    sanitized[-1][1] = begin - 1
                    sanitized[-1][3] = True
                    sanitized[-1][5] = 0.5
                else:
                    sanitized.pop()
            if sanitized and begin <= sanitized[-1][1]:
                begin = sanitized[-1][1] + 1
                onset_off = 0.5
            if begin > end:
                continue
            sanitized.append(
                [
                    begin,
                    end,
                    bool(has_onset),
                    bool(has_offset),
                    float(onset_off),
                    float(offset_off),
                ]
            )

        for begin, end, has_onset, has_offset, onset_off, offset_off in sanitized:
            if int(begin) > int(end):
                continue
            instrument_ids = sorted(
                {
                    int(instrument_id)
                    for raw_begin, raw_end, instrument_id, *_ in intervals
                    if not (int(raw_end) < int(begin) or int(raw_begin) > int(end))
                    and 0 <= int(instrument_id) < NUM_INSTRUMENT_CLASSES
                }
            )
            pitch_intervals[pitch_value].append((int(begin), int(end)))
            has_onset_tracks[pitch_value].append(bool(has_onset))
            has_offset_tracks[pitch_value].append(bool(has_offset))
            onset_offsets_tracks[pitch_value].append(float(onset_off))
            offset_offsets_tracks[pitch_value].append(float(offset_off))
            instrument_sets_tracks[pitch_value].append(tuple(instrument_ids))

    return PitchIntervalTargets(
        intervals=pitch_intervals,
        has_onset=has_onset_tracks,
        has_offset=has_offset_tracks,
        onset_offsets=onset_offsets_tracks,
        offset_offsets=offset_offsets_tracks,
        instrument_sets=instrument_sets_tracks,
    )
