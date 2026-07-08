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


def _empty_pair_interval_targets() -> PitchIntervalTargets:
    return PitchIntervalTargets(
        intervals=[],
        has_onset=[],
        has_offset=[],
        onset_offsets=[],
        offset_offsets=[],
        instrument_sets=[],
        positive_pair_ids=[],
        pair_presence=torch.zeros(
            (NUM_INSTRUMENT_CLASSES, NUM_PITCHES), dtype=torch.bool
        ),
    )


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
) -> PitchIntervalTargets:
    """Build sparse instrument-pitch Semi-CRF interval targets."""
    if num_frames <= 0 or active_start_ms.size == 0:
        return _empty_pair_interval_targets()

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

    onset_offsets = real_start_frames - raw_start_frames
    offset_offsets = real_end_frames - raw_end_frames_inclusive
    mapped_pitch, valid_pitch_mask = _map_model_pitch_array(active_pitch)
    valid_instrument_mask = (active_instrument.astype(np.int64, copy=False) >= 0) & (
        active_instrument.astype(np.int64, copy=False) < NUM_INSTRUMENT_CLASSES
    )
    valid_mask = valid_pitch_mask.copy()
    valid_mask[valid_pitch_mask] &= valid_instrument_mask[valid_pitch_mask]

    if not np.any(valid_mask):
        return _empty_pair_interval_targets()

    start_frames = start_frames[valid_mask]
    end_frames_exclusive = end_frames_exclusive[valid_mask]
    active_instrument = active_instrument[valid_mask].astype(np.int64, copy=False)
    active_has_onset = active_has_onset[valid_mask]
    active_has_offset = active_has_offset[valid_mask]
    onset_offsets = onset_offsets[valid_mask]
    offset_offsets = offset_offsets[valid_mask]
    mapped_pitch, _ = _map_model_pitch_array(active_pitch[valid_mask])

    raw_by_pair: dict[int, list[tuple[int, int, bool, bool, float, float]]] = {}
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
        pair_id = int(instrument_id) * NUM_PITCHES + int(pitch_value)
        raw_by_pair.setdefault(pair_id, []).append(
            (
                int(start_frame),
                int(end_frame_exclusive - 1),
                bool(has_onset),
                bool(has_offset),
                float(onset_off),
                float(offset_off),
            )
        )

    pair_presence = torch.zeros((NUM_INSTRUMENT_CLASSES, NUM_PITCHES), dtype=torch.bool)
    positive_pair_ids: list[int] = []
    pair_intervals: list[list[tuple[int, int]]] = []
    has_onset_tracks: list[list[bool]] = []
    has_offset_tracks: list[list[bool]] = []
    onset_offsets_tracks: list[list[float]] = []
    offset_offsets_tracks: list[list[float]] = []
    instrument_sets_tracks: list[list[tuple[int, ...]]] = []

    for pair_id in sorted(raw_by_pair):
        intervals = raw_by_pair[pair_id]
        intervals.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        sanitized: list[list[int | bool | float]] = []
        for begin, end, has_onset, has_offset, onset_off, offset_off in intervals:
            if sanitized and begin <= int(sanitized[-1][1]):
                previous_begin = int(sanitized[-1][0])
                if begin > previous_begin:
                    sanitized[-1][1] = begin - 1
                    sanitized[-1][3] = True
                    sanitized[-1][5] = 0.5
                else:
                    sanitized.pop()
            if sanitized and begin <= int(sanitized[-1][1]):
                begin = int(sanitized[-1][1]) + 1
                onset_off = 0.5
            if begin > end:
                continue
            sanitized.append(
                [
                    int(begin),
                    int(end),
                    bool(has_onset),
                    bool(has_offset),
                    float(onset_off),
                    float(offset_off),
                ]
            )

        if not sanitized:
            continue
        instrument_id = int(pair_id) // NUM_PITCHES
        pitch_value = int(pair_id) % NUM_PITCHES
        pair_presence[instrument_id, pitch_value] = True
        positive_pair_ids.append(int(pair_id))
        pair_intervals.append([(int(item[0]), int(item[1])) for item in sanitized])
        has_onset_tracks.append([bool(item[2]) for item in sanitized])
        has_offset_tracks.append([bool(item[3]) for item in sanitized])
        onset_offsets_tracks.append([float(item[4]) for item in sanitized])
        offset_offsets_tracks.append([float(item[5]) for item in sanitized])
        instrument_sets_tracks.append([(instrument_id,) for _ in sanitized])

    return PitchIntervalTargets(
        intervals=pair_intervals,
        has_onset=has_onset_tracks,
        has_offset=has_offset_tracks,
        onset_offsets=onset_offsets_tracks,
        offset_offsets=offset_offsets_tracks,
        instrument_sets=instrument_sets_tracks,
        positive_pair_ids=positive_pair_ids,
        pair_presence=pair_presence,
    )
