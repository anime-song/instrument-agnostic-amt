from __future__ import annotations

import numpy as np
import torch

from instrument_agnostic_amt.amt.data.constants import (
    MAX_MIDI_PITCH,
    MIN_MIDI_PITCH,
    NUM_PITCHES,
)
from instrument_agnostic_amt.amt.modeling.heads.interval_boundaries import (
    PitchIntervalTargets,
)
from instrument_agnostic_amt.amt.modeling.model import normalize_semi_crf_version
from instrument_agnostic_amt.taxonomy.instrument_classes import (
    NUM_INSTRUMENT_CLASSES,
)


def _ms_to_sample_index(ms: np.ndarray, *, sample_rate: int) -> np.ndarray:
    return np.rint(
        ms.astype(np.float64, copy=False) * float(sample_rate) / 1000.0
    ).astype(np.int64, copy=False)


def _map_model_pitch_array(pitch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pitch_i64 = pitch.astype(np.int64, copy=False)
    valid = (pitch_i64 >= MIN_MIDI_PITCH) & (pitch_i64 <= MAX_MIDI_PITCH)
    return (pitch_i64[valid] - MIN_MIDI_PITCH).astype(np.int64, copy=False), valid


def build_frame_note_targets(
    *,
    active_start_ms: np.ndarray,
    active_end_ms: np.ndarray,
    active_pitch: np.ndarray,
    sample_rate: int,
    hop_length: int,
    num_frames: int,
) -> torch.Tensor:
    targets = np.zeros((num_frames, NUM_PITCHES), dtype=np.float32)
    if num_frames <= 0 or active_start_ms.size == 0:
        return torch.from_numpy(targets)
    starts = _ms_to_sample_index(active_start_ms, sample_rate=sample_rate)
    ends = _ms_to_sample_index(active_end_ms, sample_rate=sample_rate)
    start_frames = np.clip(starts // hop_length, 0, num_frames - 1)
    end_frames = np.clip(
        np.maximum((np.maximum(ends - 1, 0) // hop_length) + 1, start_frames + 1),
        0,
        num_frames,
    )
    pitches, valid = _map_model_pitch_array(active_pitch)
    for begin, end, pitch in zip(
        start_frames[valid].tolist(), end_frames[valid].tolist(), pitches.tolist()
    ):
        if begin < num_frames:
            targets[begin:end, pitch] = 1.0
    return torch.from_numpy(targets)


def build_frame_instrument_targets(
    *,
    active_start_ms: np.ndarray,
    active_end_ms: np.ndarray,
    active_pitch: np.ndarray,
    active_instrument: np.ndarray,
    sample_rate: int,
    hop_length: int,
    num_frames: int,
) -> torch.Tensor:
    targets = np.zeros(
        (num_frames, NUM_PITCHES, NUM_INSTRUMENT_CLASSES), dtype=np.float32
    )
    if num_frames <= 0 or active_start_ms.size == 0:
        return torch.from_numpy(targets)
    starts = _ms_to_sample_index(active_start_ms, sample_rate=sample_rate)
    ends = _ms_to_sample_index(active_end_ms, sample_rate=sample_rate)
    start_frames = np.clip(starts // hop_length, 0, num_frames - 1)
    end_frames = np.clip(
        np.maximum((np.maximum(ends - 1, 0) // hop_length) + 1, start_frames + 1),
        0,
        num_frames,
    )
    pitches, valid = _map_model_pitch_array(active_pitch)
    instruments = active_instrument[valid].astype(np.int64, copy=False)
    for begin, end, pitch, instrument in zip(
        start_frames[valid].tolist(),
        end_frames[valid].tolist(),
        pitches.tolist(),
        instruments.tolist(),
    ):
        if begin < num_frames and 0 <= instrument < NUM_INSTRUMENT_CLASSES:
            targets[begin:end, pitch, instrument] = 1.0
    return torch.from_numpy(targets)


def _empty_v1_targets(num_pitch_slots: int) -> PitchIntervalTargets:
    track_count = NUM_PITCHES * max(1, int(num_pitch_slots))
    return PitchIntervalTargets(
        intervals=[[] for _ in range(track_count)],
        has_onset=[[] for _ in range(track_count)],
        has_offset=[[] for _ in range(track_count)],
        onset_offsets=[[] for _ in range(track_count)],
        offset_offsets=[[] for _ in range(track_count)],
        instrument_sets=[[] for _ in range(track_count)],
    )


def _framed_notes(
    *,
    active_start_ms: np.ndarray,
    active_end_ms: np.ndarray,
    sample_rate: int,
    hop_length: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = _ms_to_sample_index(active_start_ms, sample_rate=sample_rate)
    ends = _ms_to_sample_index(active_end_ms, sample_rate=sample_rate)
    real_starts = starts.astype(np.float64, copy=False) / float(hop_length)
    real_ends = ends.astype(np.float64, copy=False) / float(hop_length)
    raw_starts = starts // hop_length
    raw_ends = (np.maximum(ends - 1, 0) // hop_length) + 1
    start_frames = np.clip(raw_starts, 0, num_frames - 1)
    end_frames = np.clip(np.maximum(raw_ends, start_frames + 1), 0, num_frames)
    return (
        start_frames,
        end_frames,
        real_starts - raw_starts,
        real_ends - (raw_ends - 1),
    )


def build_v1_interval_targets(
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
    """Build V1 pitch/slot targets, including the original slot assignment."""

    num_pitch_slots = max(1, int(num_pitch_slots))
    targets = _empty_v1_targets(num_pitch_slots)
    if num_frames <= 0 or active_start_ms.size == 0:
        return targets
    starts, ends, onset_offsets, offset_offsets = _framed_notes(
        active_start_ms=active_start_ms,
        active_end_ms=active_end_ms,
        sample_rate=sample_rate,
        hop_length=hop_length,
        num_frames=num_frames,
    )
    pitches, valid = _map_model_pitch_array(active_pitch)
    if not np.any(valid):
        return targets
    raw_by_pitch: list[list[tuple[int, int, int, bool, bool, float, float]]] = [
        [] for _ in range(NUM_PITCHES)
    ]
    for begin, end_exclusive, pitch, instrument, has_onset, has_offset, onset, offset in zip(
        starts[valid].tolist(),
        ends[valid].tolist(),
        pitches.tolist(),
        active_instrument[valid].tolist(),
        active_has_onset[valid].tolist(),
        active_has_offset[valid].tolist(),
        onset_offsets[valid].tolist(),
        offset_offsets[valid].tolist(),
    ):
        if begin < num_frames and end_exclusive > begin:
            raw_by_pitch[pitch].append(
                (
                    int(begin),
                    int(end_exclusive - 1),
                    int(instrument),
                    bool(has_onset),
                    bool(has_offset),
                    float(onset),
                    float(offset),
                )
            )

    if num_pitch_slots > 1:
        for pitch, intervals in enumerate(raw_by_pitch):
            intervals.sort(key=lambda item: (item[0], item[1], item[3], item[4], item[2]))
            last_end = [-1] * num_pitch_slots
            for begin, end, instrument, has_onset, has_offset, onset, offset in intervals:
                slot = next(
                    (index for index, previous_end in enumerate(last_end) if begin > previous_end),
                    None,
                )
                if slot is None or begin > end:
                    continue
                track = pitch * num_pitch_slots + slot
                targets.intervals[track].append((begin, end))
                targets.has_onset[track].append(has_onset)
                targets.has_offset[track].append(has_offset)
                targets.onset_offsets[track].append(onset)
                targets.offset_offsets[track].append(offset)
                targets.instrument_sets[track].append(
                    (instrument,) if 0 <= instrument < NUM_INSTRUMENT_CLASSES else ()
                )
                last_end[slot] = end
        return targets

    for pitch, intervals in enumerate(raw_by_pitch):
        intervals.sort(key=lambda item: (item[0], item[1], item[3], item[4], item[2]))
        sanitized: list[list[int | bool | float]] = []
        for begin, end, _, has_onset, has_offset, onset, offset in intervals:
            if sanitized and begin <= int(sanitized[-1][1]):
                if begin > int(sanitized[-1][0]):
                    sanitized[-1][1] = begin - 1
                    sanitized[-1][3] = True
                    sanitized[-1][5] = 0.5
                else:
                    sanitized.pop()
            if sanitized and begin <= int(sanitized[-1][1]):
                begin = int(sanitized[-1][1]) + 1
                onset = 0.5
            if begin <= end:
                sanitized.append([begin, end, has_onset, has_offset, onset, offset])
        for begin, end, has_onset, has_offset, onset, offset in sanitized:
            instrument_ids = sorted(
                {
                    instrument
                    for raw_begin, raw_end, instrument, *_ in intervals
                    if not (raw_end < int(begin) or raw_begin > int(end))
                    and 0 <= instrument < NUM_INSTRUMENT_CLASSES
                }
            )
            targets.intervals[pitch].append((int(begin), int(end)))
            targets.has_onset[pitch].append(bool(has_onset))
            targets.has_offset[pitch].append(bool(has_offset))
            targets.onset_offsets[pitch].append(float(onset))
            targets.offset_offsets[pitch].append(float(offset))
            targets.instrument_sets[pitch].append(tuple(instrument_ids))
    return targets


def _empty_v2_targets() -> PitchIntervalTargets:
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


def build_v2_interval_targets(
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
    """Build sparse instrument-pitch targets for the overlap-capable V2 head."""

    if num_frames <= 0 or active_start_ms.size == 0:
        return _empty_v2_targets()
    starts, ends, onset_offsets, offset_offsets = _framed_notes(
        active_start_ms=active_start_ms,
        active_end_ms=active_end_ms,
        sample_rate=sample_rate,
        hop_length=hop_length,
        num_frames=num_frames,
    )
    pitches, pitch_valid = _map_model_pitch_array(active_pitch)
    instrument_values = active_instrument.astype(np.int64, copy=False)
    instrument_valid = (instrument_values >= 0) & (
        instrument_values < NUM_INSTRUMENT_CLASSES
    )
    valid = pitch_valid & instrument_valid
    if not np.any(valid):
        return _empty_v2_targets()
    # Re-map after applying the joint mask because ``pitches`` only contains pitch-valid rows.
    pitches = active_pitch[valid].astype(np.int64, copy=False) - MIN_MIDI_PITCH
    raw_by_pair: dict[int, list[tuple[int, int, bool, bool, float, float]]] = {}
    for begin, end_exclusive, pitch, instrument, has_onset, has_offset, onset, offset in zip(
        starts[valid].tolist(),
        ends[valid].tolist(),
        pitches.tolist(),
        instrument_values[valid].tolist(),
        active_has_onset[valid].tolist(),
        active_has_offset[valid].tolist(),
        onset_offsets[valid].tolist(),
        offset_offsets[valid].tolist(),
    ):
        if begin >= num_frames or end_exclusive <= begin:
            continue
        pair_id = int(instrument) * NUM_PITCHES + int(pitch)
        raw_by_pair.setdefault(pair_id, []).append(
            (
                int(begin),
                int(end_exclusive - 1),
                bool(has_onset),
                bool(has_offset),
                float(onset),
                float(offset),
            )
        )

    targets = _empty_v2_targets()
    for pair_id in sorted(raw_by_pair):
        intervals = sorted(raw_by_pair[pair_id], key=lambda item: item[:4])
        sanitized: list[list[int | bool | float]] = []
        for begin, end, has_onset, has_offset, onset, offset in intervals:
            if sanitized and begin <= int(sanitized[-1][1]):
                if begin > int(sanitized[-1][0]):
                    sanitized[-1][1] = begin - 1
                    sanitized[-1][3] = True
                    sanitized[-1][5] = 0.5
                else:
                    sanitized.pop()
            if sanitized and begin <= int(sanitized[-1][1]):
                begin = int(sanitized[-1][1]) + 1
                onset = 0.5
            if begin <= end:
                sanitized.append([begin, end, has_onset, has_offset, onset, offset])
        if not sanitized:
            continue
        instrument = pair_id // NUM_PITCHES
        pitch = pair_id % NUM_PITCHES
        assert targets.pair_presence is not None
        targets.pair_presence[instrument, pitch] = True
        targets.positive_pair_ids.append(pair_id)
        targets.intervals.append([(int(row[0]), int(row[1])) for row in sanitized])
        targets.has_onset.append([bool(row[2]) for row in sanitized])
        targets.has_offset.append([bool(row[3]) for row in sanitized])
        targets.onset_offsets.append([float(row[4]) for row in sanitized])
        targets.offset_offsets.append([float(row[5]) for row in sanitized])
        targets.instrument_sets.append([(instrument,) for _ in sanitized])
    return targets


def build_interval_targets(
    *,
    semi_crf_version: str,
    num_pitch_slots: int = 1,
    **kwargs,
) -> PitchIntervalTargets:
    version = normalize_semi_crf_version(semi_crf_version)
    if version == "v1":
        return build_v1_interval_targets(num_pitch_slots=num_pitch_slots, **kwargs)
    return build_v2_interval_targets(**kwargs)


# Backward-compatible function name for package callers. V1 is the default model mode.
build_pitch_interval_targets = build_v1_interval_targets
