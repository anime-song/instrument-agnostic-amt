from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

Interval = tuple[int, int]
FlattenedIntervalEntry = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class PitchIntervalTargets:
    intervals: list[list[Interval]]
    has_onset: list[list[bool]]
    has_offset: list[list[bool]]
    onset_offsets: list[list[float]]
    offset_offsets: list[list[float]]
    instrument_sets: list[list[tuple[int, ...]]] = field(default_factory=list)


def count_pitch_intervals(
    targets_or_intervals: PitchIntervalTargets | Sequence[Sequence[Interval]],
) -> int:
    interval_tracks = (
        targets_or_intervals.intervals
        if isinstance(targets_or_intervals, PitchIntervalTargets)
        else targets_or_intervals
    )
    return sum(len(track_intervals) for track_intervals in interval_tracks)


def flatten_interval_entries(
    intervals_batch: Sequence[Sequence[Sequence[Interval]]],
) -> list[FlattenedIntervalEntry]:
    entries: list[FlattenedIntervalEntry] = []
    for batch_index, sample_intervals in enumerate(intervals_batch):
        for pitch_index, track_intervals in enumerate(sample_intervals):
            for interval_index, (begin, end) in enumerate(track_intervals):
                entries.append(
                    (
                        int(batch_index),
                        int(pitch_index),
                        int(interval_index),
                        int(begin),
                        int(end),
                    )
                )
    return entries


def gather_interval_endpoint_features(
    frame_features: torch.Tensor,
    intervals_batch: Sequence[Sequence[Sequence[Interval]]],
) -> tuple[torch.Tensor, list[FlattenedIntervalEntry]]:
    if frame_features.dim() != 4:
        raise ValueError("frame_features must have shape [B, T, P, D]")

    entries = flatten_interval_entries(intervals_batch)
    feature_dim = int(frame_features.shape[-1])
    if not entries:
        return frame_features.new_zeros((0, feature_dim * 3)), []

    device = frame_features.device
    batch_indices = torch.tensor(
        [entry[0] for entry in entries],
        device=device,
        dtype=torch.long,
    )
    pitch_indices = torch.tensor(
        [entry[1] for entry in entries],
        device=device,
        dtype=torch.long,
    )
    begin_indices = torch.tensor(
        [entry[3] for entry in entries],
        device=device,
        dtype=torch.long,
    )
    end_indices = torch.tensor(
        [entry[4] for entry in entries],
        device=device,
        dtype=torch.long,
    )

    begin_features = frame_features[batch_indices, begin_indices, pitch_indices]
    end_features = frame_features[batch_indices, end_indices, pitch_indices]
    stacked_features = torch.cat(
        [
            begin_features,
            end_features,
            begin_features * end_features,
        ],
        dim=-1,
    )
    return stacked_features, entries


def gather_interval_sequence_features(
    frame_features: torch.Tensor,
    intervals_batch: Sequence[Sequence[Sequence[Interval]]],
) -> tuple[torch.Tensor, list[FlattenedIntervalEntry]]:
    """Gather endpoint and pooled features for each interval entry."""

    if frame_features.dim() != 4:
        raise ValueError("frame_features must have shape [B, T, P, D]")

    entries = flatten_interval_entries(intervals_batch)
    feature_dim = int(frame_features.shape[-1])
    if not entries:
        return frame_features.new_zeros((0, feature_dim * 4 + 1)), []

    device = frame_features.device
    batch_indices = torch.tensor(
        [entry[0] for entry in entries],
        device=device,
        dtype=torch.long,
    )
    pitch_indices = torch.tensor(
        [entry[1] for entry in entries],
        device=device,
        dtype=torch.long,
    )
    begin_indices = torch.tensor(
        [entry[3] for entry in entries],
        device=device,
        dtype=torch.long,
    )
    end_indices = torch.tensor(
        [entry[4] for entry in entries],
        device=device,
        dtype=torch.long,
    )

    begin_features = frame_features[batch_indices, begin_indices, pitch_indices]
    end_features = frame_features[batch_indices, end_indices, pitch_indices]

    prefix_sum = frame_features.cumsum(dim=1)
    interval_sum = prefix_sum[batch_indices, end_indices, pitch_indices]
    has_previous = begin_indices > 0
    if bool(torch.any(has_previous).item()):
        interval_sum = interval_sum.clone()
        interval_sum[has_previous] = (
            interval_sum[has_previous]
            - prefix_sum[
                batch_indices[has_previous],
                begin_indices[has_previous] - 1,
                pitch_indices[has_previous],
            ]
        )

    interval_lengths = (end_indices - begin_indices + 1).clamp_min(1)
    mean_features = interval_sum / interval_lengths.unsqueeze(-1).to(
        dtype=interval_sum.dtype
    )
    normalized_lengths = interval_lengths.unsqueeze(-1).to(
        dtype=interval_sum.dtype
    ) / float(max(int(frame_features.shape[1]), 1))

    stacked_features = torch.cat(
        [
            begin_features,
            end_features,
            mean_features,
            begin_features * end_features,
            normalized_lengths,
        ],
        dim=-1,
    )
    return stacked_features, entries


def gather_boundary_targets(
    targets_batch: Sequence[PitchIntervalTargets],
    entries: Sequence[FlattenedIntervalEntry],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not entries:
        empty = torch.zeros((0,), device=device, dtype=torch.float32)
        return empty, empty, empty, empty

    has_onset = torch.tensor(
        [
            float(targets_batch[batch_index].has_onset[pitch_index][interval_index])
            for batch_index, pitch_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    has_offset = torch.tensor(
        [
            float(targets_batch[batch_index].has_offset[pitch_index][interval_index])
            for batch_index, pitch_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    onset_offsets = torch.tensor(
        [
            float(targets_batch[batch_index].onset_offsets[pitch_index][interval_index])
            for batch_index, pitch_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    offset_offsets = torch.tensor(
        [
            float(
                targets_batch[batch_index].offset_offsets[pitch_index][interval_index]
            )
            for batch_index, pitch_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    return has_onset, has_offset, onset_offsets, offset_offsets


def gather_instrument_targets(
    targets_batch: Sequence[PitchIntervalTargets],
    entries: Sequence[FlattenedIntervalEntry],
    *,
    num_instruments: int,
    device: torch.device,
) -> torch.Tensor:
    """Gather multi-label instrument targets for each interval entry."""

    if num_instruments <= 0:
        raise ValueError("num_instruments must be positive")
    if not entries:
        return torch.zeros((0, num_instruments), device=device, dtype=torch.float32)

    targets = torch.zeros(
        (len(entries), int(num_instruments)),
        device=device,
        dtype=torch.float32,
    )
    for row_index, (batch_index, pitch_index, interval_index, _, _) in enumerate(
        entries
    ):
        instrument_sets = targets_batch[batch_index].instrument_sets
        if not instrument_sets or pitch_index >= len(instrument_sets):
            continue
        if interval_index >= len(instrument_sets[pitch_index]):
            continue
        for instrument_id in instrument_sets[pitch_index][interval_index]:
            if 0 <= int(instrument_id) < int(num_instruments):
                targets[row_index, int(instrument_id)] = 1.0
    return targets
