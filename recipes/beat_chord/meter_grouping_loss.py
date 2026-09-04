from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from instrument_agnostic_amt.beat_chord.meter_grouping import (
    MeterGroupingSpec,
    grouping_boundary_offsets,
    grouping_spec_for_meter,
)


@dataclass(frozen=True)
class MajorGroupingLossResult:
    loss: torch.Tensor
    valid_pattern_loss: torch.Tensor
    accent_alignment_loss: torch.Tensor
    supervised_bar_count: int


def _pooled_logits(logits: torch.Tensor, tolerance: int) -> torch.Tensor:
    if tolerance <= 0:
        return logits
    return F.max_pool1d(
        logits.unsqueeze(1),
        kernel_size=2 * int(tolerance) + 1,
        stride=1,
        padding=int(tolerance),
    ).squeeze(1)


def _onset_salience(midi_frames: torch.Tensor) -> torch.Tensor:
    if midi_frames.dim() != 4:
        raise ValueError("midi_frames must have shape [B, C, T, P]")
    channel_count = int(midi_frames.shape[1])
    if channel_count < 2 or channel_count % 2 != 0:
        raise ValueError("MIDI frame channels must contain sustain/onset halves")
    onset_frames = midi_frames[:, channel_count // 2 :, :, :]
    return onset_frames.float().sum(dim=(1, 3))


def _candidate_boundary_labels(
    reference: torch.Tensor,
    spec: MeterGroupingSpec,
) -> torch.Tensor:
    numerator = int(spec.numerator)
    return torch.tensor(
        [
            [
                1.0 if offset in expected_offsets else 0.0
                for offset in range(1, numerator)
            ]
            for expected_offsets in (
                set(grouping_boundary_offsets(pattern)) for pattern in spec.patterns
            )
        ],
        device=reference.device,
        dtype=reference.dtype,
    )


def _candidate_log_probabilities(
    boundary_logits: torch.Tensor,
    candidate_labels: torch.Tensor,
) -> torch.Tensor:
    positive_log_probs = F.logsigmoid(boundary_logits).unsqueeze(0)
    negative_log_probs = F.logsigmoid(-boundary_logits).unsqueeze(0)
    return (
        candidate_labels * positive_log_probs
        + (1.0 - candidate_labels) * negative_log_probs
    ).sum(dim=1)


def _candidate_accent_scores(
    beat_salience: torch.Tensor,
    candidate_labels: torch.Tensor,
) -> torch.Tensor:
    normalized = (beat_salience - beat_salience.mean()) / beat_salience.std(
        unbiased=False
    ).clamp_min(1e-4)
    boundary_counts = candidate_labels.sum(dim=1).clamp_min(1.0)
    return (candidate_labels * normalized.unsqueeze(0)).sum(dim=1) / boundary_counts


def major_grouping_loss(
    *,
    group_boundary_logits: torch.Tensor,
    beat_targets: torch.Tensor,
    downbeat_targets: torch.Tensor,
    meter_targets: torch.Tensor,
    beat_mask: torch.Tensor | None,
    midi_frames: torch.Tensor,
    meter_classes: Sequence[tuple[int, int]],
    tolerance: int,
    accent_loss_weight: float,
    accent_temperature: float,
) -> MajorGroupingLossResult:
    """Train fixed/latent major boundaries on complete annotated bars."""

    expected_shape = group_boundary_logits.shape
    for name, tensor in (
        ("beat_targets", beat_targets),
        ("downbeat_targets", downbeat_targets),
        ("meter_targets", meter_targets),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} must match group_boundary_logits shape")
    if beat_mask is not None and beat_mask.shape != expected_shape:
        raise ValueError("beat_mask must match group_boundary_logits shape")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if accent_loss_weight < 0.0:
        raise ValueError("accent_loss_weight must be non-negative")
    if accent_temperature <= 0.0:
        raise ValueError("accent_temperature must be positive")

    pooled_logits = _pooled_logits(group_boundary_logits.float(), tolerance)
    onset_salience = _pooled_logits(
        _onset_salience(midi_frames).to(pooled_logits.device),
        tolerance,
    )
    valid_losses: list[torch.Tensor] = []
    accent_losses: list[torch.Tensor] = []

    for batch_index in range(int(group_boundary_logits.shape[0])):
        valid_frames = meter_targets[batch_index] >= 0
        if beat_mask is not None:
            valid_frames = valid_frames & (beat_mask[batch_index] > 0.0)
        downbeat_frames = torch.nonzero(
            (downbeat_targets[batch_index] > 0.5) & valid_frames,
            as_tuple=False,
        ).flatten()
        if downbeat_frames.numel() < 2:
            continue

        for start_tensor, end_tensor in zip(
            downbeat_frames[:-1],
            downbeat_frames[1:],
        ):
            start_frame = int(start_tensor.item())
            end_frame = int(end_tensor.item())
            if end_frame <= start_frame + 1:
                continue
            interval_valid = valid_frames[start_frame:end_frame]
            interval_meters = meter_targets[batch_index, start_frame:end_frame][
                interval_valid
            ]
            if interval_meters.numel() == 0:
                continue
            meter_index = int(torch.mode(interval_meters).values.item())
            if meter_index < 0 or meter_index >= len(meter_classes):
                continue
            meter_num, meter_den = meter_classes[meter_index]
            spec = grouping_spec_for_meter(meter_num, meter_den)
            if spec is None:
                continue

            beat_frames = torch.nonzero(
                (beat_targets[batch_index, start_frame:end_frame] > 0.5)
                & interval_valid,
                as_tuple=False,
            ).flatten()
            if beat_frames.numel() != int(meter_num):
                continue
            beat_frames = beat_frames + start_frame
            internal_frames = beat_frames[1:]
            if internal_frames.numel() != int(meter_num) - 1:
                continue

            internal_logits = pooled_logits[batch_index, internal_frames]
            candidate_labels = _candidate_boundary_labels(
                internal_logits,
                spec,
            )
            candidate_log_probs = _candidate_log_probabilities(
                internal_logits,
                candidate_labels,
            )
            normalizer = float(max(1, int(meter_num) - 1))
            valid_losses.append(
                -torch.logsumexp(candidate_log_probs, dim=0) / normalizer
            )

            if len(spec.patterns) <= 1 or accent_loss_weight <= 0.0:
                continue
            internal_salience = onset_salience[batch_index, internal_frames]
            accent_scores = _candidate_accent_scores(
                internal_salience,
                candidate_labels,
            )
            accent_distribution = torch.softmax(
                accent_scores / float(accent_temperature),
                dim=0,
            ).detach()
            accent_losses.append(
                -(
                    accent_distribution * torch.log_softmax(candidate_log_probs, dim=0)
                ).sum()
            )

    zero = group_boundary_logits.sum() * 0.0
    if not valid_losses:
        return MajorGroupingLossResult(zero, zero, zero, 0)

    valid_pattern_loss = torch.stack(valid_losses).mean()
    accent_alignment_loss = torch.stack(accent_losses).mean() if accent_losses else zero
    total = valid_pattern_loss + (float(accent_loss_weight) * accent_alignment_loss)
    return MajorGroupingLossResult(
        loss=total,
        valid_pattern_loss=valid_pattern_loss,
        accent_alignment_loss=accent_alignment_loss,
        supervised_bar_count=len(valid_losses),
    )
