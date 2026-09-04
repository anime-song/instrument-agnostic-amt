from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from instrument_agnostic_amt.amt.modeling.heads.interval_boundaries import (
    gather_boundary_targets,
    gather_instrument_targets,
)
from instrument_agnostic_amt.amt.modeling.heads.semi_crf import compute_pitch_interval_loss
from instrument_agnostic_amt.amt.modeling.model import AudioSemiCRFTransformer
from .loss_config import AMTLossConfig


def _instrument_loss(
    logits: torch.Tensor, targets: torch.Tensor, *, loss_type: str
) -> torch.Tensor:
    loss_type = str(loss_type).strip().lower()
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets)
    if loss_type != "ce":
        raise ValueError(f"Unsupported instrument_loss_type: {loss_type}")
    valid = targets.sum(dim=-1) == 1.0
    if not torch.any(valid):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid], targets.argmax(dim=-1)[valid])


def compute_v1_losses(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, Any],
    *,
    config: AMTLossConfig,
    model: AudioSemiCRFTransformer,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    interval_query = outputs["interval_query"]
    interval_key = outputs["interval_key"]
    interval_diag = outputs["interval_diag"]
    interval_targets = batch["interval_targets"]
    frame_valid_mask = outputs["frame_valid_mask"]

    valid_lengths = frame_valid_mask.long().sum(dim=-1)
    semi_crf_loss, track_count, interval_count = compute_pitch_interval_loss(
        interval_query,
        interval_key,
        interval_diag,
        [target.intervals for target in interval_targets],
        valid_lengths,
        length_scaling=model.config.semi_crf_length_scaling,
        length_penalty=model.config.semi_crf_length_penalty,
        track_batch_size=config.semi_crf_track_batch_size,
        false_negative_cost=config.semi_crf_false_negative_cost,
        false_positive_cost=config.semi_crf_false_positive_cost,
        backend=config.semi_crf_loss_backend,
    )
    total_loss = semi_crf_loss * config.semi_crf_loss_weight
    zero = interval_query.sum() * 0.0
    interval_presence_loss = zero
    interval_offset_loss = zero
    boundary_count = 0

    interval_features = outputs["interval_features"]
    if model.supports_interval_boundaries():
        boundary_logits, entries = model.predict_interval_boundaries(
            interval_features, [target.intervals for target in interval_targets]
        )
        boundary_count = len(entries)
        if entries:
            has_onset, has_offset, onset_offsets, offset_offsets = gather_boundary_targets(
                interval_targets, entries, device=boundary_logits.device
            )
            presence_logits, offset_logits = boundary_logits.chunk(2, dim=-1)
            interval_presence_loss = F.binary_cross_entropy_with_logits(
                presence_logits, torch.stack([has_onset, has_offset], dim=-1)
            )
            offset_targets = torch.stack([onset_offsets, offset_offsets], dim=-1)
            offset_targets = offset_targets.clamp(0.0, 1.0) * 0.99 + 0.005
            interval_offset_loss = -torch.distributions.ContinuousBernoulli(
                logits=offset_logits
            ).log_prob(offset_targets).sum(dim=-1).mean()
            total_loss = total_loss + (
                interval_presence_loss * config.interval_presence_loss_weight
                + interval_offset_loss * config.interval_offset_loss_weight
            )

    instrument_loss = zero
    instrument_features = outputs["instrument_features"]
    if (
        model.supports_interval_instruments()
        and model._use_interval_instrument_head
    ):
        logits, entries = model.predict_interval_instruments(
            instrument_features, [target.intervals for target in interval_targets]
        )
        if entries:
            targets = gather_instrument_targets(
                interval_targets,
                entries,
                num_instruments=model.config.num_instrument_classes,
                device=logits.device,
            )
            keep = torch.ones(len(entries), dtype=torch.bool, device=logits.device)
            mask_flag = batch.get("mask_instrument_loss")
            if mask_flag is not None:
                entry_batches = torch.tensor(
                    [entry[0] for entry in entries], device=logits.device
                )
                keep &= ~mask_flag.to(logits.device)[entry_batches]
            if torch.any(keep):
                instrument_loss = _instrument_loss(
                    logits[keep],
                    targets[keep],
                    loss_type=config.instrument_loss_type,
                )
    else:
        logits = outputs.get("instrument_logits")
        targets = batch.get("frame_instrument_targets")
        active = batch.get("frame_active_targets")
        if logits is not None and targets is not None and active is not None:
            targets = targets.to(logits.device)
            active = active.to(logits.device)
            keep = (active > 0.5) & frame_valid_mask.unsqueeze(-1)
            mask_flag = batch.get("mask_instrument_loss")
            if mask_flag is not None:
                keep &= ~mask_flag.to(logits.device).view(-1, 1, 1)
            if torch.any(keep):
                instrument_loss = _instrument_loss(
                    logits[keep],
                    targets[keep],
                    loss_type=config.instrument_loss_type,
                )
    total_loss = total_loss + instrument_loss * config.instrument_loss_weight

    device = interval_query.device
    return total_loss, {
        "total_loss": total_loss,
        "semi_crf_loss": semi_crf_loss,
        "semi_crf_track_count": torch.tensor(track_count, device=device),
        "semi_crf_interval_count": torch.tensor(interval_count, device=device),
        "selected_pair_count": torch.tensor(track_count, device=device),
        "semi_crf_false_negative_cost": interval_query.new_tensor(
            config.semi_crf_false_negative_cost
        ),
        "semi_crf_false_positive_cost": interval_query.new_tensor(
            config.semi_crf_false_positive_cost
        ),
        "interval_boundary_loss": interval_presence_loss + interval_offset_loss,
        "interval_presence_loss": interval_presence_loss,
        "interval_offset_loss": interval_offset_loss,
        "interval_boundary_interval_count": torch.tensor(boundary_count, device=device),
        "instrument_loss": instrument_loss,
    }
