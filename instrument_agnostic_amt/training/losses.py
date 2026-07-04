from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..modeling.heads.interval_boundaries import gather_boundary_targets
from ..modeling.heads.semi_crf import compute_pitch_interval_loss
from ..modeling.model import AudioSemiCRFTransformer


def compute_losses(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, Any],
    args: Any | None = None,
    model: AudioSemiCRFTransformer | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    interval_query = outputs.get("interval_query")
    interval_key = outputs.get("interval_key")
    interval_diag = outputs.get("interval_diag")
    frame_valid_mask = outputs.get("frame_valid_mask")
    pitch_query_features = outputs.get("pitch_query_features")
    interval_features = outputs.get("interval_features")
    interval_targets = batch.get("interval_targets")

    if (
        interval_query is None
        or interval_key is None
        or interval_diag is None
        or interval_targets is None
        or frame_valid_mask is None
    ):
        raise ValueError("SemiCRF training requires interval outputs and targets")

    semi_crf_loss_weight = (
        1.0 if args is None else float(getattr(args, "semi_crf_loss_weight", 1.0))
    )
    semi_crf_false_negative_cost = (
        0.0
        if args is None
        else float(getattr(args, "semi_crf_false_negative_cost", 0.0))
    )
    semi_crf_false_positive_cost = (
        0.0
        if args is None
        else float(getattr(args, "semi_crf_false_positive_cost", 0.0))
    )
    interval_presence_loss_weight = (
        1.0
        if args is None
        else float(getattr(args, "interval_presence_loss_weight", 1.0))
    )
    interval_offset_loss_weight = (
        1.0
        if args is None
        else float(getattr(args, "interval_offset_loss_weight", 1.0))
    )

    valid_lengths = frame_valid_mask.to(dtype=torch.long).sum(dim=-1)
    if model is not None and hasattr(model, "config"):
        length_scaling = model.config.semi_crf_length_scaling
        length_penalty = model.config.semi_crf_length_penalty
    else:
        length_scaling = (
            "linear"
            if args is None
            else str(getattr(args, "semi_crf_length_scaling", "linear"))
        )
        length_penalty = (
            0.0
            if args is None
            else float(getattr(args, "semi_crf_length_penalty", 0.0))
        )

    semi_crf_loss, track_count, interval_count = compute_pitch_interval_loss(
        interval_query,
        interval_key,
        interval_diag,
        [target.intervals for target in interval_targets],
        valid_lengths,
        length_scaling=length_scaling,
        length_penalty=length_penalty,
        track_batch_size=128
        if args is None
        else int(getattr(args, "semi_crf_track_batch_size", 128)),
        false_negative_cost=semi_crf_false_negative_cost,
        false_positive_cost=semi_crf_false_positive_cost,
    )
    total_loss = semi_crf_loss * semi_crf_loss_weight

    zero = interval_query.sum() * 0.0
    interval_presence_loss = zero
    interval_offset_loss = zero
    interval_boundary_loss = zero
    interval_boundary_interval_count = torch.tensor(
        0, device=interval_query.device, dtype=torch.long
    )

    if model is not None and model.supports_interval_boundaries():
        boundary_features = (
            interval_features if interval_features is not None else pitch_query_features
        )
        if boundary_features is None:
            raise ValueError("interval boundary loss requires interval features")
        boundary_logits, entries = model.predict_interval_boundaries(
            boundary_features,
            [target.intervals for target in interval_targets],
        )
        if entries:
            has_onset, has_offset, onset_offsets, offset_offsets = (
                gather_boundary_targets(
                    interval_targets,
                    entries,
                    device=boundary_logits.device,
                )
            )
            presence_logits, offset_logits = boundary_logits.chunk(2, dim=-1)
            boundary_targets = torch.stack([has_onset, has_offset], dim=-1)
            interval_presence_loss = F.binary_cross_entropy_with_logits(
                presence_logits,
                boundary_targets,
            )

            offset_targets = torch.stack([onset_offsets, offset_offsets], dim=-1)
            offset_targets = torch.clamp(offset_targets, 0.0, 1.0) * 0.99 + 0.005
            offset_dist = torch.distributions.ContinuousBernoulli(logits=offset_logits)
            interval_offset_loss = (
                -offset_dist.log_prob(offset_targets).sum(dim=-1).mean()
            )
            interval_boundary_loss = interval_presence_loss + interval_offset_loss
            interval_boundary_interval_count = torch.tensor(
                len(entries),
                device=interval_query.device,
                dtype=torch.long,
            )
            total_loss = total_loss + (
                interval_presence_loss * interval_presence_loss_weight
                + interval_offset_loss * interval_offset_loss_weight
            )

    return total_loss, {
        "total_loss": total_loss,
        "semi_crf_loss": semi_crf_loss,
        "semi_crf_track_count": torch.tensor(
            track_count, device=interval_query.device, dtype=torch.long
        ),
        "semi_crf_interval_count": torch.tensor(
            interval_count, device=interval_query.device, dtype=torch.long
        ),
        "semi_crf_false_negative_cost": interval_query.new_tensor(
            semi_crf_false_negative_cost
        ),
        "semi_crf_false_positive_cost": interval_query.new_tensor(
            semi_crf_false_positive_cost
        ),
        "interval_boundary_loss": interval_boundary_loss,
        "interval_presence_loss": interval_presence_loss,
        "interval_offset_loss": interval_offset_loss,
        "interval_boundary_interval_count": interval_boundary_interval_count,
    }
