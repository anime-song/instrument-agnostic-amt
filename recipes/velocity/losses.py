"""Losses and metrics for velocity training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VelocityLossConfig:
    velocity_ce_weight: float = 1.0
    velocity_expected_weight: float = 0.25
    stem_gain_weight: float = 0.0
    label_smoothing: float = 0.02
    gain_huber_delta_db: float = 1.0


def _zero_from(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return outputs["velocity_logits"].sum() * 0.0


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _center_stem_gains(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    center = (values * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    return values - center


def compute_velocity_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    config: VelocityLossConfig | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Velocity loss with optional legacy relative-stem-gain supervision."""

    if config is None:
        config = VelocityLossConfig()
    note_mask = batch["note_mask"].bool()
    gain_mask = batch["stem_gain_mask"].bool()
    gain_enabled = config.stem_gain_weight > 0.0
    logits = outputs["velocity_logits"]
    target_velocity = batch["target_velocity"].long()
    zero = _zero_from(outputs)

    if note_mask.any():
        active_logits = logits[note_mask]
        active_target = target_velocity[note_mask]
        if int(active_target.min()) < 1 or int(active_target.max()) > 127:
            raise ValueError("target_velocity must be within 1..127")
        velocity_ce = F.cross_entropy(
            active_logits,
            active_target - 1,
            label_smoothing=config.label_smoothing,
        )
        expected = outputs["velocity_expected"][note_mask]
        velocity_expected_loss = F.smooth_l1_loss(
            (expected - 1.0) / 126.0,
            (active_target.float() - 1.0) / 126.0,
            beta=0.05,
        )
        velocity_mae = (expected - active_target.float()).abs().mean()
        predicted = logits[note_mask].argmax(dim=-1) + 1
        within_5 = ((predicted - active_target).abs() <= 5).float().mean()
        within_10 = ((predicted - active_target).abs() <= 10).float().mean()
    else:
        velocity_ce = zero
        velocity_expected_loss = zero
        velocity_mae = zero.detach()
        within_5 = zero.detach()
        within_10 = zero.detach()

    if gain_enabled and "stem_gain_db" not in outputs:
        raise ValueError(
            "stem_gain_weight is positive but the model has no stem-gain head"
        )
    if gain_enabled and gain_mask.any():
        predicted_gain = _center_stem_gains(outputs["stem_gain_db"], gain_mask)
        target_gain = _center_stem_gains(batch["stem_gain_db"].float(), gain_mask)
        gain_errors = F.smooth_l1_loss(
            predicted_gain,
            target_gain,
            reduction="none",
            beta=config.gain_huber_delta_db,
        )
        stem_gain_loss = _masked_mean(gain_errors, gain_mask)
        stem_gain_mae_db = _masked_mean(
            (predicted_gain - target_gain).abs(), gain_mask
        )
    else:
        stem_gain_loss = zero
        stem_gain_mae_db = zero.detach()

    loss = (
        config.velocity_ce_weight * velocity_ce
        + config.velocity_expected_weight * velocity_expected_loss
        + config.stem_gain_weight * stem_gain_loss
    )
    metrics = {
        "loss": loss.detach(),
        "velocity_ce": velocity_ce.detach(),
        "velocity_expected_loss": velocity_expected_loss.detach(),
        "velocity_mae": velocity_mae.detach(),
        "velocity_within_5": within_5.detach(),
        "velocity_within_10": within_10.detach(),
        "stem_gain_loss": stem_gain_loss.detach(),
        "stem_gain_mae_db": stem_gain_mae_db.detach(),
        "note_count": note_mask.sum().detach(),
        "stem_gain_count": (
            gain_mask.sum().detach()
            if gain_enabled
            else torch.zeros((), dtype=torch.long, device=gain_mask.device)
        ),
    }
    return loss, metrics
