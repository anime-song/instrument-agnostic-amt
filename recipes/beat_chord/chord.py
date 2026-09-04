from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ChordConfig:
    chord_boundary_weight: float = 3.0
    root_chord_weight: float = 1.0
    bass_weight: float = 1.0
    key_boundary_weight: float = 3.0
    key_weight: float = 1.0
    chord_pitch_weight: float = 10.0
    boundary_pos_weight: float = 5.0
    key_boundary_pos_weight: float = 150.0
    chord_boundary_loss_tolerance: int = 1
    key_boundary_loss_tolerance: int = 8
    focal_tversky_alpha: float = 0.3
    focal_tversky_gamma: float = 1.5

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if isinstance(value, (int, float)) and value < 0:
                raise ValueError(f"{field} must be non-negative")


def chord_config_from_args(args: Any) -> ChordConfig:
    return ChordConfig(
        chord_boundary_loss_tolerance=int(
            getattr(args, "chord_boundary_loss_tolerance", 1)
        ),
        key_boundary_loss_tolerance=int(
            getattr(args, "key_boundary_loss_tolerance", 8)
        ),
    )


class BalancedSoftmaxLoss(nn.Module):
    def __init__(
        self, class_counts: torch.Tensor, tau: float = 1.0, ignore_index: int = -100
    ) -> None:
        super().__init__()
        log_prior = torch.log(torch.clamp(class_counts, min=1.0))
        self.register_buffer("log_prior", log_prior)
        self.tau = tau
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))
            labels = labels.reshape(-1)
        valid = labels != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        logits, labels = logits[valid], labels[valid]
        adjusted_logits = logits + self.tau * self.log_prior
        return F.cross_entropy(adjusted_logits, labels)


class SafeCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[valid], labels[valid])


class ShiftTolerantBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0, tolerance: int = 1) -> None:
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.tolerance = tolerance

    def spread(self, values: torch.Tensor, factor: int = 1) -> torch.Tensor:
        if self.tolerance == 0:
            return values
        return F.max_pool1d(values, 1 + 2 * factor * self.tolerance, 1)

    def crop(self, values: torch.Tensor, factor: int = 1) -> torch.Tensor:
        trim = factor * self.tolerance
        if trim == 0:
            return values
        return values[..., trim : -trim or None]

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        if targets.dim() == 2:
            targets = targets.unsqueeze(1)

        spread_logits = self.crop(self.spread(logits))
        cropped_targets = self.crop(targets, factor=2)
        weights = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            weights = weights * self.crop(mask, factor=2)

        return F.binary_cross_entropy_with_logits(
            spread_logits,
            cropped_targets,
            weight=weights,
            pos_weight=self.pos_weight,
        )


class FocalTverskyLoss(nn.Module):
    def __init__(
        self, alpha: float = 0.3, gamma: float = 1.5, smooth: float = 1e-6
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        probabilities = torch.sigmoid(logits)
        if mask is not None:
            if mask.dim() == targets.dim() - 1:
                mask = mask.unsqueeze(-1)
            mask = mask.to(device=logits.device, dtype=logits.dtype)
            probabilities = probabilities * mask
            targets = targets * mask
        true_positives = (targets * probabilities).sum(dim=(0, 1))
        false_positives = ((1 - targets) * probabilities).sum(dim=(0, 1))
        false_negatives = (targets * (1 - probabilities)).sum(dim=(0, 1))
        score = (true_positives + self.smooth) / (
            true_positives
            + self.alpha * false_positives
            + (1 - self.alpha) * false_negatives
            + self.smooth
        )
        return torch.pow(1 - score.mean(), self.gamma)


class ChordLoss(nn.Module):
    def __init__(self, config: ChordConfig, root_chord_counts: torch.Tensor) -> None:
        super().__init__()
        self.config = config
        self.chord_bce = ShiftTolerantBCELoss(
            pos_weight=config.boundary_pos_weight,
            tolerance=config.chord_boundary_loss_tolerance,
        )
        self.key_bce = ShiftTolerantBCELoss(
            pos_weight=config.key_boundary_pos_weight,
            tolerance=config.key_boundary_loss_tolerance,
        )
        self.rc_loss = BalancedSoftmaxLoss(root_chord_counts, tau=0.3)
        self.ce_loss = SafeCrossEntropyLoss(ignore_index=-100)
        self.ft_loss = FocalTverskyLoss(
            alpha=config.focal_tversky_alpha,
            gamma=config.focal_tversky_gamma,
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        chord_boundary_mask = batch.get("chord_boundary_mask")
        if chord_boundary_mask is not None:
            chord_boundary_mask = chord_boundary_mask.to(
                outputs["chord_boundary_logits"].device
            )
        key_boundary_mask = batch.get("key_boundary_mask")
        if key_boundary_mask is not None:
            key_boundary_mask = key_boundary_mask.to(
                outputs["key_boundary_logits"].device
            )
        chord_pitch_mask = batch.get("chord_pitch_mask")
        if chord_pitch_mask is not None:
            chord_pitch_mask = chord_pitch_mask.to(outputs["chord_pitch_logits"].device)

        chord_boundary_loss = self.chord_bce(
            outputs["chord_boundary_logits"],
            batch["chord_boundary"].to(outputs["chord_boundary_logits"].device),
            chord_boundary_mask,
        )
        root_chord_loss = self.rc_loss(
            outputs["root_chord_logits"],
            batch["root_chord_targets"].to(outputs["root_chord_logits"].device),
        )
        bass_loss = self.ce_loss(
            outputs["bass_logits"],
            batch["bass_targets"].to(outputs["bass_logits"].device),
        )
        key_boundary_loss = self.key_bce(
            outputs["key_boundary_logits"],
            batch["key_boundary"].to(outputs["key_boundary_logits"].device),
            key_boundary_mask,
        )
        key_loss = self.ce_loss(
            outputs["key_logits"],
            batch["key_targets"].to(outputs["key_logits"].device),
        )
        chord_pitch_loss = self.ft_loss(
            outputs["chord_pitch_logits"],
            batch["chord_pitch_targets"].to(outputs["chord_pitch_logits"].device),
            chord_pitch_mask,
        )

        total = (
            chord_boundary_loss * self.config.chord_boundary_weight
            + root_chord_loss * self.config.root_chord_weight
            + bass_loss * self.config.bass_weight
            + key_boundary_loss * self.config.key_boundary_weight
            + key_loss * self.config.key_weight
            + chord_pitch_loss * self.config.chord_pitch_weight
        )
        return total, {
            "chord_total": total,
            "chord_boundary": chord_boundary_loss,
            "root_chord": root_chord_loss,
            "bass": bass_loss,
            "key_boundary": key_boundary_loss,
            "key": key_loss,
            "chord_pitch": chord_pitch_loss,
        }
