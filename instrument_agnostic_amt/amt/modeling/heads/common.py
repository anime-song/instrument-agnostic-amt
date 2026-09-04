"""Layers shared by AMT prediction heads."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskFeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class IntervalScorer(nn.Module):
    def __init__(self, input_dim: int, head_dim: int) -> None:
        super().__init__()
        if input_dim <= 0 or head_dim <= 0:
            raise ValueError("interval scorer dimensions must be positive")
        self.head_dim = int(head_dim)
        self.proj = nn.Linear(input_dim, self.head_dim * 2 + 1)
        self.query_scale = 1.0 / math.sqrt(float(self.head_dim))

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.proj(features)
        return self._split_projection(projected)

    def project_factorized(
        self,
        pitch_features: torch.Tensor,
        instrument_features: torch.Tensor,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Project additive pitch/instrument features without forming every pair.

        The pitch projection includes the Linear bias exactly once.  The
        instrument projection intentionally omits it, so adding both projections
        is mathematically identical to ``self.proj(pitch + instrument)``.
        """

        pitch_projected = self.proj(pitch_features)
        instrument_projected = F.linear(
            instrument_features,
            self.proj.weight,
            bias=None,
        )
        return (
            self._split_projection(pitch_projected),
            self._split_projection(instrument_projected),
        )

    def _split_projection(
        self, projected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query, key, diagonal = torch.split(
            projected, [self.head_dim, self.head_dim, 1], dim=-1
        )
        return query * self.query_scale, key, diagonal.squeeze(-1)


class IntervalBoundaryPredictor(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 4),
        )

    def forward(self, interval_features: torch.Tensor) -> torch.Tensor:
        return self.net(interval_features)

