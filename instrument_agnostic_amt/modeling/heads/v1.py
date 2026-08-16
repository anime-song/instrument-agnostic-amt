from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from .common import IntervalBoundaryPredictor, IntervalScorer, TaskFeatureAdapter
from .interval_boundaries import (
    gather_interval_endpoint_features,
    gather_interval_sequence_features,
)


class IntervalInstrumentPredictor(nn.Module):
    def __init__(
        self, *, input_dim: int, num_instrument_classes: int, dropout: float
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_instrument_classes = int(num_instrument_classes)
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim * 4 + 1),
            nn.Linear(self.input_dim * 4 + 1, self.input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.input_dim, self.num_instrument_classes),
        )

    def forward(self, interval_features: torch.Tensor) -> torch.Tensor:
        return self.net(interval_features)


class V1SemiCRFHead(nn.Module):
    """The original pitch/slot Semi-CRF and instrument prediction heads."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_pitch_slots: int,
        semi_crf_head_dim: int,
        num_instrument_classes: int,
        dropout: float,
        use_interval_boundary_head: bool,
    ) -> None:
        super().__init__()
        if num_pitch_slots <= 0:
            raise ValueError("num_pitch_slots must be positive")
        self.num_pitch_slots = int(num_pitch_slots)
        self.interval_adapter = TaskFeatureAdapter(input_dim, dropout)
        self.instrument_adapter = TaskFeatureAdapter(input_dim, dropout)
        self.slot_embedding = nn.Embedding(self.num_pitch_slots, input_dim)
        self.interval_scorer = IntervalScorer(input_dim, semi_crf_head_dim)
        self.interval_boundary_predictor = (
            IntervalBoundaryPredictor(input_dim, dropout)
            if use_interval_boundary_head
            else None
        )
        self.interval_instrument_predictor = IntervalInstrumentPredictor(
            input_dim=input_dim,
            num_instrument_classes=num_instrument_classes,
            dropout=dropout,
        )
        self.instrument_classifier = nn.Linear(input_dim, num_instrument_classes)

    def _expand_with_slots(self, features: torch.Tensor) -> torch.Tensor:
        if self.num_pitch_slots == 1:
            return features
        batch_size, time_steps, num_pitches, feature_dim = features.shape
        expanded = features.unsqueeze(3) + self.slot_embedding.weight.view(
            1, 1, 1, self.num_pitch_slots, feature_dim
        )
        return expanded.reshape(
            batch_size,
            time_steps,
            num_pitches * self.num_pitch_slots,
            feature_dim,
        )

    def forward(
        self,
        pitch_features: torch.Tensor,
        frame_valid_mask: torch.Tensor,
        *,
        include_frame_instrument_logits: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        interval_features = self._expand_with_slots(
            self.interval_adapter(pitch_features)
        )
        pitch_instrument_features = self.instrument_adapter(pitch_features)
        instrument_features = self._expand_with_slots(pitch_instrument_features)
        interval_query, interval_key, interval_diag = self.interval_scorer(
            interval_features
        )
        return {
            "interval_query": interval_query,
            "interval_key": interval_key,
            "interval_diag": interval_diag,
            "interval_features": interval_features,
            "instrument_features": instrument_features,
            "instrument_logits": (
                self.instrument_classifier(pitch_instrument_features)
                if include_frame_instrument_logits
                else None
            ),
            "frame_valid_mask": frame_valid_mask,
        }

    def supports_interval_boundaries(self) -> bool:
        return self.interval_boundary_predictor is not None

    def supports_interval_instruments(self) -> bool:
        return True

    def predict_interval_boundaries(
        self,
        features: torch.Tensor,
        interval_batch: Sequence[Sequence[Sequence[tuple[int, int]]]],
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int, int]]]:
        if self.interval_boundary_predictor is None:
            return features.new_zeros(
                (0, 4),
                dtype=features.dtype if compute_dtype is None else compute_dtype,
            ), []
        interval_features, entries = gather_interval_endpoint_features(
            features,
            interval_batch,
            compute_dtype=compute_dtype,
        )
        if not entries:
            return interval_features.new_zeros((0, 4)), []
        return self.interval_boundary_predictor(interval_features), entries

    def predict_interval_instruments(
        self,
        features: torch.Tensor,
        interval_batch: Sequence[Sequence[Sequence[tuple[int, int]]]],
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int, int]]]:
        interval_features, entries = gather_interval_sequence_features(
            features,
            interval_batch,
            compute_dtype=compute_dtype,
        )
        if not entries:
            return interval_features.new_zeros(
                (0, self.interval_instrument_predictor.num_instrument_classes)
            ), []
        return self.interval_instrument_predictor(interval_features), entries
