from __future__ import annotations

from typing import Sequence

import einops
import torch
import torch.nn as nn

from .common import IntervalBoundaryPredictor, IntervalScorer, TaskFeatureAdapter


class InstrumentPairGate(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        gate_dim: int,
        num_instruments: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if gate_dim <= 0:
            raise ValueError("gate_dim must be positive")
        self.pitch_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, gate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_dim, gate_dim),
        )
        self.instrument_proj = nn.Linear(input_dim, gate_dim, bias=False)
        self.instrument_bias = nn.Parameter(torch.zeros(num_instruments))
        self.scale = float(gate_dim) ** -0.5

    def forward(
        self,
        pitch_features: torch.Tensor,
        instrument_embeddings: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = frame_valid_mask.to(dtype=pitch_features.dtype).unsqueeze(-1).unsqueeze(-1)
        pooled = (pitch_features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pitch_gate = self.pitch_proj(pooled)
        instrument_gate = self.instrument_proj(instrument_embeddings)
        logits = torch.einsum("bpg,ig->bip", pitch_gate, instrument_gate) * self.scale
        return logits + einops.rearrange(self.instrument_bias, "i -> 1 i 1")


class V2OverlapSemiCRFHead(nn.Module):
    """Independent instrument-pitch Semi-CRFs, allowing cross-instrument overlap."""

    def __init__(
        self,
        *,
        input_dim: int,
        semi_crf_head_dim: int,
        instrument_pair_gate_dim: int,
        num_instrument_classes: int,
        dropout: float,
        use_interval_boundary_head: bool,
    ) -> None:
        super().__init__()
        self.num_instrument_classes = int(num_instrument_classes)
        self.interval_adapter = TaskFeatureAdapter(input_dim, dropout)
        self.instrument_embedding = nn.Embedding(num_instrument_classes, input_dim)
        self.pair_gate = InstrumentPairGate(
            input_dim=input_dim,
            gate_dim=instrument_pair_gate_dim,
            num_instruments=num_instrument_classes,
            dropout=dropout,
        )
        self.interval_scorer = IntervalScorer(input_dim, semi_crf_head_dim)
        self.interval_boundary_predictor = (
            IntervalBoundaryPredictor(input_dim, dropout)
            if use_interval_boundary_head
            else None
        )

    def forward(
        self,
        pitch_features: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        interval_features = self.interval_adapter(pitch_features)
        return {
            "interval_features": interval_features,
            "pair_gate_logits": self.pair_gate(
                interval_features,
                self.instrument_embedding.weight,
                frame_valid_mask,
            ),
            "frame_valid_mask": frame_valid_mask,
        }

    def supports_interval_boundaries(self) -> bool:
        return self.interval_boundary_predictor is not None

    def build_pair_interval_features(
        self,
        interval_features: torch.Tensor,
        selected_pair_ids: Sequence[Sequence[int] | torch.Tensor],
        *,
        num_pitches: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, time_steps, _, feature_dim = interval_features.shape
        if len(selected_pair_ids) != int(batch_size):
            raise ValueError("selected_pair_ids length must match batch size")

        feature_chunks: list[torch.Tensor] = []
        batch_indices: list[torch.Tensor] = []
        pair_chunks: list[torch.Tensor] = []
        for batch_index, values in enumerate(selected_pair_ids):
            pair_ids = (
                values.to(device=interval_features.device, dtype=torch.long)
                if isinstance(values, torch.Tensor)
                else torch.tensor(
                    list(values), device=interval_features.device, dtype=torch.long
                )
            )
            if pair_ids.numel() == 0:
                continue
            instrument_ids = torch.div(pair_ids, num_pitches, rounding_mode="floor")
            pitch_ids = pair_ids.remainder(num_pitches)
            if torch.any(instrument_ids < 0) or torch.any(
                instrument_ids >= self.num_instrument_classes
            ):
                raise ValueError("selected pair contains an invalid instrument id")
            pitch_features = interval_features[batch_index, :, pitch_ids, :]
            instrument_features = self.instrument_embedding(instrument_ids).unsqueeze(0)
            feature_chunks.append(pitch_features + instrument_features)
            batch_indices.append(
                torch.full(
                    (int(pair_ids.numel()),),
                    batch_index,
                    device=interval_features.device,
                    dtype=torch.long,
                )
            )
            pair_chunks.append(pair_ids)

        if not feature_chunks:
            return (
                interval_features.new_zeros((time_steps, 0, feature_dim)),
                torch.zeros((0,), device=interval_features.device, dtype=torch.long),
                torch.zeros((0,), device=interval_features.device, dtype=torch.long),
            )
        return (
            torch.cat(feature_chunks, dim=1).contiguous(),
            torch.cat(batch_indices),
            torch.cat(pair_chunks),
        )

    def score_pair_interval_features(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.interval_scorer(features)

    def predict_flat_interval_boundaries(
        self,
        features: torch.Tensor,
        interval_batch: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        if self.interval_boundary_predictor is None:
            return features.new_zeros((0, 4)), []
        entries = [
            (track_index, interval_index, int(begin), int(end))
            for track_index, intervals in enumerate(interval_batch)
            for interval_index, (begin, end) in enumerate(intervals)
        ]
        if not entries:
            return features.new_zeros((0, 4)), []
        device = features.device
        track = torch.tensor([entry[0] for entry in entries], device=device)
        begin = torch.tensor([entry[2] for entry in entries], device=device)
        end = torch.tensor([entry[3] for entry in entries], device=device)
        begin_features = features[begin, track]
        end_features = features[end, track]
        endpoints = torch.cat(
            [begin_features, end_features, begin_features * end_features], dim=-1
        )
        return self.interval_boundary_predictor(endpoints), entries

