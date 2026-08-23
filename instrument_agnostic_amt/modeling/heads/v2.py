from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import einops
import torch
import torch.nn as nn

from .common import IntervalBoundaryPredictor, IntervalScorer, TaskFeatureAdapter


@dataclass(frozen=True)
class SelectedPairIndices:
    """Flat selected tracks mapped back to their batch, instrument, and pitch."""

    batch_indices: torch.Tensor
    instrument_indices: torch.Tensor
    pitch_indices: torch.Tensor
    pair_ids: torch.Tensor

    @property
    def track_count(self) -> int:
        return int(self.pair_ids.numel())


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
    """Factorized pitch/instrument scorer with independent pair Semi-CRFs."""

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
        pitch_projection, instrument_projection = (
            self.interval_scorer.project_factorized(
                interval_features,
                self.instrument_embedding.weight,
            )
        )
        pitch_query, pitch_key, pitch_diag = pitch_projection
        instrument_query, instrument_key, instrument_diag = instrument_projection
        return {
            "interval_features": interval_features,
            "pair_gate_logits": self.pair_gate(
                interval_features,
                self.instrument_embedding.weight,
                frame_valid_mask,
            ),
            "pitch_interval_query": pitch_query,
            "pitch_interval_key": pitch_key,
            "pitch_interval_diag": pitch_diag,
            "instrument_interval_query": instrument_query,
            "instrument_interval_key": instrument_key,
            "instrument_interval_diag": instrument_diag,
            "frame_valid_mask": frame_valid_mask,
        }

    def supports_interval_boundaries(self) -> bool:
        return self.interval_boundary_predictor is not None

    def build_selected_pair_indices(
        self,
        selected_pair_ids: Sequence[Sequence[int] | torch.Tensor],
        *,
        num_pitches: int,
    ) -> SelectedPairIndices:
        if num_pitches <= 0:
            raise ValueError("num_pitches must be positive")

        device = self.instrument_embedding.weight.device
        batch_indices: list[torch.Tensor] = []
        pair_chunks: list[torch.Tensor] = []
        for batch_index, values in enumerate(selected_pair_ids):
            pair_ids = (
                values.to(device=device, dtype=torch.long)
                if isinstance(values, torch.Tensor)
                else torch.tensor(
                    list(values), device=device, dtype=torch.long
                )
            )
            if pair_ids.numel() == 0:
                continue
            instrument_ids = torch.div(pair_ids, num_pitches, rounding_mode="floor")
            if torch.any(instrument_ids < 0) or torch.any(
                instrument_ids >= self.num_instrument_classes
            ):
                raise ValueError("selected pair contains an invalid instrument id")
            batch_indices.append(
                torch.full(
                    (int(pair_ids.numel()),),
                    batch_index,
                    device=device,
                    dtype=torch.long,
                )
            )
            pair_chunks.append(pair_ids)

        if not pair_chunks:
            empty = torch.zeros((0,), device=device, dtype=torch.long)
            return SelectedPairIndices(
                batch_indices=empty,
                instrument_indices=empty,
                pitch_indices=empty,
                pair_ids=empty,
            )
        flat_pair_ids = torch.cat(pair_chunks)
        return SelectedPairIndices(
            batch_indices=torch.cat(batch_indices),
            instrument_indices=torch.div(
                flat_pair_ids, num_pitches, rounding_mode="floor"
            ),
            pitch_indices=flat_pair_ids.remainder(num_pitches),
            pair_ids=flat_pair_ids,
        )

    def predict_flat_interval_boundaries(
        self,
        interval_features: torch.Tensor,
        selected_pairs: SelectedPairIndices,
        interval_batch: Sequence[Sequence[tuple[int, int]]],
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        output_dtype = (
            interval_features.dtype if compute_dtype is None else compute_dtype
        )
        if self.interval_boundary_predictor is None:
            return torch.zeros(
                (0, 4),
                device=interval_features.device,
                dtype=output_dtype,
            ), []
        if interval_features.dim() != 4:
            raise ValueError("interval_features must have shape [B, T, P, D]")
        if len(interval_batch) != selected_pairs.track_count:
            raise ValueError("interval_batch length must match selected pair count")
        entries = [
            (track_index, interval_index, int(begin), int(end))
            for track_index, intervals in enumerate(interval_batch)
            for interval_index, (begin, end) in enumerate(intervals)
        ]
        if not entries:
            return torch.zeros(
                (0, 4),
                device=interval_features.device,
                dtype=output_dtype,
            ), []

        # Only decoded/gold endpoints need instrument-conditioned features.  This
        # avoids materializing the full [T, selected-pair, D] tensor used by the
        # previous V2 implementation while preserving identical boundary logits.
        device = interval_features.device
        track = torch.tensor(
            [entry[0] for entry in entries], device=device, dtype=torch.long
        )
        begin = torch.tensor(
            [entry[2] for entry in entries], device=device, dtype=torch.long
        )
        end = torch.tensor(
            [entry[3] for entry in entries], device=device, dtype=torch.long
        )
        batch_indices = selected_pairs.batch_indices.index_select(0, track)
        instrument_indices = selected_pairs.instrument_indices.index_select(0, track)
        pitch_indices = selected_pairs.pitch_indices.index_select(0, track)
        instrument_features = self.instrument_embedding(instrument_indices)
        begin_features = interval_features[batch_indices, begin, pitch_indices]
        end_features = interval_features[batch_indices, end, pitch_indices]
        if compute_dtype is not None:
            begin_features = begin_features.to(dtype=compute_dtype)
            end_features = end_features.to(dtype=compute_dtype)
        begin_features = begin_features + instrument_features
        end_features = end_features + instrument_features
        endpoints = torch.cat(
            [begin_features, end_features, begin_features * end_features], dim=-1
        )
        return self.interval_boundary_predictor(endpoints), entries
