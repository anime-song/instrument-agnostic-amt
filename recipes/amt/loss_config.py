"""Configuration shared by the V1 and V2 AMT losses."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AMTLossConfig:
    semi_crf_loss_weight: float
    semi_crf_false_negative_cost: float
    semi_crf_false_positive_cost: float
    semi_crf_track_batch_size: int
    semi_crf_loss_backend: str
    interval_presence_loss_weight: float
    interval_offset_loss_weight: float
    instrument_loss_weight: float
    instrument_loss_type: str
    instrument_pair_train_topk: int
    instrument_pair_random_negatives: int
    instrument_pair_max_pairs: int
    pair_gate_loss_weight: float
