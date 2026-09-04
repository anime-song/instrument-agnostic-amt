"""Types shared by core AMT inference stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictedNote:
    instrument_id: int
    pitch: int
    start_sample: int
    end_sample: int
    velocity: int
    slot_index: int = 0
    has_onset: bool = True
    has_offset: bool = True
    instrument_candidates: tuple[int, ...] = ()


@dataclass(frozen=True)
class InferenceSettings:
    window_ms: int
    stride_ms: int
    track_batch_size: int
    window_batch_size: int
    merge_gap_ms: float | None
    merge_onset_ms: float
    silence_gate_rms_dbfs: float | None
    note_bias: float
    disable_tqdm: bool
    use_boundary_head: bool = True
    instrument_probability_mode: str = "sigmoid"
    semi_crf_backend: str = "torch"
    semi_crf_sparse_decode: bool = False
    semi_crf_sparse_topk_per_start: int = 16
    semi_crf_sparse_score_threshold: float | None = None
    semi_crf_sparse_max_span_frames: int | None = None
    instrument_pair_infer_topk: int = 256
    instrument_pair_gate_threshold: float = -3.0
    instrument_pair_max_pairs: int = 512
    allowed_instrument_ids: tuple[int, ...] | None = None
