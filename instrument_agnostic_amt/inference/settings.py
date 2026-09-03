from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Protocol

from ..modeling.model import SemiCRFModelConfig
from .types import InferenceSettings


class InferenceOptionValues(Protocol):
    window_ms: int | None
    stride_ms: int | None
    window_batch_size: int
    semi_crf_track_batch_size: int | None
    semi_crf_backend: str
    semi_crf_sparse_decode: bool
    semi_crf_sparse_topk_per_start: int
    semi_crf_sparse_score_threshold: float | None
    semi_crf_sparse_max_span_ms: float | None
    instrument_pair_infer_topk: int
    instrument_pair_gate_threshold: float
    instrument_pair_max_pairs: int
    merge_gap_ms: float | None
    merge_onset_ms: float
    silence_gate_rms_dbfs: float | None
    note_bias: float
    use_boundary_head: bool
    show_progress: bool


def resolve_inference_settings(
    config: SemiCRFModelConfig,
    checkpoint_args: Mapping[str, object],
    options: InferenceOptionValues,
    *,
    allowed_instrument_ids: tuple[int, ...] | None,
) -> InferenceSettings:
    """Resolve checkpoint defaults and typed call options without CLI state."""

    default_window_ms = int(checkpoint_args.get("window_ms") or 8000)
    window_ms = (
        int(options.window_ms)
        if options.window_ms is not None
        else default_window_ms
    )
    stride_ms = (
        int(options.stride_ms)
        if options.stride_ms is not None
        else max(1, window_ms // 2)
    )
    track_batch_size = (
        int(options.semi_crf_track_batch_size)
        if options.semi_crf_track_batch_size is not None
        else int(checkpoint_args.get("semi_crf_track_batch_size") or 128)
    )
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if stride_ms <= 0:
        raise ValueError("stride_ms must be positive")
    if track_batch_size <= 0:
        raise ValueError("semi_crf_track_batch_size must be positive")

    sparse_max_span_frames = None
    if options.semi_crf_sparse_max_span_ms is not None:
        sparse_max_span_frames = max(
            1,
            math.ceil(
                float(options.semi_crf_sparse_max_span_ms)
                * float(config.sample_rate)
                / 1000.0
                / float(config.hop_length)
            ),
        )
    if round(window_ms * int(config.sample_rate) / 1000.0) < int(
        config.n_fft
    ):
        raise ValueError(f"window_ms={window_ms} is too short for n_fft={config.n_fft}")

    return InferenceSettings(
        window_ms=window_ms,
        stride_ms=stride_ms,
        track_batch_size=track_batch_size,
        window_batch_size=int(options.window_batch_size),
        merge_gap_ms=options.merge_gap_ms,
        merge_onset_ms=float(options.merge_onset_ms),
        silence_gate_rms_dbfs=options.silence_gate_rms_dbfs,
        note_bias=float(options.note_bias),
        disable_tqdm=not bool(options.show_progress),
        use_boundary_head=bool(options.use_boundary_head),
        instrument_probability_mode=(
            "softmax"
            if checkpoint_args.get("instrument_loss_type") == "ce"
            else "sigmoid"
        ),
        semi_crf_backend=str(options.semi_crf_backend),
        semi_crf_sparse_decode=bool(options.semi_crf_sparse_decode),
        semi_crf_sparse_topk_per_start=int(
            options.semi_crf_sparse_topk_per_start
        ),
        semi_crf_sparse_score_threshold=options.semi_crf_sparse_score_threshold,
        semi_crf_sparse_max_span_frames=sparse_max_span_frames,
        instrument_pair_infer_topk=int(options.instrument_pair_infer_topk),
        instrument_pair_gate_threshold=float(
            options.instrument_pair_gate_threshold
        ),
        instrument_pair_max_pairs=int(options.instrument_pair_max_pairs),
        allowed_instrument_ids=allowed_instrument_ids,
    )
