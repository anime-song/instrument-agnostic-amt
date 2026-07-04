from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictedNote:
    pitch: int
    start_sample: int
    end_sample: int
    velocity: int
    slot_index: int = 0
    has_onset: bool = True
    has_offset: bool = True


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
