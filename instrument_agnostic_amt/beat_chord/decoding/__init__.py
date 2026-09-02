"""Structured decoders for MIDI-frame beat/chord inference."""

from .audio_tempo import (
    TempoPrior,
    TempoPriorConfig,
    compute_pulse_curve,
    compute_tempo_prior,
    focused_tempo_prior,
)
from .beat_grid import (
    BeatGridDPConfig,
    BeatGridDecodeResult,
    MeterGridSegment,
    decode_beats_with_meter_grid_dp,
)
from .beat_refine import (
    TempoSegment,
    beat_to_quarter_ratio,
    fit_piecewise_constant_tempo,
    resample_metrical_level,
    rescale_meter,
    score_beat_grid,
    select_metrical_level,
    snap_to_curve,
)
from .legacy_grid import (
    build_grid_candidate,
    decode_beats_with_meter_grid,
    detect_peaks,
    log_softmax_numpy,
)

__all__ = [
    "BeatGridDPConfig",
    "BeatGridDecodeResult",
    "MeterGridSegment",
    "TempoPrior",
    "TempoPriorConfig",
    "TempoSegment",
    "beat_to_quarter_ratio",
    "build_grid_candidate",
    "decode_beats_with_meter_grid",
    "compute_pulse_curve",
    "compute_tempo_prior",
    "decode_beats_with_meter_grid_dp",
    "detect_peaks",
    "fit_piecewise_constant_tempo",
    "focused_tempo_prior",
    "log_softmax_numpy",
    "resample_metrical_level",
    "rescale_meter",
    "score_beat_grid",
    "select_metrical_level",
    "snap_to_curve",
]
