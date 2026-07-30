"""Structured decoders for MIDI-frame beat/chord inference."""

from .beat_grid import (
    BeatGridDPConfig,
    BeatGridDecodeResult,
    MeterGridSegment,
    decode_beats_with_meter_grid_dp,
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
    "build_grid_candidate",
    "decode_beats_with_meter_grid",
    "decode_beats_with_meter_grid_dp",
    "detect_peaks",
    "log_softmax_numpy",
]
