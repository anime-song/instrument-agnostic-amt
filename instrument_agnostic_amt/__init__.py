"""Instrument-agnostic AMT package."""

from .transcription import (
    DecodedAudio,
    MidiExportOptions,
    Transcriber,
    TranscriberBusyError,
    TranscriptionModelInfo,
    TranscriptionOptions,
    TranscriptionResult,
)
from .velocity.estimator import (
    VelocityEstimator,
    VelocityEstimatorBusyError,
    VelocityModelInfo,
    VelocityOptions,
    VelocityResult,
)

__all__ = [
    "DecodedAudio",
    "MidiExportOptions",
    "Transcriber",
    "TranscriberBusyError",
    "TranscriptionModelInfo",
    "TranscriptionOptions",
    "TranscriptionResult",
    "VelocityEstimator",
    "VelocityEstimatorBusyError",
    "VelocityModelInfo",
    "VelocityOptions",
    "VelocityResult",
]
