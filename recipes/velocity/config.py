"""Configuration for the velocity data-preparation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class PseudoLabelConfig:
    """Parameters for inexpensive waveform-derived note-strength labels."""

    frame_ms: float = 10.0
    onset_signal_start_ms: float = 15.0
    onset_signal_end_ms: float = 140.0
    onset_noise_start_ms: float = -140.0
    onset_noise_end_ms: float = -20.0
    collision_window_ms: float = 45.0
    min_note_duration_ms: float = 25.0
    min_signal_dbfs: float = -72.0
    min_snr_db: float = 3.0
    full_confidence_snr_db: float = 24.0
    pseudo_velocity_min: int = 32
    pseudo_velocity_max: int = 112
    min_rank_notes: int = 4

    def __post_init__(self) -> None:
        if self.frame_ms <= 0.0:
            raise ValueError("frame_ms must be positive")
        if self.onset_signal_end_ms <= self.onset_signal_start_ms:
            raise ValueError("the onset signal window must have positive length")
        if self.onset_noise_end_ms <= self.onset_noise_start_ms:
            raise ValueError("the onset noise window must have positive length")
        if self.full_confidence_snr_db <= self.min_snr_db:
            raise ValueError("full_confidence_snr_db must exceed min_snr_db")
        if not 1 <= self.pseudo_velocity_min <= self.pseudo_velocity_max <= 127:
            raise ValueError("pseudo velocity bounds must be within MIDI 1..127")
        if self.min_rank_notes < 1:
            raise ValueError("min_rank_notes must be positive")


@dataclass(frozen=True)
class VelocityPipelineConfig:
    """Shared configuration for data preparation and target-SF2 calibration."""

    render_sample_rate: int = 44_100
    canonical_velocity: int = 80
    sweep_velocities: tuple[int, ...] = (16, 32, 48, 64, 80, 96, 112, 127)
    sweep_pitches: tuple[int, ...] = (36, 48, 60, 72, 84)
    sweep_drum_pitches: tuple[int, ...] = (35, 36, 38, 40, 42, 46, 49, 51, 57)
    sweep_note_seconds: float = 0.5
    sweep_gap_seconds: float = 0.15
    pseudo_labels: PseudoLabelConfig = PseudoLabelConfig()

    def __post_init__(self) -> None:
        if self.render_sample_rate <= 0:
            raise ValueError("render_sample_rate must be positive")
        if not 1 <= self.canonical_velocity <= 127:
            raise ValueError("canonical_velocity must be within MIDI 1..127")
        if not self.sweep_velocities or any(
            value < 1 or value > 127 for value in self.sweep_velocities
        ):
            raise ValueError("sweep_velocities must contain MIDI velocities 1..127")
        if not self.sweep_pitches or any(
            value < 0 or value > 127 for value in self.sweep_pitches
        ):
            raise ValueError("sweep_pitches must contain MIDI pitches 0..127")
        if not self.sweep_drum_pitches or any(
            value < 0 or value > 127 for value in self.sweep_drum_pitches
        ):
            raise ValueError("sweep_drum_pitches must contain MIDI pitches 0..127")
        if self.sweep_note_seconds <= 0.0 or self.sweep_gap_seconds < 0.0:
            raise ValueError("invalid sweep note/gap duration")


def _known_dataclass_values(
    dataclass_type: type,
    values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    values = dict(values or {})
    known = {field.name for field in fields(dataclass_type)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(
            f"Unknown {dataclass_type.__name__} config key(s): {', '.join(unknown)}"
        )
    return values


def load_pipeline_config(path: str | Path) -> VelocityPipelineConfig:
    """Load a strict YAML configuration for the velocity sub-pipeline."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("velocity config root must be a mapping")

    values = _known_dataclass_values(VelocityPipelineConfig, raw)
    pseudo_values = _known_dataclass_values(
        PseudoLabelConfig,
        values.pop("pseudo_labels", None),
    )
    if "sweep_velocities" in values:
        values["sweep_velocities"] = tuple(
            int(value) for value in values["sweep_velocities"]
        )
    if "sweep_pitches" in values:
        values["sweep_pitches"] = tuple(int(value) for value in values["sweep_pitches"])
    if "sweep_drum_pitches" in values:
        values["sweep_drum_pitches"] = tuple(
            int(value) for value in values["sweep_drum_pitches"]
        )
    values["pseudo_labels"] = PseudoLabelConfig(**pseudo_values)
    return VelocityPipelineConfig(**values)
