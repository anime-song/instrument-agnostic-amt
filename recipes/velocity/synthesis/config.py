"""Configuration for synthetic velocity training data."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class SyntheticDataConfig:
    """Sampling and rendering settings for synthetic velocity examples."""

    canonical_velocity: int = 80
    velocity_min: int = 16
    velocity_max: int = 127
    velocity_center_min: float = 64.0
    velocity_center_max: float = 100.0
    velocity_span_min: float = 36.0
    velocity_span_max: float = 80.0
    velocity_jitter_std: float = 4.0
    independent_velocity_probability: float = 0.05
    gain_jitter_std_db: float = 2.5
    gain_clip_db: float = 18.0
    master_gain_min_db: float = -12.0
    master_gain_max_db: float = 0.0
    render_sample_rate: int = 22_050
    render_synth_gain: float = 0.5
    mixture_sample_rate: int = 22_050
    mix_peak_limit_dbfs: float = -1.0
    use_gain_augmentation: bool = False

    def __post_init__(self) -> None:
        if not 1 <= self.canonical_velocity <= 127:
            raise ValueError("canonical_velocity must be within MIDI 1..127")
        if not 1 <= self.velocity_min <= self.velocity_max <= 127:
            raise ValueError("velocity bounds must be within MIDI 1..127")
        if self.velocity_center_max < self.velocity_center_min:
            raise ValueError("invalid velocity center range")
        if (
            self.velocity_span_min < 0.0
            or self.velocity_span_max < self.velocity_span_min
        ):
            raise ValueError("invalid velocity span range")
        if self.velocity_jitter_std < 0.0:
            raise ValueError("velocity_jitter_std must be nonnegative")
        if not 0.0 <= self.independent_velocity_probability <= 1.0:
            raise ValueError("independent_velocity_probability must be within 0..1")
        if self.gain_jitter_std_db < 0.0 or self.gain_clip_db <= 0.0:
            raise ValueError("invalid stem gain settings")
        if self.master_gain_max_db < self.master_gain_min_db:
            raise ValueError("invalid master gain range")
        if self.render_sample_rate <= 0 or self.render_synth_gain <= 0.0:
            raise ValueError("invalid render settings")
        if self.mixture_sample_rate <= 0:
            raise ValueError("mixture_sample_rate must be positive")
        if self.mix_peak_limit_dbfs > 0.0:
            raise ValueError("mix_peak_limit_dbfs must not exceed 0 dBFS")


def load_synthetic_config(path: str | Path) -> SyntheticDataConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, Mapping):
        raise ValueError("synthetic config root must be an object")
    known = {field.name for field in fields(SyntheticDataConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown synthetic config key(s): {', '.join(unknown)}")
    values: dict[str, Any] = dict(raw)
    return SyntheticDataConfig(**values)
