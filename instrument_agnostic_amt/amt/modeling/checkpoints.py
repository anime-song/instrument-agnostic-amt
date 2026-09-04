"""Core AMT checkpoint loading and compatibility helpers."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch

from .model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
    remap_legacy_v1_state_dict,
)

V1_BACKBONE_CONFIG_FIELDS = (
    "sample_rate",
    "hop_length",
    "n_fft",
    "cqt_fmin",
    "cqt_n_bins",
    "cqt_bins_per_octave",
    "cqt_filter_scale",
    "harmonics",
    "cqt_log_scale",
    "input_audio_channels",
    "hidden_size",
    "base_ch",
    "encoder_num_layers",
    "encoder_num_heads",
    "dropout",
    "pitch_query_count",
    "use_beat_head",
    "use_chord_head",
)


@dataclass(frozen=True)
class CheckpointLoadReport:
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]

    @property
    def loaded_backbone_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self.loaded_keys if key.startswith("backbone."))


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def extract_model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    raw_config = checkpoint.get("model_config")
    run_config = checkpoint.get("config")
    if raw_config is None and isinstance(run_config, Mapping):
        raw_config = run_config.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint does not contain model_config")
    return dict(raw_config)


def extract_training_args(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    run_config = checkpoint.get("config")
    if not isinstance(run_config, Mapping):
        return {}
    args = run_config.get("args")
    return dict(args) if isinstance(args, Mapping) else {}


def coerce_model_config(
    raw_config: Mapping[str, Any],
    *,
    for_inference: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> SemiCRFModelConfig:
    allowed = {field.name for field in fields(SemiCRFModelConfig)}
    values = {key: value for key, value in raw_config.items() if key in allowed}
    # V1 predates the discriminator. Old V2 checkpoints are intentionally unsupported.
    if "semi_crf_version" not in values:
        if "instrument_pair_gate_dim" in raw_config and "num_pitch_slots" not in raw_config:
            raise ValueError(
                "The previous V2 architecture is not compatible with this implementation. "
                "Use a V1 checkpoint or a new selectable-head checkpoint."
            )
        values["semi_crf_version"] = "v1"
    if overrides:
        values.update({key: value for key, value in overrides.items() if key in allowed})
    if for_inference:
        values["use_gradient_checkpoint"] = False
        values["spec_augment_params"] = None
    return SemiCRFModelConfig(**values)


def select_state_dict(
    checkpoint: Mapping[str, Any], *, prefer_ema: bool
) -> Mapping[str, torch.Tensor]:
    state_dict = checkpoint.get("ema_state_dict") if prefer_ema else None
    if state_dict is None:
        state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a model state dict")
    return state_dict


def load_compatible_weights(
    model: AudioSemiCRFTransformer,
    checkpoint: Mapping[str, Any],
    *,
    prefer_ema: bool = False,
    require_complete_backbone: bool = True,
) -> CheckpointLoadReport:
    source = remap_legacy_v1_state_dict(
        select_state_dict(checkpoint, prefer_ema=prefer_ema)
    )
    target = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatches: list[str] = []
    for key, value in source.items():
        if key not in target:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(value.shape)} model={tuple(target[key].shape)}"
            )
            continue
        compatible[key] = value

    required_backbone = {key for key in target if key.startswith("backbone.")}
    loaded_backbone = {key for key in compatible if key.startswith("backbone.")}
    missing_backbone = sorted(required_backbone - loaded_backbone)
    if require_complete_backbone and missing_backbone:
        details = ", ".join(missing_backbone[:8])
        if len(missing_backbone) > 8:
            details += f", ... (+{len(missing_backbone) - 8})"
        raise ValueError(f"Checkpoint is not V1-backbone compatible; missing: {details}")

    incompatible = model.load_state_dict(compatible, strict=False)
    loaded = tuple(sorted(compatible))
    if model.semi_crf_version == "v1":
        model._use_interval_instrument_head = any(
            key.startswith("head.interval_instrument_predictor.") for key in loaded
        )
    return CheckpointLoadReport(
        loaded_keys=loaded,
        missing_keys=tuple(sorted(incompatible.missing_keys)),
        unexpected_keys=tuple(sorted(unexpected)),
        shape_mismatches=tuple(sorted(shape_mismatches)),
    )



