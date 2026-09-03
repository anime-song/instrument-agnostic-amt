from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch

from ...modeling.checkpoints import (
    extract_model_config,
    load_checkpoint,
    select_state_dict,
)
from ...modeling.model import remap_legacy_v1_state_dict
from .model import VelocityModelConfig, VelocityPredictionModel


@dataclass(frozen=True)
class BackboneLoadReport:
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


@dataclass(frozen=True)
class VelocityCheckpointLoadReport:
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


def _coerce_velocity_config(raw_config: object) -> VelocityModelConfig:
    if isinstance(raw_config, VelocityModelConfig):
        return raw_config
    if raw_config is None:
        return VelocityModelConfig()
    if not isinstance(raw_config, Mapping):
        raise TypeError("Velocity checkpoint config must be a mapping")
    allowed = {field.name for field in fields(VelocityModelConfig)}
    unknown = sorted(set(raw_config) - allowed)
    if unknown:
        raise ValueError(
            "Unknown velocity model config fields: " + ", ".join(unknown)
        )
    values = dict(raw_config)
    for key in ("harmonics", "local_frame_offsets"):
        if key in values and isinstance(values[key], list):
            values[key] = tuple(values[key])
    return VelocityModelConfig(**values)


def load_velocity_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[
    VelocityPredictionModel,
    VelocityModelConfig,
    VelocityCheckpointLoadReport,
]:
    """Load a complete velocity-only model from a trusted local checkpoint."""

    checkpoint = load_checkpoint(checkpoint_path)
    config = _coerce_velocity_config(
        checkpoint.get("config") or checkpoint.get("model_config")
    )
    model = VelocityPredictionModel(config)
    legacy_state_dict = checkpoint.get("state_dict")
    selected_state_dict = (
        legacy_state_dict
        if checkpoint.get("ema_state_dict") is None
        and checkpoint.get("model_state_dict") is None
        and isinstance(legacy_state_dict, Mapping)
        else select_state_dict(checkpoint, prefer_ema=True)
    )
    source = remap_legacy_v1_state_dict(selected_state_dict)
    target = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatches: list[str] = []
    for key, value in source.items():
        if key not in target or not isinstance(value, torch.Tensor):
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(value.shape)}, model={tuple(target[key].shape)}"
            )
            continue
        compatible[key] = value

    missing = sorted(set(target) - set(compatible))
    optional_prefixes = (
        ()
        if config.predict_stem_gain
        else ("global_audio_projection.", "stem_gain_head.")
    )
    missing_required = [
        key for key in missing if not key.startswith(optional_prefixes)
    ]
    required_shape_mismatches = [
        mismatch
        for mismatch in shape_mismatches
        if not mismatch.startswith(optional_prefixes)
    ]
    if missing_required or required_shape_mismatches:
        details = [
            *required_shape_mismatches,
            *(f"missing: {key}" for key in missing_required),
        ]
        preview = ", ".join(details[:8])
        if len(details) > 8:
            preview += f", ... (+{len(details) - 8})"
        raise ValueError(f"Velocity checkpoint is incomplete: {preview}")

    model.load_state_dict(compatible, strict=False)
    model.to(device)
    model.eval()
    return model, config, VelocityCheckpointLoadReport(
        loaded_keys=tuple(sorted(compatible)),
        missing_keys=tuple(missing),
        unexpected_keys=tuple(sorted(unexpected)),
        shape_mismatches=tuple(sorted(shape_mismatches)),
    )


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(value)
    return value


def _validate_backbone_config(
    model: VelocityPredictionModel,
    checkpoint: Mapping[str, Any],
) -> None:
    try:
        source = extract_model_config(checkpoint)
    except ValueError:
        return
    target = model.config
    comparisons = {
        "sample_rate": target.sample_rate,
        "hop_length": target.hop_length,
        "cqt_fmin": target.cqt_fmin,
        "cqt_n_bins": target.cqt_n_bins,
        "cqt_bins_per_octave": target.cqt_bins_per_octave,
        "cqt_filter_scale": target.cqt_filter_scale,
        "harmonics": target.harmonics,
        "cqt_log_scale": target.cqt_log_scale,
        "input_audio_channels": target.input_audio_channels,
        "hidden_size": target.hidden_size,
        "base_ch": target.base_ch,
        "encoder_num_layers": target.encoder_num_layers,
        "encoder_num_heads": target.encoder_num_heads,
        "pitch_query_count": target.pitch_query_count,
    }
    mismatches = []
    for key, target_value in comparisons.items():
        if key not in source:
            continue
        source_value = _normalize_config_value(source[key])
        if source_value != _normalize_config_value(target_value):
            mismatches.append(f"{key}: checkpoint={source_value!r}, model={target_value!r}")
    if mismatches:
        raise ValueError(
            "AMT checkpoint backbone config does not match velocity model: "
            + "; ".join(mismatches)
        )


def load_amt_backbone(
    model: VelocityPredictionModel,
    checkpoint_or_path: Mapping[str, Any] | str | Path,
    *,
    prefer_ema: bool = True,
    require_complete: bool = True,
) -> BackboneLoadReport:
    """Load only V1 backbone weights from an AMT checkpoint."""

    checkpoint = (
        load_checkpoint(checkpoint_or_path)
        if isinstance(checkpoint_or_path, (str, Path))
        else checkpoint_or_path
    )
    _validate_backbone_config(model, checkpoint)
    source = remap_legacy_v1_state_dict(
        select_state_dict(checkpoint, prefer_ema=prefer_ema)
    )
    target = model.backbone.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    shape_mismatches: list[str] = []
    for key, value in source.items():
        if not key.startswith("backbone."):
            continue
        backbone_key = key[len("backbone.") :]
        if backbone_key not in target:
            continue
        if tuple(value.shape) != tuple(target[backbone_key].shape):
            shape_mismatches.append(
                f"{backbone_key}: checkpoint={tuple(value.shape)}, "
                f"model={tuple(target[backbone_key].shape)}"
            )
            continue
        compatible[backbone_key] = value
    missing = sorted(set(target) - set(compatible))
    if require_complete and (missing or shape_mismatches):
        details = list(shape_mismatches) + [f"missing: {key}" for key in missing]
        preview = ", ".join(details[:8])
        if len(details) > 8:
            preview += f", ... (+{len(details) - 8})"
        raise ValueError(f"AMT backbone is not fully compatible: {preview}")
    model.backbone.load_state_dict(compatible, strict=False)
    return BackboneLoadReport(
        loaded_keys=tuple(sorted(compatible)),
        missing_keys=tuple(missing),
        shape_mismatches=tuple(sorted(shape_mismatches)),
    )
