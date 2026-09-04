from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from ...amt.modeling.checkpoints import (
    extract_model_config,
    load_checkpoint,
    select_state_dict,
)
from ...amt.modeling.model import remap_legacy_v1_state_dict
from .model import VelocityPredictionModel


@dataclass(frozen=True)
class BackboneLoadReport:
    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


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
