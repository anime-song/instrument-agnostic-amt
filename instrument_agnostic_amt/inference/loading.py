"""Shared loading helpers for trusted local inference checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from ..modeling.checkpoints import (
    CheckpointLoadReport,
    coerce_model_config,
    extract_inference_args,
    extract_model_config,
    load_checkpoint,
    load_compatible_weights,
    require_complete_inference_head,
)
from ..modeling.model import AudioSemiCRFTransformer, SemiCRFModelConfig
from ..taxonomy.instrument_classes import INSTRUMENT_CLASSES


@dataclass(frozen=True, slots=True)
class LoadedInferenceModel:
    """Loaded AMT model and metadata required for inference."""

    model: AudioSemiCRFTransformer
    config: SemiCRFModelConfig
    checkpoint_args: Mapping[str, object]
    report: CheckpointLoadReport


def resolve_checkpoint_path(checkpoint_path: str | Path) -> Path:
    """Resolve one existing checkpoint path and require a regular file."""

    resolved_checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not resolved_checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint is not a file: {resolved_checkpoint}")
    return resolved_checkpoint


def load_inference_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> LoadedInferenceModel:
    """Load an AMT model and its inference metadata from a trusted checkpoint."""

    checkpoint = load_checkpoint(checkpoint_path)
    config = coerce_model_config(
        extract_model_config(checkpoint),
        for_inference=True,
    )
    if int(config.num_instrument_classes) > len(INSTRUMENT_CLASSES):
        raise ValueError(
            "Checkpoint instrument head has "
            f"{int(config.num_instrument_classes)} classes, but the bundled "
            f"taxonomy defines {len(INSTRUMENT_CLASSES)}"
        )
    model = AudioSemiCRFTransformer(config)
    report = load_compatible_weights(
        model,
        checkpoint,
        prefer_ema=True,
        require_complete_backbone=True,
    )
    require_complete_inference_head(model, report)
    model.to(device)
    model.eval()
    return LoadedInferenceModel(
        model=model,
        config=config,
        checkpoint_args=extract_inference_args(checkpoint),
        report=report,
    )
