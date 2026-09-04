from __future__ import annotations

from typing import Any

import torch

from instrument_agnostic_amt.velocity.modeling.model import VelocityPredictionModel


MODEL_BATCH_FIELDS = (
    "audio",
    "valid_audio_frames",
    "note_start_seconds",
    "note_end_seconds",
    "note_pitch",
    "note_program",
    "note_is_drum",
    "note_stem_index",
    "stem_class_id",
    "note_mask",
    "stem_mask",
)


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True)
        if isinstance(value, torch.Tensor)
        else value
        for key, value in batch.items()
    }


def forward_velocity_batch(
    model: VelocityPredictionModel,
    batch: dict[str, Any],
    *,
    include_aux_outputs: bool = False,
) -> dict[str, torch.Tensor]:
    missing = [field for field in MODEL_BATCH_FIELDS if field not in batch]
    if missing:
        raise KeyError(f"Velocity batch is missing: {', '.join(missing)}")
    return model(
        batch["audio"],
        valid_audio_frames=batch["valid_audio_frames"],
        note_start_seconds=batch["note_start_seconds"],
        note_end_seconds=batch["note_end_seconds"],
        note_pitch=batch["note_pitch"],
        note_program=batch["note_program"],
        note_is_drum=batch["note_is_drum"],
        note_stem_index=batch["note_stem_index"],
        stem_class_id=batch["stem_class_id"],
        note_mask=batch["note_mask"],
        stem_mask=batch["stem_mask"],
        include_aux_outputs=include_aux_outputs,
    )
