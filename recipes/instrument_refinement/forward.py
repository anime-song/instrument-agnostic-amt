"""バッチをモデルへ流す薄いラッパ。

バッチには loss 用のフィールドも混ざっているので、モデルが必要とするものだけを
選んで渡す。欠けている場合は forward の途中ではなく、ここで分かりやすく落とす。
"""

from __future__ import annotations

from typing import Any

import torch

from instrument_agnostic_amt.instrument_refinement.modeling.model import (
    InstrumentRefinementModel,
)

# モデルの forward に渡すフィールド。これ以外（正解ラベルなど）は loss 側で使う。
MODEL_BATCH_FIELDS = (
    "audio",
    "valid_audio_frames",
    "note_start_seconds",
    "note_end_seconds",
    "note_pitch",
    "note_prior_class",
    "note_confidence",
    "note_mask",
    "stem_context_id",
)


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Tensor だけを device へ移す。ID や名前などの文字列はそのまま残す。"""
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def forward_refinement_batch(
    model: InstrumentRefinementModel,
    batch: dict[str, Any],
    *,
    window_seconds: float,
    include_aux_outputs: bool = False,
) -> dict[str, torch.Tensor]:
    """バッチから必要なフィールドを取り出してモデルを呼ぶ。"""
    missing = [field for field in MODEL_BATCH_FIELDS if field not in batch]
    if missing:
        raise KeyError(f"Refinement batch is missing: {', '.join(missing)}")
    return model(
        batch["audio"],
        valid_audio_frames=batch["valid_audio_frames"],
        note_start_seconds=batch["note_start_seconds"],
        note_end_seconds=batch["note_end_seconds"],
        note_pitch=batch["note_pitch"],
        note_prior_class=batch["note_prior_class"],
        note_confidence=batch["note_confidence"],
        note_mask=batch["note_mask"],
        stem_context_id=batch["stem_context_id"],
        window_seconds=window_seconds,
        include_aux_outputs=include_aux_outputs,
    )
