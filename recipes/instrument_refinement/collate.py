"""Dataset の出力をバッチにまとめる collate。

やることは 2 つ:

1. Dataset が返す ``{"views": [...]}``（同じ音源の複数窓）を平坦化して、
   1 サンプル = 1 窓に並べ直す。どの窓が同じ元サンプル由来かは ``view_group`` に残す。
2. 窓ごとに異なるノート数を、最大数に合わせて 0 埋めする。埋めた部分は
   ``note_mask`` が False になり、モデルと loss の両方で無視される。
"""

from __future__ import annotations

# This module defines the training batch boundary used by the recipe.

from typing import Any

import torch

# ノート数ぶんの長さを持つ（= 0 埋めが必要な）フィールド。
NOTE_FIELDS = (
    "note_start_seconds",
    "note_end_seconds",
    "note_pitch",
    "note_prior_class",
    "note_confidence",
)


def _pad_1d(values: list[torch.Tensor], *, width: int, fill_value: float | int = 0) -> torch.Tensor:
    output = torch.full((len(values), width), fill_value=fill_value, dtype=values[0].dtype)
    for batch_index, value in enumerate(values):
        output[batch_index, : value.numel()] = value
    return output


def collate_refinement_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """複数サンプルを 1 バッチへ。views は展開し、ノート長は 0 埋めでそろえる。"""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    # views を展開する。展開後のバッチサイズは batch_size * windows_per_source になる。
    flat: list[dict[str, Any]] = []
    view_group: list[int] = []
    for group_index, item in enumerate(batch):
        views = item.get("views")
        if views is None:
            flat.append(item)
            view_group.append(group_index)
        else:
            flat.extend(views)
            view_group.extend([group_index] * len(views))
    max_notes = max(int(item["note_pitch"].numel()) for item in flat)
    collated: dict[str, Any] = {
        "audio": torch.stack([item["audio"] for item in flat]),
        "valid_audio_frames": torch.tensor([item["valid_audio_frames"] for item in flat], dtype=torch.long),
        "note_mask": torch.zeros((len(flat), max_notes), dtype=torch.bool),
        "class_target_mask": torch.stack([item["class_target_mask"] for item in flat]),
        "family_target_mask": torch.stack([item["family_target_mask"] for item in flat]),
        "primary_class_id": torch.tensor([item["primary_class_id"] for item in flat], dtype=torch.long),
        "stem_context_id": torch.tensor([item["stem_context_id"] for item in flat], dtype=torch.long),
        "sample_weight": torch.tensor([item["sample_weight"] for item in flat], dtype=torch.float32),
        "window_sample_weight": torch.tensor([item["window_sample_weight"] for item in flat], dtype=torch.float32),
        "source_hash": torch.tensor([item["source_hash"] for item in flat], dtype=torch.long),
        "view_group": torch.tensor(view_group, dtype=torch.long),
        "window_start_seconds": torch.tensor([item["window_start_seconds"] for item in flat], dtype=torch.float32),
        "source_id": [item["source_id"] for item in flat],
        "song_id": [item["song_id"] for item in flat],
        "stem_name": [item["stem_name"] for item in flat],
        "track_type": [item["track_type"] for item in flat],
        "is_composite": torch.tensor([item["is_composite"] for item in flat], dtype=torch.bool),
    }
    # 実ノートがある位置だけ True にする（残りは 0 埋め）。
    for batch_index, item in enumerate(flat):
        collated["note_mask"][batch_index, : item["note_pitch"].numel()] = True
    for field in NOTE_FIELDS:
        collated[field] = _pad_1d(
            [item[field] for item in flat],
            width=max_notes,
            fill_value=-1 if field == "note_prior_class" else 0,
        )
    # ノート x クラスの 2 次元マスクも同様に 0 埋めしてそろえる。
    class_count = int(flat[0]["class_target_mask"].numel())
    family_count = int(flat[0]["family_target_mask"].numel())
    collated["note_class_target_mask"] = torch.zeros(len(flat), max_notes, class_count, dtype=torch.bool)
    collated["note_family_target_mask"] = torch.zeros(len(flat), max_notes, family_count, dtype=torch.bool)
    for batch_index, item in enumerate(flat):
        count = int(item["note_pitch"].numel())
        collated["note_class_target_mask"][batch_index, :count] = item["note_class_target_mask"]
        collated["note_family_target_mask"][batch_index, :count] = item["note_family_target_mask"]
    for field, fill_value in (
        ("note_primary_class_id", -1),
        ("note_source_hash", 0),
        ("note_sample_weight", 0),
        ("note_contrastive_mask", 0),
    ):
        collated[field] = _pad_1d([item[field] for item in flat], width=max_notes, fill_value=fill_value)
    return collated
