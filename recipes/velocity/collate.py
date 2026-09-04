"""Batch collation for velocity training."""

from __future__ import annotations

from typing import Any

import torch


NOTE_FIELDS = (
    "note_start_seconds",
    "note_end_seconds",
    "note_pitch",
    "note_program",
    "note_is_drum",
    "note_track_index",
    "note_stem_index",
    "target_velocity",
    "target_velocity_unit",
    "source_pseudo_confidence",
    "rank_source",
    "independently_randomized",
)


def _pad_1d(
    values: list[torch.Tensor],
    *,
    width: int,
    fill_value: float | int | bool = 0,
) -> torch.Tensor:
    output = torch.full(
        (len(values), width),
        fill_value=fill_value,
        dtype=values[0].dtype,
    )
    for batch_index, value in enumerate(values):
        output[batch_index, : value.numel()] = value
    return output


def collate_velocity_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable note/stem dimensions while preserving exact masks."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    max_notes = max(int(item["target_velocity"].numel()) for item in batch)
    max_stems = max(int(item["stem_gain_db"].numel()) for item in batch)
    audio_shape = batch[0]["audio"].shape
    if len(audio_shape) != 3:
        raise ValueError("Each item audio must have shape [S, C, T]")
    if any(tuple(item["audio"].shape[1:]) != tuple(audio_shape[1:]) for item in batch):
        raise ValueError("All stem audio must share channel and frame dimensions")
    audio = torch.zeros(
        (len(batch), max_stems, int(audio_shape[1]), int(audio_shape[2])),
        dtype=batch[0]["audio"].dtype,
    )
    valid_audio_frames = torch.zeros((len(batch), max_stems), dtype=torch.long)
    for batch_index, item in enumerate(batch):
        stem_count = int(item["stem_gain_db"].numel())
        audio[batch_index, :stem_count] = item["audio"]
        valid_audio_frames[batch_index, :stem_count] = item["valid_audio_frames"]
    collated: dict[str, Any] = {
        "audio": audio,
        "valid_audio_frames": valid_audio_frames,
        "note_mask": torch.zeros((len(batch), max_notes), dtype=torch.bool),
        "stem_mask": torch.zeros((len(batch), max_stems), dtype=torch.bool),
        "example_id": [item["example_id"] for item in batch],
        "song_id": [item["song_id"] for item in batch],
        "variation": torch.tensor(
            [int(item["variation"]) for item in batch],
            dtype=torch.long,
        ),
        "window_start_seconds": torch.tensor(
            [float(item["window_start_seconds"]) for item in batch],
            dtype=torch.float32,
        ),
        "master_gain_db": torch.tensor(
            [float(item["master_gain_db"]) for item in batch],
            dtype=torch.float32,
        ),
        "peak_limiter_gain_db": torch.tensor(
            [float(item["peak_limiter_gain_db"]) for item in batch],
            dtype=torch.float32,
        ),
        "stem_names": [item["stem_names"] for item in batch],
    }
    for batch_index, item in enumerate(batch):
        collated["note_mask"][batch_index, : item["target_velocity"].numel()] = True
        collated["stem_mask"][batch_index, : item["stem_gain_db"].numel()] = True
    for field in NOTE_FIELDS:
        collated[field] = _pad_1d(
            [item[field] for item in batch],
            width=max_notes,
        )
    collated["stem_gain_db"] = _pad_1d(
        [item["stem_gain_db"] for item in batch],
        width=max_stems,
    )
    collated["stem_class_id"] = _pad_1d(
        [item["stem_class_id"] for item in batch],
        width=max_stems,
        fill_value=-1,
    )
    collated["stem_active"] = _pad_1d(
        [item["stem_active"] for item in batch],
        width=max_stems,
    )
    collated["stem_gain_mask"] = collated["stem_mask"] & collated["stem_active"]
    return collated
