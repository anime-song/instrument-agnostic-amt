from __future__ import annotations

import json
from pathlib import Path, PurePath

import pytest
import torch

from scripts.distill_model import build_public_checkpoint, distill_model


def _source_checkpoint() -> dict[str, object]:
    model_config = {
        "sample_rate": 100,
        "hop_length": 10,
        "num_meter_classes": 2,
        "num_root_chord_classes": 25,
    }
    return {
        "ema_state_dict": {"weight": torch.tensor([2.0])},
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "optimizer_state_dict": {"private": "training-only"},
        "model_config": model_config,
        "beat_meter_classes": [[3, 4], [4, 4]],
        "config": {
            "model_config": model_config,
            "beat_meter_classes": [[3, 4], [4, 4]],
            "args": {
                "window_ms": 25_000,
                "midi_dir": Path("/private/training/midi"),
                "save_dir": Path("/private/checkpoints"),
            },
        },
    }


def _contains_path(value: object) -> bool:
    if isinstance(value, PurePath):
        return True
    if isinstance(value, dict):
        return any(_contains_path(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_path(child) for child in value)
    return False


def test_distill_model_builds_self_contained_path_free_checkpoint(
    tmp_path: Path,
) -> None:
    quality_path = tmp_path / "quality.json"
    quality_path.write_text(
        json.dumps({"0": "", "1": "m", "2": "N"}),
        encoding="utf-8",
    )
    input_path = tmp_path / "training.pth"
    output_path = tmp_path / "public.pth"
    torch.save(_source_checkpoint(), input_path)

    distill_model(
        input_path,
        output_path,
        quality_json_path=quality_path,
    )
    public = torch.load(output_path, map_location="cpu", weights_only=False)

    assert set(public) == {
        "checkpoint_format",
        "model_state_dict",
        "model_config",
        "beat_meter_classes",
        "chord_quality_map",
        "inference_config",
    }
    assert torch.equal(public["model_state_dict"]["weight"], torch.tensor([2.0]))
    assert public["beat_meter_classes"] == [[3, 4], [4, 4]]
    assert public["chord_quality_map"] == {"0": "", "1": "m", "2": "N"}
    assert public["inference_config"] == {"window_ms": 25_000}
    assert not _contains_path(public)


def test_public_beat_chord_checkpoint_requires_quality_vocabulary() -> None:
    with pytest.raises(ValueError, match="quality map"):
        build_public_checkpoint(_source_checkpoint())


def test_distill_rejects_ambiguous_non_state_dict_checkpoint() -> None:
    source = _source_checkpoint()
    source.pop("ema_state_dict")
    source.pop("model_state_dict")

    with pytest.raises(ValueError, match="refusing to publish"):
        build_public_checkpoint(
            source,
            quality_json_path=None,
        )
