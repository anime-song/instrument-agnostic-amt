from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Mapping
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import Any

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from instrument_agnostic_amt.beat_chord.chord_vocabulary import (
    load_chord_quality_map_json,
    normalize_chord_quality_map,
)


# Windows環境でLinux/Colab由来のPosixPathを含む信頼済みcheckpointを読む。
if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath


CHECKPOINT_FORMAT = "instrument_agnostic_amt_inference_v1"


def _is_tensor_state_dict(value: object) -> bool:
    return (
        bool(value)
        and isinstance(value, Mapping)
        and all(
            isinstance(key, str) and torch.is_tensor(tensor)
            for key, tensor in value.items()
        )
    )


def _select_inference_state_dict(
    checkpoint: Mapping[str, Any],
) -> Mapping[str, Any]:
    for key in ("ema_state_dict", "model_state_dict"):
        state_dict = checkpoint.get(key)
        if _is_tensor_state_dict(state_dict):
            return state_dict
    if _is_tensor_state_dict(checkpoint):
        return checkpoint
    raise ValueError(
        "Checkpoint does not contain a tensor-only ema_state_dict or "
        "model_state_dict; refusing to publish the whole checkpoint as weights."
    )


def _extract_model_config(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    raw_config = checkpoint.get("model_config")
    nested_config = checkpoint.get("config")
    if raw_config is None and isinstance(nested_config, Mapping):
        raw_config = nested_config.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Checkpoint does not contain a model_config mapping")
    return dict(raw_config)


def _extract_meter_classes(
    checkpoint: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> list[list[int]] | None:
    raw_classes = checkpoint.get("beat_meter_classes")
    nested_config = checkpoint.get("config")
    if raw_classes is None and isinstance(nested_config, Mapping):
        raw_classes = nested_config.get("beat_meter_classes")
    if raw_classes is None:
        if "num_meter_classes" in model_config:
            raise ValueError("Beat/chord checkpoint has no beat_meter_classes metadata")
        return None

    meter_classes: list[list[int]] = []
    for item in raw_classes:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("beat_meter_classes entries must be [num, den]")
        meter_num, meter_den = int(item[0]), int(item[1])
        if meter_num <= 0 or meter_den <= 0:
            raise ValueError("beat_meter_classes values must be positive")
        meter_classes.append([meter_num, meter_den])

    expected_count = model_config.get("num_meter_classes")
    if expected_count is not None and len(meter_classes) != int(expected_count):
        raise ValueError(
            "beat_meter_classes count does not match model_config: "
            f"{len(meter_classes)} != {int(expected_count)}"
        )
    return meter_classes


def _extract_chord_quality_map(
    checkpoint: Mapping[str, Any],
    model_config: Mapping[str, Any],
    quality_json_path: Path | None,
) -> dict[str, str] | None:
    raw_quality_map: object | None = checkpoint.get("chord_quality_map")
    if quality_json_path is not None:
        return load_chord_quality_map_json(
            quality_json_path,
            expected_root_chord_classes=(
                None
                if model_config.get("num_root_chord_classes") is None
                else int(model_config["num_root_chord_classes"])
            ),
        )

    expected_count = model_config.get("num_root_chord_classes")
    if raw_quality_map is None:
        if expected_count is not None and int(expected_count) > 1:
            raise ValueError(
                "Beat/chord checkpoint has no chord quality map. "
                "Pass --quality-json so the public checkpoint is self-contained."
            )
        return None
    return normalize_chord_quality_map(
        raw_quality_map,
        expected_root_chord_classes=(
            None if expected_count is None else int(expected_count)
        ),
    )


def _extract_inference_config(
    checkpoint: Mapping[str, Any],
) -> dict[str, int] | None:
    raw_inference_config = checkpoint.get("inference_config")
    raw_window_ms: object | None = None
    if isinstance(raw_inference_config, Mapping):
        raw_window_ms = raw_inference_config.get("window_ms")
    if raw_window_ms is None:
        nested_config = checkpoint.get("config")
        raw_args = (
            nested_config.get("args") if isinstance(nested_config, Mapping) else None
        )
        if isinstance(raw_args, Mapping):
            raw_window_ms = raw_args.get("window_ms")
    if raw_window_ms is None:
        return None
    window_ms = int(raw_window_ms)
    if window_ms <= 0:
        raise ValueError("checkpoint window_ms must be positive")
    return {"window_ms": window_ms}


def _absolute_path_description(value: object, trail: str = "checkpoint") -> str | None:
    if isinstance(value, PurePath):
        return f"{trail} contains a pathlib path"
    if isinstance(value, str) and (
        PureWindowsPath(value).is_absolute() or PurePosixPath(value).is_absolute()
    ):
        return f"{trail} contains an absolute path string"
    if isinstance(value, Mapping):
        for key, child in value.items():
            found = _absolute_path_description(child, f"{trail}.{key}")
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _absolute_path_description(child, f"{trail}[{index}]")
            if found is not None:
                return found
    return None


def build_public_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    quality_json_path: Path | None = None,
) -> dict[str, Any]:
    """Create an inference-only checkpoint from an explicit metadata allowlist."""

    model_config = _extract_model_config(checkpoint)
    public_checkpoint: dict[str, Any] = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "model_state_dict": _select_inference_state_dict(checkpoint),
        "model_config": model_config,
    }
    meter_classes = _extract_meter_classes(checkpoint, model_config)
    if meter_classes is not None:
        public_checkpoint["beat_meter_classes"] = meter_classes
    quality_map = _extract_chord_quality_map(
        checkpoint,
        model_config,
        quality_json_path,
    )
    if quality_map is not None:
        public_checkpoint["chord_quality_map"] = quality_map
    inference_config = _extract_inference_config(checkpoint)
    if inference_config is not None:
        public_checkpoint["inference_config"] = inference_config

    path_error = _absolute_path_description(public_checkpoint)
    if path_error is not None:
        raise ValueError(f"Refusing to save public checkpoint: {path_error}")
    return public_checkpoint


def distill_model(
    input_path: str | Path,
    output_path: str | Path,
    *,
    quality_json_path: str | Path | None = None,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output checkpoint paths must differ")

    print(f"Loading trusted checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a mapping")
    public_checkpoint = build_public_checkpoint(
        checkpoint,
        quality_json_path=(
            None if quality_json_path is None else Path(quality_json_path)
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving public checkpoint: {output_path}")
    torch.save(public_checkpoint, output_path)

    old_size = input_path.stat().st_size / (1024 * 1024)
    new_size = output_path.stat().st_size / (1024 * 1024)
    print("Done!")
    print(f"Original size: {old_size:.2f} MB")
    print(f"Distilled size: {new_size:.2f} MB")
    print(f"Keys in public checkpoint: {list(public_checkpoint.keys())}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a path-free inference checkpoint for distribution"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--quality-json",
        type=Path,
        default=None,
        help=(
            "Training quality.json to embed in a beat/chord checkpoint. "
            "Required for legacy beat/chord checkpoints."
        ),
    )
    args = parser.parse_args()
    distill_model(
        args.input,
        args.output,
        quality_json_path=args.quality_json,
    )


if __name__ == "__main__":
    main()
