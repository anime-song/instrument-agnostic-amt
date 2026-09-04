from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

sys.path.append(str(Path(__file__).resolve().parent.parent))
from instrument_agnostic_amt.amt.modeling.model import remap_legacy_v1_state_dict


STATE_DICT_KEYS = ("ema_state_dict", "model_state_dict")
OVERLAP_MERGE_KEYS = {
    "head.instrument_classifier.weight",
    "head.instrument_classifier.bias",
    "head.interval_instrument_predictor.net.4.weight",
    "head.interval_instrument_predictor.net.4.bias",
}


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return checkpoint


def _select_inference_state_dict(
    checkpoint: Mapping[str, Any],
) -> tuple[Mapping[str, torch.Tensor], str]:
    """Match the AMT inference loader: prefer EMA, then model_state_dict, then raw."""
    for key in STATE_DICT_KEYS:
        state_dict = checkpoint.get(key)
        if isinstance(state_dict, Mapping):
            return remap_legacy_v1_state_dict(state_dict), key

    if checkpoint and all(
        isinstance(value, torch.Tensor) for value in checkpoint.values()
    ):
        return remap_legacy_v1_state_dict(checkpoint), "raw_state_dict"

    raise ValueError("Checkpoint does not contain a model state dict")


def _lerp_tensor(
    old_tensor: torch.Tensor,
    new_tensor: torch.Tensor,
    *,
    new_weight: float,
) -> torch.Tensor:
    if not (torch.is_floating_point(old_tensor) and torch.is_floating_point(new_tensor)):
        return new_tensor.detach().clone()

    old_value = old_tensor.detach().to(dtype=torch.float32)
    new_value = new_tensor.detach().to(dtype=torch.float32)
    merged = torch.lerp(old_value, new_value, new_weight)
    return merged.to(dtype=new_tensor.dtype)


def _can_merge_first_dimension_overlap(
    key: str,
    old_tensor: torch.Tensor,
    new_tensor: torch.Tensor,
) -> bool:
    return (
        key in OVERLAP_MERGE_KEYS
        and old_tensor.ndim == new_tensor.ndim
        and old_tensor.ndim >= 1
        and old_tensor.shape[1:] == new_tensor.shape[1:]
    )


def merge_state_dicts(
    old_state: Mapping[str, torch.Tensor],
    new_state: Mapping[str, torch.Tensor],
    *,
    new_weight: float,
) -> tuple[dict[str, torch.Tensor], dict[str, list[str]]]:
    """Merge compatible weights while retaining the newer model architecture."""
    if not 0.0 <= new_weight <= 1.0:
        raise ValueError("new_weight must be in [0, 1]")

    merged_state: dict[str, torch.Tensor] = {}
    report: dict[str, list[str]] = {
        "merged": [],
        "merged_overlap": [],
        "kept_new_missing_old": [],
        "kept_new_shape_mismatch": [],
        "kept_new_non_floating": [],
        "ignored_old_only": sorted(set(old_state) - set(new_state)),
    }

    for key, new_tensor in new_state.items():
        if not isinstance(new_tensor, torch.Tensor):
            raise TypeError(f"New state value is not a tensor: {key}")

        old_tensor = old_state.get(key)
        if not isinstance(old_tensor, torch.Tensor):
            merged_state[key] = new_tensor.detach().clone()
            report["kept_new_missing_old"].append(key)
            continue

        if old_tensor.shape == new_tensor.shape:
            merged_state[key] = _lerp_tensor(
                old_tensor,
                new_tensor,
                new_weight=new_weight,
            )
            if torch.is_floating_point(new_tensor):
                report["merged"].append(key)
            else:
                report["kept_new_non_floating"].append(key)
            continue

        if _can_merge_first_dimension_overlap(key, old_tensor, new_tensor):
            overlap = min(int(old_tensor.shape[0]), int(new_tensor.shape[0]))
            merged_tensor = new_tensor.detach().clone()
            merged_tensor[:overlap] = _lerp_tensor(
                old_tensor[:overlap],
                new_tensor[:overlap],
                new_weight=new_weight,
            )
            merged_state[key] = merged_tensor
            report["merged_overlap"].append(key)
            continue

        merged_state[key] = new_tensor.detach().clone()
        report["kept_new_shape_mismatch"].append(key)

    return merged_state, report


def _model_config(checkpoint: Mapping[str, Any]) -> Any:
    run_config = checkpoint.get("config")
    if isinstance(run_config, Mapping):
        nested_model_config = run_config.get("model_config")
        if nested_model_config is not None:
            return nested_model_config
    return checkpoint.get("model_config")


def build_merged_checkpoint(
    old_checkpoint: Mapping[str, Any],
    new_checkpoint: Mapping[str, Any],
    *,
    old_path: Path,
    new_path: Path,
    new_weight: float,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    old_state, old_source = _select_inference_state_dict(old_checkpoint)
    new_state, new_source = _select_inference_state_dict(new_checkpoint)
    merged_state, report = merge_state_dicts(
        old_state,
        new_state,
        new_weight=new_weight,
    )

    model_config = _model_config(new_checkpoint)
    if model_config is None:
        raise ValueError("New checkpoint does not contain model_config")

    output_checkpoint = {
        "model_state_dict": merged_state,
        "model_config": model_config,
        "config": new_checkpoint.get("config"),
        "merge_metadata": {
            "method": "linear_interpolation_new_architecture",
            "old_checkpoint": str(old_path),
            "new_checkpoint": str(new_path),
            "old_state_source": old_source,
            "new_state_source": new_source,
            "old_weight": 1.0 - new_weight,
            "new_weight": new_weight,
            "overlap_merge_keys": sorted(OVERLAP_MERGE_KEYS),
            "report_counts": {key: len(value) for key, value in report.items()},
        },
    }
    return output_checkpoint, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Linearly interpolate two AMT checkpoints while retaining the newer "
            "checkpoint's architecture and newly added heads."
        )
    )
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--new-weight",
        type=float,
        default=0.2,
        help="Interpolation weight for the new checkpoint (default: 0.2).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.new_weight <= 1.0:
        raise ValueError("--new-weight must be in [0, 1]")
    if not args.old.is_file():
        raise FileNotFoundError(args.old)
    if not args.new.is_file():
        raise FileNotFoundError(args.new)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Pass --overwrite to replace it."
        )

    old_checkpoint = _load_checkpoint(args.old)
    new_checkpoint = _load_checkpoint(args.new)
    output_checkpoint, report = build_merged_checkpoint(
        old_checkpoint,
        new_checkpoint,
        old_path=args.old,
        new_path=args.new,
        new_weight=args.new_weight,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_name(f".{args.output.name}.tmp")
    torch.save(output_checkpoint, temporary_path)
    temporary_path.replace(args.output)

    metadata = output_checkpoint["merge_metadata"]
    print(f"Saved merged checkpoint: {args.output}")
    print(
        f"Weights: old={metadata['old_weight']:.3f}, "
        f"new={metadata['new_weight']:.3f}"
    )
    print(f"State sources: old={metadata['old_state_source']}, new={metadata['new_state_source']}")
    for category, keys in report.items():
        print(f"{category}: {len(keys)}")
        if category in {"merged_overlap", "kept_new_shape_mismatch", "kept_new_missing_old"}:
            for key in keys:
                print(f"  - {key}")


if __name__ == "__main__":
    main()
