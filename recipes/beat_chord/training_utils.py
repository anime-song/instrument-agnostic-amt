from __future__ import annotations

import logging
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class ModelEma:
    def __init__(self, model: Any, decay: float = 0.9997) -> None:
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = float(decay)

    def update(self, model: Any) -> None:
        import torch

        with torch.no_grad():
            for ema_param, model_param in zip(
                self.module.state_dict().values(),
                model.state_dict().values(),
            ):
                ema_param.copy_(
                    self.decay * ema_param
                    + (1.0 - self.decay) * model_param.to(ema_param.device)
                )


def resolve_training_amp_dtype(*, torch_module: Any, device: Any, use_amp: bool) -> Any:
    if not use_amp or device.type != "cuda":
        return None
    if torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16
    return torch_module.float16


def clear_startup_cuda_cache(torch_module: Any) -> None:
    if not torch_module.cuda.is_available():
        return
    torch_module.cuda.empty_cache()
    logger.info("Cleared CUDA cache before training startup.")


@dataclass(frozen=True)
class ShapeMismatch:
    """checkpoint と現在の model で shape が違う state_dict entry。"""

    key: str
    checkpoint_shape: tuple[int, ...]
    model_shape: tuple[int, ...]


@dataclass(frozen=True)
class ModelStateLoadReport:
    """checkpoint から model state をどの程度読み込めたかを記録する。"""

    loaded_key_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]


def select_checkpoint_model_state(
    checkpoint: Any,
    *,
    load_ema: bool,
) -> dict[str, Any]:
    """checkpoint から実際に model へ読む state_dict を選ぶ。"""

    if isinstance(checkpoint, dict):
        if load_ema and isinstance(checkpoint.get("ema_state_dict"), dict):
            return checkpoint["ema_state_dict"]
        if isinstance(checkpoint.get("model_state_dict"), dict):
            return checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint model state must be a dict")
    return checkpoint


def load_model_state(
    *,
    model: Any,
    state_dict: dict[str, Any],
    allow_shape_mismatch: bool,
) -> ModelStateLoadReport:
    """model state を読み込む。init 用なら shape 違いの tensor だけ除外する。"""

    if not allow_shape_mismatch:
        load_result = model.load_state_dict(state_dict, strict=True)
        return ModelStateLoadReport(
            loaded_key_count=len(state_dict),
            missing_keys=tuple(load_result.missing_keys),
            unexpected_keys=tuple(load_result.unexpected_keys),
            shape_mismatches=(),
        )

    current_state = model.state_dict()
    compatible_state: dict[str, Any] = {}
    unexpected_keys: list[str] = []
    shape_mismatches: list[ShapeMismatch] = []

    # 1. 現在の model に存在し、shape も一致する tensor だけを init 対象にする。
    for key, value in state_dict.items():
        if key not in current_state:
            unexpected_keys.append(str(key))
            continue

        checkpoint_shape = tuple(int(dim) for dim in value.shape)
        model_shape = tuple(int(dim) for dim in current_state[key].shape)
        if checkpoint_shape != model_shape:
            shape_mismatches.append(
                ShapeMismatch(
                    key=str(key),
                    checkpoint_shape=checkpoint_shape,
                    model_shape=model_shape,
                )
            )
            continue
        compatible_state[key] = value

    # 2. 部分ロードなので strict=False。missing は未初期化の確認ログに使う。
    load_result = model.load_state_dict(compatible_state, strict=False)
    return ModelStateLoadReport(
        loaded_key_count=len(compatible_state),
        missing_keys=tuple(load_result.missing_keys),
        unexpected_keys=tuple(unexpected_keys) + tuple(load_result.unexpected_keys),
        shape_mismatches=tuple(shape_mismatches),
    )


def log_model_state_load_report(
    *,
    logger: logging.Logger,
    source_path: Path,
    report: ModelStateLoadReport,
    allow_shape_mismatch: bool,
    max_examples: int = 8,
) -> None:
    """部分ロード時に skip された key を読みやすくログへ出す。"""

    logger.info(
        "Loaded %d model tensors from %s",
        report.loaded_key_count,
        source_path,
    )
    if not allow_shape_mismatch:
        return

    if report.shape_mismatches:
        examples = ", ".join(
            f"{item.key}: {item.checkpoint_shape} -> {item.model_shape}"
            for item in report.shape_mismatches[:max_examples]
        )
        logger.warning(
            "Skipped %d tensors with shape mismatch. Examples: %s",
            len(report.shape_mismatches),
            examples,
        )
    if report.unexpected_keys:
        logger.warning(
            "Skipped %d unexpected checkpoint tensors. Examples: %s",
            len(report.unexpected_keys),
            ", ".join(report.unexpected_keys[:max_examples]),
        )
    if report.missing_keys:
        logger.info(
            "Model kept %d tensors randomly initialized. Examples: %s",
            len(report.missing_keys),
            ", ".join(report.missing_keys[:max_examples]),
        )


def save_checkpoint(
    *,
    torch_module: Any,
    checkpoint_path: Path,
    epoch: int,
    global_step: int,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    model_config: Any,
    args: Any,
    avg_total_loss: float,
    avg_beat_loss: float | None,
    avg_chord_loss: float | None,
    ema_model: ModelEma | None,
    avg_key_only_loss: float | None = None,
    beat_meter_classes: Any | None = None,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    serialized_meter_classes = (
        None
        if beat_meter_classes is None
        else [[int(num), int(den)] for num, den in beat_meter_classes]
    )
    save_dict: dict[str, Any] = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": float(avg_total_loss),
        "beat_loss": None if avg_beat_loss is None else float(avg_beat_loss),
        "chord_loss": None if avg_chord_loss is None else float(avg_chord_loss),
        "key_only_loss": (
            None if avg_key_only_loss is None else float(avg_key_only_loss)
        ),
        "model_config": asdict(model_config),
        "beat_meter_classes": serialized_meter_classes,
        "config": {
            "model_config": asdict(model_config),
            "args": vars(args),
            "beat_meter_classes": serialized_meter_classes,
        },
    }
    if ema_model is not None:
        save_dict["ema_state_dict"] = ema_model.module.state_dict()
    torch_module.save(save_dict, checkpoint_path)
    logger.info("Saved checkpoint to %s", checkpoint_path)


def _serialize_for_wandb(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize_for_wandb(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_wandb(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scalarize_metric_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except (TypeError, ValueError, RuntimeError):
            return value
    return value


def prefix_metric_dict(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}{key}": _scalarize_metric_value(value)
        for key, value in metrics.items()
    }


def normalize_meter_classes(
    raw_meter_classes: Any | None,
) -> tuple[tuple[int, int], ...]:
    """checkpoint や JSON 由来の meter class 表を `(num, den)` の tuple にそろえる。"""

    if raw_meter_classes is None:
        return ()

    meter_classes: list[tuple[int, int]] = []
    for raw_meter in raw_meter_classes:
        if not isinstance(raw_meter, (list, tuple)) or len(raw_meter) != 2:
            raise ValueError("meter class entries must be [num, den]")
        meter_num = int(raw_meter[0])
        meter_den = int(raw_meter[1])
        if meter_num <= 0 or meter_den <= 0:
            raise ValueError("meter class values must be positive")
        meter_classes.append((meter_num, meter_den))
    return tuple(meter_classes)


def checkpoint_meter_classes(
    checkpoint: dict[str, Any] | None,
) -> tuple[tuple[int, int], ...]:
    """checkpoint 内に保存された beat 用 meter class 表を取り出す。"""

    if checkpoint is None:
        return ()

    raw_meter_classes = checkpoint.get("beat_meter_classes")
    checkpoint_config = checkpoint.get("config", {})
    if raw_meter_classes is None and isinstance(checkpoint_config, dict):
        raw_meter_classes = checkpoint_config.get("beat_meter_classes")
    return normalize_meter_classes(raw_meter_classes)


def build_wandb_config(
    *,
    args: Any,
    model_config: Any,
    device: Any,
    use_amp: bool,
    beat_dataset: Any,
    chord_dataset: Any,
    trainable_parameter_count: int,
    key_only_dataset: Any = None,
) -> dict[str, Any]:
    return {
        "args": _serialize_for_wandb(vars(args)),
        "model_config": _serialize_for_wandb(asdict(model_config)),
        "runtime": {
            "device": str(device),
            "use_amp": bool(use_amp),
            "trainable_parameter_count": int(trainable_parameter_count),
        },
        "datasets": {
            "beat_num_items": 0 if beat_dataset is None else int(len(beat_dataset)),
            "chord_num_items": 0 if chord_dataset is None else int(len(chord_dataset)),
            "key_only_num_items": (
                0 if key_only_dataset is None else int(len(key_only_dataset))
            ),
        },
    }


def initialize_wandb_run(
    *,
    args: Any,
    model_config: Any,
    device: Any,
    use_amp: bool,
    beat_dataset: Any,
    chord_dataset: Any,
    trainable_parameter_count: int,
    key_only_dataset: Any = None,
) -> Any | None:
    if not args.wandb:
        return None
    if not HAS_WANDB:
        logger.warning(
            "wandb is not installed. Please `pip install wandb` to use it. Falling back to console logging."
        )
        return None
    return wandb.init(
        project=args.project_name,
        name=args.run_name,
        config=build_wandb_config(
            args=args,
            model_config=model_config,
            device=device,
            use_amp=use_amp,
            beat_dataset=beat_dataset,
            chord_dataset=chord_dataset,
            trainable_parameter_count=trainable_parameter_count,
            key_only_dataset=key_only_dataset,
        ),
    )
