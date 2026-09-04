from __future__ import annotations

import argparse
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from instrument_agnostic_amt.velocity.modeling.checkpoints import load_amt_backbone
from instrument_agnostic_amt.velocity.modeling.model import (
    VelocityModelConfig,
    VelocityPredictionModel,
    load_velocity_model_config,
)
from .collate import collate_velocity_batch
from .forward import forward_velocity_batch, move_batch_to_device
from .losses import VelocityLossConfig, compute_velocity_losses
from .stem_dataset import SyntheticStemVelocityDataset
from .synthesis.config import SyntheticDataConfig, load_synthetic_config


LOGGER = logging.getLogger(__name__)
VELOCITY_ROOT = (
    Path(__file__).resolve().parents[2] / "instrument_agnostic_amt" / "velocity"
)
DEFAULT_ROOT = VELOCITY_ROOT / "artifacts" / "synthetic"
DEFAULT_MODEL_CONFIG = VELOCITY_ROOT / "configs" / "model.json"
DEFAULT_SYNTHETIC_CONFIG = VELOCITY_ROOT / "configs" / "synthetic.json"
DEFAULT_OUTPUT_DIR = VELOCITY_ROOT / "artifacts" / "checkpoints_velocity_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the post-AMT note velocity model."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument(
        "--synthetic-config",
        type=Path,
        default=DEFAULT_SYNTHETIC_CONFIG,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--init-amt", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--prefer-non-ema", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--hop-seconds", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.9)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Skip examples whose mixture is not assembled yet (intended for dry-runs).",
    )
    parser.add_argument(
        "--max-steps-per-epoch",
        "--max_steps_per_epoch",
        "--max-steps_per_epoch",
        type=int,
        default=None,
        help="1エポックあたりで実行する最大ステップ数",
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--velocity-ce-weight", type=float, default=1.0)
    parser.add_argument("--velocity-expected-weight", type=float, default=0.25)
    parser.add_argument(
        "--stem-gain-weight",
        type=float,
        default=0.0,
        help="Legacy opt-in weight for relative stem-gain supervision.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--gain-huber-delta-db", type=float, default=1.0)
    args = parser.parse_args()
    if args.init_amt is not None and args.resume is not None:
        parser.error("--init-amt and --resume cannot be used together")
    if args.freeze_backbone and args.init_amt is None and args.resume is None:
        parser.error("--freeze-backbone requires --init-amt or --resume")
    if args.max_steps_per_epoch is not None and args.max_steps_per_epoch <= 0:
        parser.error("--max-steps-per-epoch には正の整数を指定してください。")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _config_from_mapping(raw: Mapping[str, Any]) -> VelocityModelConfig:
    values = dict(raw)
    for key in ("harmonics", "local_frame_offsets"):
        if key in values:
            values[key] = tuple(values[key])
    return VelocityModelConfig(**values)


def _make_dataset(
    args: argparse.Namespace,
    config: VelocityModelConfig,
    data_config: SyntheticDataConfig,
    *,
    split: str,
) -> SyntheticStemVelocityDataset:
    return SyntheticStemVelocityDataset(
        args.root,
        split=split,
        sample_rate=config.sample_rate,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        use_gain_augmentation=data_config.use_gain_augmentation,
        gain_jitter_std_db=data_config.gain_jitter_std_db,
        gain_clip_db=data_config.gain_clip_db,
        master_gain_min_db=data_config.master_gain_min_db,
        master_gain_max_db=data_config.master_gain_max_db,
        allow_incomplete=args.allow_incomplete,
        max_examples=args.max_examples,
    )


def _make_loader(
    dataset: SyntheticStemVelocityDataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_velocity_batch,
    )


def _format_metrics(metrics: Mapping[str, float]) -> str:
    text = (
        f"loss={metrics.get('loss', 0.0):.4f}, "
        f"velocity_mae={metrics.get('velocity_mae', 0.0):.2f}"
    )
    if metrics.get("stem_gain_count", 0.0) > 0.0:
        text += f", gain_mae_db={metrics.get('stem_gain_mae_db', 0.0):.2f}"
    return text


def _run_epoch(
    model: VelocityPredictionModel,
    loader: DataLoader,
    *,
    device: torch.device,
    loss_config: VelocityLossConfig,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    amp_dtype: torch.dtype,
    grad_clip: float,
    max_steps: int | None,
    description: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    metric_sums: dict[str, float] = {}
    step_count = 0
    total_steps = len(loader) if hasattr(loader, "__len__") else None
    if max_steps is not None:
        total_steps = min(total_steps, max_steps) if total_steps is not None else max_steps
    progress: Iterable[dict[str, Any]] = tqdm(loader, total=total_steps, desc=description)
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for batch in progress:
            if max_steps is not None and step_count >= max_steps:
                break
            batch = move_batch_to_device(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = forward_velocity_batch(model, batch)
                loss, metrics = compute_velocity_losses(
                    outputs,
                    batch,
                    config=loss_config,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite velocity loss; rerun with --no-amp to diagnose"
                )
            if optimizer is not None:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
            step_count += 1
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value.item())
            if isinstance(progress, tqdm):
                progress.set_postfix(loss=f"{float(loss.detach()):.4f}")
    if step_count == 0:
        raise RuntimeError("The DataLoader produced no training batches")
    return {key: value / step_count for key, value in metric_sums.items()}


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: VelocityPredictionModel,
    optimizer: torch.optim.Optimizer,
    model_config: VelocityModelConfig,
    loss_config: VelocityLossConfig,
    args: argparse.Namespace,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "checkpoint_format_version": 1,
            "task": (
                "velocity_and_relative_stem_gain"
                if (
                    model_config.predict_stem_gain
                    and loss_config.stem_gain_weight > 0.0
                )
                else "note_velocity"
            ),
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "loss_config": asdict(loss_config),
            "train_metrics": dict(train_metrics),
            "validation_metrics": (
                dict(validation_metrics) if validation_metrics is not None else None
            ),
            "training_args": vars(args),
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        if not isinstance(resume_checkpoint.get("model_config"), Mapping):
            raise ValueError("Resume checkpoint does not contain model_config")
        model_config = _config_from_mapping(resume_checkpoint["model_config"])
    else:
        model_config = load_velocity_model_config(args.model_config)
    loss_config = VelocityLossConfig(
        velocity_ce_weight=args.velocity_ce_weight,
        velocity_expected_weight=args.velocity_expected_weight,
        stem_gain_weight=args.stem_gain_weight,
        label_smoothing=args.label_smoothing,
        gain_huber_delta_db=args.gain_huber_delta_db,
    )
    if loss_config.stem_gain_weight > 0.0 and not model_config.predict_stem_gain:
        raise ValueError(
            "--stem-gain-weight requires predict_stem_gain=true in model config"
        )
    data_config = load_synthetic_config(args.synthetic_config)
    model = VelocityPredictionModel(model_config)
    if args.init_amt is not None:
        report = load_amt_backbone(
            model,
            args.init_amt,
            prefer_ema=not args.prefer_non_ema,
        )
        LOGGER.info("Loaded %d AMT backbone tensors", len(report.loaded_keys))
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    if resume_checkpoint is not None:
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1

    train_dataset = _make_dataset(
        args, model_config, data_config, split="train"
    )
    train_loader = _make_loader(train_dataset, args, shuffle=True)
    validation_loader: DataLoader | None = None
    if not args.skip_validation:
        try:
            validation_dataset = _make_dataset(
                args, model_config, data_config, split="validation"
            )
        except ValueError as error:
            if not str(error).startswith("No usable examples"):
                raise
            LOGGER.warning("Validation disabled: %s", error)
        else:
            validation_loader = _make_loader(validation_dataset, args, shuffle=False)
    LOGGER.info(
        "device=%s, train_examples=%d, train_windows=%d, train_songs=%d",
        device,
        len(train_dataset.examples),
        len(train_dataset),
        train_dataset.song_count,
    )
    use_amp = device.type == "cuda" and not args.no_amp
    amp_dtype = (
        torch.bfloat16
        if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if use_amp and amp_dtype == torch.float16
        else None
    )
    final_epoch = start_epoch if args.dry_run else args.epochs
    max_steps = 1 if args.dry_run else args.max_steps_per_epoch
    for epoch in range(start_epoch, final_epoch + 1):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            loss_config=loss_config,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            grad_clip=args.grad_clip,
            max_steps=max_steps,
            description=f"train {epoch}",
        )
        LOGGER.info("epoch=%d train %s", epoch, _format_metrics(train_metrics))
        validation_metrics = None
        if validation_loader is not None and not args.dry_run:
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device=device,
                loss_config=loss_config,
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                grad_clip=args.grad_clip,
                max_steps=args.max_steps_per_epoch,
                description=f"valid {epoch}",
            )
            LOGGER.info(
                "epoch=%d validation %s",
                epoch,
                _format_metrics(validation_metrics),
            )
        if not args.dry_run and (
            epoch % args.save_every == 0 or epoch == final_epoch
        ):
            checkpoint_path = args.output_dir / f"checkpoint_epoch_{epoch:04d}.pth"
            _save_checkpoint(
                checkpoint_path,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                model_config=model_config,
                loss_config=loss_config,
                args=args,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
            )
            LOGGER.info("Saved %s", checkpoint_path)
    if args.dry_run:
        LOGGER.info("Dry run completed; no checkpoint was written")


if __name__ == "__main__":
    main()
