from __future__ import annotations

import argparse
import logging
import os
import random
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from ..data.dataset import StemDataset
from ..modeling.model import AudioSemiCRFTransformer, SemiCRFModelConfig
from ..modeling.stem_splitter_init import load_stem_splitter_initialization
from ..training.losses import compute_losses

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelEma(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, decay: float = 0.9997):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_param, model_param in zip(
            self.module.state_dict().values(),
            model.state_dict().values(),
        ):
            ema_param.copy_(
                self.decay * ema_param
                + (1.0 - self.decay) * model_param.to(ema_param.device)
            )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "audio": torch.stack([item["audio"] for item in batch]),
        "frame_active_targets": torch.stack(
            [item["frame_active_targets"] for item in batch]
        ),
        "interval_targets": [item["interval_targets"] for item in batch],
        "valid_audio_frames": torch.tensor(
            [item.get("valid_audio_frames", item["audio"].shape[-1]) for item in batch],
            dtype=torch.long,
        ),
        "condition_instrument_ids": torch.tensor(
            [int(item["condition_instrument_id"]) for item in batch],
            dtype=torch.long,
        ),
    }


def resolve_training_amp_dtype(
    device: torch.device, *, use_amp: bool
) -> torch.dtype | None:
    if not use_amp or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def count_parameters(parameters) -> int:
    return sum(parameter.numel() for parameter in parameters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train V2 conditioned Semi-CRF AMT model"
    )
    parser.add_argument("--manifest_path", type=str, default="manifest.csv")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="configs/datasets/dataset_config.yaml",
        help="Path to dataset config YAML. Overrides manifest_path when present.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--window_ms", type=int, default=8000)
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--encoder_num_layers", type=int, default=6)
    parser.add_argument("--encoder_num_heads", type=int, default=8)
    parser.add_argument("--encoder_head_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lwr_ratio", type=int, default=4)
    parser.add_argument("--num_pitch_slots", type=int, default=1)
    parser.add_argument("--semi_crf_head_dim", type=int, default=256)
    parser.add_argument(
        "--semi_crf_length_scaling", choices=("linear", "sqrt", "none"), default="none"
    )
    parser.add_argument("--semi_crf_length_penalty", type=float, default=0.0)
    parser.add_argument("--semi_crf_loss_weight", type=float, default=1.0)
    parser.add_argument("--semi_crf_false_negative_cost", type=float, default=0.0)
    parser.add_argument("--semi_crf_false_positive_cost", type=float, default=0.0)
    parser.add_argument("--semi_crf_track_batch_size", type=int, default=128)
    parser.add_argument("--interval_presence_loss_weight", type=float, default=1.0)
    parser.add_argument("--interval_offset_loss_weight", type=float, default=1.0)
    parser.add_argument("--condition_negative_prob", type=float, default=0.25)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--project_name", type=str, default="instrument_agnostic_amt_v2"
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--ir_folder", type=str, default=None)
    parser.add_argument("--noise_folder", type=str, default=None)
    parser.add_argument("--drum_folder", type=str, default=None)
    parser.add_argument("--p_drum_mix", type=float, default=0.1)
    parser.add_argument("--p_augment", type=float, default=1.0)
    parser.add_argument("--p_intra_drop", type=float, default=0.3)
    parser.add_argument("--p_cross_mix", type=float, default=0.5)
    parser.add_argument("--max_cross_stems", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--init-from", type=str, default=None)
    parser.add_argument(
        "--init-from-stem-splitter",
        type=str,
        default=None,
        help="Initialize compatible V2 backbone weights from an MSS BS-RoFormer stem splitter checkpoint.",
    )
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--disable-gradient-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation_steps must be positive")
    if args.semi_crf_track_batch_size <= 0:
        raise ValueError("--semi_crf_track_batch_size must be positive")
    if args.init_from and args.init_from_stem_splitter:
        raise ValueError(
            "--init-from and --init-from-stem-splitter are mutually exclusive"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    amp_dtype = resolve_training_amp_dtype(device, use_amp=use_amp)
    use_grad_scaler = amp_dtype == torch.float16

    if args.wandb:
        if not HAS_WANDB:
            logger.warning("wandb is not installed; disabling wandb logging.")
            args.wandb = False
        else:
            wandb.init(project=args.project_name, name=args.run_name, config=vars(args))

    os.makedirs(args.save_dir, exist_ok=True)
    dataset = StemDataset(
        manifest_path=args.manifest_path,
        dataset_config_path=args.dataset_config,
        window_ms=args.window_ms,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        num_pitch_slots=args.num_pitch_slots,
        p_intra_drop=args.p_intra_drop,
        p_cross_mix=args.p_cross_mix,
        max_cross_stems=args.max_cross_stems,
        p_augment=args.p_augment,
        ir_folder=args.ir_folder,
        noise_folder=args.noise_folder,
        drum_folder=args.drum_folder,
        p_drum_mix=args.p_drum_mix,
        condition_negative_prob=args.condition_negative_prob,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    config = SemiCRFModelConfig(
        sample_rate=dataset.sample_rate,
        hop_length=dataset.hop_length,
        n_fft=dataset.n_fft,
        hidden_size=args.hidden_size,
        encoder_num_layers=args.encoder_num_layers,
        encoder_num_heads=args.encoder_num_heads,
        encoder_head_dim=args.encoder_head_dim,
        dropout=args.dropout,
        lwr_ratio=args.lwr_ratio,
        num_pitch_slots=args.num_pitch_slots,
        semi_crf_head_dim=args.semi_crf_head_dim,
        semi_crf_length_scaling=args.semi_crf_length_scaling,
        semi_crf_length_penalty=args.semi_crf_length_penalty,
        use_gradient_checkpoint=not args.disable_gradient_checkpoint,
    )
    model = AudioSemiCRFTransformer(config).to(device)

    if args.init_from_stem_splitter:
        report = load_stem_splitter_initialization(
            model,
            args.init_from_stem_splitter,
        )
        logger.info(
            "Loaded stem splitter init from %s. tensors=%d numel=%d categories=%s category_numel=%s skipped_missing=%d skipped_shape=%d",
            args.init_from_stem_splitter,
            report.loaded_tensors,
            report.loaded_numel,
            report.categories,
            report.category_numel,
            report.skipped_missing,
            report.skipped_shape,
        )
        if report.skipped_shape_examples:
            logger.info(
                "Stem splitter shape-skip examples: %s",
                report.skipped_shape_examples[:10],
            )

    if args.init_from:
        checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
        raw_config = (
            checkpoint.get("model_config") if isinstance(checkpoint, dict) else None
        )
        if (
            isinstance(raw_config, dict)
            and int(raw_config.get("architecture_version", 1)) != 2
        ):
            raise ValueError("V2 training cannot initialize from a V1 checkpoint")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        incompatible = model.load_state_dict(state_dict, strict=False)
        logger.info(
            "Loaded init checkpoint. missing=%s unexpected=%s",
            incompatible.missing_keys,
            incompatible.unexpected_keys,
        )

    if args.wandb:
        wandb.config.update({"model_config": asdict(config)})

    use_ema = args.ema_decay > 0.0
    ema_model = ModelEma(model, decay=args.ema_decay) if use_ema else None
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    logger.info(
        "Trainable parameters: %d / %d",
        count_parameters(trainable_parameters),
        count_parameters(model.parameters()),
    )

    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=1e-4)

    def lr_lambda(step: int) -> float:
        if args.warmup_steps <= 0:
            return 1.0
        return min((step + 1) / args.warmup_steps, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(device.type) if use_grad_scaler else None

    def flush_optimizer_step() -> None:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < scale_before
        else:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
            optimizer.step()
            skipped = False
        if not skipped:
            scheduler.step()
            if ema_model is not None:
                ema_model.update(model)
        optimizer.zero_grad(set_to_none=True)

    logger.info(
        "Starting V2 training on %s AMP=%s dtype=%s", device, use_amp, amp_dtype
    )
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        dataset.set_epoch(epoch)
        epoch_loss = 0.0
        num_batches = len(dataloader)
        micro_steps = 0
        current_accumulation = args.accumulation_steps
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch_idx, batch in enumerate(progress, start=1):
            if micro_steps == 0:
                remaining = num_batches - batch_idx + 1
                current_accumulation = min(args.accumulation_steps, remaining)

            audio = batch["audio"].to(device)
            valid_audio_frames = batch["valid_audio_frames"].to(device)
            condition_ids = batch["condition_instrument_ids"].to(device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(
                    audio,
                    condition_instrument_ids=condition_ids,
                    valid_audio_frames=valid_audio_frames,
                )
                total_loss, loss_dict = compute_losses(
                    outputs, batch, args=args, model=model
                )

            loss_value = float(total_loss.item())
            (
                total_loss / current_accumulation
            ).backward() if scaler is None else scaler.scale(
                total_loss / current_accumulation
            ).backward()
            micro_steps += 1
            if micro_steps == current_accumulation:
                flush_optimizer_step()
                micro_steps = 0

            epoch_loss += loss_value
            global_step += 1
            if args.wandb:
                wandb.log(
                    {
                        f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v
                        for k, v in loss_dict.items()
                    }
                    | {"train/learning_rate": optimizer.param_groups[0]["lr"]},
                    step=global_step,
                )
            progress.set_postfix({"loss": f"{loss_value:.4f}"})

        avg_loss = epoch_loss / max(1, num_batches)
        logger.info("Epoch %d complete. Average loss %.4f", epoch, avg_loss)
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(
                args.save_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
                "model_config": asdict(config),
                "config": {"model_config": asdict(config), "args": vars(args)},
            }
            if ema_model is not None:
                save_dict["ema_state_dict"] = ema_model.module.state_dict()
            torch.save(save_dict, checkpoint_path)
            logger.info("Saved checkpoint to %s", checkpoint_path)

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
