import argparse
import logging
import os
import random
from copy import deepcopy
from typing import Any
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from dataset import StemDataset
from losses import compute_losses
from models.beat import (
    BeatDataset,
    BeatLoss,
    add_beat_training_args,
    beat_collate_fn,
    compute_beat_batch_loss,
    beat_config_from_args,
    beat_dataset_has_wav_audio,
)
from models.chord import (
    ChordDataset,
    ChordLoss,
    add_chord_training_args,
    chord_collate_fn,
    compute_chord_batch_loss,
    chord_config_from_args,
)
from models.model import AudioSemiCRFTransformer, SemiCRFModelConfig

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelEma(torch.nn.Module):
    """モデルパラメータの指数移動平均を保持するラッパー。"""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9997):
        super().__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clear_startup_cuda_cache() -> None:
    """学習開始前に未使用の CUDA cache を解放する。"""
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    logger.info("Cleared CUDA cache before training startup.")


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """カスタムバッチ作成関数。テンソル以外のデータ(interval_targets)も適切にまとめる。"""
    audio = torch.stack([b["audio"] for b in batch])
    frame_active_targets = torch.stack([b["frame_active_targets"] for b in batch])
    frame_instrument_targets = torch.stack(
        [b["frame_instrument_targets"] for b in batch]
    )
    interval_targets = [b["interval_targets"] for b in batch]
    valid_audio_frames = torch.tensor(
        [b.get("valid_audio_frames", b["audio"].shape[-1]) for b in batch],
        dtype=torch.long,
    )
    # 楽器ラベルなしデータの楽器分類ロスマスク (True = マスクしてロス計算しない)
    mask_instrument_loss = torch.tensor(
        [b.get("mask_instrument_loss", False) for b in batch],
        dtype=torch.bool,
    )
    return {
        "audio": audio,
        "frame_active_targets": frame_active_targets,
        "frame_instrument_targets": frame_instrument_targets,
        "interval_targets": interval_targets,
        "valid_audio_frames": valid_audio_frames,
        "mask_instrument_loss": mask_instrument_loss,
    }


def should_run_auxiliary_update(step_index: int, interval: int) -> bool:
    """AMT の学習 step に対して補助タスクをこの回で更新するか判定する。"""
    if interval <= 0:
        raise ValueError("Auxiliary update interval must be positive")
    return step_index % interval == 0


def resolve_training_amp_dtype(
    device: torch.device,
    *,
    use_amp: bool,
) -> torch.dtype | None:
    """学習時の autocast dtype を決める。CUDA では bf16 を優先する。"""
    if not use_amp or device.type != "cuda":
        return None
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def main():
    parser = argparse.ArgumentParser(description="Train Stem-based AMT Model")
    parser.add_argument("--manifest_path", type=str, default="manifest.csv")
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="dataset_config.yaml",
        help="Path to dataset config YAML (dataset_config.yaml). Overrides manifest_path.",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients",
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=1000,
        help="Number of optimizer steps for linear learning-rate warmup",
    )
    parser.add_argument(
        "--epochs", type=int, default=3000, help="Number of training epochs"
    )
    parser.add_argument(
        "--sample_rate", type=int, default=22050, help="Audio sampling rate"
    )
    parser.add_argument(
        "--window_ms", type=int, default=8000, help="Audio window size in milliseconds"
    )
    parser.add_argument(
        "--num_pitch_slots",
        type=int,
        default=1,
        help="Number of deterministic pitch slots used to separate same-pitch overlaps.",
    )
    parser.add_argument(
        "--semi_crf_false_negative_cost",
        type=float,
        default=0.0,
        help=(
            "Structured training cost for leaving gold-active frames uncovered. "
            "Increase to push recall up."
        ),
    )
    parser.add_argument(
        "--semi_crf_false_positive_cost",
        type=float,
        default=0.0,
        help=(
            "Structured training cost for covering gold-silent frames with an interval. "
            "Increase to push precision up."
        ),
    )
    parser.add_argument(
        "--instrument_loss_type",
        choices=("bce", "ce"),
        default="bce",
        help=(
            "Instrument loss type. Use 'bce' for multi-label BCE or 'ce' for "
            "single-label softmax cross-entropy."
        ),
    )
    parser.add_argument(
        "--no_amp", action="store_true", help="Disable mixed precision training"
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--project_name",
        type=str,
        default="instrument_agnostic_amt",
        help="Wandb project name",
    )
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--ir_folder",
        type=str,
        default=None,
        help="Path to IRs folder for reverb augmentation",
    )
    parser.add_argument(
        "--noise_folder",
        type=str,
        default=None,
        help="Path to noise folder for background noise augmentation",
    )
    parser.add_argument(
        "--drum_folder",
        type=str,
        default=None,
        help="Path to separate drum folder for drum robustness augmentation",
    )
    parser.add_argument(
        "--p_drum_mix",
        type=float,
        default=0.1,
        help="Probability of mixing a separate drum track",
    )
    parser.add_argument(
        "--p_augment",
        type=float,
        default=1.0,
        help="Probability of applying audio augmentations",
    )
    parser.add_argument(
        "--p_intra_drop",
        type=float,
        default=0.3,
        help="Probability of dropping intra-song stems",
    )
    parser.add_argument(
        "--p_cross_mix",
        type=float,
        default=0.5,
        help="Probability of mixing cross-song stems",
    )
    parser.add_argument(
        "--max_cross_stems",
        type=int,
        default=5,
        help="Maximum number of cross-song stems to mix into one sample",
    )
    parser.add_argument(
        "--sa_freq_max",
        type=int,
        default=0,
        help="SpecAugment maximum frequency mask bins",
    )
    parser.add_argument(
        "--sa_time_max",
        type=int,
        default=0,
        help="SpecAugment maximum time mask frames",
    )
    parser.add_argument(
        "--sa_num_freq",
        type=int,
        default=0,
        help="SpecAugment number of frequency masks",
    )
    parser.add_argument(
        "--sa_num_time", type=int, default=0, help="SpecAugment number of time masks"
    )
    parser.add_argument(
        "--sa_p", type=float, default=0.0, help="SpecAugment probability"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of DataLoader workers"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save_interval", type=int, default=5, help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="Path to pre-trained checkpoint for weight initialization",
    )
    parser.add_argument(
        "--ema-decay",
        type=float,
        default=0.99,
        help="EMA decay rate. Set to 0 to disable EMA.",
    )
    add_beat_training_args(parser)
    add_chord_training_args(parser)

    args = parser.parse_args()

    # 1. 学習開始前に未使用の CUDA cache を解放する
    clear_startup_cuda_cache()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    amp_dtype = resolve_training_amp_dtype(device, use_amp=use_amp)
    use_grad_scaler = amp_dtype == torch.float16

    if args.wandb:
        if not HAS_WANDB:
            logger.warning(
                "wandb is not installed. Please `pip install wandb` to use it. Falling back to console logging."
            )
            args.wandb = False
        else:
            wandb.init(project=args.project_name, name=args.run_name, config=vars(args))

    os.makedirs(args.save_dir, exist_ok=True)

    logger.info("Initializing Dataset and DataLoader...")
    dataset = StemDataset(
        manifest_path=args.manifest_path,
        dataset_config_path=args.dataset_config,
        window_ms=args.window_ms,
        sample_rate=args.sample_rate,
        num_pitch_slots=args.num_pitch_slots,
        p_intra_drop=args.p_intra_drop,
        p_cross_mix=args.p_cross_mix,
        max_cross_stems=args.max_cross_stems,
        p_augment=args.p_augment,
        ir_folder=args.ir_folder,
        noise_folder=args.noise_folder,
        drum_folder=args.drum_folder,
        p_drum_mix=args.p_drum_mix,
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

    # beat用
    beat_dataset = None
    beat_dataloader = None
    beat_config = None
    if args.enable_beat_training:
        if os.path.exists(args.beat_dataset_path):
            if beat_dataset_has_wav_audio(args.beat_dataset_path):
                beat_config = beat_config_from_args(args)
                beat_dataset = BeatDataset(
                    args.beat_dataset_path,
                    window_ms=args.window_ms,
                    sample_rate=dataset.sample_rate,
                    hop_length=dataset.hop_length,
                    seed=args.seed,
                )
                beat_batch_size = (
                    args.batch_size
                    if args.beat_batch_size <= 0
                    else args.beat_batch_size
                )
                beat_num_workers = (
                    args.num_workers
                    if args.beat_num_workers < 0
                    else args.beat_num_workers
                )
                beat_dataloader = DataLoader(
                    beat_dataset,
                    batch_size=beat_batch_size,
                    shuffle=True,
                    collate_fn=beat_collate_fn,
                    num_workers=beat_num_workers,
                    pin_memory=False,
                )
                logger.info(
                    "Beat training enabled: %d songs, %d meter classes",
                    len(beat_dataset),
                    beat_dataset.num_meter_classes,
                )
            else:
                logger.warning(
                    "Beat dataset contains no wav files at %s/audio. Beat training disabled.",
                    args.beat_dataset_path,
                )
        else:
            logger.warning(
                "Beat dataset not found at %s. Beat training disabled.",
                args.beat_dataset_path,
            )

    # chord用
    chord_dataset = None
    chord_dataloader = None
    chord_config = None
    if args.enable_chord_training:
        if os.path.exists(args.chord_dataset_path):
            chord_config = chord_config_from_args(args)
            chord_dataset = ChordDataset(
                args.chord_dataset_path,
                window_ms=args.window_ms,
                sample_rate=dataset.sample_rate,
                hop_length=dataset.hop_length,
                seed=args.seed,
            )
            chord_batch_size = (
                args.batch_size if args.chord_batch_size <= 0 else args.chord_batch_size
            )
            chord_num_workers = (
                args.num_workers
                if args.chord_num_workers < 0
                else args.chord_num_workers
            )
            chord_dataloader = DataLoader(
                chord_dataset,
                batch_size=chord_batch_size,
                shuffle=True,
                collate_fn=chord_collate_fn,
                num_workers=chord_num_workers,
                pin_memory=False,
            )
            logger.info(
                "Chord training enabled: %d songs, %d root_chord classes",
                len(chord_dataset),
                chord_dataset.num_root_chord_classes,
            )
        else:
            logger.warning(
                "Chord dataset not found at %s. Chord training disabled.",
                args.chord_dataset_path,
            )

    logger.info("Initializing Model...")
    spec_augment_params = {
        "freq_mask_max": args.sa_freq_max,
        "time_mask_max": args.sa_time_max,
        "num_freq_masks": args.sa_num_freq,
        "num_time_masks": args.sa_num_time,
        "p": args.sa_p,
        "mask_fill_mode": "random",
    }
    config = SemiCRFModelConfig(
        sample_rate=dataset.sample_rate,
        hop_length=dataset.hop_length,
        n_fft=dataset.n_fft,
        num_pitch_slots=args.num_pitch_slots,
        spec_augment_params=spec_augment_params if args.sa_p > 0.0 else None,
        use_beat_head=beat_dataset is not None,
        num_meter_classes=(
            1 if beat_dataset is None else beat_dataset.num_meter_classes
        ),
        use_chord_head=chord_dataset is not None,
        num_root_chord_classes=(
            745 if chord_dataset is None else chord_dataset.num_root_chord_classes
        ),
    )
    model = AudioSemiCRFTransformer(config).to(device)
    beat_loss_fn = (
        BeatLoss(beat_config, beat_dataset.meter_class_counts).to(device)
        if beat_dataset is not None and beat_config is not None
        else None
    )
    chord_loss_fn = (
        ChordLoss(chord_config, chord_dataset.root_chord_counts).to(device)
        if chord_dataset is not None and chord_config is not None
        else None
    )

    if args.init_from:
        if os.path.exists(args.init_from):
            logger.info(f"Initializing weights from {args.init_from}...")
            checkpoint = torch.load(args.init_from, map_location=device)
            # 'model_state_dict'キーがあればそれを、なければ辞書全体をロード
            state_dict = checkpoint.get("model_state_dict", checkpoint)

            model_state = model.state_dict()
            for key in list(state_dict.keys()):
                if key in model_state:
                    if state_dict[key].shape != model_state[key].shape:
                        logger.warning(
                            "Shape mismatch for %s: checkpoint %s, model %s. Adapting...",
                            key, state_dict[key].shape, model_state[key].shape
                        )
                        ckpt_shape = state_dict[key].shape
                        mod_shape = model_state[key].shape
                        
                        if len(ckpt_shape) == 1:
                            min_len = min(ckpt_shape[0], mod_shape[0])
                            new_tensor = model_state[key].clone()
                            new_tensor[:min_len] = state_dict[key][:min_len]
                            state_dict[key] = new_tensor
                        elif len(ckpt_shape) == 2:
                            min_d0 = min(ckpt_shape[0], mod_shape[0])
                            min_d1 = min(ckpt_shape[1], mod_shape[1])
                            new_tensor = model_state[key].clone()
                            new_tensor[:min_d0, :min_d1] = state_dict[key][:min_d0, :min_d1]
                            state_dict[key] = new_tensor
                        else:
                            logger.warning("Unsupported shape mismatch dimension %s. Skipping key %s.", len(ckpt_shape), key)
                            del state_dict[key]

            incompatible = model.load_state_dict(state_dict, strict=False)
            if incompatible.missing_keys:
                logger.info(
                    "Missing keys while loading checkpoint, initialized randomly: %s",
                    incompatible.missing_keys,
                )
            if incompatible.unexpected_keys:
                logger.warning(
                    "Unexpected keys while loading checkpoint: %s",
                    incompatible.unexpected_keys,
                )
            logger.info("Weights initialized successfully.")
        else:
            logger.error(f"Checkpoint not found at {args.init_from}")
            raise FileNotFoundError(f"Checkpoint not found at {args.init_from}")

    if args.wandb:
        wandb.config.update({"model_config": asdict(config)})

    # EMA
    use_ema = args.ema_decay > 0.0
    ema_model = ModelEma(model, decay=args.ema_decay) if use_ema else None
    if use_ema:
        logger.info(f"EMA enabled (decay={args.ema_decay})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation_steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup_steps must be non-negative")
    if args.beat_update_interval <= 0:
        raise ValueError("--beat_update_interval must be positive")
    if args.chord_update_interval <= 0:
        raise ValueError("--chord_update_interval must be positive")

    def lr_lambda(step: int) -> float:
        if args.warmup_steps == 0:
            return 1.0
        return min((step + 1) / args.warmup_steps, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler(device.type) if use_grad_scaler else None

    def flush_optimizer_step() -> None:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer_step_was_skipped = scaler.get_scale() < scale_before_step
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer_step_was_skipped = False

        if not optimizer_step_was_skipped:
            scheduler.step()
            if ema_model is not None:
                ema_model.update(model)

        optimizer.zero_grad(set_to_none=True)

    logger.info(
        "Starting training on %s (AMP: %s, AMP dtype: %s, GradScaler: %s, warmup_steps: %s)",
        device,
        use_amp,
        amp_dtype,
        use_grad_scaler,
        args.warmup_steps,
    )

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        dataset.set_epoch(epoch)  # シードの更新
        beat_iterator = None
        if beat_dataset is not None and beat_dataloader is not None:
            beat_dataset.set_epoch(epoch)
            beat_iterator = iter(beat_dataloader)

        chord_iterator = None
        if chord_dataset is not None and chord_dataloader is not None:
            chord_dataset.set_epoch(epoch)
            chord_iterator = iter(chord_dataloader)

        epoch_loss = 0.0
        beat_epoch_loss = 0.0
        beat_step_count = 0
        chord_epoch_loss = 0.0
        chord_step_count = 0
        num_train_batches = len(dataloader)
        micro_steps_in_window = 0
        current_accumulation_steps = args.accumulation_steps
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")

        for batch_idx, batch in enumerate(progress_bar, start=1):
            if micro_steps_in_window == 0:
                remaining_batches = num_train_batches - batch_idx + 1
                current_accumulation_steps = min(
                    args.accumulation_steps, remaining_batches
                )

            step_index = global_step + 1
            audio = batch["audio"].to(device)
            valid_audio_frames = batch["valid_audio_frames"].to(device)

            # AMT / beat / chord を順に backward し、optimizer.step() は 1 回だけ行う。
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(audio, valid_audio_frames=valid_audio_frames)
                total_loss, loss_dict = compute_losses(
                    outputs, batch, args=args, model=model
                )

            loss_val = float(total_loss.item())
            total_loss = total_loss / current_accumulation_steps

            if scaler is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

            beat_loss_val = None
            beat_loss_dict = None
            if (
                beat_dataloader is not None
                and beat_iterator is not None
                and beat_loss_fn is not None
                and should_run_auxiliary_update(step_index, args.beat_update_interval)
            ):
                try:
                    beat_batch = next(beat_iterator)
                except StopIteration:
                    beat_iterator = iter(beat_dataloader)
                    beat_batch = next(beat_iterator)
                beat_total_loss, beat_loss_dict = compute_beat_batch_loss(
                    model=model,
                    batch=beat_batch,
                    loss_fn=beat_loss_fn,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    device=device,
                    loss_scale=args.beat_loss_scale,
                )

                beat_loss_val = float(beat_total_loss.item())
                beat_total_loss = beat_total_loss / current_accumulation_steps

                if scaler is not None:
                    scaler.scale(beat_total_loss).backward()
                else:
                    beat_total_loss.backward()
                beat_epoch_loss += beat_loss_val
                beat_step_count += 1

            chord_loss_val = None
            chord_loss_dict = None
            if (
                chord_dataloader is not None
                and chord_iterator is not None
                and chord_loss_fn is not None
                and should_run_auxiliary_update(step_index, args.chord_update_interval)
            ):
                try:
                    chord_batch = next(chord_iterator)
                except StopIteration:
                    chord_iterator = iter(chord_dataloader)
                    chord_batch = next(chord_iterator)
                chord_total_loss, chord_loss_dict = compute_chord_batch_loss(
                    model=model,
                    batch=chord_batch,
                    loss_fn=chord_loss_fn,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    device=device,
                    loss_scale=args.chord_loss_scale,
                )

                chord_loss_val = float(chord_total_loss.item())
                chord_total_loss = chord_total_loss / current_accumulation_steps

                if scaler is not None:
                    scaler.scale(chord_total_loss).backward()
                else:
                    chord_total_loss.backward()
                chord_epoch_loss += chord_loss_val
                chord_step_count += 1

            micro_steps_in_window += 1
            if micro_steps_in_window == current_accumulation_steps:
                flush_optimizer_step()
                micro_steps_in_window = 0

            epoch_loss += loss_val
            global_step += 1

            if args.wandb:
                wandb_log_dict = {
                    f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in loss_dict.items()
                }
                if beat_loss_dict is not None:
                    wandb_log_dict.update(
                        {
                            f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in beat_loss_dict.items()
                        }
                    )
                if chord_loss_dict is not None:
                    wandb_log_dict.update(
                        {
                            f"train/{k}": v.item() if isinstance(v, torch.Tensor) else v
                            for k, v in chord_loss_dict.items()
                        }
                    )
                wandb_log_dict["train/learning_rate"] = optimizer.param_groups[0]["lr"]
                wandb.log(wandb_log_dict, step=global_step)

            postfix = {"loss": f"{loss_val:.4f}"}
            if beat_loss_val is not None:
                postfix["beat_loss"] = f"{beat_loss_val:.4f}"
            if chord_loss_val is not None:
                postfix["chord_loss"] = f"{chord_loss_val:.4f}"
            progress_bar.set_postfix(postfix)

        avg_epoch_loss = epoch_loss / num_train_batches
        log_msg = f"Epoch {epoch} completed. Average Loss: {avg_epoch_loss:.4f}"
        if beat_step_count > 0:
            avg_beat_loss = beat_epoch_loss / beat_step_count
            log_msg += f", Beat Loss: {avg_beat_loss:.4f}"
        if chord_step_count > 0:
            avg_chord_loss = chord_epoch_loss / chord_step_count
            log_msg += f", Chord Loss: {avg_chord_loss:.4f}"
        logger.info(log_msg)

        if epoch % args.save_interval == 0 or epoch == args.epochs:
            checkpoint_path = os.path.join(
                args.save_dir, f"checkpoint_epoch_{epoch}.pth"
            )
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_epoch_loss,
                "model_config": asdict(config),
                "config": {
                    "model_config": asdict(config),
                    "args": vars(args),
                },
            }
            if ema_model is not None:
                save_dict["ema_state_dict"] = ema_model.module.state_dict()
            torch.save(save_dict, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    if args.wandb:
        wandb.finish()

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
