from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any

from instrument_agnostic_amt.beat_chord import MidiFrameBeatChordModel
from .datasets import (
    MidiAugmentConfig,
    MidiBeatPretrainDataset,
    MeterAwareCropConfig,
    midi_beat_pretrain_collate_fn,
    read_beat_label_meter_classes,
)
from .training_utils import (
    ModelEma,
    checkpoint_meter_classes,
    clear_startup_cuda_cache,
    initialize_wandb_run,
    load_model_state,
    log_model_state_load_report,
    prefix_metric_dict,
    resolve_training_amp_dtype,
    save_checkpoint,
    select_checkpoint_model_state,
    set_seed,
)
from .train import (
    build_model_config,
    compute_beat_batch_loss,
    resolve_checkpoint,
    resolve_init_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_PRETRAIN_MIDI_DIR = Path("beat_chord_dataset/beat_pretrain_dataset/midis")
DEFAULT_BEAT_DATASET_PATH = Path("beat_chord_dataset/beat_dataset")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain the MIDI-frame beat head from MIDI tempo/signature maps."
    )
    parser.add_argument(
        "--pretrain_midi_dir",
        type=Path,
        default=DEFAULT_PRETRAIN_MIDI_DIR,
        help="Directory containing beat pretrain .mid files.",
    )
    parser.add_argument(
        "--beat_dataset_path",
        type=Path,
        default=DEFAULT_BEAT_DATASET_PATH,
        help="Existing beat dataset root. Its meter classes are added to pretrain.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume checkpoint for optimizer/scheduler/model state.",
    )
    parser.add_argument(
        "--init-from",
        dest="init_from",
        type=Path,
        default=None,
        help=(
            "Initialize model weights from a checkpoint without resuming optimizer "
            "state. Tensors with different shapes are skipped."
        ),
    )
    parser.add_argument(
        "--load_ema",
        action="store_true",
        help="Load EMA weights from the checkpoint when available.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--window_ms", type=int, default=25000)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--pitch_min", type=int, default=21)
    parser.add_argument("--pitch_max", type=int, default=108)
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_global_tokens", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--beat_loss_scale", type=float, default=1.0)
    parser.add_argument("--midi_pitch_shift_min", type=int, default=-5)
    parser.add_argument("--midi_pitch_shift_max", type=int, default=6)
    parser.add_argument("--midi_time_stretch_min", type=float, default=0.9)
    parser.add_argument("--midi_time_stretch_max", type=float, default=1.1)
    parser.add_argument(
        "--midi_rubato_prob",
        type=float,
        default=0.5,
        help="Probability of applying non-uniform tempo rubato to a MIDI window.",
    )
    parser.add_argument(
        "--midi_rubato_strength",
        type=float,
        default=0.10,
        help="Maximum local timing-slope variation (0.10 is about +/-10%%).",
    )
    parser.add_argument(
        "--midi_rubato_period_sec",
        type=float,
        default=4.0,
        help="Approximate duration of one compensated rubato gesture.",
    )
    parser.add_argument("--midi_drop_drum_prob", type=float, default=0.5)
    parser.add_argument("--midi_drop_note_prob", type=float, default=0.05)
    parser.add_argument("--downbeat_pos_weight", type=float, default=20.0)
    parser.add_argument("--beat_pos_weight", type=float, default=5.0)
    parser.add_argument("--meter_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--meter_grid_ranking_loss_weight",
        "--grid_consistency_loss_weight",
        type=float,
        default=1.0,
        help="Weight for bar-level ranking between meter-derived beat grids.",
    )
    parser.add_argument("--meter_grid_ranking_margin", type=float, default=0.1)
    parser.add_argument("--meter_grid_kl_loss_weight", type=float, default=1.0)
    parser.add_argument("--meter_grid_kl_temperature", type=float, default=0.2)
    parser.add_argument("--beat_loss_tolerance", type=int, default=1)
    parser.add_argument("--major_grouping_loss_weight", type=float, default=1.0)
    parser.add_argument(
        "--major_grouping_accent_loss_weight", type=float, default=0.25
    )
    parser.add_argument(
        "--major_grouping_accent_temperature", type=float, default=0.5
    )
    parser.add_argument(
        "--meter_aware_crop_probability",
        type=float,
        default=0.75,
        help=(
            "Probability of centering a crop on a complete bar whose meter "
            "requires major grouping. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--meter_aware_crop_rarity_power",
        type=float,
        default=0.5,
        help="Inverse-frequency exponent used to choose among eligible meters.",
    )
    parser.add_argument(
        "--meter_aware_crop_margin_frames",
        type=int,
        default=2,
        help="Required context frames outside both target downbeats.",
    )
    parser.add_argument("--disable_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("beat_chord_checkpoints/midi_frame_beat_pretrain"),
    )
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="Use only the first N MIDI files. 0 means all files.",
    )
    parser.add_argument(
        "--strict_midi",
        action="store_true",
        help="Fail on invalid MIDI files instead of skipping them with a warning.",
    )
    parser.add_argument(
        "--invalid_midi_cache_path",
        type=Path,
        default=None,
        help=(
            "Path to invalid MIDI cache JSON. "
            "Defaults to <pretrain_midi_dir>/../invalid_midi_cache.json."
        ),
    )
    parser.add_argument(
        "--disable_invalid_midi_cache",
        action="store_true",
        help="Do not read or write the invalid MIDI cache.",
    )
    parser.add_argument(
        "--midi_metadata_cache_path",
        type=Path,
        default=None,
        help=(
            "Path to valid MIDI metadata cache JSON. "
            "Defaults to <pretrain_midi_dir>/../midi_metadata_cache.json."
        ),
    )
    parser.add_argument(
        "--disable_midi_metadata_cache",
        action="store_true",
        help="Do not read or write the valid MIDI metadata cache.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--project_name",
        type=str,
        default="midi_frame_beat_pretrain",
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--inter_refine_layers",
        type=int,
        nargs="*",
        default=[1, 2],
        help="0-indexed layer indices after which intermediate prediction is performed.",
    )
    parser.add_argument(
        "--inter_loss_weight",
        type=float,
        default=0.5,
        help="Loss weight for averaged intermediate predictions.",
    )
    parser.add_argument("--time_mask_prob", type=float, default=0.5)
    parser.add_argument("--time_mask_duration_ms", type=float, default=1000.0)
    return parser.parse_args()


def build_pretrain_meter_classes(
    *,
    args: argparse.Namespace,
    checkpoint: dict[str, Any] | None,
) -> tuple[tuple[int, int], ...]:
    """既存 beat dataset と resume checkpoint の meter class を pretrain に持ち込む。"""

    meter_classes = set(read_beat_label_meter_classes(args.beat_dataset_path))
    meter_classes.update(checkpoint_meter_classes(checkpoint))
    return tuple(sorted(meter_classes))


def build_lr_scheduler(
    *,
    optimizer: Any,
    warmup_steps: int,
    total_optimizer_steps: int,
) -> Any:
    """pretrain 用の warmup + cosine decay scheduler を作る。"""

    import torch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(
            max(1, total_optimizer_steps - warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        eta_min = 0.01
        return eta_min + (1.0 - eta_min) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def main() -> None:
    args = parse_arguments()
    if args.resume_from is not None and args.init_from is not None:
        raise ValueError("--resume-from and --init-from cannot be used together")
    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation_steps must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.max_files < 0:
        raise ValueError("--max_files must be non-negative")

    set_seed(args.seed)

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from .beat import BeatConfig, BeatLoss

    clear_startup_cuda_cache(torch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    amp_dtype = resolve_training_amp_dtype(
        torch_module=torch,
        device=device,
        use_amp=use_amp,
    )
    use_grad_scaler = amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type) if use_grad_scaler else None

    resume_from = resolve_checkpoint(args)
    init_from = resolve_init_checkpoint(args)
    model_state_checkpoint_path = resume_from or init_from
    if model_state_checkpoint_path is not None:
        checkpoint = torch.load(
            model_state_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        logger.info(
            "Loaded pretrain checkpoint metadata from %s",
            model_state_checkpoint_path,
        )
    else:
        checkpoint = None

    midi_augment_config = MidiAugmentConfig(
        pitch_shift_min=args.midi_pitch_shift_min,
        pitch_shift_max=args.midi_pitch_shift_max,
        time_stretch_min=args.midi_time_stretch_min,
        time_stretch_max=args.midi_time_stretch_max,
        rubato_prob=args.midi_rubato_prob,
        rubato_strength=args.midi_rubato_strength,
        rubato_period_sec=args.midi_rubato_period_sec,
        drop_drum_prob=args.midi_drop_drum_prob,
        drop_note_prob=args.midi_drop_note_prob,
    )
    meter_aware_crop_config = MeterAwareCropConfig(
        probability=args.meter_aware_crop_probability,
        rarity_power=args.meter_aware_crop_rarity_power,
        boundary_margin_frames=args.meter_aware_crop_margin_frames,
    )
    extra_meter_classes = build_pretrain_meter_classes(
        args=args,
        checkpoint=checkpoint,
    )
    invalid_midi_cache_path = None
    if not args.disable_invalid_midi_cache:
        invalid_midi_cache_path = (
            args.invalid_midi_cache_path
            if args.invalid_midi_cache_path is not None
            else args.pretrain_midi_dir.resolve().parent / "invalid_midi_cache.json"
        )
    midi_metadata_cache_path = None
    if not args.disable_midi_metadata_cache:
        midi_metadata_cache_path = (
            args.midi_metadata_cache_path
            if args.midi_metadata_cache_path is not None
            else args.pretrain_midi_dir.resolve().parent / "midi_metadata_cache.json"
        )

    dataset = MidiBeatPretrainDataset(
        args.pretrain_midi_dir.resolve(),
        window_ms=args.window_ms,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
        num_input_channels=72,
        seed=args.seed,
        augment_config=midi_augment_config,
        meter_aware_crop_config=meter_aware_crop_config,
        extra_meter_classes=extra_meter_classes,
        max_files=args.max_files,
        skip_invalid=not args.strict_midi,
        invalid_midi_cache_path=invalid_midi_cache_path,
        midi_metadata_cache_path=midi_metadata_cache_path,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=midi_beat_pretrain_collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    loss_fn = BeatLoss(
        BeatConfig(
            downbeat_pos_weight=args.downbeat_pos_weight,
            beat_pos_weight=args.beat_pos_weight,
            meter_loss_weight=args.meter_loss_weight,
            meter_grid_ranking_loss_weight=args.meter_grid_ranking_loss_weight,
            meter_grid_ranking_margin=args.meter_grid_ranking_margin,
            meter_grid_kl_loss_weight=args.meter_grid_kl_loss_weight,
            meter_grid_kl_temperature=args.meter_grid_kl_temperature,
            loss_tolerance=args.beat_loss_tolerance,
            major_grouping_loss_weight=args.major_grouping_loss_weight,
            major_grouping_accent_loss_weight=(
                args.major_grouping_accent_loss_weight
            ),
            major_grouping_accent_temperature=(
                args.major_grouping_accent_temperature
            ),
        ),
        dataset.meter_class_counts,
        meter_classes=dataset.meter_classes,
    ).to(device)
    logger.info(
        "Beat pretrain enabled: %d MIDI files, %d meter classes, "
        "meter-aware crop probability %.2f",
        len(dataset),
        dataset.num_meter_classes,
        meter_aware_crop_config.probability,
    )

    model_config = build_model_config(
        args=args,
        beat_dataset=dataset,
        chord_dataset=None,
    )
    model = MidiFrameBeatChordModel(model_config).to(device)
    if checkpoint is not None:
        state_dict = select_checkpoint_model_state(
            checkpoint,
            load_ema=bool(args.load_ema),
        )
        report = load_model_state(
            model=model,
            state_dict=state_dict,
            allow_shape_mismatch=init_from is not None,
        )
        log_model_state_load_report(
            logger=logger,
            source_path=model_state_checkpoint_path,
            report=report,
            allow_shape_mismatch=init_from is not None,
        )
        if resume_from is not None:
            logger.info("Loaded model state from %s", resume_from)
        else:
            logger.info("Initialized model state from %s", init_from)
    else:
        logger.info("Beat pretraining from scratch.")

    trainable_params = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_parameter_count = sum(parameter.numel() for parameter in trainable_params)
    logger.info("Trainable parameters: %d", trainable_parameter_count)

    wandb_run = initialize_wandb_run(
        args=args,
        model_config=model_config,
        device=device,
        use_amp=use_amp,
        beat_dataset=dataset,
        chord_dataset=None,
        trainable_parameter_count=trainable_parameter_count,
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    steps_per_epoch = len(dataloader)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError("No pretrain steps available. Check dataset settings.")

    optimizer_steps_per_epoch = math.ceil(
        float(steps_per_epoch) / float(args.accumulation_steps)
    )
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=args.warmup_steps,
        total_optimizer_steps=optimizer_steps_per_epoch * args.epochs,
    )
    ema_model = (
        None
        if args.disable_ema or args.ema_decay <= 0.0
        else ModelEma(model, decay=args.ema_decay)
    )

    start_epoch = 1
    global_step = 0
    if resume_from is not None and checkpoint is not None:
        if isinstance(checkpoint.get("optimizer_state_dict"), dict):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if isinstance(checkpoint.get("scheduler_state_dict"), dict):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if ema_model is not None and isinstance(checkpoint.get("ema_state_dict"), dict):
            ema_model.module.load_state_dict(checkpoint["ema_state_dict"], strict=False)
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        logger.info("Resuming beat pretrain from epoch %d", start_epoch)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        dataset.set_epoch(epoch)

        total_epoch_loss = 0.0
        performed_steps = 0
        micro_steps_in_window = 0
        current_accumulation_steps = args.accumulation_steps
        progress_bar = tqdm(
            range(steps_per_epoch),
            desc=f"Epoch {epoch}/{args.epochs}",
        )
        data_iterator = iter(dataloader)

        for step_in_epoch in progress_bar:
            if micro_steps_in_window == 0:
                remaining_steps = steps_per_epoch - step_in_epoch
                current_accumulation_steps = min(
                    args.accumulation_steps,
                    remaining_steps,
                )

            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                batch = next(data_iterator)

            total_loss, loss_dict = compute_beat_batch_loss(
                model=model,
                batch=batch,
                loss_fn=loss_fn,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                device=device,
                loss_scale=args.beat_loss_scale,
            )
            loss_value = float(total_loss.item())
            backward_loss = total_loss / current_accumulation_steps
            if scaler is not None:
                scaler.scale(backward_loss).backward()
            else:
                backward_loss.backward()

            optimizer_step_was_skipped = None
            micro_steps_in_window += 1
            if micro_steps_in_window == current_accumulation_steps:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    scale_before_step = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step_was_skipped = scaler.get_scale() < scale_before_step
                else:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()
                    optimizer_step_was_skipped = False

                if not optimizer_step_was_skipped:
                    scheduler.step()
                    if ema_model is not None:
                        ema_model.update(model)

                optimizer.zero_grad(set_to_none=True)
                micro_steps_in_window = 0

            performed_steps += 1
            global_step += 1
            total_epoch_loss += loss_value

            if wandb_run is not None:
                wandb_log_dict: dict[str, Any] = {
                    "train/total_loss": loss_value,
                    "train/epoch": int(epoch),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                wandb_log_dict.update(prefix_metric_dict("train/beat/", loss_dict))
                if optimizer_step_was_skipped is not None:
                    wandb_log_dict["train/optimizer_step_skipped"] = int(
                        optimizer_step_was_skipped
                    )
                wandb_run.log(wandb_log_dict, step=global_step)

            progress_bar.set_postfix({"loss": f"{loss_value:.4f}"})

        avg_total_loss = total_epoch_loss / max(1, performed_steps)
        logger.info(
            "Epoch %d completed. Avg beat pretrain loss: %.4f",
            epoch,
            avg_total_loss,
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch/avg_total_loss": float(avg_total_loss),
                    "epoch/index": int(epoch),
                    "epoch/performed_steps": int(performed_steps),
                },
                step=global_step,
            )

        should_save = (
            args.save_interval > 0 and epoch % args.save_interval == 0
        ) or epoch == args.epochs
        if should_save:
            checkpoint_path = args.save_dir.resolve() / f"checkpoint_epoch_{epoch}.pth"
            save_checkpoint(
                torch_module=torch,
                checkpoint_path=checkpoint_path,
                epoch=epoch,
                global_step=global_step,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                model_config=model_config,
                args=args,
                avg_total_loss=avg_total_loss,
                avg_beat_loss=avg_total_loss,
                avg_chord_loss=None,
                ema_model=ema_model,
                beat_meter_classes=dataset.meter_classes,
            )
            if wandb_run is not None:
                wandb_run.summary["last_checkpoint"] = str(checkpoint_path)
                wandb_run.summary["last_checkpoint_epoch"] = int(epoch)

    if wandb_run is not None:
        wandb_run.finish()
    logger.info("MIDI-frame beat pretraining complete.")


if __name__ == "__main__":
    main()
