from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from instrument_agnostic_amt.beat_chord import (
    MidiFrameBeatChordModel,
    MidiFrameModelConfig,
)
from .datasets import (
    MidiAugmentConfig,
    MidiBeatDataset,
    MidiChordDataset,
    MidiKeyOnlyDataset,
    MeterAwareCropConfig,
    midi_beat_collate_fn,
    midi_chord_collate_fn,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_BEAT_DATASET_PATH = Path("beat_chord_dataset/beat_dataset")
DEFAULT_CHORD_DATASET_PATH = Path("beat_chord_dataset/chord_dataset")
DEFAULT_KEY_ONLY_DATASET_PATH = Path("beat_chord_dataset/key_only_dataset")
DEFAULT_MIDI_DIR = Path("midi_dataset/merged")


def is_key_only_training_step(
    *,
    completed_steps: int,
    interval: int,
) -> bool:
    """Return whether the next training step should include key-only data."""

    if interval <= 0:
        raise ValueError("interval must be positive")
    return (int(completed_steps) + 1) % int(interval) == 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a beat/chord model from scratch on MIDI-frame inputs."
    )
    parser.add_argument(
        "--beat_dataset_path",
        type=Path,
        default=DEFAULT_BEAT_DATASET_PATH,
        help="Path to beat dataset root containing audio/ and label/.",
    )
    parser.add_argument(
        "--chord_dataset_path",
        type=Path,
        default=DEFAULT_CHORD_DATASET_PATH,
        help="Path to chord dataset root containing audio/, chord_label/, and key_label/.",
    )
    parser.add_argument(
        "--midi_dir",
        type=Path,
        default=DEFAULT_MIDI_DIR,
        help="Directory containing merged AMT MIDI files named <song>.mid.",
    )
    parser.add_argument(
        "--key_only_dataset_path",
        type=Path,
        default=DEFAULT_KEY_ONLY_DATASET_PATH,
        help=(
            "Path to corrected prediction MIDIs under midis/. Only "
            "key_signature events are used as labels."
        ),
    )
    parser.add_argument(
        "--skip_key_only",
        action="store_true",
        help="Disable corrected key-only auxiliary training.",
    )
    parser.add_argument("--skip_beat", action="store_true", help="Skip beat training.")
    parser.add_argument(
        "--skip_chord", action="store_true", help="Skip chord training."
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
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs.")
    parser.add_argument(
        "--window_ms",
        type=int,
        default=25000,
        help="Training crop length in milliseconds.",
    )
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
    parser.add_argument("--beat_batch_size", type=int, default=0)
    parser.add_argument("--chord_batch_size", type=int, default=0)
    parser.add_argument("--key_only_batch_size", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--beat_num_workers", type=int, default=-1)
    parser.add_argument("--chord_num_workers", type=int, default=-1)
    parser.add_argument("--key_only_num_workers", type=int, default=-1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--beat_loss_scale", type=float, default=1.0)
    parser.add_argument("--midi_pitch_shift_min", type=int, default=-5)
    parser.add_argument("--midi_pitch_shift_max", type=int, default=6)
    parser.add_argument("--midi_time_stretch_min", type=float, default=0.8)
    parser.add_argument("--midi_time_stretch_max", type=float, default=1.2)
    parser.add_argument(
        "--midi_rubato_prob",
        type=float,
        default=0.5,
        help="Probability of applying non-uniform tempo rubato to a MIDI window.",
    )
    parser.add_argument(
        "--midi_rubato_strength",
        type=float,
        default=0.12,
        help="Maximum local timing-slope variation (0.12 is about +/-12%%).",
    )
    parser.add_argument(
        "--midi_rubato_period_sec",
        type=float,
        default=4.0,
        help="Approximate duration of one compensated rubato gesture.",
    )
    parser.add_argument("--midi_drop_drum_prob", type=float, default=0.5)
    parser.add_argument("--midi_drop_note_prob", type=float, default=0.05)
    parser.add_argument("--chord_loss_scale", type=float, default=1.0)
    parser.add_argument("--key_only_loss_scale", type=float, default=1.0)
    parser.add_argument("--chord_modulation_prob", type=float, default=0.0)
    parser.add_argument("--chord_boundary_loss_tolerance", type=int, default=1)
    parser.add_argument("--key_boundary_loss_tolerance", type=int, default=8)
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
    parser.add_argument("--major_grouping_accent_loss_weight", type=float, default=0.25)
    parser.add_argument("--major_grouping_accent_temperature", type=float, default=0.5)
    parser.add_argument(
        "--meter_aware_crop_probability",
        type=float,
        default=0.75,
        help=(
            "Probability of centering a beat crop on a complete bar whose "
            "meter requires major grouping. Set to 0 to disable."
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
        default=Path("beat_chord_checkpoints/midi_frame"),
    )
    parser.add_argument("--save_interval", type=int, default=1)
    parser.add_argument("--max_steps_per_epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--project_name",
        type=str,
        default="midi_frame_beat_chord",
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
    parser.add_argument(
        "--time_mask_prob",
        type=float,
        default=0.5,
        help="Probability of applying time masking data augmentation during training (0.0 to 1.0).",
    )
    parser.add_argument(
        "--time_mask_duration_ms",
        type=float,
        default=1000.0,
        help="Duration of the time mask in milliseconds (default: 1000.0ms = 1s).",
    )
    parser.add_argument(
        "--key_only_step_interval",
        type=int,
        default=10,
        help=(
            "Use one corrected key-only batch every N training steps. "
            "Set to 1 for the previous every-step behavior."
        ),
    )
    return parser.parse_args()


def resolve_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.resume_from is None:
        return None
    resume_from = args.resume_from.resolve()
    if not resume_from.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_from}")
    return resume_from


def resolve_init_checkpoint(args: argparse.Namespace) -> Path | None:
    if args.init_from is None:
        return None
    init_from = args.init_from.resolve()
    if not init_from.exists():
        raise FileNotFoundError(f"Init checkpoint not found: {init_from}")
    return init_from


def next_batch(iterator: Any, dataloader: Any) -> tuple[Any, Any]:
    try:
        batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(dataloader)
        batch = next(iterator)
        return batch, iterator


def build_model_config(
    *,
    args: argparse.Namespace,
    beat_dataset: Any | None,
    chord_dataset: Any | None,
) -> MidiFrameModelConfig:
    num_meter_classes = (
        1 if beat_dataset is None else int(beat_dataset.num_meter_classes)
    )
    num_root_chord_classes = (
        745 if chord_dataset is None else int(chord_dataset.num_root_chord_classes)
    )
    return MidiFrameModelConfig(
        sample_rate=int(args.sample_rate),
        hop_length=int(args.hop_length),
        pitch_min=int(args.pitch_min),
        pitch_max=int(args.pitch_max),
        num_input_channels=72,
        base_ch=int(args.base_ch),
        hidden_size=int(args.hidden_size),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        num_global_tokens=int(args.num_global_tokens),
        dropout=float(args.dropout),
        num_meter_classes=num_meter_classes,
        num_root_chord_classes=num_root_chord_classes,
        inter_refine_layers=tuple(args.inter_refine_layers)
        if args.inter_refine_layers
        else (),
        inter_loss_weight=float(args.inter_loss_weight),
        time_mask_prob=float(args.time_mask_prob),
        time_mask_duration_ms=float(args.time_mask_duration_ms),
    )


def compute_beat_batch_loss(
    *,
    model: MidiFrameBeatChordModel,
    batch: dict[str, Any],
    loss_fn: Any,
    use_amp: bool,
    amp_dtype: Any,
    device: Any,
    loss_scale: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        outputs = model(
            batch["midi_frames"],
            include_beat=True,
            include_chord=False,
            feedback_beat=True,
            feedback_chord=True,
            detach_chord_feedback=True,
        )
        total_loss, loss_dict = loss_fn(outputs, batch)

        # 中間予測ロスの平均化と加算
        inter_outputs_dict = outputs.get("intermediate_outputs")
        if inter_outputs_dict:
            inter_losses = []
            for layer_str, inter_outputs in inter_outputs_dict.items():
                if "beat" in inter_outputs:
                    inter_loss_val, inter_loss_components = loss_fn(
                        inter_outputs["beat"], batch
                    )
                    inter_losses.append(inter_loss_val)
                    for key, val in inter_loss_components.items():
                        loss_dict[f"inter_L{layer_str}_{key}"] = val

            if len(inter_losses) > 0:
                avg_inter_loss = torch.stack(inter_losses).mean()
                loss_dict["inter_beat_loss_avg"] = avg_inter_loss
                inter_weight = float(model.config.inter_loss_weight)
                total_loss = total_loss + avg_inter_loss * inter_weight

        total_loss = total_loss * loss_scale
    return total_loss, loss_dict


def compute_chord_batch_loss(
    *,
    model: MidiFrameBeatChordModel,
    batch: dict[str, Any],
    loss_fn: Any,
    use_amp: bool,
    amp_dtype: Any,
    device: Any,
    loss_scale: float,
    detach_chord_feedback: bool = False,
) -> tuple[Any, dict[str, Any]]:
    import torch

    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    with torch.amp.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=use_amp,
    ):
        outputs = model(
            batch["midi_frames"],
            include_beat=False,
            include_chord=True,
            feedback_beat=True,
            feedback_chord=True,
            detach_beat_feedback=True,
            detach_chord_feedback=detach_chord_feedback,
        )
        total_loss, loss_dict = loss_fn(outputs, batch)

        # 中間予測ロスの平均化と加算
        inter_outputs_dict = outputs.get("intermediate_outputs")
        if inter_outputs_dict:
            inter_losses = []
            for layer_str, inter_outputs in inter_outputs_dict.items():
                if "chord" in inter_outputs:
                    inter_loss_val, inter_loss_components = loss_fn(
                        inter_outputs["chord"], batch
                    )
                    inter_losses.append(inter_loss_val)
                    for key, val in inter_loss_components.items():
                        loss_dict[f"inter_L{layer_str}_{key}"] = val

            if len(inter_losses) > 0:
                avg_inter_loss = torch.stack(inter_losses).mean()
                loss_dict["inter_chord_loss_avg"] = avg_inter_loss
                inter_weight = float(model.config.inter_loss_weight)
                total_loss = total_loss + avg_inter_loss * inter_weight

        total_loss = total_loss * loss_scale
    return total_loss, loss_dict


def main() -> None:
    args = parse_arguments()
    if args.skip_beat and args.skip_chord:
        raise ValueError("Nothing to train: both --skip_beat and --skip_chord were set")
    if args.resume_from is not None and args.init_from is not None:
        raise ValueError("--resume-from and --init-from cannot be used together")
    if args.accumulation_steps <= 0:
        raise ValueError("--accumulation_steps must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.key_only_loss_scale < 0.0:
        raise ValueError("--key_only_loss_scale must be non-negative")

    if args.key_only_step_interval <= 0:
        raise ValueError("--key_only_step_interval must be positive")

    set_seed(args.seed)

    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from .datasets.modulation import (
        ModulationAugmentConfig,
    )
    from .beat import BeatConfig, BeatLoss
    from .chord import ChordLoss, chord_config_from_args

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
        extra_beat_meter_classes = checkpoint_meter_classes(checkpoint)
    else:
        checkpoint = None
        extra_beat_meter_classes = ()

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

    beat_dataset = None
    beat_dataloader = None
    beat_loss_fn = None
    if not args.skip_beat:
        beat_dataset = MidiBeatDataset(
            args.beat_dataset_path.resolve(),
            midi_dir=args.midi_dir.resolve(),
            window_ms=args.window_ms,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            num_input_channels=72,
            seed=args.seed,
            augment_config=midi_augment_config,
            meter_aware_crop_config=meter_aware_crop_config,
            extra_meter_classes=extra_beat_meter_classes,
        )
        beat_batch_size = (
            args.batch_size if args.beat_batch_size <= 0 else args.beat_batch_size
        )
        beat_num_workers = (
            args.num_workers if args.beat_num_workers < 0 else args.beat_num_workers
        )
        beat_dataloader = DataLoader(
            beat_dataset,
            batch_size=beat_batch_size,
            shuffle=True,
            collate_fn=midi_beat_collate_fn,
            num_workers=beat_num_workers,
            pin_memory=False,
        )
        beat_loss_fn = BeatLoss(
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
            beat_dataset.meter_class_counts,
            meter_classes=beat_dataset.meter_classes,
        ).to(device)
        logger.info(
            "Beat training enabled: %d songs, %d meter classes, "
            "meter-aware crop probability %.2f",
            len(beat_dataset),
            beat_dataset.num_meter_classes,
            meter_aware_crop_config.probability,
        )

    chord_dataset = None
    chord_dataloader = None
    key_only_dataset = None
    key_only_dataloader = None
    chord_loss_fn = None
    if not args.skip_chord:
        chord_dataset = MidiChordDataset(
            args.chord_dataset_path.resolve(),
            midi_dir=args.midi_dir.resolve(),
            window_ms=args.window_ms,
            sample_rate=args.sample_rate,
            hop_length=args.hop_length,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            num_input_channels=72,
            seed=args.seed,
            modulation_config=ModulationAugmentConfig(
                prob=args.chord_modulation_prob,
            ),
            augment_config=midi_augment_config,
        )
        chord_batch_size = (
            args.batch_size if args.chord_batch_size <= 0 else args.chord_batch_size
        )
        chord_num_workers = (
            args.num_workers if args.chord_num_workers < 0 else args.chord_num_workers
        )
        chord_dataloader = DataLoader(
            chord_dataset,
            batch_size=chord_batch_size,
            shuffle=True,
            collate_fn=midi_chord_collate_fn,
            num_workers=chord_num_workers,
            pin_memory=False,
        )
        chord_loss_fn = ChordLoss(
            chord_config_from_args(args),
            chord_dataset.root_chord_counts,
        ).to(device)
        logger.info(
            "Chord training enabled: %d songs, %d root-chord classes",
            len(chord_dataset),
            chord_dataset.num_root_chord_classes,
        )
        if not args.skip_key_only:
            key_only_dataset = MidiKeyOnlyDataset(
                args.key_only_dataset_path.resolve(),
                window_ms=args.window_ms,
                sample_rate=args.sample_rate,
                hop_length=args.hop_length,
                pitch_min=args.pitch_min,
                pitch_max=args.pitch_max,
                num_input_channels=72,
                seed=args.seed,
                augment_config=midi_augment_config,
            )
            key_only_batch_size = (
                chord_batch_size
                if args.key_only_batch_size <= 0
                else args.key_only_batch_size
            )
            key_only_num_workers = (
                args.num_workers
                if args.key_only_num_workers < 0
                else args.key_only_num_workers
            )
            key_only_dataloader = DataLoader(
                key_only_dataset,
                batch_size=key_only_batch_size,
                shuffle=True,
                collate_fn=midi_chord_collate_fn,
                num_workers=key_only_num_workers,
                pin_memory=False,
            )
            logger.info(
                "Key-only training enabled: %d corrected songs, %d crops",
                key_only_dataset.num_songs,
                len(key_only_dataset),
            )

    model_config = build_model_config(
        args=args,
        beat_dataset=beat_dataset,
        chord_dataset=chord_dataset,
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
        logger.info("Training from scratch.")

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
        beat_dataset=beat_dataset,
        chord_dataset=chord_dataset,
        trainable_parameter_count=trainable_parameter_count,
        key_only_dataset=key_only_dataset,
    )

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 総学習ステップ数を計算（ウォームアップ後のコサイン減衰用）
    steps_per_epoch = max(
        len(beat_dataloader) if beat_dataloader else 0,
        len(chord_dataloader) if chord_dataloader else 0,
        len(key_only_dataloader) if key_only_dataloader else 0,
    )
    total_steps = steps_per_epoch * args.epochs

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))

        # ウォームアップ終了後のコサインアニーリング減衰
        progress = float(step - args.warmup_steps) / float(
            max(1, total_steps - args.warmup_steps)
        )
        import math

        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        # 最低学習率は初期学習率の 1% に設定
        eta_min = 0.01
        return eta_min + (1.0 - eta_min) * cosine_decay

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
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
        logger.info("Resuming from epoch %d", start_epoch)

    steps_per_epoch = max(
        len(beat_dataloader) if beat_dataloader is not None else 0,
        len(chord_dataloader) if chord_dataloader is not None else 0,
        len(key_only_dataloader) if key_only_dataloader is not None else 0,
    )
    if steps_per_epoch <= 0:
        raise ValueError("No training steps available. Check dataset settings.")
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        if beat_dataset is not None:
            beat_dataset.set_epoch(epoch)
        if chord_dataset is not None:
            chord_dataset.set_epoch(epoch)
        if key_only_dataset is not None:
            key_only_dataset.set_epoch(epoch)

        beat_iterator = iter(beat_dataloader) if beat_dataloader is not None else None
        chord_iterator = (
            iter(chord_dataloader) if chord_dataloader is not None else None
        )
        key_only_iterator = (
            iter(key_only_dataloader) if key_only_dataloader is not None else None
        )

        total_epoch_loss = 0.0
        beat_epoch_loss = 0.0
        chord_epoch_loss = 0.0
        key_only_epoch_loss = 0.0
        beat_step_count = 0
        chord_step_count = 0
        key_only_step_count = 0
        performed_steps = 0
        micro_steps_in_window = 0
        current_accumulation_steps = args.accumulation_steps
        progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}")

        for step_in_epoch in progress_bar:
            if micro_steps_in_window == 0:
                remaining_steps = steps_per_epoch - step_in_epoch
                current_accumulation_steps = min(
                    args.accumulation_steps,
                    remaining_steps,
                )

            step_total_raw_loss = 0.0
            beat_loss_val = None
            chord_loss_val = None
            key_only_loss_val = None
            optimizer_step_was_skipped = None

            if beat_iterator is not None and beat_loss_fn is not None:
                beat_batch, beat_iterator = next_batch(beat_iterator, beat_dataloader)

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
                step_total_raw_loss += beat_loss_val
                beat_epoch_loss += beat_loss_val
                beat_step_count += 1
                backward_loss = beat_total_loss / current_accumulation_steps
                if scaler is not None:
                    scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()
            else:
                beat_loss_dict = None

            if chord_iterator is not None and chord_loss_fn is not None:
                chord_batch, chord_iterator = next_batch(
                    chord_iterator,
                    chord_dataloader,
                )

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
                step_total_raw_loss += chord_loss_val
                chord_epoch_loss += chord_loss_val
                chord_step_count += 1
                backward_loss = chord_total_loss / current_accumulation_steps
                if scaler is not None:
                    scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()
            else:
                chord_loss_dict = None

            key_only_batch = None
            if (
                key_only_iterator is not None
                and chord_loss_fn is not None
                and is_key_only_training_step(
                    completed_steps=global_step,
                    interval=args.key_only_step_interval,
                )
            ):
                try:
                    key_only_batch = next(key_only_iterator)
                except StopIteration:
                    key_only_iterator = None
            if key_only_batch is not None and chord_loss_fn is not None:
                key_only_total_loss, key_only_loss_dict = compute_chord_batch_loss(
                    model=model,
                    batch=key_only_batch,
                    loss_fn=chord_loss_fn,
                    use_amp=use_amp,
                    amp_dtype=amp_dtype,
                    device=device,
                    loss_scale=args.key_only_loss_scale,
                    detach_chord_feedback=True,
                )
                key_only_loss_val = float(key_only_total_loss.item())
                step_total_raw_loss += key_only_loss_val
                key_only_epoch_loss += key_only_loss_val
                key_only_step_count += 1
                backward_loss = key_only_total_loss / current_accumulation_steps
                if scaler is not None:
                    scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()
            else:
                key_only_loss_dict = None

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
            total_epoch_loss += step_total_raw_loss

            if wandb_run is not None:
                wandb_log_dict: dict[str, Any] = {
                    "train/total_loss": float(step_total_raw_loss),
                    "train/epoch": int(epoch),
                    "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                if beat_loss_val is not None:
                    wandb_log_dict["train/beat_scaled_total_loss"] = float(
                        beat_loss_val
                    )
                if chord_loss_val is not None:
                    wandb_log_dict["train/chord_scaled_total_loss"] = float(
                        chord_loss_val
                    )
                if key_only_loss_val is not None:
                    wandb_log_dict["train/key_only_scaled_total_loss"] = float(
                        key_only_loss_val
                    )
                if beat_loss_dict is not None:
                    wandb_log_dict.update(
                        prefix_metric_dict("train/beat/", beat_loss_dict)
                    )
                if chord_loss_dict is not None:
                    wandb_log_dict.update(
                        prefix_metric_dict("train/chord/", chord_loss_dict)
                    )
                if key_only_loss_dict is not None:
                    wandb_log_dict.update(
                        prefix_metric_dict("train/key_only/", key_only_loss_dict)
                    )
                if optimizer_step_was_skipped is not None:
                    wandb_log_dict["train/optimizer_step_skipped"] = int(
                        optimizer_step_was_skipped
                    )
                wandb_run.log(wandb_log_dict, step=global_step)

            postfix: dict[str, str] = {"loss": f"{step_total_raw_loss:.4f}"}
            if beat_loss_val is not None:
                postfix["beat"] = f"{beat_loss_val:.4f}"
            if chord_loss_val is not None:
                postfix["chord"] = f"{chord_loss_val:.4f}"
            if key_only_loss_val is not None:
                postfix["key"] = f"{key_only_loss_val:.4f}"
            progress_bar.set_postfix(postfix)

        avg_total_loss = total_epoch_loss / max(1, performed_steps)
        avg_beat_loss = (
            beat_epoch_loss / beat_step_count if beat_step_count > 0 else None
        )
        avg_chord_loss = (
            chord_epoch_loss / chord_step_count if chord_step_count > 0 else None
        )
        avg_key_only_loss = (
            key_only_epoch_loss / key_only_step_count
            if key_only_step_count > 0
            else None
        )
        log_message = f"Epoch {epoch} completed. Avg loss: {avg_total_loss:.4f}"
        if avg_beat_loss is not None:
            log_message += f", Beat: {avg_beat_loss:.4f}"
        if avg_chord_loss is not None:
            log_message += f", Chord: {avg_chord_loss:.4f}"
        if avg_key_only_loss is not None:
            log_message += f", Key-only: {avg_key_only_loss:.4f}"
        logger.info(log_message)

        if wandb_run is not None:
            epoch_log_dict: dict[str, Any] = {
                "epoch/avg_total_loss": float(avg_total_loss),
                "epoch/index": int(epoch),
                "epoch/performed_steps": int(performed_steps),
            }
            if avg_beat_loss is not None:
                epoch_log_dict["epoch/avg_beat_loss"] = float(avg_beat_loss)
            if avg_chord_loss is not None:
                epoch_log_dict["epoch/avg_chord_loss"] = float(avg_chord_loss)
            if avg_key_only_loss is not None:
                epoch_log_dict["epoch/avg_key_only_loss"] = float(avg_key_only_loss)
            wandb_run.log(epoch_log_dict, step=global_step)

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
                avg_beat_loss=avg_beat_loss,
                avg_chord_loss=avg_chord_loss,
                ema_model=ema_model,
                avg_key_only_loss=avg_key_only_loss,
                beat_meter_classes=(
                    None if beat_dataset is None else beat_dataset.meter_classes
                ),
            )
            if wandb_run is not None:
                wandb_run.summary["last_checkpoint"] = str(checkpoint_path)
                wandb_run.summary["last_checkpoint_epoch"] = int(epoch)

    if wandb_run is not None:
        wandb_run.finish()
    logger.info("MIDI-frame training complete.")


if __name__ == "__main__":
    main()
