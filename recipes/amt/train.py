from __future__ import annotations

import argparse
import logging
import os
import random
from copy import deepcopy
from dataclasses import asdict
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from instrument_agnostic_amt.amt.modeling.checkpoints import (
    V1_BACKBONE_CONFIG_FIELDS,
    extract_model_config,
    load_checkpoint,
    load_compatible_weights,
)
from instrument_agnostic_amt.amt.modeling.model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)
from instrument_agnostic_amt.taxonomy.instrument_classes import NUM_INSTRUMENT_CLASSES

from .dataset import StemDataset
from .loss_config import AMTLossConfig
from .losses import compute_losses

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ModelEma(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        super().__init__()
        self.module = deepcopy(model).eval()
        self.decay = float(decay)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_value, model_value in zip(
            self.module.state_dict().values(), model.state_dict().values()
        ):
            source = model_value.to(ema_value.device)
            if torch.is_floating_point(ema_value):
                ema_value.copy_(self.decay * ema_value + (1.0 - self.decay) * source)
            else:
                ema_value.copy_(source)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated: dict[str, Any] = {
        "audio": torch.stack([item["audio"] for item in batch]),
        "frame_active_targets": torch.stack(
            [item["frame_active_targets"] for item in batch]
        ),
        "interval_targets": [item["interval_targets"] for item in batch],
        "valid_audio_frames": torch.tensor(
            [item.get("valid_audio_frames", item["audio"].shape[-1]) for item in batch],
            dtype=torch.long,
        ),
        "mask_instrument_loss": torch.tensor(
            [bool(item.get("mask_instrument_loss", False)) for item in batch],
            dtype=torch.bool,
        ),
        "cross_aug_density_skipped": torch.tensor(
            [int(item.get("cross_aug_density_skipped", 0)) for item in batch],
            dtype=torch.long,
        ),
    }
    frame_instrument = [item.get("frame_instrument_targets") for item in batch]
    if all(value is not None for value in frame_instrument):
        collated["frame_instrument_targets"] = torch.stack(frame_instrument)
    return collated


def parse_harmonics(value: str) -> tuple[float, ...]:
    harmonics = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not harmonics or any(harmonic <= 0.0 for harmonic in harmonics):
        raise argparse.ArgumentTypeError("harmonics must contain positive values")
    return harmonics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the V1 backbone with a V1 or overlap-capable V2 Semi-CRF"
    )
    parser.add_argument("--manifest_path", default="manifest.csv")
    parser.add_argument(
        "--dataset_config",
        default="configs/datasets/dataset_config_other.yaml",
    )
    parser.add_argument(
        "--semi-crf-version", "--semi_crf_version",
        choices=("v1", "v2"),
        default="v1",
        help="v1 keeps pitch/slot compatibility; v2 uses instrument-pitch tracks.",
    )
    parser.add_argument(
        "--compile-transformer",
        "--compile_transformer",
        choices=("off", "default", "reduce-overhead", "max-autotune"),
        default="off",
        help="バックボーンの時間/バンド Transformer だけを torch.compile する。",
    )
    parser.add_argument("--init-from", "--init_from", type=str, default=None)
    parser.add_argument("--freeze-backbone", "--freeze_backbone", action="store_true")
    parser.add_argument(
        "--freeze-stem-conv",
        action="store_true",
        help="Freeze only the V1 StemConv, matching the V1 training option.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--window_ms", type=int, default=8000)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--cqt_fmin", type=float, default=27.5)
    parser.add_argument("--cqt_n_bins", type=int, default=312)
    parser.add_argument("--cqt_bins_per_octave", type=int, default=36)
    parser.add_argument("--cqt_filter_scale", type=float, default=0.475)
    parser.add_argument(
        "--harmonics", type=parse_harmonics, default=(1.0, 2.0, 3.0, 4.0, 5.0)
    )
    parser.add_argument("--harmonic_dropout_p", type=float, default=0.0)
    parser.add_argument("--cqt_log_scale", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--encoder_num_layers", type=int, default=6)
    parser.add_argument("--encoder_num_heads", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-pitch-slots", "--num_pitch_slots", type=int, default=1)
    parser.add_argument("--semi_crf_head_dim", type=int, default=256)
    parser.add_argument("--instrument_pair_gate_dim", type=int, default=128)
    parser.add_argument(
        "--semi_crf_length_scaling", choices=("linear", "sqrt", "none"), default="none"
    )
    parser.add_argument("--semi_crf_length_penalty", type=float, default=0.0)
    parser.add_argument("--semi_crf_loss_weight", type=float, default=1.0)
    parser.add_argument("--semi_crf_false_negative_cost", type=float, default=0.0)
    parser.add_argument("--semi_crf_false_positive_cost", type=float, default=0.0)
    parser.add_argument("--semi_crf_track_batch_size", type=int, default=128)
    parser.add_argument(
        "--semi-crf-loss-backend",
        "--semi_crf_loss_backend",
        choices=("torch", "triton"),
        default="torch",
        help="Use the optional fused Triton log-partition on CUDA during training.",
    )
    parser.add_argument("--instrument_loss_weight", type=float, default=1.0)
    parser.add_argument("--instrument_loss_type", choices=("bce", "ce"), default="bce")
    parser.add_argument("--instrument_pair_train_topk", type=int, default=256)
    parser.add_argument("--instrument_pair_random_negatives", type=int, default=128)
    parser.add_argument("--instrument_pair_max_pairs", type=int, default=512)
    parser.add_argument("--pair_gate_loss_weight", type=float, default=1.0)
    parser.add_argument("--interval_presence_loss_weight", type=float, default=1.0)
    parser.add_argument("--interval_offset_loss_weight", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project_name", default="instrument_agnostic_amt_semicrf")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--ir_folder", default=None)
    parser.add_argument("--noise_folder", default=None)
    parser.add_argument("--drum_folder", default=None)
    parser.add_argument("--p_drum_mix", type=float, default=0.1)
    parser.add_argument("--p_augment", type=float, default=0.5)
    parser.add_argument("--p_intra_drop", type=float, default=0.3)
    parser.add_argument("--p_cross_mix", type=float, default=0.5)
    parser.add_argument("--p_cross_mix_decay", type=float, default=0.3)
    parser.add_argument("--max_cross_stems", type=int, default=5)
    parser.add_argument("--max_cross_aug_positive_pairs", type=int, default=0)
    parser.add_argument("--max_cross_aug_intervals", type=int, default=1000)
    parser.add_argument("--sa_freq_max", type=int, default=0)
    parser.add_argument("--sa_time_max", type=int, default=0)
    parser.add_argument("--sa_num_freq", type=int, default=0)
    parser.add_argument("--sa_num_time", type=int, default=0)
    parser.add_argument("--sa_p", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--ema-decay", "--ema_decay", type=float, default=0.99)
    parser.add_argument("--disable-gradient-checkpoint", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "batch_size",
        "accumulation_steps",
        "epochs",
        "window_ms",
        "sample_rate",
        "n_fft",
        "hop_length",
        "cqt_n_bins",
        "cqt_bins_per_octave",
        "hidden_size",
        "base_ch",
        "encoder_num_layers",
        "encoder_num_heads",
        "num_pitch_slots",
        "semi_crf_head_dim",
        "semi_crf_track_batch_size",
        "instrument_pair_max_pairs",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.hidden_size % args.encoder_num_heads != 0:
        raise ValueError("hidden_size must be divisible by encoder_num_heads")
    if (args.hidden_size // args.encoder_num_heads) % 2 != 0:
        raise ValueError("hidden_size / encoder_num_heads must be even for RoPE")
    if not 0.0 <= args.harmonic_dropout_p < 1.0:
        raise ValueError("harmonic_dropout_p must be in [0, 1)")
    if args.instrument_pair_train_topk < 0 or args.instrument_pair_random_negatives < 0:
        raise ValueError("instrument-pair negative counts must be non-negative")
    if args.semi_crf_loss_backend == "triton" and not torch.cuda.is_available():
        raise ValueError("Triton Semi-CRF loss requires CUDA")


def _source_value(
    source_config: Mapping[str, Any] | None, name: str, fallback: Any
) -> Any:
    if source_config is None:
        return fallback
    return source_config.get(name, fallback)


def build_model_config(
    args: argparse.Namespace,
    *,
    source_config: Mapping[str, Any] | None,
) -> SemiCRFModelConfig:
    values = {
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "n_fft": args.n_fft,
        "cqt_fmin": args.cqt_fmin,
        "cqt_n_bins": args.cqt_n_bins,
        "cqt_bins_per_octave": args.cqt_bins_per_octave,
        "cqt_filter_scale": args.cqt_filter_scale,
        "harmonics": args.harmonics,
        "cqt_log_scale": args.cqt_log_scale,
        "input_audio_channels": 2,
        "hidden_size": args.hidden_size,
        "base_ch": args.base_ch,
        "encoder_num_layers": args.encoder_num_layers,
        "encoder_num_heads": args.encoder_num_heads,
        "dropout": args.dropout,
        "pitch_query_count": 88,
        "use_beat_head": False,
        "use_chord_head": False,
    }
    if source_config is not None:
        if (
            "semi_crf_version" not in source_config
            and "instrument_pair_gate_dim" in source_config
            and "num_pitch_slots" not in source_config
        ):
            raise ValueError("Previous V2 checkpoints cannot initialize this architecture")
        for name in V1_BACKBONE_CONFIG_FIELDS:
            if name in source_config:
                values[name] = source_config[name]
        logger.info("Using V1 backbone configuration from --init-from")

    head_dim = int(_source_value(source_config, "semi_crf_head_dim", args.semi_crf_head_dim))
    spec_augment_params = (
        {
            "freq_mask_max": args.sa_freq_max,
            "time_mask_max": args.sa_time_max,
            "num_freq_masks": args.sa_num_freq,
            "num_time_masks": args.sa_num_time,
            "p": args.sa_p,
            "mask_fill_mode": "random",
        }
        if args.sa_p > 0.0
        else None
    )
    slots = (
        int(_source_value(source_config, "num_pitch_slots", args.num_pitch_slots))
        if args.semi_crf_version == "v1"
        else 1
    )
    return SemiCRFModelConfig(
        **values,
        semi_crf_version=args.semi_crf_version,
        harmonic_dropout_p=args.harmonic_dropout_p,
        spec_augment_params=spec_augment_params,
        use_gradient_checkpoint=not args.disable_gradient_checkpoint,
        num_pitch_slots=slots,
        semi_crf_head_dim=head_dim,
        semi_crf_length_scaling=args.semi_crf_length_scaling,
        semi_crf_length_penalty=args.semi_crf_length_penalty,
        num_instrument_classes=NUM_INSTRUMENT_CLASSES,
        instrument_pair_gate_dim=args.instrument_pair_gate_dim,
    )


def drop_unsearchable_path_entries() -> list[str]:
    """PATH から探索できないエントリを落とし、落としたものを返す。

    torch 2.13 の inductor は生成コードのコメント用に codegen 中へ無条件で
    ``nvcc --version`` を実行するが、torch 側は FileNotFoundError しか捕捉していない。
    WSL では PATH に紛れ込んだ実在しない Windows 側ディレクトリのせいで execvp が
    ENOENT ではなく EACCES を返し、PermissionError でコンパイル全体が落ちる。
    nvcc 自体はコンパイルに不要（Triton が ptxas を同梱している）。
    """
    entries = os.environ.get("PATH", "").split(os.pathsep)
    searchable = [
        entry for entry in entries if os.path.isdir(entry) and os.access(entry, os.X_OK)
    ]
    dropped = [entry for entry in entries if entry not in searchable]
    if dropped:
        os.environ["PATH"] = os.pathsep.join(searchable)
    return dropped


def compile_backbone_transformers(model: AudioSemiCRFTransformer, mode: str) -> None:
    """バックボーンの時間軸/バンド軸 Transformer だけを torch.compile する。

    プロファイル上 1 step の 65% が backward で、その大半は Transformer の
    elementwise 演算（RMSNorm・RoPE・残差・FFN）だった。matmul 律速ではないので
    カーネル融合がそのまま効き、RTX 4070 Ti で 1 step が約 1.7 倍速くなる。
    Semi-CRF ロスは動的な形状を持つ自前 DP なのでコンパイル対象にしない。

    nn.Module.compile は state_dict のキーを変えないため、チェックポイント互換性は保たれる。
    """
    for time_transformer, band_transformer in model.backbone.layers:
        time_transformer.compile(mode=mode)
        band_transformer.compile(mode=mode)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    source_checkpoint = load_checkpoint(args.init_from) if args.init_from else None
    source_config = (
        extract_model_config(source_checkpoint) if source_checkpoint is not None else None
    )
    config = build_model_config(args, source_config=source_config)
    loss_config = AMTLossConfig(
        semi_crf_loss_weight=args.semi_crf_loss_weight,
        semi_crf_false_negative_cost=args.semi_crf_false_negative_cost,
        semi_crf_false_positive_cost=args.semi_crf_false_positive_cost,
        semi_crf_track_batch_size=args.semi_crf_track_batch_size,
        semi_crf_loss_backend=args.semi_crf_loss_backend,
        interval_presence_loss_weight=args.interval_presence_loss_weight,
        interval_offset_loss_weight=args.interval_offset_loss_weight,
        instrument_loss_weight=args.instrument_loss_weight,
        instrument_loss_type=args.instrument_loss_type,
        instrument_pair_train_topk=args.instrument_pair_train_topk,
        instrument_pair_random_negatives=args.instrument_pair_random_negatives,
        instrument_pair_max_pairs=args.instrument_pair_max_pairs,
        pair_gate_loss_weight=args.pair_gate_loss_weight,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = not args.no_amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16

    if args.wandb:
        if not HAS_WANDB:
            logger.warning("wandb is not installed; disabling it")
            args.wandb = False
        else:
            wandb.init(project=args.project_name, name=args.run_name, config=vars(args))

    dataset = StemDataset(
        manifest_path=args.manifest_path,
        dataset_config_path=args.dataset_config,
        window_ms=args.window_ms,
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        semi_crf_version=config.semi_crf_version,
        num_pitch_slots=config.num_pitch_slots,
        p_intra_drop=args.p_intra_drop,
        p_cross_mix=args.p_cross_mix,
        p_cross_mix_decay=args.p_cross_mix_decay,
        max_cross_stems=args.max_cross_stems,
        max_cross_aug_positive_pairs=args.max_cross_aug_positive_pairs,
        max_cross_aug_intervals=args.max_cross_aug_intervals,
        p_augment=args.p_augment,
        ir_folder=args.ir_folder,
        noise_folder=args.noise_folder,
        drum_folder=args.drum_folder,
        p_drum_mix=args.p_drum_mix,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    model = AudioSemiCRFTransformer(config)
    if source_checkpoint is not None:
        report = load_compatible_weights(
            model, source_checkpoint, prefer_ema=False, require_complete_backbone=True
        )
        logger.info(
            "Initialized %d backbone and %d head tensors; skipped %d unexpected and %d shape-mismatched tensors",
            len(report.loaded_backbone_keys),
            len(report.loaded_keys) - len(report.loaded_backbone_keys),
            len(report.unexpected_keys),
            len(report.shape_mismatches),
        )
    model.to(device)
    if args.freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
    elif args.freeze_stem_conv:
        for parameter in model.backbone.stem.parameters():
            parameter.requires_grad = False

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            1.0 if args.warmup_steps <= 0 else min((step + 1) / args.warmup_steps, 1.0)
        ),
    )
    scaler = torch.amp.GradScaler(device.type) if use_scaler else None
    # EMA は非コンパイルのコピーを持たせたいので、複製してから compile する。
    ema_model = ModelEma(model, args.ema_decay) if args.ema_decay > 0.0 else None
    if args.compile_transformer != "off":
        dropped = drop_unsearchable_path_entries()
        if dropped:
            logger.info("Dropped %d unsearchable PATH entries: %s", len(dropped), dropped)
        compile_backbone_transformers(model, args.compile_transformer)
        logger.info(
            "Compiled backbone transformers (mode=%s); the first steps include compilation",
            args.compile_transformer,
        )
    os.makedirs(args.save_dir, exist_ok=True)
    if args.wandb:
        wandb.config.update({"model_config": asdict(config)})

    def optimizer_step() -> None:
        if scaler is not None:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if scaler is None:
            optimizer.step()
            skipped = False
        else:
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped = scaler.get_scale() < before
        if not skipped:
            scheduler.step()
            if ema_model is not None:
                ema_model.update(model)
        optimizer.zero_grad(set_to_none=True)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    logger.info(
        "Training Semi-CRF %s on %s with %s loss backend",
        config.semi_crf_version,
        device,
        args.semi_crf_loss_backend,
    )
    for epoch in range(1, args.epochs + 1):
        model.train()
        dataset.set_epoch(epoch)
        epoch_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}")
        accumulated = 0
        accumulation_target = args.accumulation_steps
        for batch_index, batch in enumerate(progress, start=1):
            if accumulated == 0:
                accumulation_target = min(
                    args.accumulation_steps, len(loader) - batch_index + 1
                )
            audio = batch["audio"].to(device)
            valid_frames = batch["valid_audio_frames"].to(device)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                outputs = model(
                    audio,
                    valid_audio_frames=valid_frames,
                    include_aux_outputs=False,
                )
                loss, metrics = compute_losses(
                    outputs,
                    batch,
                    config=loss_config,
                    model=model,
                )
            if scaler is None:
                (loss / accumulation_target).backward()
            else:
                scaler.scale(loss / accumulation_target).backward()
            accumulated += 1
            if accumulated == accumulation_target:
                optimizer_step()
                accumulated = 0

            loss_value = float(loss.detach().item())
            epoch_loss += loss_value
            global_step += 1
            progress.set_postfix(
                loss=f"{loss_value:.4f}",
                tracks=int(metrics["semi_crf_track_count"].item()),
                intervals=int(metrics["semi_crf_interval_count"].item()),
            )
            if args.wandb:
                wandb.log(
                    {
                        f"train/{key}": value.item()
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in metrics.items()
                    },
                    step=global_step,
                )

        average_loss = epoch_loss / max(1, len(loader))
        if epoch % args.save_interval == 0 or epoch == args.epochs:
            path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth")
            saved: dict[str, Any] = {
                "checkpoint_format_version": 2,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": average_loss,
                "model_config": asdict(config),
                "config": {"model_config": asdict(config), "args": vars(args)},
            }
            if ema_model is not None:
                saved["ema_state_dict"] = ema_model.module.state_dict()
            torch.save(saved, path)
            logger.info("Saved %s", path)
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
