"""学習の CLI。

典型的な使い方は、AMT の学習済み重みで backbone を初期化してから学習を始める::

    python -m recipes.instrument_refinement.train --init-amt checkpoints/best_model.pth

チェックポイントは毎エポック（--save-every）と、スコア更新時に best_model.pth を保存する。
--dry-run は 1 エポック 1 ステップだけ流して、設定とデータの通り道を確認する用。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from instrument_agnostic_amt.instrument_refinement.modeling.checkpoints import (
    load_amt_backbone,
    save_refinement_checkpoint,
)
from instrument_agnostic_amt.instrument_refinement.modeling.model import (
    InstrumentRefinementConfig,
    InstrumentRefinementModel,
    _coerce_config_values,
    load_refinement_model_config,
)
from .collate import collate_refinement_batch
from .dataset import (
    ClassBalancedSampler,
    ManifestInstrumentRefinementDataset,
)
from .forward import forward_refinement_batch, move_batch_to_device
from .losses import (
    InstrumentRefinementLossConfig,
    compute_refinement_losses,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REFINEMENT_ROOT = PROJECT_ROOT / "instrument_agnostic_amt" / "instrument_refinement"
DEFAULT_MANIFEST = REFINEMENT_ROOT / "artifacts" / "datasets" / "manifest.csv"
DEFAULT_MODEL_CONFIG = REFINEMENT_ROOT / "configs" / "model.json"
DEFAULT_OUTPUT = REFINEMENT_ROOT / "artifacts" / "checkpoints"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the drum-free post-AMT instrument refinement model."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    # --init-amt: AMT の重みで backbone を初期化して新規学習を始める。
    # --resume:   このモデルの学習を途中から再開する。両立しない。
    parser.add_argument("--init-amt", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--prefer-non-ema", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--hop-seconds", type=float, default=4.0)
    # 1 サンプルにつき同じ音源から切り出す窓の数。一貫性 loss と対照学習に必要なので 2 以上推奨。
    parser.add_argument("--windows-per-source", type=int, default=2)
    # 実際のバッチサイズは batch_size * windows_per_source になる点に注意。
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    # backbone は学習済みなので、ヘッドより 1 桁小さい学習率で微調整する。
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sources", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--no-balanced-sampling", action="store_true")
    parser.add_argument(
        "--dataset-weights",
        type=Path,
        default=None,
        metavar="FILE",
        help=(
            "JSON or YAML file mapping dataset name to sampling weight. Without it a "
            "class is drawn from its datasets in proportion to their source counts, "
            "which buries small datasets (real_recordings is 0.2%% of orchestral_woodwind). "
            "Supplying weights switches every class to weighted draws; datasets missing "
            "from the file default to 1.0 and a weight of 0 removes one."
        ),
    )
    parser.add_argument(
        "--dataset-weight",
        action="append",
        default=[],
        metavar="DATASET=WEIGHT",
        help="Override one entry of --dataset-weights, e.g. --dataset-weight real_recordings=3.",
    )
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--random-gain-db", type=float, default=3.0)
    parser.add_argument("--composite-probability", type=float, default=0.5)
    parser.add_argument(
        "--cross-song-composite-ratio",
        type=float,
        default=0.5,
        help=(
            "Probability that a mixture partner comes from a different song. "
            "Same-song data rarely contains two different instruments in one stem "
            "(for example no song has acoustic and distorted guitar together), so "
            "crossing songs is what makes multi-instrument stems learnable. Set 0 "
            "to keep mixtures inside one song."
        ),
    )
    parser.add_argument(
        "--max-mixture-sources",
        type=int,
        default=None,
        help="Optional cap for sources mixed inside one inference stem group.",
    )
    parser.add_argument(
        "--ambiguous-note-onset-ms",
        type=float,
        default=30.0,
        help="Merge same-pitch cross-source notes within this onset tolerance.",
    )
    parser.add_argument(
        "--augment-probability",
        type=float,
        default=0.25,
        help=(
            "Probability of running AudioAugmentor (EQ, micro pitch shift, chorus/phaser, "
            "reverb, noise) on each source before it is mixed, matching how the main AMT "
            "StemDataset augments stems. Costs roughly one extra pass over the window per "
            "source, so raise it only if the timbre robustness is worth the slower epoch. "
            "0 disables it and skips importing audiomentations entirely."
        ),
    )
    parser.add_argument(
        "--ir-folder",
        default=None,
        help="Impulse response folder for convolution reverb during augmentation.",
    )
    parser.add_argument(
        "--noise-folder",
        default=None,
        help="Background noise folder for augmentation. Without it only Gaussian noise is added.",
    )
    parser.add_argument("--max-contrastive-notes", type=int, default=256)
    parser.add_argument("--window-class-weight", type=float, default=1.0)
    parser.add_argument("--note-class-weight", type=float, default=0.5)
    parser.add_argument("--family-weight", type=float, default=0.3)
    parser.add_argument("--contrastive-weight", type=float, default=0.2)
    parser.add_argument("--consistency-weight", type=float, default=0.2)
    parser.add_argument("--agreement-weight", type=float, default=0.1)
    args = parser.parse_args()
    if args.resume is not None and args.init_amt is not None:
        parser.error("--resume and --init-amt cannot be used together")
    if args.freeze_backbone and args.resume is None and args.init_amt is None:
        parser.error("--freeze-backbone requires --resume or --init-amt")
    if args.max_steps_per_epoch is not None and args.max_steps_per_epoch <= 0:
        parser.error("--max-steps-per-epoch must be positive")
    if args.max_mixture_sources is not None and args.max_mixture_sources < 2:
        parser.error("--max-mixture-sources must be at least two")
    if args.ambiguous_note_onset_ms < 0.0:
        parser.error("--ambiguous-note-onset-ms must be nonnegative")
    if args.max_contrastive_notes <= 0:
        parser.error("--max-contrastive-notes must be positive")
    if not 0.0 <= args.cross_song_composite_ratio <= 1.0:
        parser.error("--cross-song-composite-ratio must be in [0, 1]")
    return args


def _coerce_weight(name: str, value: Any, source: str) -> float:
    try:
        weight = float(value)
    except (TypeError, ValueError):
        raise SystemExit(
            f"{source}: weight for {name!r} is not a number: {value!r}"
        ) from None
    if weight < 0.0:
        raise SystemExit(
            f"{source}: weight for {name!r} must be non-negative, got {weight}"
        )
    return weight


def load_dataset_weights(path: Path | str | None) -> dict[str, float]:
    """データセット重みのファイルを読む。拡張子で JSON / YAML を切り替える。

    中身は ``{データセット名: 重み}`` の 1 段の対応表。
    """
    if path is None:
        return {}
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"--dataset-weights file not found: {file}")
    text = file.read_text(encoding="utf-8")
    if file.suffix.lower() in (".yaml", ".yml"):
        import yaml

        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    # 重みだけを書いた表でも、"weights:" の下にまとめた表でも読めるようにする。
    if isinstance(raw, Mapping) and isinstance(raw.get("weights"), Mapping):
        raw = raw["weights"]
    if not isinstance(raw, Mapping):
        raise SystemExit(f"{file}: expected a mapping of dataset name to weight")
    return {
        str(name): _coerce_weight(str(name), value, str(file))
        for name, value in raw.items()
    }


def parse_dataset_weights(values: Iterable[str]) -> dict[str, float]:
    """``--dataset-weight name=w`` の並びを dict にする。空なら重み付けしない。"""
    weights: dict[str, float] = {}
    for value in values:
        name, separator, raw = str(value).partition("=")
        if not separator or not name:
            raise SystemExit(f"--dataset-weight expects DATASET=WEIGHT, got {value!r}")
        weights[name] = _coerce_weight(name, raw, "--dataset-weight")
    return weights


def resolve_dataset_weights(
    args: argparse.Namespace, known: Iterable[str]
) -> dict[str, float]:
    """ファイルとコマンドラインの重みを合成し、名前の綴りを検証する。

    存在しないデータセット名は黙って無視すると効果が出ないまま学習が回るので、
    まとめて弾く。
    """
    weights = load_dataset_weights(getattr(args, "dataset_weights", None))
    weights.update(parse_dataset_weights(args.dataset_weight))
    unknown = sorted(set(weights) - set(known))
    if unknown:
        raise SystemExit(
            f"Unknown dataset name(s) in weights: {', '.join(unknown)}. Known: {', '.join(sorted(known))}"
        )
    return weights


def _make_dataset(
    args: argparse.Namespace,
    config: InstrumentRefinementConfig,
    *,
    split: str,
) -> ManifestInstrumentRefinementDataset:
    return ManifestInstrumentRefinementDataset(
        args.manifest,
        split=split,
        sample_rate=config.sample_rate,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        sampling="random" if split == "train" else "fixed",
        windows_per_source=args.windows_per_source,
        require_midi=True,
        random_gain_db=args.random_gain_db if split == "train" else 0.0,
        composite_probability=(args.composite_probability if split == "train" else 0.0),
        cross_song_composite_ratio=(
            args.cross_song_composite_ratio if split == "train" else 0.0
        ),
        max_mixture_sources=args.max_mixture_sources,
        ambiguous_note_onset_tolerance_seconds=(args.ambiguous_note_onset_ms / 1000.0),
        max_sources=args.max_sources,
        # 検証を毎エポック同じ音で測りたいので、拡張は学習 split だけに掛ける。
        augment_probability=(args.augment_probability if split == "train" else 0.0),
        ir_folder=args.ir_folder,
        noise_folder=args.noise_folder,
    )


def _make_loader(
    dataset: ManifestInstrumentRefinementDataset,
    args: argparse.Namespace,
    *,
    training: bool,
) -> DataLoader:
    """DataLoader を作る。学習時は既定でクラス均衡サンプリングを使う。"""
    sampler = None
    shuffle = training
    if training and not args.no_balanced_sampling:
        dataset_weights = resolve_dataset_weights(
            args, {source.dataset for source in dataset.sources}
        )
        if dataset_weights:
            print(f"Dataset sampling weights: {dict(sorted(dataset_weights.items()))}")
        sampler = ClassBalancedSampler(
            dataset.class_to_unit_indices,
            num_samples=len(dataset),
            seed=args.seed,
            class_dataset_unit_indices=dataset.class_dataset_unit_indices
            if dataset_weights
            else None,
            dataset_weights=dataset_weights,
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        collate_fn=collate_refinement_batch,
    )


def _run_epoch(
    model: InstrumentRefinementModel,
    loader: DataLoader,
    *,
    device: torch.device,
    window_seconds: float,
    loss_config: InstrumentRefinementLossConfig,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    grad_clip: float,
    max_steps: int | None,
    description: str,
) -> dict[str, float]:
    """1 エポック分を回す。optimizer が None なら検証（勾配を計算しない）。"""
    training = optimizer is not None
    model.train(training)
    sums: dict[str, float] = {}
    step_count = 0
    total = min(len(loader), max_steps) if max_steps is not None else len(loader)
    progress: Iterable[dict[str, Any]] = tqdm(loader, total=total, desc=description)
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
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                outputs = forward_refinement_batch(
                    model, batch, window_seconds=window_seconds
                )
                loss, metrics = compute_refinement_losses(
                    outputs, batch, config=loss_config
                )
            # NaN/Inf のまま学習を続けると重みが壊れるので、その場で止める。
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite refinement loss")
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
                sums[key] = sums.get(key, 0.0) + float(value.item())
            if isinstance(progress, tqdm):
                progress.set_postfix(
                    loss=f"{float(loss.detach()):.4f}",
                    acc=f"{float(metrics['window_accuracy']):.3f}",
                )
    if step_count == 0:
        raise RuntimeError("The DataLoader produced no batches")
    return {key: value / step_count for key, value in sums.items()}


def _build_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[
    InstrumentRefinementModel, InstrumentRefinementConfig, dict[str, Any] | None
]:
    """モデルを用意する。--resume なら設定と重みをチェックポイントから復元する。"""
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        raw_config = resume_checkpoint.get("model_config")
        if not isinstance(raw_config, Mapping):
            raise ValueError("Resume checkpoint does not contain model_config")
        config = InstrumentRefinementConfig(**_coerce_config_values(raw_config))
    else:
        config = load_refinement_model_config(args.model_config)
    model = InstrumentRefinementModel(config)
    if args.init_amt is not None:
        report = load_amt_backbone(
            model, args.init_amt, prefer_ema=not args.prefer_non_ema
        )
        print(f"loaded {len(report.loaded_keys)} AMT backbone tensors")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
    if args.freeze_backbone:
        model.backbone.requires_grad_(False)
    model.to(device)
    return model, config, resume_checkpoint


def _build_optimizer(
    model: InstrumentRefinementModel,
    args: argparse.Namespace,
    resume_checkpoint: Mapping[str, Any] | None,
) -> torch.optim.Optimizer:
    """backbone とヘッドで学習率を変えるため、パラメータを 2 群に分けて AdamW を作る。"""
    backbone_parameters = [
        parameter
        for parameter in model.backbone.parameters()
        if parameter.requires_grad
    ]
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_parameter_ids
    ]
    parameter_groups = []
    if backbone_parameters:
        parameter_groups.append(
            {"params": backbone_parameters, "lr": args.backbone_learning_rate}
        )
    if head_parameters:
        parameter_groups.append({"params": head_parameters, "lr": args.learning_rate})
    optimizer = torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)
    if resume_checkpoint is not None and "optimizer_state_dict" in resume_checkpoint:
        try:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        except ValueError:
            print(
                "optimizer parameter groups changed (for example after unfreezing "
                "the backbone); starting with a fresh optimizer"
            )
    return optimizer


def _format_epoch_line(
    epoch: int,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float] | None,
) -> str:
    line = f"epoch={epoch + 1} train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['window_accuracy']:.3f}"
    if validation_metrics is None:
        return line
    return (
        line + f" validation_loss={validation_metrics['loss']:.4f} "
        f"validation_acc={validation_metrics['window_accuracy']:.3f}"
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, resume_checkpoint = _build_model(args, device)

    train_loader = _make_loader(
        _make_dataset(args, config, split="train"), args, training=True
    )
    validation_loader = None
    if not args.skip_validation:
        validation_loader = _make_loader(
            _make_dataset(args, config, split="validation"), args, training=False
        )
    optimizer = _build_optimizer(model, args, resume_checkpoint)
    use_amp = bool(not args.no_amp and device.type == "cuda")
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp) if device.type == "cuda" else None
    )
    loss_config = InstrumentRefinementLossConfig(
        window_class_weight=args.window_class_weight,
        note_class_weight=args.note_class_weight,
        family_weight=args.family_weight,
        contrastive_weight=args.contrastive_weight,
        consistency_weight=args.consistency_weight,
        note_window_agreement_weight=args.agreement_weight,
        max_contrastive_notes=args.max_contrastive_notes,
    )
    start_epoch = (
        int(resume_checkpoint.get("epoch", -1)) + 1 if resume_checkpoint else 0
    )
    epoch_count = 1 if args.dry_run else args.epochs
    max_steps = 1 if args.dry_run else args.max_steps_per_epoch
    best_validation = float("inf")
    for epoch in range(start_epoch, start_epoch + epoch_count):
        train_metrics = _run_epoch(
            model,
            train_loader,
            device=device,
            window_seconds=args.window_seconds,
            loss_config=loss_config,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip=args.grad_clip,
            max_steps=max_steps,
            description=f"train {epoch + 1}",
        )
        validation_metrics = None
        if validation_loader is not None:
            validation_metrics = _run_epoch(
                model,
                validation_loader,
                device=device,
                window_seconds=args.window_seconds,
                loss_config=loss_config,
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
                grad_clip=args.grad_clip,
                max_steps=max_steps if args.dry_run else None,
                description=f"validation {epoch + 1}",
            )
        print(_format_epoch_line(epoch, train_metrics, validation_metrics))
        if args.dry_run:
            continue
        extra = {
            "loss_config": asdict(loss_config),
            "training_args": vars(args),
            "train_metrics": train_metrics,
            "validation_metrics": validation_metrics,
        }
        destinations = []
        if (epoch + 1) % args.save_every == 0:
            destinations.append(args.output_dir / f"epoch_{epoch + 1:03d}.pth")
        # best の基準は検証 loss。--skip-validation のときは学習 loss で代用する。
        score = (
            validation_metrics["loss"] if validation_metrics else train_metrics["loss"]
        )
        if score < best_validation:
            best_validation = score
            destinations.append(args.output_dir / "best_model.pth")
        for destination in destinations:
            save_refinement_checkpoint(
                destination, model=model, epoch=epoch, optimizer=optimizer, extra=extra
            )


if __name__ == "__main__":
    main()
