"""チェックポイントの入出力。

役割は 2 つ:

- ``load_amt_backbone``: 学習済み AMT のチェックポイントから backbone 部分だけを
  借りてきて、このモデルの初期値にする（学習の出発点を作る）。
- ``save_refinement_checkpoint`` / ``load_refinement_model``: このモデル自身の
  チェックポイントを読み書きする。

AMT のチェックポイントと refinement のチェックポイントは中身が別物なので、
``task`` フィールドで取り違えを弾いている。
"""

from __future__ import annotations

import os
import pathlib
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ...amt.modeling.checkpoints import (
    extract_model_config,
    load_checkpoint,
    select_state_dict,
)
from ...amt.modeling.model import remap_legacy_v1_state_dict
from .model import (
    InstrumentRefinementConfig,
    InstrumentRefinementModel,
    _coerce_config_values,
)


@contextmanager
def _posix_path_compatibility() -> Generator[None]:
    """WSL で保存したチェックポイントを Windows で読めるようにする。

    チェックポイントには学習時の引数（``training_args``）が入っており、その中の
    パスは保存側の OS の Path として pickle される。WSL で学習して Windows で
    推論すると PosixPath を復元できず torch.load 全体が失敗するため、読み込みの
    間だけ WindowsPath として解決させる。パスは付随情報で、モデルの復元には
    使わないので値が Windows 形式になっても問題ない。
    """
    if os.name != "nt":
        yield
        return
    original_posix_path = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        yield
    finally:
        pathlib.PosixPath = original_posix_path


@dataclass(frozen=True)
class BackboneLoadReport:
    """backbone 移植の結果。何が入って何が入らなかったかを呼び出し側へ返す。"""

    loaded_keys: tuple[str, ...]
    missing_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


def _normalize(value: Any) -> Any:
    """JSON 経由で list になった設定値を tuple にそろえて比較できるようにする。"""
    return tuple(value) if isinstance(value, list) else value


def _validate_backbone_config(model: InstrumentRefinementModel, checkpoint: Mapping[str, Any]) -> None:
    """AMT 側と backbone の構成が一致しているかを先に確認する。

    ずれたまま重みだけ入れると、形状は合っていても特徴の意味が変わって静かに劣化する。
    そのため CQT やエンコーダの設定が違う場合はここで失敗させる。
    """
    try:
        source = extract_model_config(checkpoint)
    except ValueError:
        return
    target = model.config
    comparisons = {
        "sample_rate": target.sample_rate,
        "hop_length": target.hop_length,
        "cqt_fmin": target.cqt_fmin,
        "cqt_n_bins": target.cqt_n_bins,
        "cqt_bins_per_octave": target.cqt_bins_per_octave,
        "cqt_filter_scale": target.cqt_filter_scale,
        "harmonics": target.harmonics,
        "cqt_log_scale": target.cqt_log_scale,
        "input_audio_channels": target.input_audio_channels,
        "hidden_size": target.hidden_size,
        "base_ch": target.base_ch,
        "encoder_num_layers": target.encoder_num_layers,
        "encoder_num_heads": target.encoder_num_heads,
        "pitch_query_count": target.pitch_query_count,
    }
    mismatches = [
        f"{key}: checkpoint={_normalize(source[key])!r}, model={_normalize(value)!r}"
        for key, value in comparisons.items()
        if key in source and _normalize(source[key]) != _normalize(value)
    ]
    if mismatches:
        raise ValueError("AMT checkpoint backbone config does not match refinement model: " + "; ".join(mismatches))


def load_amt_backbone(
    model: InstrumentRefinementModel,
    checkpoint_or_path: Mapping[str, Any] | str | Path,
    *,
    prefer_ema: bool = True,
    require_complete: bool = True,
) -> BackboneLoadReport:
    """AMT のチェックポイントから backbone の重みだけを取り込む。

    Args:
        prefer_ema: EMA 重みがあればそちらを使う（通常はこちらの方が良い）。
        require_complete: 1 つでも欠けたら失敗させる。中途半端な初期化を防ぐため既定は True。
    """
    checkpoint = (
        load_checkpoint(checkpoint_or_path) if isinstance(checkpoint_or_path, (str, Path)) else checkpoint_or_path
    )
    _validate_backbone_config(model, checkpoint)
    source = remap_legacy_v1_state_dict(select_state_dict(checkpoint, prefer_ema=prefer_ema))
    target = model.backbone.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    shape_mismatches: list[str] = []
    # AMT 側の state_dict は "backbone." 接頭辞つき。ヘッド類は取り込まない。
    for key, value in source.items():
        if not key.startswith("backbone."):
            continue
        backbone_key = key[len("backbone.") :]
        if backbone_key not in target:
            continue
        if tuple(value.shape) != tuple(target[backbone_key].shape):
            shape_mismatches.append(
                f"{backbone_key}: checkpoint={tuple(value.shape)}, model={tuple(target[backbone_key].shape)}"
            )
            continue
        compatible[backbone_key] = value
    missing = sorted(set(target) - set(compatible))
    if require_complete and (missing or shape_mismatches):
        details = [*shape_mismatches, *(f"missing: {key}" for key in missing)]
        preview = ", ".join(details[:8])
        if len(details) > 8:
            preview += f", ... (+{len(details) - 8})"
        raise ValueError(f"AMT backbone is not fully compatible: {preview}")
    model.backbone.load_state_dict(compatible, strict=False)
    return BackboneLoadReport(
        loaded_keys=tuple(sorted(compatible)),
        missing_keys=tuple(missing),
        shape_mismatches=tuple(sorted(shape_mismatches)),
    )


def save_refinement_checkpoint(
    path: str | Path,
    *,
    model: InstrumentRefinementModel,
    epoch: int,
    optimizer: torch.optim.Optimizer | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """チェックポイントを保存する。書き込み中に落ちても壊れないよう一時ファイル経由。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload: dict[str, Any] = {
        "checkpoint_format_version": 1,
        "task": "instrument_refinement",
        "epoch": int(epoch),
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if extra:
        payload.update(dict(extra))
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_refinement_model(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[InstrumentRefinementModel, InstrumentRefinementConfig, dict[str, Any]]:
    """推論用にモデルを復元する。構成はチェックポイント内の model_config から作る。

    Returns:
        (eval 状態のモデル, その構成, チェックポイントの生 dict)。
    """
    with _posix_path_compatibility():
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("task") != "instrument_refinement":
        raise ValueError(f"Not an instrument refinement checkpoint: {path}")
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, Mapping):
        raise ValueError("Refinement checkpoint does not contain model_config")
    config = InstrumentRefinementConfig(**_coerce_config_values(raw_config))
    model = InstrumentRefinementModel(config)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("Refinement checkpoint does not contain model_state_dict")
    model.load_state_dict(state, strict=True)
    model.to(torch.device(device))
    model.eval()
    return model, config, checkpoint
