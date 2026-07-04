from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


@dataclass(frozen=True)
class StemSplitterInitReport:
    loaded_tensors: int
    loaded_numel: int
    categories: dict[str, int]
    category_numel: dict[str, int]
    skipped_missing: int
    skipped_shape: int
    skipped_shape_examples: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Stem splitter checkpoint must be a state_dict or dict")

    for key in ("ema_state_dict", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break

    if not all(torch.is_tensor(value) for value in checkpoint.values()):
        raise ValueError(
            "Stem splitter checkpoint does not look like a tensor state_dict"
        )
    return checkpoint


def _source_layer_key(target_key: str) -> str | None:
    match = re.match(
        r"backbone\.layers\.(\d+)\.(time_transformer|band_transformer)\.(.+)",
        target_key,
    )
    if match is None:
        return None

    layer_index, axis_name, rest = match.groups()
    axis_index = "0" if axis_name == "time_transformer" else "1"
    rest = rest.replace("norm_q.gamma", "norm.gamma")
    rest = rest.replace("norm_context.gamma", "norm.gamma")
    return f"layers.{layer_index}.{axis_index}.{rest}"


def _source_qkv_key(target_key: str) -> tuple[str, int] | None:
    for qkv_index, name in enumerate(("to_q", "to_k", "to_v")):
        marker = f".{name}."
        if marker in target_key:
            source_key = _source_layer_key(target_key.replace(marker, ".to_qkv."))
            if source_key is None:
                return None
            return source_key, qkv_index
    return None


def _copy_if_shape_matches(
    *,
    init_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    target_key: str,
    source_key: str,
    target_tensor: torch.Tensor,
    category: str,
    categories: Counter[str],
    category_numel: Counter[str],
    skipped_shape_examples: list[tuple[str, tuple[int, ...], tuple[int, ...]]],
) -> bool | None:
    source_tensor = source_state.get(source_key)
    if source_tensor is None:
        return None
    if tuple(source_tensor.shape) != tuple(target_tensor.shape):
        if len(skipped_shape_examples) < 20:
            skipped_shape_examples.append(
                (target_key, tuple(source_tensor.shape), tuple(target_tensor.shape))
            )
        return False

    init_state[target_key] = source_tensor.detach().clone()
    categories[category] += 1
    category_numel[category] += int(target_tensor.numel())
    return True


def _copy_qkv_chunk_if_shape_matches(
    *,
    init_state: dict[str, torch.Tensor],
    source_state: dict[str, torch.Tensor],
    target_key: str,
    source_key: str,
    qkv_index: int,
    target_tensor: torch.Tensor,
    categories: Counter[str],
    category_numel: Counter[str],
    skipped_shape_examples: list[tuple[str, tuple[int, ...], tuple[int, ...]]],
) -> bool | None:
    source_tensor = source_state.get(source_key)
    if source_tensor is None:
        return None

    if source_tensor.shape[0] % 3 != 0:
        if len(skipped_shape_examples) < 20:
            skipped_shape_examples.append(
                (target_key, tuple(source_tensor.shape), tuple(target_tensor.shape))
            )
        return False

    source_chunk = torch.chunk(source_tensor, chunks=3, dim=0)[qkv_index]
    if tuple(source_chunk.shape) != tuple(target_tensor.shape):
        if len(skipped_shape_examples) < 20:
            skipped_shape_examples.append(
                (target_key, tuple(source_chunk.shape), tuple(target_tensor.shape))
            )
        return False

    init_state[target_key] = source_chunk.detach().clone()
    categories["transformer_qkv_split"] += 1
    category_numel["transformer_qkv_split"] += int(target_tensor.numel())
    return True


def load_stem_splitter_initialization(
    model: nn.Module,
    checkpoint_path: str | Path,
) -> StemSplitterInitReport:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source_state = _extract_state_dict(checkpoint)
    target_state = model.state_dict()
    init_state: dict[str, torch.Tensor] = {}
    categories: Counter[str] = Counter()
    category_numel: Counter[str] = Counter()
    skipped_missing = 0
    skipped_shape = 0
    skipped_shape_examples: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

    for target_key, target_tensor in target_state.items():
        result: bool | None = None
        if target_key.startswith("backbone.band_split."):
            result = _copy_if_shape_matches(
                init_state=init_state,
                source_state=source_state,
                target_key=target_key,
                source_key=target_key.removeprefix("backbone."),
                target_tensor=target_tensor,
                category="band_split",
                categories=categories,
                category_numel=category_numel,
                skipped_shape_examples=skipped_shape_examples,
            )
        elif target_key.startswith("backbone.final_norm."):
            result = _copy_if_shape_matches(
                init_state=init_state,
                source_state=source_state,
                target_key=target_key,
                source_key=target_key.removeprefix("backbone."),
                target_tensor=target_tensor,
                category="final_norm",
                categories=categories,
                category_numel=category_numel,
                skipped_shape_examples=skipped_shape_examples,
            )
        elif target_key.startswith("backbone.layers."):
            source_key = _source_layer_key(target_key)
            if source_key is not None:
                result = _copy_if_shape_matches(
                    init_state=init_state,
                    source_state=source_state,
                    target_key=target_key,
                    source_key=source_key,
                    target_tensor=target_tensor,
                    category="transformer_direct",
                    categories=categories,
                    category_numel=category_numel,
                    skipped_shape_examples=skipped_shape_examples,
                )
            if result is None:
                qkv_source = _source_qkv_key(target_key)
                if qkv_source is not None:
                    source_key, qkv_index = qkv_source
                    result = _copy_qkv_chunk_if_shape_matches(
                        init_state=init_state,
                        source_state=source_state,
                        target_key=target_key,
                        source_key=source_key,
                        qkv_index=qkv_index,
                        target_tensor=target_tensor,
                        categories=categories,
                        category_numel=category_numel,
                        skipped_shape_examples=skipped_shape_examples,
                    )

        if result is None:
            skipped_missing += 1
        elif result is False:
            skipped_shape += 1

    if not init_state:
        raise ValueError(
            f"No compatible stem splitter weights found in {checkpoint_path}"
        )

    model.load_state_dict(init_state, strict=False)
    return StemSplitterInitReport(
        loaded_tensors=len(init_state),
        loaded_numel=sum(int(tensor.numel()) for tensor in init_state.values()),
        categories=dict(categories),
        category_numel=dict(category_numel),
        skipped_missing=skipped_missing,
        skipped_shape=skipped_shape,
        skipped_shape_examples=tuple(skipped_shape_examples),
    )
