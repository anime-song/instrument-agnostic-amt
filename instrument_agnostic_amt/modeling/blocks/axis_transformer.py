from __future__ import annotations

from typing import Any

import einops
import torch
import torch.nn as nn
import torch.utils.checkpoint

from .lwr import LayerWiseResampling
from .transformer import Transformer


def _checkpoint_module(
    module: nn.Module,
    x: torch.Tensor,
    *,
    use_checkpoint: bool,
    **kwargs: Any,
) -> torch.Tensor:
    if not use_checkpoint:
        return module(x, **kwargs)

    def forward_fn(tensor: torch.Tensor) -> torch.Tensor:
        return module(tensor, **kwargs)

    return torch.utils.checkpoint.checkpoint(forward_fn, x, use_reentrant=False)


class DualAxisTransformerLayer(nn.Module):
    """BS-RoFormer style dual-axis transformer layer."""

    def __init__(
        self,
        *,
        dim: int,
        head_dim: int,
        num_heads: int,
        dropout: float,
        ffn_hidden_size_factor: int = 4,
        lwr_ratio: int = 4,
        lwr_residual_mode: str = "delta",
        lwr_resampling_mode: str = "mean",
        axis_order: str = "time_band",
    ) -> None:
        super().__init__()
        if lwr_residual_mode not in {"delta", "residual"}:
            raise ValueError("lwr_residual_mode must be one of {'delta', 'residual'}")
        if lwr_resampling_mode not in {"mean", "conv1d"}:
            raise ValueError("lwr_resampling_mode must be one of {'mean', 'conv1d'}")
        if axis_order not in {"time_band", "band_time"}:
            raise ValueError("axis_order must be one of {'time_band', 'band_time'}")
        self.time_transformer = Transformer(
            input_dim=dim,
            head_dim=head_dim,
            num_layers=1,
            num_heads=num_heads,
            ffn_hidden_size_factor=ffn_hidden_size_factor,
            dropout=dropout,
        )
        self.band_transformer = Transformer(
            input_dim=dim,
            head_dim=head_dim,
            num_layers=1,
            num_heads=num_heads,
            ffn_hidden_size_factor=ffn_hidden_size_factor,
            dropout=dropout,
        )
        self.resampling = LayerWiseResampling(
            ratio=lwr_ratio,
            dim=dim,
            mode=str(lwr_resampling_mode),
        )
        self.lwr_ratio = int(lwr_ratio)
        self.lwr_residual_mode = str(lwr_residual_mode)
        self.lwr_resampling_mode = str(lwr_resampling_mode)
        self.axis_order = str(axis_order)

    def _run_time_axis(self, x: torch.Tensor, *, use_checkpoint: bool) -> torch.Tensor:
        batch_size, _, num_tokens, _ = x.shape
        x = einops.rearrange(x, "b t k d -> (b k) t d")
        x = _checkpoint_module(
            self.time_transformer,
            x,
            use_checkpoint=use_checkpoint,
        )
        return einops.rearrange(
            x,
            "(b k) t d -> b t k d",
            b=batch_size,
            k=num_tokens,
        )

    def _run_band_axis(self, x: torch.Tensor, *, use_checkpoint: bool) -> torch.Tensor:
        batch_size, time_steps, num_tokens, _ = x.shape
        x = einops.rearrange(x, "b t k d -> (b t) k d")
        x = _checkpoint_module(
            self.band_transformer,
            x,
            use_checkpoint=use_checkpoint,
        )
        return einops.rearrange(
            x,
            "(b t) k d -> b t k d",
            b=batch_size,
            t=time_steps,
            k=num_tokens,
        )

    def _run_axes(self, x: torch.Tensor, *, use_checkpoint: bool) -> torch.Tensor:
        if self.axis_order == "band_time":
            x = self._run_band_axis(x, use_checkpoint=use_checkpoint)
            return self._run_time_axis(x, use_checkpoint=use_checkpoint)
        x = self._run_time_axis(x, use_checkpoint=use_checkpoint)
        return self._run_band_axis(x, use_checkpoint=use_checkpoint)

    def forward(self, x: torch.Tensor, *, use_checkpoint: bool = False) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, T, K, D]")
        if self.lwr_ratio <= 1:
            return self._run_axes(x, use_checkpoint=use_checkpoint)

        target_T = int(x.shape[1])
        low = self.resampling.downsample_time(x)
        processed_low = self._run_axes(low, use_checkpoint=use_checkpoint)
        if self.lwr_residual_mode == "residual":
            return x + self.resampling.upsample_time(processed_low, target_T)
        delta_low = processed_low - low
        return x + self.resampling.upsample_time(delta_low, target_T)
