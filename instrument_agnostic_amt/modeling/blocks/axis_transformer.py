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
    """BS-RoFormer style time-axis then band-axis transformer layer."""

    def __init__(
        self,
        *,
        dim: int,
        head_dim: int,
        num_heads: int,
        dropout: float,
        ffn_hidden_size_factor: int = 4,
        lwr_ratio: int = 4,
    ) -> None:
        super().__init__()
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
        self.resampling = LayerWiseResampling(ratio=lwr_ratio)
        self.lwr_ratio = int(lwr_ratio)

    def _run_axes(self, x: torch.Tensor, *, use_checkpoint: bool) -> torch.Tensor:
        batch_size, time_steps, num_tokens, dim = x.shape
        x = einops.rearrange(x, "b t k d -> (b k) t d")
        x = _checkpoint_module(
            self.time_transformer,
            x,
            use_checkpoint=use_checkpoint,
        )
        x = einops.rearrange(
            x,
            "(b k) t d -> b t k d",
            b=batch_size,
            k=num_tokens,
        )

        x = x.reshape(batch_size * int(x.shape[1]), num_tokens, dim)
        x = _checkpoint_module(
            self.band_transformer,
            x,
            use_checkpoint=use_checkpoint,
        )
        return x.reshape(batch_size, -1, num_tokens, dim)

    def forward(self, x: torch.Tensor, *, use_checkpoint: bool = False) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, T, K, D]")
        if self.lwr_ratio <= 1:
            return self._run_axes(x, use_checkpoint=use_checkpoint)

        target_T = int(x.shape[1])
        low = self.resampling.downsample_time(x)
        processed_low = self._run_axes(low, use_checkpoint=use_checkpoint)
        delta_low = processed_low - low
        return x + self.resampling.upsample_time(delta_low, target_T)
