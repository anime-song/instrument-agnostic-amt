from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerWiseResampling(nn.Module):
    """Block-mean downsampling and repeat upsampling for LWR."""

    def __init__(self, ratio: int = 4) -> None:
        super().__init__()
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        self.ratio = int(ratio)

    def downsample_time(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, T, K, D]")
        if self.ratio == 1:
            return x

        batch_size, time_steps, num_tokens, dim = x.shape
        x_bkd = x.permute(0, 2, 3, 1).reshape(batch_size * num_tokens, dim, time_steps)
        pooled = F.avg_pool1d(
            x_bkd,
            kernel_size=self.ratio,
            stride=self.ratio,
            ceil_mode=True,
        )
        low_time = int(pooled.shape[-1])
        return pooled.reshape(batch_size, num_tokens, dim, low_time).permute(0, 3, 1, 2)

    def upsample_time(self, x: torch.Tensor, target_T: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, T, K, D]")
        if self.ratio == 1:
            return x[:, :target_T]

        upsampled = x.repeat_interleave(self.ratio, dim=1)
        if int(upsampled.shape[1]) < int(target_T):
            upsampled = F.pad(
                upsampled, (0, 0, 0, 0, 0, int(target_T) - int(upsampled.shape[1]))
            )
        return upsampled[:, :target_T]
