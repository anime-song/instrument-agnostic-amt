from __future__ import annotations

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerWiseResampling(nn.Module):
    """Time-axis resampling for LWR.

    ``mean`` preserves the previous block-mean downsampling and repeat upsampling.
    ``conv1d`` uses depthwise Conv1d / ConvTranspose1d initialized to the same
    mean/repeat behavior, then learns the resampling filters.
    """

    def __init__(
        self,
        ratio: int = 4,
        *,
        dim: int | None = None,
        mode: str = "mean",
    ) -> None:
        super().__init__()
        if ratio <= 0:
            raise ValueError("ratio must be positive")
        if mode not in {"mean", "conv1d"}:
            raise ValueError("mode must be one of {'mean', 'conv1d'}")
        if mode == "conv1d" and dim is None:
            raise ValueError("dim is required when mode='conv1d'")
        if dim is not None and dim <= 0:
            raise ValueError("dim must be positive")

        self.ratio = int(ratio)
        self.mode = str(mode)
        self.dim = int(dim) if dim is not None else None
        self.down_conv: nn.Conv1d | None = None
        self.up_conv: nn.ConvTranspose1d | None = None
        if self.mode == "conv1d" and self.ratio > 1:
            if self.dim is None:
                raise RuntimeError("dim must be set for conv1d resampling")
            self.down_conv = nn.Conv1d(
                self.dim,
                self.dim,
                kernel_size=self.ratio,
                stride=self.ratio,
                groups=self.dim,
                bias=False,
            )
            self.up_conv = nn.ConvTranspose1d(
                self.dim,
                self.dim,
                kernel_size=self.ratio,
                stride=self.ratio,
                groups=self.dim,
                bias=False,
            )
            self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.down_conv is None or self.up_conv is None:
            return
        with torch.no_grad():
            self.down_conv.weight.fill_(1.0 / float(self.ratio))
            self.up_conv.weight.fill_(1.0)

    @staticmethod
    def _flatten_time(x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        batch_size, _, num_tokens, _ = x.shape
        return (
            einops.rearrange(x, "b t k d -> (b k) d t").contiguous(),
            int(batch_size),
            int(num_tokens),
        )

    @staticmethod
    def _unflatten_time(
        x: torch.Tensor,
        *,
        batch_size: int,
        num_tokens: int,
    ) -> torch.Tensor:
        return einops.rearrange(
            x,
            "(b k) d t -> b t k d",
            b=int(batch_size),
            k=int(num_tokens),
        ).contiguous()

    def _pad_to_ratio(self, x: torch.Tensor) -> torch.Tensor:
        remainder = int(x.shape[-1]) % self.ratio
        if remainder == 0:
            return x
        pad = self.ratio - remainder
        return F.pad(x, (0, pad), mode="replicate")

    def downsample_time(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, T, K, D]")
        if self.ratio == 1 or int(x.shape[1]) == 0:
            return x

        flat, batch_size, num_tokens = self._flatten_time(x)
        if self.mode == "mean":
            pooled = F.avg_pool1d(
                flat,
                kernel_size=self.ratio,
                stride=self.ratio,
                ceil_mode=True,
            )
            return self._unflatten_time(
                pooled,
                batch_size=batch_size,
                num_tokens=num_tokens,
            )

        if self.down_conv is None:
            raise RuntimeError("down_conv is not initialized")
        return self._unflatten_time(
            self.down_conv(self._pad_to_ratio(flat)),
            batch_size=batch_size,
            num_tokens=num_tokens,
        )

    def upsample_time(self, x: torch.Tensor, target_T: int) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, T, K, D]")
        if target_T < 0:
            raise ValueError("target_T must be non-negative")
        if self.ratio == 1:
            return x[:, :target_T]
        if int(x.shape[1]) == 0:
            return x.new_zeros(
                (
                    int(x.shape[0]),
                    int(target_T),
                    int(x.shape[2]),
                    int(x.shape[3]),
                )
            )

        if self.mode == "mean":
            upsampled = x.repeat_interleave(self.ratio, dim=1)
        else:
            if self.up_conv is None:
                raise RuntimeError("up_conv is not initialized")
            flat, batch_size, num_tokens = self._flatten_time(x)
            upsampled = self._unflatten_time(
                self.up_conv(flat),
                batch_size=batch_size,
                num_tokens=num_tokens,
            )

        if int(upsampled.shape[1]) < int(target_T):
            upsampled = F.pad(
                upsampled, (0, 0, 0, 0, 0, int(target_T) - int(upsampled.shape[1]))
            )
        return upsampled[:, :target_T]
