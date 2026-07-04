from __future__ import annotations

from typing import Sequence

import einops
import numpy as np
import torch
import torch.nn as nn

from ..blocks.transformer import RMSNorm


def build_band_indices(
    *,
    sample_rate: int = 22_050,
    n_fft: int = 1024,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Build the BS-RoFormer style contiguous frequency bands."""

    freqs = np.fft.rfftfreq(int(n_fft), d=1 / int(sample_rate))

    def bins_per_band(freq_hz: float) -> int:
        if freq_hz < 1_000:
            return 2
        if freq_hz < 2_000:
            return 4
        if freq_hz < 4_000:
            return 12
        if freq_hz < 8_000:
            return 24
        if freq_hz < 16_000:
            return 48
        return -1

    bands: list[np.ndarray] = []
    idx = 0
    n_bins = len(freqs)
    while idx < n_bins:
        if freqs[idx] >= 16_000:
            remaining = np.arange(idx, n_bins, dtype=np.int32)
            half = len(remaining) // 2
            if half > 0:
                bands.append(remaining[:half])
                bands.append(remaining[half:])
            elif len(remaining) > 0:
                bands.append(remaining)
            break

        size = bins_per_band(float(freqs[idx]))
        band = np.arange(idx, min(idx + size, n_bins), dtype=np.int32)
        if len(band) > 0:
            bands.append(band)
        idx += size

    return freqs, bands


class BandSplit(nn.Module):
    """Project variable-width STFT frequency bands into fixed-size tokens."""

    def __init__(
        self,
        *,
        hidden_size: int,
        band_indices: Sequence[np.ndarray],
        num_channels: int,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if not band_indices:
            raise ValueError("band_indices must be non-empty")

        self.hidden_size = int(hidden_size)
        self.num_channels = int(num_channels)
        self.num_bands = len(band_indices)
        self.to_features = nn.ModuleList()

        for index, raw_indices in enumerate(band_indices):
            indices = torch.as_tensor(raw_indices, dtype=torch.long)
            if int(indices.numel()) <= 0:
                raise ValueError("all bands must contain at least one bin")
            self.register_buffer(f"band_idx_{index}", indices, persistent=False)
            input_dim = int(indices.numel()) * self.num_channels
            self.to_features.append(
                nn.Sequential(
                    RMSNorm(input_dim),
                    nn.Linear(input_dim, self.hidden_size),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("x must have shape [B, T, C, F]")
        if int(x.shape[2]) != self.num_channels:
            raise ValueError(
                f"x has {x.shape[2]} channels, expected {self.num_channels}"
            )

        bands = []
        for index, projection in enumerate(self.to_features):
            band_indices = getattr(self, f"band_idx_{index}")
            sub_band = x.index_select(dim=-1, index=band_indices)
            sub_band = einops.rearrange(sub_band, "b t c f -> b t (f c)")
            bands.append(projection(sub_band))
        return torch.stack(bands, dim=2)
