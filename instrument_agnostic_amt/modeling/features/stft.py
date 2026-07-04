from __future__ import annotations

import math
from dataclasses import dataclass

import einops
import torch
import torch.nn as nn


@dataclass(frozen=True)
class STFTFeatureOutput:
    """STFT features consumed by the V2 backbone."""

    spec: torch.Tensor  # [B, T, C * 2, F]
    crop_length: int


class STFTFeatureExtractor(nn.Module):
    """Convert stereo waveform into real/imag STFT band-split input."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int = 2048,
        hop_length: int = 512,
        win_length: int | None = None,
        center: bool = True,
        input_audio_channels: int = 2,
        peak_normalize_waveform: bool = False,
    ) -> None:
        super().__init__()
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if input_audio_channels <= 0:
            raise ValueError("input_audio_channels must be positive")

        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.center = bool(center)
        self.input_audio_channels = int(input_audio_channels)
        self.output_channels = self.input_audio_channels * 2
        self.n_bins = self.n_fft // 2 + 1
        self.peak_normalize_waveform = bool(peak_normalize_waveform)
        self.register_buffer(
            "window",
            torch.hann_window(self.win_length),
            persistent=False,
        )

    def forward(self, waveform: torch.Tensor) -> STFTFeatureOutput:
        if waveform.ndim != 3 or waveform.shape[1] != self.input_audio_channels:
            raise ValueError(
                f"waveform must have shape [B, {self.input_audio_channels}, T]"
            )

        crop_length = math.ceil(int(waveform.shape[-1]) / float(self.hop_length))
        x = waveform
        if self.peak_normalize_waveform:
            peak = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            x = x / peak

        flat = einops.rearrange(x.float(), "b c t -> (b c) t")
        spec = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=flat.device, dtype=flat.dtype),
            center=self.center,
            return_complex=True,
        )
        spec = einops.rearrange(
            spec,
            "(b c) f t -> b c f t",
            b=int(waveform.shape[0]),
            c=self.input_audio_channels,
        )
        spec = torch.view_as_real(spec)
        spec = einops.rearrange(spec, "b c f t r -> b t (c r) f")

        if spec.shape[1] < crop_length:
            pad = crop_length - int(spec.shape[1])
            spec = torch.nn.functional.pad(spec, (0, 0, 0, 0, 0, pad))
        elif spec.shape[1] > crop_length:
            spec = spec[:, :crop_length]

        return STFTFeatureOutput(
            spec=spec.to(dtype=waveform.dtype), crop_length=crop_length
        )
