from __future__ import annotations

import math
from dataclasses import dataclass
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from .blocks.transformer import RMSNorm, Transformer
from .features.cqt import RecursiveCQT
from .features.spec_augment import SpecAugment


@dataclass(frozen=True)
class BackboneContext:
    spec: torch.Tensor
    crop_length: int


@dataclass(frozen=True)
class BackboneOutput:
    band_features: torch.Tensor | None
    global_features: torch.Tensor | None
    pitch_query_features: torch.Tensor
    lowres_band_features: torch.Tensor | None
    lowres_global_features: torch.Tensor | None
    lowres_pitch_query_features: torch.Tensor | None


class AudioFeatureExtractor(nn.Module):
    """The HCQT feature extractor used by V1 checkpoints."""

    def __init__(
        self,
        *,
        sampling_rate: int,
        hop_length: int,
        cqt_fmin: float = 27.5,
        cqt_n_bins: int = 312,
        cqt_bins_per_octave: int = 36,
        cqt_filter_scale: float = 0.475,
        harmonics: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
        harmonic_dropout_p: float = 0.0,
        spec_augment_params: dict[str, float | int | str] | None = None,
        peak_normalize_waveform: bool = False,
        cqt_log_scale: bool = False,
        input_audio_channels: int = 2,
    ) -> None:
        super().__init__()
        self.sampling_rate = int(sampling_rate)
        self.hop_length = int(hop_length)
        self.cqt_fmin = float(cqt_fmin)
        self.cqt_n_bins = int(cqt_n_bins)
        self.cqt_bins_per_octave = int(cqt_bins_per_octave)
        self.cqt_filter_scale = float(cqt_filter_scale)
        self.harmonics = tuple(float(harmonic) for harmonic in harmonics)
        self.harmonic_dropout_p = float(harmonic_dropout_p)
        self.peak_normalize_waveform = bool(peak_normalize_waveform)
        self.cqt_log_scale = bool(cqt_log_scale)
        self.input_audio_channels = int(input_audio_channels)

        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.input_audio_channels <= 0:
            raise ValueError("input_audio_channels must be positive")
        if self.cqt_n_bins <= 0:
            raise ValueError("cqt_n_bins must be positive")
        if not self.harmonics or any(harmonic <= 0.0 for harmonic in self.harmonics):
            raise ValueError("harmonics must contain only positive values")

        self.num_harmonics = len(self.harmonics)
        self.num_audio_channels = self.input_audio_channels * self.num_harmonics
        self.n_bins = self.cqt_n_bins
        min_harmonic = min(self.harmonics)
        max_harmonic = max(self.harmonics)
        self.fmin_large = self.cqt_fmin * min_harmonic
        self.n_bins_large = math.ceil(
            self.cqt_n_bins
            + self.cqt_bins_per_octave * math.log2(max_harmonic / min_harmonic)
        )

        nyquist = self.sampling_rate / 2.0
        max_valid_bins = math.floor(
            self.cqt_bins_per_octave * math.log2(nyquist / self.fmin_large) + 1
        )
        self.actual_cqt_bins = min(self.n_bins_large, max_valid_bins)
        if self.actual_cqt_bins <= 0:
            raise ValueError("CQT configuration has no bins below Nyquist")

        self.cqt = RecursiveCQT(
            sr=self.sampling_rate,
            hop_length=self.hop_length,
            fmin=self.fmin_large,
            n_bins=self.actual_cqt_bins,
            bins_per_octave=self.cqt_bins_per_octave,
            filter_scale=self.cqt_filter_scale,
        )
        self.register_buffer(
            "harmonic_shifts",
            torch.tensor(
                [
                    self.cqt_bins_per_octave
                    * math.log2(harmonic / min_harmonic)
                    for harmonic in self.harmonics
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        base_bins = torch.arange(self.cqt_n_bins, dtype=torch.float32)
        harmonic_positions = (
            base_bins.unsqueeze(0) + self.harmonic_shifts.unsqueeze(1)
        ).clamp(0, self.n_bins_large - 1)
        harmonic_lower_indices = harmonic_positions.floor().long()
        self.register_buffer(
            "harmonic_lower_indices",
            harmonic_lower_indices,
            persistent=False,
        )
        self.register_buffer(
            "harmonic_upper_indices",
            (harmonic_lower_indices + 1).clamp(max=self.n_bins_large - 1),
            persistent=False,
        )
        self.register_buffer(
            "harmonic_interpolation_alpha",
            harmonic_positions - harmonic_lower_indices,
            persistent=False,
        )
        self.spec_augment = (
            SpecAugment(**spec_augment_params) if spec_augment_params else None
        )

    @staticmethod
    def _normalize_spec(spec: torch.Tensor) -> torch.Tensor:
        reduce_dims = tuple(range(1, spec.ndim))
        mean = spec.mean(dim=reduce_dims, keepdim=True)
        std = spec.std(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
        return (spec - mean) / std

    def _apply_spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        if self.spec_augment is None or not self.training:
            return spec
        spec_btf = spec.transpose(1, 2)
        reduce_dims = tuple(range(1, spec_btf.ndim))
        mean = spec_btf.mean(dim=reduce_dims, keepdim=True)
        std = spec_btf.std(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
        augmented, _ = self.spec_augment((spec_btf - mean) / std)
        return (augmented * std + mean).transpose(1, 2).clamp_min(0.0)

    def _interpolate_harmonic_specs(
        self,
        large_spec: torch.Tensor,
    ) -> torch.Tensor:
        if not self.training and not torch.is_grad_enabled():
            lower_values = large_spec[:, self.harmonic_lower_indices, :]
            spec = large_spec[:, self.harmonic_upper_indices, :]
            spec.sub_(lower_values).mul_(
                self.harmonic_interpolation_alpha.unsqueeze(0).unsqueeze(-1)
            ).add_(lower_values)
            if self.cqt_log_scale:
                spec = torch.log(spec + 1e-8)
            return spec

        harmonic_specs: list[torch.Tensor] = []
        for index in range(self.num_harmonics):
            lower = self.harmonic_lower_indices[index]
            upper = self.harmonic_upper_indices[index]
            alpha = (
                self.harmonic_interpolation_alpha[index]
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            value = large_spec[:, lower, :] + alpha * (
                large_spec[:, upper, :] - large_spec[:, lower, :]
            )
            if self.cqt_log_scale:
                value = torch.log(value + 1e-8)
            harmonic_specs.append(value)
        return torch.stack(harmonic_specs, dim=1)

    def forward(self, waveform: torch.Tensor) -> BackboneContext:
        if waveform.ndim != 3 or int(waveform.shape[1]) != self.input_audio_channels:
            raise ValueError(
                f"waveform must have shape [B, {self.input_audio_channels}, T]"
            )

        crop_length = math.ceil(int(waveform.shape[-1]) / self.hop_length)
        batch_size = int(waveform.shape[0])
        x = waveform
        if self.peak_normalize_waveform:
            x = x / x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)

        flat = einops.rearrange(x, "b c t -> (b c) t").float()
        large_spec = self.cqt(flat)
        if self.actual_cqt_bins < self.n_bins_large:
            large_spec = F.pad(
                large_spec, (0, 0, 0, self.n_bins_large - self.actual_cqt_bins)
            )
        large_spec = self._apply_spec_augment(large_spec)

        spec = self._interpolate_harmonic_specs(large_spec)
        spec = einops.rearrange(
            spec,
            "(b c) h f t -> b c h f t",
            b=batch_size,
            c=self.input_audio_channels,
        )
        spec = self._normalize_spec(spec)
        if self.training and self.harmonic_dropout_p > 0.0:
            keep_mask = (
                torch.rand(
                    batch_size,
                    1,
                    self.num_harmonics,
                    1,
                    1,
                    device=spec.device,
                )
                >= self.harmonic_dropout_p
            )
            fundamental_mask = torch.tensor(
                [abs(harmonic - 1.0) <= 1e-5 for harmonic in self.harmonics],
                device=spec.device,
                dtype=torch.bool,
            ).view(1, 1, self.num_harmonics, 1, 1)
            spec = spec * (keep_mask | fundamental_mask).to(dtype=spec.dtype)

        spec = einops.rearrange(spec, "b c h f t -> b (c h) t f").contiguous()
        return BackboneContext(spec=spec.to(dtype=waveform.dtype), crop_length=crop_length)


class StemConv(nn.Module):
    """The V1 convolutional stem (time /8, frequency /4)."""

    def __init__(self, *, in_ch: int, base_ch: int, n_bins: int = 312) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, base_ch, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=5, padding=2)
        self.freq_embed = nn.Parameter(torch.zeros(1, base_ch, 1, n_bins))
        nn.init.normal_(self.freq_embed, std=0.02)
        self.block1 = self._block(base_ch, base_ch * 2, stride=(2, 1))
        self.block2 = self._block(base_ch * 2, base_ch * 4, stride=(2, 2))
        self.block3 = self._block(base_ch * 4, base_ch * 4, stride=(2, 2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, 3, padding=1),
            nn.GroupNorm(4, base_ch * 4),
        )
        self.out_ch = base_ch * 4

    @staticmethod
    def _block(in_ch: int, out_ch: int, *, stride: tuple[int, int]) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.GELU(),
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) + self.freq_embed[:, :, :, : int(x.shape[-1])]
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (
            not self.training
            and torch.is_autocast_enabled(x.device.type)
            and torch.get_autocast_dtype(x.device.type) == torch.float16
        ):
            # FP16 AMP の推論時は、StemConv 全体を FP32 で実行して
            # 畳み込みのオーバーフローを防ぐ。
            with torch.amp.autocast(device_type=x.device.type, enabled=False):
                return self._forward_impl(x.float())
        return self._forward_impl(x)


class PitchQueryEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, midi_pitches: torch.Tensor) -> torch.Tensor:
        x = midi_pitches.float()
        return self.mlp(
            torch.stack(
                [
                    x / 128.0,
                    torch.sin(2 * math.pi * x / 12.0),
                    torch.cos(2 * math.pi * x / 12.0),
                    torch.ones_like(x),
                ],
                dim=-1,
            )
        )


def _checkpoint(module: nn.Module, x: torch.Tensor, *, enabled: bool) -> torch.Tensor:
    if not enabled:
        return module(x)

    def forward_fn(tensor: torch.Tensor) -> torch.Tensor:
        return module(tensor)

    return torch.utils.checkpoint.checkpoint(forward_fn, x, use_reentrant=False)


class V1Backbone(nn.Module):
    """Checkpoint-compatible V1 HCQT/StemConv dual-axis Transformer backbone."""

    def __init__(
        self,
        *,
        feature_extractor: AudioFeatureExtractor,
        hidden_size: int,
        base_ch: int,
        num_layers: int,
        num_heads: int,
        num_pitch_queries: int,
        use_global_token: bool,
        dropout: float,
        use_gradient_checkpoint: bool,
    ) -> None:
        super().__init__()
        if hidden_size <= 0 or base_ch <= 0 or num_layers < 0 or num_heads <= 0:
            raise ValueError("invalid V1 backbone dimensions")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (hidden_size // num_heads) % 2 != 0:
            raise ValueError("hidden_size / num_heads must be even for RoPE")

        self.feature_extractor = feature_extractor
        self.hidden_size = int(hidden_size)
        self.base_ch = int(base_ch)
        self.output_dim = int(hidden_size)
        self.use_gradient_checkpoint = bool(use_gradient_checkpoint)
        self.num_pitch_queries = int(num_pitch_queries)
        self.use_global_token = bool(use_global_token)
        self.num_audio_channels = feature_extractor.num_audio_channels
        self.n_bins = feature_extractor.n_bins
        self.model_dim = int(hidden_size)

        self.stem = StemConv(
            in_ch=self.num_audio_channels,
            base_ch=self.base_ch,
            n_bins=self.n_bins,
        )
        token_dim = self.base_ch * 4
        self.pitch_query_embed = PitchQueryEmbedding(dim=token_dim)
        self.register_buffer(
            "midi_pitches",
            torch.arange(21, 21 + self.num_pitch_queries, dtype=torch.float32),
            persistent=False,
        )
        self.band_type_embed = nn.Parameter(torch.zeros(1, 1, 1, token_dim))
        self.global_token = (
            nn.Parameter(torch.zeros(1, 1, 1, token_dim))
            if self.use_global_token
            else None
        )
        self.global_type_embed = (
            nn.Parameter(torch.zeros(1, 1, 1, token_dim))
            if self.use_global_token
            else None
        )
        self.pitch_type_embed = nn.Parameter(torch.zeros(1, 1, 1, token_dim))

        self.layers = nn.ModuleList()
        for _ in range(int(num_layers)):
            time_transformer = Transformer(
                input_dim=token_dim,
                head_dim=int(hidden_size) // int(num_heads),
                num_layers=1,
                num_heads=int(num_heads),
                ffn_hidden_size_factor=4,
                dropout=float(dropout),
            )
            band_transformer = Transformer(
                input_dim=token_dim,
                head_dim=int(hidden_size) // int(num_heads),
                num_layers=1,
                num_heads=int(num_heads),
                ffn_hidden_size_factor=4,
                dropout=float(dropout),
            )
            self.layers.append(nn.ModuleList([time_transformer, band_transformer]))

        self.stem_dim = token_dim
        self.query_feature_dim = token_dim
        self.final_norm = RMSNorm(token_dim)
        self.up_conv = nn.ConvTranspose1d(token_dim, token_dim, kernel_size=8, stride=8)
        self.global_up_conv = (
            nn.ConvTranspose1d(token_dim, token_dim, kernel_size=8, stride=8)
            if self.use_global_token
            else None
        )

    @staticmethod
    def _match_time_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
        if int(x.shape[2]) < int(target_length):
            return F.pad(x, (0, 0, 0, int(target_length) - int(x.shape[2])))
        return x[:, :, :target_length]

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        include_aux_outputs: bool = True,
    ) -> BackboneOutput:
        context = self.feature_extractor(waveform)
        x = einops.rearrange(self.stem(context.spec), "b d t f -> b t f d")
        batch_size, _, num_bands, _ = x.shape

        pitch_query = self.pitch_query_embed(self.midi_pitches)
        pitch_query = pitch_query.unsqueeze(0).unsqueeze(0).expand(
            batch_size, int(x.shape[1]), -1, -1
        )
        tokens = [x + self.band_type_embed]
        if self.use_global_token:
            if self.global_token is None or self.global_type_embed is None:
                raise RuntimeError("global token parameters are not initialized")
            tokens.append(
                self.global_token.expand(batch_size, int(x.shape[1]), -1, -1)
                + self.global_type_embed
            )
        tokens.append(pitch_query + self.pitch_type_embed)
        x = torch.cat(tokens, dim=2)

        use_checkpoint = (
            self.use_gradient_checkpoint and self.training and torch.is_grad_enabled()
        )
        for time_transformer, band_transformer in self.layers:
            batch_size, time_steps, token_count, token_dim = x.shape
            x = x.reshape(batch_size * time_steps, token_count, token_dim).contiguous()
            x = _checkpoint(band_transformer, x, enabled=use_checkpoint)
            x = x.reshape(batch_size, time_steps, token_count, token_dim)
            x = einops.rearrange(x, "b t k d -> (b k) t d").contiguous()
            x = _checkpoint(time_transformer, x, enabled=use_checkpoint)
            x = einops.rearrange(x, "(b k) t d -> b t k d", k=token_count)
        x = self.final_norm(x)

        band_part = x[:, :, :num_bands, :]
        pitch_start = int(num_bands)
        global_part = None
        if self.use_global_token:
            global_part = x[:, :, pitch_start : pitch_start + 1, :]
            pitch_start += 1
        pitch_part = x[:, :, pitch_start:, :]

        batch_size, _, num_pitches, _ = pitch_part.shape
        pitch_part = einops.rearrange(pitch_part, "b t p d -> (b p) d t")
        pitch_part = self.up_conv(pitch_part)
        pitch_part = einops.rearrange(
            pitch_part, "(b p) d t -> b p t d", b=batch_size, p=num_pitches
        )
        pitch_part = self._match_time_length(pitch_part, context.crop_length)

        lowres_global_features = (
            global_part.squeeze(2).contiguous()
            if global_part is not None and include_aux_outputs
            else None
        )
        global_features = None
        if global_part is not None and include_aux_outputs:
            if self.global_up_conv is None:
                raise RuntimeError("global_up_conv is not initialized")
            global_part = einops.rearrange(global_part, "b t g d -> (b g) d t")
            global_part = self.global_up_conv(global_part)
            global_part = einops.rearrange(
                global_part, "(b g) d t -> b g t d", b=batch_size, g=1
            )
            global_part = self._match_time_length(global_part, context.crop_length)
            global_features = global_part.squeeze(1).contiguous()

        return BackboneOutput(
            band_features=(
                band_part.permute(0, 2, 1, 3).contiguous()
                if include_aux_outputs
                else None
            ),
            global_features=global_features,
            pitch_query_features=pitch_part.permute(0, 2, 1, 3).contiguous(),
            lowres_band_features=(
                band_part.contiguous() if include_aux_outputs else None
            ),
            lowres_global_features=lowres_global_features,
            lowres_pitch_query_features=(
                x[:, :, pitch_start:, :].contiguous()
                if include_aux_outputs
                else None
            ),
        )
