from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks.axis_transformer import DualAxisTransformerLayer
from .blocks.transformer import RMSNorm
from .features.cqt import RecursiveCQT
from .features.spec_augment import SpecAugment
from .heads.interval_boundaries import gather_interval_endpoint_features
from ..taxonomy.instrument_classes import NUM_INSTRUMENT_CLASSES

MIN_MIDI_PITCH = 21
MAX_MIDI_PITCH = 108
NUM_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1


def compute_model_frames(num_audio_frames: int, n_fft: int, hop_length: int) -> int:
    _ = n_fft
    return math.ceil(num_audio_frames / hop_length)


def resolve_lwr_layers(
    lwr_layers: str | Sequence[int] | None,
    encoder_num_layers: int,
) -> tuple[int, ...]:
    if encoder_num_layers <= 0:
        raise ValueError("encoder_num_layers must be positive")
    if lwr_layers is None:
        return tuple(range(int(encoder_num_layers)))

    raw_indices: list[int] = []
    if isinstance(lwr_layers, str):
        value = lwr_layers.strip().lower()
        if value in {"all", "*"}:
            return tuple(range(int(encoder_num_layers)))
        if value.startswith("last"):
            suffix = value[4:].lstrip("_-")
            if suffix and not suffix.isdigit():
                raise ValueError("last-N lwr layer count must be a positive integer")
            count = 3 if not suffix else int(suffix)
            if count <= 0:
                raise ValueError("last-N lwr layer count must be positive")
            start = max(0, int(encoder_num_layers) - count)
            return tuple(range(start, int(encoder_num_layers)))
        if value in {"none", "off", "false", ""}:
            return ()
        if len(value) == int(encoder_num_layers) and all(
            char in {"0", "1"} for char in value
        ):
            raw_indices = [index for index, flag in enumerate(value) if flag == "1"]
        else:
            for token in value.replace(";", ",").split(","):
                token = token.strip()
                if not token:
                    continue
                if "-" in token:
                    bounds = [part.strip() for part in token.split("-", maxsplit=1)]
                    if len(bounds) == 2 and all(part.isdigit() for part in bounds):
                        start, end = (int(bounds[0]), int(bounds[1]))
                        if end < start:
                            raise ValueError("lwr layer range end must be >= start")
                        raw_indices.extend(range(start, end + 1))
                        continue
                try:
                    raw_indices.append(int(token))
                except ValueError as exc:
                    raise ValueError(
                        "lwr_layers must be 'all', 'none', 'last3', a binary mask, "
                        "or comma-separated 0-based layer indices/ranges"
                    ) from exc
    else:
        raw_indices = [int(index) for index in lwr_layers]

    seen: set[int] = set()
    resolved: list[int] = []
    for index in raw_indices:
        if index < 0 or index >= int(encoder_num_layers):
            raise ValueError(
                f"lwr layer index {index} is out of range for "
                f"{encoder_num_layers} encoder layers"
            )
        if index in seen:
            continue
        seen.add(index)
        resolved.append(index)
    return tuple(resolved)


@dataclass(frozen=True)
class SemiCRFModelConfig:
    sample_rate: int
    hop_length: int
    n_fft: int = 2048
    architecture_version: int = 2
    cqt_fmin: float = 27.5
    cqt_n_bins: int = 312
    cqt_bins_per_octave: int = 36
    cqt_filter_scale: float = 0.475
    harmonics: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    harmonic_dropout_p: float = 0.0
    cqt_log_scale: bool = False
    input_audio_channels: int = 2
    hidden_size: int = 384
    base_ch: int = 64
    encoder_num_layers: int = 6
    encoder_num_heads: int = 12
    dropout: float = 0.1
    lwr_layers: str | tuple[int, ...] = "last3"
    lwr_ratio: int = 8
    lwr_residual_mode: str = "delta"
    lwr_resampling_mode: str = "conv1d"
    use_gradient_checkpoint: bool = True
    pitch_query_count: int = NUM_PITCHES
    semi_crf_head_dim: int = 256
    semi_crf_length_scaling: str = "none"
    semi_crf_length_penalty: float = 0.0
    use_interval_boundary_head: bool = True
    num_instrument_classes: int = NUM_INSTRUMENT_CLASSES
    instrument_pair_gate_dim: int = 128
    spec_augment_params: dict[str, float | int] | None = None


@dataclass(frozen=True)
class CQTFeatureOutput:
    spec: torch.Tensor  # [B, C * harmonics, T, F]
    crop_length: int


@dataclass(frozen=True)
class BackboneOutput:
    band_features: torch.Tensor  # [B, K, T, D]
    pitch_query_features: torch.Tensor  # [B, T, P, D]


class AudioFeatureExtractor(nn.Module):
    """Extract V1 HCQT features from waveform input."""

    def __init__(
        self,
        *,
        sampling_rate: int,
        hop_length: int,
        input_audio_channels: int = 2,
        cqt_fmin: float = 27.5,
        cqt_n_bins: int = 312,
        cqt_bins_per_octave: int = 36,
        cqt_filter_scale: float = 0.475,
        harmonics: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0),
        harmonic_dropout_p: float = 0.0,
        spec_augment_params: dict[str, float | int] | None = None,
        peak_normalize_waveform: bool = False,
        cqt_log_scale: bool = False,
    ) -> None:
        super().__init__()
        self.sampling_rate = int(sampling_rate)
        self.hop_length = int(hop_length)
        self.input_audio_channels = int(input_audio_channels)
        self.cqt_fmin = float(cqt_fmin)
        self.cqt_n_bins = int(cqt_n_bins)
        self.cqt_bins_per_octave = int(cqt_bins_per_octave)
        self.cqt_filter_scale = float(cqt_filter_scale)
        self.harmonics = tuple(float(harmonic) for harmonic in harmonics)
        self.harmonic_dropout_p = float(harmonic_dropout_p)
        self.peak_normalize_waveform = bool(peak_normalize_waveform)
        self.cqt_log_scale = bool(cqt_log_scale)

        if self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.input_audio_channels <= 0:
            raise ValueError("input_audio_channels must be positive")
        if self.cqt_n_bins <= 0:
            raise ValueError("cqt_n_bins must be positive")
        if not self.harmonics:
            raise ValueError("harmonics must be non-empty")
        if any(harmonic <= 0.0 for harmonic in self.harmonics):
            raise ValueError("harmonics must contain only positive values")

        self.num_harmonics = len(self.harmonics)
        self.num_audio_channels = self.input_audio_channels * self.num_harmonics
        self.n_bins = self.cqt_n_bins
        self.min_harmonic = min(self.harmonics)
        self.max_harmonic = max(self.harmonics)
        self.fmin_large = self.cqt_fmin * self.min_harmonic
        self.n_bins_large = math.ceil(
            self.cqt_n_bins
            + self.cqt_bins_per_octave
            * math.log2(self.max_harmonic / self.min_harmonic)
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
                    self.cqt_bins_per_octave * math.log2(harmonic / self.min_harmonic)
                    for harmonic in self.harmonics
                ],
                dtype=torch.float32,
            ),
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

    def _apply_spec_augment_to_large_cqt(
        self, large_cqt_spec: torch.Tensor
    ) -> torch.Tensor:
        if self.spec_augment is None or not self.training:
            return large_cqt_spec

        large_cqt_btf = large_cqt_spec.transpose(1, 2)
        reduce_dims = tuple(range(1, large_cqt_btf.ndim))
        mean = large_cqt_btf.mean(dim=reduce_dims, keepdim=True)
        std = large_cqt_btf.std(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
        normalized = (large_cqt_btf - mean) / std
        augmented, _ = self.spec_augment(normalized)
        restored = augmented * std + mean
        return restored.transpose(1, 2).clamp_min(0.0)

    def forward(self, waveform: torch.Tensor) -> CQTFeatureOutput:
        if waveform.ndim != 3 or int(waveform.shape[1]) != self.input_audio_channels:
            raise ValueError(
                f"waveform must have shape [B, {self.input_audio_channels}, T]"
            )

        crop_length = math.ceil(int(waveform.shape[-1]) / float(self.hop_length))
        batch_size = int(waveform.shape[0])
        x = waveform
        if self.peak_normalize_waveform:
            peak = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
            x = x / peak

        flat = einops.rearrange(x, "b c t -> (b c) t").float()
        with torch.amp.autocast(device_type=flat.device.type, enabled=False):
            large_cqt_spec = self.cqt(flat.float())
        if self.actual_cqt_bins < self.n_bins_large:
            large_cqt_spec = F.pad(
                large_cqt_spec, (0, 0, 0, self.n_bins_large - self.actual_cqt_bins)
            )
        large_cqt_spec = self._apply_spec_augment_to_large_cqt(large_cqt_spec)

        freq_bins_large = int(large_cqt_spec.shape[1])
        base_bins = torch.arange(
            self.cqt_n_bins, device=large_cqt_spec.device, dtype=large_cqt_spec.dtype
        )
        harmonic_specs: list[torch.Tensor] = []
        for index in range(self.num_harmonics):
            position = (base_bins + self.harmonic_shifts[index]).clamp(
                0, freq_bins_large - 1
            )
            lower = torch.floor(position).long()
            upper = (lower + 1).clamp(max=freq_bins_large - 1)
            alpha = (position - lower).unsqueeze(0).unsqueeze(-1)
            value = large_cqt_spec[:, lower, :] + alpha * (
                large_cqt_spec[:, upper, :] - large_cqt_spec[:, lower, :]
            )
            if self.cqt_log_scale:
                value = torch.log(value + 1e-8)
            harmonic_specs.append(value)

        spec = torch.stack(harmonic_specs, dim=1)
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
        return CQTFeatureOutput(
            spec=spec.to(dtype=waveform.dtype), crop_length=crop_length
        )


class StemConv(nn.Module):
    """V1 convolutional stem: [B, C, T, F] -> [B, 4 * base_ch, T, F/4]."""

    def __init__(
        self,
        *,
        in_ch: int,
        base_ch: int,
        kernel_size: int = 3,
        n_bins: int = 312,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        _ = dropout
        if in_ch <= 0:
            raise ValueError("in_ch must be positive")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive")
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(in_ch, base_ch, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=5, padding=2)
        self.freq_embed = nn.Parameter(torch.zeros(1, base_ch, 1, n_bins))
        nn.init.normal_(self.freq_embed, std=0.02)

        self.block1 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, kernel_size, stride=(1, 1), padding=pad),
            nn.GroupNorm(4, base_ch * 2),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                base_ch * 2, base_ch * 4, kernel_size, stride=(1, 2), padding=pad
            ),
            nn.GroupNorm(4, base_ch * 4),
            nn.GELU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                base_ch * 4, base_ch * 4, kernel_size, stride=(1, 2), padding=pad
            ),
            nn.GroupNorm(4, base_ch * 4),
            nn.GELU(),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size, padding=pad),
            nn.GroupNorm(4, base_ch * 4),
        )
        self.out_ch = base_ch * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x) + self.freq_embed[:, :, :, : int(x.shape[-1])]
        x = self.conv2(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.block4(x)


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
        features = torch.stack(
            [
                x / 128.0,
                torch.sin(2 * math.pi * x / 12.0),
                torch.cos(2 * math.pi * x / 12.0),
                torch.ones_like(x),
            ],
            dim=-1,
        )
        return self.mlp(features)


class V1CQTStemBackbone(nn.Module):
    def __init__(self, config: SemiCRFModelConfig) -> None:
        super().__init__()
        if config.architecture_version != 2:
            raise ValueError("Semi-CRF model requires architecture_version=2")
        if config.lwr_ratio <= 0:
            raise ValueError("lwr_ratio must be positive")
        if config.lwr_residual_mode not in {"delta", "residual"}:
            raise ValueError("lwr_residual_mode must be one of {'delta', 'residual'}")
        if config.lwr_resampling_mode not in {"mean", "conv1d"}:
            raise ValueError("lwr_resampling_mode must be one of {'mean', 'conv1d'}")
        if config.encoder_num_layers <= 0:
            raise ValueError("encoder_num_layers must be positive")
        if config.encoder_num_heads <= 0:
            raise ValueError("encoder_num_heads must be positive")
        if config.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if config.hidden_size % config.encoder_num_heads != 0:
            raise ValueError("hidden_size must be divisible by encoder_num_heads")

        head_dim = int(config.hidden_size) // int(config.encoder_num_heads)
        if head_dim % 2 != 0:
            raise ValueError("encoder head dim must be even for RoPE")

        lwr_layer_indices = resolve_lwr_layers(
            config.lwr_layers, int(config.encoder_num_layers)
        )
        lwr_layer_index_set = set(lwr_layer_indices)

        self.config = config
        self.feature_extractor = AudioFeatureExtractor(
            sampling_rate=config.sample_rate,
            hop_length=config.hop_length,
            input_audio_channels=config.input_audio_channels,
            cqt_fmin=config.cqt_fmin,
            cqt_n_bins=config.cqt_n_bins,
            cqt_bins_per_octave=config.cqt_bins_per_octave,
            cqt_filter_scale=config.cqt_filter_scale,
            harmonics=config.harmonics,
            harmonic_dropout_p=config.harmonic_dropout_p,
            cqt_log_scale=config.cqt_log_scale,
            spec_augment_params=config.spec_augment_params,
        )
        self.stem = StemConv(
            in_ch=self.feature_extractor.num_audio_channels,
            base_ch=int(config.base_ch),
            n_bins=self.feature_extractor.n_bins,
            dropout=float(config.dropout),
        )
        self.num_bands = math.ceil(self.feature_extractor.n_bins / 4)
        self.model_dim = int(self.stem.out_ch)
        self.query_feature_dim = self.model_dim
        self.pitch_query_embed = PitchQueryEmbedding(dim=self.model_dim)
        self.register_buffer(
            "midi_pitches",
            torch.arange(
                MIN_MIDI_PITCH,
                MIN_MIDI_PITCH + int(config.pitch_query_count),
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.band_type_embed = nn.Parameter(torch.zeros(1, 1, 1, self.model_dim))
        self.pitch_type_embed = nn.Parameter(torch.zeros(1, 1, 1, self.model_dim))
        self.lwr_layer_indices = lwr_layer_indices
        self.layers = nn.ModuleList(
            [
                DualAxisTransformerLayer(
                    dim=self.model_dim,
                    head_dim=head_dim,
                    num_heads=int(config.encoder_num_heads),
                    dropout=float(config.dropout),
                    lwr_ratio=(
                        int(config.lwr_ratio)
                        if layer_index in lwr_layer_index_set
                        else 1
                    ),
                    lwr_residual_mode=str(config.lwr_residual_mode),
                    lwr_resampling_mode=str(config.lwr_resampling_mode),
                    axis_order="band_time",
                )
                for layer_index in range(int(config.encoder_num_layers))
            ]
        )
        self.final_norm = RMSNorm(self.model_dim)

    @staticmethod
    def _match_time_length_pitch(tokens: torch.Tensor, target_T: int) -> torch.Tensor:
        if int(tokens.shape[1]) < int(target_T):
            return F.pad(tokens, (0, 0, 0, 0, 0, int(target_T) - int(tokens.shape[1])))
        return tokens[:, :target_T]

    def forward(self, waveform: torch.Tensor) -> BackboneOutput:
        context = self.feature_extractor(waveform)
        stem_features = self.stem(context.spec)
        tokens = einops.rearrange(stem_features, "b d t f -> b t f d")
        batch_size, time_steps, num_bands, _ = tokens.shape

        pitch_query = self.pitch_query_embed(self.midi_pitches.to(waveform.device))
        pitch_query = einops.repeat(
            pitch_query, "p d -> b t p d", b=batch_size, t=time_steps
        )
        tokens = torch.cat(
            [tokens + self.band_type_embed, pitch_query + self.pitch_type_embed],
            dim=2,
        )

        use_checkpoint = (
            self.config.use_gradient_checkpoint
            and self.training
            and torch.is_grad_enabled()
        )
        for layer in self.layers:
            tokens = layer(tokens, use_checkpoint=use_checkpoint)
        tokens = self.final_norm(tokens)

        band_tokens = tokens[:, :, :num_bands, :].contiguous()
        pitch_tokens = self._match_time_length_pitch(
            tokens[:, :, num_bands:, :].contiguous(),
            target_T=int(context.crop_length),
        )

        return BackboneOutput(
            band_features=einops.rearrange(
                band_tokens, "b t k d -> b k t d"
            ).contiguous(),
            pitch_query_features=pitch_tokens.contiguous(),
        )


class TaskFeatureAdapter(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class IntervalScorer(nn.Module):
    def __init__(self, input_dim: int, head_dim: int) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        self.head_dim = int(head_dim)
        self.proj = nn.Linear(input_dim, self.head_dim * 2 + 1)
        self.query_scale = 1.0 / math.sqrt(float(self.head_dim))

    def forward(
        self, pitch_query_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        interval_proj = self.proj(pitch_query_features)
        interval_query, interval_key, interval_diag = torch.split(
            interval_proj, [self.head_dim, self.head_dim, 1], dim=-1
        )
        return (
            interval_query * self.query_scale,
            interval_key,
            interval_diag.squeeze(-1),
        )


class InstrumentPairGate(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        gate_dim: int,
        num_instruments: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if gate_dim <= 0:
            raise ValueError("gate_dim must be positive")
        self.gate_dim = int(gate_dim)
        self.pitch_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.gate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.gate_dim, self.gate_dim),
        )
        self.instrument_proj = nn.Linear(input_dim, self.gate_dim, bias=False)
        self.instrument_bias = nn.Parameter(torch.zeros(num_instruments))
        self.scale = 1.0 / math.sqrt(float(self.gate_dim))

    def forward(
        self,
        pitch_features: torch.Tensor,
        instrument_embeddings: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pitch_features.dim() != 4:
            raise ValueError("pitch_features must have shape [B, T, P, D]")
        if frame_valid_mask.dim() != 2:
            raise ValueError("frame_valid_mask must have shape [B, T]")
        mask = (
            frame_valid_mask.to(dtype=pitch_features.dtype).unsqueeze(-1).unsqueeze(-1)
        )
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled_pitch = (pitch_features * mask).sum(dim=1) / denom
        pitch_gate = self.pitch_proj(pooled_pitch)
        instrument_gate = self.instrument_proj(instrument_embeddings)
        logits = torch.einsum("bpg,ig->bip", pitch_gate, instrument_gate) * self.scale
        return logits + einops.rearrange(self.instrument_bias, "i -> 1 i 1")


class IntervalBoundaryPredictor(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 4),
        )

    def forward(self, interval_features: torch.Tensor) -> torch.Tensor:
        return self.net(interval_features)


class AudioSemiCRFTransformer(nn.Module):
    def __init__(self, config: SemiCRFModelConfig) -> None:
        super().__init__()
        if config.pitch_query_count <= 0:
            raise ValueError("pitch_query_count must be positive")
        if int(config.pitch_query_count) != NUM_PITCHES:
            raise ValueError(f"model requires pitch_query_count={NUM_PITCHES}")
        if config.semi_crf_length_scaling not in {"linear", "sqrt", "none"}:
            raise ValueError(
                "semi_crf_length_scaling must be one of {'linear', 'sqrt', 'none'}"
            )
        if config.num_instrument_classes <= 0:
            raise ValueError("num_instrument_classes must be positive")
        if int(config.num_instrument_classes) != NUM_INSTRUMENT_CLASSES:
            raise ValueError(
                f"model requires num_instrument_classes={NUM_INSTRUMENT_CLASSES}"
            )

        self.config = config
        self.backbone = V1CQTStemBackbone(config)
        self.interval_adapter = TaskFeatureAdapter(
            input_dim=self.backbone.query_feature_dim,
            dropout=float(config.dropout),
        )
        self.instrument_embedding = nn.Embedding(
            int(config.num_instrument_classes), self.backbone.query_feature_dim
        )
        self.pair_gate = InstrumentPairGate(
            input_dim=self.backbone.query_feature_dim,
            gate_dim=int(config.instrument_pair_gate_dim),
            num_instruments=int(config.num_instrument_classes),
            dropout=float(config.dropout),
        )
        self.interval_scorer = IntervalScorer(
            input_dim=self.backbone.query_feature_dim,
            head_dim=int(config.semi_crf_head_dim),
        )
        self.interval_boundary_predictor = (
            IntervalBoundaryPredictor(
                input_dim=self.backbone.query_feature_dim,
                dropout=float(config.dropout),
            )
            if config.use_interval_boundary_head
            else None
        )

    def supports_interval_boundaries(self) -> bool:
        return self.interval_boundary_predictor is not None

    def _build_frame_valid_mask(
        self,
        *,
        batch_size: int,
        num_frames: int,
        valid_audio_frames: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if valid_audio_frames is None:
            return torch.ones(batch_size, num_frames, dtype=torch.bool, device=device)
        lengths = [
            compute_model_frames(
                int(frame_count), int(self.config.n_fft), int(self.config.hop_length)
            )
            for frame_count in valid_audio_frames.tolist()
        ]
        lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
        positions = torch.arange(num_frames, device=device, dtype=torch.long).unsqueeze(
            0
        )
        return positions < lengths_tensor.unsqueeze(1)

    def build_pair_interval_features(
        self,
        interval_features: torch.Tensor,
        selected_pair_ids: Sequence[Sequence[int] | torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if interval_features.dim() != 4:
            raise ValueError("interval_features must have shape [B, T, P, D]")
        batch_size, time_steps, num_pitches, feature_dim = interval_features.shape
        if int(num_pitches) != NUM_PITCHES:
            raise ValueError(
                f"expected {NUM_PITCHES} pitch features, got {num_pitches}"
            )
        if len(selected_pair_ids) != int(batch_size):
            raise ValueError("selected_pair_ids length must match batch size")

        pair_feature_chunks: list[torch.Tensor] = []
        batch_indices: list[torch.Tensor] = []
        pair_id_chunks: list[torch.Tensor] = []
        for batch_index, pair_ids_for_sample in enumerate(selected_pair_ids):
            pair_ids = (
                pair_ids_for_sample.to(
                    device=interval_features.device, dtype=torch.long
                )
                if isinstance(pair_ids_for_sample, torch.Tensor)
                else torch.tensor(
                    list(pair_ids_for_sample),
                    device=interval_features.device,
                    dtype=torch.long,
                )
            )
            if int(pair_ids.numel()) == 0:
                continue
            instrument_ids = torch.div(pair_ids, NUM_PITCHES, rounding_mode="floor")
            pitch_ids = pair_ids.remainder(NUM_PITCHES)
            if torch.any(instrument_ids < 0) or torch.any(
                instrument_ids >= int(self.config.num_instrument_classes)
            ):
                raise ValueError("selected pair contains out-of-range instrument id")
            pitch_features = interval_features[batch_index, :, pitch_ids, :]
            instrument_features = self.instrument_embedding(instrument_ids).unsqueeze(0)
            pair_feature_chunks.append(pitch_features + instrument_features)
            batch_indices.append(
                torch.full(
                    (int(pair_ids.numel()),),
                    int(batch_index),
                    device=interval_features.device,
                    dtype=torch.long,
                )
            )
            pair_id_chunks.append(pair_ids)

        if not pair_feature_chunks:
            return (
                interval_features.new_zeros((int(time_steps), 0, int(feature_dim))),
                torch.zeros((0,), device=interval_features.device, dtype=torch.long),
                torch.zeros((0,), device=interval_features.device, dtype=torch.long),
            )
        return (
            torch.cat(pair_feature_chunks, dim=1).contiguous(),
            torch.cat(batch_indices, dim=0),
            torch.cat(pair_id_chunks, dim=0),
        )

    def score_pair_interval_features(
        self, pair_interval_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.interval_scorer(pair_interval_features)

    def predict_interval_boundaries(
        self,
        pitch_query_features: torch.Tensor,
        interval_batch: Sequence[Sequence[Sequence[tuple[int, int]]]],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int, int]]]:
        if self.interval_boundary_predictor is None:
            return pitch_query_features.new_zeros((0, 4)), []
        interval_features, entries = gather_interval_endpoint_features(
            pitch_query_features, interval_batch
        )
        if not entries:
            return pitch_query_features.new_zeros((0, 4)), []
        return self.interval_boundary_predictor(interval_features), entries

    def predict_flat_interval_boundaries(
        self,
        pair_interval_features: torch.Tensor,
        interval_batch: Sequence[Sequence[tuple[int, int]]],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        if self.interval_boundary_predictor is None:
            return pair_interval_features.new_zeros((0, 4)), []
        if pair_interval_features.dim() != 3:
            raise ValueError("pair_interval_features must have shape [T, N, D]")
        entries: list[tuple[int, int, int, int]] = []
        for track_index, track_intervals in enumerate(interval_batch):
            for interval_index, (begin, end) in enumerate(track_intervals):
                entries.append(
                    (int(track_index), int(interval_index), int(begin), int(end))
                )
        feature_dim = int(pair_interval_features.shape[-1])
        if not entries:
            return pair_interval_features.new_zeros((0, 4)), []
        device = pair_interval_features.device
        track_indices = torch.tensor(
            [entry[0] for entry in entries], device=device, dtype=torch.long
        )
        begin_indices = torch.tensor(
            [entry[2] for entry in entries], device=device, dtype=torch.long
        )
        end_indices = torch.tensor(
            [entry[3] for entry in entries], device=device, dtype=torch.long
        )
        begin_features = pair_interval_features[begin_indices, track_indices]
        end_features = pair_interval_features[end_indices, track_indices]
        endpoint_features = torch.cat(
            [begin_features, end_features, begin_features * end_features], dim=-1
        )
        if int(endpoint_features.shape[-1]) != feature_dim * 3:
            raise ValueError("unexpected boundary feature shape")
        return self.interval_boundary_predictor(endpoint_features), entries

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        valid_audio_frames: Optional[torch.Tensor] = None,
        include_amt: bool = True,
        include_aux_outputs: bool = True,
        **_: object,
    ) -> dict[str, torch.Tensor | None]:
        if not include_amt:
            raise ValueError("model only supports AMT inference/training")

        backbone_output = self.backbone(waveform)
        pitch_query_features = backbone_output.pitch_query_features
        interval_features = self.interval_adapter(pitch_query_features)
        frame_valid_mask = self._build_frame_valid_mask(
            batch_size=int(waveform.shape[0]),
            num_frames=int(interval_features.shape[1]),
            valid_audio_frames=valid_audio_frames,
            device=waveform.device,
        )
        pair_gate_logits = self.pair_gate(
            interval_features,
            self.instrument_embedding.weight,
            frame_valid_mask,
        )

        return {
            "band_features": backbone_output.band_features
            if include_aux_outputs
            else None,
            "global_features": None,
            "pitch_query_features": pitch_query_features
            if include_aux_outputs
            else None,
            "interval_features": interval_features,
            "pair_gate_logits": pair_gate_logits,
            "frame_valid_mask": frame_valid_mask,
        }
