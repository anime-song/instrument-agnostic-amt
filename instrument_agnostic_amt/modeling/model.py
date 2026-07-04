from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import einops
import torch
import torch.nn as nn

from .bands.split import BandSplit, build_band_indices
from .blocks.axis_transformer import DualAxisTransformerLayer
from .blocks.transformer import RMSNorm
from .conditioning import InstrumentConditioner
from .features.stft import STFTFeatureExtractor
from .heads.interval_boundaries import gather_interval_endpoint_features
from ..taxonomy.instrument_classes import NUM_INSTRUMENT_CLASSES

MIN_MIDI_PITCH = 21
MAX_MIDI_PITCH = 108
NUM_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1


def compute_model_frames(num_audio_frames: int, n_fft: int, hop_length: int) -> int:
    return math.ceil(num_audio_frames / hop_length)


@dataclass(frozen=True)
class SemiCRFModelConfig:
    sample_rate: int
    hop_length: int
    n_fft: int = 2048
    architecture_version: int = 2
    feature_extractor: str = "stft"
    band_split_type: str = "bs"
    lwr_mode: str = "all"
    lwr_ratio: int = 4
    hidden_size: int = 384
    encoder_num_layers: int = 6
    encoder_num_heads: int = 12
    dropout: float = 0.1
    use_gradient_checkpoint: bool = True
    pitch_query_count: int = NUM_PITCHES
    num_pitch_slots: int = 1
    semi_crf_head_dim: int = 256
    semi_crf_length_scaling: str = "none"
    semi_crf_length_penalty: float = 0.0
    use_interval_boundary_head: bool = True
    num_instrument_classes: int = NUM_INSTRUMENT_CLASSES
    spec_augment_params: dict[str, float | int] | None = None


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


class ConditionedBandSplitBackbone(nn.Module):
    def __init__(self, config: SemiCRFModelConfig) -> None:
        super().__init__()
        if config.architecture_version != 2:
            raise ValueError("V2 model requires architecture_version=2")
        if config.feature_extractor != "stft":
            raise ValueError("V2 model only supports feature_extractor='stft'")
        if config.band_split_type != "bs":
            raise ValueError("V2 model currently only supports band_split_type='bs'")
        if config.lwr_mode != "all":
            raise ValueError("V2 model currently only supports lwr_mode='all'")
        if config.hidden_size % config.encoder_num_heads != 0:
            raise ValueError("hidden_size must be divisible by encoder_num_heads")

        self.config = config
        self.feature_extractor = STFTFeatureExtractor(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            input_audio_channels=2,
        )
        _, band_indices = build_band_indices(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
        )
        self.band_split = BandSplit(
            hidden_size=config.hidden_size,
            band_indices=band_indices,
            num_channels=self.feature_extractor.output_channels,
        )
        self.num_bands = self.band_split.num_bands
        self.model_dim = int(config.hidden_size)
        self.query_feature_dim = self.model_dim
        self.pitch_query_embed = PitchQueryEmbedding(dim=self.model_dim)
        self.conditioner = InstrumentConditioner(
            num_instruments=config.num_instrument_classes,
            dim=self.model_dim,
            dropout=config.dropout,
        )
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
        self.condition_type_embed = nn.Parameter(torch.zeros(1, 1, 1, self.model_dim))
        self.layers = nn.ModuleList(
            [
                DualAxisTransformerLayer(
                    dim=self.model_dim,
                    head_dim=self.model_dim // int(config.encoder_num_heads),
                    num_heads=int(config.encoder_num_heads),
                    dropout=float(config.dropout),
                    lwr_ratio=int(config.lwr_ratio),
                )
                for _ in range(int(config.encoder_num_layers))
            ]
        )
        self.final_norm = RMSNorm(self.model_dim)

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        condition_instrument_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(waveform)
        band_tokens = self.band_split(features.spec)
        batch_size, time_steps, _, _ = band_tokens.shape

        condition = self.conditioner(condition_instrument_ids.to(waveform.device))
        condition_tokens = condition.view(batch_size, 1, 1, self.model_dim).expand(
            -1, time_steps, -1, -1
        )
        condition_tokens = condition_tokens + self.condition_type_embed

        pitch_query = self.pitch_query_embed(self.midi_pitches.to(waveform.device))
        pitch_query = pitch_query.view(1, 1, -1, self.model_dim).expand(
            batch_size, time_steps, -1, -1
        )
        pitch_query = (
            pitch_query
            + self.pitch_type_embed
            + condition.view(batch_size, 1, 1, self.model_dim)
        )
        band_tokens = band_tokens + self.band_type_embed
        tokens = torch.cat([band_tokens, condition_tokens, pitch_query], dim=2)

        use_checkpoint = (
            self.config.use_gradient_checkpoint
            and self.training
            and torch.is_grad_enabled()
        )
        for layer in self.layers:
            tokens = layer(tokens, use_checkpoint=use_checkpoint)

        tokens = self.final_norm(tokens)
        pitch_start = self.num_bands + 1
        pitch_tokens = tokens[:, :, pitch_start:, :]
        return pitch_tokens.contiguous(), tokens[:, :, : self.num_bands, :].contiguous()


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
        self,
        pitch_query_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        interval_proj = self.proj(pitch_query_features)
        interval_query, interval_key, interval_diag = torch.split(
            interval_proj,
            [self.head_dim, self.head_dim, 1],
            dim=-1,
        )
        return (
            interval_query * self.query_scale,
            interval_key,
            interval_diag.squeeze(-1),
        )


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
        if config.num_pitch_slots <= 0:
            raise ValueError("num_pitch_slots must be positive")
        if config.semi_crf_length_scaling not in {"linear", "sqrt", "none"}:
            raise ValueError(
                "semi_crf_length_scaling must be one of {'linear', 'sqrt', 'none'}"
            )

        self.config = config
        self.backbone = ConditionedBandSplitBackbone(config)
        self.interval_adapter = TaskFeatureAdapter(
            input_dim=self.backbone.query_feature_dim,
            dropout=config.dropout,
        )
        self.slot_embedding = nn.Embedding(
            config.num_pitch_slots,
            self.backbone.query_feature_dim,
        )
        self.interval_scorer = IntervalScorer(
            input_dim=self.backbone.query_feature_dim,
            head_dim=config.semi_crf_head_dim,
        )
        self.interval_boundary_predictor = (
            IntervalBoundaryPredictor(
                input_dim=self.backbone.query_feature_dim,
                dropout=config.dropout,
            )
            if config.use_interval_boundary_head
            else None
        )

    def supports_interval_boundaries(self) -> bool:
        return self.interval_boundary_predictor is not None

    def _expand_pitch_features_with_slots(
        self,
        pitch_features: torch.Tensor,
    ) -> torch.Tensor:
        if int(self.config.num_pitch_slots) <= 1:
            return pitch_features
        batch_size, time_steps, num_pitches, feature_dim = pitch_features.shape
        slot_embeddings = self.slot_embedding.weight.to(
            device=pitch_features.device,
            dtype=pitch_features.dtype,
        )
        expanded = pitch_features.unsqueeze(3) + slot_embeddings.view(
            1,
            1,
            1,
            int(self.config.num_pitch_slots),
            feature_dim,
        )
        return expanded.reshape(
            batch_size,
            time_steps,
            num_pitches * int(self.config.num_pitch_slots),
            feature_dim,
        )

    def predict_interval_boundaries(
        self,
        pitch_query_features: torch.Tensor,
        interval_batch: Sequence[Sequence[Sequence[tuple[int, int]]]],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int, int]]]:
        if self.interval_boundary_predictor is None:
            return pitch_query_features.new_zeros((0, 4)), []
        interval_features, entries = gather_interval_endpoint_features(
            pitch_query_features,
            interval_batch,
        )
        if not entries:
            return pitch_query_features.new_zeros((0, 4)), []
        return self.interval_boundary_predictor(interval_features), entries

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
                int(frame_count),
                self.config.n_fft,
                self.config.hop_length,
            )
            for frame_count in valid_audio_frames.tolist()
        ]
        lengths_tensor = torch.tensor(lengths, device=device, dtype=torch.long)
        positions = torch.arange(num_frames, device=device, dtype=torch.long).unsqueeze(
            0
        )
        return positions < lengths_tensor.unsqueeze(1)

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        condition_instrument_ids: torch.Tensor,
        valid_audio_frames: Optional[torch.Tensor] = None,
        include_amt: bool = True,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        if not include_amt:
            raise ValueError("V2 model only supports AMT inference/training")

        pitch_query_features, band_features = self.backbone(
            waveform,
            condition_instrument_ids=condition_instrument_ids,
        )
        interval_features = self.interval_adapter(pitch_query_features)
        interval_track_features = self._expand_pitch_features_with_slots(
            interval_features
        )
        interval_query, interval_key, interval_diag = self.interval_scorer(
            interval_track_features
        )
        frame_valid_mask = self._build_frame_valid_mask(
            batch_size=int(waveform.shape[0]),
            num_frames=int(interval_query.shape[1]),
            valid_audio_frames=valid_audio_frames,
            device=waveform.device,
        )

        return {
            "band_features": band_features.permute(0, 2, 1, 3).contiguous(),
            "global_features": None,
            "pitch_query_features": pitch_query_features,
            "interval_query": interval_query,
            "interval_key": interval_key,
            "interval_diag": interval_diag,
            "interval_features": interval_track_features,
            "frame_valid_mask": frame_valid_mask,
        }
