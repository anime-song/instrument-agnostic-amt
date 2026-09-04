from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...amt.modeling.backbone import AudioFeatureExtractor, V1Backbone


@dataclass(frozen=True)
class VelocityModelConfig:
    sample_rate: int = 22_050
    hop_length: int = 512
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
    use_gradient_checkpoint: bool = True
    pitch_min: int = 21
    pitch_max: int = 108
    num_stem_classes: int = 7
    note_hidden_size: int = 256
    local_frame_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2, 4)
    # Defaults preserve checkpoints written before these options were added.
    use_absolute_velocity_energy: bool = False
    predict_stem_gain: bool = True

    def __post_init__(self) -> None:
        if self.sample_rate <= 0 or self.hop_length <= 0:
            raise ValueError("sample_rate and hop_length must be positive")
        if self.input_audio_channels != 2:
            raise ValueError("the reusable V1 backbone requires stereo input")
        if self.pitch_min < 0 or self.pitch_max > 127 or self.pitch_min > self.pitch_max:
            raise ValueError("invalid MIDI pitch range")
        if self.num_stem_classes <= 0 or self.note_hidden_size <= 0:
            raise ValueError("head dimensions must be positive")
        if not self.local_frame_offsets:
            raise ValueError("local_frame_offsets cannot be empty")

    @property
    def pitch_query_count(self) -> int:
        return self.pitch_max - self.pitch_min + 1


def load_velocity_model_config(path: str | Path) -> VelocityModelConfig:
    with Path(path).open("r", encoding="utf-8") as file:
        raw = json.load(file)
    if not isinstance(raw, Mapping):
        raise ValueError("velocity model config must be a JSON object")
    allowed = {field.name for field in fields(VelocityModelConfig)}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown velocity model config fields: {', '.join(unknown)}")
    values: dict[str, Any] = dict(raw)
    for key in ("harmonics", "local_frame_offsets"):
        if key in values:
            values[key] = tuple(values[key])
    return VelocityModelConfig(**values)


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    dim: int,
) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator


class VelocityPredictionModel(nn.Module):
    """Predict per-note MIDI velocity and, optionally, relative stem gain.

    The AMT V1 HCQT backbone is shared architecturally, but this model and its
    checkpoints remain separate from the transcription model.  MIDI notes act
    as queries into pitch-conditioned audio frames around each onset.
    """

    def __init__(
        self,
        config: VelocityModelConfig,
        *,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if backbone is None:
            feature_extractor = AudioFeatureExtractor(
                sampling_rate=config.sample_rate,
                hop_length=config.hop_length,
                cqt_fmin=config.cqt_fmin,
                cqt_n_bins=config.cqt_n_bins,
                cqt_bins_per_octave=config.cqt_bins_per_octave,
                cqt_filter_scale=config.cqt_filter_scale,
                harmonics=config.harmonics,
                harmonic_dropout_p=config.harmonic_dropout_p,
                cqt_log_scale=config.cqt_log_scale,
                input_audio_channels=config.input_audio_channels,
            )
            backbone = V1Backbone(
                feature_extractor=feature_extractor,
                hidden_size=config.hidden_size,
                base_ch=config.base_ch,
                num_layers=config.encoder_num_layers,
                num_heads=config.encoder_num_heads,
                num_pitch_queries=config.pitch_query_count,
                use_global_token=False,
                dropout=config.dropout,
                use_gradient_checkpoint=config.use_gradient_checkpoint,
            )
        if not hasattr(backbone, "query_feature_dim"):
            raise ValueError("backbone must expose query_feature_dim")
        self.backbone = backbone
        backbone_dim = int(getattr(backbone, "query_feature_dim"))
        hidden = int(config.note_hidden_size)

        self.pitch_embedding = nn.Embedding(128, 32)
        self.program_embedding = nn.Embedding(128, 32)
        self.drum_embedding = nn.Embedding(2, 8)
        # The final index is reserved for padded/unknown input at the head.
        self.stem_embedding = nn.Embedding(config.num_stem_classes + 1, 24)
        self.duration_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 16),
        )
        metadata_dim = 32 + 32 + 8 + 24 + 16
        self.note_query = nn.Sequential(
            nn.Linear(metadata_dim, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, hidden),
        )

        # Three log-energy channels are appended because the HCQT backbone
        # normalizes away overall level. New velocity-only models retain dBFS.
        self.local_audio_projection = nn.Linear(backbone_dim + 3, hidden)
        self.local_attention = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.note_fusion = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.velocity_head = nn.Linear(hidden, 127)

        if config.predict_stem_gain:
            self.global_audio_projection: nn.Module | None = nn.Sequential(
                nn.Linear(backbone_dim, hidden),
                nn.GELU(),
            )
            self.stem_gain_head: nn.Module | None = nn.Sequential(
                nn.Linear(hidden * 2 + 24 + 2, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, 1),
            )
        else:
            self.global_audio_projection = None
            self.stem_gain_head = None
        self.register_buffer(
            "local_offsets",
            torch.tensor(config.local_frame_offsets, dtype=torch.float32),
            persistent=False,
        )

    def _frame_valid_mask(
        self,
        audio: torch.Tensor,
        *,
        num_frames: int,
        valid_audio_frames: torch.Tensor | None,
    ) -> torch.Tensor:
        if valid_audio_frames is None:
            lengths = torch.full(
                (audio.shape[0], audio.shape[1]),
                num_frames,
                dtype=torch.long,
                device=audio.device,
            )
        else:
            lengths = torch.div(
                valid_audio_frames.to(device=audio.device, dtype=torch.long)
                + self.config.hop_length
                - 1,
                self.config.hop_length,
                rounding_mode="floor",
            ).clamp(max=num_frames)
        return (
            torch.arange(num_frames, device=audio.device)[None, None, :]
            < lengths[:, :, None]
        )

    def _relative_energy(
        self,
        audio: torch.Tensor,
        *,
        num_frames: int,
        frame_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, stem_count, channel_count, sample_count = audio.shape
        x = audio.float().reshape(
            batch_size * stem_count, channel_count, sample_count
        )
        channel_power = F.avg_pool1d(
            x.square(),
            kernel_size=max(1, self.config.hop_length * 2),
            stride=self.config.hop_length,
            padding=self.config.hop_length // 2,
            ceil_mode=True,
        )
        mono = x.mean(dim=1, keepdim=True)
        mono_power = F.avg_pool1d(
            mono.square(),
            kernel_size=max(1, self.config.hop_length * 2),
            stride=self.config.hop_length,
            padding=self.config.hop_length // 2,
            ceil_mode=True,
        )
        energy = torch.cat((channel_power, mono_power), dim=1)
        energy = F.interpolate(energy, size=num_frames, mode="linear", align_corners=False)
        log_energy = 10.0 * torch.log10(energy.clamp_min(1e-10))
        log_energy = log_energy.transpose(1, 2).reshape(
            batch_size, stem_count, num_frames, 3
        )
        if self.config.use_absolute_velocity_energy:
            # float audio dBFS: -100 dB maps to -2 and 0 dB maps to +2.
            return ((log_energy + 50.0) / 25.0).clamp(-2.0, 2.0)
        center = _masked_mean(log_energy, frame_valid_mask, dim=2).unsqueeze(2)
        return (log_energy - center).clamp(-60.0, 60.0) / 30.0

    def _observed_relative_stem_level(
        self,
        audio: torch.Tensor,
        *,
        valid_audio_frames: torch.Tensor | None,
        stem_mask: torch.Tensor,
    ) -> torch.Tensor:
        sample_count = int(audio.shape[-1])
        if valid_audio_frames is None:
            valid_audio_frames = torch.full(
                audio.shape[:2],
                sample_count,
                dtype=torch.long,
                device=audio.device,
            )
        sample_mask = (
            torch.arange(sample_count, device=audio.device)[None, None, :]
            < valid_audio_frames.to(device=audio.device)[:, :, None]
        )
        mono_power = audio.float().square().mean(dim=2)
        power = (mono_power * sample_mask).sum(dim=2) / sample_mask.sum(
            dim=2
        ).clamp_min(1)
        level_db = 10.0 * torch.log10(power.clamp_min(1e-10))
        center = _masked_mean(level_db, stem_mask, dim=1).unsqueeze(1)
        return ((level_db - center) / 20.0).clamp(-3.0, 3.0).unsqueeze(-1)

    def _gather_local_features(
        self,
        pitch_features: torch.Tensor,
        energy_features: torch.Tensor,
        *,
        note_start_seconds: torch.Tensor,
        note_pitch: torch.Tensor,
        note_stem_index: torch.Tensor,
        note_mask: torch.Tensor,
        frame_valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, num_frames, _, _ = pitch_features.shape
        num_notes = int(note_pitch.shape[1])
        offsets = self.local_offsets.to(device=pitch_features.device)
        positions = (
            note_start_seconds.to(dtype=torch.float32)
            * (float(self.config.sample_rate) / float(self.config.hop_length))
        )[:, :, None] + offsets[None, None, :]
        lower_unclamped = torch.floor(positions).long()
        upper_unclamped = lower_unclamped + 1
        alpha = (positions - lower_unclamped.to(dtype=positions.dtype)).to(
            dtype=pitch_features.dtype
        )
        lower = lower_unclamped.clamp(0, max(0, num_frames - 1))
        upper = upper_unclamped.clamp(0, max(0, num_frames - 1))
        pitch_index = (note_pitch - self.config.pitch_min).clamp(
            0, self.config.pitch_query_count - 1
        )
        safe_stem_index = note_stem_index.clamp(
            0, max(0, pitch_features.shape[1] - 1)
        )
        batch_index = torch.arange(batch_size, device=pitch_features.device).view(
            batch_size, 1, 1
        )
        batch_index = batch_index.expand(batch_size, num_notes, offsets.numel())
        pitch_index = pitch_index[:, :, None].expand_as(lower)
        stem_index = safe_stem_index[:, :, None].expand_as(lower)
        lower_pitch = pitch_features[batch_index, stem_index, lower, pitch_index]
        upper_pitch = pitch_features[batch_index, stem_index, upper, pitch_index]
        pitch_local = lower_pitch + alpha[..., None] * (upper_pitch - lower_pitch)
        lower_energy = energy_features[batch_index, stem_index, lower]
        upper_energy = energy_features[batch_index, stem_index, upper]
        energy_local = lower_energy + alpha[..., None].float() * (
            upper_energy - lower_energy
        )
        local = torch.cat(
            (pitch_local, energy_local.to(dtype=pitch_local.dtype)),
            dim=-1,
        )
        valid_lengths = frame_valid_mask.sum(dim=2).gather(1, safe_stem_index)
        local_mask = (
            note_mask[:, :, None]
            & (positions >= 0.0)
            & (positions < valid_lengths[:, :, None].to(dtype=positions.dtype))
        )
        return local, local_mask

    def _note_metadata(
        self,
        *,
        note_start_seconds: torch.Tensor,
        note_end_seconds: torch.Tensor,
        note_pitch: torch.Tensor,
        note_program: torch.Tensor,
        note_is_drum: torch.Tensor,
        note_stem_index: torch.Tensor,
        stem_class_id: torch.Tensor,
    ) -> torch.Tensor:
        safe_stem_index = note_stem_index.clamp(0, max(0, stem_class_id.shape[1] - 1))
        note_stem_class = stem_class_id.gather(1, safe_stem_index)
        note_stem_class = torch.where(
            (note_stem_class >= 0)
            & (note_stem_class < self.config.num_stem_classes),
            note_stem_class,
            torch.full_like(note_stem_class, self.config.num_stem_classes),
        )
        duration = (note_end_seconds - note_start_seconds).clamp_min(0.0)
        duration_feature = self.duration_encoder(torch.log1p(duration).unsqueeze(-1))
        metadata = torch.cat(
            (
                self.pitch_embedding(note_pitch.clamp(0, 127)),
                self.program_embedding(note_program.clamp(0, 127)),
                self.drum_embedding(note_is_drum.long().clamp(0, 1)),
                self.stem_embedding(note_stem_class),
                duration_feature,
            ),
            dim=-1,
        )
        return self.note_query(metadata)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_scores = scores.masked_fill(~mask, -torch.inf)
        has_value = mask.any(dim=-1, keepdim=True)
        safe_scores = torch.where(has_value, masked_scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=-1) * mask.to(dtype=scores.dtype)
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _pool_stem_notes(
        self,
        note_features: torch.Tensor,
        *,
        note_stem_index: torch.Tensor,
        note_mask: torch.Tensor,
        num_stems: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, hidden = note_features.shape
        sums = note_features.new_zeros(batch_size, num_stems, hidden)
        counts = note_features.new_zeros(batch_size, num_stems, 1)
        safe_index = note_stem_index.clamp(0, max(0, num_stems - 1))
        weights = note_mask.to(dtype=note_features.dtype).unsqueeze(-1)
        sums.scatter_add_(1, safe_index.unsqueeze(-1).expand(-1, -1, hidden), note_features * weights)
        counts.scatter_add_(1, safe_index.unsqueeze(-1), weights)
        return sums / counts.clamp_min(1.0), counts

    def forward(
        self,
        audio: torch.Tensor,
        *,
        note_start_seconds: torch.Tensor,
        note_end_seconds: torch.Tensor,
        note_pitch: torch.Tensor,
        note_program: torch.Tensor,
        note_is_drum: torch.Tensor,
        note_stem_index: torch.Tensor,
        stem_class_id: torch.Tensor,
        note_mask: torch.Tensor | None = None,
        stem_mask: torch.Tensor | None = None,
        valid_audio_frames: torch.Tensor | None = None,
        include_aux_outputs: bool = False,
        include_stem_gain: bool = True,
    ) -> dict[str, torch.Tensor]:
        if audio.ndim != 4 or audio.shape[2] != 2:
            raise ValueError("audio must have shape [B, S, 2, T]")
        if note_mask is None:
            note_mask = torch.ones_like(note_pitch, dtype=torch.bool)
        if stem_mask is None:
            stem_mask = stem_class_id >= 0
        batch_size, stem_count, channel_count, sample_count = audio.shape
        flat_audio = audio.reshape(
            batch_size * stem_count, channel_count, sample_count
        )
        if isinstance(self.backbone, V1Backbone):
            backbone_output = self.backbone(
                flat_audio,
                include_aux_outputs=False,
            )
        else:
            backbone_output = self.backbone(flat_audio)
        flat_pitch_features = backbone_output.pitch_query_features
        if flat_pitch_features.shape[2] != self.config.pitch_query_count:
            raise ValueError("backbone pitch-query count does not match model config")
        pitch_features = flat_pitch_features.reshape(
            batch_size,
            stem_count,
            flat_pitch_features.shape[1],
            flat_pitch_features.shape[2],
            flat_pitch_features.shape[3],
        )
        frame_valid_mask = self._frame_valid_mask(
            audio,
            num_frames=int(pitch_features.shape[2]),
            valid_audio_frames=valid_audio_frames,
        )
        energy_features = self._relative_energy(
            audio,
            num_frames=int(pitch_features.shape[2]),
            frame_valid_mask=frame_valid_mask,
        )
        local, local_mask = self._gather_local_features(
            pitch_features,
            energy_features,
            note_start_seconds=note_start_seconds,
            note_pitch=note_pitch,
            note_stem_index=note_stem_index,
            note_mask=note_mask,
            frame_valid_mask=frame_valid_mask,
        )
        query = self._note_metadata(
            note_start_seconds=note_start_seconds,
            note_end_seconds=note_end_seconds,
            note_pitch=note_pitch,
            note_program=note_program,
            note_is_drum=note_is_drum,
            note_stem_index=note_stem_index,
            stem_class_id=stem_class_id,
        )
        projected_local = self.local_audio_projection(local)
        attention_scores = self.local_attention(
            projected_local + query.unsqueeze(2)
        ).squeeze(-1)
        attention = self._masked_softmax(attention_scores, local_mask)
        pooled_local = (projected_local * attention.unsqueeze(-1)).sum(dim=2)
        note_features = self.note_fusion(pooled_local + query)
        note_features = note_features * note_mask.unsqueeze(-1).to(note_features.dtype)
        velocity_logits = self.velocity_head(note_features)
        velocity_values = torch.arange(
            1,
            128,
            device=velocity_logits.device,
            dtype=velocity_logits.dtype,
        )
        velocity_expected = (
            torch.softmax(velocity_logits, dim=-1) * velocity_values
        ).sum(dim=-1)

        outputs = {
            "velocity_logits": velocity_logits,
            "velocity_expected": velocity_expected,
        }
        if (
            include_stem_gain
            and self.stem_gain_head is not None
            and self.global_audio_projection is not None
        ):
            global_frames = pitch_features.mean(dim=3)
            global_audio = _masked_mean(global_frames, frame_valid_mask, dim=2)
            global_audio = self.global_audio_projection(global_audio)
            num_stems = int(stem_class_id.shape[1])
            stem_note_features, stem_note_counts = self._pool_stem_notes(
                note_features,
                note_stem_index=note_stem_index,
                note_mask=note_mask,
                num_stems=num_stems,
            )
            safe_stem_class = torch.where(
                (stem_class_id >= 0)
                & (stem_class_id < self.config.num_stem_classes),
                stem_class_id,
                torch.full_like(stem_class_id, self.config.num_stem_classes),
            )
            observed_relative_level = self._observed_relative_stem_level(
                audio,
                valid_audio_frames=valid_audio_frames,
                stem_mask=stem_mask,
            )
            stem_features = torch.cat(
                (
                    global_audio,
                    stem_note_features,
                    self.stem_embedding(safe_stem_class),
                    torch.log1p(stem_note_counts),
                    observed_relative_level,
                ),
                dim=-1,
            )
            raw_stem_gain_db = self.stem_gain_head(stem_features).squeeze(-1)
            gain_center = _masked_mean(
                raw_stem_gain_db, stem_mask, dim=1
            ).unsqueeze(1)
            stem_gain_db = (raw_stem_gain_db - gain_center) * stem_mask.to(
                dtype=raw_stem_gain_db.dtype
            )
            outputs.update(
                {
                    "stem_gain_db": stem_gain_db,
                    "raw_stem_gain_db": raw_stem_gain_db,
                }
            )
        if include_aux_outputs:
            outputs.update(
                {
                    "note_features": note_features,
                    "local_attention": attention,
                    "frame_valid_mask": frame_valid_mask,
                }
            )
        return outputs
