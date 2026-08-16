from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..taxonomy.instrument_classes import NUM_INSTRUMENT_CLASSES
from .backbone import AudioFeatureExtractor, V1Backbone
from .heads.v1 import V1SemiCRFHead
from .heads.v2 import SelectedPairIndices, V2OverlapSemiCRFHead

MIN_MIDI_PITCH = 21
MAX_MIDI_PITCH = 108
NUM_PITCHES = MAX_MIDI_PITCH - MIN_MIDI_PITCH + 1
SEMI_CRF_VERSIONS = ("v1", "v2")


def compute_model_frames(num_audio_frames: int, n_fft: int, hop_length: int) -> int:
    _ = n_fft
    return math.ceil(num_audio_frames / hop_length)


def normalize_semi_crf_version(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "legacy": "v1",
        "pitch": "v1",
        "overlap": "v2",
        "instrument_pitch": "v2",
        "instrument-pitch": "v2",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SEMI_CRF_VERSIONS:
        raise ValueError(
            f"semi_crf_version must be one of {SEMI_CRF_VERSIONS}, got {value!r}"
        )
    return normalized


@dataclass(frozen=True)
class SemiCRFModelConfig:
    sample_rate: int
    hop_length: int
    n_fft: int = 2048
    semi_crf_version: str = "v1"
    cqt_fmin: float = 27.5
    cqt_n_bins: int = 312
    cqt_bins_per_octave: int = 36
    cqt_filter_scale: float = 0.475
    harmonics: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    harmonic_dropout_p: float = 0.0
    cqt_log_scale: bool = False
    spec_augment_params: dict[str, float | int | str] | None = None
    input_audio_channels: int = 2
    hidden_size: int = 384
    base_ch: int = 64
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
    instrument_pair_gate_dim: int = 128
    use_beat_head: bool = False
    use_chord_head: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "semi_crf_version", normalize_semi_crf_version(self.semi_crf_version)
        )


LEGACY_V1_HEAD_PREFIXES = (
    "interval_adapter.",
    "instrument_adapter.",
    "slot_embedding.",
    "interval_scorer.",
    "interval_boundary_predictor.",
    "interval_instrument_predictor.",
    "instrument_classifier.",
)


def remap_legacy_v1_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Map the flat V1 head layout to the explicit ``head`` module."""

    remapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    for raw_key, value in state_dict.items():
        key = raw_key[7:] if raw_key.startswith("module.") else raw_key
        if key.startswith(LEGACY_V1_HEAD_PREFIXES):
            key = f"head.{key}"
        remapped[key] = value
    return remapped


class AudioSemiCRFTransformer(nn.Module):
    """V1 backbone with a selectable V1 or overlap-capable V2 Semi-CRF head."""

    def __init__(self, config: SemiCRFModelConfig) -> None:
        super().__init__()
        if int(config.pitch_query_count) != NUM_PITCHES:
            raise ValueError(f"pitch_query_count must be {NUM_PITCHES}")
        if config.input_audio_channels != 2:
            raise ValueError("the V1 backbone requires stereo input_audio_channels=2")
        if config.semi_crf_length_scaling not in {"linear", "sqrt", "none"}:
            raise ValueError("invalid semi_crf_length_scaling")
        if config.num_instrument_classes <= 0:
            raise ValueError("num_instrument_classes must be positive")

        self.config = config
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
            spec_augment_params=config.spec_augment_params,
            input_audio_channels=config.input_audio_channels,
        )
        self.backbone = V1Backbone(
            feature_extractor=feature_extractor,
            hidden_size=config.hidden_size,
            base_ch=config.base_ch,
            num_layers=config.encoder_num_layers,
            num_heads=config.encoder_num_heads,
            num_pitch_queries=config.pitch_query_count,
            use_global_token=config.use_beat_head or config.use_chord_head,
            dropout=config.dropout,
            use_gradient_checkpoint=config.use_gradient_checkpoint,
        )
        if config.semi_crf_version == "v1":
            self.head: V1SemiCRFHead | V2OverlapSemiCRFHead = V1SemiCRFHead(
                input_dim=self.backbone.query_feature_dim,
                num_pitch_slots=config.num_pitch_slots,
                semi_crf_head_dim=config.semi_crf_head_dim,
                num_instrument_classes=config.num_instrument_classes,
                dropout=config.dropout,
                use_interval_boundary_head=config.use_interval_boundary_head,
            )
        else:
            self.head = V2OverlapSemiCRFHead(
                input_dim=self.backbone.query_feature_dim,
                semi_crf_head_dim=config.semi_crf_head_dim,
                instrument_pair_gate_dim=config.instrument_pair_gate_dim,
                num_instrument_classes=config.num_instrument_classes,
                dropout=config.dropout,
                use_interval_boundary_head=config.use_interval_boundary_head,
            )
        self._use_interval_instrument_head = config.semi_crf_version == "v1"

    @property
    def semi_crf_version(self) -> str:
        return self.config.semi_crf_version

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
        if valid_audio_frames.dim() != 1:
            raise ValueError("valid_audio_frames must be a 1D tensor")
        audio_lengths = valid_audio_frames.to(device=device, dtype=torch.long)
        hop_length = int(self.config.hop_length)
        complete_frames = torch.div(
            audio_lengths,
            hop_length,
            rounding_mode="floor",
        )
        has_partial_frame = torch.remainder(audio_lengths, hop_length).ne(0)
        lengths = complete_frames + has_partial_frame.to(dtype=torch.long)
        frame_indices = torch.arange(num_frames, device=device).unsqueeze(0)
        return frame_indices < lengths.unsqueeze(1)

    def forward(
        self,
        waveform: torch.Tensor,
        *,
        valid_audio_frames: Optional[torch.Tensor] = None,
        include_amt: bool = True,
        include_aux_outputs: bool = True,
        include_frame_instrument_logits: bool = True,
        **_: object,
    ) -> dict[str, torch.Tensor | None]:
        if not include_amt:
            raise ValueError("this model exposes only the AMT task")
        backbone_output = self.backbone(
            waveform,
            include_aux_outputs=include_aux_outputs,
        )
        pitch_features = backbone_output.pitch_query_features
        frame_valid_mask = self._build_frame_valid_mask(
            batch_size=int(waveform.shape[0]),
            num_frames=int(pitch_features.shape[1]),
            valid_audio_frames=valid_audio_frames,
            device=waveform.device,
        )
        if isinstance(self.head, V1SemiCRFHead):
            outputs: dict[str, torch.Tensor | None] = self.head(
                pitch_features,
                frame_valid_mask,
                include_frame_instrument_logits=include_frame_instrument_logits,
            )
        else:
            outputs = self.head(pitch_features, frame_valid_mask)
        outputs.update(
            {
                "band_features": (
                    backbone_output.band_features if include_aux_outputs else None
                ),
                "global_features": (
                    backbone_output.global_features if include_aux_outputs else None
                ),
                "pitch_query_features": (
                    pitch_features if include_aux_outputs else None
                ),
            }
        )
        return outputs

    def supports_interval_boundaries(self) -> bool:
        return self.head.supports_interval_boundaries()

    def supports_interval_instruments(self) -> bool:
        return isinstance(self.head, V1SemiCRFHead)

    def predict_interval_boundaries(
        self,
        features: torch.Tensor,
        interval_batch: Sequence[Sequence[Sequence[tuple[int, int]]]],
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int, int]]]:
        if not isinstance(self.head, V1SemiCRFHead):
            raise RuntimeError("predict_interval_boundaries is a V1-head operation")
        return self.head.predict_interval_boundaries(
            features,
            interval_batch,
            compute_dtype=compute_dtype,
        )

    def predict_interval_instruments(
        self,
        features: torch.Tensor,
        interval_batch: Sequence[Sequence[Sequence[tuple[int, int]]]],
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int, int]]]:
        if not isinstance(self.head, V1SemiCRFHead):
            raise RuntimeError("predict_interval_instruments is a V1-head operation")
        return self.head.predict_interval_instruments(
            features,
            interval_batch,
            compute_dtype=compute_dtype,
        )

    def build_selected_pair_indices(
        self,
        selected_pair_ids: Sequence[Sequence[int] | torch.Tensor],
    ) -> SelectedPairIndices:
        if not isinstance(self.head, V2OverlapSemiCRFHead):
            raise RuntimeError("selected pair indices require the V2 head")
        return self.head.build_selected_pair_indices(
            selected_pair_ids, num_pitches=NUM_PITCHES
        )

    def predict_flat_interval_boundaries(
        self,
        interval_features: torch.Tensor,
        selected_pairs: SelectedPairIndices,
        interval_batch: Sequence[Sequence[tuple[int, int]]],
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        if not isinstance(self.head, V2OverlapSemiCRFHead):
            raise RuntimeError("flat interval boundaries require the V2 head")
        return self.head.predict_flat_interval_boundaries(
            interval_features,
            selected_pairs,
            interval_batch,
            compute_dtype=compute_dtype,
        )

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        return super().load_state_dict(
            remap_legacy_v1_state_dict(state_dict), strict=strict, assign=assign
        )
