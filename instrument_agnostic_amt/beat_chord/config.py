from __future__ import annotations

from dataclasses import dataclass

from instrument_agnostic_amt.taxonomy.instrument_classes import NUM_INSTRUMENT_CLASSES


@dataclass(frozen=True)
class MidiFrameModelConfig:
    sample_rate: int
    hop_length: int
    pitch_min: int = 21
    pitch_max: int = 108
    num_input_channels: int = NUM_INSTRUMENT_CLASSES
    base_ch: int = 32
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    num_global_tokens: int = 4
    dropout: float = 0.1
    stem_kernel_size: int = 3
    ffn_multiplier: int = 4
    time_downsample_factor: int = 8
    use_gradient_checkpoint: bool = False
    num_meter_classes: int = 1
    num_root_chord_classes: int = 745
    beat_head_hidden_dim: int | None = None
    chord_head_hidden_dim: int | None = None
    inter_refine_layers: tuple[int, ...] = ()
    inter_loss_weight: float = 0.5
    time_mask_prob: float = 0.0
    time_mask_duration_ms: float = 1000.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.time_mask_prob < 0.0 or self.time_mask_prob > 1.0:
            raise ValueError("time_mask_prob must be between 0.0 and 1.0")
        if self.time_mask_duration_ms <= 0.0:
            raise ValueError("time_mask_duration_ms must be positive")
        if self.inter_loss_weight < 0.0:
            raise ValueError("inter_loss_weight must be non-negative")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.pitch_max < self.pitch_min:
            raise ValueError("pitch_max must be >= pitch_min")
        if self.num_input_channels <= 0:
            raise ValueError("num_input_channels must be positive")
        if self.base_ch <= 0:
            raise ValueError("base_ch must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_global_tokens <= 0:
            raise ValueError("num_global_tokens must be positive")
        if self.dropout < 0.0:
            raise ValueError("dropout must be non-negative")
        if self.stem_kernel_size <= 0 or self.stem_kernel_size % 2 == 0:
            raise ValueError("stem_kernel_size must be a positive odd integer")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if self.time_downsample_factor <= 0:
            raise ValueError("time_downsample_factor must be positive")
        if self.num_meter_classes <= 0:
            raise ValueError("num_meter_classes must be positive")
        if self.num_root_chord_classes <= 0:
            raise ValueError("num_root_chord_classes must be positive")

    @property
    def num_pitch_bins(self) -> int:
        return int(self.pitch_max - self.pitch_min + 1)

    @property
    def stem_output_dim(self) -> int:
        return int(self.base_ch * 4)
