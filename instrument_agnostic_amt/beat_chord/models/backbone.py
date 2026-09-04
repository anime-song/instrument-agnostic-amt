from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import einops
import torch
import torch.nn.functional as F
from torch import nn

from instrument_agnostic_amt.amt.modeling.blocks.transformer import RMSNorm, Transformer

from ..config import MidiFrameModelConfig
from .stem import MidiFrameStem


@dataclass
class MidiFrameBackboneOutput:
    lowres_global_features: torch.Tensor
    global_features: torch.Tensor
    lowres_global_tokens: torch.Tensor
    global_token_features: torch.Tensor


def checkpoint_module(
    module: nn.Module,
    x: torch.Tensor,
    *,
    use_checkpoint: bool,
) -> torch.Tensor:
    if not use_checkpoint:
        return module(x)

    def forward_fn(tensor: torch.Tensor) -> torch.Tensor:
        return module(tensor)

    return torch.utils.checkpoint.checkpoint(
        forward_fn,
        x,
        use_reentrant=False,
    )


class MidiFrameBackbone(nn.Module):
    def __init__(self, config: MidiFrameModelConfig) -> None:
        super().__init__()
        self.config = config
        self.use_gradient_checkpoint = bool(config.use_gradient_checkpoint)
        self.time_downsample_factor = int(config.time_downsample_factor)

        self.stem = MidiFrameStem(
            in_ch=config.num_input_channels,
            base_ch=config.base_ch,
            pitch_bins=config.num_pitch_bins,
            kernel_size=config.stem_kernel_size,
            dropout=config.dropout,
        )
        # stem の時間圧縮率と復元率がずれると、出力フレームの時刻対応が崩れる。
        if self.time_downsample_factor != self.stem.time_downsample_factor:
            raise ValueError(
                "config.time_downsample_factor must match MidiFrameStem "
                f"downsample factor ({self.stem.time_downsample_factor}), "
                f"got {self.time_downsample_factor}"
            )
        self.input_proj = nn.Linear(self.stem.out_ch, config.hidden_size)
        self.pitch_pos_embed = nn.Parameter(
            torch.zeros(1, 1, config.num_pitch_bins, config.hidden_size)
        )
        self.pitch_type_embed = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))

        nn.init.normal_(self.pitch_pos_embed, std=0.02)
        nn.init.normal_(self.pitch_type_embed, std=0.02)

        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Transformer(
                            input_dim=config.hidden_size,
                            head_dim=config.hidden_size // config.num_heads,
                            num_layers=1,
                            num_heads=config.num_heads,
                            ffn_hidden_size_factor=config.ffn_multiplier,
                            dropout=config.dropout,
                        ),
                        Transformer(
                            input_dim=config.hidden_size,
                            head_dim=config.hidden_size // config.num_heads,
                            num_layers=1,
                            num_heads=config.num_heads,
                            ffn_hidden_size_factor=config.ffn_multiplier,
                            dropout=config.dropout,
                        ),
                    ]
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = RMSNorm(config.hidden_size)

        # グローバルトークンと対応する埋め込みの定義
        self.num_global_tokens = config.num_global_tokens
        self.global_tokens = nn.Parameter(
            torch.zeros(1, 1, config.num_global_tokens, config.hidden_size)
        )
        self.global_type_embed = nn.Parameter(torch.zeros(1, 1, 1, config.hidden_size))
        nn.init.normal_(self.global_tokens, std=0.02)
        nn.init.normal_(self.global_type_embed, std=0.02)

        # 時間方向のアップサンプリングを行う転置畳み込み
        self.global_up_conv = nn.ConvTranspose1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=self.time_downsample_factor,
            stride=self.time_downsample_factor,
        )

        # 4つのグローバルトークン特徴量を1つにマージするための Linear
        self.global_token_merge = nn.Linear(
            config.hidden_size * config.num_global_tokens, config.hidden_size
        )

    @staticmethod
    def _match_time_length(x: torch.Tensor, target_frames: int) -> torch.Tensor:
        if x.shape[1] < target_frames:
            return F.pad(x, (0, 0, 0, target_frames - x.shape[1]))
        return x[:, :target_frames, :]

    def forward(
        self,
        midi_frames: torch.Tensor,
        *,
        intermediate_callback: Callable[[torch.Tensor, int], torch.Tensor]
        | None = None,
    ) -> MidiFrameBackboneOutput:
        if midi_frames.dim() != 4:
            raise ValueError("midi_frames must have shape [B, C, T, P]")

        use_checkpoint = (
            self.use_gradient_checkpoint and self.training and torch.is_grad_enabled()
        )
        target_frames = int(midi_frames.shape[2])
        stem_features = self.stem(midi_frames)
        x = einops.rearrange(stem_features, "b d t p -> b t p d")
        x = self.input_proj(x)
        x = x + self.pitch_pos_embed[:, :, : x.shape[2], :] + self.pitch_type_embed

        batch_size, time_steps, pitch_steps, hidden_size = x.shape

        # グローバルトークンをピッチ方向に結合
        global_tokens = self.global_tokens.expand(batch_size, time_steps, -1, -1)
        global_tokens = global_tokens + self.global_type_embed
        x = torch.cat([x, global_tokens], dim=2)  # [B, T, P + G, D]

        for layer_idx, (pitch_transformer, time_transformer) in enumerate(self.layers):
            current_token_count = x.shape[2]
            x = x.reshape(batch_size * time_steps, current_token_count, hidden_size)
            x = checkpoint_module(
                pitch_transformer,
                x,
                use_checkpoint=use_checkpoint,
            )
            x = x.reshape(batch_size, time_steps, current_token_count, hidden_size)

            x = einops.rearrange(x, "b t k d -> (b k) t d")
            x = checkpoint_module(
                time_transformer,
                x,
                use_checkpoint=use_checkpoint,
            )
            x = einops.rearrange(
                x,
                "(b k) t d -> b t k d",
                b=batch_size,
                k=current_token_count,
            )

            if intermediate_callback is not None:
                x = intermediate_callback(x, layer_idx)

        x = self.final_norm(x)  # [B, T_low, P + G + R, D]

        # ピッチ特徴とグローバル特徴を分離 (追加された予測トークンは無視)
        pitch_part = x[:, :, :pitch_steps, :]  # [B, T_low, P, D]
        global_part = x[
            :, :, pitch_steps : pitch_steps + self.num_global_tokens, :
        ]  # [B, T_low, G, D]

        # 2. グローバルトークン（global_part）の時間方向アップサンプリング
        upsampled_global = einops.rearrange(global_part, "b t g d -> (b g) d t")
        upsampled_global = self.global_up_conv(upsampled_global)
        upsampled_global = einops.rearrange(
            upsampled_global,
            "(b g) d t -> b t g d",
            b=batch_size,
            g=self.num_global_tokens,
        )
        highres_global_tokens = self._match_time_length(
            upsampled_global, target_frames
        )  # [B, T_high, G, D]

        # 3. 高解像度グローバルトークンを1つにマージ
        highres_global_flat = einops.rearrange(
            highres_global_tokens, "b t g d -> b t (g d)"
        )
        global_features = self.global_token_merge(highres_global_flat)  # [B, T_high, D]

        # 4. 低解像度グローバルトークンを1つにマージ
        lowres_global_flat = einops.rearrange(global_part, "b t g d -> b t (g d)")
        lowres_global_features = self.global_token_merge(
            lowres_global_flat
        )  # [B, T_low, D]

        return MidiFrameBackboneOutput(
            lowres_global_features=lowres_global_features.contiguous(),
            global_features=global_features.contiguous(),
            lowres_global_tokens=global_part.contiguous(),
            global_token_features=highres_global_tokens.contiguous(),
        )
