from __future__ import annotations

import torch
from torch import nn


class MidiFrameStem(nn.Module):
    """MIDI roll を pitch 方向は保ち、時間方向だけ段階的に圧縮する stem。"""

    def __init__(
        self,
        *,
        in_ch: int,
        base_ch: int,
        pitch_bins: int,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if in_ch <= 0:
            raise ValueError("in_ch must be positive")
        if base_ch <= 0:
            raise ValueError("base_ch must be positive")
        if (base_ch * 2) % 4 != 0:
            raise ValueError("base_ch must be even for GroupNorm channel groups")
        if pitch_bins <= 0:
            raise ValueError("pitch_bins must be positive")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        pad = kernel_size // 2
        self.time_downsample_factor = 2 * 2 * 2
        # Depthwise Conv: 各楽器チャンネルを独立して畳み込み
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=7, padding=3, groups=in_ch)
        self.conv2 = nn.Conv2d(in_ch, in_ch, kernel_size=5, padding=2, groups=in_ch)
        # Pointwise Conv: チャンネル数を圧縮・混合
        self.channel_proj = nn.Conv2d(in_ch, base_ch, kernel_size=1)
        self.pitch_embed = nn.Parameter(torch.zeros(1, base_ch, 1, pitch_bins))
        nn.init.normal_(self.pitch_embed, std=0.02)

        self.block1 = nn.Sequential(
            nn.Conv2d(
                base_ch,
                base_ch * 2,
                kernel_size=(kernel_size, kernel_size),
                stride=(2, 1),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(
                base_ch * 2,
                base_ch * 4,
                kernel_size=(kernel_size, kernel_size),
                stride=(2, 1),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(
                base_ch * 4,
                base_ch * 4,
                kernel_size=(kernel_size, kernel_size),
                stride=(2, 1),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 4),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(
                base_ch * 4,
                base_ch * 4,
                kernel_size=(kernel_size, kernel_size),
                padding=(pad, pad),
            ),
            nn.GroupNorm(4, base_ch * 4),
        )
        self.out_ch = int(base_ch * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("x must have shape [B, C, T, P]")
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.channel_proj(x) + self.pitch_embed[:, :, :, : x.shape[-1]]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x
