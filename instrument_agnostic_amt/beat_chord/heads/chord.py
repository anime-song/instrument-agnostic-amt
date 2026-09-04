from __future__ import annotations

import torch
from torch import nn


class ChordHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_root_chord_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 256

        self.shared = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.boundary = nn.Linear(hidden_dim, 1)
        self.root_chord = nn.Linear(hidden_dim, num_root_chord_classes)
        self.bass = nn.Linear(hidden_dim, 13)
        self.key_boundary = nn.Linear(hidden_dim, 1)
        self.key = nn.Linear(hidden_dim, 13)
        self.pitch = nn.Linear(hidden_dim, 25)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            "chord_boundary_logits": self.boundary(h).squeeze(-1),
            "root_chord_logits": self.root_chord(h),
            "bass_logits": self.bass(h),
            "key_boundary_logits": self.key_boundary(h).squeeze(-1),
            "key_logits": self.key(h),
            "chord_pitch_logits": self.pitch(h),
        }
