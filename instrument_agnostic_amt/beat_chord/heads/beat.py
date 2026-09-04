from __future__ import annotations

import contextlib
from typing import Any

import torch
from torch import nn


class BeatHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_meter_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        phase_peak_sharpness: float = 12.0,
        min_phase_velocity: float = 1e-4,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if num_meter_classes <= 0:
            raise ValueError("num_meter_classes must be positive")
        if hidden_dim is None:
            hidden_dim = input_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if phase_peak_sharpness <= 0.0:
            raise ValueError("phase_peak_sharpness must be positive")
        if min_phase_velocity <= 0.0:
            raise ValueError("min_phase_velocity must be positive")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_meter_classes = int(num_meter_classes)
        self.phase_peak_sharpness = float(phase_peak_sharpness)
        self.min_phase_velocity = float(min_phase_velocity)

        self.shared = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.frame_proj = nn.Linear(self.hidden_dim, self.num_meter_classes + 2)
        self.group_boundary_proj = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.group_boundary_proj.weight)
        nn.init.zeros_(self.group_boundary_proj.bias)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        # The boundary projection did not exist in legacy checkpoints. A zero
        # projection is neutral for decoding and keeps strict loading compatible.
        for parameter_name in ("weight", "bias"):
            key = f"{prefix}group_boundary_proj.{parameter_name}"
            if key not in state_dict:
                parameter = getattr(self.group_boundary_proj, parameter_name)
                state_dict[key] = parameter.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.dim() != 3:
            raise ValueError("features must have shape [B, T, D]")
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"features last dim must be {self.input_dim}, got {features.shape[-1]}"
            )

        shared_features = self.shared(features)
        frame_outputs = self.frame_proj(shared_features)
        group_boundary_logits = self.group_boundary_proj(shared_features).squeeze(-1)
        beat_logits, downbeat_logits, meter_logits = torch.split(
            frame_outputs,
            [1, 1, self.num_meter_classes],
            dim=-1,
        )

        beat_logits = beat_logits.squeeze(-1)
        downbeat_logits = downbeat_logits.squeeze(-1)
        if hasattr(
            torch.amp, "is_autocast_available"
        ) and not torch.amp.is_autocast_available(beat_logits.device.type):
            autocast_context = contextlib.nullcontext()
        else:
            autocast_context = torch.autocast(beat_logits.device.type, enabled=False)
        # Keep the sum in float32 because downbeats are also beats and low precision
        # addition can produce NaNs under AMP.
        with autocast_context:
            beat_logits = beat_logits.float() + downbeat_logits.float()

        return {
            "beat_logits": beat_logits,
            "downbeat_logits": downbeat_logits,
            "meter_logits": meter_logits,
            "group_boundary_logits": group_boundary_logits,
        }
