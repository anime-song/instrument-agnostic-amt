from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from instrument_agnostic_amt.velocity.modeling.model import (
    VelocityModelConfig,
    VelocityPredictionModel,
)
from instrument_agnostic_amt.velocity.training.forward import forward_velocity_batch
from instrument_agnostic_amt.velocity.training.losses import compute_velocity_losses


class _FakePitchBackbone(nn.Module):
    query_feature_dim = 12

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv1d(2, 88 * self.query_feature_dim, 5, stride=10, padding=2)

    def forward(self, audio: torch.Tensor) -> SimpleNamespace:
        values = self.projection(audio).transpose(1, 2)
        values = values.reshape(audio.shape[0], values.shape[1], 88, self.query_feature_dim)
        return SimpleNamespace(pitch_query_features=values)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "audio": torch.randn(2, 2, 2, 100),
        "valid_audio_frames": torch.tensor([[100, 100], [80, 80]]),
        "note_start_seconds": torch.tensor([[0.1, 0.5, 0.0], [0.2, 0.6, 0.7]]),
        "note_end_seconds": torch.tensor([[0.4, 0.8, 0.0], [0.5, 0.9, 1.0]]),
        "note_pitch": torch.tensor([[40, 60, 0], [36, 72, 84]]),
        "note_program": torch.tensor([[32, 0, 0], [0, 40, 48]]),
        "note_is_drum": torch.tensor([[False, False, False], [True, False, False]]),
        "note_stem_index": torch.tensor([[0, 1, 0], [0, 1, 1]]),
        "stem_class_id": torch.tensor([[0, 4], [1, 3]]),
        "note_mask": torch.tensor([[True, True, False], [True, True, True]]),
        "stem_mask": torch.ones(2, 2, dtype=torch.bool),
        "stem_gain_mask": torch.ones(2, 2, dtype=torch.bool),
        "target_velocity": torch.tensor([[32, 80, 0], [48, 96, 112]]),
        "stem_gain_db": torch.tensor([[-3.0, 3.0], [2.0, -2.0]]),
    }


def test_velocity_model_forward_loss_and_backward() -> None:
    config = VelocityModelConfig(
        sample_rate=100,
        hop_length=10,
        hidden_size=24,
        base_ch=8,
        encoder_num_layers=0,
        encoder_num_heads=3,
        note_hidden_size=32,
        dropout=0.0,
        use_gradient_checkpoint=False,
    )
    model = VelocityPredictionModel(config, backbone=_FakePitchBackbone())
    batch = _batch()

    outputs = forward_velocity_batch(model, batch, include_aux_outputs=True)

    assert outputs["velocity_logits"].shape == (2, 3, 127)
    assert outputs["velocity_expected"].shape == (2, 3)
    assert outputs["stem_gain_db"].shape == (2, 2)
    assert torch.allclose(outputs["stem_gain_db"].sum(dim=1), torch.zeros(2), atol=1e-6)
    assert outputs["local_attention"][0, 2].sum() == 0

    loss, metrics = compute_velocity_losses(outputs, batch)
    assert torch.isfinite(loss)
    assert int(metrics["note_count"]) == 5
    assert int(metrics["stem_gain_count"]) == 4
    loss.backward()
    assert model.velocity_head.weight.grad is not None
    assert model.stem_gain_head[-1].weight.grad is not None


def test_velocity_loss_handles_empty_note_targets() -> None:
    batch = _batch()
    batch["note_mask"].zero_()
    batch["stem_gain_mask"].zero_()
    outputs = {
        "velocity_logits": torch.randn(2, 3, 127, requires_grad=True),
        "velocity_expected": torch.randn(2, 3),
        "stem_gain_db": torch.randn(2, 2),
    }

    loss, metrics = compute_velocity_losses(outputs, batch)

    assert loss.item() == 0.0
    assert metrics["note_count"].item() == 0
    loss.backward()
