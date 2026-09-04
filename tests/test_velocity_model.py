from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from instrument_agnostic_amt.amt.modeling.backbone import V1Backbone
from instrument_agnostic_amt.velocity.modeling.model import (
    VelocityModelConfig,
    VelocityPredictionModel,
)
from recipes.velocity.forward import forward_velocity_batch
from recipes.velocity.losses import (
    VelocityLossConfig,
    compute_velocity_losses,
)


class _FakePitchBackbone(nn.Module):
    query_feature_dim = 12

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv1d(2, 88 * self.query_feature_dim, 5, stride=10, padding=2)

    def forward(self, audio: torch.Tensor) -> SimpleNamespace:
        values = self.projection(audio).transpose(1, 2)
        values = values.reshape(audio.shape[0], values.shape[1], 88, self.query_feature_dim)
        return SimpleNamespace(pitch_query_features=values)


class _AuxAwareFakePitchBackbone(V1Backbone):
    query_feature_dim = 12

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.projection = nn.Conv1d(
            2,
            88 * self.query_feature_dim,
            5,
            stride=10,
            padding=2,
        )
        self.include_aux_outputs: list[bool] = []

    def forward(
        self,
        audio: torch.Tensor,
        *,
        include_aux_outputs: bool = True,
    ) -> SimpleNamespace:
        self.include_aux_outputs.append(include_aux_outputs)
        values = self.projection(audio).transpose(1, 2)
        values = values.reshape(
            audio.shape[0],
            values.shape[1],
            88,
            self.query_feature_dim,
        )
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

    loss, metrics = compute_velocity_losses(
        outputs,
        batch,
        config=VelocityLossConfig(stem_gain_weight=0.1),
    )
    assert torch.isfinite(loss)
    assert int(metrics["note_count"]) == 5
    assert int(metrics["stem_gain_count"]) == 4
    loss.backward()
    assert model.velocity_head.weight.grad is not None
    assert model.stem_gain_head[-1].weight.grad is not None


def test_velocity_only_model_retains_absolute_audio_level() -> None:
    config = VelocityModelConfig(
        use_absolute_velocity_energy=True,
        predict_stem_gain=False,
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
    outputs = forward_velocity_batch(model, batch)

    assert "stem_gain_db" not in outputs
    assert model.stem_gain_head is None

    frame_mask = torch.ones(1, 2, 10, dtype=torch.bool)
    loud = torch.ones(1, 2, 100)
    quiet = loud * 0.1
    audio = torch.stack((loud, quiet), dim=1)
    energy = model._relative_energy(
        audio,
        num_frames=10,
        frame_valid_mask=frame_mask,
    )
    assert energy[:, 0].mean() > energy[:, 1].mean()

    loss, metrics = compute_velocity_losses(outputs, batch)
    assert int(metrics["stem_gain_count"]) == 0
    loss.backward()
    assert model.velocity_head.weight.grad is not None


def test_velocity_model_can_skip_unused_stem_gain_head() -> None:
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
        predict_stem_gain=True,
    )
    model = VelocityPredictionModel(config, backbone=_FakePitchBackbone()).eval()
    batch = _batch()

    with torch.inference_mode():
        reference = forward_velocity_batch(model, batch)
        velocity_only = model(
            batch["audio"],
            valid_audio_frames=batch["valid_audio_frames"],
            note_start_seconds=batch["note_start_seconds"],
            note_end_seconds=batch["note_end_seconds"],
            note_pitch=batch["note_pitch"],
            note_program=batch["note_program"],
            note_is_drum=batch["note_is_drum"],
            note_stem_index=batch["note_stem_index"],
            stem_class_id=batch["stem_class_id"],
            note_mask=batch["note_mask"],
            stem_mask=batch["stem_mask"],
            include_stem_gain=False,
        )

    assert "stem_gain_db" in reference
    assert "stem_gain_db" not in velocity_only
    assert torch.equal(
        velocity_only["velocity_expected"],
        reference["velocity_expected"],
    )


def test_velocity_model_skips_backbone_auxiliary_outputs() -> None:
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
        predict_stem_gain=False,
    )
    backbone = _AuxAwareFakePitchBackbone()
    model = VelocityPredictionModel(config, backbone=backbone).eval()

    with torch.inference_mode():
        forward_velocity_batch(model, _batch())

    assert backbone.include_aux_outputs == [False]


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
