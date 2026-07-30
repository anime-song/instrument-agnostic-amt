from __future__ import annotations

import torch

from instrument_agnostic_amt.beat_chord.heads.beat import BeatHead


def test_beat_head_adds_downbeat_logit_to_beat_logit() -> None:
    head = BeatHead(
        input_dim=3,
        num_meter_classes=2,
        hidden_dim=4,
        dropout=0.0,
    ).eval()
    features = torch.tensor(
        [[[0.2, -0.4, 0.8], [1.0, 0.5, -0.5]]],
        dtype=torch.float32,
    )

    with torch.no_grad():
        frame_outputs = head.frame_proj(head.shared(features))
        raw_beat_logits = frame_outputs[..., 0]
        raw_downbeat_logits = frame_outputs[..., 1]
        outputs = head(features)

    assert torch.allclose(
        outputs["beat_logits"],
        raw_beat_logits + raw_downbeat_logits,
    )
    assert torch.allclose(outputs["downbeat_logits"], raw_downbeat_logits)


def test_beat_loss_gradient_reaches_both_sum_head_channels() -> None:
    head = BeatHead(
        input_dim=2,
        num_meter_classes=1,
        hidden_dim=2,
        dropout=0.0,
    )
    outputs = head(torch.zeros(1, 3, 2))

    outputs["beat_logits"].sum().backward()

    bias_gradient = head.frame_proj.bias.grad
    assert bias_gradient is not None
    assert bias_gradient[0] == 3.0
    assert bias_gradient[1] == 3.0
    assert bias_gradient[2] == 0.0


def test_sum_head_uses_float32_under_autocast() -> None:
    head = BeatHead(
        input_dim=2,
        num_meter_classes=1,
        hidden_dim=2,
        dropout=0.0,
    ).eval()

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        outputs = head(torch.zeros(1, 2, 2))

    assert outputs["beat_logits"].dtype == torch.float32
    assert outputs["downbeat_logits"].dtype == torch.bfloat16


def test_legacy_beat_head_state_loads_with_neutral_group_boundary() -> None:
    original = BeatHead(
        input_dim=2,
        num_meter_classes=1,
        hidden_dim=2,
        dropout=0.0,
    )
    legacy_state = {
        key: value
        for key, value in original.state_dict().items()
        if "group_boundary_proj" not in key
    }
    restored = BeatHead(
        input_dim=2,
        num_meter_classes=1,
        hidden_dim=2,
        dropout=0.0,
    )

    restored.load_state_dict(legacy_state, strict=True)
    outputs = restored(torch.zeros(1, 3, 2))

    assert torch.allclose(
        torch.sigmoid(outputs["group_boundary_logits"]),
        torch.full((1, 3), 0.5),
    )
