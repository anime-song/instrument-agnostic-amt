from __future__ import annotations

import pytest
import torch

from instrument_agnostic_amt.modeling.blocks import transformer


def _capture_attention_dtypes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[torch.dtype, torch.dtype, torch.dtype]]:
    captured: list[tuple[torch.dtype, torch.dtype, torch.dtype]] = []

    def fake_scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **_: object,
    ) -> torch.Tensor:
        captured.append((query.dtype, key.dtype, value.dtype))
        return value

    monkeypatch.setattr(
        transformer.F,
        "scaled_dot_product_attention",
        fake_scaled_dot_product_attention,
    )
    return captured


def test_inference_attention_stays_float32_without_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_attention_dtypes(monkeypatch)
    attention = transformer.MultiHeadAttention(
        input_dim=8,
        head_dim=4,
        num_heads=2,
        dropout=0.0,
    ).eval()
    attention.lowp_dtype = torch.float16

    output = attention(torch.randn(1, 5, 8))

    assert captured == [(torch.float32, torch.float32, torch.float32)]
    assert output.dtype is torch.float32


def test_inference_attention_uses_the_requested_autocast_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_attention_dtypes(monkeypatch)
    attention = transformer.MultiHeadAttention(
        input_dim=8,
        head_dim=4,
        num_heads=2,
        dropout=0.0,
    ).eval()
    attention.lowp_dtype = torch.float32

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        attention(torch.randn(1, 5, 8))

    assert captured == [(torch.bfloat16, torch.bfloat16, torch.bfloat16)]


def test_training_attention_preserves_legacy_low_precision_cast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_attention_dtypes(monkeypatch)
    attention = transformer.MultiHeadAttention(
        input_dim=8,
        head_dim=4,
        num_heads=2,
        dropout=0.0,
    ).train()
    attention.lowp_dtype = torch.bfloat16

    attention(torch.randn(1, 5, 8))

    assert captured == [(torch.bfloat16, torch.bfloat16, torch.bfloat16)]
