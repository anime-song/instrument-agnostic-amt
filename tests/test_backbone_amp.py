from __future__ import annotations

import torch

from instrument_agnostic_amt.amt.modeling.backbone import StemConv


def _small_stem() -> StemConv:
    return StemConv(in_ch=2, base_ch=4, n_bins=16)


def test_stem_conv_uses_float32_during_float16_inference_autocast() -> None:
    stem = _small_stem().eval()
    inputs = torch.randn(1, 2, 16, 16)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        output = stem(inputs)

    assert output.dtype == torch.float32


def test_stem_conv_float16_inference_autocast_prevents_overflow() -> None:
    torch.manual_seed(0)
    stem = _small_stem().eval()
    with torch.no_grad():
        for module in stem.modules():
            if isinstance(module, torch.nn.Conv2d):
                module.weight.mul_(50.0)
    inputs = torch.randn(1, 2, 16, 16) * 10.0

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.float16):
        output = stem(inputs)

    assert torch.isfinite(output).all()


def test_stem_conv_keeps_bfloat16_inference_autocast() -> None:
    stem = _small_stem().eval()
    inputs = torch.randn(1, 2, 16, 16)

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = stem(inputs)

    assert output.dtype == torch.bfloat16


def test_stem_conv_keeps_float16_training_autocast() -> None:
    stem = _small_stem().train()
    inputs = torch.randn(1, 2, 16, 16)

    with torch.autocast("cpu", dtype=torch.float16):
        output = stem(inputs)

    assert output.dtype == torch.float16


def test_stem_conv_keeps_float32_without_autocast() -> None:
    stem = _small_stem().eval()

    with torch.no_grad():
        output = stem(torch.randn(1, 2, 16, 16))

    assert output.dtype == torch.float32
