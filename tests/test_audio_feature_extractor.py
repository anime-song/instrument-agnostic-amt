from __future__ import annotations

import pytest
import torch

from instrument_agnostic_amt.amt.modeling.backbone import AudioFeatureExtractor


def _extractor(*, cqt_log_scale: bool = False) -> AudioFeatureExtractor:
    return AudioFeatureExtractor(
        sampling_rate=16_000,
        hop_length=64,
        cqt_fmin=100.0,
        cqt_n_bins=24,
        cqt_bins_per_octave=12,
        cqt_filter_scale=0.5,
        harmonics=(1.0, 1.5, 2.0, 3.0, 5.0),
        cqt_log_scale=cqt_log_scale,
        input_audio_channels=2,
    )


def _legacy_harmonic_interpolation(
    extractor: AudioFeatureExtractor,
    large_spec: torch.Tensor,
) -> torch.Tensor:
    base_bins = torch.arange(
        extractor.cqt_n_bins,
        device=large_spec.device,
        dtype=large_spec.dtype,
    )
    harmonic_specs: list[torch.Tensor] = []
    for index in range(extractor.num_harmonics):
        position = (base_bins + extractor.harmonic_shifts[index]).clamp(
            0,
            int(large_spec.shape[1]) - 1,
        )
        lower = torch.floor(position).long()
        upper = (lower + 1).clamp(max=int(large_spec.shape[1]) - 1)
        alpha = (position - lower).unsqueeze(0).unsqueeze(-1)
        value = large_spec[:, lower, :] + alpha * (
            large_spec[:, upper, :] - large_spec[:, lower, :]
        )
        if extractor.cqt_log_scale:
            value = torch.log(value + 1e-8)
        harmonic_specs.append(value)
    return torch.stack(harmonic_specs, dim=1)


@pytest.mark.parametrize("cqt_log_scale", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="MPS is not available",
            ),
        ),
    ],
)
def test_harmonic_interpolation_is_bit_exact_in_inference(
    cqt_log_scale: bool,
    device: str,
) -> None:
    extractor = _extractor(cqt_log_scale=cqt_log_scale).eval().to(device)
    torch.manual_seed(20260817)
    large_spec = torch.rand(2, extractor.n_bins_large, 17, device=device)

    with torch.inference_mode():
        reference = _legacy_harmonic_interpolation(extractor, large_spec)
        actual = extractor._interpolate_harmonic_specs(large_spec)

    assert torch.equal(actual, reference)


def test_harmonic_interpolation_keeps_training_gradient_order() -> None:
    extractor = _extractor().train()
    torch.manual_seed(20260817)
    reference_input = torch.rand(
        2,
        extractor.n_bins_large,
        17,
        requires_grad=True,
    )
    actual_input = reference_input.detach().clone().requires_grad_(True)
    weight = torch.randn(2, extractor.num_harmonics, extractor.cqt_n_bins, 17)

    reference = _legacy_harmonic_interpolation(extractor, reference_input)
    actual = extractor._interpolate_harmonic_specs(actual_input)
    (reference * weight).sum().backward()
    (actual * weight).sum().backward()

    assert torch.equal(actual, reference)
    assert torch.equal(actual_input.grad, reference_input.grad)


def test_harmonic_interpolation_indices_are_nonpersistent_buffers() -> None:
    extractor = _extractor()

    assert extractor.harmonic_lower_indices.shape == (
        extractor.num_harmonics,
        extractor.cqt_n_bins,
    )
    assert extractor.harmonic_upper_indices.shape == (
        extractor.num_harmonics,
        extractor.cqt_n_bins,
    )
    assert extractor.harmonic_interpolation_alpha.shape == (
        extractor.num_harmonics,
        extractor.cqt_n_bins,
    )
    assert extractor.harmonic_lower_indices.dtype == torch.long
    assert extractor.harmonic_upper_indices.dtype == torch.long
    assert extractor.harmonic_interpolation_alpha.dtype == torch.float32
    assert "harmonic_lower_indices" not in extractor.state_dict()
    assert "harmonic_upper_indices" not in extractor.state_dict()
    assert "harmonic_interpolation_alpha" not in extractor.state_dict()
