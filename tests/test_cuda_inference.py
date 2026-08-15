from __future__ import annotations

import copy
import os

import pytest
import torch

from instrument_agnostic_amt.runtime import maybe_compile_forward, resolve_device
from tests.test_mps_inference import (
    _small_amt_model,
    _small_velocity_model,
    _velocity_inputs,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available",
)


def _run_core_forward(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    valid_frames: torch.Tensor,
    *,
    amp: bool,
) -> dict[str, torch.Tensor | None]:
    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=amp,
    ):
        return model(
            waveform,
            valid_audio_frames=valid_frames,
            include_aux_outputs=False,
        )


def test_auto_device_selects_cuda_on_cuda_host() -> None:
    assert resolve_device("auto").type == "cuda"


@pytest.mark.parametrize("amp", [False, True])
def test_core_amt_forward_runs_on_cuda(amp: bool) -> None:
    model = _small_amt_model().to("cuda")
    waveform = torch.randn(1, 2, 4096, device="cuda")
    valid_frames = torch.tensor([4096], device="cuda")

    outputs = _run_core_forward(
        model,
        waveform,
        valid_frames,
        amp=amp,
    )

    tensors = [value for value in outputs.values() if isinstance(value, torch.Tensor)]
    assert tensors
    assert all(value.device.type == "cuda" for value in tensors)
    assert all(torch.isfinite(value).all() for value in tensors)


def test_core_amt_cuda_matches_cpu_in_fp32() -> None:
    torch.manual_seed(11)
    cpu_model = _small_amt_model()
    cuda_model = copy.deepcopy(cpu_model).to("cuda")
    waveform = torch.randn(1, 2, 4096)
    valid_frames = torch.tensor([4096])

    cpu_outputs = cpu_model(
        waveform,
        valid_audio_frames=valid_frames,
        include_aux_outputs=False,
    )
    cuda_outputs = _run_core_forward(
        cuda_model,
        waveform.to("cuda"),
        valid_frames.to("cuda"),
        amp=False,
    )

    for name, cpu_value in cpu_outputs.items():
        cuda_value = cuda_outputs[name]
        if not isinstance(cpu_value, torch.Tensor):
            assert cuda_value is None
            continue
        assert isinstance(cuda_value, torch.Tensor)
        torch.testing.assert_close(
            cuda_value.cpu(),
            cpu_value,
            rtol=5e-3,
            atol=5e-4,
        )


@pytest.mark.skipif(
    os.environ.get("RUN_ACCELERATOR_COMPILE_TEST") != "1",
    reason="accelerator compile regression is opt-in",
)
def test_core_amt_compiled_forward_runs_on_cuda() -> None:
    torch.manual_seed(29)
    model = _small_amt_model().to("cuda")
    eager_model = copy.deepcopy(model)
    compiled_forward = maybe_compile_forward(model, enabled=True)
    waveform = torch.randn(1, 2, 4096).to("cuda")
    valid_frames = torch.tensor([4096], device="cuda")

    for amp in (False, True):
        eager_outputs = _run_core_forward(
            eager_model,
            waveform,
            valid_frames,
            amp=amp,
        )
        outputs = _run_core_forward(
            compiled_forward,
            waveform,
            valid_frames,
            amp=amp,
        )
        tensors = [
            value for value in outputs.values() if isinstance(value, torch.Tensor)
        ]
        assert tensors
        assert all(value.device.type == "cuda" for value in tensors)
        assert all(torch.isfinite(value).all() for value in tensors)
        for name, eager_value in eager_outputs.items():
            compiled_value = outputs[name]
            if not isinstance(eager_value, torch.Tensor):
                assert compiled_value is None
                continue
            assert isinstance(compiled_value, torch.Tensor)
            if eager_value.dtype == torch.float16:
                rtol, atol = 2e-2, 2e-3
            elif eager_value.is_floating_point() or eager_value.is_complex():
                rtol, atol = 1e-4, 1e-5
            else:
                rtol, atol = 0.0, 0.0
            torch.testing.assert_close(
                compiled_value,
                eager_value,
                rtol=rtol,
                atol=atol,
            )


@pytest.mark.skipif(
    os.environ.get("RUN_ACCELERATOR_COMPILE_TEST") != "1",
    reason="accelerator compile regression is opt-in",
)
def test_velocity_compiled_forward_handles_varying_note_counts_on_cuda() -> None:
    device = torch.device("cuda")
    model = _small_velocity_model(
        device,
        predict_stem_gain=False,
        encoder_num_layers=1,
    )
    eager_model = copy.deepcopy(model)
    compiled_forward = maybe_compile_forward(model, enabled=True)

    with torch.inference_mode():
        for note_count, sample_count in ((1, 1024), (2, 768), (3, 640)):
            inputs = _velocity_inputs(
                note_count,
                device=device,
                sample_count=sample_count,
            )
            eager_outputs = eager_model(**inputs)
            compiled_outputs = compiled_forward(**inputs)

            assert compiled_outputs.keys() == eager_outputs.keys()
            for name, eager_value in eager_outputs.items():
                compiled_value = compiled_outputs[name]
                assert compiled_value.device.type == "cuda"
                assert torch.isfinite(compiled_value).all()
                torch.testing.assert_close(
                    compiled_value,
                    eager_value,
                    rtol=1e-4,
                    atol=1e-5,
                )
