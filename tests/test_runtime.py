from __future__ import annotations

import sys
from inspect import signature

import pytest
import torch

import instrument_agnostic_amt.runtime as runtime
from instrument_agnostic_amt.cli.infer import parse_args, process_file
from instrument_agnostic_amt.runtime import (
    empty_device_cache,
    is_amp_supported,
    maybe_compile_forward,
    resolve_amp_dtype,
    resolve_device,
)


def _set_available_devices(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool,
    mps: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize(
    ("cuda_available", "mps_available", "expected_type"),
    [
        (True, True, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_resolve_device_auto_prioritizes_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
    mps_available: bool,
    expected_type: str,
) -> None:
    _set_available_devices(
        monkeypatch,
        cuda=cuda_available,
        mps=mps_available,
    )

    assert resolve_device("auto").type == expected_type


@pytest.mark.parametrize("device_name", ["cuda", "mps"])
def test_resolve_device_rejects_an_unavailable_accelerator(
    monkeypatch: pytest.MonkeyPatch,
    device_name: str,
) -> None:
    _set_available_devices(monkeypatch, cuda=False, mps=False)

    with pytest.raises(RuntimeError, match=device_name.upper()):
        resolve_device(device_name)


def test_amp_is_available_for_cuda_and_mps_only() -> None:
    assert is_amp_supported(torch.device("cuda")) is True
    assert is_amp_supported(torch.device("mps")) is True
    assert is_amp_supported(torch.device("cpu")) is False


def test_amp_dtype_defaults_to_float16_when_bfloat16_support_is_unknown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda *, including_emulation=True: False,
    )

    assert resolve_amp_dtype(torch.device("mps"), None) is torch.float16
    assert resolve_amp_dtype(torch.device("cuda"), None) is torch.float16
    assert resolve_amp_dtype(torch.device("mps"), "fp16") is torch.float16


def test_amp_dtype_ignores_emulated_cuda_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_modes: list[bool] = []

    def fake_bfloat16_support(*, including_emulation: bool = True) -> bool:
        checked_modes.append(including_emulation)
        return including_emulation

    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        fake_bfloat16_support,
    )

    assert resolve_amp_dtype(torch.device("cuda"), None) is torch.float16
    assert checked_modes == [False]


def test_amp_dtype_uses_native_cuda_bfloat16_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked_modes: list[bool] = []

    def fake_bfloat16_support(*, including_emulation: bool = True) -> bool:
        checked_modes.append(including_emulation)
        return True

    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        fake_bfloat16_support,
    )

    assert resolve_amp_dtype(torch.device("cuda"), None) is torch.bfloat16
    assert checked_modes == [False]


def test_explicit_bfloat16_does_not_require_native_cuda_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_probe(*, including_emulation: bool = True) -> bool:
        raise AssertionError("Explicit dtype must not probe CUDA capabilities")

    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        unexpected_probe,
    )

    assert resolve_amp_dtype(torch.device("cuda"), "bf16") is torch.bfloat16


def test_core_inference_cli_defaults_to_auto_device_and_device_amp_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["amt-infer", "--audio", "input.wav"])

    args = parse_args()

    assert args.device == "auto"
    assert args.amp_dtype is None
    assert args.compile is False
    assert args.compile_mode == "default"


def test_core_inference_cli_accepts_compile_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "amt-infer",
            "--audio",
            "input.wav",
            "--compile",
            "--compile-mode",
            "max-autotune",
        ],
    )

    args = parse_args()

    assert args.compile is True
    assert args.compile_mode == "max-autotune"


def test_process_file_keeps_forward_model_optional_for_existing_callers() -> None:
    parameter = signature(process_file).parameters["forward_model"]

    assert parameter.default is None


class _RegionalTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.compile_calls: list[dict[str, object]] = []

    def compile(self, **kwargs: object) -> None:
        self.compile_calls.append(kwargs)


class _RegionalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = _RegionalTarget()
        self.backbone = torch.nn.Module()
        self.backbone.layers = torch.nn.ModuleList(
            [torch.nn.ModuleList([_RegionalTarget(), _RegionalTarget()])]
        )
        self.head = _RegionalTarget()


def test_regional_compile_is_default_and_only_compiles_transformer_children() -> None:
    model = _RegionalModel()
    state_keys = tuple(model.state_dict())
    targets = [module for pair in model.backbone.layers for module in pair]

    forward = maybe_compile_forward(model, enabled=True, mode="reduce-overhead")

    assert forward is model
    assert tuple(model.state_dict()) == state_keys
    assert all(
        target.compile_calls
        == [
            {
                "backend": "inductor",
                "mode": "reduce-overhead",
                "fullgraph": False,
                "dynamic": None,
            }
        ]
        for target in targets
    )
    assert model.feature_extractor.compile_calls == []
    assert model.head.compile_calls == []


def test_empty_device_cache_supports_cuda_and_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("cuda"))
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append("mps"))

    empty_device_cache(torch.device("cuda"))
    empty_device_cache(torch.device("mps"))
    empty_device_cache(torch.device("cpu"))

    assert calls == ["cuda", "mps"]


def test_copy_tensors_to_cpu_once_preserves_cpu_tensors() -> None:
    tensors = (
        torch.arange(6, dtype=torch.float32).reshape(2, 3),
        torch.arange(4, dtype=torch.float32).reshape(1, 4),
    )

    copied = runtime.copy_tensors_to_cpu_once(tensors)

    assert len(copied) == len(tensors)
    assert all(torch.equal(actual, expected) for actual, expected in zip(copied, tensors))
    assert [value.dtype for value in copied] == [value.dtype for value in tensors]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS上のdevice-to-host転送回数を検査するテストです",
)
def test_copy_tensors_to_cpu_once_uses_one_mps_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cpu_calls: list[tuple[int, ...]] = []
    original_cpu = torch.Tensor.cpu

    def counted_cpu(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        cpu_calls.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    tensors = (
        torch.arange(6, device="mps", dtype=torch.float32).reshape(2, 3),
        torch.arange(4, device="mps", dtype=torch.float32).reshape(1, 4),
    )

    copied = runtime.copy_tensors_to_cpu_once(tensors)

    assert cpu_calls == [(10,)]
    assert torch.equal(copied[0], torch.arange(6, dtype=torch.float32).reshape(2, 3))
    assert torch.equal(copied[1], torch.arange(4, dtype=torch.float32).reshape(1, 4))
