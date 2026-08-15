from __future__ import annotations

import sys
from inspect import signature

import pytest
import torch

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
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    assert resolve_amp_dtype(torch.device("mps"), None) is torch.float16
    assert resolve_amp_dtype(torch.device("cuda"), None) is torch.float16
    assert resolve_amp_dtype(torch.device("mps"), "fp16") is torch.float16


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


def test_compile_forward_is_opt_in_and_preserves_the_eager_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eager_model = torch.nn.Linear(2, 2)
    compiled_forward = object()
    compile_calls: list[tuple[object, dict[str, object]]] = []

    def fake_compile(model: object, **kwargs: object) -> object:
        compile_calls.append((model, kwargs))
        return compiled_forward

    monkeypatch.setattr(torch, "compile", fake_compile)

    assert maybe_compile_forward(eager_model, enabled=False) is eager_model
    assert compile_calls == []
    assert (
        maybe_compile_forward(eager_model, enabled=True, mode="max-autotune")
        is compiled_forward
    )
    assert compile_calls == [
        (
            eager_model,
            {
                "backend": "inductor",
                "mode": "max-autotune",
                "fullgraph": False,
            },
        )
    ]


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
