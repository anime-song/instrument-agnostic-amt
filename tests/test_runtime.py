from __future__ import annotations

import sys
from inspect import signature

import pytest
import torch

import instrument_agnostic_amt.runtime as runtime
from instrument_agnostic_amt.beat_chord.key_only_candidates import parse_arguments
from instrument_agnostic_amt.amt.cli.infer import parse_args, process_file
from instrument_agnostic_amt.instrument_refinement.cli.infer import (
    parse_args as parse_refinement_args,
)
from instrument_agnostic_amt.runtime import (
    empty_device_cache,
    is_amp_supported,
    resolve_amp_dtype,
    resolve_device,
)
from instrument_agnostic_amt.velocity.cli.infer_velocity import (
    parse_args as parse_velocity_args,
)


def _set_available_devices(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cuda: bool,
    mps: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


def test_core_inference_cli_defaults_to_auto_device_and_device_amp_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["amt-infer", "--audio", "input.wav"])

    args = parse_args()

    assert (args.device, args.amp_dtype) == ("auto", None)


def test_core_inference_cli_accepts_explicit_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "amt-infer",
            "--audio",
            "input.wav",
            "--device",
            "mps",
            "--amp",
            "--amp-dtype",
            "fp16",
        ],
    )

    args = parse_args()

    assert (args.device, args.amp, args.amp_dtype) == ("mps", True, "fp16")


def test_core_inference_cli_exposes_opt_in_compile_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["amt-infer", "--audio", "input.wav"])
    defaults = parse_args()
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
    enabled = parse_args()

    assert (
        getattr(defaults, "compile", None),
        getattr(defaults, "compile_mode", None),
        getattr(enabled, "compile", None),
        getattr(enabled, "compile_mode", None),
    ) == (False, "default", True, "max-autotune")


def test_process_file_keeps_forward_model_optional_for_existing_callers() -> None:
    parameter = signature(process_file).parameters.get("forward_model")

    assert parameter is not None and parameter.default is None


def test_secondary_inference_clis_default_to_auto_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["refine-instruments", "--audio", "input.wav", "--midi", "input.mid"],
    )
    refinement_args = parse_refinement_args()
    monkeypatch.setattr(sys, "argv", ["infer-velocity", "--midi", "input.mid"])
    velocity_args = parse_velocity_args()
    batch_args = parse_arguments([])

    assert (
        refinement_args.device,
        velocity_args.device,
        batch_args.device,
    ) == ("auto", "auto", "auto")


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


def test_resolve_device_accepts_available_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_available_devices(monkeypatch, cuda=False, mps=True)

    assert resolve_device("mps") == torch.device("mps")


def test_amp_is_available_for_cuda_and_mps_only() -> None:
    assert (
        is_amp_supported(torch.device("cuda")),
        is_amp_supported(torch.device("mps")),
        is_amp_supported(torch.device("cpu")),
    ) == (True, True, False)


def test_amp_dtype_uses_device_safe_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "is_bf16_supported",
        lambda *, including_emulation=True: False,
    )

    assert (
        resolve_amp_dtype(torch.device("cuda"), None),
        resolve_amp_dtype(torch.device("mps"), None),
        resolve_amp_dtype(torch.device("cpu"), None),
    ) == (torch.float16, torch.float16, torch.float32)


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


def test_explicit_bfloat16_does_not_probe_cuda_support(
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
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(copied, tensors)
    )
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

    def counted_cpu(
        tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
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


class _RegionalCompileTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.compile_calls: list[dict[str, object]] = []

    def compile(self, **kwargs: object) -> None:
        self.compile_calls.append(kwargs)


class _RegionalCompileModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_extractor = _RegionalCompileTarget()
        self.backbone = torch.nn.Module()
        self.backbone.layers = torch.nn.ModuleList(
            [
                torch.nn.ModuleList(
                    [_RegionalCompileTarget(), _RegionalCompileTarget()]
                )
            ]
        )
        self.head = _RegionalCompileTarget()


def test_compile_is_opt_in_and_targets_only_shared_transformer_regions() -> None:
    maybe_compile_forward = getattr(runtime, "maybe_compile_forward", None)
    assert callable(maybe_compile_forward)
    model = _RegionalCompileModel()
    state_keys = tuple(model.state_dict())
    targets = [module for pair in model.backbone.layers for module in pair]

    assert maybe_compile_forward(model, enabled=False) is model
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
                "dynamic": True,
            }
        ]
        for target in targets
    )
    assert model.feature_extractor.compile_calls == []
    assert model.head.compile_calls == []
