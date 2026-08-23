from __future__ import annotations

from collections.abc import Callable

import pytest
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

from instrument_agnostic_amt.inference import v1_windowed
from instrument_agnostic_amt.inference.types import InferenceSettings
from instrument_agnostic_amt.inference.v1_windowed import decode_v1_notes
from instrument_agnostic_amt.modeling.heads.interval_boundaries import (
    FlattenedIntervalEntry,
    gather_interval_endpoint_features,
    gather_interval_sequence_features,
)
from instrument_agnostic_amt.modeling.model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)


IntervalGather = Callable[
    [torch.Tensor, list[list[list[tuple[int, int]]]]],
    tuple[torch.Tensor, list[FlattenedIntervalEntry]],
]


class _FloatCastNumelRecorder(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.numels: list[int] = []

    def __torch_dispatch__(
        self,
        func: object,
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        call_kwargs = {} if kwargs is None else kwargs
        if "_to_copy" in str(func) and args and isinstance(args[0], torch.Tensor):
            source = args[0]
            if (
                source.dtype != torch.float32
                and call_kwargs.get("dtype") == torch.float32
            ):
                self.numels.append(int(source.numel()))
        return func(*args, **call_kwargs)  # type: ignore[operator]


def _small_v1_model(
    *,
    use_interval_boundary_head: bool = True,
) -> AudioSemiCRFTransformer:
    torch.manual_seed(20260822)
    return AudioSemiCRFTransformer(
        SemiCRFModelConfig(
            sample_rate=16_000,
            hop_length=64,
            n_fft=256,
            cqt_fmin=250.0,
            cqt_n_bins=12,
            cqt_bins_per_octave=12,
            cqt_filter_scale=0.5,
            harmonics=(1.0,),
            hidden_size=8,
            base_ch=4,
            encoder_num_layers=0,
            encoder_num_heads=2,
            dropout=0.0,
            use_gradient_checkpoint=False,
            semi_crf_head_dim=8,
            use_interval_boundary_head=use_interval_boundary_head,
            num_instrument_classes=2,
        )
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
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
@pytest.mark.parametrize(
    "gather_features",
    [
        pytest.param(gather_interval_endpoint_features, id="endpoint"),
        pytest.param(gather_interval_sequence_features, id="sequence"),
    ],
)
def test_interval_compute_dtype_matches_full_fp32_output_and_input_gradient(
    dtype: torch.dtype,
    device: str,
    gather_features: IntervalGather,
) -> None:
    torch.manual_seed(20260822)
    source = torch.randn(1, 6, 2, 4, dtype=dtype, device=device)
    intervals = [[[(0, 2), (2, 5)], [(1, 4)]]]
    reference_input = source.detach().clone().requires_grad_(True)
    actual_input = source.detach().clone().requires_grad_(True)

    reference, reference_entries = gather_features(
        reference_input.float(),
        intervals,
    )
    actual, actual_entries = gather_features(
        actual_input,
        intervals,
        compute_dtype=torch.float32,
    )
    output_gradient = torch.randn_like(reference)
    reference.backward(output_gradient)
    actual.backward(output_gradient)

    assert actual_entries == reference_entries
    assert torch.equal(actual, reference)
    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        rtol=2 * torch.finfo(dtype).eps,
        atol=2 * torch.finfo(dtype).eps,
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_v1_public_compute_dtype_matches_default_fp32_training_path(
    dtype: torch.dtype,
) -> None:
    model = _small_v1_model().train()
    intervals = [[[(0, 2), (2, 5)], [(1, 4)]]]
    feature_dim = model.head.interval_instrument_predictor.input_dim
    source = torch.randn(1, 6, 2, feature_dim, dtype=dtype)
    reference_input = source.detach().clone().requires_grad_(True)
    actual_input = source.detach().clone().requires_grad_(True)

    reference_boundary, reference_boundary_entries = (
        model.predict_interval_boundaries(reference_input.float(), intervals)
    )
    reference_instrument, reference_instrument_entries = (
        model.predict_interval_instruments(reference_input.float(), intervals)
    )
    (reference_boundary.sum() + reference_instrument.sum()).backward()
    reference_parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.head.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)

    actual_boundary, actual_boundary_entries = model.predict_interval_boundaries(
        actual_input,
        intervals,
        compute_dtype=torch.float32,
    )
    actual_instrument, actual_instrument_entries = model.predict_interval_instruments(
        actual_input,
        intervals,
        compute_dtype=torch.float32,
    )
    (actual_boundary.sum() + actual_instrument.sum()).backward()
    actual_parameter_grads = {
        name: parameter.grad
        for name, parameter in model.head.named_parameters()
        if parameter.grad is not None
    }

    assert actual_boundary_entries == reference_boundary_entries
    assert actual_instrument_entries == reference_instrument_entries
    assert torch.equal(actual_boundary, reference_boundary)
    assert torch.equal(actual_instrument, reference_instrument)
    torch.testing.assert_close(
        actual_input.grad,
        reference_input.grad,
        rtol=2 * torch.finfo(dtype).eps,
        atol=2 * torch.finfo(dtype).eps,
    )
    assert actual_parameter_grads.keys() == reference_parameter_grads.keys()
    assert all(
        torch.equal(actual_parameter_grads[name], reference_gradient)
        for name, reference_gradient in reference_parameter_grads.items()
    )


@pytest.mark.parametrize(
    ("gather_features", "expected_width"),
    [
        pytest.param(gather_interval_endpoint_features, 12, id="endpoint"),
        pytest.param(gather_interval_sequence_features, 17, id="sequence"),
    ],
)
def test_empty_interval_features_honor_compute_dtype(
    gather_features: IntervalGather,
    expected_width: int,
) -> None:
    features = torch.randn(1, 6, 2, 4, dtype=torch.float16)

    actual, entries = gather_features(
        features,
        [[[], []]],
        compute_dtype=torch.float32,
    )

    assert entries == []
    assert actual.shape == (0, expected_width)
    assert actual.dtype == torch.float32


def test_v1_empty_logits_honor_compute_dtype() -> None:
    model = _small_v1_model()
    model_without_boundary_head = _small_v1_model(
        use_interval_boundary_head=False
    )
    feature_dim = model.head.interval_instrument_predictor.input_dim
    features = torch.randn(1, 6, 2, feature_dim, dtype=torch.float16)
    empty_intervals = [[[], []]]
    nonempty_intervals = [[[(0, 2)], []]]

    boundary_logits, boundary_entries = model.predict_interval_boundaries(
        features,
        empty_intervals,
        compute_dtype=torch.float32,
    )
    disabled_logits, disabled_entries = (
        model_without_boundary_head.predict_interval_boundaries(
            features,
            nonempty_intervals,
            compute_dtype=torch.float32,
        )
    )
    instrument_logits, instrument_entries = model.predict_interval_instruments(
        features,
        empty_intervals,
        compute_dtype=torch.float32,
    )

    assert (boundary_entries, disabled_entries, instrument_entries) == ([], [], [])
    assert (
        boundary_logits.dtype,
        disabled_logits.dtype,
        instrument_logits.dtype,
    ) == (torch.float32, torch.float32, torch.float32)


def test_v1_inference_gathers_before_converting_full_features_to_fp32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_dtypes: list[tuple[str, torch.dtype, torch.dtype | None]] = []
    entries = [(0, 0, 0, 0, 2)]

    class LowPrecisionV1Model:
        _use_interval_instrument_head = True

        @staticmethod
        def supports_interval_boundaries() -> bool:
            return True

        @staticmethod
        def supports_interval_instruments() -> bool:
            return True

        def predict_interval_boundaries(
            self,
            features: torch.Tensor,
            _interval_batch: object,
            *,
            compute_dtype: torch.dtype | None = None,
        ) -> tuple[torch.Tensor, list[FlattenedIntervalEntry]]:
            recorded_dtypes.append(("boundary", features.dtype, compute_dtype))
            return torch.tensor([[1.0, 1.0, 0.0, 0.0]]), entries

        def predict_interval_instruments(
            self,
            features: torch.Tensor,
            _interval_batch: object,
            *,
            compute_dtype: torch.dtype | None = None,
        ) -> tuple[torch.Tensor, list[FlattenedIntervalEntry]]:
            recorded_dtypes.append(("instrument", features.dtype, compute_dtype))
            return torch.tensor([[1.0, 0.0]]), entries

    def low_precision_forward(
        waveform: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor | None]:
        batch_size = int(waveform.shape[0])
        frame_count = 20
        return {
            "interval_query": torch.zeros(batch_size, frame_count, 88, 1),
            "interval_key": torch.zeros(batch_size, frame_count, 88, 1),
            "interval_diag": torch.zeros(batch_size, frame_count, 88),
            "frame_valid_mask": torch.ones(
                batch_size,
                frame_count,
                dtype=torch.bool,
            ),
            "interval_features": torch.zeros(
                batch_size,
                frame_count,
                88,
                4,
                dtype=torch.float16,
            ),
            "instrument_features": torch.zeros(
                batch_size,
                frame_count,
                88,
                4,
                dtype=torch.float16,
            ),
            "instrument_logits": None,
        }

    monkeypatch.setattr(
        v1_windowed,
        "decode_pitch_intervals",
        lambda *_args, **_kwargs: [[[(0, 2)], *([[]] * 87)]],
    )
    config = SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version="v1",
        num_instrument_classes=2,
    )
    settings = InferenceSettings(
        window_ms=200,
        stride_ms=200,
        track_batch_size=128,
        window_batch_size=1,
        merge_gap_ms=None,
        merge_onset_ms=0.0,
        silence_gate_rms_dbfs=None,
        note_bias=0.0,
        disable_tqdm=True,
        use_boundary_head=True,
        allowed_instrument_ids=(0, 1),
    )

    decode_v1_notes(
        LowPrecisionV1Model(),  # type: ignore[arg-type]
        config,
        torch.ones(2, 200),
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=settings,
        velocity=100,
        forward_model=low_precision_forward,
    )

    assert recorded_dtypes == [
        ("boundary", torch.float16, torch.float32),
        ("instrument", torch.float16, torch.float32),
    ]


@pytest.mark.parametrize(
    "gather_features",
    [
        pytest.param(gather_interval_endpoint_features, id="endpoint"),
        pytest.param(gather_interval_sequence_features, id="sequence"),
    ],
)
def test_interval_compute_dtype_does_not_cast_full_frame_features(
    gather_features: IntervalGather,
) -> None:
    features = torch.randn(1, 6, 2, 4, dtype=torch.float16)
    intervals = [[[(0, 2), (2, 5)], [(1, 4)]]]
    recorder = _FloatCastNumelRecorder()

    with recorder:
        gather_features(
            features,
            intervals,
            compute_dtype=torch.float32,
        )

    assert int(features.numel()) not in recorder.numels


def test_interval_sequence_avoids_scalar_read_and_redundant_clone() -> None:
    features = torch.randn(1, 6, 2, 4)
    intervals = [[[(0, 2), (2, 5)], [(1, 4)]]]

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        gather_interval_sequence_features(features, intervals)

    operation_counts = {
        event.key: event.count for event in profiler.key_averages()
    }
    assert operation_counts.get("aten::_local_scalar_dense", 0) == 0
    assert operation_counts.get("aten::clone", 0) == 0
