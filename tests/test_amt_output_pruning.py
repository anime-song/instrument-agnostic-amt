from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from instrument_agnostic_amt.inference import v1_windowed
from instrument_agnostic_amt.inference.types import InferenceSettings
from instrument_agnostic_amt.inference.v1_windowed import decode_v1_notes
from instrument_agnostic_amt.modeling.model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)


def _small_model() -> AudioSemiCRFTransformer:
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
            num_instrument_classes=2,
        )
    )


def _tensor_outputs_are_equal(
    actual: dict[str, torch.Tensor | None],
    expected: dict[str, torch.Tensor | None],
) -> bool:
    return actual.keys() == expected.keys() and all(
        (left is None and right is None)
        or (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and torch.equal(left, right)
        )
        for left, right in zip(actual.values(), expected.values())
    )


class _PitchOnlyBackbone(torch.nn.Module):
    def forward(
        self,
        waveform: torch.Tensor,
        **_kwargs: object,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            band_features=torch.zeros(1, device=waveform.device),
            global_features=None,
            pitch_query_features=torch.zeros(
                int(waveform.shape[0]),
                4,
                88,
                16,
                device=waveform.device,
            ),
        )


class _RejectingInstrumentClassifier(torch.nn.Module):
    def forward(self, _features: torch.Tensor) -> torch.Tensor:
        raise AssertionError("未使用のframe instrument classifierが呼ばれました")


class _DecodeV1Model:
    def __init__(self, *, use_interval_instrument_head: bool) -> None:
        self._use_interval_instrument_head = use_interval_instrument_head

    @staticmethod
    def supports_interval_boundaries() -> bool:
        return False

    @staticmethod
    def supports_interval_instruments() -> bool:
        return True


class _FrameInstrumentFlagForward:
    def __init__(self) -> None:
        self.include_frame_instrument_logits: list[object] = []

    def __call__(
        self,
        waveform: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | None]:
        self.include_frame_instrument_logits.append(
            kwargs.get("include_frame_instrument_logits")
        )
        batch_size = int(waveform.shape[0])
        frame_count = 20
        return {
            "interval_query": torch.zeros(
                batch_size, frame_count, 88, 1, device=waveform.device
            ),
            "interval_key": torch.zeros(
                batch_size, frame_count, 88, 1, device=waveform.device
            ),
            "interval_diag": torch.zeros(
                batch_size, frame_count, 88, device=waveform.device
            ),
            "frame_valid_mask": torch.ones(
                batch_size,
                frame_count,
                dtype=torch.bool,
                device=waveform.device,
            ),
            "interval_features": None,
            "instrument_features": None,
            "instrument_logits": None,
        }


def test_training_defaults_keep_all_outputs_and_gradients() -> None:
    torch.manual_seed(20260822)
    model = _small_model().train()
    waveform = torch.randn(1, 2, 1_024, requires_grad=True)

    default_outputs = model(waveform)
    explicit_outputs = model(
        waveform,
        include_aux_outputs=True,
        include_frame_instrument_logits=True,
    )
    instrument_logits = default_outputs["instrument_logits"]
    band_features = default_outputs["band_features"]
    assert isinstance(instrument_logits, torch.Tensor)
    assert isinstance(band_features, torch.Tensor)
    (instrument_logits.sum() + band_features.sum()).backward()

    assert _tensor_outputs_are_equal(default_outputs, explicit_outputs)
    assert model.head.instrument_classifier.weight.grad is not None  # type: ignore[union-attr]
    assert model.backbone.stem.conv1.weight.grad is not None


def test_model_can_skip_v1_frame_instrument_logits() -> None:
    model = _small_model().eval()
    model.backbone = _PitchOnlyBackbone()
    model.head.instrument_classifier = _RejectingInstrumentClassifier()  # type: ignore[union-attr]

    outputs = model(
        torch.zeros(1, 2, 256),
        include_aux_outputs=False,
        include_frame_instrument_logits=False,
    )

    assert outputs["instrument_logits"] is None


@pytest.mark.parametrize(
    ("use_interval_instrument_head", "expected_frame_logits"),
    [(True, False), (False, True)],
)
def test_v1_requests_frame_logits_only_for_checkpoint_fallback(
    monkeypatch: pytest.MonkeyPatch,
    use_interval_instrument_head: bool,
    expected_frame_logits: bool,
) -> None:
    monkeypatch.setattr(
        v1_windowed,
        "decode_pitch_intervals",
        lambda interval_query, *_args, **_kwargs: [
            [[] for _ in range(88)]
            for _ in range(int(interval_query.shape[0]))
        ],
    )
    forward = _FrameInstrumentFlagForward()

    decode_v1_notes(
        _DecodeV1Model(  # type: ignore[arg-type]
            use_interval_instrument_head=use_interval_instrument_head
        ),
        SemiCRFModelConfig(
            sample_rate=1_000,
            hop_length=10,
            n_fft=128,
            semi_crf_version="v1",
            num_instrument_classes=2,
        ),
        torch.randn(2, 200),
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=InferenceSettings(
            window_ms=200,
            stride_ms=200,
            track_batch_size=128,
            window_batch_size=1,
            merge_gap_ms=None,
            merge_onset_ms=0.0,
            silence_gate_rms_dbfs=None,
            note_bias=0.0,
            disable_tqdm=True,
            allowed_instrument_ids=(0,),
        ),
        velocity=100,
        forward_model=forward,
    )

    assert forward.include_frame_instrument_logits == [expected_frame_logits]


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
def test_aux_pruning_preserves_amt_outputs_and_skips_contiguous_copies(
    device: str,
) -> None:
    torch.manual_seed(20260822)
    model = _small_model().eval().to(device)
    waveform = torch.randn(1, 2, 1_024, device=device)

    def run(include_aux_outputs: bool) -> tuple[dict[str, torch.Tensor | None], int]:
        with torch.inference_mode(), profile(
            activities=[ProfilerActivity.CPU]
        ) as profiler:
            outputs = model(waveform, include_aux_outputs=include_aux_outputs)
        contiguous_count = sum(
            event.count
            for event in profiler.key_averages()
            if event.key == "aten::contiguous"
        )
        return outputs, contiguous_count

    full, full_contiguous = run(True)
    minimal, minimal_contiguous = run(False)
    major_keys = (
        "interval_query",
        "interval_key",
        "interval_diag",
        "interval_features",
        "instrument_features",
        "instrument_logits",
        "frame_valid_mask",
    )

    assert all(
        torch.equal(full[key], minimal[key])  # type: ignore[arg-type]
        for key in major_keys
    )
    assert full_contiguous - minimal_contiguous == 3
    assert full["band_features"] is not None
    assert full["pitch_query_features"] is not None
    assert minimal["band_features"] is None
    assert minimal["global_features"] is None
    assert minimal["pitch_query_features"] is None
