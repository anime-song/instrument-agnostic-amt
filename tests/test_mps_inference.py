from __future__ import annotations

import copy
import os
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from instrument_agnostic_amt.amt.inference import v1_windowed
from instrument_agnostic_amt.amt.inference.types import InferenceSettings
from instrument_agnostic_amt.amt.inference.windowed import decode_notes
from instrument_agnostic_amt.amt.modeling.model import (
    NUM_PITCHES,
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)
from instrument_agnostic_amt.runtime import maybe_compile_forward
from instrument_agnostic_amt.velocity.modeling.model import (
    VelocityModelConfig,
    VelocityPredictionModel,
)

pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available",
)


class _EmptyV1Model:
    _use_interval_instrument_head = False

    @staticmethod
    def supports_interval_boundaries() -> bool:
        return False

    @staticmethod
    def supports_interval_instruments() -> bool:
        return False

    def __call__(
        self,
        waveform: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(waveform.shape[0])
        return {
            "frame_valid_mask": torch.ones(
                batch_size,
                1,
                dtype=torch.bool,
                device=waveform.device,
            ),
            "interval_query": torch.empty(0, device=waveform.device),
            "interval_key": torch.empty(0, device=waveform.device),
            "interval_diag": torch.empty(0, device=waveform.device),
        }


class _StopV2Model:
    @staticmethod
    def supports_interval_boundaries() -> bool:
        return False

    def __call__(self, *_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
        raise RuntimeError("model called")


def _small_amt_model() -> AudioSemiCRFTransformer:
    config = SemiCRFModelConfig(
        sample_rate=16_000,
        hop_length=64,
        n_fft=1_024,
        semi_crf_version="v1",
        cqt_fmin=250.0,
        cqt_n_bins=12,
        cqt_bins_per_octave=12,
        cqt_filter_scale=0.5,
        harmonics=(1.0,),
        hidden_size=8,
        base_ch=4,
        encoder_num_layers=1,
        encoder_num_heads=2,
        dropout=0.0,
        use_gradient_checkpoint=False,
        semi_crf_head_dim=8,
        num_instrument_classes=2,
        instrument_pair_gate_dim=8,
    )
    return AudioSemiCRFTransformer(config).eval()


def _small_velocity_model(
    device: torch.device | str = "cpu",
) -> VelocityPredictionModel:
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model = VelocityPredictionModel(
            VelocityModelConfig(
                sample_rate=16_000,
                hop_length=64,
                cqt_fmin=250.0,
                cqt_n_bins=12,
                cqt_bins_per_octave=12,
                cqt_filter_scale=0.5,
                harmonics=(1.0,),
                hidden_size=8,
                base_ch=4,
                encoder_num_layers=1,
                encoder_num_heads=2,
                dropout=0.0,
                use_gradient_checkpoint=False,
                pitch_min=21,
                pitch_max=32,
                note_hidden_size=8,
                num_stem_classes=1,
                local_frame_offsets=(0,),
                predict_stem_gain=False,
            )
        ).eval()
    return model.to(device)


def _velocity_inputs(
    note_count: int,
    *,
    device: torch.device | str,
    sample_count: int,
) -> dict[str, torch.Tensor]:
    waveform = torch.linspace(-0.25, 0.25, steps=2 * sample_count).reshape(
        1, 1, 2, sample_count
    )
    note_starts = torch.linspace(0.004, 0.030, steps=note_count).reshape(
        1, note_count
    )
    return {
        "audio": waveform.to(device),
        "note_start_seconds": note_starts.to(device),
        "note_end_seconds": (note_starts + 0.008).to(device),
        "note_pitch": (
            (torch.arange(note_count) % 12 + 21).reshape(1, note_count).to(device)
        ),
        "note_program": torch.zeros(
            1, note_count, dtype=torch.long, device=device
        ),
        "note_is_drum": torch.zeros(
            1, note_count, dtype=torch.bool, device=device
        ),
        "note_stem_index": torch.zeros(
            1, note_count, dtype=torch.long, device=device
        ),
        "stem_class_id": torch.zeros(1, 1, dtype=torch.long, device=device),
        "valid_audio_frames": torch.full(
            (1, 1), sample_count, dtype=torch.long, device=device
        ),
    }


def test_v1_windowed_amp_enables_mps_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autocast_enabled: list[bool] = []

    def fake_autocast(**kwargs: object) -> nullcontext[None]:
        autocast_enabled.append(bool(kwargs["enabled"]))
        return nullcontext()

    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)
    monkeypatch.setattr(
        v1_windowed,
        "decode_pitch_intervals",
        lambda *_args, **_kwargs: [[[] for _ in range(NUM_PITCHES)]],
    )
    config = SimpleNamespace(
        semi_crf_version="v1",
        num_instrument_classes=1,
        sample_rate=16_000,
        hop_length=64,
        num_pitch_slots=1,
        semi_crf_length_scaling=1.0,
        semi_crf_length_penalty=0.0,
    )
    settings = InferenceSettings(
        window_ms=64,
        stride_ms=64,
        track_batch_size=1,
        window_batch_size=1,
        merge_gap_ms=None,
        merge_onset_ms=0.0,
        silence_gate_rms_dbfs=None,
        note_bias=0.0,
        disable_tqdm=True,
    )

    decode_notes(
        _EmptyV1Model(),
        config,
        torch.ones(2, 1_024),
        instrument_filter_id=None,
        device=torch.device("mps"),
        amp_enabled=True,
        amp_dtype=torch.float16,
        settings=settings,
        velocity=100,
    )

    assert autocast_enabled == [True]


def test_v2_windowed_amp_enables_mps_autocast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    autocast_enabled: list[bool] = []

    def fake_autocast(**kwargs: object) -> nullcontext[None]:
        autocast_enabled.append(bool(kwargs["enabled"]))
        return nullcontext()

    monkeypatch.setattr(torch.amp, "autocast", fake_autocast)
    config = SimpleNamespace(
        semi_crf_version="v2",
        input_audio_channels=2,
        sample_rate=16_000,
        hop_length=64,
        n_fft=1_024,
    )
    settings = InferenceSettings(
        window_ms=64,
        stride_ms=64,
        track_batch_size=1,
        window_batch_size=1,
        merge_gap_ms=None,
        merge_onset_ms=0.0,
        silence_gate_rms_dbfs=None,
        note_bias=0.0,
        disable_tqdm=True,
    )

    with pytest.raises(RuntimeError, match="model called"):
        decode_notes(
            _StopV2Model(),
            config,
            torch.ones(2, 1_024),
            instrument_filter_id=None,
            device=torch.device("mps"),
            amp_enabled=True,
            amp_dtype=torch.float16,
            settings=settings,
            velocity=100,
        )

    assert autocast_enabled == [True]


@pytest.mark.skipif(
    os.environ.get("RUN_ACCELERATOR_COMPILE_TEST") != "1",
    reason="accelerator compile regression is opt-in",
)
def test_core_amt_compiled_forward_runs_on_mps() -> None:
    torch.manual_seed(23)
    model = _small_amt_model().to("mps")
    eager_model = copy.deepcopy(model)
    compiled_forward = maybe_compile_forward(model, enabled=True)
    waveform = torch.randn(1, 2, 4_096, device="mps")
    valid_frames = torch.tensor([4_096], device="mps")

    eager_outputs = eager_model(
        waveform,
        valid_audio_frames=valid_frames,
        include_aux_outputs=False,
    )
    compiled_outputs = compiled_forward(
        waveform,
        valid_audio_frames=valid_frames,
        include_aux_outputs=False,
    )

    tensors = [
        value for value in compiled_outputs.values() if isinstance(value, torch.Tensor)
    ]
    assert tensors
    assert all(value.device.type == "mps" for value in tensors)
    assert all(torch.isfinite(value).all() for value in tensors)
    for name, eager_value in eager_outputs.items():
        compiled_value = compiled_outputs[name]
        if not isinstance(eager_value, torch.Tensor):
            assert compiled_value is None
            continue
        assert isinstance(compiled_value, torch.Tensor)
        if eager_value.is_floating_point() or eager_value.is_complex():
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
def test_velocity_compiled_forward_handles_varying_shapes_on_mps() -> None:
    device = torch.device("mps")
    model = _small_velocity_model(device)
    eager_model = copy.deepcopy(model)
    compiled_forward = maybe_compile_forward(model, enabled=True)

    with torch.inference_mode():
        for note_count, sample_count in ((1, 1_024), (2, 768), (3, 640)):
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
                assert compiled_value.device.type == "mps"
                assert torch.isfinite(compiled_value).all()
                torch.testing.assert_close(
                    compiled_value,
                    eager_value,
                    rtol=1e-4,
                    atol=1e-5,
                )
