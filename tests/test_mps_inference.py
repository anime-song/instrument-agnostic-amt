from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from instrument_agnostic_amt.inference import v1_windowed
from instrument_agnostic_amt.inference.types import InferenceSettings
from instrument_agnostic_amt.inference.windowed import decode_notes
from instrument_agnostic_amt.modeling.model import NUM_PITCHES


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
