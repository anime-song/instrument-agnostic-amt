from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from instrument_agnostic_amt.inference import v1_windowed
from instrument_agnostic_amt.inference.types import InferenceSettings, PredictedNote
from instrument_agnostic_amt.inference.windowed import decode_notes
from instrument_agnostic_amt.modeling.model import SemiCRFModelConfig


class _WindowStateV1Model:
    _use_interval_instrument_head = False

    def __init__(self, *, first_end_frame: int = 15) -> None:
        self.batch_sizes: list[int] = []
        self.first_end_frame = int(first_end_frame)

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
    ) -> dict[str, torch.Tensor | None]:
        batch_size = int(waveform.shape[0])
        self.batch_sizes.append(batch_size)
        frame_count = 20
        interval_query = torch.zeros(
            batch_size, frame_count, 88, 1, device=waveform.device
        )
        interval_key = torch.zeros_like(interval_query)
        interval_diag = torch.full(
            (batch_size, frame_count, 88), -100.0, device=waveform.device
        )
        # 第1窓は指定frameで閉じ、第2窓はstate未反映時だけ[0, 18]を復号する。
        end_frames = torch.where(
            waveform[:, 0, 0] > 0.5,
            torch.full((batch_size,), 18, device=waveform.device, dtype=torch.long),
            torch.full(
                (batch_size,),
                self.first_end_frame,
                device=waveform.device,
                dtype=torch.long,
            ),
        )
        interval_query[:, 0, 0, 0] = 1.0
        interval_key[
            torch.arange(batch_size, device=waveform.device),
            end_frames,
            0,
            0,
        ] = 10.0
        return {
            "interval_query": interval_query,
            "interval_key": interval_key,
            "interval_diag": interval_diag,
            "frame_valid_mask": torch.ones(
                batch_size,
                frame_count,
                dtype=torch.bool,
                device=waveform.device,
            ),
            "interval_features": torch.zeros(
                batch_size,
                frame_count,
                88,
                1,
                device=waveform.device,
            ),
            "instrument_features": None,
            "instrument_logits": None,
        }


class _WindowStateBoundaryV1Model(_WindowStateV1Model):
    @staticmethod
    def supports_interval_boundaries() -> bool:
        return True

    @staticmethod
    def predict_interval_boundaries(
        features: torch.Tensor,
        interval_batch: list[list[list[tuple[int, int]]]],
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int, int]]]:
        entries: list[tuple[int, int, int, int, int]] = []
        for batch_index, sample in enumerate(interval_batch):
            for track, intervals in enumerate(sample):
                for interval_index, (begin, end) in enumerate(intervals):
                    entries.append((batch_index, track, interval_index, begin, end))
        logits = torch.tensor(
            [[1.0, 1.0, 0.0, 0.0] for _ in entries],
            dtype=torch.float32,
            device=features.device,
        )
        return logits, entries


def _config() -> SemiCRFModelConfig:
    return SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version="v1",
        semi_crf_length_scaling="none",
        num_instrument_classes=2,
    )


def _settings(
    *,
    window_batch_size: int,
    use_boundary_head: bool = False,
) -> InferenceSettings:
    return InferenceSettings(
        window_ms=200,
        stride_ms=100,
        track_batch_size=128,
        window_batch_size=window_batch_size,
        merge_gap_ms=None,
        merge_onset_ms=0.0,
        silence_gate_rms_dbfs=None,
        note_bias=0.0,
        disable_tqdm=True,
        use_boundary_head=use_boundary_head,
        allowed_instrument_ids=(0,),
    )


def _decode(
    model: _WindowStateV1Model,
    *,
    window_batch_size: int,
    use_boundary_head: bool = False,
) -> tuple[list[PredictedNote], dict[str, int]]:
    waveform = torch.ones(2, 300)
    waveform[:, :100] = 0.0
    return decode_notes(
        model,  # type: ignore[arg-type]
        _config(),
        waveform,
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=_settings(
            window_batch_size=window_batch_size,
            use_boundary_head=use_boundary_head,
        ),
        velocity=100,
    )


def test_v1_window_batch_applies_closed_interval_before_next_decode() -> None:
    sequential_model = _WindowStateV1Model()
    batched_model = _WindowStateV1Model()

    sequential_notes, sequential_stats = _decode(sequential_model, window_batch_size=1)
    batched_notes, batched_stats = _decode(batched_model, window_batch_size=2)

    assert [
        (note.pitch, note.start_sample, note.end_sample) for note in sequential_notes
    ] == [(21, 0, 160)]
    assert batched_notes == sequential_notes
    assert batched_stats == sequential_stats
    assert sequential_model.batch_sizes == [1, 1]
    assert batched_model.batch_sizes == [2]


def test_v1_window_batch_applies_boundary_close_before_next_decode() -> None:
    sequential_model = _WindowStateBoundaryV1Model(first_end_frame=19)
    batched_model = _WindowStateBoundaryV1Model(first_end_frame=19)

    sequential_notes, sequential_stats = _decode(
        sequential_model,
        window_batch_size=1,
        use_boundary_head=True,
    )
    batched_notes, batched_stats = _decode(
        batched_model,
        window_batch_size=2,
        use_boundary_head=True,
    )

    assert batched_notes == sequential_notes
    assert batched_stats == sequential_stats
    assert sequential_stats["decoded_interval_count"] == 1
    assert sequential_stats["boundary_interval_count"] == 1
    assert sequential_model.batch_sizes == [1, 1]
    assert batched_model.batch_sizes == [2]


def test_v1_decode_uses_sparse_decoder_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sparse_calls: list[dict[str, object]] = []

    def reject_dense_decoder(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("sparse指定時にdense decoderを呼び出している")

    def sparse_decoder(
        interval_query: torch.Tensor,
        *_args: object,
        **kwargs: object,
    ) -> list[list[list[tuple[int, int]]]]:
        sparse_calls.append(kwargs)
        return [
            [[] for _ in range(88)]
            for _ in range(int(interval_query.shape[0]))
        ]

    monkeypatch.setattr(v1_windowed, "decode_pitch_intervals", reject_dense_decoder)
    monkeypatch.setattr(
        v1_windowed,
        "decode_pitch_intervals_sparse",
        sparse_decoder,
        raising=False,
    )

    notes, _ = decode_notes(
        _WindowStateV1Model(),  # type: ignore[arg-type]
        _config(),
        torch.ones(2, 200),
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=replace(
            _settings(window_batch_size=1),
            semi_crf_sparse_decode=True,
            semi_crf_sparse_topk_per_start=5,
            semi_crf_sparse_score_threshold=-0.25,
            semi_crf_sparse_max_span_frames=7,
        ),
        velocity=100,
    )

    assert notes == []
    assert len(sparse_calls) == 1
    assert sparse_calls[0]["sparse_topk_per_start"] == 5
    assert sparse_calls[0]["sparse_score_threshold"] == -0.25
    assert sparse_calls[0]["sparse_max_span_frames"] == 7
