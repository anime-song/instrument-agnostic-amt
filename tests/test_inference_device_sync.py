from __future__ import annotations

import pytest
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

from instrument_agnostic_amt.inference import v1_windowed
from instrument_agnostic_amt.inference.v1_windowed import (
    _decode_boundary_map,
    _decode_instrument_map,
    decode_v1_notes,
)
from instrument_agnostic_amt.inference.types import InferenceSettings
from instrument_agnostic_amt.inference.windowed import (
    _decode_flat_boundary_features,
    _rank_instrument_candidates_by_pitch,
    _select_pair_candidates,
    decode_notes,
)
from instrument_agnostic_amt.modeling.model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)
from instrument_agnostic_amt.modeling.heads.semi_crf import viterbiBackward


def _inference_settings(**overrides: object) -> InferenceSettings:
    values: dict[str, object] = {
        "window_ms": 8_000,
        "stride_ms": 4_000,
        "track_batch_size": 128,
        "window_batch_size": 1,
        "merge_gap_ms": None,
        "merge_onset_ms": 50.0,
        "silence_gate_rms_dbfs": None,
        "note_bias": 0.0,
        "disable_tqdm": True,
        "instrument_pair_infer_topk": 2,
        "instrument_pair_gate_threshold": 1.0,
        "instrument_pair_max_pairs": 3,
        "allowed_instrument_ids": (0, 1),
    }
    values.update(overrides)
    return InferenceSettings(**values)


def _local_scalar_read_count(profiler: profile) -> int:
    return sum(
        event.count
        for event in profiler.key_averages()
        if event.key == "aten::_local_scalar_dense"
    )


class _NonCpuScalarReadCounter(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def __torch_dispatch__(
        self,
        func: object,
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if "_local_scalar_dense" in str(func):
            self.count += sum(
                int(isinstance(value, torch.Tensor) and value.device.type != "cpu")
                for value in args
            )
        return func(*args, **({} if kwargs is None else kwargs))  # type: ignore[operator]


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
    ).eval()


class _NoCandidateModel:
    @staticmethod
    def supports_interval_boundaries() -> bool:
        return False

    @staticmethod
    def build_selected_pair_indices(*_args: object) -> object:
        raise AssertionError("no candidate pair should reach the decoder")


class _NoCandidateForward:
    def __call__(
        self,
        waveform: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(waveform.shape[0])
        frame_count = 20
        return {
            "interval_features": torch.zeros(
                batch_size, frame_count, 88, 1, device=waveform.device
            ),
            "pair_gate_logits": torch.full(
                (batch_size, 2, 88),
                float("-inf"),
                device=waveform.device,
            ),
            "pitch_interval_query": torch.zeros(
                batch_size, frame_count, 88, 1, device=waveform.device
            ),
            "pitch_interval_key": torch.zeros(
                batch_size, frame_count, 88, 1, device=waveform.device
            ),
            "pitch_interval_diag": torch.zeros(
                batch_size, frame_count, 88, device=waveform.device
            ),
            "instrument_interval_query": torch.zeros(2, 1, device=waveform.device),
            "instrument_interval_key": torch.zeros(2, 1, device=waveform.device),
            "instrument_interval_diag": torch.zeros(2, device=waveform.device),
            "frame_valid_mask": torch.ones(
                batch_size, frame_count, dtype=torch.bool, device=waveform.device
            ),
        }


class _V1NoIntervalModel:
    _use_interval_instrument_head = False

    @staticmethod
    def supports_interval_boundaries() -> bool:
        return False

    @staticmethod
    def supports_interval_instruments() -> bool:
        return True


class _V1NoIntervalForward:
    def __call__(
        self,
        waveform: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor | None]:
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
                batch_size, frame_count, dtype=torch.bool, device=waveform.device
            ),
            "interval_features": None,
            "instrument_features": None,
            "instrument_logits": None,
        }


class _V1FrameInstrumentForward(_V1NoIntervalForward):
    def __call__(
        self,
        waveform: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | None]:
        outputs = super().__call__(waveform, **kwargs)
        frame_logits = torch.zeros(
            int(waveform.shape[0]),
            20,
            88,
            2,
            device=waveform.device,
        )
        frame_logits[0, :, 0, 0] = 2.0
        frame_logits[0, :, 0, 1] = -1.0
        frame_logits[1, :, 0, 0] = -1.0
        frame_logits[1, :, 0, 1] = 2.0
        outputs["instrument_logits"] = frame_logits
        return outputs


def test_pair_candidate_selection_avoids_scalar_tensor_reads() -> None:
    logits = torch.full((3, 88), float("-inf"))
    logits[0, 0] = 2.0
    logits[0, 1] = 2.0
    logits[0, 2] = float("nan")
    logits[1, 0] = 1.5
    logits[2, 0] = 100.0

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        selected = _select_pair_candidates(
            logits,
            settings=_inference_settings(),
            instrument_filter_id=None,
        )

    assert selected == [0, 1, 88]
    assert _local_scalar_read_count(profiler) == 0


def test_profiler_detects_deliberate_scalar_tensor_read() -> None:
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        torch.tensor(1.0).item()

    assert _local_scalar_read_count(profiler) == 1


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS上のdevice scalar readを検査するテストです",
)
def test_dense_viterbi_reuses_cpu_diagonal_for_final_frame() -> None:
    score = torch.zeros(4, 4, 3, device="mps")
    score[-1, -1] = 1.0
    noise_score = torch.zeros(3, 3, device="mps")
    counter = _NonCpuScalarReadCounter()

    with counter:
        decoded = viterbiBackward(score, noise_score)

    assert decoded == [[(3, 3)], [(3, 3)], [(3, 3)]]
    assert counter.count == 0


def test_dense_viterbi_does_not_materialize_transposed_score() -> None:
    score = torch.zeros(5, 5, 2)
    score[1, 0] = 2.0
    score[4, 4] = 1.0
    noise_score = torch.zeros(4, 2)

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        decoded = viterbiBackward(score, noise_score)

    contiguous_count = sum(
        event.count
        for event in profiler.key_averages()
        if event.key == "aten::contiguous"
    )
    assert decoded == [[(0, 1), (4, 4)], [(0, 1), (4, 4)]]
    assert contiguous_count == 0


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available",
)
def test_pair_candidate_topk_runs_on_mps_and_matches_cpu() -> None:
    logits = torch.full((2, 88), -10.0)
    logits[0, 3] = 2.0
    logits[0, 5] = 1.0
    settings = _inference_settings(
        instrument_pair_gate_threshold=100.0,
        instrument_pair_infer_topk=2,
        instrument_pair_max_pairs=2,
        allowed_instrument_ids=(0,),
    )

    cpu_selected = _select_pair_candidates(
        logits,
        settings=settings,
        instrument_filter_id=None,
    )
    mps_selected = _select_pair_candidates(
        logits.to("mps"),
        settings=settings,
        instrument_filter_id=None,
    )

    assert cpu_selected == [3, 5]
    assert mps_selected == cpu_selected


def test_pair_candidate_selection_filters_nonfinite_topk_padding() -> None:
    logits = torch.full((2, 88), float("nan"))
    logits[0, 5] = -2.0
    logits[0, 6] = float("inf")
    logits[1, 3] = -1.0

    selected = _select_pair_candidates(
        logits,
        settings=_inference_settings(
            instrument_pair_gate_threshold=10.0,
            instrument_pair_infer_topk=10,
            instrument_pair_max_pairs=0,
        ),
        instrument_filter_id=None,
    )

    assert selected == [91, 5]


def test_pair_gate_instrument_ranking_avoids_scalar_tensor_reads() -> None:
    logits = torch.empty(5, 88)
    logits[0] = 100.0
    logits[1] = 2.0
    logits[2] = 1.0
    logits[3] = float("inf")
    logits[4] = float("nan")

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        ranked = _rank_instrument_candidates_by_pitch(
            logits,
            settings=_inference_settings(allowed_instrument_ids=(4, 3, 1, 2)),
            instrument_filter_id=None,
        )

    assert ranked == [(1, 2, 4, 3)] * 88
    assert _local_scalar_read_count(profiler) == 0


def test_v2_boundary_decoding_bulk_reads_interval_results() -> None:
    logits = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    entries = [
        (1, 0, 2, 3),
        (0, 0, 1, 4),
        (1, 1, 5, 6),
    ]

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        decoded = _decode_flat_boundary_features(logits, entries, num_tracks=2)

    assert decoded == [
        [(False, True, 0.5, 0.5)],
        [(True, False, 0.5, 0.5), (True, True, 0.5, 0.5)],
    ]
    # ContinuousBernoulliの引数検証に伴う1回だけは許容する。
    assert _local_scalar_read_count(profiler) <= 1


def test_v1_boundary_decoding_bulk_reads_interval_results() -> None:
    logits = torch.tensor(
        [
            [1.0, -1.0, 0.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0],
        ]
    )
    entries = [
        (1, 2, 3, 4, 5),
        (0, 6, 7, 8, 9),
    ]

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        decoded = _decode_boundary_map(logits, entries)

    assert decoded == {
        (1, 2, 3): (True, False, 0.5, 0.5),
        (0, 6, 7): (False, True, 0.5, 0.5),
    }
    # ContinuousBernoulliの引数検証に伴う1回だけは許容する。
    assert _local_scalar_read_count(profiler) <= 1


@pytest.mark.parametrize("probability_mode", ["softmax", "sigmoid"])
def test_v1_instrument_decoding_bulk_reads_candidate_orders(
    monkeypatch: pytest.MonkeyPatch,
    probability_mode: str,
) -> None:
    logits = torch.tensor(
        [
            [0.0, 3.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 2.0],
            [0.0, 4.0, 0.0, -2.0],
        ]
    )
    entries = [
        (0, 1, 2, 3, 4),
        (0, 2, 3, 4, 5),
        (1, 0, 1, 2, 3),
    ]
    tolist_shapes: list[tuple[int, ...]] = []
    original_tolist = torch.Tensor.tolist

    def counted_tolist(tensor: torch.Tensor) -> list[object]:
        tolist_shapes.append(tuple(tensor.shape))
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)

    decoded = _decode_instrument_map(
        logits,
        entries,
        probability_mode=probability_mode,
        allowed_instrument_ids=(1, 3),
    )

    assert decoded == {
        (0, 1, 2): (1, (1, 3)),
        (0, 2, 3): (3, (3, 1)),
        (1, 0, 1): (1, (1, 3)),
    }
    assert tolist_shapes == [(3, 2)]


def test_frame_valid_mask_keeps_length_calculation_on_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model()
    valid_audio_frames = torch.tensor(
        [-1, 0, 1, 63, 64, 65, 2**25 + 1, 2**62]
    )
    expected = torch.tensor(
        [
            [False, False],
            [False, False],
            [True, False],
            [True, False],
            [True, False],
            [True, True],
            [True, True],
            [True, True],
        ]
    )
    tolist_shapes: list[tuple[int, ...]] = []
    original_tolist = torch.Tensor.tolist

    def counted_tolist(tensor: torch.Tensor) -> list[object]:
        tolist_shapes.append(tuple(tensor.shape))
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)

    mask = model._build_frame_valid_mask(
        batch_size=8,
        num_frames=2,
        valid_audio_frames=valid_audio_frames,
        device=torch.device("cpu"),
    )

    assert torch.equal(mask, expected)
    assert tolist_shapes == []


def test_frame_valid_mask_handles_none_and_rejects_non_vector_lengths() -> None:
    model = _small_model()

    full_mask = model._build_frame_valid_mask(
        batch_size=2,
        num_frames=3,
        valid_audio_frames=None,
        device=torch.device("cpu"),
    )

    assert torch.equal(full_mask, torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(ValueError, match="1D tensor"):
        model._build_frame_valid_mask(
            batch_size=2,
            num_frames=3,
            valid_audio_frames=torch.ones(2, 1, dtype=torch.long),
            device=torch.device("cpu"),
        )


def test_v2_decode_bulk_reads_valid_lengths_for_a_window_batch() -> None:
    config = SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version="v2",
        num_instrument_classes=2,
    )
    settings = _inference_settings(
        window_ms=200,
        stride_ms=200,
        window_batch_size=2,
        silence_gate_rms_dbfs=-60.0,
        instrument_pair_infer_topk=0,
        instrument_pair_gate_threshold=0.5,
        instrument_pair_max_pairs=1,
    )

    waveform = torch.zeros(2, 400)
    waveform[:, 200:] = 1.0
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        notes, stats = decode_notes(
            _NoCandidateModel(),  # type: ignore[arg-type]
            config,
            waveform,
            instrument_filter_id=None,
            device=torch.device("cpu"),
            amp_enabled=False,
            amp_dtype=torch.float32,
            settings=settings,
            velocity=100,
            forward_model=_NoCandidateForward(),
        )

    assert notes == []
    assert stats["decoded_window_count"] == 1
    assert stats["skipped_silent_window_count"] == 1
    # CPU上の無音窓数集計に伴う1回だけは許容する。
    assert _local_scalar_read_count(profiler) <= 1


def test_v1_decode_bulk_reads_valid_lengths_for_a_window_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version="v1",
        num_instrument_classes=2,
    )
    settings = _inference_settings(
        window_ms=200,
        stride_ms=200,
        window_batch_size=2,
        silence_gate_rms_dbfs=-60.0,
        allowed_instrument_ids=(0,),
    )

    def no_intervals(
        interval_query: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> list[list[list[tuple[int, int]]]]:
        return [
            [[] for _ in range(88)]
            for _ in range(int(interval_query.shape[0]))
        ]

    monkeypatch.setattr(v1_windowed, "decode_pitch_intervals", no_intervals)

    waveform = torch.zeros(2, 400)
    waveform[:, 200:] = 1.0
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        notes, stats = decode_v1_notes(
            _V1NoIntervalModel(),  # type: ignore[arg-type]
            config,
            waveform,
            instrument_filter_id=None,
            device=torch.device("cpu"),
            amp_enabled=False,
            amp_dtype=torch.float32,
            settings=settings,
            velocity=100,
            forward_model=_V1NoIntervalForward(),
        )

    assert notes == []
    assert stats["decoded_window_count"] == 1
    assert stats["skipped_silent_window_count"] == 1
    assert _local_scalar_read_count(profiler) == 0


def test_v1_decode_uses_sparse_decoder_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version="v1",
        num_instrument_classes=2,
    )
    settings = _inference_settings(
        window_ms=200,
        stride_ms=200,
        semi_crf_sparse_decode=True,
        semi_crf_sparse_topk_per_start=5,
        semi_crf_sparse_score_threshold=-0.25,
        semi_crf_sparse_max_span_frames=7,
        allowed_instrument_ids=(0,),
    )
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

    notes, _ = decode_v1_notes(
        _V1NoIntervalModel(),  # type: ignore[arg-type]
        config,
        torch.randn(2, 200),
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=settings,
        velocity=100,
        forward_model=_V1NoIntervalForward(),
    )

    assert notes == []
    assert len(sparse_calls) == 1
    assert sparse_calls[0]["sparse_topk_per_start"] == 5
    assert sparse_calls[0]["sparse_score_threshold"] == -0.25
    assert sparse_calls[0]["sparse_max_span_frames"] == 7


def test_v1_frame_instrument_fallback_bulk_reads_candidate_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version="v1",
        num_instrument_classes=2,
    )
    settings = _inference_settings(
        window_ms=200,
        stride_ms=200,
        window_batch_size=2,
        allowed_instrument_ids=(0, 1),
    )

    def two_intervals(
        interval_query: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> list[list[list[tuple[int, int]]]]:
        samples = []
        for sample_index in range(int(interval_query.shape[0])):
            tracks = [[] for _ in range(88)]
            tracks[0] = [(sample_index * 2, sample_index * 2 + 1)]
            samples.append(tracks)
        return samples

    monkeypatch.setattr(v1_windowed, "decode_pitch_intervals", two_intervals)
    tolist_shapes: list[tuple[int, ...]] = []
    original_tolist = torch.Tensor.tolist

    def counted_tolist(tensor: torch.Tensor) -> list[object]:
        tolist_shapes.append(tuple(tensor.shape))
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)

    notes, stats = decode_v1_notes(
        _V1NoIntervalModel(),  # type: ignore[arg-type]
        config,
        torch.randn(2, 400),
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=settings,
        velocity=100,
        forward_model=_V1FrameInstrumentForward(),
    )

    assert [note.instrument_id for note in notes] == [0, 1]
    assert stats["decoded_interval_count"] == 2
    assert tolist_shapes == [(2,), (2, 2)]
