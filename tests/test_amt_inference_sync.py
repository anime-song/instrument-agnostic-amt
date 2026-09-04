from __future__ import annotations

import weakref

import pytest
import torch
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

from instrument_agnostic_amt.amt.inference import v1_windowed, windowed
from instrument_agnostic_amt.amt.inference.v1_windowed import (
    _decode_boundary_map,
    _decode_frame_instrument_map,
    _decode_instrument_map,
    decode_v1_notes,
)
from instrument_agnostic_amt.amt.inference.types import InferenceSettings, PredictedNote
from instrument_agnostic_amt.amt.inference.windowed import (
    _decode_flat_boundary_features,
    _rank_instrument_candidates_by_pitch,
    decode_notes,
)
from instrument_agnostic_amt.amt.modeling.model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)


def _settings(**overrides: object) -> InferenceSettings:
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


class _NonCpuToCpuCopyRecorder(TorchDispatchMode):
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
            target_device = call_kwargs.get("device")
            if (
                source.device.type != "cpu"
                and target_device is not None
                and torch.device(target_device).type == "cpu"
            ):
                self.numels.append(int(source.numel()))
        return func(*args, **call_kwargs)  # type: ignore[operator]


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


def _config(version: str) -> SemiCRFModelConfig:
    return SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version=version,
        num_instrument_classes=2,
    )


def _record_tolist_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    original_tolist = torch.Tensor.tolist

    def counted_tolist(tensor: torch.Tensor) -> list[object]:
        shapes.append(tuple(tensor.shape))
        return original_tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", counted_tolist)
    return shapes


def _empty_v1_intervals(
    interval_query: torch.Tensor,
    *_args: object,
    **_kwargs: object,
) -> list[list[list[tuple[int, int]]]]:
    return [[[] for _ in range(88)] for _ in range(int(interval_query.shape[0]))]


def _run_v2_decoder(
    forward: object,
    *,
    window_batch_size: int,
) -> tuple[list[PredictedNote], dict[str, int]]:
    return decode_notes(
        _NoCandidateV2Model(),  # type: ignore[arg-type]
        _config("v2"),
        torch.randn(2, 400),
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=_settings(
            window_ms=200,
            stride_ms=200,
            window_batch_size=window_batch_size,
        ),
        velocity=100,
        forward_model=forward,  # type: ignore[arg-type]
    )


def _run_v1_decoder(
    forward: object,
    *,
    window_batch_size: int,
    allowed_instrument_ids: tuple[int, ...] = (0,),
) -> tuple[list[PredictedNote], dict[str, int]]:
    return decode_v1_notes(
        _NoIntervalV1Model(),  # type: ignore[arg-type]
        _config("v1"),
        torch.randn(2, 400),
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=_settings(
            window_ms=200,
            stride_ms=200,
            window_batch_size=window_batch_size,
            allowed_instrument_ids=allowed_instrument_ids,
        ),
        velocity=100,
        forward_model=forward,  # type: ignore[arg-type]
    )


class _NoCandidateV2Model:
    @staticmethod
    def supports_interval_boundaries() -> bool:
        return False

    @staticmethod
    def build_selected_pair_indices(*_args: object) -> object:
        raise AssertionError("候補がない窓は復号へ進みません")


class _NoCandidateV2Forward:
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
                (batch_size, 2, 88), float("-inf"), device=waveform.device
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
                batch_size,
                frame_count,
                dtype=torch.bool,
                device=waveform.device,
            ),
        }


class _LifetimeV2Forward(_NoCandidateV2Forward):
    def __init__(self) -> None:
        self.previous_outputs: list[weakref.ReferenceType[torch.Tensor]] = []
        self.alive_before_forward: list[int] = []

    def __call__(
        self,
        waveform: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor]:
        self.alive_before_forward.append(
            sum(reference() is not None for reference in self.previous_outputs)
        )
        outputs = super().__call__(waveform, **kwargs)
        self.previous_outputs = [weakref.ref(value) for value in outputs.values()]
        return outputs


class _NoIntervalV1Model:
    _use_interval_instrument_head = False

    @staticmethod
    def supports_interval_boundaries() -> bool:
        return False

    @staticmethod
    def supports_interval_instruments() -> bool:
        return True


class _NoIntervalV1Forward:
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
                batch_size,
                frame_count,
                dtype=torch.bool,
                device=waveform.device,
            ),
            "interval_features": None,
            "instrument_features": None,
            "instrument_logits": None,
        }


class _LifetimeV1Forward(_NoIntervalV1Forward):
    def __init__(self) -> None:
        self.previous_outputs: list[weakref.ReferenceType[torch.Tensor]] = []
        self.alive_before_forward: list[int] = []

    def __call__(
        self,
        waveform: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, torch.Tensor | None]:
        self.alive_before_forward.append(
            sum(reference() is not None for reference in self.previous_outputs)
        )
        outputs = super().__call__(waveform, **kwargs)
        self.previous_outputs = [
            weakref.ref(value)
            for value in outputs.values()
            if isinstance(value, torch.Tensor)
        ]
        return outputs


class _FrameInstrumentV1Forward(_NoIntervalV1Forward):
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


def test_scalar_read_profiler_detects_a_deliberate_item_call() -> None:
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        torch.tensor(1.0).item()

    assert _local_scalar_read_count(profiler) == 1


def test_instrument_orders_are_read_in_bulk_without_scalar_tensor_reads() -> None:
    logits = torch.empty(5, 88)
    logits[0] = 100.0
    logits[1] = 2.0
    logits[2] = 1.0
    logits[3] = float("inf")
    logits[4] = float("nan")

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        ranked = _rank_instrument_candidates_by_pitch(
            logits,
            settings=_settings(allowed_instrument_ids=(4, 3, 1, 2)),
            instrument_filter_id=None,
        )

    assert ranked == [(1, 2, 4, 3)] * 88
    assert _local_scalar_read_count(profiler) == 0


def test_v2_boundaries_are_read_in_bulk_and_keep_the_same_result() -> None:
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


def test_v1_boundaries_are_read_in_bulk_and_keep_the_same_result() -> None:
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


def test_v1_instrument_orders_use_one_bulk_host_read(
    monkeypatch: pytest.MonkeyPatch,
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
    tolist_shapes = _record_tolist_shapes(monkeypatch)

    decoded = _decode_instrument_map(
        logits,
        entries,
        probability_mode="softmax",
        allowed_instrument_ids=(1, 3),
    )

    assert decoded == {
        (0, 1, 2): (1, (1, 3)),
        (0, 2, 3): (3, (3, 1)),
        (1, 0, 1): (1, (1, 3)),
    }
    assert tolist_shapes == [(3, 2)]


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS上のdevice-to-host転送を検査するテストです",
)
def test_mps_postprocess_outputs_use_one_host_transfer_per_result() -> None:
    rank_logits = torch.stack(
        (
            torch.full((88,), 2.0),
            torch.full((88,), 1.0),
        )
    ).to("mps")
    v2_boundary_logits = torch.tensor(
        [[1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0]],
        device="mps",
    )
    v1_boundary_logits = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0]],
        device="mps",
    )
    interval_instrument_logits = torch.tensor(
        [[3.0, 1.0], [1.0, 3.0]],
        device="mps",
    )
    frame_instrument_logits = torch.tensor(
        [
            [[[3.0, 1.0]], [[3.0, 1.0]], [[0.0, 0.0]]],
            [[[0.0, 0.0]], [[1.0, 3.0]], [[1.0, 3.0]]],
        ],
        device="mps",
    )
    recorder = _NonCpuToCpuCopyRecorder()

    with recorder:
        ranked = _rank_instrument_candidates_by_pitch(
            rank_logits,
            settings=_settings(allowed_instrument_ids=(0, 1)),
            instrument_filter_id=None,
        )
        v2_boundaries = _decode_flat_boundary_features(
            v2_boundary_logits,
            [(0, 0, 1, 2), (1, 1, 2, 3)],
            num_tracks=2,
        )
        v1_boundaries = _decode_boundary_map(
            v1_boundary_logits,
            [(0, 0, 0, 1, 2)],
        )
        interval_instruments = _decode_instrument_map(
            interval_instrument_logits,
            [(0, 0, 0, 1, 2), (1, 0, 0, 1, 2)],
            probability_mode="softmax",
            allowed_instrument_ids=(0, 1),
        )
        frame_instruments = _decode_frame_instrument_map(
            frame_instrument_logits,
            [[[(0, 1)]], [[(1, 2)]]],
            num_pitch_slots=1,
            allowed_instrument_ids=(0, 1),
            excluded_keys=set(),
        )

    assert ranked == [(0, 1)] * 88
    assert v2_boundaries == [
        [(True, False, 0.5, 0.5)],
        [(False, True, 0.5, 0.5)],
    ]
    assert v1_boundaries == {(0, 0, 0): (True, True, 0.5, 0.5)}
    assert interval_instruments == {
        (0, 0, 0): (0, (0, 1)),
        (1, 0, 0): (1, (1, 0)),
    }
    assert frame_instruments == {
        (0, 0, 0): (0, (0, 1)),
        (1, 0, 0): (1, (1, 0)),
    }
    assert recorder.numels == [176, 8, 4, 4, 4]


def test_frame_valid_lengths_stay_on_the_inference_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_model()
    valid_audio_frames = torch.tensor([-1, 0, 1, 63, 64, 65, 2**25 + 1, 2**62])
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
    tolist_shapes = _record_tolist_shapes(monkeypatch)

    mask = model._build_frame_valid_mask(
        batch_size=8,
        num_frames=2,
        valid_audio_frames=valid_audio_frames,
        device=torch.device("cpu"),
    )

    assert torch.equal(mask, expected)
    assert tolist_shapes == []


def test_v2_window_batch_reads_valid_lengths_without_per_window_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 候補選択の同期回数は専用PRで検証し、ここでは有効長readだけを測る。
    monkeypatch.setattr(
        windowed,
        "_select_pair_candidates",
        lambda *_args, **_kwargs: [],
    )
    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        notes, stats = _run_v2_decoder(
            _NoCandidateV2Forward(),
            window_batch_size=2,
        )

    assert notes == []
    assert stats["decoded_window_count"] == 2
    assert _local_scalar_read_count(profiler) == 0


def test_v2_releases_previous_outputs_before_the_next_forward() -> None:
    forward = _LifetimeV2Forward()

    notes, stats = _run_v2_decoder(forward, window_batch_size=1)

    assert notes == []
    assert stats["decoded_window_count"] == 2
    assert forward.alive_before_forward == [0, 0]


def test_v1_window_batch_reads_valid_lengths_without_per_window_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v1_windowed, "decode_pitch_intervals", _empty_v1_intervals)

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        notes, stats = _run_v1_decoder(
            _NoIntervalV1Forward(),
            window_batch_size=2,
        )

    assert notes == []
    assert stats["decoded_window_count"] == 2
    assert _local_scalar_read_count(profiler) == 0


def test_v1_releases_previous_outputs_before_the_next_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(v1_windowed, "decode_pitch_intervals", _empty_v1_intervals)
    forward = _LifetimeV1Forward()

    notes, stats = _run_v1_decoder(forward, window_batch_size=1)

    assert notes == []
    assert stats["decoded_window_count"] == 2
    assert forward.alive_before_forward == [0, 0]


def test_v1_frame_fallback_reads_all_candidate_orders_in_one_transfer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decode_call = 0

    def one_interval_per_window(
        _interval_query: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> list[list[list[tuple[int, int]]]]:
        nonlocal decode_call
        tracks = [[] for _ in range(88)]
        begin = decode_call * 2
        tracks[0] = [(begin, begin + 1)]
        decode_call += 1
        return [tracks]

    monkeypatch.setattr(
        v1_windowed,
        "decode_pitch_intervals",
        one_interval_per_window,
    )
    tolist_shapes = _record_tolist_shapes(monkeypatch)

    notes, stats = _run_v1_decoder(
        _FrameInstrumentV1Forward(),
        window_batch_size=2,
        allowed_instrument_ids=(0, 1),
    )

    assert [note.instrument_id for note in notes] == [0, 1]
    assert stats["decoded_interval_count"] == 2
    assert tolist_shapes == [(2,), (2, 2)]
