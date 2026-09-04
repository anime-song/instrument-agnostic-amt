from __future__ import annotations

import pytest
import torch

from instrument_agnostic_amt.amt.inference import v1_windowed
from instrument_agnostic_amt.amt.inference.types import InferenceSettings
from instrument_agnostic_amt.amt.inference.windowed import decode_notes
from instrument_agnostic_amt.amt.modeling.heads.v2 import SelectedPairIndices
from instrument_agnostic_amt.amt.modeling.model import (
    AudioSemiCRFTransformer,
    NUM_PITCHES,
    SemiCRFModelConfig,
)
from instrument_agnostic_amt.runtime import maybe_compile_forward


class _EagerV2DecoderModel:
    def __init__(self) -> None:
        self.build_selected_pair_indices_call_count = 0

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("the eager forward must not be called")

    @staticmethod
    def supports_interval_boundaries() -> bool:
        return False

    def build_selected_pair_indices(
        self,
        selected_pair_ids: list[list[int]],
    ) -> SelectedPairIndices:
        self.build_selected_pair_indices_call_count += 1
        assert selected_pair_ids == [[0]]
        indices = torch.tensor([0], dtype=torch.long)
        return SelectedPairIndices(
            batch_indices=indices,
            instrument_indices=indices,
            pitch_indices=indices,
            pair_ids=indices,
        )


class _CompiledV2Forward:
    def __init__(self) -> None:
        self.valid_audio_frames: list[list[int]] = []

    def __call__(
        self,
        waveform: torch.Tensor,
        *,
        valid_audio_frames: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        self.valid_audio_frames.append(valid_audio_frames.tolist())
        batch_size = int(waveform.shape[0])
        frame_count = 20
        pair_gate_logits = torch.full((batch_size, 2, NUM_PITCHES), -100.0)
        pair_gate_logits[:, 0, 0] = 100.0
        return {
            "interval_features": torch.zeros(
                batch_size, frame_count, NUM_PITCHES, 1
            ),
            "pair_gate_logits": pair_gate_logits,
            "pitch_interval_query": torch.zeros(
                batch_size, frame_count, NUM_PITCHES, 1
            ),
            "pitch_interval_key": torch.zeros(
                batch_size, frame_count, NUM_PITCHES, 1
            ),
            "pitch_interval_diag": torch.zeros(
                batch_size, frame_count, NUM_PITCHES
            ),
            "instrument_interval_query": torch.zeros(2, 1),
            "instrument_interval_key": torch.zeros(2, 1),
            "instrument_interval_diag": torch.zeros(2),
            "frame_valid_mask": torch.ones(
                batch_size, frame_count, dtype=torch.bool
            ),
        }


def _settings() -> InferenceSettings:
    return InferenceSettings(
        window_ms=200,
        stride_ms=200,
        track_batch_size=1,
        window_batch_size=1,
        merge_gap_ms=None,
        merge_onset_ms=0.0,
        silence_gate_rms_dbfs=None,
        note_bias=0.0,
        disable_tqdm=True,
        use_boundary_head=False,
        instrument_pair_infer_topk=0,
        instrument_pair_gate_threshold=0.5,
        instrument_pair_max_pairs=1,
    )


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
        encoder_num_layers=2,
        encoder_num_heads=2,
        dropout=0.0,
        use_gradient_checkpoint=False,
        semi_crf_head_dim=8,
        num_instrument_classes=2,
        instrument_pair_gate_dim=8,
    )
    return AudioSemiCRFTransformer(config).eval()


def test_regional_compile_reuses_graph_for_variable_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _small_amt_model()
    graph_count = 0

    def counting_backend(
        graph_module: torch.fx.GraphModule,
        _example_inputs: list[torch.Tensor],
    ) -> object:
        nonlocal graph_count
        graph_count += 1
        return graph_module.forward

    def compile_with_counting_backend(
        module: torch.nn.Module,
        *,
        backend: str,
        mode: str,
        fullgraph: bool,
        dynamic: bool | None,
    ) -> None:
        assert backend == "inductor"
        assert mode == "default"
        module.forward = torch.compile(  # type: ignore[method-assign]
            module.forward,
            backend=counting_backend,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

    monkeypatch.setattr(torch.nn.Module, "compile", compile_with_counting_backend)
    compiled_model = maybe_compile_forward(model, enabled=True)

    with torch.inference_mode():
        compiled_model(
            torch.randn(4, 2, 4_096),
            valid_audio_frames=torch.full((4,), 4_096),
            include_aux_outputs=False,
        )
        initial_graph_count = graph_count
        for batch_size in (1, 2):
            compiled_model(
                torch.randn(batch_size, 2, 4_096),
                valid_audio_frames=torch.full((batch_size,), 4_096),
                include_aux_outputs=False,
            )

    assert initial_graph_count > 0
    assert graph_count == initial_graph_count


def test_v2_reuses_compiled_forward_for_full_and_partial_windows() -> None:
    config = SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version="v2",
        num_instrument_classes=2,
    )
    eager_model = _EagerV2DecoderModel()
    compiled_forward = _CompiledV2Forward()

    for total_audio_frames in (200, 150):
        notes, stats = decode_notes(
            eager_model,  # type: ignore[arg-type]
            config,
            torch.randn(2, total_audio_frames),
            instrument_filter_id=None,
            device=torch.device("cpu"),
            amp_enabled=False,
            amp_dtype=torch.float32,
            settings=_settings(),
            velocity=100,
            forward_model=compiled_forward,
        )
        assert isinstance(notes, list)
        assert stats["selected_pair_count"] == 1

    assert compiled_forward.valid_audio_frames == [[200], [150]]
    assert eager_model.build_selected_pair_indices_call_count == 2


class _EagerV1DecoderModel:
    _use_interval_instrument_head = False

    def __call__(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("the eager forward must not be called")

    @staticmethod
    def supports_interval_boundaries() -> bool:
        return False

    @staticmethod
    def supports_interval_instruments() -> bool:
        return False


class _CompiledV1Forward:
    def __init__(self) -> None:
        self.valid_audio_frames: list[list[int]] = []

    def __call__(
        self,
        waveform: torch.Tensor,
        *,
        valid_audio_frames: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        self.valid_audio_frames.append(valid_audio_frames.tolist())
        batch_size = int(waveform.shape[0])
        return {
            "frame_valid_mask": torch.ones(batch_size, 1, dtype=torch.bool),
            "interval_query": torch.zeros(batch_size, 1),
            "interval_key": torch.zeros(batch_size, 1),
            "interval_diag": torch.zeros(batch_size, 1),
        }


def test_v1_reuses_compiled_forward_for_full_and_partial_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version="v1",
        num_instrument_classes=2,
    )

    def no_intervals(
        interval_query: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> list[list[list[tuple[int, int]]]]:
        return [
            [[] for _ in range(NUM_PITCHES)]
            for _ in range(int(interval_query.shape[0]))
        ]

    monkeypatch.setattr(v1_windowed, "decode_pitch_intervals", no_intervals)
    eager_model = _EagerV1DecoderModel()
    compiled_forward = _CompiledV1Forward()

    for total_audio_frames in (200, 150):
        notes, stats = decode_notes(
            eager_model,  # type: ignore[arg-type]
            config,
            torch.randn(2, total_audio_frames),
            instrument_filter_id=None,
            device=torch.device("cpu"),
            amp_enabled=False,
            amp_dtype=torch.float32,
            settings=_settings(),
            velocity=100,
            forward_model=compiled_forward,
        )
        assert notes == []
        assert stats["decoded_window_count"] == 1

    assert compiled_forward.valid_audio_frames == [[200], [150]]
