from __future__ import annotations

import torch

from instrument_agnostic_amt.inference.types import InferenceSettings
from instrument_agnostic_amt.inference.windowed import decode_notes
from instrument_agnostic_amt.modeling.model import (
    NUM_PITCHES,
    SemiCRFModelConfig,
)
from instrument_agnostic_amt.modeling.heads.v2 import SelectedPairIndices


class _EagerDecoderModel:
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


class _CompiledForward:
    def __init__(self) -> None:
        self.call_count = 0

    def __call__(
        self,
        waveform: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        self.call_count += 1
        batch_size = int(waveform.shape[0])
        frame_count = 20
        pair_gate_logits = torch.full(
            (batch_size, 2, NUM_PITCHES), -100.0
        )
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


def test_decode_notes_uses_compiled_forward_but_keeps_eager_decoder_model() -> None:
    config = SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version="v2",
        num_instrument_classes=2,
    )
    settings = InferenceSettings(
        window_ms=200,
        stride_ms=100,
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
    eager_model = _EagerDecoderModel()
    compiled_forward = _CompiledForward()

    notes, stats = decode_notes(
        eager_model,  # type: ignore[arg-type]
        config,
        torch.randn(2, 200),
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=settings,
        velocity=100,
        forward_model=compiled_forward,
    )

    assert isinstance(notes, list)
    assert stats["selected_pair_count"] == 1
    assert compiled_forward.call_count == 1
    assert eager_model.build_selected_pair_indices_call_count == 1
