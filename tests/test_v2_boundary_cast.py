from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from instrument_agnostic_amt.amt.inference import windowed
from instrument_agnostic_amt.amt.inference.types import InferenceSettings
from instrument_agnostic_amt.amt.inference.windowed import decode_notes
from instrument_agnostic_amt.amt.modeling.heads.v2 import V2OverlapSemiCRFHead
from instrument_agnostic_amt.amt.modeling.model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)


def _inference_settings(**overrides: object) -> InferenceSettings:
    values: dict[str, object] = {
        "window_ms": 200,
        "stride_ms": 200,
        "track_batch_size": 128,
        "window_batch_size": 1,
        "merge_gap_ms": None,
        "merge_onset_ms": 50.0,
        "silence_gate_rms_dbfs": None,
        "note_bias": 0.0,
        "disable_tqdm": True,
        "use_boundary_head": True,
        "instrument_pair_infer_topk": 2,
        "instrument_pair_gate_threshold": 1.0,
        "instrument_pair_max_pairs": 3,
        "allowed_instrument_ids": (0,),
    }
    values.update(overrides)
    return InferenceSettings(**values)


def _small_v2_model(
    *,
    use_interval_boundary_head: bool = True,
) -> AudioSemiCRFTransformer:
    return AudioSemiCRFTransformer(
        SemiCRFModelConfig(
            sample_rate=16_000,
            hop_length=64,
            n_fft=256,
            semi_crf_version="v2",
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
            use_interval_boundary_head=use_interval_boundary_head,
        )
    ).eval()


class _V2BoundaryModel:
    def __init__(self) -> None:
        self.boundary_dtypes: list[tuple[torch.dtype, torch.dtype | None]] = []

    @staticmethod
    def supports_interval_boundaries() -> bool:
        return True

    @staticmethod
    def build_selected_pair_indices(*_args: object) -> SimpleNamespace:
        return SimpleNamespace(
            batch_indices=torch.tensor([0]),
            instrument_indices=torch.tensor([0]),
            pitch_indices=torch.tensor([0]),
        )

    def predict_flat_interval_boundaries(
        self,
        features: torch.Tensor,
        _selected_pairs: object,
        _interval_batch: object,
        *,
        compute_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
        self.boundary_dtypes.append((features.dtype, compute_dtype))
        return torch.zeros(0, 4, dtype=torch.float32), []


class _V2BoundaryForward:
    def __call__(
        self,
        waveform: torch.Tensor,
        **_kwargs: object,
    ) -> dict[str, torch.Tensor]:
        batch_size = int(waveform.shape[0])
        frame_count = 20
        pair_gate_logits = torch.full(
            (batch_size, 2, 88),
            float("-inf"),
            device=waveform.device,
        )
        pair_gate_logits[:, 0, 0] = 2.0
        return {
            "interval_features": torch.zeros(
                batch_size,
                frame_count,
                88,
                1,
                device=waveform.device,
                dtype=torch.float16,
            ),
            "pair_gate_logits": pair_gate_logits,
            "pitch_interval_query": torch.zeros(
                batch_size, frame_count, 88, 1, device=waveform.device
            ),
            "pitch_interval_key": torch.zeros(
                batch_size, frame_count, 88, 1, device=waveform.device
            ),
            "pitch_interval_diag": torch.zeros(
                batch_size, frame_count, 88, device=waveform.device
            ),
            "instrument_interval_query": torch.zeros(
                2, 1, device=waveform.device
            ),
            "instrument_interval_key": torch.zeros(
                2, 1, device=waveform.device
            ),
            "instrument_interval_diag": torch.zeros(2, device=waveform.device),
            "frame_valid_mask": torch.ones(
                batch_size,
                frame_count,
                dtype=torch.bool,
                device=waveform.device,
            ),
        }


@pytest.mark.parametrize("feature_dtype", [torch.float16, torch.bfloat16])
def test_v2_boundary_endpoint_cast_matches_full_float32_logits_and_gradients(
    feature_dtype: torch.dtype,
) -> None:
    torch.manual_seed(7)
    model = _small_v2_model()
    assert isinstance(model.head, V2OverlapSemiCRFHead)
    assert model.head.interval_boundary_predictor is not None

    feature_dim = model.backbone.query_feature_dim
    source = torch.randn(1, 5, 88, feature_dim, dtype=feature_dtype)
    reference_features = source.clone().requires_grad_()
    actual_features = source.clone().requires_grad_()
    selected_pairs = model.build_selected_pair_indices([[0, 88 + 3]])
    interval_batch = [[(0, 2)], [(1, 4)]]

    reference_logits, reference_entries = model.predict_flat_interval_boundaries(
        reference_features.float(),
        selected_pairs,
        interval_batch,
    )
    actual_logits, actual_entries = model.predict_flat_interval_boundaries(
        actual_features,
        selected_pairs,
        interval_batch,
        compute_dtype=torch.float32,
    )
    loss_weights = torch.randn_like(reference_logits)
    parameter_inputs = (
        model.head.instrument_embedding.weight,
        *model.head.interval_boundary_predictor.parameters(),
    )
    reference_gradients = torch.autograd.grad(
        (reference_logits * loss_weights).sum(),
        (reference_features, *parameter_inputs),
    )
    actual_gradients = torch.autograd.grad(
        (actual_logits * loss_weights).sum(),
        (actual_features, *parameter_inputs),
    )

    assert actual_entries == reference_entries
    assert torch.equal(actual_logits, reference_logits)
    for actual, reference in zip(actual_gradients, reference_gradients):
        assert torch.equal(actual, reference)


@pytest.mark.parametrize("use_interval_boundary_head", [False, True])
def test_v2_empty_boundary_logits_honor_compute_dtype(
    use_interval_boundary_head: bool,
) -> None:
    model = _small_v2_model(
        use_interval_boundary_head=use_interval_boundary_head,
    )
    feature_dim = model.backbone.query_feature_dim
    features = torch.zeros(1, 3, 88, feature_dim, dtype=torch.float16)
    selected_pairs = model.build_selected_pair_indices([[0]])

    logits, entries = model.predict_flat_interval_boundaries(
        features,
        selected_pairs,
        [[]],
        compute_dtype=torch.float32,
    )

    assert entries == []
    assert logits.shape == (0, 4)
    assert logits.dtype == torch.float32


def test_v2_decoder_casts_only_selected_boundary_endpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SemiCRFModelConfig(
        sample_rate=1_000,
        hop_length=10,
        n_fft=128,
        semi_crf_version="v2",
        num_instrument_classes=2,
    )
    monkeypatch.setattr(
        windowed,
        "decode_factorized_pair_intervals",
        lambda *_args, **_kwargs: [[]],
    )
    model = _V2BoundaryModel()

    decode_notes(
        model,  # type: ignore[arg-type]
        config,
        torch.randn(2, 200),
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=_inference_settings(),
        velocity=100,
        forward_model=_V2BoundaryForward(),
    )

    assert model.boundary_dtypes == [(torch.float16, torch.float32)]
