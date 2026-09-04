from __future__ import annotations

import torch

from instrument_agnostic_amt.amt.modeling.heads.semi_crf import (
    _build_interval_score,
    build_factorized_interval_score,
    compute_factorized_pair_interval_loss,
    compute_flat_interval_loss,
    decode_factorized_pair_intervals,
    decode_factorized_pair_intervals_sparse,
    decode_pitch_intervals,
    decode_pitch_intervals_sparse,
)
from instrument_agnostic_amt.amt.modeling.heads.interval_boundaries import (
    PitchIntervalTargets,
)
from instrument_agnostic_amt.amt.modeling.heads.v2 import V2OverlapSemiCRFHead
from instrument_agnostic_amt.amt.modeling.model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)
from recipes.amt.loss_config import AMTLossConfig
from recipes.amt.losses import compute_losses


def _expanded_pair_features(
    head: V2OverlapSemiCRFHead,
    interval_features: torch.Tensor,
    selected_pair_ids: list[list[int]],
    *,
    num_pitches: int,
) -> torch.Tensor:
    """Reproduce the pre-factorization V2 feature expansion for comparison."""

    chunks = []
    for batch_index, pair_values in enumerate(selected_pair_ids):
        pair_ids = torch.tensor(pair_values, dtype=torch.long)
        instrument_indices = torch.div(
            pair_ids, num_pitches, rounding_mode="floor"
        )
        pitch_indices = pair_ids.remainder(num_pitches)
        chunks.append(
            interval_features[batch_index, :, pitch_indices]
            + head.instrument_embedding(instrument_indices).unsqueeze(0)
        )
    return torch.cat(chunks, dim=1)


def _factorized_projections(
    head: V2OverlapSemiCRFHead,
    interval_features: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    pitch_projection, instrument_projection = (
        head.interval_scorer.project_factorized(
            interval_features,
            head.instrument_embedding.weight,
        )
    )
    return (*pitch_projection, *instrument_projection)


def test_factorized_scores_losses_and_gradients_match_expanded_v2() -> None:
    torch.manual_seed(7)
    num_pitches = 5
    head = V2OverlapSemiCRFHead(
        input_dim=12,
        semi_crf_head_dim=7,
        instrument_pair_gate_dim=4,
        num_instrument_classes=4,
        dropout=0.0,
        use_interval_boundary_head=True,
    )
    interval_features = torch.randn(2, 6, num_pitches, 12, requires_grad=True)
    # Two instruments share pitch index 2, which is the V2 overlap case that
    # must remain represented by independent Semi-CRF tracks.
    selected_pair_ids = [
        [0 * num_pitches + 2, 1 * num_pitches + 2, 3 * num_pitches + 4],
        [2 * num_pitches + 2, 0 * num_pitches + 1],
    ]
    selected = head.build_selected_pair_indices(
        selected_pair_ids,
        num_pitches=num_pitches,
    )

    expanded_features = _expanded_pair_features(
        head,
        interval_features,
        selected_pair_ids,
        num_pitches=num_pitches,
    )
    expanded_query, expanded_key, expanded_diag = head.interval_scorer(
        expanded_features
    )
    expanded_score = _build_interval_score(
        expanded_query,
        expanded_key,
        expanded_diag,
        length_scaling="sqrt",
        length_penalty=0.03,
        note_bias=0.2,
    )

    (
        pitch_query,
        pitch_key,
        pitch_diag,
        instrument_query,
        instrument_key,
        instrument_diag,
    ) = _factorized_projections(head, interval_features)
    factorized_score = build_factorized_interval_score(
        pitch_query,
        pitch_key,
        pitch_diag,
        instrument_query,
        instrument_key,
        instrument_diag,
        selected.batch_indices,
        selected.instrument_indices,
        selected.pitch_indices,
        length_scaling="sqrt",
        length_penalty=0.03,
        note_bias=0.2,
    )
    torch.testing.assert_close(factorized_score, expanded_score, atol=2e-5, rtol=2e-5)

    score_weights = torch.randn_like(expanded_score)
    gradient_inputs = (
        interval_features,
        head.instrument_embedding.weight,
        head.interval_scorer.proj.weight,
        head.interval_scorer.proj.bias,
    )
    expanded_gradients = torch.autograd.grad(
        (expanded_score * score_weights).sum(),
        gradient_inputs,
        retain_graph=True,
    )
    factorized_gradients = torch.autograd.grad(
        (factorized_score * score_weights).sum(),
        gradient_inputs,
    )
    for factorized, expanded in zip(factorized_gradients, expanded_gradients):
        torch.testing.assert_close(factorized, expanded, atol=5e-5, rtol=5e-5)

    targets = [[(0, 1), (3, 4)], [(1, 3)], [], [(0, 0), (2, 4)], [(1, 2)]]
    sample_lengths = torch.tensor([6, 5], dtype=torch.long)
    expanded_lengths = sample_lengths.index_select(0, selected.batch_indices)
    expanded_loss, expanded_tracks, expanded_intervals = compute_flat_interval_loss(
        expanded_query,
        expanded_key,
        expanded_diag,
        targets,
        expanded_lengths,
        length_scaling="none",
        track_batch_size=3,
    )
    factorized_loss, factorized_tracks, factorized_intervals = (
        compute_factorized_pair_interval_loss(
            pitch_query,
            pitch_key,
            pitch_diag,
            instrument_query,
            instrument_key,
            instrument_diag,
            selected.batch_indices,
            selected.instrument_indices,
            selected.pitch_indices,
            targets,
            sample_lengths,
            length_scaling="none",
            track_batch_size=3,
        )
    )
    torch.testing.assert_close(factorized_loss, expanded_loss, atol=2e-5, rtol=2e-5)
    assert (factorized_tracks, factorized_intervals) == (
        expanded_tracks,
        expanded_intervals,
    )


def test_factorized_dense_and_sparse_decode_match_expanded_v2() -> None:
    torch.manual_seed(11)
    num_pitches = 4
    head = V2OverlapSemiCRFHead(
        input_dim=8,
        semi_crf_head_dim=5,
        instrument_pair_gate_dim=4,
        num_instrument_classes=3,
        dropout=0.0,
        use_interval_boundary_head=False,
    )
    interval_features = torch.randn(1, 7, num_pitches, 8)
    selected_pair_ids = [
        [0 * num_pitches + 1, 1 * num_pitches + 1, 2 * num_pitches + 3]
    ]
    selected = head.build_selected_pair_indices(
        selected_pair_ids,
        num_pitches=num_pitches,
    )
    expanded_features = _expanded_pair_features(
        head,
        interval_features,
        selected_pair_ids,
        num_pitches=num_pitches,
    )
    expanded_query, expanded_key, expanded_diag = head.interval_scorer(
        expanded_features
    )
    projections = _factorized_projections(head, interval_features)

    expanded_dense = decode_pitch_intervals(
        expanded_query.unsqueeze(0),
        expanded_key.unsqueeze(0),
        expanded_diag.unsqueeze(0),
        [7],
        length_scaling="none",
        note_bias=0.4,
        track_batch_size=2,
    )[0]
    factorized_dense = decode_factorized_pair_intervals(
        *projections,
        selected.batch_indices,
        selected.instrument_indices,
        selected.pitch_indices,
        [7],
        length_scaling="none",
        note_bias=0.4,
        track_batch_size=2,
    )
    assert factorized_dense == expanded_dense

    for length_scaling in ("none", "sqrt"):
        for max_span_frames in (None, 3):
            expanded_sparse = decode_pitch_intervals_sparse(
                expanded_query.unsqueeze(0),
                expanded_key.unsqueeze(0),
                expanded_diag.unsqueeze(0),
                [7],
                length_scaling=length_scaling,
                length_penalty=0.03,
                note_bias=0.4,
                track_batch_size=2,
                sparse_topk_per_start=3,
                sparse_max_span_frames=max_span_frames,
            )[0]
            factorized_sparse = decode_factorized_pair_intervals_sparse(
                *projections,
                selected.batch_indices,
                selected.instrument_indices,
                selected.pitch_indices,
                [7],
                length_scaling=length_scaling,
                length_penalty=0.03,
                note_bias=0.4,
                track_batch_size=2,
                sparse_topk_per_start=3,
                sparse_max_span_frames=max_span_frames,
            )
            assert factorized_sparse == expanded_sparse


def test_factorized_boundary_head_reconstructs_only_selected_endpoints() -> None:
    torch.manual_seed(13)
    num_pitches = 4
    head = V2OverlapSemiCRFHead(
        input_dim=8,
        semi_crf_head_dim=5,
        instrument_pair_gate_dim=4,
        num_instrument_classes=3,
        dropout=0.0,
        use_interval_boundary_head=True,
    )
    interval_features = torch.randn(2, 6, num_pitches, 8)
    selected_pair_ids = [
        [0 * num_pitches + 1, 2 * num_pitches + 1],
        [1 * num_pitches + 3],
    ]
    intervals = [[(0, 2), (4, 5)], [(1, 4)], [(2, 3)]]
    selected = head.build_selected_pair_indices(
        selected_pair_ids,
        num_pitches=num_pitches,
    )

    factorized_logits, entries = head.predict_flat_interval_boundaries(
        interval_features,
        selected,
        intervals,
    )
    expanded_features = _expanded_pair_features(
        head,
        interval_features,
        selected_pair_ids,
        num_pitches=num_pitches,
    )
    begin_features = torch.stack(
        [expanded_features[begin, track] for track, _, begin, _ in entries]
    )
    end_features = torch.stack(
        [expanded_features[end, track] for track, _, _, end in entries]
    )
    expanded_logits = head.interval_boundary_predictor(
        torch.cat(
            [begin_features, end_features, begin_features * end_features],
            dim=-1,
        )
    )
    torch.testing.assert_close(factorized_logits, expanded_logits)


def test_v2_training_loss_uses_factorized_outputs_end_to_end() -> None:
    torch.manual_seed(17)
    config = SemiCRFModelConfig(
        sample_rate=8_000,
        hop_length=128,
        n_fft=256,
        semi_crf_version="v2",
        cqt_n_bins=48,
        cqt_bins_per_octave=12,
        harmonics=(1.0,),
        hidden_size=16,
        base_ch=4,
        encoder_num_layers=0,
        encoder_num_heads=1,
        dropout=0.0,
        semi_crf_head_dim=5,
        instrument_pair_gate_dim=4,
        num_instrument_classes=3,
        use_gradient_checkpoint=False,
    )
    model = AudioSemiCRFTransformer(config)
    interval_features = torch.randn(1, 5, 88, model.backbone.query_feature_dim)
    frame_valid_mask = torch.ones(1, 5, dtype=torch.bool)
    outputs = model.head(interval_features, frame_valid_mask)

    instrument_id = 1
    pitch_index = 39
    pair_id = instrument_id * 88 + pitch_index
    pair_presence = torch.zeros(3, 88, dtype=torch.bool)
    pair_presence[instrument_id, pitch_index] = True
    target = PitchIntervalTargets(
        intervals=[[(1, 3)]],
        has_onset=[[True]],
        has_offset=[[True]],
        onset_offsets=[[0.2]],
        offset_offsets=[[0.8]],
        instrument_sets=[[(instrument_id,)]],
        positive_pair_ids=[pair_id],
        pair_presence=pair_presence,
    )
    loss_config = AMTLossConfig(
        semi_crf_loss_weight=1.0,
        semi_crf_false_negative_cost=0.0,
        semi_crf_false_positive_cost=0.0,
        semi_crf_track_batch_size=2,
        semi_crf_loss_backend="torch",
        interval_presence_loss_weight=1.0,
        interval_offset_loss_weight=1.0,
        instrument_loss_weight=1.0,
        instrument_loss_type="bce",
        instrument_pair_train_topk=2,
        instrument_pair_random_negatives=1,
        instrument_pair_max_pairs=4,
        pair_gate_loss_weight=1.0,
    )

    loss, metrics = compute_losses(
        outputs,
        {"interval_targets": [target]},
        config=loss_config,
        model=model,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert int(metrics["selected_pair_count"].item()) == 4
    assert model.head.interval_scorer.proj.weight.grad is not None
    assert torch.all(torch.isfinite(model.head.interval_scorer.proj.weight.grad))
