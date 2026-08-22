from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile

from instrument_agnostic_amt.modeling.heads.semi_crf import (
    _build_dense_sparse_candidates,
    _select_sparse_candidates,
    viterbiBackward,
)


def _contiguous_count(profiler: profile) -> int:
    return sum(
        event.count
        for event in profiler.key_averages()
        if event.key == "aten::contiguous"
    )


def test_dense_viterbi_does_not_materialize_transposed_score() -> None:
    score = torch.zeros(5, 5, 2)
    score[1, 0] = 2.0
    score[4, 4] = 1.0
    noise_score = torch.zeros(4, 2)

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        decoded = viterbiBackward(score, noise_score)

    assert decoded == [[(0, 1), (4, 4)], [(0, 1), (4, 4)]]
    assert _contiguous_count(profiler) == 0


def test_sparse_candidates_do_not_copy_transposed_score() -> None:
    score = torch.arange(5 * 5 * 2, dtype=torch.float32).reshape(5, 5, 2)
    begin_index = torch.arange(5).unsqueeze(1)
    end_index = torch.arange(5).unsqueeze(0)
    valid_mask = end_index > begin_index
    reference_scores = score.transpose(0, 1).contiguous().masked_fill(
        ~valid_mask.unsqueeze(-1),
        float("-inf"),
    )
    reference = _select_sparse_candidates(
        reference_scores,
        end_index.expand(5, 5),
        topk_per_start=2,
        score_threshold=None,
    )

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        actual = _build_dense_sparse_candidates(
            score,
            topk_per_start=2,
            score_threshold=None,
        )

    assert torch.equal(actual[0], reference[0])
    assert torch.equal(actual[1], reference[1])
    assert _contiguous_count(profiler) == 0
