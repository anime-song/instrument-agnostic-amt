from __future__ import annotations

import torch
from torch.profiler import ProfilerActivity, profile

from instrument_agnostic_amt.amt.modeling.heads.semi_crf import (
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


def _legacy_viterbi_backward(
    score: torch.Tensor,
    noise_score: torch.Tensor,
    forced_start_pos: list[int] | None = None,
) -> list[list[tuple[int, int]]]:
    """高速化前の候補結合と選択規則を、回帰テスト用に再現する。"""
    time_steps = int(score.shape[0])
    track_count = int(score.shape[2])
    q = score.new_zeros(time_steps, track_count)
    pointers = []

    q[time_steps - 1] = score[time_steps - 1, time_steps - 1, :] * (
        score[time_steps - 1, time_steps - 1, :] > 0
    )
    for offset in range(1, time_steps):
        begin = time_steps - offset - 1
        candidates = torch.cat(
            [
                q[begin + 1 : begin + 2, :] + noise_score[begin, :],
                q[begin + 1 :, :] + score[begin + 1 :, begin, :],
            ],
            dim=0,
        )
        best_value, selection = candidates.max(dim=0)
        pointers.append(selection - 1)

        singleton_mask = score[begin, begin, :] > 0
        q[begin] = best_value + score[begin, begin, :] * singleton_mask

    pointer_values = torch.stack(pointers, dim=0).cpu()
    diag_inclusion = (torch.diagonal(score, dim1=0, dim2=1) > 0).cpu()
    if forced_start_pos is None:
        forced_start_pos = [0] * track_count

    result: list[list[tuple[int, int]]] = []
    for track in range(track_count):
        position = forced_start_pos[track]
        track_result: list[tuple[int, int]] = []
        current_diag = diag_inclusion[track]
        while position < time_steps - 1:
            selection = int(pointer_values[time_steps - position - 2][track])
            if bool(current_diag[position]):
                track_result.append((position, position))
            if selection < 0:
                position += 1
            else:
                end = selection + position + 1
                track_result.append((position, end))
                position = end
        if bool(current_diag[time_steps - 1]):
            track_result.append((time_steps - 1, time_steps - 1))
        result.append(track_result)
    return result


def test_dense_viterbi_does_not_materialize_transposed_score() -> None:
    score = torch.zeros(5, 5, 2)
    score[1, 0] = 2.0
    score[4, 4] = 1.0
    noise_score = torch.zeros(4, 2)

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        decoded = viterbiBackward(score, noise_score)

    assert decoded == [[(0, 1), (4, 4)], [(0, 1), (4, 4)]]
    assert _contiguous_count(profiler) == 0


def test_dense_viterbi_matches_legacy_recurrence() -> None:
    for seed in range(20):
        generator = torch.Generator().manual_seed(seed)
        score = torch.randn(17, 17, 7, generator=generator)
        noise_score = torch.randn(16, 7, generator=generator)
        forced_start_pos = [(seed + track * 3) % 17 for track in range(7)]

        expected = _legacy_viterbi_backward(
            score,
            noise_score,
            forced_start_pos,
        )
        actual = viterbiBackward(score, noise_score, forced_start_pos)

        assert actual == expected


def test_dense_viterbi_preserves_legacy_tie_breaking() -> None:
    # skipと区間が同点なら、旧実装で候補の先頭だったskipを優先する。
    skip_tie_score = torch.zeros(6, 6, 1)
    skip_tie_noise = torch.zeros(5, 1)
    assert viterbiBackward(skip_tie_score, skip_tie_noise) == [[]]

    # 複数区間が同点なら、旧実装と同じく最も早い終了位置を選ぶ。
    interval_tie_score = torch.zeros(6, 6, 1)
    interval_tie_score[2, 0, 0] = 1.0
    interval_tie_score[3, 0, 0] = 1.0
    assert viterbiBackward(interval_tie_score, skip_tie_noise) == [[(0, 2)]]


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
