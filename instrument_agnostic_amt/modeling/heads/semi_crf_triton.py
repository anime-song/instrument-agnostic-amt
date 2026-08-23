from __future__ import annotations

import torch
import triton
import triton.language as tl


IntervalBatch = list[list[tuple[int, int]]]


@triton.jit
def _viterbi_backward_kernel(
    score_by_track_begin_end,
    noise_by_track,
    q,
    pointer_end,
    time_steps: tl.constexpr,
    block_time: tl.constexpr,
):
    """1 programで1トラック分のbackward Viterbiを処理する。"""
    track = tl.program_id(0)
    end_offsets = tl.arange(0, block_time)
    score_track_offset = track * time_steps * time_steps
    q_track_offset = track * time_steps
    noise_track_offset = track * (time_steps - 1)

    # 1. 最終時刻のsingleton scoreをDPの初期値にする。
    last = time_steps - 1
    last_diag = tl.load(
        score_by_track_begin_end + score_track_offset + last * time_steps + last
    )
    tl.store(q + q_track_offset + last, last_diag * (last_diag > 0.0))
    tl.debug_barrier()

    # 2. 後ろから順に、区間開始とskipの最良候補を選ぶ。
    for offset in tl.range(1, time_steps, loop_unroll_factor=1):
        begin = time_steps - offset - 1
        valid_end = (end_offsets > begin) & (end_offsets < time_steps)
        future_q = tl.load(
            q + q_track_offset + end_offsets,
            mask=valid_end,
            other=-float("inf"),
        )
        interval_score = tl.load(
            score_by_track_begin_end
            + score_track_offset
            + begin * time_steps
            + end_offsets,
            mask=valid_end,
            other=-float("inf"),
        )
        interval_values = future_q + interval_score
        best_interval_value = tl.max(interval_values, axis=0)
        best_interval_end = tl.argmax(
            interval_values,
            axis=0,
            tie_break_left=True,
        )
        skip_value = tl.load(q + q_track_offset + begin + 1) + tl.load(
            noise_by_track + noise_track_offset + begin
        )

        # torch版はskipを先頭候補としていたため、同点時はskipを維持する。
        use_interval = best_interval_value > skip_value
        tl.store(
            pointer_end + q_track_offset + begin,
            tl.where(use_interval, best_interval_end, -1),
        )

        diag = tl.load(
            score_by_track_begin_end
            + score_track_offset
            + begin * time_steps
            + begin
        )
        best_value = tl.where(use_interval, best_interval_value, skip_value)
        tl.store(
            q + q_track_offset + begin,
            best_value + diag * (diag > 0.0),
        )

        # 次の反復では別warpも今保存したq[begin]を参照する。
        tl.debug_barrier()


def _traceback(
    pointer_end: torch.Tensor,
    diag_inclusion: torch.Tensor,
    forced_start_pos: list[int] | None,
) -> IntervalBatch:
    """GPUから一括転送した絶対終了位置を区間列へ戻す。"""
    pointer_values = pointer_end.cpu().tolist()
    diag_values = diag_inclusion.cpu().tolist()
    track_count = int(pointer_end.shape[0])
    time_steps = int(pointer_end.shape[1])
    if forced_start_pos is None:
        forced_start_pos = [0] * track_count

    result: IntervalBatch = []
    for track in range(track_count):
        position = int(forced_start_pos[track])
        track_result: list[tuple[int, int]] = []
        current_diag = diag_values[track]
        while position < time_steps - 1:
            end = int(pointer_values[track][position])
            if bool(current_diag[position]):
                track_result.append((position, position))
            if end < 0:
                position += 1
            else:
                track_result.append((position, end))
                position = end
        if bool(current_diag[time_steps - 1]):
            track_result.append((time_steps - 1, time_steps - 1))
        result.append(track_result)
    return result


def viterbi_backward_triton(
    score: torch.Tensor,
    noise_score: torch.Tensor,
    forced_start_pos: list[int] | None = None,
) -> IntervalBatch:
    """CUDA上のdense scoreをTritonで厳密Viterbiデコードする。"""
    if score.device.type != "cuda":
        raise ValueError("Triton Semi-CRF decoding requires a CUDA tensor")
    if score.dtype != torch.float32 or noise_score.dtype != torch.float32:
        raise ValueError("Triton Semi-CRF decoding requires float32 scores")
    if score.dim() != 3 or score.shape[0] != score.shape[1]:
        raise ValueError("score must have shape [T, T, B]")

    time_steps = int(score.shape[0])
    track_count = int(score.shape[2])
    if noise_score.shape != (time_steps - 1, track_count):
        raise ValueError("noise_score must have shape [T-1, B]")
    if noise_score.device != score.device:
        raise ValueError("score and noise_score must be on the same CUDA device")
    if time_steps < 2:
        raise ValueError("Triton Semi-CRF decoding requires at least two frames")

    with torch.cuda.device(score.device):
        # 1. 各programが連続した[begin, end]を読むtrack-major配置へ変換する。
        score_by_track_begin_end = score.permute(2, 1, 0).contiguous()
        noise_by_track = noise_score.transpose(0, 1).contiguous()
        q = torch.empty(
            (track_count, time_steps),
            dtype=score.dtype,
            device=score.device,
        )
        pointer_end = torch.empty(
            (track_count, time_steps),
            dtype=torch.int32,
            device=score.device,
        )

        # 2. 1トラックを1programへ割り当て、時刻方向のkernel起動を融合する。
        _viterbi_backward_kernel[(track_count,)](
            score_by_track_begin_end,
            noise_by_track,
            q,
            pointer_end,
            time_steps=time_steps,
            block_time=triton.next_power_of_2(time_steps),
            num_warps=16,
        )

        # 3. singletonの採否とpointerだけをCPUへ転送してtracebackする。
        diag_inclusion = torch.diagonal(
            score_by_track_begin_end,
            dim1=1,
            dim2=2,
        ) > 0
    return _traceback(pointer_end, diag_inclusion, forced_start_pos)
