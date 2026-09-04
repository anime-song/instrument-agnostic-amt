"""Triton log-partition backend used by Semi-CRF training."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _softplus(value):
    """大きな正値でもoverflowしないsoftplusを計算する。"""
    return tl.maximum(value, 0.0) + libdevice.log1p(libdevice.exp(-tl.abs(value)))


@triton.jit
def _forward_kernel(
    score_by_track,
    noise_by_track,
    alpha,
    log_z,
    time_steps,
    block_time: tl.constexpr,
):
    """1 programで1トラックのforward DPを計算する。"""
    track = tl.program_id(0)
    positions = tl.arange(0, block_time)
    score_track_offset = track * time_steps * time_steps
    state_track_offset = track * time_steps
    noise_track_offset = track * (time_steps - 1)

    # 左から右へalphaを計算する。
    first_diag = tl.load(score_by_track + score_track_offset)
    tl.store(alpha + state_track_offset, _softplus(first_diag))
    tl.debug_barrier()

    for end in tl.range(1, time_steps, loop_unroll_factor=1):
        valid_begin = positions < end
        interval_values = tl.load(
            alpha + state_track_offset + positions,
            mask=valid_begin,
            other=-float("inf"),
        ) + tl.load(
            score_by_track + score_track_offset + end * time_steps + positions,
            mask=valid_begin,
            other=0.0,
        )
        skip_value = tl.load(alpha + state_track_offset + end - 1) + tl.load(
            noise_by_track + noise_track_offset + end - 1
        )

        # torch版と同じくintervalのlogsumexp後にskipをlogaddexpする。
        interval_maximum = tl.max(interval_values, axis=0)
        interval_sum = tl.sum(
            libdevice.exp(interval_values - interval_maximum), axis=0
        )
        interval_log_sum = interval_maximum + libdevice.log(interval_sum)
        maximum = tl.maximum(interval_log_sum, skip_value)
        combined = maximum + libdevice.log1p(
            libdevice.exp(-tl.abs(interval_log_sum - skip_value))
        )
        diag = tl.load(
            score_by_track + score_track_offset + end * time_steps + end
        )
        tl.store(
            alpha + state_track_offset + end,
            combined + _softplus(diag),
        )
        tl.debug_barrier()

    tl.store(log_z + track, tl.load(alpha + state_track_offset + time_steps - 1))


@triton.jit
def _backward_kernel(
    score_by_track,
    noise_by_track,
    beta,
    time_steps,
    block_time: tl.constexpr,
):
    """1 programで1トラックのbackward DPを計算する。"""
    track = tl.program_id(0)
    positions = tl.arange(0, block_time)
    score_track_offset = track * time_steps * time_steps
    state_track_offset = track * time_steps
    noise_track_offset = track * (time_steps - 1)

    # 右から左へbetaを計算する。
    last = time_steps - 1
    last_diag = tl.load(
        score_by_track + score_track_offset + last * time_steps + last
    )
    tl.store(beta + state_track_offset + last, _softplus(last_diag))
    tl.debug_barrier()

    for reverse_offset in tl.range(1, time_steps, loop_unroll_factor=1):
        begin = time_steps - reverse_offset - 1
        valid_end = (positions > begin) & (positions < time_steps)
        interval_values = tl.load(
            beta + state_track_offset + positions,
            mask=valid_end,
            other=-float("inf"),
        ) + tl.load(
            score_by_track
            + score_track_offset
            + positions * time_steps
            + begin,
            mask=valid_end,
            other=0.0,
        )
        skip_value = tl.load(beta + state_track_offset + begin + 1) + tl.load(
            noise_by_track + noise_track_offset + begin
        )
        interval_maximum = tl.max(interval_values, axis=0)
        interval_sum = tl.sum(
            libdevice.exp(interval_values - interval_maximum), axis=0
        )
        interval_log_sum = interval_maximum + libdevice.log(interval_sum)
        maximum = tl.maximum(interval_log_sum, skip_value)
        combined = maximum + libdevice.log1p(
            libdevice.exp(-tl.abs(interval_log_sum - skip_value))
        )
        diag = tl.load(
            score_by_track + score_track_offset + begin * time_steps + begin
        )
        tl.store(
            beta + state_track_offset + begin,
            combined + _softplus(diag),
        )
        tl.debug_barrier()


@triton.jit
def _score_marginal_kernel(
    score_by_track,
    alpha,
    beta,
    log_z,
    score_grad_by_track,
    time_steps,
    block_time: tl.constexpr,
):
    """1 programで1終了時刻に対する区間周辺確率を計算する。"""
    track = tl.program_id(0)
    end = tl.program_id(1)
    positions = tl.arange(0, block_time)
    score_track_offset = track * time_steps * time_steps
    state_track_offset = track * time_steps
    valid_begin = positions <= end
    interval_score = tl.load(
        score_by_track + score_track_offset + end * time_steps + positions,
        mask=valid_begin,
        other=0.0,
    )
    log_marginal = (
        tl.load(
            alpha + state_track_offset + positions,
            mask=valid_begin,
            other=0.0,
        )
        + tl.load(beta + state_track_offset + end)
        - tl.load(log_z + track)
        + interval_score
    )

    # singletonはalphaとbetaの両方に含まれるため、その寄与を2回除く。
    log_marginal -= tl.where(
        positions == end,
        2.0 * _softplus(interval_score),
        0.0,
    )
    tl.store(
        score_grad_by_track + score_track_offset + end * time_steps + positions,
        libdevice.exp(log_marginal),
        mask=valid_begin,
    )


@triton.jit
def _noise_marginal_kernel(
    noise_by_track,
    alpha,
    beta,
    log_z,
    noise_grad_by_track,
    time_steps,
    block_time: tl.constexpr,
):
    """1 programで1トラックのskip周辺確率を計算する。"""
    track = tl.program_id(0)
    positions = tl.arange(0, block_time)
    state_track_offset = track * time_steps
    noise_track_offset = track * (time_steps - 1)
    valid_noise = positions < time_steps - 1
    noise_marginal = libdevice.exp(
        tl.load(
            alpha + state_track_offset + positions,
            mask=valid_noise,
            other=0.0,
        )
        + tl.load(
            beta + state_track_offset + positions + 1,
            mask=valid_noise,
            other=0.0,
        )
        + tl.load(
            noise_by_track + noise_track_offset + positions,
            mask=valid_noise,
            other=0.0,
        )
        - tl.load(log_z + track)
    )
    tl.store(
        noise_grad_by_track + noise_track_offset + positions,
        noise_marginal,
        mask=valid_noise,
    )


def _compute_log_z_and_marginals(
    score: torch.Tensor,
    noise_score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton向け配置へ変換し、log Zと解析的勾配を返す。"""
    if score.device.type != "cuda":
        raise ValueError("Triton Semi-CRF loss requires a CUDA tensor")
    if score.dtype != torch.float32 or noise_score.dtype != torch.float32:
        raise ValueError("Triton Semi-CRF loss requires float32 scores")
    if score.dim() != 3 or score.shape[0] != score.shape[1]:
        raise ValueError("score must have shape [T, T, B]")

    time_steps = int(score.shape[0])
    track_count = int(score.shape[2])
    if noise_score.shape != (time_steps - 1, track_count):
        raise ValueError("noise_score must have shape [T-1, B]")
    if noise_score.device != score.device:
        raise ValueError("score and noise_score must be on the same CUDA device")
    if time_steps < 1:
        raise ValueError("Triton Semi-CRF loss requires at least one frame")

    with torch.cuda.device(score.device):
        # 1. 各programが時刻方向を連続アクセスできるtrack-major配置を使う。
        score_by_track = score.permute(2, 0, 1).contiguous()
        noise_by_track = noise_score.transpose(0, 1).contiguous()
        alpha = torch.empty(
            (track_count, time_steps), dtype=score.dtype, device=score.device
        )
        beta = torch.empty_like(alpha)
        log_z = torch.empty(track_count, dtype=score.dtype, device=score.device)
        score_grad_by_track = torch.zeros_like(score_by_track)
        noise_grad_by_track = torch.empty_like(noise_by_track)
        block_time = triton.next_power_of_2(time_steps)

        # 2. DPと周辺確率を時刻単位のPyTorch kernel起動なしで計算する。
        _forward_kernel[(track_count,)](
            score_by_track,
            noise_by_track,
            alpha,
            log_z,
            time_steps,
            block_time=block_time,
            num_warps=16,
        )
        _backward_kernel[(track_count,)](
            score_by_track,
            noise_by_track,
            beta,
            time_steps,
            block_time=block_time,
            num_warps=16,
        )
        _score_marginal_kernel[(track_count, time_steps)](
            score_by_track,
            alpha,
            beta,
            log_z,
            score_grad_by_track,
            time_steps,
            block_time=block_time,
            num_warps=8,
        )
        if time_steps > 1:
            _noise_marginal_kernel[(track_count,)](
                noise_by_track,
                alpha,
                beta,
                log_z,
                noise_grad_by_track,
                time_steps,
                block_time=triton.next_power_of_2(time_steps - 1),
                num_warps=8,
            )
    return log_z, score_grad_by_track, noise_grad_by_track


class _ComputeLogZTriton(torch.autograd.Function):
    """Tritonで計算した周辺確率をPyTorchの学習グラフへ接続する。"""

    @staticmethod
    def forward(ctx, score: torch.Tensor, noise_score: torch.Tensor) -> torch.Tensor:
        log_z, score_grad, noise_grad = _compute_log_z_and_marginals(
            score, noise_score
        )
        ctx.save_for_backward(score_grad, noise_grad)
        return log_z

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        score_grad, noise_grad = ctx.saved_tensors
        scaled_score_grad = score_grad * grad_output[:, None, None]
        scaled_noise_grad = noise_grad * grad_output[:, None]
        return (
            scaled_score_grad.permute(1, 2, 0),
            scaled_noise_grad.transpose(0, 1),
        )


compute_log_z_triton = _ComputeLogZTriton.apply
