from dataclasses import dataclass

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple

"""
Neural semi-CRF for multiple tracks of non-overlapping closed intervals.

Original author: Yujia Yan
Refactor notes:
- Keep the public API and numerical behavior unchanged.
- Remove unused / legacy code paths.
- Add shape comments and helper functions for readability.
"""

Interval = Tuple[int, int]
IntervalBatch = List[List[Interval]]
PitchIntervalBatch = List[IntervalBatch]


@dataclass(frozen=True)
class _FactorizedTrackComponents:
    """Pitch-shared and track-specific projections for one selected-pair chunk."""

    unique_pitch_query: torch.Tensor
    unique_pitch_key: torch.Tensor
    source_inverse: torch.Tensor
    begin_adjustment: torch.Tensor
    end_adjustment: torch.Tensor
    instrument_constant: torch.Tensor
    track_diag: torch.Tensor


@torch.jit.script
def _validate_shapes(score: torch.Tensor, noiseScore: torch.Tensor) -> int:
    """Validate tensor shapes and return sequence length T."""
    assert score.dim() == 3
    assert noiseScore.dim() == 2
    assert score.shape[0] == score.shape[1]
    assert noiseScore.shape[0] == score.shape[0] - 1
    return int(score.shape[0])


@torch.jit.script
def _strictly_lower_triangular_mask(T: int, device: torch.device) -> torch.Tensor:
    """Mask that keeps only valid interval entries (end >= begin)."""
    return torch.ones(T, T, device=device).tril().unsqueeze(-1)


@torch.jit.script
def _viterbi_backward_tensors(
    score: torch.Tensor,
    noise_score: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Viterbi DPをデバイス上で実行し、tracebackに必要な状態を返す。"""
    time_steps = _validate_shapes(score, noise_score)
    track_count = int(score.shape[2])

    q = score.new_zeros(time_steps, track_count)
    pointers = torch.jit.annotate(List[torch.Tensor], [])
    q[time_steps - 1] = score[time_steps - 1, time_steps - 1, :] * (
        score[time_steps - 1, time_steps - 1, :] > 0
    )

    for offset in range(1, time_steps):
        begin = time_steps - offset - 1

        # 区間候補とskip候補を別々に評価し、時刻ごとの一時的なcatを作らない。
        interval_values = q[begin + 1 :, :] + score[begin + 1 :, begin, :]
        best_interval_value, interval_selection = interval_values.max(dim=0)
        skip_value = q[begin + 1, :] + noise_score[begin, :]

        # 旧実装ではskipが候補の先頭にあり、同点時はskipが選ばれていた。
        use_interval = best_interval_value > skip_value
        pointers.append(torch.where(use_interval, interval_selection, -1))
        best_value = torch.where(use_interval, best_interval_value, skip_value)

        singleton_mask = score[begin, begin, :] > 0
        q[begin] = best_value + score[begin, begin, :] * singleton_mask

    # traceback順のpointerをtrack-majorで直接積み、転置コピーを発生させない。
    return (
        torch.stack(pointers, dim=1),
        torch.diagonal(score, dim1=0, dim2=1) > 0,
    )


def viterbiBackward(
    score: torch.Tensor,
    noiseScore: torch.Tensor,
    forcedStartPos: Optional[List[int]] = None,
    *,
    backend: str = "torch",
) -> IntervalBatch:
    """
    左から右へ最良の区間列をデコードする。

    Args:
        score: [T, T, B]。score[end, begin, batch]は[begin, end]のスコア。
        noiseScore: [T-1, B]。位置[t, t+1]をskipするスコア。
        forcedStartPos: 分割デコードで使用するトラックごとの開始位置。
        backend: dense Viterbiを実行するバックエンド。torchまたはtriton。
    """
    if backend not in {"torch", "triton"}:
        raise ValueError("backend must be one of {'torch', 'triton'}")
    if backend == "triton":
        if score.device.type != "cuda":
            raise ValueError("Triton Semi-CRF decoding requires a CUDA tensor")
        try:
            from .semi_crf_triton import viterbi_backward_triton
        except ModuleNotFoundError as exc:
            if exc.name is None or not exc.name.startswith("triton"):
                raise
            raise RuntimeError(
                "Triton Semi-CRF decoding requires the triton package"
            ) from exc
        return viterbi_backward_triton(score, noiseScore, forcedStartPos)

    pointers, diag_inclusion = _viterbi_backward_tensors(score, noiseScore)
    time_steps = int(score.shape[0])
    track_count = int(score.shape[2])

    # TensorのCPUスカラー参照は高コストなため、一括転送後はPython list上を辿る。
    pointer_values = pointers.cpu().tolist()
    diag_values = diag_inclusion.cpu().tolist()

    if forcedStartPos is None:
        forcedStartPos = [0] * track_count

    result: IntervalBatch = []
    for track in range(track_count):
        position = forcedStartPos[track]
        track_result: List[Interval] = []
        current_diag = diag_values[track]

        while position < time_steps - 1:
            selection = int(pointer_values[track][time_steps - position - 2])

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


@torch.jit.script
def viterbi(
    score: torch.Tensor,
    noiseScore: torch.Tensor,
    forcedStartPos: Optional[List[int]] = None,
) -> IntervalBatch:
    """
    Decode intervals from right to left, then reverse them.

    Args:
        score: [T, T, B] where score[end, begin, batch] is the score for [begin, end].
        noiseScore: [T-1, B] score for a non-event transition [t, t+1].
        forcedStartPos: per-batch forced end position for segmented decoding.
    """
    T = _validate_shapes(score, noiseScore)
    nBatch = int(score.shape[2])

    v = score.new_zeros(T, nBatch)
    ptr = []

    v[0] = score[0, 0, :] * (score[0, 0, :] > 0)

    for end in range(1, T):
        subScore = score[end, :end, :]
        candidates = torch.cat(
            [
                v[end - 1 : end, :] + noiseScore[end - 1, :],  # skip
                v[:end, :] + subScore,  # interval [begin, end]
            ],
            dim=0,
        )

        bestValue, selection = candidates.max(dim=0)
        ptr.append(selection - 1)

        singletonMask = score[end, end, :] > 0
        v[end] = bestValue + score[end, end, :] * singletonMask

    ptr = torch.stack(ptr, dim=0).cpu()
    diagInclusion = (torch.diagonal(score, dim1=0, dim2=1) > 0).cpu()

    if forcedStartPos is None:
        forcedStartPos = [T - 1] * nBatch

    result: IntervalBatch = []
    for batchIdx in range(nBatch):
        pos = forcedStartPos[batchIdx]
        batchResult: List[Interval] = []
        curDiag = diagInclusion[batchIdx]

        while pos > 0:
            selection = int(ptr[pos - 1][batchIdx])

            if bool(curDiag[pos]):
                batchResult.append((pos, pos))

            if selection < 0:
                pos -= 1
            else:
                begin = selection
                batchResult.append((begin, pos))
                pos = begin

        if score[0, 0, batchIdx] > 0:
            batchResult.append((0, 0))

        batchResult.reverse()
        result.append(batchResult)

    return result


@torch.jit.script
def computeLogZ(score: torch.Tensor, noiseScore: torch.Tensor) -> torch.Tensor:
    """
    Compute log-partition function log Z for each batch.
    """
    T = _validate_shapes(score, noiseScore)

    v = F.softplus(score[0, 0, :]).unsqueeze(0)

    for end in range(1, T):
        subScore = score[end, :end, :]
        candidates = torch.cat(
            [
                v[end - 1 : end, :] + noiseScore[end - 1, :],  # skip
                v[:end, :] + subScore,  # interval [begin, end]
            ],
            dim=0,
        )
        curValue = candidates.logsumexp(dim=0) + F.softplus(score[end, end, :])
        v = torch.cat([v, curValue.unsqueeze(0)], dim=0)

    return v[-1]


@torch.jit.script
def forward_backward(score: torch.Tensor, noiseScore: torch.Tensor):
    """
    Compute log Z and exact gradients w.r.t. score / noiseScore.

    This version folds the forward and backward recurrences into one batched pass
    by concatenating the original sequence and the time-reversed sequence.
    """
    T = _validate_shapes(score, noiseScore)
    nBatch = int(score.shape[2])

    scoreFlip = torch.flip(score, dims=[0, 1]).transpose(0, 1)
    noiseScoreFlip = torch.flip(noiseScore, dims=(0,))

    scoreFB = torch.cat([score, scoreFlip], dim=-1)  # [T, T, 2B]
    noiseScoreFB = torch.cat([noiseScore, noiseScoreFlip], dim=-1)

    singleScoreSP = F.softplus(torch.diagonal(scoreFB, dim1=0, dim2=1)).transpose(
        -1, -2
    )

    v = score.new_zeros(T, nBatch * 2)
    v[0] = singleScoreSP[0, :]

    for end in range(1, T):
        subScore = scoreFB[end, :end, :]
        v[end] = torch.logaddexp(
            v[end - 1, :] + noiseScoreFB[end - 1, :],  # skip
            torch.logsumexp(v[:end, :] + subScore, dim=0),
        )
        v[end] += singleScoreSP[end, :]

    v, q = torch.chunk(v, 2, dim=-1)
    q = torch.flip(q, dims=(0,))
    logZ = v[-1]

    diag_softplus = F.softplus(torch.diagonal(score, dim1=0, dim2=1))
    grad = v.unsqueeze(0) + (q.unsqueeze(1) - logZ) + score
    grad = grad - 2 * torch.diag_embed(diag_softplus, dim1=0, dim2=1)

    lowerMask = _strictly_lower_triangular_mask(T, grad.device)
    grad = (grad * lowerMask).exp() * lowerMask

    gradNoise = (v[:-1] + q[1:] + noiseScore - logZ).exp()

    return logZ, grad, gradNoise


class ComputeLogZFasterGrad(torch.autograd.Function):
    """
    Custom autograd wrapper around forward_backward().
    """

    @staticmethod
    def forward(ctx, score: torch.Tensor, noiseScore: torch.Tensor) -> torch.Tensor:
        logz, grad, gradNoise = forward_backward(score, noiseScore)
        ctx.save_for_backward(grad, gradNoise)
        return logz

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad, gradNoise = ctx.saved_tensors
        assert grad_output.shape[-1] == grad.shape[-1]
        return grad * grad_output, gradNoise * grad_output


computeLogZFasterGrad = ComputeLogZFasterGrad.apply


def _validate_loss_backend(backend: str, device: torch.device) -> None:
    """Semi-CRF loss backend名と実行deviceの組み合わせを検証する。"""
    if backend not in {"torch", "triton"}:
        raise ValueError("backend must be one of {'torch', 'triton'}")
    if backend == "triton" and device.type != "cuda":
        raise ValueError("Triton Semi-CRF loss requires a CUDA tensor")


def evalPath(
    intervals: IntervalBatch, score: torch.Tensor, noiseScore: torch.Tensor
) -> torch.Tensor:
    """
    Compute the unnormalized path score for each batch item.

    score[end, begin, batch] stores the score of the closed interval [begin, end].
    """
    assert score.dim() == 3
    assert score.shape[0] == score.shape[1]

    T = score.shape[0]
    nBatch = score.shape[2]
    device = score.device

    paddedNoise = F.pad(noiseScore, (0, 0, 1, 0))
    noiseScoreCum = torch.cumsum(paddedNoise, dim=0)

    flatIntervalIndices = [
        batchIdx + begin * nBatch + end * nBatch * T
        for batchIdx, batchIntervals in enumerate(intervals)
        for begin, end in batchIntervals
    ]
    batchIndices = [
        batchIdx
        for batchIdx, batchIntervals in enumerate(intervals)
        for _ in batchIntervals
    ]
    noiseStartIndices = [
        batchIdx + begin * nBatch
        for batchIdx, batchIntervals in enumerate(intervals)
        for begin, end in batchIntervals
    ]
    noiseEndIndices = [
        batchIdx + end * nBatch
        for batchIdx, batchIntervals in enumerate(intervals)
        for begin, end in batchIntervals
    ]

    flatIntervalIndices = torch.tensor(
        flatIntervalIndices, device=device, dtype=torch.long
    )
    batchIndices = torch.tensor(batchIndices, device=device, dtype=torch.long)
    noiseStartIndices = torch.tensor(noiseStartIndices, device=device, dtype=torch.long)
    noiseEndIndices = torch.tensor(noiseEndIndices, device=device, dtype=torch.long)

    gatheredNoiseStart = noiseScoreCum.reshape(-1).gather(0, noiseStartIndices)
    gatheredNoiseEnd = noiseScoreCum.reshape(-1).gather(0, noiseEndIndices)
    gatheredIntervalScores = score.reshape(-1).gather(0, flatIntervalIndices)

    gathered = gatheredIntervalScores - (gatheredNoiseEnd - gatheredNoiseStart)

    result = gathered.new_zeros(nBatch, device=device)
    result = result.scatter_add(-1, batchIndices, gathered)
    result = result + noiseScoreCum[-1, :]

    return result


class NeuralSemiCRFInterval:
    """
    Output layer for multiple tracks of non-overlapping closed intervals.

    Args:
        score:
            [T, T, B], where score[end, begin, batch] is the score of [begin, end].
        noiseScore:
            [T-1, B], where noiseScore[t, batch] is the score of the non-event
            interval [t, t+1].
    """

    def __init__(self, score: torch.Tensor, noiseScore: torch.Tensor):
        self.score = score
        self.noiseScore = noiseScore

    def decode(
        self,
        forcedStartPos: Optional[List[int]] = None,
        forward: bool = False,
        *,
        backend: str = "torch",
    ) -> IntervalBatch:
        """指定バックエンドで最良の区間列をデコードする。"""
        if forward:
            if backend != "torch":
                raise ValueError("forward Viterbi decoding only supports torch")
            return viterbi(self.score, self.noiseScore, forcedStartPos)
        return viterbiBackward(
            self.score,
            self.noiseScore,
            forcedStartPos,
            backend=backend,
        )

    def evalPath(self, intervals: IntervalBatch) -> torch.Tensor:
        """Compute the unnormalized score of a given interval path."""
        return evalPath(intervals, self.score, self.noiseScore)

    def computeLogZ(
        self,
        noBackward: bool = False,
        *,
        backend: str = "torch",
    ) -> torch.Tensor:
        """
        Compute log Z.

        noBackward=True uses the plain scripted DP.
        noBackward=False uses the custom-autograd implementation with faster gradients.
        """
        _validate_loss_backend(backend, self.score.device)
        if noBackward:
            if backend != "torch":
                raise ValueError("noBackward log-partition only supports torch")
            return computeLogZ(self.score, self.noiseScore)
        if backend == "triton":
            try:
                from .semi_crf_loss_triton import compute_log_z_triton
            except ModuleNotFoundError as exc:
                if exc.name is None or not exc.name.startswith("triton"):
                    raise
                raise RuntimeError(
                    "Triton Semi-CRF loss requires the triton package"
                ) from exc
            return compute_log_z_triton(self.score, self.noiseScore)
        return computeLogZFasterGrad(self.score, self.noiseScore)

    def logProb(
        self,
        intervals: IntervalBatch,
        noBackward: bool = False,
        *,
        backend: str = "torch",
    ) -> torch.Tensor:
        """Compute log p(intervals)."""
        return self.evalPath(intervals) - self.computeLogZ(
            noBackward=noBackward,
            backend=backend,
        )


def _flatten_pitch_interval_batch(
    intervals: PitchIntervalBatch,
    *,
    num_pitches: int,
) -> IntervalBatch:
    flat: IntervalBatch = []
    for sample_intervals in intervals:
        if len(sample_intervals) != num_pitches:
            raise ValueError(
                f"Expected {num_pitches} pitch tracks, got {len(sample_intervals)}"
            )
        flat.extend(sample_intervals)
    return flat


def _expand_track_lengths(
    valid_lengths: torch.Tensor | List[int],
    *,
    batch_size: int,
    num_pitches: int,
    device: torch.device,
) -> torch.Tensor:
    lengths = (
        valid_lengths.to(device=device, dtype=torch.long)
        if isinstance(valid_lengths, torch.Tensor)
        else torch.tensor(valid_lengths, device=device, dtype=torch.long)
    )
    if lengths.dim() != 1 or int(lengths.shape[0]) != batch_size:
        raise ValueError(
            f"valid_lengths must have shape [{batch_size}], got {tuple(lengths.shape)}"
        )
    return lengths.unsqueeze(1).expand(batch_size, num_pitches).reshape(-1)


def _expand_forced_start_positions(
    forced_start_pos: Optional[torch.Tensor | List[int] | List[List[int]]],
    *,
    batch_size: int,
    num_pitches: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Convert segmented decode start positions to flat track shape."""
    if forced_start_pos is None:
        return None

    positions = (
        forced_start_pos.to(device=device, dtype=torch.long)
        if isinstance(forced_start_pos, torch.Tensor)
        else torch.tensor(forced_start_pos, device=device, dtype=torch.long)
    )
    if positions.dim() == 1:
        if int(positions.shape[0]) != batch_size * num_pitches:
            raise ValueError(
                "forced_start_pos must have shape "
                f"[{batch_size * num_pitches}] or [{batch_size}, {num_pitches}], "
                f"got {tuple(positions.shape)}"
            )
        return positions.clamp_min(0)
    if positions.dim() == 2 and tuple(positions.shape) == (batch_size, num_pitches):
        return positions.reshape(-1).clamp_min(0)
    raise ValueError(
        "forced_start_pos must have shape "
        f"[{batch_size * num_pitches}] or [{batch_size}, {num_pitches}], "
        f"got {tuple(positions.shape)}"
    )


def _flatten_interval_diag(
    interval_diag: torch.Tensor,
    *,
    batch_size: int,
    time_steps: int,
    num_pitches: int,
) -> torch.Tensor:
    if interval_diag.dim() == 4:
        if int(interval_diag.shape[-1]) != 1:
            raise ValueError(
                "interval_diag with 4 dims must have trailing singleton dim"
            )
        interval_diag = interval_diag.squeeze(-1)
    if interval_diag.shape != (batch_size, time_steps, num_pitches):
        raise ValueError(
            f"interval_diag must have shape [B, T, P], got {tuple(interval_diag.shape)}"
        )
    return interval_diag.permute(1, 0, 2).reshape(
        time_steps,
        batch_size * num_pitches,
    )


def _build_length_scale(
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    length_scaling: str,
) -> Optional[torch.Tensor]:
    if length_scaling == "none":
        return None
    if length_scaling not in {"linear", "sqrt"}:
        raise ValueError("length_scaling must be one of {'linear', 'sqrt', 'none'}")
    end_index = torch.arange(length, device=device)
    begin_index = torch.arange(length, device=device)
    interval_length = (end_index.unsqueeze(1) - begin_index.unsqueeze(0)).abs()
    scale = interval_length.to(dtype=dtype)
    if length_scaling == "sqrt":
        scale = scale.sqrt()
    return scale


def _build_length_penalty(
    length: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    penalty: float,
) -> Optional[torch.Tensor]:
    penalty = float(penalty)
    if penalty == 0.0:
        return None
    end_index = torch.arange(length, device=device)
    begin_index = torch.arange(length, device=device)
    interval_span = (end_index.unsqueeze(1) - begin_index.unsqueeze(0)).clamp_min(0)
    return interval_span.to(dtype=dtype) * penalty


def _build_interval_score(
    interval_query: torch.Tensor,
    interval_key: torch.Tensor,
    interval_diag: torch.Tensor,
    *,
    length_scaling: str,
    length_penalty: float = 0.0,
    note_bias: float = 0.0,
    length_scale: Optional[torch.Tensor] = None,
    length_penalty_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    score = torch.einsum("tnd,snd->stn", interval_query, interval_key).float()

    if float(note_bias) != 0.0:
        score = score + float(note_bias)

    if length_scale is None and length_scaling != "none":
        length_scale = _build_length_scale(
            int(score.shape[0]),
            device=score.device,
            dtype=score.dtype,
            length_scaling=length_scaling,
        )

    if length_scale is not None:
        score = score * length_scale.unsqueeze(-1)

    if length_penalty_matrix is None and float(length_penalty) != 0.0:
        length_penalty_matrix = _build_length_penalty(
            int(score.shape[0]),
            device=score.device,
            dtype=score.dtype,
            penalty=length_penalty,
        )

    if length_penalty_matrix is not None:
        score = score - length_penalty_matrix.unsqueeze(-1)

    diagonal_indices = torch.arange(score.shape[0], device=score.device)
    score[diagonal_indices, diagonal_indices, :] = (
        score[diagonal_indices, diagonal_indices, :] + interval_diag.float()
    )
    return score


def _validate_factorized_interval_inputs(
    pitch_query: torch.Tensor,
    pitch_key: torch.Tensor,
    pitch_diag: torch.Tensor,
    instrument_query: torch.Tensor,
    instrument_key: torch.Tensor,
    instrument_diag: torch.Tensor,
    pair_batch_indices: torch.Tensor,
    pair_instrument_indices: torch.Tensor,
    pair_pitch_indices: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    if pitch_query.shape != pitch_key.shape or pitch_query.dim() != 4:
        raise ValueError(
            "pitch_query and pitch_key must share shape [B, T, P, D]"
        )
    batch_size, time_steps, num_pitches, head_dim = pitch_query.shape
    if pitch_diag.shape != (batch_size, time_steps, num_pitches):
        raise ValueError("pitch_diag must have shape [B, T, P]")
    if instrument_query.shape != instrument_key.shape or instrument_query.dim() != 2:
        raise ValueError(
            "instrument_query and instrument_key must share shape [I, D]"
        )
    num_instruments = int(instrument_query.shape[0])
    if int(instrument_query.shape[1]) != int(head_dim):
        raise ValueError("pitch and instrument projection dimensions must match")
    if instrument_diag.shape != (num_instruments,):
        raise ValueError("instrument_diag must have shape [I]")

    track_count = int(pair_batch_indices.numel())
    for name, indices in (
        ("pair_batch_indices", pair_batch_indices),
        ("pair_instrument_indices", pair_instrument_indices),
        ("pair_pitch_indices", pair_pitch_indices),
    ):
        if indices.shape != (track_count,) or indices.dtype != torch.long:
            raise ValueError(f"{name} must be a 1D torch.long tensor of length N")
        if indices.device != pitch_query.device:
            raise ValueError(f"{name} must be on the projection device")

    projection_tensors = (
        pitch_key,
        pitch_diag,
        instrument_query,
        instrument_key,
        instrument_diag,
    )
    if any(tensor.device != pitch_query.device for tensor in projection_tensors):
        raise ValueError("all factorized projections must be on one device")
    if track_count > 0:
        if bool(
            torch.any(
                (pair_batch_indices < 0) | (pair_batch_indices >= batch_size)
            ).item()
        ):
            raise ValueError("pair_batch_indices contains an invalid batch index")
        if bool(
            torch.any(
                (pair_instrument_indices < 0)
                | (pair_instrument_indices >= num_instruments)
            ).item()
        ):
            raise ValueError(
                "pair_instrument_indices contains an invalid instrument index"
            )
        if bool(
            torch.any(
                (pair_pitch_indices < 0) | (pair_pitch_indices >= num_pitches)
            ).item()
        ):
            raise ValueError("pair_pitch_indices contains an invalid pitch index")
    return (
        int(batch_size),
        int(time_steps),
        int(num_pitches),
        int(head_dim),
        track_count,
    )


def _gather_factorized_track_components(
    pitch_query: torch.Tensor,
    pitch_key: torch.Tensor,
    pitch_diag: torch.Tensor,
    instrument_query: torch.Tensor,
    instrument_key: torch.Tensor,
    instrument_diag: torch.Tensor,
    pair_batch_indices: torch.Tensor,
    pair_instrument_indices: torch.Tensor,
    pair_pitch_indices: torch.Tensor,
    *,
    length: int,
) -> _FactorizedTrackComponents:
    """Gather one copy of each batch/pitch sequence and map selected tracks to it."""

    num_pitches = int(pitch_query.shape[2])
    source_ids = pair_batch_indices * num_pitches + pair_pitch_indices
    unique_source_ids, source_inverse = torch.unique(
        source_ids,
        sorted=True,
        return_inverse=True,
    )
    unique_batch_indices = torch.div(
        unique_source_ids, num_pitches, rounding_mode="floor"
    )
    unique_pitch_indices = unique_source_ids.remainder(num_pitches)

    # Advanced indexing produces [U, T, D].  Time is moved first because the
    # Semi-CRF score convention is [end, begin, track].
    unique_pitch_query = pitch_query[
        unique_batch_indices, :length, unique_pitch_indices, :
    ].transpose(0, 1).contiguous()
    unique_pitch_key = pitch_key[
        unique_batch_indices, :length, unique_pitch_indices, :
    ].transpose(0, 1).contiguous()
    unique_pitch_diag = pitch_diag[
        unique_batch_indices, :length, unique_pitch_indices
    ].transpose(0, 1)
    unique_instrument_indices, instrument_inverse = torch.unique(
        pair_instrument_indices,
        sorted=True,
        return_inverse=True,
    )
    unique_instrument_query = instrument_query.index_select(
        0, unique_instrument_indices
    )
    unique_instrument_key = instrument_key.index_select(
        0, unique_instrument_indices
    )

    # Cross terms have no T x T component.  Compute the small [T, U, J]
    # pitch/instrument grid, then gather only the selected tracks.  In
    # particular, no [T, selected-pair, D] tensor is materialized here.
    begin_grid = torch.einsum(
        "tud,jd->tuj", unique_pitch_query, unique_instrument_key
    ).float()
    end_grid = torch.einsum(
        "jd,tud->tuj", unique_instrument_query, unique_pitch_key
    ).float()
    unique_instrument_constant = (
        unique_instrument_query * unique_instrument_key
    ).sum(dim=-1).float()
    return _FactorizedTrackComponents(
        unique_pitch_query=unique_pitch_query,
        unique_pitch_key=unique_pitch_key,
        source_inverse=source_inverse,
        begin_adjustment=begin_grid[:, source_inverse, instrument_inverse],
        end_adjustment=end_grid[:, source_inverse, instrument_inverse],
        instrument_constant=unique_instrument_constant.index_select(
            0, instrument_inverse
        ),
        track_diag=(
            unique_pitch_diag.index_select(1, source_inverse)
            + instrument_diag.index_select(0, unique_instrument_indices)
            .index_select(0, instrument_inverse)
            .unsqueeze(0)
        ),
    )


def _build_factorized_interval_score(
    pitch_query: torch.Tensor,
    pitch_key: torch.Tensor,
    pitch_diag: torch.Tensor,
    instrument_query: torch.Tensor,
    instrument_key: torch.Tensor,
    instrument_diag: torch.Tensor,
    pair_batch_indices: torch.Tensor,
    pair_instrument_indices: torch.Tensor,
    pair_pitch_indices: torch.Tensor,
    *,
    length: int,
    length_scaling: str,
    length_penalty: float = 0.0,
    note_bias: float = 0.0,
    length_scale: Optional[torch.Tensor] = None,
    length_penalty_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    track_count = int(pair_batch_indices.numel())
    if track_count == 0:
        return pitch_query.new_zeros(
            (int(length), int(length), 0), dtype=torch.float32
        )
    # Projection is factorized before pair selection.  Recombine q/k only for
    # this CRF chunk, then use one high-throughput batched dot product.  A
    # literal four-term expansion saves multiplications for duplicate pitches,
    # but in eager CUDA it must gather and repeatedly update the much larger
    # [begin, end, pair] buffer; that is slower and uses more transient memory.
    # This fused form never recreates the old [T, pair, model-D] input to the
    # projection and remains algebraically identical:
    #   (q_pitch + q_inst) dot (k_pitch + k_inst).
    pair_query = (
        pitch_query[
            pair_batch_indices, : int(length), pair_pitch_indices, :
        ].transpose(0, 1)
        + instrument_query.index_select(0, pair_instrument_indices).unsqueeze(0)
    )
    pair_key = (
        pitch_key[
            pair_batch_indices, : int(length), pair_pitch_indices, :
        ].transpose(0, 1)
        + instrument_key.index_select(0, pair_instrument_indices).unsqueeze(0)
    )
    pair_diag = (
        pitch_diag[
            pair_batch_indices, : int(length), pair_pitch_indices
        ].transpose(0, 1)
        + instrument_diag.index_select(0, pair_instrument_indices).unsqueeze(0)
    )
    return _build_interval_score(
        pair_query,
        pair_key,
        pair_diag,
        length_scaling=length_scaling,
        length_penalty=length_penalty,
        note_bias=note_bias,
        length_scale=length_scale,
        length_penalty_matrix=length_penalty_matrix,
    )


def build_factorized_interval_score(
    pitch_query: torch.Tensor,
    pitch_key: torch.Tensor,
    pitch_diag: torch.Tensor,
    instrument_query: torch.Tensor,
    instrument_key: torch.Tensor,
    instrument_diag: torch.Tensor,
    pair_batch_indices: torch.Tensor,
    pair_instrument_indices: torch.Tensor,
    pair_pitch_indices: torch.Tensor,
    *,
    length_scaling: str = "none",
    length_penalty: float = 0.0,
    note_bias: float = 0.0,
) -> torch.Tensor:
    """Build exact V2 pair scores from separate pitch/instrument projections."""

    _, time_steps, _, _, _ = _validate_factorized_interval_inputs(
        pitch_query,
        pitch_key,
        pitch_diag,
        instrument_query,
        instrument_key,
        instrument_diag,
        pair_batch_indices,
        pair_instrument_indices,
        pair_pitch_indices,
    )
    return _build_factorized_interval_score(
        pitch_query,
        pitch_key,
        pitch_diag,
        instrument_query,
        instrument_key,
        instrument_diag,
        pair_batch_indices,
        pair_instrument_indices,
        pair_pitch_indices,
        length=time_steps,
        length_scaling=length_scaling,
        length_penalty=length_penalty,
        note_bias=note_bias,
    )


def _factorized_track_lengths(
    valid_lengths: torch.Tensor | List[int],
    *,
    batch_size: int,
    pair_batch_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    sample_lengths = (
        valid_lengths.to(device=device, dtype=torch.long)
        if isinstance(valid_lengths, torch.Tensor)
        else torch.tensor(valid_lengths, device=device, dtype=torch.long)
    )
    if sample_lengths.shape != (int(batch_size),):
        raise ValueError(
            f"valid_lengths must have shape [{int(batch_size)}], "
            f"got {tuple(sample_lengths.shape)}"
        )
    return sample_lengths.index_select(0, pair_batch_indices)


def _factorized_forced_start_positions(
    forced_start_pos: Optional[torch.Tensor | List[int]],
    *,
    track_count: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if forced_start_pos is None:
        return None
    positions = (
        forced_start_pos.to(device=device, dtype=torch.long)
        if isinstance(forced_start_pos, torch.Tensor)
        else torch.tensor(forced_start_pos, device=device, dtype=torch.long)
    )
    if positions.shape != (int(track_count),):
        raise ValueError(
            f"forced_start_pos must have shape [{int(track_count)}], "
            f"got {tuple(positions.shape)}"
        )
    return positions.clamp_min(0)


def _sanitize_track_intervals(
    track_intervals: List[Interval],
    *,
    length: int,
) -> List[Interval]:
    if length <= 0 or not track_intervals:
        return []

    sanitized: List[Interval] = []
    for begin, end in sorted(track_intervals):
        begin = max(0, int(begin))
        end = min(int(end), length - 1)
        if end < begin:
            continue
        if sanitized and begin <= sanitized[-1][1]:
            begin = sanitized[-1][1] + 1
        if end < begin:
            continue
        sanitized.append((begin, end))
    return sanitized


def _sanitize_interval_batch(
    intervals: IntervalBatch,
    *,
    length: int,
) -> IntervalBatch:
    return [
        _sanitize_track_intervals(track_intervals, length=length)
        for track_intervals in intervals
    ]


def _zero_noise_score(
    length: int,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.zeros(
        max(0, length - 1),
        batch_size,
        device=device,
    )


def _build_interval_active_cost(
    interval_targets: IntervalBatch,
    *,
    length: int,
    device: torch.device,
    dtype: torch.dtype,
    false_negative_cost: float,
    false_positive_cost: float,
) -> Optional[torch.Tensor]:
    false_negative_cost = float(false_negative_cost)
    false_positive_cost = float(false_positive_cost)
    if false_negative_cost == 0.0 and false_positive_cost == 0.0:
        return None
    if length <= 0 or not interval_targets:
        return None

    batch_size = len(interval_targets)
    active_mask = torch.zeros(batch_size, length, device=device, dtype=dtype)
    for batch_index, track_intervals in enumerate(interval_targets):
        for begin, end in track_intervals:
            begin = max(0, int(begin))
            end = min(int(end), length - 1)
            if end < begin:
                continue
            active_mask[batch_index, begin : end + 1] = 1.0

    active_prefix = F.pad(active_mask.cumsum(dim=1), (1, 0))
    end_prefix = active_prefix[:, 1:].transpose(0, 1).unsqueeze(1)
    begin_prefix = active_prefix[:, :-1].transpose(0, 1).unsqueeze(0)
    true_positive_frames = end_prefix - begin_prefix

    end_index = torch.arange(length, device=device)
    begin_index = torch.arange(length, device=device)
    interval_length = (end_index.unsqueeze(1) - begin_index.unsqueeze(0) + 1).clamp_min(
        0
    )
    interval_length = interval_length.to(dtype=dtype).unsqueeze(-1)

    # Up to a gold-dependent constant, this equals weighted framewise
    # Hamming loss over the predicted active mask.
    cost = (
        float(false_positive_cost) * interval_length
        - float(false_negative_cost + false_positive_cost) * true_positive_frames
    )
    lower_mask = _strictly_lower_triangular_mask(length, device).to(dtype=dtype)
    return cost * lower_mask


def _build_sparse_diag_score(
    interval_query: torch.Tensor,
    interval_key: torch.Tensor,
    interval_diag: torch.Tensor,
    *,
    length_scaling: str,
    note_bias: float,
) -> torch.Tensor:
    if length_scaling not in {"linear", "sqrt", "none"}:
        raise ValueError("length_scaling must be one of {'linear', 'sqrt', 'none'}")

    diag_score = interval_diag.float()
    if length_scaling == "none":
        diag_score = diag_score + (interval_query.float() * interval_key.float()).sum(
            dim=-1
        )
        if float(note_bias) != 0.0:
            diag_score = diag_score + float(note_bias)
    return diag_score


def _select_sparse_candidates(
    candidate_scores: torch.Tensor,
    candidate_ends: torch.Tensor,
    *,
    topk_per_start: int,
    score_threshold: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_scores.dim() != 3:
        raise ValueError("candidate_scores must have shape [T, K, B]")
    if topk_per_start <= 0:
        raise ValueError("topk_per_start must be positive")

    if score_threshold is not None:
        candidate_scores = candidate_scores.masked_fill(
            candidate_scores < float(score_threshold), float("-inf")
        )

    batch_size = int(candidate_scores.shape[2])
    candidate_count = int(candidate_scores.shape[1])
    topk = min(int(topk_per_start), candidate_count)
    if candidate_ends.dim() == 2:
        end_source = candidate_ends.unsqueeze(-1).expand(-1, -1, batch_size)
    elif candidate_ends.dim() == 3:
        end_source = candidate_ends
    else:
        raise ValueError("candidate_ends must have shape [T, K] or [T, K, B]")

    if topk < candidate_count:
        top_scores, top_positions = torch.topk(candidate_scores, k=topk, dim=1)
        top_ends = torch.gather(end_source, 1, top_positions)
        return top_ends.contiguous(), top_scores.contiguous()

    return end_source.contiguous(), candidate_scores.contiguous()


def _build_dense_sparse_candidates(
    score: torch.Tensor,
    *,
    topk_per_start: int,
    score_threshold: Optional[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    length = int(score.shape[0])
    device = score.device
    begin_index = torch.arange(length, device=device).unsqueeze(1)
    end_index = torch.arange(length, device=device).unsqueeze(0)
    valid_mask = end_index > begin_index

    score_by_begin = score.transpose(0, 1)
    score_by_begin = score_by_begin.masked_fill(
        ~valid_mask.unsqueeze(-1), float("-inf")
    )
    return _select_sparse_candidates(
        score_by_begin,
        end_index.expand(length, length),
        topk_per_start=topk_per_start,
        score_threshold=score_threshold,
    )


def _build_banded_sparse_candidates(
    interval_query: torch.Tensor,
    interval_key: torch.Tensor,
    *,
    length_scaling: str,
    length_penalty: float,
    note_bias: float,
    topk_per_start: int,
    score_threshold: Optional[float],
    max_span_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length_scaling not in {"linear", "sqrt", "none"}:
        raise ValueError("length_scaling must be one of {'linear', 'sqrt', 'none'}")

    length = int(interval_query.shape[0])
    batch_size = int(interval_query.shape[1])
    device = interval_query.device
    candidate_count = max(1, min(max(1, int(max_span_frames)), max(1, length - 1)))

    candidate_scores = interval_query.new_full(
        (length, candidate_count, batch_size),
        float("-inf"),
        dtype=torch.float32,
    )
    begin_positions = torch.arange(length, device=device, dtype=torch.long)
    candidate_ends = torch.empty(
        length,
        candidate_count,
        device=device,
        dtype=torch.long,
    )

    for span_index in range(candidate_count):
        offset = span_index + 1
        end_positions = begin_positions + offset
        candidate_ends[:, span_index] = end_positions.clamp_max(max(0, length - 1))
        valid_count = length - offset
        if valid_count <= 0:
            continue

        scores = (
            interval_query[:valid_count].float()
            * interval_key[offset : offset + valid_count].float()
        ).sum(dim=-1)
        if float(note_bias) != 0.0:
            scores = scores + float(note_bias)

        if length_scaling == "linear":
            scores = scores * float(offset)
        elif length_scaling == "sqrt":
            scores = scores * (float(offset) ** 0.5)

        if float(length_penalty) != 0.0:
            scores = scores - float(offset) * float(length_penalty)

        candidate_scores[:valid_count, span_index, :] = scores

    return _select_sparse_candidates(
        candidate_scores,
        candidate_ends,
        topk_per_start=topk_per_start,
        score_threshold=score_threshold,
    )


def _build_factorized_sparse_diag_score(
    pitch_query: torch.Tensor,
    pitch_key: torch.Tensor,
    pitch_diag: torch.Tensor,
    instrument_query: torch.Tensor,
    instrument_key: torch.Tensor,
    instrument_diag: torch.Tensor,
    pair_batch_indices: torch.Tensor,
    pair_instrument_indices: torch.Tensor,
    pair_pitch_indices: torch.Tensor,
    *,
    length: int,
    length_scaling: str,
    note_bias: float,
) -> torch.Tensor:
    if length_scaling not in {"linear", "sqrt", "none"}:
        raise ValueError("length_scaling must be one of {'linear', 'sqrt', 'none'}")
    components = _gather_factorized_track_components(
        pitch_query,
        pitch_key,
        pitch_diag,
        instrument_query,
        instrument_key,
        instrument_diag,
        pair_batch_indices,
        pair_instrument_indices,
        pair_pitch_indices,
        length=int(length),
    )
    diag_score = components.track_diag.float()
    if length_scaling == "none":
        pitch_diag_score = (
            components.unique_pitch_query.float()
            * components.unique_pitch_key.float()
        ).sum(dim=-1)
        diag_score = (
            diag_score
            + pitch_diag_score.index_select(1, components.source_inverse)
            + components.begin_adjustment
            + components.end_adjustment
            + components.instrument_constant.unsqueeze(0)
        )
        if float(note_bias) != 0.0:
            diag_score = diag_score + float(note_bias)
    return diag_score


def _build_factorized_banded_sparse_candidates(
    pitch_query: torch.Tensor,
    pitch_key: torch.Tensor,
    pitch_diag: torch.Tensor,
    instrument_query: torch.Tensor,
    instrument_key: torch.Tensor,
    instrument_diag: torch.Tensor,
    pair_batch_indices: torch.Tensor,
    pair_instrument_indices: torch.Tensor,
    pair_pitch_indices: torch.Tensor,
    *,
    length: int,
    length_scaling: str,
    length_penalty: float,
    note_bias: float,
    topk_per_start: int,
    score_threshold: Optional[float],
    max_span_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if length_scaling not in {"linear", "sqrt", "none"}:
        raise ValueError("length_scaling must be one of {'linear', 'sqrt', 'none'}")

    components = _gather_factorized_track_components(
        pitch_query,
        pitch_key,
        pitch_diag,
        instrument_query,
        instrument_key,
        instrument_diag,
        pair_batch_indices,
        pair_instrument_indices,
        pair_pitch_indices,
        length=int(length),
    )
    track_count = int(pair_batch_indices.numel())
    candidate_count = max(
        1, min(max(1, int(max_span_frames)), max(1, int(length) - 1))
    )
    candidate_scores = pitch_query.new_full(
        (int(length), candidate_count, track_count),
        float("-inf"),
        dtype=torch.float32,
    )
    begin_positions = torch.arange(
        int(length), device=pitch_query.device, dtype=torch.long
    )
    candidate_ends = torch.empty(
        int(length),
        candidate_count,
        device=pitch_query.device,
        dtype=torch.long,
    )
    for span_index in range(candidate_count):
        offset = span_index + 1
        end_positions = begin_positions + offset
        candidate_ends[:, span_index] = end_positions.clamp_max(
            max(0, int(length) - 1)
        )
        valid_count = int(length) - offset
        if valid_count <= 0:
            continue

        pitch_score = (
            components.unique_pitch_query[:valid_count].float()
            * components.unique_pitch_key[offset : offset + valid_count].float()
        ).sum(dim=-1)
        scores = pitch_score.index_select(1, components.source_inverse)
        scores = scores + components.begin_adjustment[:valid_count]
        scores = scores + components.end_adjustment[offset : offset + valid_count]
        scores = scores + components.instrument_constant.unsqueeze(0)
        if float(note_bias) != 0.0:
            scores = scores + float(note_bias)
        if length_scaling == "linear":
            scores = scores * float(offset)
        elif length_scaling == "sqrt":
            scores = scores * (float(offset) ** 0.5)
        if float(length_penalty) != 0.0:
            scores = scores - float(offset) * float(length_penalty)
        candidate_scores[:valid_count, span_index, :] = scores

    return _select_sparse_candidates(
        candidate_scores,
        candidate_ends,
        topk_per_start=topk_per_start,
        score_threshold=score_threshold,
    )


def _viterbi_backward_sparse(
    candidate_ends: torch.Tensor,
    candidate_scores: torch.Tensor,
    diag_score: torch.Tensor,
    noiseScore: torch.Tensor,
    forcedStartPos: Optional[List[int]] = None,
) -> IntervalBatch:
    if candidate_ends.shape != candidate_scores.shape:
        raise ValueError("candidate_ends and candidate_scores must share shape")
    if candidate_scores.dim() != 3:
        raise ValueError("candidate_scores must have shape [T, K, B]")
    if diag_score.dim() != 2:
        raise ValueError("diag_score must have shape [T, B]")

    T = int(diag_score.shape[0])
    nBatch = int(diag_score.shape[1])
    if T <= 0:
        return [[] for _ in range(nBatch)]
    if candidate_scores.shape[0] != T or candidate_scores.shape[2] != nBatch:
        raise ValueError("candidate tensor shapes do not match diag_score")
    if noiseScore.shape != (max(0, T - 1), nBatch):
        raise ValueError("noiseScore must have shape [T - 1, B]")

    q = diag_score.new_zeros(T, nBatch)
    ptr = torch.full(
        (max(0, T - 1), nBatch),
        -1,
        device=diag_score.device,
        dtype=torch.long,
    )
    q[T - 1] = diag_score[T - 1, :] * (diag_score[T - 1, :] > 0)

    for offset in range(1, T):
        t = T - offset - 1
        ends = candidate_ends[t].clamp(0, T - 1)
        scores = candidate_scores[t]
        candidate_values = q.gather(0, ends) + scores
        best_interval_value, best_interval_index = candidate_values.max(dim=0)
        best_interval_end = ends.gather(0, best_interval_index.unsqueeze(0)).squeeze(0)

        skip_value = q[t + 1, :] + noiseScore[t, :]
        use_interval = best_interval_value > skip_value
        best_value = torch.where(use_interval, best_interval_value, skip_value)
        ptr[t, :] = torch.where(
            use_interval & torch.isfinite(best_interval_value),
            best_interval_end,
            ptr[t, :],
        )

        singleton_mask = diag_score[t, :] > 0
        q[t, :] = best_value + diag_score[t, :] * singleton_mask

    ptr_cpu = ptr.cpu()
    diag_inclusion = (diag_score > 0).cpu()

    if forcedStartPos is None:
        forcedStartPos = [0] * nBatch

    result: IntervalBatch = []
    for batchIdx in range(nBatch):
        pos = max(0, min(int(forcedStartPos[batchIdx]), T - 1))
        batchResult: List[Interval] = []
        curDiag = diag_inclusion[:, batchIdx]

        while pos < T - 1:
            selection = int(ptr_cpu[pos, batchIdx])

            if bool(curDiag[pos]):
                batchResult.append((pos, pos))

            if selection < 0:
                pos += 1
            else:
                batchResult.append((pos, selection))
                pos = selection

        if bool(curDiag[T - 1]):
            batchResult.append((T - 1, T - 1))

        result.append(batchResult)

    return result


def compute_flat_interval_loss(
    flat_query: torch.Tensor,
    flat_key: torch.Tensor,
    flat_diag: torch.Tensor,
    interval_targets: IntervalBatch,
    valid_lengths: torch.Tensor | List[int],
    *,
    length_scaling: str = "linear",
    length_penalty: float = 0.0,
    track_batch_size: int = 128,
    false_negative_cost: float = 0.0,
    false_positive_cost: float = 0.0,
    backend: str = "torch",
) -> tuple[torch.Tensor, int, int]:
    """Compute Semi-CRF NLL for an already-selected flat track set."""
    _validate_loss_backend(backend, flat_query.device)
    if flat_query.shape != flat_key.shape:
        raise ValueError(
            "flat_query and flat_key must share the same shape, "
            f"got {tuple(flat_query.shape)} vs {tuple(flat_key.shape)}"
        )
    if flat_query.dim() != 3:
        raise ValueError("flat_query must have shape [T, N, D]")
    time_steps, track_count, _ = flat_query.shape
    if flat_diag.shape != (time_steps, track_count):
        raise ValueError(
            "flat_diag must have shape [T, N], "
            f"got {tuple(flat_diag.shape)} expected {(time_steps, track_count)}"
        )
    if len(interval_targets) != int(track_count):
        raise ValueError(
            f"interval_targets must contain {int(track_count)} tracks, "
            f"got {len(interval_targets)}"
        )

    flat_lengths = (
        valid_lengths.to(device=flat_query.device, dtype=torch.long)
        if isinstance(valid_lengths, torch.Tensor)
        else torch.tensor(valid_lengths, device=flat_query.device, dtype=torch.long)
    )
    if flat_lengths.shape != (track_count,):
        raise ValueError(
            f"valid_lengths must have shape [{int(track_count)}], "
            f"got {tuple(flat_lengths.shape)}"
        )

    total_loss_sum = flat_query.new_zeros(())
    total_tracks = 0
    total_intervals = 0
    chunk_size = max(1, int(track_batch_size))
    length_scale_cache: dict[int, Optional[torch.Tensor]] = {}
    length_penalty_cache: dict[int, Optional[torch.Tensor]] = {}

    unique_lengths = sorted(
        {int(length) for length in flat_lengths.tolist() if int(length) > 0}
    )
    for length in unique_lengths:
        length_scale = length_scale_cache.get(length)
        if length not in length_scale_cache:
            length_scale = _build_length_scale(
                length,
                device=flat_query.device,
                dtype=torch.float32,
                length_scaling=length_scaling,
            )
            length_scale_cache[length] = length_scale
        length_penalty_matrix = length_penalty_cache.get(length)
        if length not in length_penalty_cache:
            length_penalty_matrix = _build_length_penalty(
                length,
                device=flat_query.device,
                dtype=torch.float32,
                penalty=length_penalty,
            )
            length_penalty_cache[length] = length_penalty_matrix
        track_indices = (flat_lengths == length).nonzero(as_tuple=False).flatten()
        if int(track_indices.numel()) == 0:
            continue
        for chunk_indices in track_indices.split(chunk_size):
            score = _build_interval_score(
                flat_query[:length, chunk_indices, :],
                flat_key[:length, chunk_indices, :],
                flat_diag[:length, chunk_indices],
                length_scaling=length_scaling,
                length_penalty=length_penalty,
                length_scale=length_scale,
                length_penalty_matrix=length_penalty_matrix,
            )
            chunk_targets = [
                interval_targets[int(index)] for index in chunk_indices.tolist()
            ]
            chunk_targets = _sanitize_interval_batch(chunk_targets, length=length)
            noise_score = _zero_noise_score(
                length,
                batch_size=int(chunk_indices.numel()),
                device=score.device,
            )
            interval_cost = _build_interval_active_cost(
                chunk_targets,
                length=length,
                device=score.device,
                dtype=score.dtype,
                false_negative_cost=false_negative_cost,
                false_positive_cost=false_positive_cost,
            )
            if interval_cost is None:
                semi_crf = NeuralSemiCRFInterval(score, noise_score)
                chunk_loss = -semi_crf.logProb(chunk_targets, backend=backend)
            else:
                augmented_score = score + interval_cost
                semi_crf = NeuralSemiCRFInterval(augmented_score, noise_score)
                gold_score = evalPath(chunk_targets, score, noise_score)
                chunk_loss = semi_crf.computeLogZ(backend=backend) - gold_score
            total_loss_sum = total_loss_sum + chunk_loss.sum()
            total_tracks += int(chunk_indices.numel())
            total_intervals += sum(len(track) for track in chunk_targets)

    if total_tracks <= 0:
        return flat_query.sum() * 0.0, 0, 0
    return total_loss_sum / float(total_tracks), total_tracks, total_intervals


def compute_factorized_pair_interval_loss(
    pitch_query: torch.Tensor,
    pitch_key: torch.Tensor,
    pitch_diag: torch.Tensor,
    instrument_query: torch.Tensor,
    instrument_key: torch.Tensor,
    instrument_diag: torch.Tensor,
    pair_batch_indices: torch.Tensor,
    pair_instrument_indices: torch.Tensor,
    pair_pitch_indices: torch.Tensor,
    interval_targets: IntervalBatch,
    valid_lengths: torch.Tensor | List[int],
    *,
    length_scaling: str = "linear",
    length_penalty: float = 0.0,
    track_batch_size: int = 128,
    false_negative_cost: float = 0.0,
    false_positive_cost: float = 0.0,
    backend: str = "torch",
) -> tuple[torch.Tensor, int, int]:
    """Compute V2 Semi-CRF NLL from additive pitch/instrument projections."""
    _validate_loss_backend(backend, pitch_query.device)

    batch_size, _, _, _, track_count = (
        _validate_factorized_interval_inputs(
            pitch_query,
            pitch_key,
            pitch_diag,
            instrument_query,
            instrument_key,
            instrument_diag,
            pair_batch_indices,
            pair_instrument_indices,
            pair_pitch_indices,
        )
    )
    if len(interval_targets) != track_count:
        raise ValueError(
            f"interval_targets must contain {track_count} tracks, "
            f"got {len(interval_targets)}"
        )
    flat_lengths = _factorized_track_lengths(
        valid_lengths,
        batch_size=batch_size,
        pair_batch_indices=pair_batch_indices,
        device=pitch_query.device,
    )

    total_loss_sum = pitch_query.new_zeros(())
    total_tracks = 0
    total_intervals = 0
    chunk_size = max(1, int(track_batch_size))
    length_scale_cache: dict[int, Optional[torch.Tensor]] = {}
    length_penalty_cache: dict[int, Optional[torch.Tensor]] = {}

    unique_lengths = sorted(
        {int(length) for length in flat_lengths.tolist() if int(length) > 0}
    )
    for length in unique_lengths:
        length_scale = length_scale_cache.get(length)
        if length not in length_scale_cache:
            length_scale = _build_length_scale(
                length,
                device=pitch_query.device,
                dtype=torch.float32,
                length_scaling=length_scaling,
            )
            length_scale_cache[length] = length_scale
        length_penalty_matrix = length_penalty_cache.get(length)
        if length not in length_penalty_cache:
            length_penalty_matrix = _build_length_penalty(
                length,
                device=pitch_query.device,
                dtype=torch.float32,
                penalty=length_penalty,
            )
            length_penalty_cache[length] = length_penalty_matrix
        track_indices = (flat_lengths == length).nonzero(as_tuple=False).flatten()
        if int(track_indices.numel()) == 0:
            continue
        for chunk_indices in track_indices.split(chunk_size):
            score = _build_factorized_interval_score(
                pitch_query,
                pitch_key,
                pitch_diag,
                instrument_query,
                instrument_key,
                instrument_diag,
                pair_batch_indices.index_select(0, chunk_indices),
                pair_instrument_indices.index_select(0, chunk_indices),
                pair_pitch_indices.index_select(0, chunk_indices),
                length=length,
                length_scaling=length_scaling,
                length_penalty=length_penalty,
                length_scale=length_scale,
                length_penalty_matrix=length_penalty_matrix,
            )
            chunk_targets = [
                interval_targets[int(index)] for index in chunk_indices.tolist()
            ]
            chunk_targets = _sanitize_interval_batch(chunk_targets, length=length)
            noise_score = _zero_noise_score(
                length,
                batch_size=int(chunk_indices.numel()),
                device=score.device,
            )
            interval_cost = _build_interval_active_cost(
                chunk_targets,
                length=length,
                device=score.device,
                dtype=score.dtype,
                false_negative_cost=false_negative_cost,
                false_positive_cost=false_positive_cost,
            )
            if interval_cost is None:
                semi_crf = NeuralSemiCRFInterval(score, noise_score)
                chunk_loss = -semi_crf.logProb(chunk_targets, backend=backend)
            else:
                augmented_score = score + interval_cost
                semi_crf = NeuralSemiCRFInterval(augmented_score, noise_score)
                gold_score = evalPath(chunk_targets, score, noise_score)
                chunk_loss = semi_crf.computeLogZ(backend=backend) - gold_score
            total_loss_sum = total_loss_sum + chunk_loss.sum()
            total_tracks += int(chunk_indices.numel())
            total_intervals += sum(len(track) for track in chunk_targets)

    if total_tracks <= 0:
        return (pitch_query.sum() + instrument_query.sum()) * 0.0, 0, 0
    return total_loss_sum / float(total_tracks), total_tracks, total_intervals


def compute_pitch_interval_loss(
    interval_query: torch.Tensor,
    interval_key: torch.Tensor,
    interval_diag: torch.Tensor,
    interval_targets: PitchIntervalBatch,
    valid_lengths: torch.Tensor | List[int],
    *,
    length_scaling: str = "linear",
    length_penalty: float = 0.0,
    track_batch_size: int = 128,
    false_negative_cost: float = 0.0,
    false_positive_cost: float = 0.0,
    backend: str = "torch",
) -> tuple[torch.Tensor, int, int]:
    """
    Compute pitch-wise semi-CRF NLL from interval query/key features.

    Args:
        interval_query: [B, T, P, D]
        interval_key: [B, T, P, D]
        interval_diag: [B, T, P]
        interval_targets: nested intervals as [B][P][(begin, end), ...]
        valid_lengths: valid frame count per batch item, shape [B]
        track_batch_size: chunk size over flattened B*P tracks.
        false_negative_cost:
            Cost added during training for active frames that are left uncovered.
        false_positive_cost:
            Cost added during training for silent frames covered by an interval.
        backend: log-partitionを計算するtorchまたはtriton backend。
    """
    _validate_loss_backend(backend, interval_query.device)
    if interval_query.shape != interval_key.shape:
        raise ValueError(
            "interval_query and interval_key must share the same shape, "
            f"got {tuple(interval_query.shape)} vs {tuple(interval_key.shape)}"
        )
    if interval_query.dim() != 4:
        raise ValueError("interval_query must have shape [B, T, P, D]")

    batch_size, time_steps, num_pitches, feature_dim = interval_query.shape
    del feature_dim
    flat_diag = _flatten_interval_diag(
        interval_diag,
        batch_size=int(batch_size),
        time_steps=int(time_steps),
        num_pitches=int(num_pitches),
    )

    flat_targets = _flatten_pitch_interval_batch(
        interval_targets,
        num_pitches=int(num_pitches),
    )
    flat_lengths = _expand_track_lengths(
        valid_lengths,
        batch_size=int(batch_size),
        num_pitches=int(num_pitches),
        device=interval_query.device,
    )

    flat_query = interval_query.permute(1, 0, 2, 3).reshape(
        time_steps,
        batch_size * num_pitches,
        -1,
    )
    flat_key = interval_key.permute(1, 0, 2, 3).reshape(
        time_steps,
        batch_size * num_pitches,
        -1,
    )

    total_loss_sum = interval_query.new_zeros(())
    total_tracks = 0
    total_intervals = 0
    chunk_size = max(1, int(track_batch_size))
    length_scale_cache: dict[int, Optional[torch.Tensor]] = {}
    length_penalty_cache: dict[int, Optional[torch.Tensor]] = {}

    unique_lengths = sorted(
        {int(length) for length in flat_lengths.tolist() if int(length) > 0}
    )
    for length in unique_lengths:
        length_scale = length_scale_cache.get(length)
        if length not in length_scale_cache:
            length_scale = _build_length_scale(
                length,
                device=interval_query.device,
                dtype=torch.float32,
                length_scaling=length_scaling,
            )
            length_scale_cache[length] = length_scale
        length_penalty_matrix = length_penalty_cache.get(length)
        if length not in length_penalty_cache:
            length_penalty_matrix = _build_length_penalty(
                length,
                device=interval_query.device,
                dtype=torch.float32,
                penalty=length_penalty,
            )
            length_penalty_cache[length] = length_penalty_matrix
        track_indices = (flat_lengths == length).nonzero(as_tuple=False).flatten()
        if int(track_indices.numel()) == 0:
            continue
        for chunk_indices in track_indices.split(chunk_size):
            score = _build_interval_score(
                flat_query[:length, chunk_indices, :],
                flat_key[:length, chunk_indices, :],
                flat_diag[:length, chunk_indices],
                length_scaling=length_scaling,
                length_penalty=length_penalty,
                length_scale=length_scale,
                length_penalty_matrix=length_penalty_matrix,
            )
            chunk_targets = [
                flat_targets[int(index)] for index in chunk_indices.tolist()
            ]
            chunk_targets = _sanitize_interval_batch(chunk_targets, length=length)
            noise_score = _zero_noise_score(
                length,
                batch_size=int(chunk_indices.numel()),
                device=score.device,
            )
            interval_cost = _build_interval_active_cost(
                chunk_targets,
                length=length,
                device=score.device,
                dtype=score.dtype,
                false_negative_cost=false_negative_cost,
                false_positive_cost=false_positive_cost,
            )
            if interval_cost is None:
                semi_crf = NeuralSemiCRFInterval(score, noise_score)
                chunk_loss = -semi_crf.logProb(chunk_targets, backend=backend)
            else:
                augmented_score = score + interval_cost
                semi_crf = NeuralSemiCRFInterval(augmented_score, noise_score)
                gold_score = evalPath(chunk_targets, score, noise_score)
                chunk_loss = semi_crf.computeLogZ(backend=backend) - gold_score
            total_loss_sum = total_loss_sum + chunk_loss.sum()
            total_tracks += int(chunk_indices.numel())
            total_intervals += sum(len(track) for track in chunk_targets)

    if total_tracks <= 0:
        return interval_query.sum() * 0.0, 0, 0
    return total_loss_sum / float(total_tracks), total_tracks, total_intervals


@torch.no_grad()
def decode_pitch_intervals_sparse(
    interval_query: torch.Tensor,
    interval_key: torch.Tensor,
    interval_diag: torch.Tensor,
    valid_lengths: torch.Tensor | List[int],
    *,
    length_scaling: str = "linear",
    length_penalty: float = 0.0,
    note_bias: float = 0.0,
    track_batch_size: int = 128,
    forced_start_pos: Optional[torch.Tensor | List[int] | List[List[int]]] = None,
    sparse_topk_per_start: int = 16,
    sparse_score_threshold: Optional[float] = None,
    sparse_max_span_frames: Optional[int] = None,
) -> PitchIntervalBatch:
    """
    Decode pitch-wise intervals using a sparse candidate set per start frame.

    If sparse_max_span_frames is set, only that local band is scored. Otherwise
    dense interval scores are used only to select the top-k candidates before the
    sparse Viterbi pass.
    """
    if interval_query.shape != interval_key.shape:
        raise ValueError(
            "interval_query and interval_key must share the same shape, "
            f"got {tuple(interval_query.shape)} vs {tuple(interval_key.shape)}"
        )
    if interval_query.dim() != 4:
        raise ValueError("interval_query must have shape [B, T, P, D]")
    if sparse_topk_per_start <= 0:
        raise ValueError("sparse_topk_per_start must be positive")
    if sparse_max_span_frames is not None and int(sparse_max_span_frames) <= 0:
        raise ValueError("sparse_max_span_frames must be positive when set")

    batch_size, time_steps, num_pitches, _ = interval_query.shape
    flat_diag = _flatten_interval_diag(
        interval_diag,
        batch_size=int(batch_size),
        time_steps=int(time_steps),
        num_pitches=int(num_pitches),
    )
    flat_lengths = _expand_track_lengths(
        valid_lengths,
        batch_size=int(batch_size),
        num_pitches=int(num_pitches),
        device=interval_query.device,
    )
    flat_forced_start_pos = _expand_forced_start_positions(
        forced_start_pos,
        batch_size=int(batch_size),
        num_pitches=int(num_pitches),
        device=interval_query.device,
    )
    flat_query = interval_query.permute(1, 0, 2, 3).reshape(
        time_steps,
        batch_size * num_pitches,
        -1,
    )
    flat_key = interval_key.permute(1, 0, 2, 3).reshape(
        time_steps,
        batch_size * num_pitches,
        -1,
    )

    decoded_flat: IntervalBatch = [[] for _ in range(int(batch_size * num_pitches))]
    chunk_size = max(1, int(track_batch_size))
    length_scale_cache: dict[int, Optional[torch.Tensor]] = {}
    length_penalty_cache: dict[int, Optional[torch.Tensor]] = {}
    max_span_frames = (
        None if sparse_max_span_frames is None else max(1, int(sparse_max_span_frames))
    )
    unique_lengths = sorted(
        {int(length) for length in flat_lengths.tolist() if int(length) > 0}
    )
    for length in unique_lengths:
        track_indices = (flat_lengths == length).nonzero(as_tuple=False).flatten()
        if int(track_indices.numel()) == 0:
            continue

        length_scale = None
        length_penalty_matrix = None
        if max_span_frames is None:
            length_scale = length_scale_cache.get(length)
            if length not in length_scale_cache:
                length_scale = _build_length_scale(
                    length,
                    device=interval_query.device,
                    dtype=torch.float32,
                    length_scaling=length_scaling,
                )
                length_scale_cache[length] = length_scale
            length_penalty_matrix = length_penalty_cache.get(length)
            if length not in length_penalty_cache:
                length_penalty_matrix = _build_length_penalty(
                    length,
                    device=interval_query.device,
                    dtype=torch.float32,
                    penalty=length_penalty,
                )
                length_penalty_cache[length] = length_penalty_matrix

        for chunk_indices in track_indices.split(chunk_size):
            chunk_query = flat_query[:length, chunk_indices, :]
            chunk_key = flat_key[:length, chunk_indices, :]
            chunk_diag = flat_diag[:length, chunk_indices]
            if max_span_frames is None:
                score = _build_interval_score(
                    chunk_query,
                    chunk_key,
                    chunk_diag,
                    length_scaling=length_scaling,
                    length_penalty=length_penalty,
                    note_bias=note_bias,
                    length_scale=length_scale,
                    length_penalty_matrix=length_penalty_matrix,
                )
                candidate_ends, candidate_scores = _build_dense_sparse_candidates(
                    score,
                    topk_per_start=int(sparse_topk_per_start),
                    score_threshold=sparse_score_threshold,
                )
                diag_score = (
                    torch.diagonal(score, dim1=0, dim2=1).transpose(0, 1).contiguous()
                )
            else:
                candidate_ends, candidate_scores = _build_banded_sparse_candidates(
                    chunk_query,
                    chunk_key,
                    length_scaling=length_scaling,
                    length_penalty=length_penalty,
                    note_bias=note_bias,
                    topk_per_start=int(sparse_topk_per_start),
                    score_threshold=sparse_score_threshold,
                    max_span_frames=max_span_frames,
                )
                diag_score = _build_sparse_diag_score(
                    chunk_query,
                    chunk_key,
                    chunk_diag,
                    length_scaling=length_scaling,
                    note_bias=note_bias,
                )

            if flat_forced_start_pos is not None:
                chunk_forced_start_pos = torch.clamp(
                    flat_forced_start_pos.index_select(0, chunk_indices),
                    min=0,
                    max=max(0, length - 1),
                ).tolist()
            else:
                chunk_forced_start_pos = None

            decoded_chunk = _viterbi_backward_sparse(
                candidate_ends,
                candidate_scores,
                diag_score,
                _zero_noise_score(
                    length,
                    batch_size=int(chunk_indices.numel()),
                    device=diag_score.device,
                ),
                forcedStartPos=chunk_forced_start_pos,
            )
            for flat_index, intervals in zip(chunk_indices.tolist(), decoded_chunk):
                decoded_flat[int(flat_index)] = intervals

    return [
        decoded_flat[batch_index * num_pitches : (batch_index + 1) * num_pitches]
        for batch_index in range(int(batch_size))
    ]


@torch.no_grad()
def decode_pitch_intervals(
    interval_query: torch.Tensor,
    interval_key: torch.Tensor,
    interval_diag: torch.Tensor,
    valid_lengths: torch.Tensor | List[int],
    *,
    length_scaling: str = "linear",
    length_penalty: float = 0.0,
    note_bias: float = 0.0,
    track_batch_size: int = 128,
    forced_start_pos: Optional[torch.Tensor | List[int] | List[List[int]]] = None,
    backend: str = "torch",
) -> PitchIntervalBatch:
    """
    Decode pitch-wise best non-overlapping intervals from interval query/key features.
    """
    if interval_query.shape != interval_key.shape:
        raise ValueError(
            "interval_query and interval_key must share the same shape, "
            f"got {tuple(interval_query.shape)} vs {tuple(interval_key.shape)}"
        )
    if interval_query.dim() != 4:
        raise ValueError("interval_query must have shape [B, T, P, D]")

    batch_size, time_steps, num_pitches, _ = interval_query.shape
    flat_diag = _flatten_interval_diag(
        interval_diag,
        batch_size=int(batch_size),
        time_steps=int(time_steps),
        num_pitches=int(num_pitches),
    )
    flat_lengths = _expand_track_lengths(
        valid_lengths,
        batch_size=int(batch_size),
        num_pitches=int(num_pitches),
        device=interval_query.device,
    )
    flat_forced_start_pos = _expand_forced_start_positions(
        forced_start_pos,
        batch_size=int(batch_size),
        num_pitches=int(num_pitches),
        device=interval_query.device,
    )
    flat_query = interval_query.permute(1, 0, 2, 3).reshape(
        time_steps,
        batch_size * num_pitches,
        -1,
    )
    flat_key = interval_key.permute(1, 0, 2, 3).reshape(
        time_steps,
        batch_size * num_pitches,
        -1,
    )

    decoded_flat: IntervalBatch = [[] for _ in range(int(batch_size * num_pitches))]
    chunk_size = max(1, int(track_batch_size))
    length_scale_cache: dict[int, Optional[torch.Tensor]] = {}
    length_penalty_cache: dict[int, Optional[torch.Tensor]] = {}
    unique_lengths = sorted(
        {int(length) for length in flat_lengths.tolist() if int(length) > 0}
    )
    for length in unique_lengths:
        length_scale = length_scale_cache.get(length)
        if length not in length_scale_cache:
            length_scale = _build_length_scale(
                length,
                device=interval_query.device,
                dtype=torch.float32,
                length_scaling=length_scaling,
            )
            length_scale_cache[length] = length_scale
        length_penalty_matrix = length_penalty_cache.get(length)
        if length not in length_penalty_cache:
            length_penalty_matrix = _build_length_penalty(
                length,
                device=interval_query.device,
                dtype=torch.float32,
                penalty=length_penalty,
            )
            length_penalty_cache[length] = length_penalty_matrix
        track_indices = (flat_lengths == length).nonzero(as_tuple=False).flatten()
        if int(track_indices.numel()) == 0:
            continue
        for chunk_indices in track_indices.split(chunk_size):
            score = _build_interval_score(
                flat_query[:length, chunk_indices, :],
                flat_key[:length, chunk_indices, :],
                flat_diag[:length, chunk_indices],
                length_scaling=length_scaling,
                length_penalty=length_penalty,
                note_bias=note_bias,
                length_scale=length_scale,
                length_penalty_matrix=length_penalty_matrix,
            )
            semi_crf = NeuralSemiCRFInterval(
                score,
                _zero_noise_score(
                    length,
                    batch_size=int(chunk_indices.numel()),
                    device=score.device,
                ),
            )
            if flat_forced_start_pos is not None:
                chunk_forced_start_pos = torch.clamp(
                    flat_forced_start_pos.index_select(0, chunk_indices),
                    min=0,
                    max=max(0, length - 1),
                ).tolist()
            else:
                chunk_forced_start_pos = None
            decoded_chunk = semi_crf.decode(
                forcedStartPos=chunk_forced_start_pos,
                backend=backend,
            )
            for flat_index, intervals in zip(chunk_indices.tolist(), decoded_chunk):
                decoded_flat[int(flat_index)] = intervals

    return [
        decoded_flat[batch_index * num_pitches : (batch_index + 1) * num_pitches]
        for batch_index in range(int(batch_size))
    ]


@torch.no_grad()
def decode_factorized_pair_intervals(
    pitch_query: torch.Tensor,
    pitch_key: torch.Tensor,
    pitch_diag: torch.Tensor,
    instrument_query: torch.Tensor,
    instrument_key: torch.Tensor,
    instrument_diag: torch.Tensor,
    pair_batch_indices: torch.Tensor,
    pair_instrument_indices: torch.Tensor,
    pair_pitch_indices: torch.Tensor,
    valid_lengths: torch.Tensor | List[int],
    *,
    length_scaling: str = "linear",
    length_penalty: float = 0.0,
    note_bias: float = 0.0,
    track_batch_size: int = 128,
    forced_start_pos: Optional[torch.Tensor | List[int]] = None,
    backend: str = "torch",
) -> IntervalBatch:
    """Decode independent selected V2 tracks from factorized projections."""

    batch_size, _, _, _, track_count = (
        _validate_factorized_interval_inputs(
            pitch_query,
            pitch_key,
            pitch_diag,
            instrument_query,
            instrument_key,
            instrument_diag,
            pair_batch_indices,
            pair_instrument_indices,
            pair_pitch_indices,
        )
    )
    flat_lengths = _factorized_track_lengths(
        valid_lengths,
        batch_size=batch_size,
        pair_batch_indices=pair_batch_indices,
        device=pitch_query.device,
    )
    flat_forced_start_pos = _factorized_forced_start_positions(
        forced_start_pos,
        track_count=track_count,
        device=pitch_query.device,
    )

    decoded_flat: IntervalBatch = [[] for _ in range(track_count)]
    chunk_size = max(1, int(track_batch_size))
    length_scale_cache: dict[int, Optional[torch.Tensor]] = {}
    length_penalty_cache: dict[int, Optional[torch.Tensor]] = {}
    unique_lengths = sorted(
        {int(length) for length in flat_lengths.tolist() if int(length) > 0}
    )
    for length in unique_lengths:
        length_scale = length_scale_cache.get(length)
        if length not in length_scale_cache:
            length_scale = _build_length_scale(
                length,
                device=pitch_query.device,
                dtype=torch.float32,
                length_scaling=length_scaling,
            )
            length_scale_cache[length] = length_scale
        length_penalty_matrix = length_penalty_cache.get(length)
        if length not in length_penalty_cache:
            length_penalty_matrix = _build_length_penalty(
                length,
                device=pitch_query.device,
                dtype=torch.float32,
                penalty=length_penalty,
            )
            length_penalty_cache[length] = length_penalty_matrix
        track_indices = (flat_lengths == length).nonzero(as_tuple=False).flatten()
        for chunk_indices in track_indices.split(chunk_size):
            score = _build_factorized_interval_score(
                pitch_query,
                pitch_key,
                pitch_diag,
                instrument_query,
                instrument_key,
                instrument_diag,
                pair_batch_indices.index_select(0, chunk_indices),
                pair_instrument_indices.index_select(0, chunk_indices),
                pair_pitch_indices.index_select(0, chunk_indices),
                length=length,
                length_scaling=length_scaling,
                length_penalty=length_penalty,
                note_bias=note_bias,
                length_scale=length_scale,
                length_penalty_matrix=length_penalty_matrix,
            )
            semi_crf = NeuralSemiCRFInterval(
                score,
                _zero_noise_score(
                    length,
                    batch_size=int(chunk_indices.numel()),
                    device=score.device,
                ),
            )
            chunk_forced_start_pos = (
                torch.clamp(
                    flat_forced_start_pos.index_select(0, chunk_indices),
                    min=0,
                    max=max(0, length - 1),
                ).tolist()
                if flat_forced_start_pos is not None
                else None
            )
            decoded_chunk = semi_crf.decode(
                forcedStartPos=chunk_forced_start_pos,
                backend=backend,
            )
            for flat_index, intervals in zip(chunk_indices.tolist(), decoded_chunk):
                decoded_flat[int(flat_index)] = intervals
    return decoded_flat


@torch.no_grad()
def decode_factorized_pair_intervals_sparse(
    pitch_query: torch.Tensor,
    pitch_key: torch.Tensor,
    pitch_diag: torch.Tensor,
    instrument_query: torch.Tensor,
    instrument_key: torch.Tensor,
    instrument_diag: torch.Tensor,
    pair_batch_indices: torch.Tensor,
    pair_instrument_indices: torch.Tensor,
    pair_pitch_indices: torch.Tensor,
    valid_lengths: torch.Tensor | List[int],
    *,
    length_scaling: str = "linear",
    length_penalty: float = 0.0,
    note_bias: float = 0.0,
    track_batch_size: int = 128,
    forced_start_pos: Optional[torch.Tensor | List[int]] = None,
    sparse_topk_per_start: int = 16,
    sparse_score_threshold: Optional[float] = None,
    sparse_max_span_frames: Optional[int] = None,
) -> IntervalBatch:
    """Sparse V2 decode from separate pitch/instrument projections."""

    if sparse_topk_per_start <= 0:
        raise ValueError("sparse_topk_per_start must be positive")
    if sparse_max_span_frames is not None and int(sparse_max_span_frames) <= 0:
        raise ValueError("sparse_max_span_frames must be positive when set")
    batch_size, _, num_pitches, _, track_count = (
        _validate_factorized_interval_inputs(
            pitch_query,
            pitch_key,
            pitch_diag,
            instrument_query,
            instrument_key,
            instrument_diag,
            pair_batch_indices,
            pair_instrument_indices,
            pair_pitch_indices,
        )
    )
    flat_lengths = _factorized_track_lengths(
        valid_lengths,
        batch_size=batch_size,
        pair_batch_indices=pair_batch_indices,
        device=pitch_query.device,
    )
    flat_forced_start_pos = _factorized_forced_start_positions(
        forced_start_pos,
        track_count=track_count,
        device=pitch_query.device,
    )

    decoded_flat: IntervalBatch = [[] for _ in range(track_count)]
    chunk_size = max(1, int(track_batch_size))
    length_scale_cache: dict[int, Optional[torch.Tensor]] = {}
    length_penalty_cache: dict[int, Optional[torch.Tensor]] = {}
    source_ids = pair_batch_indices * int(num_pitches) + pair_pitch_indices
    max_span_frames = (
        None if sparse_max_span_frames is None else int(sparse_max_span_frames)
    )
    unique_lengths = sorted(
        {int(length) for length in flat_lengths.tolist() if int(length) > 0}
    )
    for length in unique_lengths:
        length_scale = None
        length_penalty_matrix = None
        if max_span_frames is None:
            length_scale = length_scale_cache.get(length)
            if length not in length_scale_cache:
                length_scale = _build_length_scale(
                    length,
                    device=pitch_query.device,
                    dtype=torch.float32,
                    length_scaling=length_scaling,
                )
                length_scale_cache[length] = length_scale
            length_penalty_matrix = length_penalty_cache.get(length)
            if length not in length_penalty_cache:
                length_penalty_matrix = _build_length_penalty(
                    length,
                    device=pitch_query.device,
                    dtype=torch.float32,
                    penalty=length_penalty,
                )
                length_penalty_cache[length] = length_penalty_matrix

        track_indices = (flat_lengths == length).nonzero(as_tuple=False).flatten()
        if max_span_frames is not None:
            # The banded sparse scorer evaluates the decomposed pitch term
            # directly, so keep identical pitches together within each chunk.
            track_indices = track_indices.index_select(
                0,
                torch.argsort(source_ids.index_select(0, track_indices)),
            )
        for chunk_indices in track_indices.split(chunk_size):
            chunk_batch_indices = pair_batch_indices.index_select(0, chunk_indices)
            chunk_instrument_indices = pair_instrument_indices.index_select(
                0, chunk_indices
            )
            chunk_pitch_indices = pair_pitch_indices.index_select(0, chunk_indices)
            if max_span_frames is None:
                score = _build_factorized_interval_score(
                    pitch_query,
                    pitch_key,
                    pitch_diag,
                    instrument_query,
                    instrument_key,
                    instrument_diag,
                    chunk_batch_indices,
                    chunk_instrument_indices,
                    chunk_pitch_indices,
                    length=length,
                    length_scaling=length_scaling,
                    length_penalty=length_penalty,
                    note_bias=note_bias,
                    length_scale=length_scale,
                    length_penalty_matrix=length_penalty_matrix,
                )
                candidate_ends, candidate_scores = _build_dense_sparse_candidates(
                    score,
                    topk_per_start=int(sparse_topk_per_start),
                    score_threshold=sparse_score_threshold,
                )
                diag_score = (
                    torch.diagonal(score, dim1=0, dim2=1)
                    .transpose(0, 1)
                    .contiguous()
                )
            else:
                candidate_ends, candidate_scores = (
                    _build_factorized_banded_sparse_candidates(
                        pitch_query,
                        pitch_key,
                        pitch_diag,
                        instrument_query,
                        instrument_key,
                        instrument_diag,
                        chunk_batch_indices,
                        chunk_instrument_indices,
                        chunk_pitch_indices,
                        length=length,
                        length_scaling=length_scaling,
                        length_penalty=length_penalty,
                        note_bias=note_bias,
                        topk_per_start=int(sparse_topk_per_start),
                        score_threshold=sparse_score_threshold,
                        max_span_frames=max_span_frames,
                    )
                )
                diag_score = _build_factorized_sparse_diag_score(
                    pitch_query,
                    pitch_key,
                    pitch_diag,
                    instrument_query,
                    instrument_key,
                    instrument_diag,
                    chunk_batch_indices,
                    chunk_instrument_indices,
                    chunk_pitch_indices,
                    length=length,
                    length_scaling=length_scaling,
                    note_bias=note_bias,
                )

            chunk_forced_start_pos = (
                torch.clamp(
                    flat_forced_start_pos.index_select(0, chunk_indices),
                    min=0,
                    max=max(0, length - 1),
                ).tolist()
                if flat_forced_start_pos is not None
                else None
            )
            decoded_chunk = _viterbi_backward_sparse(
                candidate_ends,
                candidate_scores,
                diag_score,
                _zero_noise_score(
                    length,
                    batch_size=int(chunk_indices.numel()),
                    device=diag_score.device,
                ),
                forcedStartPos=chunk_forced_start_pos,
            )
            for flat_index, intervals in zip(chunk_indices.tolist(), decoded_chunk):
                decoded_flat[int(flat_index)] = intervals
    return decoded_flat
