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
def viterbiBackward(
    score: torch.Tensor,
    noiseScore: torch.Tensor,
    forcedStartPos: Optional[List[int]] = None,
) -> IntervalBatch:
    """
    Decode intervals from left to right.

    Args:
        score: [T, T, B] where score[end, begin, batch] is the score for [begin, end].
        noiseScore: [T-1, B] score for a non-event transition [t, t+1].
        forcedStartPos: per-batch forced start position for segmented decoding.
    """
    T = _validate_shapes(score, noiseScore)
    nBatch = int(score.shape[2])

    q = score.new_zeros(T, nBatch)
    ptr = []

    scoreT = score.transpose(0, 1).contiguous()
    q[T - 1] = score[T - 1, T - 1, :] * (score[T - 1, T - 1, :] > 0)

    for offset in range(1, T):
        t = T - offset - 1
        subScore = scoreT[t, t + 1 :, :]

        candidates = torch.cat(
            [
                q[t + 1 : t + 2, :] + noiseScore[t, :],  # skip current position
                q[t + 1 :, :] + subScore,  # start an interval at t
            ],
            dim=0,
        )

        bestValue, selection = candidates.max(dim=0)
        ptr.append(selection - 1)

        singletonMask = score[t, t, :] > 0
        q[t] = bestValue + score[t, t, :] * singletonMask

    ptr = torch.stack(ptr, dim=0).cpu()
    diagInclusion = (torch.diagonal(score, dim1=0, dim2=1) > 0).cpu()

    if forcedStartPos is None:
        forcedStartPos = [0] * nBatch

    result: IntervalBatch = []
    for batchIdx in range(nBatch):
        pos = forcedStartPos[batchIdx]
        batchResult: List[Interval] = []
        curDiag = diagInclusion[batchIdx]

        while pos < T - 1:
            selection = int(ptr[T - pos - 2][batchIdx])

            if bool(curDiag[pos]):
                batchResult.append((pos, pos))

            if selection < 0:
                pos += 1
            else:
                end = selection + pos + 1
                batchResult.append((pos, end))
                pos = end

        if score[T - 1, T - 1, batchIdx] > 0:
            batchResult.append((T - 1, T - 1))

        result.append(batchResult)

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
        self, forcedStartPos: Optional[List[int]] = None, forward: bool = False
    ) -> IntervalBatch:
        """Decode the best interval sequence."""
        if forward:
            return viterbi(self.score, self.noiseScore, forcedStartPos)
        return viterbiBackward(self.score, self.noiseScore, forcedStartPos)

    def evalPath(self, intervals: IntervalBatch) -> torch.Tensor:
        """Compute the unnormalized score of a given interval path."""
        return evalPath(intervals, self.score, self.noiseScore)

    def computeLogZ(self, noBackward: bool = False) -> torch.Tensor:
        """
        Compute log Z.

        noBackward=True uses the plain scripted DP.
        noBackward=False uses the custom-autograd implementation with faster gradients.
        """
        if noBackward:
            return computeLogZ(self.score, self.noiseScore)
        return computeLogZFasterGrad(self.score, self.noiseScore)

    def logProb(
        self, intervals: IntervalBatch, noBackward: bool = False
    ) -> torch.Tensor:
        """Compute log p(intervals)."""
        return self.evalPath(intervals) - self.computeLogZ(noBackward=noBackward)


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
    """セグメント継続用の開始位置制約を flat track 形式へ変換する。"""
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
    """
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
                chunk_loss = -semi_crf.logProb(chunk_targets)
            else:
                augmented_score = score + interval_cost
                semi_crf = NeuralSemiCRFInterval(augmented_score, noise_score)
                gold_score = evalPath(chunk_targets, score, noise_score)
                chunk_loss = semi_crf.computeLogZ() - gold_score
            total_loss_sum = total_loss_sum + chunk_loss.sum()
            total_tracks += int(chunk_indices.numel())
            total_intervals += sum(len(track) for track in chunk_targets)

    if total_tracks <= 0:
        return interval_query.sum() * 0.0, 0, 0
    return total_loss_sum / float(total_tracks), total_tracks, total_intervals


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
            decoded_chunk = semi_crf.decode(forcedStartPos=chunk_forced_start_pos)
            for flat_index, intervals in zip(chunk_indices.tolist(), decoded_chunk):
                decoded_flat[int(flat_index)] = intervals

    return [
        decoded_flat[batch_index * num_pitches : (batch_index + 1) * num_pitches]
        for batch_index in range(int(batch_size))
    ]
