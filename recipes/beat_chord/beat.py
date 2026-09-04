"""Beat training configuration and losses."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union  # noqa: UP035

import torch
import torch.nn.functional as F
from torch import nn

from .meter_grouping_loss import major_grouping_loss


@dataclass(frozen=True)
class BeatConfig:
    downbeat_pos_weight: float = 20.0
    beat_pos_weight: float = 5.0
    meter_loss_weight: float = 0.05
    meter_grid_ranking_loss_weight: float = 0.0
    meter_grid_ranking_margin: float = 0.1
    meter_grid_kl_loss_weight: float = 0.0
    meter_grid_kl_temperature: float = 0.2
    loss_tolerance: int = 1
    beat_phase_loss_weight: float = 1.0
    bar_phase_loss_weight: float = 1.0
    major_grouping_loss_weight: float = 1.0
    major_grouping_accent_loss_weight: float = 0.25
    major_grouping_accent_temperature: float = 0.5

    def __post_init__(self) -> None:
        if self.downbeat_pos_weight <= 0.0:
            raise ValueError("downbeat_pos_weight must be positive")
        if self.beat_pos_weight <= 0.0:
            raise ValueError("beat_pos_weight must be positive")
        if self.meter_loss_weight < 0.0:
            raise ValueError("meter_loss_weight must be non-negative")
        if self.meter_grid_ranking_loss_weight < 0.0:
            raise ValueError("meter_grid_ranking_loss_weight must be non-negative")
        if self.meter_grid_ranking_margin < 0.0:
            raise ValueError("meter_grid_ranking_margin must be non-negative")
        if self.meter_grid_kl_loss_weight < 0.0:
            raise ValueError("meter_grid_kl_loss_weight must be non-negative")
        if self.meter_grid_kl_temperature <= 0.0:
            raise ValueError("meter_grid_kl_temperature must be positive")
        if self.loss_tolerance < 0:
            raise ValueError("loss_tolerance must be non-negative")
        if self.beat_phase_loss_weight < 0.0:
            raise ValueError("beat_phase_loss_weight must be non-negative")
        if self.bar_phase_loss_weight < 0.0:
            raise ValueError("bar_phase_loss_weight must be non-negative")
        if self.major_grouping_loss_weight < 0.0:
            raise ValueError("major_grouping_loss_weight must be non-negative")
        if self.major_grouping_accent_loss_weight < 0.0:
            raise ValueError("major_grouping_accent_loss_weight must be non-negative")
        if self.major_grouping_accent_temperature <= 0.0:
            raise ValueError("major_grouping_accent_temperature must be positive")

class BalancedSoftmaxLoss(nn.Module):
    def __init__(
        self,
        class_counts: Union[List[int], torch.Tensor],
        tau: float = 1.0,
        ignore_index: int = -100,
    ):
        """
        Args:
            class_counts (Union[List[int], torch.Tensor]):
                各クラスの出現回数のリストまたはテンソル。
                事前に Laplace 平滑化（全カウントに+1するなど）を推奨します。
            tau (float, optional): 補正のスケール係数. Defaults to 1.0.
        """
        super().__init__()

        class_counts = torch.as_tensor(class_counts, dtype=torch.float32)

        # log_prior を計算し、バッファとして登録
        # 希少クラスの出現回数が極小のときに log_prior が負の無限大に爆発するのを防ぐため、最小値を 1.0 に制限します
        log_prior = torch.log(torch.clamp(class_counts, min=1.0))

        self.register_buffer("log_prior", log_prior)
        self.tau = tau
        self.ignore_index = int(ignore_index)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits (torch.Tensor): モデルの出力ロジット (B, T, C)
            labels (torch.Tensor): 正解ラベル (B, T)

        Returns:
            torch.Tensor: 計算された損失値 (スカラー)
        """
        # 形状を合わせる
        if logits.dim() > 2:
            logits = logits.reshape(-1, logits.size(-1))  # (B*T, C)
            labels = labels.reshape(-1)  # (B*T,)

        # meter が未定義のフレームは ignore_index にして、そのまま落とす。
        valid = labels != self.ignore_index
        if not torch.any(valid):
            return logits.sum() * 0.0

        logits = logits[valid]
        labels = labels[valid]

        # ロジット補正: z_k <- z_k + τ * log(n_k)
        adjusted_logits = logits + self.tau * self.log_prior
        loss = F.cross_entropy(adjusted_logits, labels)
        return loss


def masked_l1_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    diff = (predictions - targets).abs()
    if mask is None:
        return diff.mean()

    weighted = diff * mask.to(diff.dtype)
    normalizer = mask.sum().clamp_min(1.0).to(diff.dtype)
    return weighted.sum() / normalizer


def masked_circular_phase_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    phase_error = 1.0 - torch.cos((predictions - targets) * (2.0 * math.pi))
    if mask is None:
        return phase_error.mean()

    weighted = phase_error * mask.to(phase_error.dtype)
    normalizer = mask.sum().clamp_min(1.0).to(phase_error.dtype)
    return weighted.sum() / normalizer


def make_bar_grid_mask(
    *,
    length: int,
    beat_count: int,
    tolerance: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """小節長と beat 数から、許容幅付きの等間隔 grid mask を作る。"""

    if length <= 0:
        raise ValueError("length must be positive")
    if beat_count <= 0:
        raise ValueError("beat_count must be positive")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")

    grid_mask = torch.zeros(length, device=device, dtype=dtype)
    for beat_index in range(beat_count):
        grid_position = int(
            round(float(beat_index) * float(length) / float(beat_count))
        )
        grid_position = max(0, min(length - 1, grid_position))
        grid_mask[grid_position] = 1.0

    if tolerance > 0:
        grid_mask = F.max_pool1d(
            grid_mask.view(1, 1, length),
            kernel_size=1 + 2 * tolerance,
            stride=1,
            padding=tolerance,
        ).view(length)
    return grid_mask.clamp(max=1.0)


def bar_meter_grid_ranking_loss(
    beat_logits: torch.Tensor,
    downbeat_targets: torch.Tensor,
    meter_targets: torch.Tensor,
    meter_class_beat_counts: Sequence[int],
    mask: torch.Tensor | None = None,
    *,
    tolerance: int = 1,
    margin: float = 0.1,
) -> torch.Tensor:
    """正解 meter の beat grid が他候補より高スコアになるようにする小節単位 loss。"""

    if beat_logits.shape != downbeat_targets.shape:
        raise ValueError("beat_logits and downbeat_targets must have the same shape")
    if beat_logits.shape != meter_targets.shape:
        raise ValueError("beat_logits and meter_targets must have the same shape")
    if mask is not None and beat_logits.shape != mask.shape:
        raise ValueError("beat_logits and mask must have the same shape")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if len(meter_class_beat_counts) == 0:
        raise ValueError("meter_class_beat_counts must not be empty")

    unique_beat_counts = tuple(
        sorted({int(count) for count in meter_class_beat_counts})
    )
    if len(unique_beat_counts) <= 1:
        return beat_logits.sum() * 0.0
    beat_count_to_grid_index = {
        beat_count: index for index, beat_count in enumerate(unique_beat_counts)
    }

    beat_prob = torch.sigmoid(beat_logits)
    bar_losses: list[torch.Tensor] = []

    # 1. downbeat のペアから、窓内で完結している小節だけを取り出す。
    for batch_index in range(beat_logits.shape[0]):
        if mask is None:
            valid_mask = torch.ones_like(
                downbeat_targets[batch_index], dtype=torch.bool
            )
        else:
            valid_mask = mask[batch_index] > 0.0
        downbeat_frames = torch.nonzero(
            (downbeat_targets[batch_index] > 0.5) & valid_mask,
            as_tuple=False,
        ).flatten()
        if downbeat_frames.numel() < 2:
            continue

        for start_tensor, end_tensor in zip(downbeat_frames[:-1], downbeat_frames[1:]):
            start_frame = int(start_tensor.item())
            end_frame = int(end_tensor.item())
            bar_length = end_frame - start_frame
            if bar_length <= 1:
                continue

            bar_valid_mask = valid_mask[start_frame:end_frame]
            bar_meter_targets = meter_targets[batch_index, start_frame:end_frame]
            meter_valid = (bar_meter_targets >= 0) & bar_valid_mask
            if not torch.any(meter_valid):
                continue

            # 2. 小節内で最も多い meter class を、この小節の正解 meter とみなす。
            target_meter_class = int(
                torch.mode(bar_meter_targets[meter_valid]).values.item()
            )
            if target_meter_class < 0 or target_meter_class >= len(
                meter_class_beat_counts
            ):
                continue
            target_beat_count = int(meter_class_beat_counts[target_meter_class])
            target_grid_index = beat_count_to_grid_index.get(target_beat_count)
            if target_grid_index is None:
                continue

            bar_prob = beat_prob[batch_index, start_frame:end_frame]
            bar_weight = bar_valid_mask.to(dtype=bar_prob.dtype)
            grid_scores: list[torch.Tensor] = []

            # 3. 各 meter 候補の等間隔 grid で beat 分布をスコア化する。
            for beat_count in unique_beat_counts:
                grid_mask = make_bar_grid_mask(
                    length=bar_length,
                    beat_count=beat_count,
                    tolerance=tolerance,
                    device=bar_prob.device,
                    dtype=bar_prob.dtype,
                )
                on_grid_weight = grid_mask * bar_weight
                off_grid_weight = (1.0 - grid_mask) * bar_weight
                on_grid_score = (bar_prob * on_grid_weight).sum()
                on_grid_score = on_grid_score / on_grid_weight.sum().clamp_min(1.0)
                off_grid_score = (bar_prob * off_grid_weight).sum()
                off_grid_score = off_grid_score / off_grid_weight.sum().clamp_min(1.0)
                grid_scores.append(on_grid_score - off_grid_score)

            # 4. 正解 grid が他の grid より margin 以上高くなるように ranking loss を取る。
            score_tensor = torch.stack(grid_scores)
            target_score = score_tensor[target_grid_index]
            other_mask = torch.ones(
                len(unique_beat_counts),
                device=score_tensor.device,
                dtype=torch.bool,
            )
            other_mask[target_grid_index] = False
            other_scores = score_tensor[other_mask]
            bar_losses.append(F.relu(margin + other_scores - target_score).mean())

    if not bar_losses:
        return beat_logits.sum() * 0.0
    return torch.stack(bar_losses).mean()


def estimate_bar_phase_step(
    bar_phase_targets: torch.Tensor,
    phase_mask: torch.Tensor,
) -> torch.Tensor:
    """bar phase の隣接差分から、各フレームのおおよその1フレーム進行量を推定する。"""

    phase_delta = bar_phase_targets[:, 1:] - bar_phase_targets[:, :-1]
    valid_delta = (
        (phase_delta > 0.0) & (phase_mask[:, 1:] > 0.0) & (phase_mask[:, :-1] > 0.0)
    )
    next_step = F.pad(
        torch.where(valid_delta, phase_delta, torch.zeros_like(phase_delta)),
        (0, 1),
    )
    prev_step = F.pad(
        torch.where(valid_delta, phase_delta, torch.zeros_like(phase_delta)),
        (1, 0),
    )
    return torch.where(next_step > 0.0, next_step, prev_step)


def phase_grid_masks_from_targets(
    *,
    unique_beat_counts: torch.Tensor,
    bar_phase_targets: torch.Tensor,
    phase_mask: torch.Tensor,
    tolerance: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """bar phase から各 meter 候補の grid mask と有効 frame mask を作る。"""

    phase_step = estimate_bar_phase_step(bar_phase_targets, phase_mask)
    counts = unique_beat_counts.to(
        device=bar_phase_targets.device,
        dtype=dtype,
    ).view(-1, 1, 1)
    phase = bar_phase_targets.clamp(0.0, 1.0).to(dtype).unsqueeze(0)
    phase_step_expanded = phase_step.to(dtype).unsqueeze(0).clamp_min(1e-6)
    cycle_position = phase * counts
    distance_in_cycles = (cycle_position - torch.round(cycle_position)).abs()
    distance_in_frames = (distance_in_cycles / counts) / phase_step_expanded
    grid_mask = (distance_in_frames <= float(tolerance)).to(dtype)
    return grid_mask, phase_step > 0.0


def score_phase_grids(
    *,
    beat_prob: torch.Tensor,
    grid_mask: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """候補 grid ごとの on-grid/off-grid 差分 score を計算する。"""

    candidate_weight = weight.unsqueeze(0)
    on_grid_weight = grid_mask * candidate_weight
    off_grid_weight = (1.0 - grid_mask) * candidate_weight
    beat_prob_expanded = beat_prob.unsqueeze(0)

    on_grid_score = (beat_prob_expanded * on_grid_weight).sum(dim=(1, 2))
    on_grid_score = on_grid_score / on_grid_weight.sum(dim=(1, 2)).clamp_min(1.0)
    off_grid_score = (beat_prob_expanded * off_grid_weight).sum(dim=(1, 2))
    off_grid_score = off_grid_score / off_grid_weight.sum(dim=(1, 2)).clamp_min(1.0)
    return on_grid_score - off_grid_score


def meter_logits_to_grid_probs(
    *,
    meter_logits: torch.Tensor,
    meter_class_grid_indices: torch.Tensor,
    weight: torch.Tensor,
    num_grids: int,
) -> torch.Tensor:
    """meter class 確率を、同じ beat 数を持つ grid group ごとの確率に集約する。"""

    meter_probs = torch.softmax(meter_logits, dim=-1)
    grid_probs: list[torch.Tensor] = []
    normalizer = weight.sum().clamp_min(1.0)
    class_grid_indices = meter_class_grid_indices.to(meter_logits.device)
    for grid_index in range(num_grids):
        class_mask = class_grid_indices == grid_index
        frame_grid_prob = meter_probs[..., class_mask].sum(dim=-1)
        grid_probs.append((frame_grid_prob * weight).sum() / normalizer)

    probs = torch.stack(grid_probs)
    probs = probs.clamp_min(1e-8)
    return probs / probs.sum().clamp_min(1e-8)


def phase_meter_grid_ranking_loss(
    beat_logits: torch.Tensor,
    meter_targets: torch.Tensor,
    unique_beat_counts: torch.Tensor,
    meter_class_grid_indices: torch.Tensor,
    bar_phase_targets: torch.Tensor,
    phase_mask: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    tolerance: int = 1,
    margin: float = 0.1,
) -> torch.Tensor:
    """bar phase target を使って、meter grid ranking をベクトル化して計算する。"""

    if beat_logits.shape != meter_targets.shape:
        raise ValueError("beat_logits and meter_targets must have the same shape")
    if beat_logits.shape != bar_phase_targets.shape:
        raise ValueError("beat_logits and bar_phase_targets must have the same shape")
    if beat_logits.shape != phase_mask.shape:
        raise ValueError("beat_logits and phase_mask must have the same shape")
    if mask is not None and beat_logits.shape != mask.shape:
        raise ValueError("beat_logits and mask must have the same shape")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if margin < 0.0:
        raise ValueError("margin must be non-negative")
    if unique_beat_counts.numel() <= 1:
        return beat_logits.sum() * 0.0

    dtype = beat_logits.dtype
    beat_prob = torch.sigmoid(beat_logits)
    valid_mask = (meter_targets >= 0) & (phase_mask > 0.0)
    if mask is not None:
        valid_mask = valid_mask & (mask > 0.0)

    # 1. bar phase から「grid まで何フレーム離れているか」を候補 meter ごとに計算する。
    grid_mask, valid_phase_step = phase_grid_masks_from_targets(
        unique_beat_counts=unique_beat_counts,
        bar_phase_targets=bar_phase_targets,
        phase_mask=phase_mask,
        tolerance=tolerance,
        dtype=dtype,
    )
    valid_mask = valid_mask & valid_phase_step

    # 2. meter class を「候補 beat 数の index」に変換し、同じ正解 grid ごとに集計する。
    safe_meter_targets = meter_targets.clamp(
        min=0,
        max=int(meter_class_grid_indices.numel()) - 1,
    )
    target_grid_indices = meter_class_grid_indices.to(beat_logits.device)[
        safe_meter_targets
    ]
    base_weight = valid_mask.to(dtype)
    weighted_losses: list[torch.Tensor] = []
    group_weights: list[torch.Tensor] = []

    for target_grid_index in range(int(unique_beat_counts.numel())):
        target_weight = base_weight * (target_grid_indices == target_grid_index).to(
            dtype
        )
        target_weight_sum = target_weight.sum()
        group_present = (target_weight_sum > 0.0).to(dtype)

        scores = score_phase_grids(
            beat_prob=beat_prob,
            grid_mask=grid_mask,
            weight=target_weight,
        )

        target_score = scores[target_grid_index]
        other_loss = F.relu(margin + scores - target_score)
        other_loss = other_loss.clone()
        other_loss[target_grid_index] = 0.0
        weighted_losses.append(
            other_loss.sum() / max(1, int(unique_beat_counts.numel()) - 1)
        )
        group_weights.append(group_present)

    loss_tensor = torch.stack(weighted_losses)
    weight_tensor = torch.stack(group_weights)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)


def phase_meter_grid_kl_loss(
    beat_logits: torch.Tensor,
    meter_logits: torch.Tensor,
    meter_targets: torch.Tensor,
    unique_beat_counts: torch.Tensor,
    meter_class_grid_indices: torch.Tensor,
    bar_phase_targets: torch.Tensor,
    phase_mask: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    tolerance: int = 1,
    temperature: float = 0.2,
) -> torch.Tensor:
    """detach した beat-grid 分布に meter logits の grid 分布を合わせる KL loss。"""

    if beat_logits.shape != meter_targets.shape:
        raise ValueError("beat_logits and meter_targets must have the same shape")
    if beat_logits.shape != bar_phase_targets.shape:
        raise ValueError("beat_logits and bar_phase_targets must have the same shape")
    if beat_logits.shape != phase_mask.shape:
        raise ValueError("beat_logits and phase_mask must have the same shape")
    if meter_logits.shape[:2] != beat_logits.shape:
        raise ValueError("meter_logits must have shape [B, T, C]")
    if mask is not None and beat_logits.shape != mask.shape:
        raise ValueError("beat_logits and mask must have the same shape")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if unique_beat_counts.numel() <= 1:
        return beat_logits.sum() * 0.0

    dtype = beat_logits.dtype
    valid_mask = (meter_targets >= 0) & (phase_mask > 0.0)
    if mask is not None:
        valid_mask = valid_mask & (mask > 0.0)

    grid_mask, valid_phase_step = phase_grid_masks_from_targets(
        unique_beat_counts=unique_beat_counts,
        bar_phase_targets=bar_phase_targets,
        phase_mask=phase_mask,
        tolerance=tolerance,
        dtype=dtype,
    )
    valid_mask = valid_mask & valid_phase_step

    safe_meter_targets = meter_targets.clamp(
        min=0,
        max=int(meter_class_grid_indices.numel()) - 1,
    )
    target_grid_indices = meter_class_grid_indices.to(beat_logits.device)[
        safe_meter_targets
    ]
    base_weight = valid_mask.to(dtype)
    beat_prob = torch.sigmoid(beat_logits.detach())
    losses: list[torch.Tensor] = []
    group_weights: list[torch.Tensor] = []

    # 1. 正解 grid group ごとに分け、複数拍子が混ざる batch でも分布を混ぜない。
    for target_grid_index in range(int(unique_beat_counts.numel())):
        target_weight = base_weight * (target_grid_indices == target_grid_index).to(
            dtype
        )
        group_present = (target_weight.sum() > 0.0).to(dtype)

        # 2. beat pattern から q_grid を作る。detach 済みなので beat 側へは勾配を返さない。
        with torch.no_grad():
            grid_scores = score_phase_grids(
                beat_prob=beat_prob,
                grid_mask=grid_mask,
                weight=target_weight,
            )
            q_grid = torch.softmax(grid_scores / float(temperature), dim=0)

        # 3. meter_logits を同じ grid group 空間に集約し、KL(q_grid || p_meter_grid) を取る。
        p_grid = meter_logits_to_grid_probs(
            meter_logits=meter_logits,
            meter_class_grid_indices=meter_class_grid_indices,
            weight=target_weight,
            num_grids=int(unique_beat_counts.numel()),
        )
        losses.append((q_grid * (q_grid.clamp_min(1e-8).log() - p_grid.log())).sum())
        group_weights.append(group_present)

    loss_tensor = torch.stack(losses)
    weight_tensor = torch.stack(group_weights)
    return (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp_min(1.0)


# https://github.com/CPJKU/beat_this/blob/main/beat_this/model/loss.py
class ShiftTolerantBCELoss(torch.nn.Module):
    """
    少しずれた beat/downbeat ラベルを許容する BCE loss。
    予測側を max-pooling し、正解フレーム周辺で最も強い予測に勾配を流す。
    """

    def __init__(self, pos_weight: float = 1, tolerance: int = 1):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(pos_weight, dtype=torch.get_default_dtype()),
            persistent=False,
        )
        self.tolerance = tolerance

    def spread(self, x: torch.Tensor, factor: int = 1):
        if self.tolerance == 0:
            return x
        return F.max_pool1d(x, 1 + 2 * factor * self.tolerance, 1)

    def crop(self, x: torch.Tensor, factor: int = 1):
        return x[..., factor * self.tolerance : -factor * self.tolerance or None]

    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # 予測側を tolerance 分だけ広げ、端の不確かなフレームを落とす。
        spreaded_preds = self.crop(self.spread(preds))
        cropped_targets = self.crop(targets, factor=2)
        # 正解 beat 周辺の負例は見ない。padding や未アノテーション区間も mask で落とす。
        look_at = cropped_targets + (1 - self.spread(targets, factor=2))
        if mask is not None:
            look_at = look_at * self.crop(mask, factor=2)
        return F.binary_cross_entropy_with_logits(
            spreaded_preds,
            cropped_targets,
            weight=look_at,
            pos_weight=self.pos_weight,
        )

class BeatLoss(nn.Module):
    def __init__(
        self,
        config: BeatConfig,
        meter_class_counts: Union[List[int], torch.Tensor],
        meter_classes: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.meter_classes = (
            tuple(
                (int(meter_num), int(meter_den))
                for meter_num, meter_den in meter_classes
            )
            if meter_classes is not None
            else ()
        )
        self.meter_class_beat_counts = (
            tuple(int(meter_num) for meter_num, _meter_den in self.meter_classes)
            if self.meter_classes
            else ()
        )
        unique_beat_counts = tuple(sorted(set(self.meter_class_beat_counts)))
        beat_count_to_grid_index = {
            beat_count: index for index, beat_count in enumerate(unique_beat_counts)
        }
        meter_class_grid_indices = tuple(
            beat_count_to_grid_index[beat_count]
            for beat_count in self.meter_class_beat_counts
        )
        self.register_buffer(
            "meter_grid_unique_beat_counts",
            torch.tensor(unique_beat_counts, dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "meter_class_grid_indices",
            torch.tensor(meter_class_grid_indices, dtype=torch.long),
            persistent=False,
        )
        if self.meter_class_beat_counts and len(self.meter_class_beat_counts) != int(
            torch.as_tensor(meter_class_counts).numel()
        ):
            raise ValueError("meter_classes must match meter_class_counts length")
        if (
            config.meter_grid_ranking_loss_weight > 0.0
            and not self.meter_class_beat_counts
        ):
            raise ValueError(
                "meter_classes is required when meter_grid_ranking_loss_weight is positive"
            )
        if config.meter_grid_kl_loss_weight > 0.0 and not self.meter_class_beat_counts:
            raise ValueError(
                "meter_classes is required when meter_grid_kl_loss_weight is positive"
            )
        self.beat_loss = ShiftTolerantBCELoss(
            pos_weight=config.beat_pos_weight,
            tolerance=config.loss_tolerance,
        )
        self.downbeat_loss = ShiftTolerantBCELoss(
            pos_weight=config.downbeat_pos_weight,
            tolerance=config.loss_tolerance,
        )
        self.meter_loss = BalancedSoftmaxLoss(meter_class_counts, tau=0.3)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        beat_logits = outputs.get("beat_logits")
        downbeat_logits = outputs.get("downbeat_logits")
        meter_logits = outputs.get("meter_logits")
        if beat_logits is None or downbeat_logits is None or meter_logits is None:
            raise ValueError("Beat training requires beat/downbeat/meter logits")

        beat_targets = batch["beat_targets"].to(beat_logits.device)
        downbeat_targets = batch["downbeat_targets"].to(downbeat_logits.device)
        meter_targets = batch["meter_targets"].to(meter_logits.device)
        beat_mask = batch.get("beat_mask")
        if beat_mask is not None:
            beat_mask = beat_mask.to(beat_logits.device)
        bar_phase_targets = batch.get("bar_phase_targets")
        if bar_phase_targets is not None:
            bar_phase_targets = bar_phase_targets.to(beat_logits.device)
        phase_mask = batch.get("phase_mask")
        if phase_mask is not None:
            phase_mask = phase_mask.to(beat_logits.device)

        beat_loss = self.beat_loss(
            beat_logits.unsqueeze(1),
            beat_targets.unsqueeze(1),
            None if beat_mask is None else beat_mask.unsqueeze(1),
        )
        downbeat_loss = self.downbeat_loss(
            downbeat_logits.unsqueeze(1),
            downbeat_targets.unsqueeze(1),
            None if beat_mask is None else beat_mask.unsqueeze(1),
        )
        meter_loss = self.meter_loss(meter_logits, meter_targets)
        meter_grid_ranking_loss = beat_logits.sum() * 0.0
        if self.config.meter_grid_ranking_loss_weight > 0.0:
            if bar_phase_targets is not None and phase_mask is not None:
                meter_grid_ranking_loss = phase_meter_grid_ranking_loss(
                    beat_logits,
                    meter_targets,
                    self.meter_grid_unique_beat_counts,
                    self.meter_class_grid_indices,
                    bar_phase_targets,
                    phase_mask,
                    beat_mask,
                    tolerance=self.config.loss_tolerance,
                    margin=self.config.meter_grid_ranking_margin,
                )
            else:
                meter_grid_ranking_loss = bar_meter_grid_ranking_loss(
                    beat_logits,
                    downbeat_targets,
                    meter_targets,
                    self.meter_class_beat_counts,
                    beat_mask,
                    tolerance=self.config.loss_tolerance,
                    margin=self.config.meter_grid_ranking_margin,
                )
        meter_grid_kl_loss = beat_logits.sum() * 0.0
        if self.config.meter_grid_kl_loss_weight > 0.0:
            if bar_phase_targets is None or phase_mask is None:
                raise ValueError(
                    "meter_grid_kl_loss requires bar_phase_targets and phase_mask"
                )
            meter_grid_kl_loss = phase_meter_grid_kl_loss(
                beat_logits,
                meter_logits,
                meter_targets,
                self.meter_grid_unique_beat_counts,
                self.meter_class_grid_indices,
                bar_phase_targets,
                phase_mask,
                beat_mask,
                tolerance=self.config.loss_tolerance,
                temperature=self.config.meter_grid_kl_temperature,
            )

        major_grouping_total_loss = beat_logits.sum() * 0.0
        major_grouping_valid_loss = beat_logits.sum() * 0.0
        major_grouping_accent_loss = beat_logits.sum() * 0.0
        major_grouping_bar_count = beat_logits.sum() * 0.0
        meter_aware_crop_rate = beat_logits.sum() * 0.0
        meter_aware_crop = batch.get("meter_aware_crop")
        if meter_aware_crop is not None:
            meter_aware_crop_rate = meter_aware_crop.to(
                device=beat_logits.device,
                dtype=beat_logits.dtype,
            ).mean()
        group_boundary_logits = outputs.get("group_boundary_logits")
        midi_frames = batch.get("midi_frames")
        if (
            self.config.major_grouping_loss_weight > 0.0
            and group_boundary_logits is not None
            and midi_frames is not None
            and self.meter_classes
        ):
            grouping_result = major_grouping_loss(
                group_boundary_logits=group_boundary_logits,
                beat_targets=beat_targets,
                downbeat_targets=downbeat_targets,
                meter_targets=meter_targets,
                beat_mask=beat_mask,
                midi_frames=midi_frames.to(group_boundary_logits.device),
                meter_classes=self.meter_classes,
                tolerance=self.config.loss_tolerance,
                accent_loss_weight=(self.config.major_grouping_accent_loss_weight),
                accent_temperature=(self.config.major_grouping_accent_temperature),
            )
            major_grouping_total_loss = grouping_result.loss
            major_grouping_valid_loss = grouping_result.valid_pattern_loss
            major_grouping_accent_loss = grouping_result.accent_alignment_loss
            major_grouping_bar_count = torch.tensor(
                float(grouping_result.supervised_bar_count),
                device=beat_logits.device,
                dtype=beat_logits.dtype,
            )

        total_loss = (
            beat_loss
            + downbeat_loss
            + meter_loss * float(self.config.meter_loss_weight)
            + meter_grid_ranking_loss
            * float(self.config.meter_grid_ranking_loss_weight)
            + meter_grid_kl_loss * float(self.config.meter_grid_kl_loss_weight)
            + major_grouping_total_loss * float(self.config.major_grouping_loss_weight)
        )
        return total_loss, {
            "beat_total_loss": total_loss,
            "beat_loss": beat_loss,
            "downbeat_loss": downbeat_loss,
            "meter_loss": meter_loss,
            "meter_grid_ranking_loss": meter_grid_ranking_loss,
            "meter_grid_kl_loss": meter_grid_kl_loss,
            "major_grouping_loss": major_grouping_total_loss,
            "major_grouping_valid_loss": major_grouping_valid_loss,
            "major_grouping_accent_loss": major_grouping_accent_loss,
            "major_grouping_bar_count": major_grouping_bar_count,
            "meter_aware_crop_rate": meter_aware_crop_rate,
        }
