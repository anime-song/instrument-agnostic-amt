from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ..modeling.heads.interval_boundaries import PitchIntervalTargets
from ..modeling.heads.semi_crf import compute_factorized_pair_interval_loss
from ..modeling.model import AudioSemiCRFTransformer, NUM_PITCHES
from .v1_losses import compute_v1_losses


def _stack_pair_presence(
    interval_targets: list[PitchIntervalTargets],
    *,
    device: torch.device,
    num_instruments: int,
) -> torch.Tensor:
    rows = []
    for target in interval_targets:
        row = torch.zeros((num_instruments, NUM_PITCHES), dtype=torch.bool)
        if target.pair_presence is not None:
            presence = target.pair_presence.to(dtype=torch.bool, device="cpu")
            copy_instruments = min(int(presence.shape[0]), int(num_instruments))
            copy_pitches = min(int(presence.shape[1]), NUM_PITCHES)
            row[:copy_instruments, :copy_pitches] = presence[
                :copy_instruments, :copy_pitches
            ]
        rows.append(row)
    return torch.stack(rows, dim=0).to(device=device)


def _select_pair_candidates(
    pair_gate_logits: torch.Tensor,
    interval_targets: list[PitchIntervalTargets],
    *,
    train_topk: int,
    random_negatives: int,
    max_pairs: int,
) -> tuple[list[list[int]], PitchIntervalTargets]:
    if pair_gate_logits.dim() != 3:
        raise ValueError("pair_gate_logits must have shape [B, I, P]")

    batch_size, num_instruments, num_pitches = pair_gate_logits.shape
    if int(num_pitches) != NUM_PITCHES:
        raise ValueError(f"expected {NUM_PITCHES} pitches, got {int(num_pitches)}")
    if len(interval_targets) != int(batch_size):
        raise ValueError("interval_targets length must match batch size")

    selected_pair_ids: list[list[int]] = []
    flat_intervals: list[list[tuple[int, int]]] = []
    flat_has_onset: list[list[bool]] = []
    flat_has_offset: list[list[bool]] = []
    flat_onset_offsets: list[list[float]] = []
    flat_offset_offsets: list[list[float]] = []
    flat_instrument_sets: list[list[tuple[int, ...]]] = []
    flat_positive_pair_ids: list[int] = []
    total_pairs = int(num_instruments) * int(num_pitches)

    for batch_index, target in enumerate(interval_targets):
        positive_pair_ids = [int(pair_id) for pair_id in target.positive_pair_ids]
        positive_lookup = {
            int(pair_id): pair_index
            for pair_index, pair_id in enumerate(positive_pair_ids)
        }
        selected: list[int] = []
        seen: set[int] = set()
        for pair_id in positive_pair_ids:
            if pair_id not in seen:
                selected.append(pair_id)
                seen.add(pair_id)

        negative_cap = (
            max(0, int(max_pairs) - len(selected))
            if int(max_pairs) > 0
            else total_pairs
        )
        hard_cap = min(max(0, int(train_topk)), negative_cap)
        logits_flat = pair_gate_logits[batch_index].detach().reshape(-1)
        negative_mask = torch.ones(
            total_pairs, dtype=torch.bool, device=logits_flat.device
        )
        if positive_pair_ids:
            positive_tensor = torch.tensor(
                positive_pair_ids,
                device=logits_flat.device,
                dtype=torch.long,
            )
            negative_mask[positive_tensor] = False

        hard_ids: list[int] = []
        if hard_cap > 0 and bool(torch.any(negative_mask).item()):
            masked_scores = logits_flat.masked_fill(~negative_mask, float("-inf"))
            topk = min(hard_cap, int(torch.isfinite(masked_scores).sum().item()))
            if topk > 0:
                hard_ids = [
                    int(value)
                    for value in torch.topk(masked_scores, k=topk).indices.tolist()
                ]
                for pair_id in hard_ids:
                    if pair_id not in seen:
                        selected.append(pair_id)
                        seen.add(pair_id)
                        negative_mask[pair_id] = False

        random_cap = min(
            max(0, int(random_negatives)), max(0, negative_cap - len(hard_ids))
        )
        if random_cap > 0 and bool(torch.any(negative_mask).item()):
            candidates = negative_mask.nonzero(as_tuple=False).flatten()
            order = torch.randperm(int(candidates.numel()), device=candidates.device)
            for pair_id in candidates.index_select(0, order[:random_cap]).tolist():
                pair_id = int(pair_id)
                if pair_id not in seen:
                    selected.append(pair_id)
                    seen.add(pair_id)

        selected_pair_ids.append(selected)
        for pair_id in selected:
            positive_index = positive_lookup.get(int(pair_id))
            if positive_index is None:
                flat_intervals.append([])
                flat_has_onset.append([])
                flat_has_offset.append([])
                flat_onset_offsets.append([])
                flat_offset_offsets.append([])
                flat_instrument_sets.append([])
                continue
            flat_intervals.append(target.intervals[positive_index])
            flat_has_onset.append(target.has_onset[positive_index])
            flat_has_offset.append(target.has_offset[positive_index])
            flat_onset_offsets.append(target.onset_offsets[positive_index])
            flat_offset_offsets.append(target.offset_offsets[positive_index])
            if target.instrument_sets:
                flat_instrument_sets.append(target.instrument_sets[positive_index])
            else:
                instrument_id = int(pair_id) // NUM_PITCHES
                flat_instrument_sets.append(
                    [(instrument_id,) for _ in target.intervals[positive_index]]
                )
            flat_positive_pair_ids.append(int(pair_id))

    flat_targets = PitchIntervalTargets(
        intervals=flat_intervals,
        has_onset=flat_has_onset,
        has_offset=flat_has_offset,
        onset_offsets=flat_onset_offsets,
        offset_offsets=flat_offset_offsets,
        instrument_sets=flat_instrument_sets,
        positive_pair_ids=flat_positive_pair_ids,
        pair_presence=None,
    )
    return selected_pair_ids, flat_targets


def _gather_flat_boundary_targets(
    targets: PitchIntervalTargets,
    entries: list[tuple[int, int, int, int]],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not entries:
        empty = torch.zeros((0,), device=device, dtype=torch.float32)
        return empty, empty, empty, empty
    has_onset = torch.tensor(
        [
            float(targets.has_onset[track_index][interval_index])
            for track_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    has_offset = torch.tensor(
        [
            float(targets.has_offset[track_index][interval_index])
            for track_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    onset_offsets = torch.tensor(
        [
            float(targets.onset_offsets[track_index][interval_index])
            for track_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    offset_offsets = torch.tensor(
        [
            float(targets.offset_offsets[track_index][interval_index])
            for track_index, interval_index, _, _ in entries
        ],
        device=device,
        dtype=torch.float32,
    )
    return has_onset, has_offset, onset_offsets, offset_offsets


def _compute_v2_losses(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, Any],
    args: Any | None = None,
    model: AudioSemiCRFTransformer | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    interval_features = outputs.get("interval_features")
    pair_gate_logits = outputs.get("pair_gate_logits")
    pitch_interval_query = outputs.get("pitch_interval_query")
    pitch_interval_key = outputs.get("pitch_interval_key")
    pitch_interval_diag = outputs.get("pitch_interval_diag")
    instrument_interval_query = outputs.get("instrument_interval_query")
    instrument_interval_key = outputs.get("instrument_interval_key")
    instrument_interval_diag = outputs.get("instrument_interval_diag")
    frame_valid_mask = outputs.get("frame_valid_mask")
    interval_targets = batch.get("interval_targets")

    if (
        interval_features is None
        or pair_gate_logits is None
        or pitch_interval_query is None
        or pitch_interval_key is None
        or pitch_interval_diag is None
        or instrument_interval_query is None
        or instrument_interval_key is None
        or instrument_interval_diag is None
        or interval_targets is None
        or frame_valid_mask is None
        or model is None
    ):
        raise ValueError(
            "Factorized V2 training requires interval projections, pair gate logits, "
            "targets, and model"
        )
    if not isinstance(interval_targets, list):
        raise ValueError("interval_targets must be a list of PitchIntervalTargets")

    semi_crf_loss_weight = (
        1.0 if args is None else float(getattr(args, "semi_crf_loss_weight", 1.0))
    )
    pair_gate_loss_weight = (
        1.0 if args is None else float(getattr(args, "pair_gate_loss_weight", 1.0))
    )
    semi_crf_false_negative_cost = (
        0.0
        if args is None
        else float(getattr(args, "semi_crf_false_negative_cost", 0.0))
    )
    semi_crf_false_positive_cost = (
        0.0
        if args is None
        else float(getattr(args, "semi_crf_false_positive_cost", 0.0))
    )
    semi_crf_loss_backend = (
        "torch"
        if args is None
        else str(getattr(args, "semi_crf_loss_backend", "torch"))
    )
    interval_presence_loss_weight = (
        1.0
        if args is None
        else float(getattr(args, "interval_presence_loss_weight", 1.0))
    )
    interval_offset_loss_weight = (
        1.0
        if args is None
        else float(getattr(args, "interval_offset_loss_weight", 1.0))
    )
    train_topk = (
        256 if args is None else int(getattr(args, "instrument_pair_train_topk", 256))
    )
    random_negatives = (
        128
        if args is None
        else int(getattr(args, "instrument_pair_random_negatives", 128))
    )
    max_pairs = (
        512 if args is None else int(getattr(args, "instrument_pair_max_pairs", 512))
    )

    valid_lengths = frame_valid_mask.to(dtype=torch.long).sum(dim=-1)
    length_scaling = model.config.semi_crf_length_scaling
    length_penalty = model.config.semi_crf_length_penalty

    pair_presence = _stack_pair_presence(
        interval_targets,
        device=pair_gate_logits.device,
        num_instruments=int(model.config.num_instrument_classes),
    )
    positive_count = pair_presence.sum().to(dtype=torch.float32)
    negative_count = pair_presence.numel() - positive_count
    pos_weight = torch.clamp(negative_count / positive_count.clamp_min(1.0), 1.0, 50.0)
    pair_gate_loss = F.binary_cross_entropy_with_logits(
        pair_gate_logits.float(),
        pair_presence.to(dtype=torch.float32),
        pos_weight=pos_weight.detach(),
    )

    selected_pair_ids, flat_targets = _select_pair_candidates(
        pair_gate_logits,
        interval_targets,
        train_topk=train_topk,
        random_negatives=random_negatives,
        max_pairs=max_pairs,
    )
    selected_pairs = model.build_selected_pair_indices(
        selected_pair_ids,
    )
    semi_crf_loss, track_count, interval_count = (
        compute_factorized_pair_interval_loss(
            pitch_interval_query,
            pitch_interval_key,
            pitch_interval_diag,
            instrument_interval_query,
            instrument_interval_key,
            instrument_interval_diag,
            selected_pairs.batch_indices,
            selected_pairs.instrument_indices,
            selected_pairs.pitch_indices,
            flat_targets.intervals,
            valid_lengths,
            length_scaling=length_scaling,
            length_penalty=length_penalty,
            track_batch_size=128
            if args is None
            else int(getattr(args, "semi_crf_track_batch_size", 128)),
            false_negative_cost=semi_crf_false_negative_cost,
            false_positive_cost=semi_crf_false_positive_cost,
            backend=semi_crf_loss_backend,
        )
    )

    total_loss = (
        semi_crf_loss * semi_crf_loss_weight + pair_gate_loss * pair_gate_loss_weight
    )

    zero = interval_features.sum() * 0.0
    interval_presence_loss = zero
    interval_offset_loss = zero
    interval_boundary_loss = zero
    interval_boundary_interval_count = torch.tensor(
        0, device=interval_features.device, dtype=torch.long
    )

    if model.supports_interval_boundaries():
        boundary_logits, entries = model.predict_flat_interval_boundaries(
            interval_features,
            selected_pairs,
            flat_targets.intervals,
        )
        if entries:
            has_onset, has_offset, onset_offsets, offset_offsets = (
                _gather_flat_boundary_targets(
                    flat_targets,
                    entries,
                    device=boundary_logits.device,
                )
            )
            presence_logits, offset_logits = boundary_logits.chunk(2, dim=-1)
            boundary_targets = torch.stack([has_onset, has_offset], dim=-1)
            interval_presence_loss = F.binary_cross_entropy_with_logits(
                presence_logits,
                boundary_targets,
            )
            offset_targets = torch.stack([onset_offsets, offset_offsets], dim=-1)
            offset_targets = torch.clamp(offset_targets, 0.0, 1.0) * 0.99 + 0.005
            offset_dist = torch.distributions.ContinuousBernoulli(logits=offset_logits)
            interval_offset_loss = (
                -offset_dist.log_prob(offset_targets).sum(dim=-1).mean()
            )
            interval_boundary_loss = interval_presence_loss + interval_offset_loss
            interval_boundary_interval_count = torch.tensor(
                len(entries), device=interval_features.device, dtype=torch.long
            )
            total_loss = total_loss + (
                interval_presence_loss * interval_presence_loss_weight
                + interval_offset_loss * interval_offset_loss_weight
            )

    return total_loss, {
        "total_loss": total_loss,
        "semi_crf_loss": semi_crf_loss,
        "pair_gate_loss": pair_gate_loss,
        "pair_gate_positive_count": positive_count.detach().to(
            device=interval_features.device
        ),
        "pair_gate_pos_weight": pos_weight.detach().to(device=interval_features.device),
        "selected_pair_count": torch.tensor(
            track_count, device=interval_features.device, dtype=torch.long
        ),
        "semi_crf_track_count": torch.tensor(
            track_count, device=interval_features.device, dtype=torch.long
        ),
        "semi_crf_interval_count": torch.tensor(
            interval_count, device=interval_features.device, dtype=torch.long
        ),
        "semi_crf_false_negative_cost": interval_features.new_tensor(
            semi_crf_false_negative_cost
        ),
        "semi_crf_false_positive_cost": interval_features.new_tensor(
            semi_crf_false_positive_cost
        ),
        "interval_boundary_loss": interval_boundary_loss,
        "interval_presence_loss": interval_presence_loss,
        "interval_offset_loss": interval_offset_loss,
        "interval_boundary_interval_count": interval_boundary_interval_count,
    }




def compute_losses(
    outputs: dict[str, torch.Tensor | None],
    batch: dict[str, Any],
    args: Any | None = None,
    model: AudioSemiCRFTransformer | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if model is None:
        raise ValueError("compute_losses requires the model")
    if model.semi_crf_version == "v1":
        return compute_v1_losses(outputs, batch, args=args, model=model)
    return _compute_v2_losses(outputs, batch, args=args, model=model)
