from __future__ import annotations

import numpy as np
import torch

from instrument_agnostic_amt.beat_chord.decoding.beat_grid import (
    BeatGridDPConfig,
    decode_beats_with_meter_grid_dp,
)
from instrument_agnostic_amt.beat_chord.decoding.meter_grouping import (
    group_boundary_log_odds_array,
    score_major_groupings,
)
from instrument_agnostic_amt.beat_chord.heads.meter_grouping import (
    major_grouping_loss,
)
from instrument_agnostic_amt.beat_chord.meter_grouping import (
    grouping_spec_for_meter,
)


def test_grouping_specs_only_cover_meters_that_need_major_groups() -> None:
    assert grouping_spec_for_meter(4, 4) is None

    six_eight = grouping_spec_for_meter(6, 8)
    assert six_eight is not None
    assert six_eight.mode == "fixed"
    assert six_eight.patterns == ((3, 3),)

    seven_four = grouping_spec_for_meter(7, 4)
    assert seven_four is not None
    assert seven_four.mode == "latent"
    assert seven_four.patterns == ((4, 3), (3, 4))

    six_sixteen = grouping_spec_for_meter(6, 16)
    assert six_sixteen is not None
    assert six_sixteen.mode == "fixed"
    assert six_sixteen.patterns == ((3, 3),)

    nine_four = grouping_spec_for_meter(9, 4)
    assert nine_four is not None
    assert nine_four.mode == "latent"
    assert {(3, 3, 3), (4, 5), (5, 4)} <= set(nine_four.patterns)

    twenty_one_eight = grouping_spec_for_meter(21, 8)
    assert twenty_one_eight is not None
    assert twenty_one_eight.mode == "latent"
    assert (3, 3, 3, 3, 3, 3, 3) in twenty_one_eight.patterns

    eleven_sixteen = grouping_spec_for_meter(11, 16)
    assert eleven_sixteen is not None
    assert eleven_sixteen.mode == "latent"
    assert all(sum(pattern) == 11 for pattern in eleven_sixteen.patterns)
    assert all(set(pattern) <= {2, 3} for pattern in eleven_sixteen.patterns)


def _bar_batch(
    *,
    meter_num: int,
    meter_den: int,
    boundary_accent_offset: int,
) -> dict[str, torch.Tensor]:
    beat_period = 5
    end_frame = meter_num * beat_period
    frame_count = end_frame + 1
    beat_targets = torch.zeros(1, frame_count)
    beat_targets[0, list(range(0, end_frame, beat_period))] = 1.0
    downbeat_targets = torch.zeros(1, frame_count)
    downbeat_targets[0, [0, end_frame]] = 1.0
    meter_targets = torch.zeros(1, frame_count, dtype=torch.long)
    beat_mask = torch.ones(1, frame_count)
    midi_frames = torch.zeros(1, 4, frame_count, 1)
    accent_frame = boundary_accent_offset * beat_period
    midi_frames[0, 2:, accent_frame, 0] = 1.0
    return {
        "beat_targets": beat_targets,
        "downbeat_targets": downbeat_targets,
        "meter_targets": meter_targets,
        "beat_mask": beat_mask,
        "midi_frames": midi_frames,
        "meter_classes": [(meter_num, meter_den)],
    }


def test_latent_grouping_uses_onset_accent_as_soft_evidence() -> None:
    batch = _bar_batch(
        meter_num=7,
        meter_den=4,
        boundary_accent_offset=4,
    )
    logits = torch.zeros_like(batch["beat_targets"], requires_grad=True)

    result = major_grouping_loss(
        group_boundary_logits=logits,
        beat_targets=batch["beat_targets"],
        downbeat_targets=batch["downbeat_targets"],
        meter_targets=batch["meter_targets"],
        beat_mask=batch["beat_mask"],
        midi_frames=batch["midi_frames"],
        meter_classes=batch["meter_classes"],
        tolerance=0,
        accent_loss_weight=1.0,
        accent_temperature=0.1,
    )
    result.loss.backward()

    assert result.supervised_bar_count == 1
    assert result.loss.item() > 0.0
    assert logits.grad is not None
    # 4+3 starts its second major group after beat four (frame 20), while
    # 3+4 would place it at frame 15. Gradient descent should prefer frame 20.
    assert logits.grad[0, 20] < logits.grad[0, 15]


def test_fixed_grouping_supervises_boundary_and_non_boundaries() -> None:
    batch = _bar_batch(
        meter_num=6,
        meter_den=8,
        boundary_accent_offset=3,
    )
    logits = torch.zeros_like(batch["beat_targets"], requires_grad=True)

    result = major_grouping_loss(
        group_boundary_logits=logits,
        beat_targets=batch["beat_targets"],
        downbeat_targets=batch["downbeat_targets"],
        meter_targets=batch["meter_targets"],
        beat_mask=batch["beat_mask"],
        midi_frames=batch["midi_frames"],
        meter_classes=batch["meter_classes"],
        tolerance=0,
        accent_loss_weight=1.0,
        accent_temperature=0.5,
    )
    result.loss.backward()

    assert result.supervised_bar_count == 1
    assert logits.grad is not None
    assert logits.grad[0, 15] < 0.0
    assert logits.grad[0, 10] > 0.0


def test_simple_meter_does_not_supervise_major_grouping() -> None:
    batch = _bar_batch(
        meter_num=4,
        meter_den=4,
        boundary_accent_offset=2,
    )
    logits = torch.zeros_like(batch["beat_targets"], requires_grad=True)

    result = major_grouping_loss(
        group_boundary_logits=logits,
        beat_targets=batch["beat_targets"],
        downbeat_targets=batch["downbeat_targets"],
        meter_targets=batch["meter_targets"],
        beat_mask=batch["beat_mask"],
        midi_frames=batch["midi_frames"],
        meter_classes=batch["meter_classes"],
        tolerance=1,
        accent_loss_weight=0.25,
        accent_temperature=0.5,
    )

    assert result.supervised_bar_count == 0
    assert result.loss.item() == 0.0


def test_decoder_group_score_selects_the_supported_five_four_pattern() -> None:
    probabilities = np.full(5, 0.5, dtype=np.float64)
    probabilities[2] = 0.9
    score, patterns = score_major_groupings(
        grid_frames=(0, 1, 2, 3, 4),
        meter_num=5,
        meter_den=4,
        bar_count=1,
        group_boundary_probabilities=probabilities,
        false_boundary_weight=0.25,
    )

    assert score > 0.0
    assert patterns == ((2, 3),)


def test_precomputed_group_boundary_log_odds_match_the_probability_path() -> None:
    # ``beat_grid._log_odds_array`` clamps to +-6, which would silently rescale
    # confident boundaries. The precomputed array must stay bit-identical to the
    # per-frame ``_logit`` it replaces.
    generator = np.random.default_rng(0)
    probabilities = 1.0 / (
        1.0 + np.exp(-generator.normal(0.0, 6.0, size=200))
    )
    grid_frames = tuple(range(48))

    from_probabilities = score_major_groupings(
        grid_frames=grid_frames,
        meter_num=6,
        meter_den=8,
        bar_count=8,
        group_boundary_probabilities=probabilities,
        false_boundary_weight=0.25,
    )
    from_log_odds = score_major_groupings(
        grid_frames=grid_frames,
        meter_num=6,
        meter_den=8,
        bar_count=8,
        group_boundary_probabilities=probabilities,
        false_boundary_weight=0.25,
        group_boundary_log_odds=group_boundary_log_odds_array(probabilities),
    )

    assert from_log_odds == from_probabilities


def test_grid_decoder_reports_major_grouping() -> None:
    frame_count = 51
    beat = np.full(frame_count, 0.02, dtype=np.float64)
    downbeat = np.full(frame_count, 0.01, dtype=np.float64)
    group = np.full(frame_count, 0.1, dtype=np.float64)
    beat[list(range(0, frame_count, 5))] = 0.95
    downbeat[[0, 25, 50]] = 0.95
    group[[10, 35]] = 0.95

    result = decode_beats_with_meter_grid_dp(
        beat_probabilities=beat,
        downbeat_probabilities=downbeat,
        group_boundary_probabilities=group,
        meter_logits=np.zeros((frame_count, 1)),
        meter_classes=[(5, 4)],
        config=BeatGridDPConfig(
            sample_rate=100,
            hop_length=10,
            tolerance_frames=1,
            max_bar_count=2,
            beam_size=24,
            max_leading_seconds=1.0,
            max_trailing_seconds=1.0,
        ),
    )

    assert result.downbeat_frames == (0, 25, 50)
    assert [segment.major_grouping for segment in result.meter_segments] == [
        (2, 3),
        (2, 3),
    ]
