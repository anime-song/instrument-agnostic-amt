import numpy as np
import pytest

from instrument_agnostic_amt.beat_chord.decoding.beat_grid import (
    BeatGridDPConfig,
    decode_beats_with_meter_grid_dp,
)


def _probabilities(
    frame_count: int,
    beat_frames: list[int],
    downbeat_frames: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    beat = np.full(frame_count, 0.03, dtype=np.float64)
    downbeat = np.full(frame_count, 0.02, dtype=np.float64)
    beat[beat_frames] = 0.95
    downbeat[downbeat_frames] = 0.95
    return beat, downbeat


def _config(**overrides: object) -> BeatGridDPConfig:
    values: dict[str, object] = {
        "sample_rate": 100,
        "hop_length": 10,
        "tolerance_frames": 1,
        "max_bar_count": 2,
        "beam_size": 48,
        "max_leading_seconds": 1.0,
        "max_trailing_seconds": 1.0,
    }
    values.update(overrides)
    return BeatGridDPConfig(**values)


def test_dp_grid_rejects_spurious_half_bar_downbeats() -> None:
    frame_count = 101
    beat, downbeat = _probabilities(
        frame_count,
        list(range(0, frame_count, 5)),
        list(range(0, frame_count, 20)),
    )
    spurious = [10, 30, 50, 70, 90]
    downbeat[spurious] = 0.75

    result = decode_beats_with_meter_grid_dp(
        beat_probabilities=beat,
        downbeat_probabilities=downbeat,
        meter_logits=np.zeros((frame_count, 1)),
        meter_classes=[(4, 4)],
        config=_config(),
    )

    assert result.downbeat_frames == (0, 20, 40, 60, 80, 100)
    assert result.rejected_downbeat_candidates == tuple(spurious)


def test_dp_grid_infers_a_downbeat_missing_from_the_downbeat_head() -> None:
    frame_count = 101
    beat, downbeat = _probabilities(
        frame_count,
        list(range(0, frame_count, 5)),
        list(range(0, frame_count, 20)),
    )
    downbeat[40] = 0.02

    result = decode_beats_with_meter_grid_dp(
        beat_probabilities=beat,
        downbeat_probabilities=downbeat,
        meter_logits=np.zeros((frame_count, 1)),
        meter_classes=[(4, 4)],
        config=_config(),
    )

    assert result.downbeat_frames == (0, 20, 40, 60, 80, 100)
    assert result.inferred_downbeat_frames == (40,)


def test_dp_grid_follows_a_large_real_tempo_change() -> None:
    downbeats = [0, 20, 40, 72, 104]
    beat_frames = [
        0,
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
        48,
        56,
        64,
        72,
        80,
        88,
        96,
        104,
    ]
    beat, downbeat = _probabilities(105, beat_frames, downbeats)

    result = decode_beats_with_meter_grid_dp(
        beat_probabilities=beat,
        downbeat_probabilities=downbeat,
        meter_logits=np.zeros((105, 1)),
        meter_classes=[(4, 4)],
        config=_config(),
    )

    assert result.downbeat_frames == tuple(downbeats)
    tempos_by_bar = [
        round(segment.quarter_note_bpm)
        for segment in result.meter_segments
        for _bar in range(segment.bar_count)
    ]
    assert tempos_by_bar == [
        120,
        120,
        75,
        75,
    ]


def test_dp_grid_prefers_repeating_seven_four_over_four_plus_three() -> None:
    frame_count = 106
    true_downbeats = [0, 35, 70, 105]
    false_split_points = [20, 55, 90]
    beat, downbeat = _probabilities(
        frame_count,
        list(range(0, frame_count, 5)),
        true_downbeats,
    )
    downbeat[false_split_points] = 0.72

    meter_classes = [(3, 4), (4, 4), (7, 4)]
    meter_logits = np.zeros((frame_count, len(meter_classes)))
    meter_logits[:, 2] = -2.0
    for bar_start in (0, 35, 70):
        meter_logits[bar_start : bar_start + 20, 1] = 2.0
        meter_logits[bar_start + 20 : bar_start + 35, 0] = 2.0

    result = decode_beats_with_meter_grid_dp(
        beat_probabilities=beat,
        downbeat_probabilities=downbeat,
        meter_logits=meter_logits,
        meter_classes=meter_classes,
        config=_config(),
    )

    assert result.downbeat_frames == tuple(true_downbeats)
    meters_by_bar = [
        segment.meter_num
        for segment in result.meter_segments
        for _bar in range(segment.bar_count)
    ]
    assert meters_by_bar == [7, 7, 7]
    assert all(
        segment.meter_evidence_source in {"4/4+3/4", "3/4+4/4"}
        for segment in result.meter_segments
    )
    assert result.rejected_downbeat_candidates == tuple(false_split_points)


def test_numba_grid_decoder_is_exactly_equal_to_python_decoder() -> None:
    pytest.importorskip("numba")
    frame_count = 106
    true_downbeats = [0, 35, 70, 105]
    beat, downbeat = _probabilities(
        frame_count,
        list(range(0, frame_count, 5)),
        true_downbeats,
    )
    meter_classes = [(3, 4), (4, 4), (7, 4)]
    meter_logits = np.zeros((frame_count, len(meter_classes)))
    meter_logits[:, 2] = -2.0
    for bar_start in (0, 35, 70):
        meter_logits[bar_start : bar_start + 20, 1] = 2.0
        meter_logits[bar_start + 20 : bar_start + 35, 0] = 2.0

    arguments = {
        "beat_probabilities": beat,
        "downbeat_probabilities": downbeat,
        "meter_logits": meter_logits,
        "meter_classes": meter_classes,
    }
    python_result = decode_beats_with_meter_grid_dp(
        **arguments,
        config=_config(use_jit_grid=False),
    )
    jit_result = decode_beats_with_meter_grid_dp(
        **arguments,
        config=_config(use_jit_grid=True),
    )

    assert jit_result == python_result
