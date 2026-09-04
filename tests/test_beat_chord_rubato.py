from __future__ import annotations

import random

import pytest
import torch

from recipes.beat_chord.datasets.augment import (
    MidiAugmentConfig,
    RubatoGesture,
    RubatoTimeMap,
    transform_event_times,
    transform_intervals,
    warp_roll_time,
)


def test_rubato_time_map_is_reproducible_monotonic_and_compensated() -> None:
    first = RubatoTimeMap.sample(
        duration_sec=25.0,
        strength=0.15,
        period_sec=4.0,
        seed=1234,
    )
    second = RubatoTimeMap.sample(
        duration_sec=25.0,
        strength=0.15,
        period_sec=4.0,
        seed=1234,
    )

    assert first == second
    assert first.gestures
    assert first.map_time(0.0) == pytest.approx(0.0)
    assert first.map_time(25.0) == pytest.approx(25.0)
    for gesture in first.gestures:
        assert first.map_time(gesture.start_sec) == pytest.approx(gesture.start_sec)
        assert first.map_time(gesture.end_sec) == pytest.approx(gesture.end_sec)

    source_times = torch.linspace(0.0, 25.0, 2001).tolist()
    mapped_times = torch.tensor([first.map_time(value) for value in source_times])
    local_slopes = torch.diff(mapped_times) / (25.0 / 2000.0)
    assert torch.all(local_slopes > 0.0)
    assert float(local_slopes.std()) > 0.01
    assert float(local_slopes.min()) >= 0.84
    assert float(local_slopes.max()) <= 1.16


def test_rubato_config_samples_an_independent_reproducible_curve_seed() -> None:
    config = MidiAugmentConfig(
        rubato_prob=1.0,
        rubato_strength=0.12,
        rubato_period_sec=3.5,
    )
    first = config.sample(random.Random(99))
    second = config.sample(random.Random(99))

    assert first == second
    assert first.rubato_strength == pytest.approx(0.12)
    assert not first.build_rubato_time_map(10.0).is_identity

    disabled = MidiAugmentConfig(rubato_prob=0.0).sample(random.Random(99))
    assert disabled.rubato_strength == 0.0
    assert disabled.build_rubato_time_map(10.0).is_identity


def test_roll_events_and_intervals_share_the_same_rubato_map() -> None:
    time_map = RubatoTimeMap(
        duration_sec=8.0,
        gestures=(RubatoGesture(0.0, 8.0, 0.8),),
    )
    roll = torch.zeros((2, 8, 1), dtype=torch.float32)
    roll[0, 4, 0] = 1.0
    roll[1, 4, 0] = 1.0

    warped = warp_roll_time(roll, time_map)
    transformed_events = transform_event_times(
        [12.0],
        source_window_start_sec=10.0,
        stretch_factor=2.0,
        target_window_sec=8.0,
        rubato_time_map=time_map,
    )
    transformed_intervals = transform_intervals(
        [(11.0, 13.0)],
        source_window_start_sec=10.0,
        stretch_factor=2.0,
        target_window_sec=8.0,
        rubato_time_map=time_map,
    )

    assert transformed_events == pytest.approx([4.8])
    assert transformed_intervals == pytest.approx([(2.45, 6.45)])
    assert warped[1, 5, 0] == 1.0
    assert warped[1, :, 0].sum() == 1.0
    assert warped[0, 5, 0] == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rubato_prob": -0.1}, "rubato_prob"),
        ({"rubato_prob": 1.1}, "rubato_prob"),
        ({"rubato_strength": 0.0}, "rubato_strength"),
        ({"rubato_strength": 0.8}, "rubato_strength"),
        ({"rubato_period_sec": 0.0}, "rubato_period_sec"),
    ],
)
def test_rubato_config_rejects_invalid_ranges(
    kwargs: dict[str, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MidiAugmentConfig(**kwargs)
