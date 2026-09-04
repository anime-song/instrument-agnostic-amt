from __future__ import annotations

import random

import pytest

from recipes.beat_chord.datasets.meter_aware_crop import (
    MeterAwareCropConfig,
    choose_meter_aware_window_start,
)


def _events_for_bars(
    bars: list[tuple[float, float, int]],
    meter_classes: tuple[tuple[int, int], ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    beats: list[float] = []
    downbeats = [bar[0] for bar in bars]
    downbeats.append(bars[-1][1])
    for start_sec, end_sec, meter_index in bars:
        numerator = meter_classes[meter_index][0]
        duration = end_sec - start_sec
        beats.extend(
            start_sec + duration * beat_index / numerator
            for beat_index in range(numerator)
        )
    return tuple(beats), tuple(downbeats)


def test_meter_aware_crop_contains_a_complete_grouping_bar() -> None:
    meter_classes = ((4, 4), (7, 4), (5, 8))
    bars = [
        (0.0, 4.0, 0),
        (4.0, 11.0, 1),
        (11.0, 18.0, 1),
        (18.0, 20.5, 2),
        (20.5, 23.0, 2),
        (23.0, 27.0, 0),
    ]
    beat_times, downbeat_times = _events_for_bars(bars, meter_classes)

    selection = choose_meter_aware_window_start(
        meter_intervals=bars,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        meter_classes=meter_classes,
        meter_class_counts=(10_000.0, 1_000.0, 100.0),
        duration_sec=27.0,
        source_window_sec=10.0,
        sample_rate=100,
        hop_length=1,
        config=MeterAwareCropConfig(probability=1.0),
        rng=random.Random(7),
    )

    assert selection.used_meter_aware
    assert selection.target_meter_index in {1, 2}
    assert selection.target_bar_start_sec is not None
    assert selection.target_bar_end_sec is not None
    assert selection.window_start_sec <= selection.target_bar_start_sec
    assert (
        selection.window_start_sec + 10.0
        > selection.target_bar_end_sec
    )


def test_meter_aware_crop_ignores_simple_meters() -> None:
    meter_classes = ((4, 4),)
    bars = [(0.0, 4.0, 0), (4.0, 8.0, 0), (8.0, 12.0, 0)]
    beat_times, downbeat_times = _events_for_bars(bars, meter_classes)

    selection = choose_meter_aware_window_start(
        meter_intervals=bars,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        meter_classes=meter_classes,
        meter_class_counts=(1.0,),
        duration_sec=12.0,
        source_window_sec=6.0,
        sample_rate=100,
        hop_length=1,
        config=MeterAwareCropConfig(probability=1.0),
        rng=random.Random(3),
    )

    assert not selection.used_meter_aware
    assert selection.target_meter_index is None
    assert 0.0 <= selection.window_start_sec <= 6.0


def test_meter_aware_crop_rejects_incomplete_beat_annotation() -> None:
    meter_classes = ((7, 4),)
    bars = [(2.0, 9.0, 0), (9.0, 16.0, 0)]
    beat_times, downbeat_times = _events_for_bars(bars, meter_classes)
    incomplete_beats = tuple(
        beat for beat in beat_times if beat not in {8.0, 15.0}
    )

    selection = choose_meter_aware_window_start(
        meter_intervals=bars,
        beat_times=incomplete_beats,
        downbeat_times=downbeat_times,
        meter_classes=meter_classes,
        meter_class_counts=(1.0,),
        duration_sec=18.0,
        source_window_sec=10.0,
        sample_rate=100,
        hop_length=1,
        config=MeterAwareCropConfig(probability=1.0),
        rng=random.Random(5),
    )

    assert not selection.used_meter_aware


def test_meter_aware_crop_can_be_disabled_and_is_reproducible() -> None:
    meter_classes = ((5, 4),)
    bars = [(1.0, 6.0, 0), (6.0, 11.0, 0)]
    beat_times, downbeat_times = _events_for_bars(bars, meter_classes)
    arguments = dict(
        meter_intervals=bars,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        meter_classes=meter_classes,
        meter_class_counts=(1.0,),
        duration_sec=14.0,
        source_window_sec=8.0,
        sample_rate=100,
        hop_length=1,
    )

    disabled = choose_meter_aware_window_start(
        **arguments,
        config=MeterAwareCropConfig(probability=0.0),
        rng=random.Random(11),
    )
    first = choose_meter_aware_window_start(
        **arguments,
        config=MeterAwareCropConfig(probability=1.0),
        rng=random.Random(11),
    )
    second = choose_meter_aware_window_start(
        **arguments,
        config=MeterAwareCropConfig(probability=1.0),
        rng=random.Random(11),
    )

    assert not disabled.used_meter_aware
    assert first == second
    assert first.used_meter_aware


@pytest.mark.parametrize(
    "config",
    [
        MeterAwareCropConfig(probability=0.0),
        MeterAwareCropConfig(probability=1.0, rarity_power=0.0),
    ],
)
def test_meter_aware_crop_config_accepts_probability_endpoints(
    config: MeterAwareCropConfig,
) -> None:
    assert 0.0 <= config.probability <= 1.0
