from __future__ import annotations

from pathlib import Path

import mido

from instrument_agnostic_amt.beat_chord.tempo_map_export import (
    MeterSegmentSpec,
    export_tempo_mapped_midi,
)


def _append_absolute_events(
    track: mido.MidiTrack,
    events: list[tuple[int, mido.Message | mido.MetaMessage]],
) -> None:
    previous_tick = 0
    for tick, message in sorted(events, key=lambda item: item[0]):
        track.append(message.copy(time=int(tick) - previous_tick))
        previous_tick = int(tick)


def _write_source_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    conductor = mido.MidiTrack()
    _append_absolute_events(
        conductor,
        [
            (0, mido.MetaMessage("track_name", name="Source Conductor")),
            (0, mido.MetaMessage("set_tempo", tempo=500_000)),
            (0, mido.MetaMessage("time_signature", numerator=4, denominator=4)),
            (0, mido.MetaMessage("key_signature", key="D")),
            (960, mido.MetaMessage("set_tempo", tempo=600_000)),
        ],
    )
    midi.tracks.append(conductor)

    performance = mido.MidiTrack()
    _append_absolute_events(
        performance,
        [
            (0, mido.MetaMessage("track_name", name="Piano")),
            (0, mido.Message("program_change", channel=0, program=0)),
            (240, mido.Message("note_on", channel=0, note=60, velocity=90)),
            (720, mido.Message("note_off", channel=0, note=60, velocity=0)),
            (1200, mido.Message("note_on", channel=0, note=64, velocity=80)),
            (1680, mido.Message("note_off", channel=0, note=64, velocity=0)),
            (3000, mido.Message("note_on", channel=0, note=67, velocity=70)),
            (3400, mido.Message("note_off", channel=0, note=67, velocity=0)),
        ],
    )
    midi.tracks.append(performance)
    midi.save(str(path))


def _merged_events_in_seconds(
    path: Path,
) -> list[tuple[float, mido.Message | mido.MetaMessage]]:
    midi = mido.MidiFile(str(path))
    tempo = 500_000
    seconds = 0.0
    events: list[tuple[float, mido.Message | mido.MetaMessage]] = []
    for message in mido.merge_tracks(midi.tracks):
        seconds += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = int(message.tempo)
        events.append((seconds, message))
    return events


def _note_event_times(path: Path) -> list[float]:
    return [
        seconds
        for seconds, message in _merged_events_in_seconds(path)
        if message.type in {"note_on", "note_off"}
    ]


def _tempo_events(path: Path) -> list[tuple[int, int]]:
    midi = mido.MidiFile(str(path))
    absolute_tick = 0
    events: list[tuple[int, int]] = []
    for message in midi.tracks[0]:
        absolute_tick += int(message.time)
        if message.type == "set_tempo":
            events.append((absolute_tick, int(message.tempo)))
    return events


def test_export_builds_continuous_tempo_meter_and_chord_map_without_note_shift(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mid"
    output_path = tmp_path / "mapped.mid"
    _write_source_midi(source_path)
    original_note_times = _note_event_times(source_path)

    beat_times = [
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        6.5,
        7.0,
        7.5,
        8.0,
    ]
    result = export_tempo_mapped_midi(
        source_midi_path=source_path,
        output_midi_path=output_path,
        beat_times=beat_times,
        meter_segments=[
            MeterSegmentSpec(1.0, 3.0, 4, 4, bar_count=1, score=0.9),
            MeterSegmentSpec(3.0, 7.0, 4, 4, bar_count=2, score=0.7),
            MeterSegmentSpec(7.0, 8.5, 3, 4, bar_count=1, score=0.8),
        ],
        chord_segments=[
            {
                "start": 0.5,
                "end": 3.0,
                "chord": "C:maj7",
                "combined_label": "C:maj7 [IM7|T]",
            },
            {"start": 3.0, "end": 7.0, "chord": "G:7"},
            {"start": 7.0, "end": 9.0, "chord": "A:min"},
        ],
        key_segments=[
            {"start": 0.5, "end": 7.0, "key": "Db"},
            {"start": 7.0, "end": 9.0, "key": "Gb"},
        ],
        duration_seconds=9.0,
        ticks_per_beat=960,
    )

    assert output_path.exists()
    assert result.used_predicted_tempo is True
    assert result.time_signature_count == 2
    assert result.chord_marker_count == 3
    assert result.key_signature_count == 2
    assert any(region["source"] == "interpolated" for region in result.regions)
    assert result.max_note_drift_seconds < 0.001

    output_note_times = _note_event_times(output_path)
    assert len(output_note_times) == len(original_note_times)
    assert (
        max(
            abs(output - original)
            for output, original in zip(output_note_times, original_note_times)
        )
        < 0.001
    )

    events = _merged_events_in_seconds(output_path)
    signatures = [
        (seconds, message.numerator, message.denominator)
        for seconds, message in events
        if message.type == "time_signature"
    ]
    assert signatures[0][1:] == (4, 4)
    assert abs(signatures[0][0]) < 0.001
    assert signatures[1][1:] == (3, 4)
    assert abs(signatures[1][0] - 7.0) < 0.001

    mapped_midi = mido.MidiFile(str(output_path))
    absolute_tick = 0
    signature_ticks: list[int] = []
    for message in mapped_midi.tracks[0]:
        absolute_tick += int(message.time)
        if message.type == "time_signature":
            signature_ticks.append(absolute_tick)
    assert signature_ticks[0] % (mapped_midi.ticks_per_beat * 4) == 0

    chord_markers = [
        (seconds, message.text)
        for seconds, message in events
        if message.type == "marker"
    ]
    assert [text for _, text in chord_markers] == [
        "C:maj7 [IM7|T]",
        "G:7",
        "A:min",
    ]
    assert (
        max(
            abs(actual - expected)
            for (actual, _), expected in zip(chord_markers, [0.5, 3.0, 7.0])
        )
        < 0.001
    )

    key_signatures = [
        (seconds, message.key)
        for seconds, message in events
        if message.type == "key_signature"
    ]
    assert [key for _, key in key_signatures] == ["Db", "Gb"]
    assert (
        max(
            abs(actual - expected)
            for (actual, _), expected in zip(key_signatures, [0.5, 7.0])
        )
        < 0.001
    )


def test_export_merges_small_grid_jitter_into_a_stable_bar_tempo(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mid"
    output_path = tmp_path / "stable.mid"
    _write_source_midi(source_path)

    result = export_tempo_mapped_midi(
        source_midi_path=source_path,
        output_midi_path=output_path,
        beat_times=[
            0.0,
            0.48,
            1.02,
            1.49,
            2.0,
            2.52,
            3.0,
            3.51,
            4.0,
            4.48,
            5.0,
            5.48,
            6.0,
        ],
        meter_segments=[
            MeterSegmentSpec(0.0, 2.0, 4, 4),
            MeterSegmentSpec(2.0, 4.0, 4, 4),
            MeterSegmentSpec(4.0, 6.0, 4, 4),
        ],
        chord_segments=[],
        duration_seconds=6.0,
    )

    assert _tempo_events(output_path) == [(0, 500_000)]
    assert result.tempo_event_count == 1
    assert all(region["tempo_mode"] == "smoothed_bar" for region in result.regions)
    assert result.max_note_drift_seconds < 0.001


def test_export_keeps_each_detected_downbeat_within_one_ms(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mid"
    output_path = tmp_path / "bar_jitter.mid"
    _write_source_midi(source_path)

    export_tempo_mapped_midi(
        source_midi_path=source_path,
        output_midi_path=output_path,
        beat_times=[
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.01,
            4.5,
            5.0,
            5.5,
            6.0,
        ],
        meter_segments=[
            MeterSegmentSpec(0.0, 2.0, 4, 4),
            MeterSegmentSpec(2.0, 4.01, 4, 4),
            MeterSegmentSpec(4.01, 6.0, 4, 4),
        ],
        chord_segments=[],
        duration_seconds=6.0,
    )

    tempo_times = [
        seconds
        for seconds, message in _merged_events_in_seconds(output_path)
        if message.type == "set_tempo"
    ]
    assert len(tempo_times) == 3
    assert max(
        abs(actual - expected)
        for actual, expected in zip(tempo_times, [0.0, 2.0, 4.01])
    ) < 0.001


def test_export_keeps_a_clear_bar_level_tempo_change(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mid"
    output_path = tmp_path / "tempo_change.mid"
    _write_source_midi(source_path)

    result = export_tempo_mapped_midi(
        source_midi_path=source_path,
        output_midi_path=output_path,
        beat_times=[
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.7,
            5.4,
            6.1,
            6.8,
        ],
        meter_segments=[
            MeterSegmentSpec(0.0, 2.0, 4, 4),
            MeterSegmentSpec(2.0, 4.0, 4, 4),
            MeterSegmentSpec(4.0, 6.8, 4, 4),
        ],
        chord_segments=[],
        duration_seconds=6.8,
    )

    assert _tempo_events(output_path) == [(0, 500_000), (7_680, 700_000)]
    assert result.tempo_event_count == 2
    assert result.max_note_drift_seconds < 0.001


def test_export_preserves_clear_ritardando_as_a_smoothed_beat_curve(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mid"
    output_path = tmp_path / "ritardando.mid"
    _write_source_midi(source_path)

    result = export_tempo_mapped_midi(
        source_midi_path=source_path,
        output_midi_path=output_path,
        beat_times=[0.0, 0.4, 0.9, 1.5],
        meter_segments=[MeterSegmentSpec(0.0, 2.3, 4, 4)],
        chord_segments=[],
        duration_seconds=5.0,
    )

    tempos = _tempo_events(output_path)
    assert [tick for tick, _tempo in tempos] == [0, 960, 1_920, 2_880]
    assert [tempo for _tick, tempo in tempos] == sorted(
        tempo for _tick, tempo in tempos
    )
    assert result.regions[0]["tempo_mode"] == "expressive_beat_curve"
    assert result.max_note_drift_seconds < 0.001


def test_export_folds_a_one_beat_meter_gap_into_the_preceding_bar(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mid"
    output_path = tmp_path / "meter_gap.mid"
    _write_source_midi(source_path)

    result = export_tempo_mapped_midi(
        source_midi_path=source_path,
        output_midi_path=output_path,
        beat_times=[
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            2.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            5.5,
        ],
        meter_segments=[
            MeterSegmentSpec(0.0, 2.0, 4, 4),
            MeterSegmentSpec(2.0, 4.0, 4, 4),
            MeterSegmentSpec(4.5, 6.0, 3, 4),
        ],
        chord_segments=[],
        duration_seconds=6.0,
    )

    signatures = [
        (seconds, message.numerator, message.denominator)
        for seconds, message in _merged_events_in_seconds(output_path)
        if message.type == "time_signature"
    ]
    assert [(num, den) for _seconds, num, den in signatures] == [
        (4, 4),
        (5, 4),
        (3, 4),
    ]
    assert max(
        abs(actual - expected)
        for (actual, _num, _den), expected in zip(signatures, [0.0, 2.0, 4.5])
    ) < 0.001
    assert any(region["source"] == "gap_extended" for region in result.regions)
    assert result.max_note_drift_seconds < 0.001


def test_export_declares_initial_odd_meter_at_tick_zero(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mid"
    _write_source_midi(source_path)

    for numerator, denominator, first_downbeat in (
        (5, 8, 0.25),
        (7, 4, 0.8),
    ):
        output_path = tmp_path / f"mapped_{numerator}_{denominator}.mid"
        beat_duration = 0.2
        bar_duration = numerator * beat_duration
        beat_times = [
            first_downbeat + beat_index * beat_duration
            for beat_index in range(numerator)
        ]
        result = export_tempo_mapped_midi(
            source_midi_path=source_path,
            output_midi_path=output_path,
            beat_times=beat_times,
            meter_segments=[
                MeterSegmentSpec(
                    first_downbeat,
                    first_downbeat + bar_duration,
                    numerator,
                    denominator,
                    bar_count=1,
                    score=0.9,
                )
            ],
            chord_segments=[],
            duration_seconds=5.0,
            ticks_per_beat=960,
        )

        mapped_midi = mido.MidiFile(str(output_path))
        absolute_tick = 0
        signature_events: list[tuple[int, int, int]] = []
        tempo_ticks: list[int] = []
        for message in mapped_midi.tracks[0]:
            absolute_tick += int(message.time)
            if message.type == "time_signature":
                signature_events.append(
                    (absolute_tick, message.numerator, message.denominator)
                )
            elif message.type == "set_tempo":
                tempo_ticks.append(absolute_tick)

        assert signature_events == [(0, numerator, denominator)]
        first_bar_ticks = int(
            round(mapped_midi.ticks_per_beat * numerator * 4.0 / denominator)
        )
        assert tempo_ticks[1] == first_bar_ticks
        assert result.max_note_drift_seconds < 0.001


def test_export_falls_back_to_original_tempo_and_meter_without_decoded_meter(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mid"
    output_path = tmp_path / "fallback.mid"
    _write_source_midi(source_path)

    result = export_tempo_mapped_midi(
        source_midi_path=source_path,
        output_midi_path=output_path,
        beat_times=[],
        meter_segments=[],
        chord_segments=[{"start": 1.0, "end": 2.0, "chord": "F:maj"}],
        duration_seconds=5.0,
    )

    assert result.used_predicted_tempo is False
    assert result.max_note_drift_seconds < 0.001
    signatures = [
        message
        for _, message in _merged_events_in_seconds(output_path)
        if message.type == "time_signature"
    ]
    assert [(message.numerator, message.denominator) for message in signatures] == [
        (4, 4)
    ]
    key_signatures = [
        message.key
        for _, message in _merged_events_in_seconds(output_path)
        if message.type == "key_signature"
    ]
    assert key_signatures == ["D"]


def test_export_raises_resolution_to_keep_extreme_meter_timing_below_one_ms(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "source.mid"
    output_path = tmp_path / "high_resolution.mid"
    _write_source_midi(source_path)

    result = export_tempo_mapped_midi(
        source_midi_path=source_path,
        output_midi_path=output_path,
        beat_times=[0.0],
        meter_segments=[MeterSegmentSpec(0.0, 1.6, 1, 32, bar_count=1, score=0.5)],
        chord_segments=[],
        duration_seconds=5.0,
        ticks_per_beat=960,
    )

    assert result.ticks_per_beat > 960
    assert result.max_note_drift_seconds < 0.001
