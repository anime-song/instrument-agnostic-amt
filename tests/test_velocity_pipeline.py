from __future__ import annotations

import csv
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from recipes.velocity.config import PseudoLabelConfig
from recipes.velocity.data.calibration import (
    build_velocity_sweep_midi,
)
from recipes.velocity.data.curve import (
    CalibrationAnalysisConfig,
    analyze_sweep_files,
    isotonic_increasing,
)
from recipes.velocity.data.index import discover_amt_cbnet_items
from instrument_agnostic_amt.velocity.data.midi import (
    MidiNoteTable,
    canonicalize_amt_midi,
)
from recipes.velocity.data.pseudo import (
    build_pseudo_labels_from_audio,
)


def _note_table(starts: np.ndarray) -> MidiNoteTable:
    count = int(starts.size)
    return MidiNoteTable(
        start_seconds=starts.astype(np.float64),
        end_seconds=(starts + 0.2).astype(np.float64),
        pitch=np.arange(60, 60 + count, dtype=np.int16),
        input_velocity=np.full(count, 100, dtype=np.int16),
        program=np.zeros(count, dtype=np.int16),
        is_drum=np.zeros(count, dtype=np.bool_),
        track_index=np.zeros(count, dtype=np.int16),
    )


def test_pseudo_velocity_rank_tracks_onset_strength() -> None:
    sample_rate = 8_000
    waveform = np.zeros((sample_rate * 3, 1), dtype=np.float32)
    starts = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    for start, amplitude in zip(starts, (0.08, 0.16, 0.32, 0.64)):
        left = int((start + 0.015) * sample_rate)
        right = int((start + 0.13) * sample_rate)
        time = np.arange(right - left, dtype=np.float32) / sample_rate
        waveform[left:right, 0] = amplitude * np.sin(2.0 * np.pi * 440.0 * time)

    arrays, summary = build_pseudo_labels_from_audio(
        waveform,
        sample_rate=sample_rate,
        note_table=_note_table(starts),
        config=PseudoLabelConfig(),
    )

    assert summary.valid_note_count == 4
    assert np.all(arrays["pseudo_valid"])
    assert np.all(np.diff(arrays["pseudo_velocity_rank"]) > 0.0)
    assert np.all(np.diff(arrays["pseudo_velocity"]) > 0)


def test_canonicalize_amt_midi_replaces_dynamics(tmp_path: Path) -> None:
    source_path = tmp_path / "source.mid"
    output_path = tmp_path / "canonical.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.extend(
        [
            mido.Message("control_change", control=7, value=84, time=0),
            mido.Message("control_change", control=11, value=96, time=0),
            mido.Message("control_change", control=64, value=127, time=0),
            mido.Message("note_on", note=60, velocity=100, time=120),
            mido.Message("note_off", note=60, velocity=0, time=240),
        ]
    )
    midi.tracks.append(track)
    midi.save(source_path)

    canonicalize_amt_midi(source_path, output_path, canonical_velocity=80)
    output = mido.MidiFile(output_path)
    messages = [message for track in output.tracks for message in track]
    controls = [message.control for message in messages if message.type == "control_change"]
    note_ons = [
        message
        for message in messages
        if message.type == "note_on" and message.velocity > 0
    ]

    assert controls == [64]
    assert [message.velocity for message in note_ons] == [80]
    assert note_ons[0].time == 120


def test_discover_amt_cbnet_items_keeps_missing_midi(tmp_path: Path) -> None:
    source_root = tmp_path / "midi_dataset"
    song_dir = source_root / "stems" / "song_a"
    song_dir.mkdir(parents=True)
    (source_root / "midis").mkdir()
    (source_root / "merged").mkdir()
    (song_dir / "song_a_bass.wav").touch()
    (song_dir / "song_a_piano.wav").touch()
    (source_root / "midis" / "song_a_bass.mid").touch()
    (source_root / "merged" / "song_a.mid").touch()

    items = discover_amt_cbnet_items(source_root)

    assert [(item.stem_name, item.has_midi) for item in items] == [
        ("bass", True),
        ("piano", False),
    ]
    assert all(item.merged_midi_path is not None for item in items)


def test_velocity_sweep_has_expected_events() -> None:
    midi, events = build_velocity_sweep_midi(
        program=24,
        pitches=(48, 60),
        velocities=(32, 96),
        note_seconds=0.5,
        gap_seconds=0.1,
    )

    assert len(events) == 4
    assert not any(event.is_drum for event in events)
    assert [note.velocity for note in midi.instruments[0].notes] == [32, 96, 32, 96]
    assert [control.number for control in midi.instruments[0].control_changes] == [7, 11]


def test_isotonic_increasing_pools_velocity_reversal() -> None:
    fitted = isotonic_increasing((0.0, 2.0, 1.0, 3.0))

    assert np.allclose(fitted, (0.0, 1.5, 1.5, 3.0))


def test_analyze_sweep_files_measures_and_fits_curve(tmp_path: Path) -> None:
    sample_rate = 8_000
    starts = (0.25, 0.85, 1.45, 2.05)
    velocities = (16, 48, 80, 127)
    amplitudes = (0.40, 0.10, 0.20, 0.40)
    waveform = np.zeros((sample_rate * 3, 1), dtype=np.float32)
    tail_left = int((starts[0] - 0.14) * sample_rate)
    tail_right = int(starts[0] * sample_rate)
    tail_time = np.arange(tail_right - tail_left, dtype=np.float32) / sample_rate
    waveform[tail_left:tail_right, 0] = 0.40 * np.sin(
        2.0 * np.pi * 220.0 * tail_time
    )
    for start, amplitude in zip(starts, amplitudes):
        left = int(start * sample_rate)
        right = int((start + 0.3) * sample_rate)
        time = np.arange(right - left, dtype=np.float32) / sample_rate
        waveform[left:right, 0] = amplitude * np.sin(2.0 * np.pi * 220.0 * time)
    wav_path = tmp_path / "program_000.wav"
    sf.write(wav_path, waveform, sample_rate, subtype="FLOAT")

    events_path = tmp_path / "sweep_events.csv"
    with events_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "program",
                "is_drum",
                "pitch",
                "velocity",
                "start_seconds",
                "end_seconds",
                "midi_path",
            ),
        )
        writer.writeheader()
        for start, velocity in zip(starts, velocities):
            writer.writerow(
                {
                    "program": 0,
                    "is_drum": 0,
                    "pitch": 60,
                    "velocity": velocity,
                    "start_seconds": start,
                    "end_seconds": start + 0.3,
                    "midi_path": "midi/program_000.mid",
                }
            )

    manifest_path = tmp_path / "render_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("program", "is_drum", "midi_path", "wav_path", "sample_rate"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "program": 0,
                "is_drum": 0,
                "midi_path": "midi/program_000.mid",
                "wav_path": wav_path.name,
                "sample_rate": sample_rate,
            }
        )

    rows, summary = analyze_sweep_files(
        events_path,
        manifest_path,
        config=CalibrationAnalysisConfig(signal_end_ms=280.0),
    )

    fitted = np.asarray([row["fitted_level_dbfs"] for row in rows], dtype=np.float64)
    assert summary["event_count"] == 4
    assert summary["valid_event_count"] == 4
    assert summary["raw_monotonic_violation_count"] == 1
    assert summary["tail_contaminated_event_count"] == 1
    assert rows[0]["fit_observed"] == 0
    assert rows[0]["tail_contaminated"] == 1
    assert np.all(np.diff(fitted) >= 0.0)
    assert rows[-1]["relative_level_db"] == 0.0
