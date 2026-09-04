from __future__ import annotations

from pathlib import Path

import mido
import torch

from recipes.beat_chord.datasets import (
    MidiAugmentConfig,
    MidiKeyOnlyDataset,
    midi_chord_collate_fn,
    read_midi_key_segments,
)
from recipes.beat_chord.train import is_key_only_training_step
from recipes.beat_chord.chord import ChordConfig, ChordLoss


def _write_key_only_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)

    conductor = mido.MidiTrack()
    conductor.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    conductor.append(mido.MetaMessage("key_signature", key="C", time=0))
    conductor.append(mido.MetaMessage("marker", text="G:7", time=240))
    conductor.append(mido.MetaMessage("key_signature", key="Dm", time=240))
    conductor.append(mido.MetaMessage("end_of_track", time=480))
    midi.tracks.append(conductor)

    notes = mido.MidiTrack()
    notes.append(mido.MetaMessage("track_name", name="piano", time=0))
    notes.append(mido.Message("program_change", program=0, time=0))
    notes.append(mido.Message("note_on", note=60, velocity=100, time=0))
    notes.append(mido.Message("note_off", note=60, velocity=0, time=960))
    notes.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(notes)
    midi.save(path)


def test_read_midi_key_segments_ignores_predicted_chord_markers(
    tmp_path: Path,
) -> None:
    midi_path = tmp_path / "corrected.mid"
    _write_key_only_midi(midi_path)

    segments, duration_sec = read_midi_key_segments(midi_path)

    assert duration_sec == 1.0
    assert segments == [
        {"start_time": 0.0, "end_time": 0.5, "key": "C"},
        {"start_time": 0.5, "end_time": 1.0, "key": "Dm"},
    ]


def test_key_only_dataset_masks_every_non_key_chord_target(tmp_path: Path) -> None:
    midi_dir = tmp_path / "midis"
    midi_dir.mkdir()
    _write_key_only_midi(midi_dir / "corrected.mid")
    dataset = MidiKeyOnlyDataset(
        tmp_path,
        window_ms=1000,
        sample_rate=100,
        hop_length=10,
        pitch_min=21,
        pitch_max=108,
        num_input_channels=72,
        augment_config=MidiAugmentConfig(),
    )

    item = dataset[0]

    assert item["supervision"] == "key_only"
    assert item["midi_frames"].sum() > 0
    assert torch.all(item["root_chord_targets"] == -100)
    assert torch.all(item["bass_targets"] == -100)
    assert item["chord_boundary_mask"].sum() == 0
    assert item["chord_pitch_mask"].sum() == 0
    assert item["key_boundary_mask"].sum() > 0
    # D minor maps to the current model's relative-major class F.
    assert set(item["key_targets"].tolist()) == {0, 5}


def test_key_only_batches_are_spaced_by_training_step_interval() -> None:
    scheduled_steps = [
        completed_steps + 1
        for completed_steps in range(12)
        if is_key_only_training_step(
            completed_steps=completed_steps,
            interval=4,
        )
    ]

    assert scheduled_steps == [4, 8, 12]
    assert all(
        is_key_only_training_step(completed_steps=step, interval=1)
        for step in range(12)
    )


def test_key_only_loss_has_no_gradient_for_chord_outputs() -> None:
    num_frames = 40
    item = {
        "midi_frames": torch.zeros(72, num_frames, 88),
        "chord_boundary": torch.zeros(num_frames),
        "root_chord_targets": torch.full((num_frames,), -100, dtype=torch.long),
        "bass_targets": torch.full((num_frames,), -100, dtype=torch.long),
        "key_boundary": torch.zeros(num_frames),
        "key_targets": torch.zeros(num_frames, dtype=torch.long),
        "chord_pitch_targets": torch.zeros(num_frames, 25),
        "chord_boundary_mask": torch.zeros(num_frames),
        "key_boundary_mask": torch.ones(num_frames),
        "chord_pitch_mask": torch.zeros(num_frames),
        "song_name": "corrected",
        "supervision": "key_only",
        "augment_pitch_shift": 0,
        "augment_time_stretch": 1.0,
        "augment_rubato_strength": 0.0,
    }
    batch = midi_chord_collate_fn([item])
    outputs = {
        "chord_boundary_logits": torch.randn(1, num_frames, requires_grad=True),
        "root_chord_logits": torch.randn(1, num_frames, 3, requires_grad=True),
        "bass_logits": torch.randn(1, num_frames, 13, requires_grad=True),
        "key_boundary_logits": torch.randn(1, num_frames, requires_grad=True),
        "key_logits": torch.randn(1, num_frames, 13, requires_grad=True),
        "chord_pitch_logits": torch.randn(1, num_frames, 25, requires_grad=True),
    }

    loss, parts = ChordLoss(ChordConfig(), torch.ones(3))(outputs, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert parts["chord_boundary"] == 0
    assert parts["root_chord"] == 0
    assert parts["bass"] == 0
    assert parts["chord_pitch"] == 0
    for name in (
        "chord_boundary_logits",
        "root_chord_logits",
        "bass_logits",
        "chord_pitch_logits",
    ):
        assert outputs[name].grad is not None
        assert outputs[name].grad.abs().sum() == 0
    assert outputs["key_boundary_logits"].grad.abs().sum() > 0
    assert outputs["key_logits"].grad.abs().sum() > 0
