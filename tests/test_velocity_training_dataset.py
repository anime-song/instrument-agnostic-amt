from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf

from recipes.velocity.collate import collate_velocity_batch
from recipes.velocity.split import assign_song_split
from recipes.velocity.stem_dataset import SyntheticStemVelocityDataset


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_training_label(
    path: Path,
    *,
    starts: tuple[float, ...],
    pitches: tuple[int, ...],
    velocities: tuple[int, ...],
    program: int,
    gain_db: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = len(starts)
    np.savez_compressed(
        path,
        note_start_seconds=np.asarray(starts, dtype=np.float64),
        note_end_seconds=np.asarray(starts, dtype=np.float64) + 0.3,
        note_pitch=np.asarray(pitches, dtype=np.int16),
        note_program=np.full(count, program, dtype=np.int16),
        note_is_drum=np.zeros(count, dtype=np.bool_),
        note_track_index=np.zeros(count, dtype=np.int16),
        target_velocity=np.asarray(velocities, dtype=np.int16),
        source_pseudo_confidence=np.ones(count, dtype=np.float32),
        rank_source=np.full(count, 2, dtype=np.int8),
        independently_randomized=np.zeros(count, dtype=np.bool_),
        stem_gain_db=np.asarray(gain_db, dtype=np.float32),
    )


def _build_training_root(tmp_path: Path) -> Path:
    root = tmp_path / "synthetic"
    sample_rate = 44_100
    time = np.arange(int(1.5 * sample_rate), dtype=np.float32) / sample_rate
    rendered_paths = {}
    for stem_name, frequency in (("bass", 220.0), ("piano", 330.0)):
        path = root / "rendered_stems" / "song_a" / "v000" / f"{stem_name}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        waveform = np.stack(
            (
                0.1 * np.sin(2.0 * np.pi * frequency * time),
                0.1 * np.sin(2.0 * np.pi * frequency * time),
            ),
            axis=1,
        )
        sf.write(path, waveform, sample_rate, subtype="FLOAT")
        rendered_paths[stem_name] = path

    label_a = root / "labels" / "song_a" / "v000" / "bass.npz"
    label_b = root / "labels" / "song_a" / "v000" / "piano.npz"
    _write_training_label(
        label_a,
        starts=(0.25, 1.25),
        pitches=(40, 43),
        velocities=(32, 96),
        program=32,
        gain_db=-3.0,
    )
    _write_training_label(
        label_b,
        starts=(0.5, 1.4),
        pitches=(60, 64),
        velocities=(64, 112),
        program=0,
        gain_db=3.0,
    )
    _write_csv(
        root / "examples.csv",
        [
            {
                "example_id": "song_a__v000",
                "song_id": "song_a",
                "variation": 0,
                "stem_count": 2,
                "mixture_path": "mixtures/song_a/v000.wav",
            }
        ],
    )
    _write_csv(
        root / "render_manifest.csv",
        [
            {
                "example_id": "song_a__v000",
                "stem_name": "bass",
                "rendered_stem_path": "rendered_stems/song_a/v000/bass.wav",
                "label_path": "labels/song_a/v000/bass.npz",
                "input_midi_path": "input_midis/song_a/bass.mid",
                "stem_gain_db": -3.0,
                "base_relative_level_db": -3.0,
            },
            {
                "example_id": "song_a__v000",
                "stem_name": "piano",
                "rendered_stem_path": "rendered_stems/song_a/v000/piano.wav",
                "label_path": "labels/song_a/v000/piano.npz",
                "input_midi_path": "input_midis/song_a/piano.mid",
                "stem_gain_db": 3.0,
                "base_relative_level_db": 3.0,
            },
        ],
    )
    return root


def test_song_split_keeps_variations_together() -> None:
    split = assign_song_split("song_a", seed=123)

    assert split == assign_song_split("song_a", seed=123)
    assert split in ("train", "validation", "test")


def test_velocity_dataset_resamples_windows_and_collates(tmp_path: Path) -> None:
    root = _build_training_root(tmp_path)
    dataset = SyntheticStemVelocityDataset(
        root,
        split="all",
        sample_rate=22_050,
        window_seconds=1.0,
        hop_seconds=1.0,
        use_gain_augmentation=True,
        gain_jitter_std_db=0.0,
        master_gain_min_db=0.0,
        master_gain_max_db=0.0,
    )

    assert len(dataset) == 2
    first = dataset[0]
    second = dataset[1]
    assert first["audio"].shape == (2, 2, 22_050)
    assert first["valid_audio_frames"].tolist() == [22_050, 22_050]
    assert first["stem_gain_db"].tolist() == [-3.0, 3.0]
    assert first["note_start_seconds"].tolist() == [0.25, 0.5]
    assert first["target_velocity"].tolist() == [32, 64]
    assert second["window_start_seconds"] == 0.5
    assert np.allclose(
        second["note_start_seconds"].numpy(),
        (0.0, 0.75, 0.9),
    )
    assert second["target_velocity"].tolist() == [64, 96, 112]

    batch = collate_velocity_batch([first, second])
    assert batch["audio"].shape == (2, 2, 2, 22_050)
    assert batch["target_velocity"].shape == (2, 3)
    assert batch["note_mask"].sum(dim=1).tolist() == [2, 3]
    assert batch["stem_gain_db"].shape == (2, 2)
    assert batch["stem_gain_mask"].all()


def test_velocity_dataset_uses_unmodified_render_level_by_default(
    tmp_path: Path,
) -> None:
    root = _build_training_root(tmp_path)
    dataset = SyntheticStemVelocityDataset(
        root,
        split="all",
        sample_rate=22_050,
        window_seconds=1.0,
        hop_seconds=1.0,
    )

    first = dataset[0]

    assert first["stem_gain_db"].tolist() == [0.0, 0.0]
    assert first["master_gain_db"] == 0.0
    assert np.isclose(float(first["audio"][0].abs().max()), 0.1, atol=2e-3)
    assert np.isclose(float(first["audio"][1].abs().max()), 0.1, atol=2e-3)


def test_velocity_dataset_uses_duration_seconds_from_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _build_training_root(tmp_path)
    _write_csv(
        root / "examples.csv",
        [
            {
                "example_id": "song_a__v000",
                "song_id": "song_a",
                "variation": 0,
                "stem_count": 2,
                "mixture_path": "mixtures/song_a/v000.wav",
                "duration_seconds": 1.5,
            }
        ],
    )

    info_call_count = 0
    original_info = sf.info

    def spy_info(*args: object, **kwargs: object) -> object:
        nonlocal info_call_count
        info_call_count += 1
        return original_info(*args, **kwargs)

    monkeypatch.setattr(sf, "info", spy_info)

    dataset = SyntheticStemVelocityDataset(
        root,
        split="all",
        sample_rate=22_050,
        window_seconds=1.0,
        hop_seconds=1.0,
    )

    assert dataset.examples[0].duration_seconds == 1.5
    assert info_call_count == 0
