from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

from instrument_agnostic_amt.velocity.data.midi import load_midi_note_table
from recipes.velocity.synthesis.config import SyntheticDataConfig
from recipes.velocity.synthesis.midi import write_target_velocity_midi
from recipes.velocity.synthesis.mix import (
    mix_rendered_stems,
    write_dataset_manifest,
)
from recipes.velocity.synthesis.plan import prepare_synthetic_plan
from recipes.velocity.synthesis.sampling import (
    sample_note_velocities,
    sample_stem_gains,
)


def _write_midi(path: Path, *, program: int, pitch_offset: int = 0) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=program)
    instrument.control_changes.extend(
        [
            pretty_midi.ControlChange(number=7, value=80, time=0.0),
            pretty_midi.ControlChange(number=11, value=90, time=0.0),
        ]
    )
    instrument.notes.extend(
        [
            pretty_midi.Note(
                velocity=100,
                pitch=64 + pitch_offset,
                start=1.0,
                end=1.3,
            ),
            pretty_midi.Note(
                velocity=100,
                pitch=60 + pitch_offset,
                start=0.25,
                end=0.55,
            ),
            pretty_midi.Note(
                velocity=100,
                pitch=62 + pitch_offset,
                start=0.6,
                end=0.9,
            ),
        ]
    )
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def _write_pseudo(path: Path, *, program: int, pitch_offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        note_start_seconds=np.asarray((0.25, 0.6, 1.0), dtype=np.float64),
        note_end_seconds=np.asarray((0.55, 0.9, 1.3), dtype=np.float64),
        note_pitch=np.asarray((60, 62, 64), dtype=np.int16) + pitch_offset,
        note_program=np.full(3, program, dtype=np.int16),
        note_is_drum=np.zeros(3, dtype=np.bool_),
        note_track_index=np.zeros(3, dtype=np.int16),
        pseudo_velocity_rank=np.asarray((0.0, np.nan, 1.0), dtype=np.float32),
        pseudo_confidence=np.asarray((1.0, 0.0, 1.0), dtype=np.float32),
        pseudo_valid=np.asarray((True, False, True), dtype=np.bool_),
    )


def test_note_velocity_sampling_is_bounded_and_rank_conditioned() -> None:
    config = SyntheticDataConfig(
        velocity_center_min=80.0,
        velocity_center_max=80.0,
        velocity_span_min=60.0,
        velocity_span_max=60.0,
        velocity_jitter_std=0.0,
        independent_velocity_probability=0.0,
    )
    sample = sample_note_velocities(
        pseudo_rank=np.asarray((0.0, np.nan, np.nan, 1.0), dtype=np.float32),
        pseudo_valid=np.asarray((True, False, False, True)),
        track_index=np.zeros(4, dtype=np.int16),
        note_start_seconds=np.asarray((0.0, 0.5, 0.5, 1.0)),
        rng=np.random.default_rng(123),
        config=config,
    )

    assert sample.target_velocity.tolist() == [50, 80, 80, 110]
    assert sample.rank_source.tolist() == [2, 1, 1, 2]


def test_stem_gain_sampling_is_zero_by_default() -> None:
    gains = sample_stem_gains(
        {"bass": -8.0, "piano": 5.0},
        rng=np.random.default_rng(123),
        config=SyntheticDataConfig(),
    )

    assert gains == {"bass": 0.0, "piano": 0.0}


def test_target_midi_uses_label_order_and_fixed_render_controllers(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mid"
    target = tmp_path / "target.mid"
    _write_midi(source, program=24)

    write_target_velocity_midi(
        source,
        target,
        np.asarray((20, 60, 110), dtype=np.int16),
    )

    table = load_midi_note_table(target)
    rendered = pretty_midi.PrettyMIDI(str(target))
    controls = {
        control.number: control.value
        for control in rendered.instruments[0].control_changes
    }
    assert table.input_velocity.tolist() == [20, 60, 110]
    assert controls[7] == 127
    assert controls[11] == 127


def test_prepare_and_mix_synthetic_examples(tmp_path: Path) -> None:
    source_root = tmp_path / "pseudo"
    manifest_path = source_root / "manifest.csv"
    source_rows = []
    for stem_name, program, level, pitch_offset in (
        ("bass", 32, -3.0, -12),
        ("piano", 0, 3.0, 0),
    ):
        midi_path = source_root / "midis" / f"{stem_name}.mid"
        pseudo_path = source_root / "labels" / f"{stem_name}.npz"
        _write_midi(midi_path, program=program, pitch_offset=pitch_offset)
        _write_pseudo(pseudo_path, program=program, pitch_offset=pitch_offset)
        source_rows.append(
            {
                "song_id": "song_a",
                "stem_name": stem_name,
                "midi_path": str(midi_path),
                "pseudo_label_path": str(pseudo_path),
                "relative_active_level_db": level,
                "has_midi": 1,
                "error": "",
            }
        )
    source_root.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(source_rows[0]))
        writer.writeheader()
        writer.writerows(source_rows)

    output_root = tmp_path / "synthetic"
    config = SyntheticDataConfig(
        velocity_jitter_std=0.0,
        independent_velocity_probability=0.0,
        gain_jitter_std_db=0.0,
        master_gain_min_db=0.0,
        master_gain_max_db=0.0,
    )
    summary = prepare_synthetic_plan(
        manifest_path,
        output_root,
        config=config,
        variations=2,
        seed=7,
        overwrite=True,
    )

    with (output_root / "render_manifest.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as file:
        render_rows = list(csv.DictReader(file))
    assert summary["example_count"] == 2
    assert summary["render_job_count"] == 4
    assert all(not Path(row["target_midi_path"]).is_absolute() for row in render_rows)
    assert all("soundfont" not in key for key in render_rows[0])

    for row in render_rows:
        wav_path = output_root / row["rendered_stem_path"]
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            wav_path,
            np.full((4_000, 1), 0.1, dtype=np.float32),
            8_000,
            subtype="FLOAT",
        )
    dataset_rows, mix_summary = mix_rendered_stems(
        output_root / "render_manifest.csv",
        output_root / "examples.csv",
        peak_limit_dbfs=-1.0,
        output_sample_rate=4_000,
    )

    assert mix_summary["mixed_example_count"] == 2
    assert len(dataset_rows) == 2
    assert all(int(row["sample_rate"]) == 4_000 for row in dataset_rows)
    assert all(
        sf.info(output_root / row["mixture_path"]).frames == 2_000
        for row in dataset_rows
    )
    assert all((output_root / row["mixture_path"]).is_file() for row in dataset_rows)
    assert all(float(row["final_peak_dbfs"]) <= -1.0 + 1e-5 for row in dataset_rows)

    reused_rows, reused_summary = mix_rendered_stems(
        output_root / "render_manifest.csv",
        output_root / "examples.csv",
        peak_limit_dbfs=-1.0,
        output_sample_rate=4_000,
    )
    assert len(reused_rows) == 2
    assert reused_summary["skipped_existing_count"] == 2

    resampled_rows, resampled_summary = mix_rendered_stems(
        output_root / "render_manifest.csv",
        output_root / "examples.csv",
        peak_limit_dbfs=-1.0,
        output_sample_rate=2_000,
    )
    assert resampled_summary["skipped_existing_count"] == 0
    assert all(
        sf.info(output_root / row["mixture_path"]).samplerate == 2_000
        for row in resampled_rows
    )

    write_dataset_manifest(output_root / "dataset_manifest.csv", resampled_rows)
    assert all(
        (output_root / row["rendered_stem_path"]).is_file()
        for row in render_rows
    )
    assert all((output_root / row["mixture_path"]).is_file() for row in resampled_rows)
