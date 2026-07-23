from __future__ import annotations

from pathlib import Path

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch

from instrument_agnostic_amt.velocity.cli.infer_velocity import (
    predict_velocity_for_midi,
    predict_velocity_for_stem_midis,
)
from instrument_agnostic_amt.velocity.modeling.model import (
    VelocityModelConfig,
    VelocityPredictionModel,
)


@pytest.fixture
def mock_velocity_checkpoint(tmp_path: Path) -> Path:
    """テスト用のVelocityPredictionModelチェックポイントを作成する。"""
    config = VelocityModelConfig(
        sample_rate=22050,
        encoder_num_layers=2,
        encoder_num_heads=4,
        hidden_size=128,
        note_hidden_size=64,
    )
    model = VelocityPredictionModel(config)
    checkpoint_path = tmp_path / "mock_velocity_model.pth"
    torch.save(
        {
            "config": config,
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


@pytest.fixture
def mock_audio_stems(tmp_path: Path) -> dict[str, Path]:
    """テスト用のダミーステム音声を作成する。"""
    sample_rate = 22050
    duration_seconds = 2.0
    num_samples = int(sample_rate * duration_seconds)
    t = np.linspace(0, duration_seconds, num_samples, endpoint=False)

    stems = {}
    freqs = {"bass": 110.0, "drums": 60.0, "guitar": 330.0, "other": 440.0}

    for name, freq in freqs.items():
        signal = 0.5 * np.sin(2.0 * np.pi * freq * t, dtype=np.float32)
        stereo = np.column_stack([signal, signal])
        file_path = tmp_path / f"stem_{name}.wav"
        sf.write(str(file_path), stereo, sample_rate)
        stems[name] = file_path

    return stems


@pytest.fixture
def mock_stem_midis(tmp_path: Path) -> dict[str, Path]:
    """各ステムごとのテスト用ダミーMIDIファイルを作成する。"""
    stem_midis = {}

    bass_pm = pretty_midi.PrettyMIDI()
    bass_inst = pretty_midi.Instrument(program=33, is_drum=False, name="bass")
    bass_inst.notes.append(pretty_midi.Note(velocity=80, pitch=36, start=0.2, end=0.8))
    bass_pm.instruments.append(bass_inst)
    bass_path = tmp_path / "bass.mid"
    bass_pm.write(str(bass_path))
    stem_midis["bass"] = bass_path

    guitar_pm = pretty_midi.PrettyMIDI()
    guitar_inst = pretty_midi.Instrument(program=25, is_drum=False, name="guitar")
    guitar_inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.5, end=1.2))
    guitar_pm.instruments.append(guitar_inst)
    guitar_path = tmp_path / "guitar.mid"
    guitar_pm.write(str(guitar_path))
    stem_midis["guitar"] = guitar_path

    drum_pm = pretty_midi.PrettyMIDI()
    drum_inst = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    drum_inst.notes.append(pretty_midi.Note(velocity=80, pitch=38, start=0.1, end=0.3))
    drum_pm.instruments.append(drum_inst)
    drum_path = tmp_path / "drums.mid"
    drum_pm.write(str(drum_path))
    stem_midis["drums"] = drum_path

    return stem_midis


def test_predict_velocity_for_stem_midis(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    mock_velocity_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """predict_velocity_for_stem_midis が各ステムMIDIの1:1対応を維持して正確に推論を行うかテスト。"""
    output_midi_path = tmp_path / "merged_velocity.mid"

    result_path = predict_velocity_for_stem_midis(
        stem_midis=mock_stem_midis,
        stem_audios=mock_audio_stems,
        output_midi_path=output_midi_path,
        checkpoint_path=mock_velocity_checkpoint,
        device="cpu",
        window_seconds=4.0,
        disable_tqdm=True,
    )

    assert isinstance(result_path, Path)
    assert result_path.exists()
    assert result_path == output_midi_path

    output_midi = pretty_midi.PrettyMIDI(str(result_path))
    assert len(output_midi.instruments) == 3

    for inst in output_midi.instruments:
        for note in inst.notes:
            assert 1 <= note.velocity <= 127


def test_predict_velocity_for_single_midi(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    mock_velocity_checkpoint: Path,
    tmp_path: Path,
) -> None:
    """単一MIDIに対する predict_velocity_for_midi 互換関数の動作テスト。"""
    merged_pm = pretty_midi.PrettyMIDI()
    for midi_path in mock_stem_midis.values():
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        merged_pm.instruments.extend(pm.instruments)

    single_midi_path = tmp_path / "single_merged.mid"
    merged_pm.write(str(single_midi_path))

    output_midi_path = tmp_path / "single_output_velocity.mid"

    result_path = predict_velocity_for_midi(
        midi_path=single_midi_path,
        stems=mock_audio_stems,
        output_midi_path=output_midi_path,
        checkpoint_path=mock_velocity_checkpoint,
        device="cpu",
        window_seconds=4.0,
        disable_tqdm=True,
    )

    assert result_path.exists()
    output_midi = pretty_midi.PrettyMIDI(str(result_path))
    assert len(output_midi.instruments) >= 1
    for inst in output_midi.instruments:
        for note in inst.notes:
            assert 1 <= note.velocity <= 127
