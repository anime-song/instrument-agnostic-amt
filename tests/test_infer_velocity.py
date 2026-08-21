from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch

import instrument_agnostic_amt.velocity.cli.infer_velocity as velocity_infer
from instrument_agnostic_amt.velocity.cli.infer_velocity import (
    predict_velocity_for_midi,
    predict_velocity_for_stem_midis,
)
from instrument_agnostic_amt.velocity.modeling.model import (
    VelocityModelConfig,
    VelocityPredictionModel,
)


def test_velocity_auto_routes_model_to_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_devices: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    def stop_after_device_resolution(
        _path: object,
        *,
        device: torch.device,
    ) -> tuple[object, object]:
        loaded_devices.append(str(device))
        raise RuntimeError("device captured")

    monkeypatch.setattr(
        velocity_infer,
        "load_velocity_model",
        stop_after_device_resolution,
    )

    with pytest.raises(RuntimeError, match="device captured"):
        predict_velocity_for_stem_midis({}, {}, device=None)

    assert loaded_devices == ["mps"]


def _midi_note_structure(path: Path) -> list[tuple[int, str, int, str, int, int]]:
    midi = mido.MidiFile(str(path))
    structure: list[tuple[int, str, int, str, int, int]] = []
    for track_index, track in enumerate(midi.tracks):
        track_name = next(
            (
                message.name
                for message in track
                if message.type == 'track_name'
            ),
            '',
        )
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            if message.type not in {'note_on', 'note_off'}:
                continue
            event_kind = (
                'on'
                if message.type == 'note_on' and int(message.velocity) > 0
                else 'off'
            )
            structure.append(
                (
                    track_index,
                    track_name,
                    absolute_tick,
                    event_kind,
                    int(message.channel),
                    int(message.note),
                )
            )
    return structure


def _midi_note_on_velocities(path: Path) -> list[int]:
    midi = mido.MidiFile(str(path))
    return [
        int(message.velocity)
        for track in midi.tracks
        for message in track
        if message.type == 'note_on' and int(message.velocity) > 0
    ]


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
    """predict_velocity_for_stem_midis が各ステムMIDIの1:1対応を維持し、CC#7 Volumeを更新するかテスト。"""
    output_midi_path = tmp_path / "merged_velocity.mid"

    result_path = predict_velocity_for_stem_midis(
        stem_midis=mock_stem_midis,
        stem_audios=mock_audio_stems,
        output_midi_path=output_midi_path,
        checkpoint_path=mock_velocity_checkpoint,
        device="cpu",
        window_seconds=4.0,
        apply_stem_gain_to_cc7=True,
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

        cc7_events = [cc for cc in inst.control_changes if cc.number == 7]
        assert len(cc7_events) >= 1
        for cc in cc7_events:
            assert 1 <= cc.value <= 127


def test_predict_velocity_uses_preloaded_waveforms_without_reloading(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    mock_velocity_checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model, config = velocity_infer.load_velocity_model(
        mock_velocity_checkpoint,
        device="cpu",
    )
    preloaded_waveforms = {
        stem_name: torch.from_numpy(
            velocity_infer._load_and_preprocess_audio(
                audio_path,
                target_sample_rate=config.sample_rate,
            )
        ).contiguous()
        for stem_name, audio_path in mock_audio_stems.items()
    }
    waveform_copies = {
        stem_name: waveform.clone()
        for stem_name, waveform in preloaded_waveforms.items()
    }
    eager_output_path = tmp_path / "eager_velocity.mid"
    predict_velocity_for_stem_midis(
        stem_midis=mock_stem_midis,
        stem_audios=mock_audio_stems,
        output_midi_path=eager_output_path,
        device="cpu",
        window_seconds=4.0,
        disable_tqdm=True,
        preloaded_model=model,
        preloaded_config=config,
    )
    monkeypatch.setattr(
        velocity_infer,
        "_load_and_preprocess_audio",
        lambda *_args, **_kwargs: pytest.fail("audio should not be reloaded"),
    )
    output_path = tmp_path / "preloaded_velocity.mid"

    result = predict_velocity_for_stem_midis(
        stem_midis=mock_stem_midis,
        stem_audios=mock_audio_stems,
        output_midi_path=output_path,
        device="cpu",
        window_seconds=4.0,
        disable_tqdm=True,
        preloaded_model=model,
        preloaded_config=config,
        preloaded_waveforms=preloaded_waveforms,
    )

    assert result == output_path
    assert output_path.read_bytes() == eager_output_path.read_bytes()
    assert all(
        torch.equal(preloaded_waveforms[name], waveform_copies[name])
        for name in preloaded_waveforms
    )


def test_template_midi_preserves_note_structure_and_uses_stem_velocities(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class StemIndexedVelocityModel:
        def __call__(self, _audio: torch.Tensor, **kwargs: torch.Tensor):
            stem_indices = kwargs['note_stem_index']
            return {
                'velocity_expected': stem_indices.to(torch.float32) * 10.0 + 30.0
            }

    config = VelocityModelConfig(sample_rate=22050, predict_stem_gain=False)
    monkeypatch.setattr(
        velocity_infer,
        'load_velocity_model',
        lambda *_args, **_kwargs: (StemIndexedVelocityModel(), config),
    )

    template_midi = pretty_midi.PrettyMIDI()
    for midi_path in mock_stem_midis.values():
        stem_midi = pretty_midi.PrettyMIDI(str(midi_path))
        template_midi.instruments.extend(stem_midi.instruments)

    # 予測元のbass noteより長いV1 noteを作り、終了位置が維持されることも確認する。
    template_midi.instruments[0].notes[0].end = 1.8
    template_path = tmp_path / 'template_v1.mid'
    template_midi.write(str(template_path))
    original_structure = _midi_note_structure(template_path)

    output_path = tmp_path / 'template_velocity.mid'
    result_path = predict_velocity_for_stem_midis(
        stem_midis=mock_stem_midis,
        stem_audios=mock_audio_stems,
        output_midi_path=output_path,
        template_midi_path=template_path,
        device='cpu',
        window_seconds=4.0,
        disable_tqdm=True,
    )

    assert result_path == output_path
    assert _midi_note_structure(output_path) == original_structure
    assert _midi_note_on_velocities(output_path) == [30, 50, 40]

    output = pretty_midi.PrettyMIDI(str(output_path))
    assert output.instruments[0].notes[0].end == pytest.approx(1.8, abs=0.01)
    for instrument in output.instruments:
        fixed_controls = [
            (control.number, control.value, control.time)
            for control in instrument.control_changes
            if control.number in (7, 11)
        ]
        assert fixed_controls == [(7, 127, 0.0), (11, 127, 0.0)]


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
    original_structure = _midi_note_structure(single_midi_path)

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
    assert _midi_note_structure(result_path) == original_structure
    output_midi = pretty_midi.PrettyMIDI(str(result_path))
    assert len(output_midi.instruments) >= 1
    for inst in output_midi.instruments:
        for note in inst.notes:
            assert 1 <= note.velocity <= 127


def test_velocity_only_inference_fixes_loudness_controls_at_maximum(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    mock_velocity_checkpoint: Path,
    tmp_path: Path,
) -> None:
    for midi_path in mock_stem_midis.values():
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        for instrument in midi.instruments:
            instrument.control_changes.extend(
                [
                    pretty_midi.ControlChange(number=7, value=72, time=0.0),
                    pretty_midi.ControlChange(number=11, value=91, time=0.0),
                    pretty_midi.ControlChange(number=64, value=127, time=0.1),
                ]
            )
        midi.write(str(midi_path))

    output_path = tmp_path / "velocity_only.mid"
    predict_velocity_for_stem_midis(
        stem_midis=mock_stem_midis,
        stem_audios=mock_audio_stems,
        output_midi_path=output_path,
        checkpoint_path=mock_velocity_checkpoint,
        device="cpu",
        window_seconds=4.0,
        disable_tqdm=True,
    )

    output = pretty_midi.PrettyMIDI(str(output_path))
    controls = [
        (control.number, control.value, control.time)
        for instrument in output.instruments
        for control in instrument.control_changes
    ]
    instrument_count = len(output.instruments)
    assert controls.count((7, 127, 0.0)) == instrument_count
    assert controls.count((11, 127, 0.0)) == instrument_count
    assert all(number not in (7, 11) or value == 127 for number, value, _ in controls)
    assert any(number == 64 for number, _, _ in controls)


def test_velocity_output_limits_melodic_tracks_and_merges_overflow(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    mock_velocity_checkpoint: Path,
    tmp_path: Path,
) -> None:
    guitar_midi = pretty_midi.PrettyMIDI()
    for index in range(17):
        instrument = pretty_midi.Instrument(
            program=index,
            name=f"guitar_class_{index:02d}",
        )
        instrument.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=48 + index,
                start=0.1 + 0.01 * index,
                end=0.8 + 0.01 * index,
            )
        )
        guitar_midi.instruments.append(instrument)
    guitar_midi.write(str(mock_stem_midis["guitar"]))

    output_path = tmp_path / "limited_velocity.mid"
    predict_velocity_for_stem_midis(
        stem_midis=mock_stem_midis,
        stem_audios=mock_audio_stems,
        output_midi_path=output_path,
        checkpoint_path=mock_velocity_checkpoint,
        device="cpu",
        window_seconds=4.0,
        max_melodic_instruments=15,
        disable_tqdm=True,
    )

    output = pretty_midi.PrettyMIDI(str(output_path))
    melodic = [instrument for instrument in output.instruments if not instrument.is_drum]
    drums = [instrument for instrument in output.instruments if instrument.is_drum]
    assert len(melodic) == 15
    assert len(drums) == 1
    assert len(output.instruments) == 16
    assert sum(len(instrument.notes) for instrument in output.instruments) == 19
    assert sum(instrument.name == "Other / Merged" for instrument in melodic) == 1
