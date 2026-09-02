from __future__ import annotations

import weakref
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
from instrument_agnostic_amt.velocity.midi_template import apply_loudness_controls
from instrument_agnostic_amt.velocity.modeling.model import (
    VelocityModelConfig,
    VelocityPredictionModel,
)
from instrument_agnostic_amt.velocity.training.dataset import STEM_CLASS_BY_NAME


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


def test_velocity_cli_exposes_opt_in_compile_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["infer_velocity", "--midi", "input.mid", "--stem-files", "piano.wav"],
    )
    defaults = velocity_infer.parse_args()
    monkeypatch.setattr(
        "sys.argv",
        [
            "infer_velocity",
            "--midi",
            "input.mid",
            "--stem-files",
            "piano.wav",
            "--compile-velocity",
            "--compile-mode",
            "max-autotune",
        ],
    )
    enabled = velocity_infer.parse_args()

    assert (
        getattr(defaults, "compile_velocity", None),
        getattr(defaults, "compile_mode", None),
        getattr(enabled, "compile_velocity", None),
        getattr(enabled, "compile_mode", None),
    ) == (False, "default", True, "max-autotune")


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


def _tiny_velocity_model() -> tuple[VelocityPredictionModel, VelocityModelConfig]:
    config = VelocityModelConfig(
        sample_rate=8_000,
        hop_length=128,
        cqt_n_bins=48,
        cqt_bins_per_octave=12,
        harmonics=(1.0,),
        hidden_size=16,
        base_ch=4,
        encoder_num_layers=1,
        encoder_num_heads=1,
        dropout=0.0,
        note_hidden_size=16,
        predict_stem_gain=False,
        use_gradient_checkpoint=False,
    )
    return VelocityPredictionModel(config).eval(), config


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


def test_velocity_compile_reuses_fixed_shape_and_back_aligns_final_window(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guitar_midi = pretty_midi.PrettyMIDI(str(mock_stem_midis["guitar"]))
    guitar_midi.instruments[0].notes.append(
        pretty_midi.Note(
            velocity=80,
            pitch=64,
            start=1.6,
            end=1.9,
        )
    )
    guitar_midi.write(str(mock_stem_midis["guitar"]))

    compiled_audio_lengths: list[int] = []
    compiled_note_starts: list[list[float]] = []
    compile_calls: list[tuple[object, bool, str]] = []

    class EagerVelocityModel:
        def __call__(
            self,
            _audio: torch.Tensor,
            **kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return {
                "velocity_expected": torch.full_like(
                    kwargs["note_start_seconds"],
                    31.0,
                )
            }

    eager_model = EagerVelocityModel()

    def compiled_forward(
        audio: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        compiled_audio_lengths.append(int(audio.shape[-1]))
        compiled_note_starts.append(kwargs["note_start_seconds"].squeeze(0).tolist())
        return {
            "velocity_expected": torch.full_like(
                kwargs["note_start_seconds"],
                91.0,
            )
        }

    config = VelocityModelConfig(sample_rate=22_050, predict_stem_gain=False)
    monkeypatch.setattr(
        velocity_infer,
        "load_velocity_model",
        lambda *_args, **_kwargs: (eager_model, config),
    )

    def fake_compile(
        model: object,
        *,
        enabled: bool,
        mode: str,
    ) -> object:
        compile_calls.append((model, enabled, mode))
        return compiled_forward

    monkeypatch.setattr(
        velocity_infer,
        "maybe_compile_forward",
        fake_compile,
        raising=False,
    )

    output_path = tmp_path / "compiled_velocity.mid"
    predict_velocity_for_stem_midis(
        stem_midis=mock_stem_midis,
        stem_audios=mock_audio_stems,
        output_midi_path=output_path,
        device="cpu",
        window_seconds=1.25,
        compile_velocity=True,
        compile_mode="reduce-overhead",
        disable_tqdm=True,
    )

    assert compile_calls == [(eager_model, True, "reduce-overhead")]
    assert compiled_audio_lengths == [
        int(1.25 * config.sample_rate),
        int(1.25 * config.sample_rate),
    ]
    final_audio_start = (
        2 * config.sample_rate - int(1.25 * config.sample_rate)
    ) / config.sample_rate
    assert compiled_note_starts[-1] == pytest.approx([1.6 - final_audio_start])
    output = pretty_midi.PrettyMIDI(str(output_path))
    velocities_by_start = {
        round(note.start, 1): note.velocity
        for instrument in output.instruments
        for note in instrument.notes
    }
    assert velocities_by_start[0.1] == 91
    assert velocities_by_start[1.6] == 91


def test_legacy_velocity_back_aligns_a_one_sample_final_window(
    tmp_path: Path,
) -> None:
    model, config = _tiny_velocity_model()
    audio_path = tmp_path / "one-sample-tail.wav"
    midi_path = tmp_path / "one-sample-tail.mid"
    output_path = tmp_path / "one-sample-tail-velocity.mid"
    sf.write(
        audio_path,
        np.zeros((8_001, 2), dtype=np.float32),
        config.sample_rate,
        subtype="FLOAT",
    )
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.Message("note_on", channel=0, note=60, velocity=80, time=960),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=1),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    midi.save(midi_path)

    result = predict_velocity_for_stem_midis(
        stem_midis={"other": midi_path},
        stem_audios={"other": audio_path},
        output_midi_path=output_path,
        device="cpu",
        window_seconds=1.0,
        disable_tqdm=True,
        preloaded_model=model,
        preloaded_config=config,
    )

    assert result == output_path
    assert output_path.exists()
    assert len(pretty_midi.PrettyMIDI(str(output_path)).instruments[0].notes) == 1


def test_legacy_velocity_keeps_notes_in_the_last_logical_window(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    tmp_path: Path,
) -> None:
    guitar_midi = pretty_midi.PrettyMIDI(str(mock_stem_midis["guitar"]))
    guitar_midi.instruments[0].notes[0].start = 3.0
    guitar_midi.instruments[0].notes[0].end = 3.1
    guitar_midi.write(str(mock_stem_midis["guitar"]))
    note_starts: list[list[float]] = []

    class RecordingModel:
        def to(self, _device: torch.device) -> RecordingModel:
            return self

        def eval(self) -> RecordingModel:
            return self

        def __call__(
            self,
            _audio: torch.Tensor,
            **kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            note_starts.append(kwargs["note_start_seconds"].squeeze(0).tolist())
            return {
                "velocity_expected": torch.full_like(
                    kwargs["note_start_seconds"],
                    93.0,
                )
            }

    output_path = tmp_path / "last-logical-window.mid"
    result = predict_velocity_for_stem_midis(
        stem_midis={"guitar": mock_stem_midis["guitar"]},
        stem_audios={"guitar": mock_audio_stems["guitar"]},
        output_midi_path=output_path,
        device="cpu",
        window_seconds=8.0,
        disable_tqdm=True,
        preloaded_model=RecordingModel(),  # type: ignore[arg-type]
        preloaded_config=VelocityModelConfig(
            sample_rate=22_050,
            predict_stem_gain=False,
        ),
    )

    assert result == output_path
    assert len(note_starts) == 1
    assert note_starts[0] == pytest.approx([3.0])
    output = pretty_midi.PrettyMIDI(str(output_path))
    assert output.instruments[0].notes[0].velocity == 93


def test_legacy_velocity_right_pads_audio_shorter_than_one_window(
    tmp_path: Path,
) -> None:
    model, config = _tiny_velocity_model()
    audio_path = tmp_path / "short.wav"
    midi_path = tmp_path / "short.mid"
    output_path = tmp_path / "short-velocity.mid"
    waveform = np.linspace(0.0, 0.5, 1_000, dtype=np.float32)
    sf.write(
        audio_path,
        np.column_stack((waveform, waveform)),
        config.sample_rate,
        subtype="FLOAT",
    )
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="other")
    instrument.notes.append(
        pretty_midi.Note(velocity=80, pitch=60, start=0.05, end=0.06)
    )
    midi.instruments.append(instrument)
    midi.write(str(midi_path))
    calls: list[tuple[torch.Size, list[list[int]], torch.Tensor]] = []

    def recording_forward(
        audio: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        calls.append(
            (
                audio.shape,
                kwargs["valid_audio_frames"].tolist(),
                audio[..., 1_000:].detach().cpu(),
            )
        )
        return {
            "velocity_expected": torch.full_like(
                kwargs["note_start_seconds"],
                64.0,
            )
        }

    result = predict_velocity_for_stem_midis(
        stem_midis={"other": midi_path},
        stem_audios={"other": audio_path},
        output_midi_path=output_path,
        device="cpu",
        window_seconds=1.0,
        disable_tqdm=True,
        preloaded_model=model,
        preloaded_config=config,
        preloaded_forward=recording_forward,
    )

    assert result == output_path
    assert len(calls) == 1
    assert calls[0][0] == torch.Size((1, 1, 2, 8_000))
    assert calls[0][1] == [[1_000]]
    assert torch.count_nonzero(calls[0][2]) == 0

    strict_audio_lengths: list[int] = []

    def strict_forward(
        audio: torch.Tensor,
        *,
        note_start_seconds: torch.Tensor,
        note_end_seconds: torch.Tensor,
        note_pitch: torch.Tensor,
        note_program: torch.Tensor,
        note_is_drum: torch.Tensor,
        note_stem_index: torch.Tensor,
        stem_class_id: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _ = (
            note_end_seconds,
            note_pitch,
            note_program,
            note_is_drum,
            note_stem_index,
            stem_class_id,
        )
        strict_audio_lengths.append(int(audio.shape[-1]))
        return {"velocity_expected": torch.full_like(note_start_seconds, 64.0)}

    strict_result = predict_velocity_for_stem_midis(
        stem_midis={"other": midi_path},
        stem_audios={"other": audio_path},
        output_midi_path=tmp_path / "short-strict-velocity.mid",
        device="cpu",
        window_seconds=1.0,
        disable_tqdm=True,
        preloaded_model=model,
        preloaded_config=config,
        preloaded_forward=strict_forward,
    )

    assert strict_result == tmp_path / "short-strict-velocity.mid"
    assert strict_audio_lengths == [8_000]


def test_velocity_inference_skips_unused_stem_gain_head(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    include_stem_gain_values: list[bool] = []
    inference_mode_values: list[bool] = []

    class FakeVelocityModel:
        def to(self, _device: torch.device) -> FakeVelocityModel:
            return self

        def eval(self) -> FakeVelocityModel:
            return self

        def __call__(
            self,
            audio: torch.Tensor,
            *,
            include_stem_gain: bool = True,
            **kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            include_stem_gain_values.append(include_stem_gain)
            inference_mode_values.append(torch.is_inference_mode_enabled())
            outputs = {
                "velocity_expected": torch.full_like(
                    kwargs["note_start_seconds"],
                    64.0,
                )
            }
            if include_stem_gain:
                outputs["stem_gain_db"] = torch.zeros(
                    (audio.shape[0], audio.shape[1]),
                    device=audio.device,
                )
            return outputs

    model = FakeVelocityModel()
    config = VelocityModelConfig(sample_rate=22_050, predict_stem_gain=True)
    monkeypatch.setattr(velocity_infer, "VelocityPredictionModel", FakeVelocityModel)
    monkeypatch.setattr(
        velocity_infer,
        "load_velocity_model",
        lambda *_args, **_kwargs: (model, config),
    )

    predict_velocity_for_stem_midis(
        stem_midis=mock_stem_midis,
        stem_audios=mock_audio_stems,
        output_midi_path=tmp_path / "velocity_without_stem_gain.mid",
        device="cpu",
        window_seconds=4.0,
        disable_tqdm=True,
    )

    assert include_stem_gain_values == [False]
    assert inference_mode_values == [True]


def test_velocity_only_inference_runs_only_note_stems_in_each_window(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guitar_midi = pretty_midi.PrettyMIDI(str(mock_stem_midis["guitar"]))
    guitar_midi.instruments[0].notes[0].start = 1.5
    guitar_midi.instruments[0].notes[0].end = 1.9
    guitar_midi.write(str(mock_stem_midis["guitar"]))
    guitar_audio, sample_rate = sf.read(
        mock_audio_stems["guitar"],
        dtype="float32",
        always_2d=True,
    )
    guitar_sample_count = int(1.75 * sample_rate)
    sf.write(
        mock_audio_stems["guitar"],
        guitar_audio[:guitar_sample_count],
        sample_rate,
        subtype="FLOAT",
    )
    stem_counts: list[int] = []
    local_stem_indices: list[list[int]] = []
    valid_frame_counts: list[list[int] | None] = []

    class RecordingVelocityModel(VelocityPredictionModel):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)

        def forward(
            self,
            audio: torch.Tensor,
            **kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            stem_counts.append(int(audio.shape[1]))
            local_stem_indices.append(
                kwargs["note_stem_index"].squeeze(0).tolist()
            )
            valid_audio_frames = kwargs.get("valid_audio_frames")
            valid_frame_counts.append(
                None
                if valid_audio_frames is None
                else valid_audio_frames.squeeze(0).tolist()
            )
            selected_classes = kwargs["stem_class_id"].gather(
                1,
                kwargs["note_stem_index"],
            )
            return {"velocity_expected": selected_classes.float() + 50.0}

    model = RecordingVelocityModel()
    config = VelocityModelConfig(sample_rate=22_050, predict_stem_gain=False)
    monkeypatch.setattr(
        velocity_infer,
        "load_velocity_model",
        lambda *_args, **_kwargs: (model, config),
    )

    output_path = tmp_path / "selected_note_stems.mid"
    predict_velocity_for_stem_midis(
        stem_midis={
            "bass": mock_stem_midis["bass"],
            "guitar": mock_stem_midis["guitar"],
        },
        stem_audios=mock_audio_stems,
        output_midi_path=output_path,
        device="cpu",
        window_seconds=1.0,
        disable_tqdm=True,
    )

    assert stem_counts == [1, 1]
    assert local_stem_indices == [[0], [0]]
    assert valid_frame_counts == [None, [guitar_sample_count - 22_050]]
    output = pretty_midi.PrettyMIDI(str(output_path))
    velocities = {
        instrument.name: instrument.notes[0].velocity
        for instrument in output.instruments
    }
    assert velocities == {
        "bass": STEM_CLASS_BY_NAME["bass"] + 50,
        "guitar": STEM_CLASS_BY_NAME["guitar"] + 50,
    }


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="MPS is not available",
            ),
        ),
    ],
)
def test_velocity_inference_releases_window_outputs_after_cpu_extraction(
    device_name: str,
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    guitar_midi = pretty_midi.PrettyMIDI(str(mock_stem_midis["guitar"]))
    guitar_midi.instruments[0].notes.append(
        pretty_midi.Note(
            velocity=80,
            pitch=64,
            start=1.6,
            end=1.9,
        )
    )
    guitar_midi.write(str(mock_stem_midis["guitar"]))

    output_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    class WindowedVelocityModel:
        def __call__(
            self,
            _audio: torch.Tensor,
            **kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            if output_refs:
                assert output_refs[-1]() is None
            output = torch.full_like(
                kwargs["note_start_seconds"],
                (37.0, 93.0)[len(output_refs)],
            )
            output_refs.append(weakref.ref(output))
            return {"velocity_expected": output}

    model = WindowedVelocityModel()
    config = VelocityModelConfig(sample_rate=22_050, predict_stem_gain=False)
    monkeypatch.setattr(
        velocity_infer,
        "load_velocity_model",
        lambda *_args, **_kwargs: (model, config),
    )

    output_path = tmp_path / "window_output_lifetime.mid"
    predict_velocity_for_stem_midis(
        stem_midis={"guitar": mock_stem_midis["guitar"]},
        stem_audios={"guitar": mock_audio_stems["guitar"]},
        output_midi_path=output_path,
        device=device_name,
        window_seconds=1.0,
        disable_tqdm=True,
    )

    assert all(output_ref() is None for output_ref in output_refs)
    output_midi = pretty_midi.PrettyMIDI(str(output_path))
    assert {
        round(note.start, 1): note.velocity
        for instrument in output_midi.instruments
        for note in instrument.notes
    } == {0.5: 37, 1.6: 93}


def test_preloaded_velocity_forward_requires_matching_eager_model() -> None:
    with pytest.raises(
        ValueError,
        match="preloaded_forward requires preloaded_model and preloaded_config",
    ):
        predict_velocity_for_stem_midis(
            stem_midis={},
            stem_audios={},
            device="cpu",
            preloaded_forward=object(),
        )


@pytest.mark.parametrize("window_seconds", [0.0, -1.0])
def test_velocity_inference_rejects_nonpositive_window(
    window_seconds: float,
) -> None:
    with pytest.raises(ValueError, match="window_seconds must be positive"):
        predict_velocity_for_stem_midis(
            stem_midis={},
            stem_audios={},
            device="cpu",
            window_seconds=window_seconds,
        )


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
    monkeypatch.setattr(
        velocity_infer,
        "maybe_compile_forward",
        lambda *_args, **_kwargs: pytest.fail("forward should not be recompiled"),
    )
    output_path = tmp_path / "preloaded_velocity.mid"

    result = predict_velocity_for_stem_midis(
        stem_midis=mock_stem_midis,
        stem_audios=mock_audio_stems,
        output_midi_path=output_path,
        device="cpu",
        window_seconds=4.0,
        disable_tqdm=True,
        compile_velocity=True,
        compile_mode="reduce-overhead",
        preloaded_model=model,
        preloaded_config=config,
        preloaded_forward=model,
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


def test_velocity_inference_can_preserve_template_loudness_controls(
    mock_stem_midis: dict[str, Path],
    mock_audio_stems: dict[str, Path],
    mock_velocity_checkpoint: Path,
    tmp_path: Path,
) -> None:
    input_path = mock_stem_midis["drums"]
    midi = pretty_midi.PrettyMIDI(str(input_path))
    midi.instruments[0].control_changes.extend(
        [
            pretty_midi.ControlChange(number=7, value=72, time=0.0),
            pretty_midi.ControlChange(number=64, value=127, time=0.1),
            pretty_midi.ControlChange(number=11, value=91, time=0.2),
        ]
    )
    midi.write(str(input_path))
    output_path = tmp_path / "preserved.mid"

    predict_velocity_for_stem_midis(
        stem_midis={"drums": input_path},
        stem_audios={"drums": mock_audio_stems["drums"]},
        output_midi_path=output_path,
        template_midi_path=input_path,
        checkpoint_path=mock_velocity_checkpoint,
        device="cpu",
        window_seconds=4.0,
        disable_tqdm=True,
        loudness_controls="preserve",
    )

    output = pretty_midi.PrettyMIDI(str(output_path))
    controls = output.instruments[0].control_changes
    assert [(control.number, control.value) for control in controls] == [
        (7, 72),
        (64, 127),
        (11, 91),
    ]
    assert [control.time for control in controls] == pytest.approx([0.0, 0.1, 0.2])


def test_strip_loudness_controls_preserves_other_event_ticks() -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    track = mido.MidiTrack()
    track.extend(
        [
            mido.Message("program_change", channel=0, program=8, time=0),
            mido.Message("control_change", channel=0, control=7, value=72, time=10),
            mido.Message("control_change", channel=0, control=64, value=127, time=5),
            mido.Message("note_on", channel=0, note=60, velocity=80, time=7),
            mido.MetaMessage("end_of_track", time=20),
        ]
    )
    midi.tracks.append(track)

    apply_loudness_controls(midi, "strip")

    absolute_tick = 0
    remaining_events: list[tuple[str, int, int]] = []
    for message in midi.tracks[0]:
        absolute_tick += int(message.time)
        if message.type == "control_change":
            remaining_events.append((message.type, int(message.control), absolute_tick))
        elif message.type == "note_on":
            remaining_events.append((message.type, int(message.note), absolute_tick))
    assert remaining_events == [
        ("control_change", 64, 15),
        ("note_on", 60, 22),
    ]


def test_velocity_only_controls_precede_tick_zero_notes_and_strip_automation_tracks() -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    note_track = mido.MidiTrack(
        [
            mido.MetaMessage("track_name", name="drums", time=0),
            mido.Message("program_change", channel=9, program=0, time=0),
            mido.Message("note_on", channel=9, note=38, velocity=80, time=0),
            mido.Message("note_off", channel=9, note=38, velocity=23, time=120),
            mido.MetaMessage("end_of_track", time=0),
        ]
    )
    automation_track = mido.MidiTrack(
        [
            mido.Message("control_change", channel=2, control=7, value=50, time=12),
            mido.Message("control_change", channel=2, control=64, value=127, time=3),
            mido.MetaMessage("end_of_track", time=0),
        ]
    )
    midi.tracks.extend([note_track, automation_track])

    apply_loudness_controls(midi, "velocity_only")

    note_events = list(midi.tracks[0])
    note_on_index = next(
        index for index, message in enumerate(note_events) if message.type == "note_on"
    )
    fixed_indices = [
        index
        for index, message in enumerate(note_events)
        if message.type == "control_change" and int(message.control) in (7, 11)
    ]
    assert fixed_indices and max(fixed_indices) < note_on_index
    assert all(int(note_events[index].time) == 0 for index in fixed_indices)

    absolute_tick = 0
    automation_events: list[tuple[int, int]] = []
    for message in midi.tracks[1]:
        absolute_tick += int(message.time)
        if message.type == "control_change":
            automation_events.append((int(message.control), absolute_tick))
    assert automation_events == [(64, 15)]


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
