from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Event

import mido
import numpy as np
import pytest
import soundfile as sf
import torch

import instrument_agnostic_amt
import instrument_agnostic_amt.velocity.estimator as estimator_module
from instrument_agnostic_amt import (
    DecodedAudio,
    VelocityEstimator,
    VelocityEstimatorBusyError,
    VelocityOptions,
)
from instrument_agnostic_amt.velocity.inference import VelocityPredictions
from instrument_agnostic_amt.velocity.modeling.model import (
    VelocityModelConfig,
    VelocityPredictionModel,
)
from instrument_agnostic_amt.velocity.training.dataset import STEM_CLASS_BY_NAME


def _write_tiny_velocity_checkpoint(
    path: Path,
    *,
    predict_stem_gain: bool = False,
) -> None:
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
        predict_stem_gain=predict_stem_gain,
        use_gradient_checkpoint=False,
    )
    model = VelocityPredictionModel(config)
    torch.save(
        {
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def _midi_semantic_snapshot(midi: mido.MidiFile) -> list[list[dict[str, object]]]:
    snapshot: list[list[dict[str, object]]] = []
    for track in midi.tracks:
        absolute_tick = 0
        events: list[dict[str, object]] = []
        for message in track:
            absolute_tick += int(message.time)
            values = message.dict()
            values.pop("time", None)
            if message.type == "note_on" and int(message.velocity) > 0:
                values["velocity"] = "predicted"
            events.append({"tick": absolute_tick, **values})
        snapshot.append(events)
    return snapshot


def test_package_exports_the_public_velocity_api() -> None:
    expected_names = {
        "VelocityEstimator",
        "VelocityEstimatorBusyError",
        "VelocityModelInfo",
        "VelocityOptions",
        "VelocityResult",
    }

    assert expected_names <= set(dir(instrument_agnostic_amt))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_seconds": 0.0}, "window_seconds must be positive"),
        ({"window_seconds": float("nan")}, "window_seconds must be positive"),
        ({"window_seconds": float("inf")}, "window_seconds must be positive"),
        ({"loudness_controls": "unknown"}, "loudness_controls must be one of"),
    ],
)
def test_velocity_options_reject_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        VelocityOptions(**kwargs)  # type: ignore[arg-type]


def test_velocity_estimator_loads_one_explicit_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)

    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")

    assert asdict(estimator.model_info) == {
        "checkpoint_path": checkpoint_path.resolve(),
        "sample_rate": 8_000,
        "device": "cpu",
        "compile_enabled": False,
        "compile_mode": None,
    }
    assert capsys.readouterr() == ("", "")


def test_velocity_estimator_rejects_a_checkpoint_without_velocity_heads(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "backbone-only.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["model_state_dict"] = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("backbone.")
    }
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="Velocity checkpoint is incomplete"):
        VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")


def test_velocity_estimator_accepts_the_legacy_state_dict_key(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "legacy-state-dict.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["state_dict"] = checkpoint.pop("model_state_dict")
    torch.save(checkpoint, checkpoint_path)

    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")

    assert estimator.load_report.missing_keys == ()
    assert estimator.load_report.shape_mismatches == ()


def test_declared_stem_gain_head_must_be_present(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "missing-stem-gain.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path, predict_stem_gain=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["model_state_dict"] = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if not key.startswith(("global_audio_projection.", "stem_gain_head."))
    }
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="Velocity checkpoint is incomplete"):
        VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")


@pytest.mark.parametrize("policy", ["velocity_only", "preserve", "strip"])
def test_zero_note_midi_returns_original_bytes_without_audio_inference(
    policy: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=960)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.MetaMessage("end_of_track", time=120),
            ]
        )
    )
    buffer = io.BytesIO()
    midi.save(file=buffer)
    source_bytes = buffer.getvalue()

    def unexpected_audio(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("zero-note MIDI must not load audio")

    monkeypatch.setattr(estimator, "_prepare_audio", unexpected_audio)

    result = estimator.estimate(
        midi=source_bytes,
        audio=tmp_path / "does-not-exist.wav",
        stem_kind="other",
        options=VelocityOptions(loudness_controls=policy),
    )

    assert result.midi_bytes == source_bytes
    assert result.velocity_applied is False
    assert result.note_count == 0
    assert result.model_info == estimator.model_info


def test_estimator_uses_explicit_stem_kind_for_every_merged_midi_track(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.MetaMessage("marker", text="merged producers", time=0),
                mido.Message("sysex", data=(1, 2, 3), time=0),
                mido.MetaMessage("end_of_track", time=240),
            ]
        )
    )
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("track_name", name="vocals", time=0),
                mido.Message("program_change", channel=9, program=0, time=0),
                mido.Message("control_change", channel=9, control=64, value=127, time=0),
                mido.Message("pitchwheel", channel=9, pitch=128, time=0),
                mido.Message("note_on", channel=9, note=38, velocity=100, time=0),
                mido.Message("note_off", channel=9, note=38, velocity=23, time=120),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("track_name", name="other_v1_5", time=0),
                mido.Message("program_change", channel=0, program=8, time=0),
                mido.Message("aftertouch", channel=0, value=42, time=0),
                mido.Message("note_on", channel=0, note=60, velocity=100, time=24),
                mido.Message("note_on", channel=0, note=60, velocity=0, time=120),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)
    source_bytes = source.getvalue()
    input_snapshot = _midi_semantic_snapshot(
        mido.MidiFile(file=io.BytesIO(source_bytes))
    )

    class FakeForward:
        def __call__(
            self,
            _audio: torch.Tensor,
            **kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            assert kwargs["stem_class_id"].tolist() == [
                [STEM_CLASS_BY_NAME["drums"]]
            ]
            assert kwargs["note_stem_index"].tolist() == [[0, 0]]
            assert kwargs["note_is_drum"].tolist() == [[1, 0]]
            assert kwargs["note_program"].tolist() == [[0, 8]]
            return {"velocity_expected": torch.tensor([[31.0, 97.0]])}

    estimator._forward_model = FakeForward()

    result = estimator.estimate(
        midi=source_bytes,
        audio=DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000),
        stem_kind="drums",
        options=VelocityOptions(loudness_controls="preserve"),
    )

    output = mido.MidiFile(file=io.BytesIO(result.midi_bytes))
    positive_velocities = [
        int(message.velocity)
        for track in output.tracks
        for message in track
        if message.type == "note_on" and int(message.velocity) > 0
    ]
    assert positive_velocities == [31, 97]
    assert result.note_count == 2
    assert result.velocity_applied is True
    assert output.type == 1
    assert output.ticks_per_beat == 480
    assert _midi_semantic_snapshot(output) == input_snapshot
    assert any(
        message.type == "note_off" and int(message.velocity) == 23
        for message in output.tracks[1]
    )
    assert any(
        message.type == "note_on" and int(message.velocity) == 0
        for message in output.tracks[2]
    )


@pytest.mark.parametrize(
    ("policy", "expected_controls"),
    [
        ("preserve", [(7, 72), (11, 91), (64, 127)]),
        ("strip", [(64, 127)]),
        ("velocity_only", [(7, 127), (11, 127), (64, 127)]),
    ],
)
def test_estimator_applies_the_selected_loudness_policy(
    policy: str,
    expected_controls: list[tuple[int, int]],
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("track_name", name="drums", time=0),
                mido.Message("program_change", channel=9, program=0, time=0),
                mido.Message("control_change", channel=9, control=7, value=72, time=0),
                mido.Message("control_change", channel=9, control=11, value=91, time=0),
                mido.Message("note_on", channel=9, note=38, velocity=100, time=0),
                mido.Message("control_change", channel=9, control=64, value=127, time=10),
                mido.Message("note_off", channel=9, note=38, velocity=0, time=110),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)

    class FakeForward:
        def __call__(
            self,
            _audio: torch.Tensor,
            **_kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return {"velocity_expected": torch.tensor([[55.0]])}

    estimator._forward_model = FakeForward()

    result = estimator.estimate(
        midi=source.getvalue(),
        audio=DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000),
        stem_kind="drums",
        options=VelocityOptions(loudness_controls=policy),
    )

    output = mido.MidiFile(file=io.BytesIO(result.midi_bytes))
    controls = [
        (int(message.control), int(message.value))
        for message in output.tracks[0]
        if message.type == "control_change"
    ]
    assert controls == expected_controls
    if policy == "velocity_only":
        events = list(output.tracks[0])
        note_on_index = next(
            index
            for index, message in enumerate(events)
            if message.type == "note_on" and int(message.velocity) > 0
        )
        fixed_indices = [
            index
            for index, message in enumerate(events)
            if message.type == "control_change" and int(message.control) in (7, 11)
        ]
        assert max(fixed_indices) < note_on_index


def test_midi_and_audio_path_inputs_match_in_memory_inputs(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    midi_path = tmp_path / "input.mid"
    audio_path = tmp_path / "input.wav"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.Message("program_change", channel=0, program=33, time=0),
                mido.Message("note_on", channel=0, note=36, velocity=80, time=96),
                mido.Message("note_off", channel=0, note=36, velocity=17, time=96),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    midi.save(midi_path)
    samples = torch.zeros(2, 8_000, dtype=torch.float32)
    sf.write(audio_path, samples.T.numpy(), 8_000, subtype="FLOAT")
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    options = VelocityOptions(loudness_controls="preserve")

    path_result = estimator.estimate(
        midi=midi_path,
        audio=audio_path,
        stem_kind="bass",
        options=options,
    )
    memory_result = estimator.estimate(
        midi=midi_path.read_bytes(),
        audio=DecodedAudio(samples, sample_rate=8_000),
        stem_kind="bass",
        options=options,
    )

    assert path_result.midi_bytes == memory_result.midi_bytes
    assert path_result.note_count == memory_result.note_count == 1
    assert path_result.velocity_applied is memory_result.velocity_applied is True


def test_estimator_keeps_duplicate_notes_from_merged_producers(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("program_change", channel=9, program=0, time=0),
                mido.Message("note_on", channel=9, note=38, velocity=100, time=0),
                mido.Message("note_on", channel=9, note=38, velocity=100, time=0),
                mido.Message("note_off", channel=9, note=38, velocity=0, time=120),
                mido.Message("note_off", channel=9, note=38, velocity=0, time=0),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)

    class FakeForward:
        def __call__(
            self,
            _audio: torch.Tensor,
            **_kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            return {"velocity_expected": torch.tensor([[30.0, 90.0]])}

    estimator._forward_model = FakeForward()

    result = estimator.estimate(
        midi=source.getvalue(),
        audio=DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000),
        stem_kind="drums",
        options=VelocityOptions(loudness_controls="preserve"),
    )

    output = mido.MidiFile(file=io.BytesIO(result.midi_bytes))
    assert result.note_count == 2
    assert [
        int(message.velocity)
        for message in output.tracks[0]
        if message.type == "note_on" and int(message.velocity) > 0
    ] == [30, 90]


def test_note_queries_preserve_overlap_ends_and_note_on_programs(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.Message("program_change", channel=0, program=1, time=0),
                mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
                mido.Message("program_change", channel=0, program=41, time=48),
                mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=48),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=48),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)

    class FakeForward:
        def __call__(
            self,
            _audio: torch.Tensor,
            **kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            torch.testing.assert_close(
                kwargs["note_start_seconds"],
                torch.tensor([[0.0, 0.05]]),
            )
            torch.testing.assert_close(
                kwargs["note_end_seconds"],
                torch.tensor([[0.1, 0.15]]),
            )
            assert kwargs["note_program"].tolist() == [[1, 41]]
            return {"velocity_expected": torch.tensor([[44.0, 88.0]])}

    estimator._forward_model = FakeForward()

    result = estimator.estimate(
        midi=source.getvalue(),
        audio=DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000),
        stem_kind="other",
        options=VelocityOptions(loudness_controls="preserve"),
    )

    output = mido.MidiFile(file=io.BytesIO(result.midi_bytes))
    assert [
        int(message.velocity)
        for message in output.tracks[0]
        if message.type == "note_on" and int(message.velocity) > 0
    ] == [44, 88]


def test_estimator_rejects_notes_starting_after_the_audio_ends(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.Message("note_on", channel=0, note=60, velocity=80, time=96),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=96),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)

    with pytest.raises(ValueError, match="outside the supplied audio"):
        estimator.estimate(
            midi=source.getvalue(),
            audio=DecodedAudio(torch.zeros(2, 400), sample_rate=8_000),
            stem_kind="other",
        )


def test_estimator_back_aligns_a_short_final_audio_window(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
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
    source = io.BytesIO()
    midi.save(file=source)

    waveform = torch.arange(8_001, dtype=torch.float32).repeat(2, 1) / 8_001

    class RecordingForward:
        def __call__(
            self,
            audio: torch.Tensor,
            **kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            assert audio.shape == (1, 1, 2, 8_000)
            torch.testing.assert_close(
                audio[0, 0, :, 0],
                waveform[:, 1],
            )
            torch.testing.assert_close(
                kwargs["note_start_seconds"],
                torch.tensor([[7_999 / 8_000]]),
            )
            return {"velocity_expected": torch.tensor([[55.0]])}

    estimator._forward_model = RecordingForward()

    result = estimator.estimate(
        midi=source.getvalue(),
        audio=DecodedAudio(waveform, sample_rate=8_000),
        stem_kind="other",
        options=VelocityOptions(window_seconds=1.0),
    )

    assert result.velocity_applied is True
    assert result.note_count == 1


def test_estimator_does_not_reassign_notes_from_overlapping_final_audio(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.MetaMessage("set_tempo", tempo=500_000, time=0),
                mido.Message("note_on", channel=0, note=59, velocity=80, time=8_640),
                mido.Message("note_off", channel=0, note=59, velocity=0, time=1),
                mido.Message("note_on", channel=0, note=60, velocity=80, time=6_719),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=1),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)
    waveform = torch.arange(17 * 8_000, dtype=torch.float32).repeat(2, 1)
    waveform /= float(waveform.shape[1])
    calls: list[tuple[list[int], float, list[float]]] = []

    class RecordingForward:
        def __call__(
            self,
            audio: torch.Tensor,
            **kwargs: torch.Tensor,
        ) -> dict[str, torch.Tensor]:
            calls.append(
                (
                    kwargs["note_pitch"].squeeze(0).tolist(),
                    float(audio[0, 0, 0, 0]),
                    kwargs["note_start_seconds"].squeeze(0).tolist(),
                )
            )
            velocity = (41.0, 99.0)[len(calls) - 1]
            return {
                "velocity_expected": torch.full_like(
                    kwargs["note_start_seconds"],
                    velocity,
                )
            }

    estimator._forward_model = RecordingForward()

    result = estimator.estimate(
        midi=source.getvalue(),
        audio=DecodedAudio(waveform, sample_rate=8_000),
        stem_kind="other",
    )

    assert [call[0] for call in calls] == [[59], [60]]
    assert [call[1] for call in calls] == pytest.approx([8 / 17, 9 / 17])
    assert [call[2] for call in calls] == [[1.0], [7.0]]
    output = mido.MidiFile(file=io.BytesIO(result.midi_bytes))
    assert [
        int(message.velocity)
        for track in output.tracks
        for message in track
        if message.type == "note_on" and int(message.velocity) > 0
    ] == [41, 99]


def test_estimator_enforces_the_model_cqt_minimum_window(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=1),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)

    with pytest.raises(
        ValueError,
        match=(
            r"window_seconds is too short for this velocity model: "
            r"requires at least 2,049 samples "
            r"\(about 0\.2561 seconds at 8,000 Hz\)"
        ),
    ):
        estimator.estimate(
            midi=source.getvalue(),
            audio=DecodedAudio(torch.zeros(2, 2_049), sample_rate=8_000),
            stem_kind="other",
            options=VelocityOptions(window_seconds=2_048 / 8_000),
        )

    result = estimator.estimate(
        midi=source.getvalue(),
        audio=DecodedAudio(torch.zeros(2, 2_049), sample_rate=8_000),
        stem_kind="other",
        options=VelocityOptions(window_seconds=2_049 / 8_000),
    )

    assert result.velocity_applied is True
    assert result.note_count == 1


def test_estimator_rejects_a_dangling_positive_note_on(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
                mido.MetaMessage("end_of_track", time=120),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)

    with pytest.raises(ValueError, match="could not be paired"):
        estimator.estimate(
            midi=source.getvalue(),
            audio=DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000),
            stem_kind="other",
        )


def test_velocity_estimator_rejects_concurrent_calls_without_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=120),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)
    audio = DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000)
    first_call_entered = Event()
    release_first_call = Event()
    call_count = 0

    def blocking_prediction(**_kwargs: object) -> VelocityPredictions:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_call_entered.set()
            assert release_first_call.wait(timeout=5.0)
        return VelocityPredictions(
            velocities=np.asarray([64], dtype=np.int32),
            stem_gains_db=None,
        )

    monkeypatch.setattr(
        estimator_module,
        "predict_velocity_values",
        blocking_prediction,
    )

    def estimate() -> object:
        return estimator.estimate(
            midi=source.getvalue(),
            audio=audio,
            stem_kind="other",
        )

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_result = executor.submit(estimate)
        assert first_call_entered.wait(timeout=5.0)
        with pytest.raises(VelocityEstimatorBusyError):
            estimate()
        release_first_call.set()
        first_result.result(timeout=5.0)

    estimate()
    assert call_count == 2


def test_velocity_estimator_rejects_calls_after_close(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")

    estimator.close()

    with pytest.raises(RuntimeError, match="closed"):
        estimator.estimate(
            midi=b"",
            audio=DecodedAudio(torch.zeros(1, 1), sample_rate=8_000),
            stem_kind="other",
        )


def test_velocity_estimator_close_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    devices: list[torch.device | str] = []
    monkeypatch.setattr(estimator_module, "empty_device_cache", devices.append)

    assert estimator.closed is False

    estimator.close()
    estimator.close()

    assert estimator.closed is True
    assert devices == [torch.device("cpu")]


def test_velocity_estimator_context_manager_closes_on_exit(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")

    with estimator as managed_estimator:
        assert managed_estimator is estimator
        assert managed_estimator.closed is False

    assert estimator.closed is True


def test_velocity_estimator_close_waits_for_an_active_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=120),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)
    audio = DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000)
    estimate_entered = Event()
    release_estimate = Event()
    close_started = Event()
    close_finished = Event()

    def blocking_prediction(**_kwargs: object) -> VelocityPredictions:
        estimate_entered.set()
        assert release_estimate.wait(timeout=5.0)
        return VelocityPredictions(
            velocities=np.asarray([64], dtype=np.int32),
            stem_gains_db=None,
        )

    monkeypatch.setattr(
        estimator_module,
        "predict_velocity_values",
        blocking_prediction,
    )

    def estimate() -> object:
        return estimator.estimate(
            midi=source.getvalue(),
            audio=audio,
            stem_kind="other",
        )

    def close() -> None:
        close_started.set()
        estimator.close()
        close_finished.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        estimate_result = executor.submit(estimate)
        assert estimate_entered.wait(timeout=5.0)
        close_result = executor.submit(close)
        assert close_started.wait(timeout=5.0)
        assert close_finished.wait(timeout=0.1) is False
        assert estimator.closed is False

        release_estimate.set()
        result = estimate_result.result(timeout=5.0)
        close_result.result(timeout=5.0)

    assert result.note_count == 1
    assert close_finished.is_set()
    assert estimator.closed is True


def test_velocity_estimator_close_empties_its_device_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    model_info = estimator.model_info
    load_report = estimator.load_report
    devices: list[torch.device | str] = []
    monkeypatch.setattr(
        estimator_module,
        "empty_device_cache",
        devices.append,
        raising=False,
    )

    estimator.close()

    assert devices == [torch.device(model_info.device)]
    assert estimator.model_info is model_info
    assert estimator.load_report is load_report


def test_estimate_leaves_decoded_audio_samples_unchanged(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "velocity.pth"
    _write_tiny_velocity_checkpoint(checkpoint_path)
    estimator = VelocityEstimator.from_checkpoint(checkpoint_path, device="cpu")
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks.append(
        mido.MidiTrack(
            [
                mido.Message("note_on", channel=0, note=60, velocity=80, time=0),
                mido.Message("note_off", channel=0, note=60, velocity=0, time=120),
                mido.MetaMessage("end_of_track", time=0),
            ]
        )
    )
    source = io.BytesIO()
    midi.save(file=source)
    samples = torch.linspace(-1.0, 1.0, 8_000).repeat(2, 1)
    samples_before = samples.clone()

    estimator.estimate(
        midi=source.getvalue(),
        audio=DecodedAudio(samples=samples, sample_rate=8_000),
        stem_kind="other",
    )

    assert torch.equal(samples, samples_before)
