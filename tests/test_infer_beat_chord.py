from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest
import torch

from instrument_agnostic_amt.beat_chord.cli import infer as beat_chord_infer
from instrument_agnostic_amt.beat_chord.cli.infer import (
    DEFAULT_BEAT_CHORD_CHECKPOINT_FILENAME,
    ensure_beat_chord_checkpoint,
    load_beat_chord_model,
    predict_beat_chord_for_midi,
)
from instrument_agnostic_amt.beat_chord import (
    MidiFrameBeatChordModel,
    MidiFrameModelConfig,
)


class _FixedWindowLoader:
    def __init__(self, config: object, midi_file: Path) -> None:
        self.config = config
        self.midi_file = midi_file

    def load_window(
        self,
        *,
        song_name: str,
        window_start_sec: float,
        num_frames: int,
    ) -> torch.Tensor:
        _ = song_name
        return torch.full(
            (
                int(self.config.num_channels),  # type: ignore[attr-defined]
                int(num_frames),
                int(self.config.num_pitch_bins),  # type: ignore[attr-defined]
            ),
            float(window_start_sec),
        )


class _LastWindowFailingLoader(_FixedWindowLoader):
    def load_window(
        self,
        *,
        song_name: str,
        window_start_sec: float,
        num_frames: int,
    ) -> torch.Tensor:
        if window_start_sec >= 3.0:
            raise RuntimeError("最後の窓だけ読込失敗")
        return super().load_window(
            song_name=song_name,
            window_start_sec=window_start_sec,
            num_frames=num_frames,
        )


class _BatchTrackingBeatChordModel:
    def __init__(self, model_config: MidiFrameModelConfig) -> None:
        self.model_config = model_config
        self.batch_sizes: list[int] = []
        self.aux_output_flags: list[bool | None] = []

    def __call__(
        self,
        roll: torch.Tensor,
        *,
        include_beat: bool,
        include_chord: bool,
        include_aux_outputs: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        assert include_beat and include_chord
        self.aux_output_flags.append(include_aux_outputs)
        batch_size, _, frame_count, _ = roll.shape
        self.batch_sizes.append(int(batch_size))
        window_value = roll[:, 0, 0, 0].view(batch_size, 1)
        frame_logits = window_value.expand(batch_size, frame_count)
        return {
            "beat_logits": frame_logits,
            "downbeat_logits": frame_logits - 0.25,
            "group_boundary_logits": frame_logits - 0.5,
            "meter_logits": torch.zeros(
                batch_size,
                frame_count,
                self.model_config.num_meter_classes,
            ),
            "root_chord_logits": torch.zeros(
                batch_size,
                frame_count,
                self.model_config.num_root_chord_classes,
            ),
            "chord_boundary_logits": frame_logits - 1.0,
            "bass_logits": torch.zeros(batch_size, frame_count, 13),
            "key_boundary_logits": frame_logits - 1.5,
            "key_logits": torch.zeros(batch_size, frame_count, 13),
        }


def _write_windowed_midi(path: Path, *, duration_seconds: float) -> None:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(
        pretty_midi.Note(
            velocity=100,
            pitch=60,
            start=0.0,
            end=duration_seconds,
        )
    )
    midi.instruments.append(instrument)
    midi.write(str(path))


def _run_windowed_inference(
    midi_path: Path,
    *,
    window_batch_size: int,
    model: _BatchTrackingBeatChordModel | None = None,
) -> tuple[list[int], beat_chord_infer.BeatChordInferenceResult]:
    model_config = MidiFrameModelConfig(
        sample_rate=10,
        hop_length=1,
        num_input_channels=1,
        num_meter_classes=1,
        num_root_chord_classes=25,
        pitch_min=21,
        pitch_max=108,
    )
    if model is None:
        model = _BatchTrackingBeatChordModel(model_config)
    config = beat_chord_infer.BeatChordInferenceConfig(
        device=torch.device("cpu"),
        window_ms_override=1_000,
        stride_ms_override=1_000,
        window_batch_size=window_batch_size,
        beat_decode_mode="peaks",
        beat_threshold=1.1,
        downbeat_threshold=1.1,
    )
    result = beat_chord_infer.run_beat_chord_inference(
        midi_path=midi_path,
        checkpoint={},
        model=model,
        model_config=model_config,
        metadata={
            "meter_classes": None,
            "quality_map": {"0": "", "1": "N"},
            "has_major_grouping_head": False,
            "checkpoint_args": {},
        },
        config=config,
    )
    return model.batch_sizes, result


def test_ensure_beat_chord_checkpoint_resolves_existing(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "custom_checkpoint.pth"
    checkpoint_path.write_bytes(b"dummy")

    resolved = ensure_beat_chord_checkpoint(checkpoint_path)
    assert resolved == checkpoint_path.resolve()


def test_load_beat_chord_model_reads_checkpoint_on_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MidiFrameModelConfig(
        sample_rate=16_000,
        hop_length=160,
        num_input_channels=1,
        num_meter_classes=1,
        num_root_chord_classes=13,
        pitch_min=21,
        pitch_max=108,
    )
    source_model = MidiFrameBeatChordModel(config)
    checkpoint_path = tmp_path / "beat_chord.pth"
    torch.save(
        {
            "model_config": config.__dict__,
            "model_state_dict": source_model.state_dict(),
            "beat_meter_classes": [[4, 4]],
            "chord_quality_map": {"0": "", "1": "N"},
        },
        checkpoint_path,
    )
    requested_map_locations: list[object] = []
    original_load = torch.load

    def tracked_load(*args: object, **kwargs: object) -> object:
        requested_map_locations.append(kwargs.get("map_location"))
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    monkeypatch.setattr(MidiFrameBeatChordModel, "to", lambda self, _device: self)

    _, _, metadata = load_beat_chord_model(checkpoint_path, device="mps")

    assert requested_map_locations == ["cpu"]
    checkpoint = metadata["checkpoint"]
    assert isinstance(checkpoint, dict)
    state_dict = checkpoint["model_state_dict"]
    assert all(tensor.device.type == "cpu" for tensor in state_dict.values())


def test_beat_chord_auto_routes_model_to_mps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_midi = tmp_path / "input.mid"
    input_midi.write_bytes(b"midi")
    loaded_devices: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        beat_chord_infer,
        "ensure_beat_chord_checkpoint",
        lambda _path: tmp_path / "checkpoint.pth",
    )

    def fake_load(
        _path: Path,
        *,
        device: torch.device,
    ) -> tuple[object, object, dict[str, object]]:
        loaded_devices.append(str(device))
        return object(), object(), {"checkpoint": {}}

    monkeypatch.setattr(beat_chord_infer, "load_beat_chord_model", fake_load)
    monkeypatch.setattr(
        beat_chord_infer,
        "run_beat_chord_inference",
        lambda **_kwargs: SimpleNamespace(
            beat_times=[],
            meter_segments=[],
            chord_segments=[],
            key_segments=[],
            duration_seconds=0.0,
            hop_length=1,
            sample_rate=1,
        ),
    )
    monkeypatch.setattr(
        beat_chord_infer,
        "export_tempo_mapped_midi",
        lambda **_kwargs: None,
    )

    predict_beat_chord_for_midi(input_midi, device=None)

    assert loaded_devices == ["mps"]


def test_predict_beat_chord_reuses_preloaded_model_and_auto_enables_jit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_midi = tmp_path / "input.mid"
    input_midi.write_bytes(b"midi")
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        beat_chord_infer,
        "ensure_beat_chord_checkpoint",
        lambda *_args, **_kwargs: pytest.fail("checkpoint should not be resolved"),
    )
    monkeypatch.setattr(
        beat_chord_infer,
        "load_beat_chord_model",
        lambda *_args, **_kwargs: pytest.fail("model should not be loaded"),
    )
    monkeypatch.setattr(
        "instrument_agnostic_amt.beat_chord.decoding.beat_grid_jit.is_jit_grid_available",
        lambda: True,
    )

    def fake_run(**kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(
            beat_times=[],
            meter_segments=[],
            chord_segments=[],
            key_segments=[],
            duration_seconds=0.0,
            hop_length=1,
            sample_rate=1,
        )

    monkeypatch.setattr(beat_chord_infer, "run_beat_chord_inference", fake_run)
    monkeypatch.setattr(
        beat_chord_infer,
        "export_tempo_mapped_midi",
        lambda **_kwargs: None,
    )
    model = torch.nn.Linear(1, 1)
    model_config = object()
    metadata = {"checkpoint_args": {}}

    predict_beat_chord_for_midi(
        input_midi,
        device="cpu",
        preloaded_model=model,
        preloaded_model_config=model_config,
        preloaded_metadata=metadata,
    )

    assert captured["model"] is model
    assert captured["model_config"] is model_config
    assert captured["checkpoint"] == {}
    assert captured["config"].grid_jit is True


def test_run_beat_chord_batches_windows_without_changing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    midi_path = tmp_path / "four_windows.mid"
    _write_windowed_midi(midi_path, duration_seconds=3.1)
    monkeypatch.setattr(
        beat_chord_infer,
        "SingleMidiFrameLoader",
        _FixedWindowLoader,
    )

    single_sizes, single = _run_windowed_inference(
        midi_path,
        window_batch_size=1,
    )
    batched_sizes, batched = _run_windowed_inference(
        midi_path,
        window_batch_size=2,
    )

    assert single_sizes == [1, 1, 1, 1]
    assert batched_sizes == [2, 2]
    np.testing.assert_array_equal(
        batched.beat_probabilities,
        single.beat_probabilities,
    )
    np.testing.assert_array_equal(
        batched.downbeat_probabilities,
        single.downbeat_probabilities,
    )
    np.testing.assert_array_equal(batched.meter_logits, single.meter_logits)
    assert batched.beat_times == single.beat_times
    assert batched.downbeat_times == single.downbeat_times
    assert batched.chord_segments == single.chord_segments
    assert batched.key_segments == single.key_segments


def test_run_beat_chord_flushes_the_final_partial_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    midi_path = tmp_path / "five_windows.mid"
    _write_windowed_midi(midi_path, duration_seconds=4.1)
    monkeypatch.setattr(
        beat_chord_infer,
        "SingleMidiFrameLoader",
        _FixedWindowLoader,
    )

    _, single = _run_windowed_inference(midi_path, window_batch_size=1)
    batch_sizes, batched = _run_windowed_inference(
        midi_path,
        window_batch_size=2,
    )

    assert batch_sizes == [2, 2, 1]
    np.testing.assert_array_equal(
        batched.beat_probabilities,
        single.beat_probabilities,
    )


def test_run_beat_chord_bulk_transfers_each_batch_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    midi_path = tmp_path / "four_windows.mid"
    _write_windowed_midi(midi_path, duration_seconds=3.1)
    monkeypatch.setattr(
        beat_chord_infer,
        "SingleMidiFrameLoader",
        _FixedWindowLoader,
    )
    transferred_tensor_counts: list[int] = []

    def tracked_copy(
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        values = tuple(tensors)
        transferred_tensor_counts.append(len(values))
        return tuple(value.cpu() for value in values)

    monkeypatch.setattr(
        beat_chord_infer,
        "copy_tensors_to_cpu_once",
        tracked_copy,
        raising=False,
    )

    _run_windowed_inference(midi_path, window_batch_size=2)

    assert transferred_tensor_counts == [9, 9]


def test_run_beat_chord_omits_auxiliary_outputs_for_the_real_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    midi_path = tmp_path / "two_windows.mid"
    _write_windowed_midi(midi_path, duration_seconds=1.1)
    monkeypatch.setattr(
        beat_chord_infer,
        "SingleMidiFrameLoader",
        _FixedWindowLoader,
    )
    model_config = MidiFrameModelConfig(
        sample_rate=10,
        hop_length=1,
        num_input_channels=1,
        num_meter_classes=1,
        num_root_chord_classes=25,
        pitch_min=21,
        pitch_max=108,
    )
    model = _BatchTrackingBeatChordModel(model_config)
    monkeypatch.setattr(
        beat_chord_infer,
        "MidiFrameBeatChordModel",
        _BatchTrackingBeatChordModel,
    )

    _run_windowed_inference(
        midi_path,
        window_batch_size=2,
        model=model,
    )

    assert model.aux_output_flags == [False]


def test_run_beat_chord_keeps_loaded_windows_when_the_last_load_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    midi_path = tmp_path / "last_window_failure.mid"
    _write_windowed_midi(midi_path, duration_seconds=3.1)
    monkeypatch.setattr(
        beat_chord_infer,
        "SingleMidiFrameLoader",
        _LastWindowFailingLoader,
    )

    single_sizes, single = _run_windowed_inference(
        midi_path,
        window_batch_size=1,
    )
    batched_sizes, batched = _run_windowed_inference(
        midi_path,
        window_batch_size=4,
    )

    assert single_sizes == [1, 1, 1]
    assert batched_sizes == [3]
    np.testing.assert_array_equal(
        batched.beat_probabilities,
        single.beat_probabilities,
    )
    assert batched.chord_segments == single.chord_segments
    assert batched.key_segments == single.key_segments


def test_predict_beat_chord_for_midi(tmp_path: Path) -> None:
    # 1. テスト用 MIDI の作成
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, name="Piano")
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=2.0))
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.5, end=2.5))
    pm.instruments.append(inst)
    midi_path = tmp_path / "sample.mid"
    pm.write(str(midi_path))

    # 2. テスト用ダミーチェックポイントの作成
    config = MidiFrameModelConfig(
        sample_rate=16000,
        hop_length=160,
        num_input_channels=1,
        num_meter_classes=1,
        num_root_chord_classes=13,
        pitch_min=21,
        pitch_max=108,
    )
    model = MidiFrameBeatChordModel(config)
    checkpoint = {
        "model_config": config.__dict__,
        "model_state_dict": model.state_dict(),
        "beat_meter_classes": [[4, 4]],
        "chord_quality_map": {"0": "", "1": "N"},
    }
    checkpoint_path = tmp_path / "best_beat_chord_key.pth"
    torch.save(checkpoint, checkpoint_path)

    # 3. 推論関数の実行
    output_path = tmp_path / "sample.beat_mapped.mid"
    result_path = predict_beat_chord_for_midi(
        input_midi_path=midi_path,
        output_midi_path=output_path,
        checkpoint_path=checkpoint_path,
        device="cpu",
        beat_decode_mode="peaks",
        window_batch_size=2,
    )

    assert result_path.exists()
    assert result_path == output_path.resolve()

    output_pm = pretty_midi.PrettyMIDI(str(result_path))
    assert len(output_pm.instruments) >= 1
