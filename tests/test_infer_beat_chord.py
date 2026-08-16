from __future__ import annotations

from pathlib import Path
import numpy as np
import pretty_midi
import torch

from infer_beat_chord import (
    DEFAULT_BEAT_CHORD_CHECKPOINT_FILENAME,
    ensure_beat_chord_checkpoint,
    load_beat_chord_model,
    predict_beat_chord_for_midi,
)
from instrument_agnostic_amt.beat_chord import (
    MidiFrameBeatChordModel,
    MidiFrameModelConfig,
)
from instrument_agnostic_amt.beat_chord.cli import infer as beat_infer


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

    def __call__(
        self,
        roll: torch.Tensor,
        *,
        include_beat: bool,
        include_chord: bool,
    ) -> dict[str, torch.Tensor]:
        assert include_beat and include_chord
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


def test_ensure_beat_chord_checkpoint_resolves_existing(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "custom_checkpoint.pth"
    checkpoint_path.write_bytes(b"dummy")

    resolved = ensure_beat_chord_checkpoint(checkpoint_path)
    assert resolved == checkpoint_path.resolve()


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


def test_run_beat_chord_batches_windows_without_changing_results(
    tmp_path: Path,
    monkeypatch,
) -> None:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.append(
        pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=3.1)
    )
    midi.instruments.append(instrument)
    midi_path = tmp_path / "four_windows.mid"
    midi.write(str(midi_path))

    model_config = MidiFrameModelConfig(
        sample_rate=10,
        hop_length=1,
        num_input_channels=1,
        num_meter_classes=1,
        num_root_chord_classes=25,
        pitch_min=21,
        pitch_max=108,
    )
    metadata = {
        "meter_classes": None,
        "quality_map": {"0": "", "1": "N"},
        "has_major_grouping_head": False,
        "checkpoint_args": {},
    }
    monkeypatch.setattr(beat_infer, "SingleMidiFrameLoader", _FixedWindowLoader)

    def run(window_batch_size: int):
        model = _BatchTrackingBeatChordModel(model_config)
        config = beat_infer.BeatChordInferenceConfig(
            device=torch.device("cpu"),
            window_ms_override=1_000,
            stride_ms_override=1_000,
            beat_decode_mode="peaks",
            beat_threshold=1.1,
            downbeat_threshold=1.1,
        )
        config.window_batch_size = window_batch_size
        result = beat_infer.run_beat_chord_inference(
            midi_path=midi_path,
            checkpoint={},
            model=model,
            model_config=model_config,
            metadata=metadata,
            config=config,
        )
        return model.batch_sizes, result

    single_sizes, single = run(1)
    batched_sizes, batched = run(2)

    assert single_sizes == [1, 1, 1, 1]
    assert batched_sizes == [2, 2]
    np.testing.assert_array_equal(batched.beat_probabilities, single.beat_probabilities)
    np.testing.assert_array_equal(
        batched.downbeat_probabilities,
        single.downbeat_probabilities,
    )
    np.testing.assert_array_equal(batched.meter_logits, single.meter_logits)
    assert batched.beat_times == single.beat_times
    assert batched.downbeat_times == single.downbeat_times
    assert batched.chord_segments == single.chord_segments
    assert batched.key_segments == single.key_segments

    monkeypatch.setattr(
        beat_infer,
        "SingleMidiFrameLoader",
        _LastWindowFailingLoader,
    )
    failed_single_sizes, failed_single = run(1)
    failed_batched_sizes, failed_batched = run(4)

    assert failed_single_sizes == [1, 1, 1]
    assert failed_batched_sizes == [3]
    np.testing.assert_array_equal(
        failed_batched.beat_probabilities,
        failed_single.beat_probabilities,
    )
    assert failed_batched.chord_segments == failed_single.chord_segments
    assert failed_batched.key_segments == failed_single.key_segments
