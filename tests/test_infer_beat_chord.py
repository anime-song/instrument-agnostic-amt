from __future__ import annotations

from pathlib import Path
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
    )

    assert result_path.exists()
    assert result_path == output_path.resolve()

    output_pm = pretty_midi.PrettyMIDI(str(result_path))
    assert len(output_pm.instruments) >= 1
