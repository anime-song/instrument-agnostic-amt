from __future__ import annotations

import json
from pathlib import Path
import pretty_midi

from recipes.beat_chord.convert_beat_dataset import (
    convert_midi_dir_to_beat_dataset,
)


def create_sample_midi(midi_path: Path) -> None:
    """テスト用のシンプルな MIDI ファイルを作成する。"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0)
    # C4 ノートを 0.0s - 2.0s 間鳴らす
    instrument.notes.append(
        pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=2.0)
    )
    midi.instruments.append(instrument)
    midi.write(str(midi_path))


def test_convert_midi_dir_to_beat_dataset(tmp_path: Path) -> None:
    midi_dir = tmp_path / "midi_input"
    output_dir = tmp_path / "beat_output"
    midi_dir.mkdir(parents=True, exist_ok=True)

    midi_file_path = midi_dir / "sample_song.mid"
    create_sample_midi(midi_file_path)

    success_count, skip_count = convert_midi_dir_to_beat_dataset(
        midi_dir=midi_dir,
        output_dataset_dir=output_dir,
    )

    assert success_count == 1
    assert skip_count == 0

    label_json_path = output_dir / "label" / "sample_song.beat.beats.json"
    audio_wav_path = output_dir / "audio" / "sample_song.wav"

    assert label_json_path.exists()
    assert audio_wav_path.exists()

    with open(label_json_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    assert "measures" in label_data
    assert len(label_data["measures"]) > 0
    first_measure = label_data["measures"][0]
    assert first_measure["downbeat_sec"] == 0.0
    assert first_measure["time_sig_num"] == 4
    assert first_measure["time_sig_den"] == 4
    assert first_measure["tempo_bpm"] == 120.0


def test_convert_midi_with_tempo_and_time_signature_changes(tmp_path: Path) -> None:
    """途中でテンポや拍子が変化する MIDI ファイルの変換結果を検証する。"""
    import mido

    midi_dir = tmp_path / "complex_midi"
    output_dir = tmp_path / "complex_beat_output"
    midi_dir.mkdir(parents=True, exist_ok=True)
    midi_path = midi_dir / "tempo_change_song.mid"

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # 1小節目: 4/4 拍子, 120 BPM (500,000 µs/beat)
    # 4/4 1小節 = 4 beats = 4 * 480 = 1920 ticks
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))  # 120 BPM
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=1920))

    # 2小節目: 3/4 拍子, 150 BPM (400,000 µs/beat)
    # 3/4 1小節 = 3 beats = 3 * 480 = 1440 ticks
    track.append(mido.MetaMessage("time_signature", numerator=3, denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=400000, time=0))  # 150 BPM
    track.append(mido.Message("note_on", note=62, velocity=64, time=0))
    track.append(mido.Message("note_off", note=62, velocity=0, time=1440))

    midi.save(str(midi_path))

    success_count, skip_count = convert_midi_dir_to_beat_dataset(
        midi_dir=midi_dir,
        output_dataset_dir=output_dir,
        export_audacity=True,
    )

    assert success_count == 1
    assert skip_count == 0

    label_json_path = output_dir / "label" / "tempo_change_song.beat.beats.json"
    audacity_txt_path = output_dir / "audacity_labels" / "tempo_change_song.beats.txt"

    assert label_json_path.exists()
    assert audacity_txt_path.exists()

    with open(label_json_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    measures = label_data["measures"]
    assert len(measures) == 2

    # 1小節目の検証 (4/4, 120 BPM)
    assert measures[0]["downbeat_sec"] == 0.0
    assert measures[0]["time_sig_num"] == 4
    assert measures[0]["time_sig_den"] == 4
    assert abs(measures[0]["tempo_bpm"] - 120.0) < 1e-3

    # 1小節目の長さ = 4 beats * (60 / 120) = 2.0 秒
    # したがって2小節目の downbeat_sec は 2.0 秒になるはず
    assert abs(measures[1]["downbeat_sec"] - 2.0) < 1e-3
    assert measures[1]["time_sig_num"] == 3
    assert measures[1]["time_sig_den"] == 4
    assert abs(measures[1]["tempo_bpm"] - 150.0) < 1e-3

    # beat_times の検証
    assert "beat_times" in measures[0]
    assert len(measures[0]["beat_times"]) == 4
    assert "beat_times" in measures[1]
    assert len(measures[1]["beat_times"]) == 3


def test_ritardando_beat_times_and_dataset_loading(tmp_path: Path) -> None:
    """1小節内で rit. (減速) がある場合に beat_times が正確に非線形計算されるかを検証する。"""
    import mido
    from recipes.beat_chord.datasets.beat import MidiBeatDataset

    midi_dir = tmp_path / "rit_midi"
    output_dir = tmp_path / "rit_beat_output"
    midi_dir.mkdir(parents=True, exist_ok=True)
    midi_path = midi_dir / "rit_song.mid"

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # 4/4 拍子
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))

    # 0.0s - 1.0s (最初の2拍): 120 BPM (500,000 µs/beat). 2 beats = 960 ticks.
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))

    # 2拍目の後 (960 ticks) に 60 BPM (1,000,000 µs/beat) に急減速 (rit.)
    # 3拍目・4拍目は各 1.0 秒かかる
    track.append(mido.MetaMessage("set_tempo", tempo=1000000, time=960))
    track.append(
        mido.Message("note_off", note=60, velocity=0, time=960)
    )  # 960 ticks = 2 beats at 60BPM = 1920 ticks total for measure

    midi.save(str(midi_path))

    success_count, skip_count = convert_midi_dir_to_beat_dataset(
        midi_dir=midi_dir,
        output_dataset_dir=output_dir,
    )

    assert success_count == 1
    assert skip_count == 0

    label_json_path = output_dir / "label" / "rit_song.beat.beats.json"
    with open(label_json_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    measure = label_data["measures"][0]
    beat_times = measure["beat_times"]

    # 期待されるビートタイム:
    # 1拍目 (tick 0)   = 0.0s
    # 2拍目 (tick 480) = 0.5s
    # 3拍目 (tick 960) = 1.0s (tempo 変更直前/直後)
    # 4拍目 (tick 1440) = 1.0s + 1.0s = 2.0s (60BPM なので 1拍 = 1秒)
    assert len(beat_times) == 4
    assert abs(beat_times[0] - 0.0) < 1e-3
    assert abs(beat_times[1] - 0.5) < 1e-3
    assert abs(beat_times[2] - 1.0) < 1e-3
    assert abs(beat_times[3] - 2.0) < 1e-3

    # MidiBeatDataset が beat_times を正しく読み込めるか検証
    dataset = MidiBeatDataset(
        root=output_dir,
        midi_dir=midi_dir,
        window_ms=10000,
        sample_rate=16000,
        hop_length=160,
        pitch_min=21,
        pitch_max=108,
        num_input_channels=4,
    )

    assert len(dataset) > 0
    item = dataset.items[0]
    # 全ビート位置が正確に渡されていること
    assert len(item["beat_times"]) >= 4
    assert abs(item["beat_times"][0] - 0.0) < 1e-3
    assert abs(item["beat_times"][1] - 0.5) < 1e-3
    assert abs(item["beat_times"][2] - 1.0) < 1e-3
    assert abs(item["beat_times"][3] - 2.0) < 1e-3


def test_duplicate_time_signature_event_no_shift(tmp_path: Path) -> None:
    """tick > 0 に重複した同一拍子イベントがあっても 1 拍ズレが生じないことを検証する。"""
    import mido

    midi_dir = tmp_path / "dup_ts_midi"
    output_dir = tmp_path / "dup_ts_output"
    midi_dir.mkdir(parents=True, exist_ok=True)
    midi_path = midi_dir / "dup_ts_song.mid"

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # tick=0 で 4/4
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))  # 120 BPM

    # tick=480 (1拍後ろ) で重複して同じ 4/4 が指定されているケース
    track.append(
        mido.MetaMessage("time_signature", numerator=4, denominator=4, time=480)
    )

    # 2小節分のノート (合計 3840 ticks)
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=3360))

    midi.save(str(midi_path))

    success_count, skip_count = convert_midi_dir_to_beat_dataset(
        midi_dir=midi_dir,
        output_dataset_dir=output_dir,
    )

    assert success_count == 1
    assert skip_count == 0

    label_json_path = output_dir / "label" / "dup_ts_song.beat.beats.json"
    with open(label_json_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    measures = label_data["measures"]
    # 1小節目が tick=0 から tick=480 (0.5s) の不完全小節として正しく切り出され、2小節目は tick=480 (0.5s) から始まること
    assert abs(measures[0]["downbeat_sec"] - 0.0) < 1e-3
    assert abs(measures[1]["downbeat_sec"] - 0.5) < 1e-3
    assert abs(measures[2]["downbeat_sec"] - 2.5) < 1e-3
