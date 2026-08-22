from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

import preprocess.resample_only as resample_only
from instrument_agnostic_amt.beat_chord.datasets.common import (
    read_audio_duration_entry,
)
from instrument_agnostic_amt.beat_chord.heads.beat import BeatDataset
from instrument_agnostic_amt.beat_chord.heads.chord import ChordDataset
from preprocess.resample_only import needs_resample, resample_in_place


def _write_mono_wav(path: Path, *, sample_rate: int, frames: int) -> None:
    waveform = np.linspace(-0.5, 0.5, frames, dtype=np.float32)
    sf.write(path, waveform, sample_rate, subtype="FLOAT")


def test_audio_duration_entry_reads_soundfile_metadata(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_mono_wav(audio_path, sample_rate=8_000, frames=4_000)

    entry = read_audio_duration_entry(audio_path)

    assert (entry.sample_rate, entry.num_frames) == (8_000, 4_000)


def test_beat_dataset_reads_and_resamples_audio_with_soundfile(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    label_dir = tmp_path / "label"
    audio_dir.mkdir()
    label_dir.mkdir()
    _write_mono_wav(audio_dir / "song.wav", sample_rate=8_000, frames=8_000)
    (label_dir / "song.beat.beats.json").write_text(
        json.dumps(
            {
                "measures": [
                    {
                        "downbeat_sec": 0.0,
                        "time_sig_num": 4,
                        "time_sig_den": 4,
                        "tempo_bpm": 240.0,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = BeatDataset(
        tmp_path,
        window_ms=1_000,
        sample_rate=4_000,
        hop_length=100,
    )

    sample = dataset[0]

    assert sample["audio"].shape == torch.Size((2, 4_000))


def test_chord_dataset_reads_and_resamples_audio_with_soundfile(tmp_path: Path) -> None:
    audio_dir = tmp_path / "audio"
    chord_label_dir = tmp_path / "chord_label"
    key_label_dir = tmp_path / "key_label"
    audio_dir.mkdir()
    chord_label_dir.mkdir()
    key_label_dir.mkdir()
    _write_mono_wav(audio_dir / "song.wav", sample_rate=8_000, frames=8_000)
    (chord_label_dir / "song.jsonl").write_text(
        json.dumps(
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "root": "C",
                "quality": "",
                "bass": "C",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (key_label_dir / "song.txt").write_text("0:1:C\n", encoding="utf-8")
    quality_map = {str(index): f"quality-{index}" for index in range(63)}
    quality_map["0"] = ""
    quality_map["62"] = "N"
    (tmp_path / "quality.json").write_text(
        json.dumps(quality_map), encoding="utf-8"
    )
    (tmp_path / "quality_freq_count.json").write_text(
        json.dumps([1] * 63), encoding="utf-8"
    )
    dataset = ChordDataset(
        tmp_path,
        window_ms=1_000,
        sample_rate=4_000,
        hop_length=100,
    )

    sample = dataset[0]

    assert sample["audio"].shape == torch.Size((2, 4_000))


def test_resample_in_place_rewrites_audio_at_target_rate(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_mono_wav(audio_path, sample_rate=8_000, frames=8_000)

    assert needs_resample(audio_path, 4_000)

    resample_in_place(audio_path, 4_000)

    info = sf.info(audio_path)
    assert (needs_resample(audio_path, 4_000), info.samplerate, info.frames) == (
        False,
        4_000,
        4_000,
    )
    assert info.subtype == "FLOAT"


def test_resample_in_place_keeps_original_when_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_mono_wav(audio_path, sample_rate=8_000, frames=8_000)
    original_bytes = audio_path.read_bytes()

    def fail_after_truncating_target(path: str, *_args: object, **_kwargs: object) -> None:
        Path(path).write_bytes(b"")
        raise RuntimeError("write failed")

    monkeypatch.setattr(resample_only.sf, "write", fail_after_truncating_target)

    with pytest.raises(RuntimeError, match="write failed"):
        resample_in_place(audio_path, 4_000)

    assert audio_path.read_bytes() == original_bytes
    assert list(tmp_path.iterdir()) == [audio_path]
