from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import preprocess.resample_only as resample_only
from recipes.beat_chord.datasets.common import (
    read_audio_duration_entry,
)
from preprocess.resample_only import needs_resample, resample_in_place


def _write_mono_wav(path: Path, *, sample_rate: int, frames: int) -> None:
    waveform = np.linspace(-0.5, 0.5, frames, dtype=np.float32)
    sf.write(path, waveform, sample_rate, subtype="FLOAT")


def test_audio_duration_entry_reads_soundfile_metadata(tmp_path: Path) -> None:
    audio_path = tmp_path / "sample.wav"
    _write_mono_wav(audio_path, sample_rate=8_000, frames=4_000)

    entry = read_audio_duration_entry(audio_path)

    assert (entry.sample_rate, entry.num_frames) == (8_000, 4_000)


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
