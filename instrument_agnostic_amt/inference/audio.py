from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as audio_functional

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
}


def load_audio(audio_path: Path, *, target_sample_rate: int) -> torch.Tensor:
    waveform_np, source_sample_rate = sf.read(
        audio_path, dtype="float32", always_2d=True
    )
    waveform = torch.from_numpy(waveform_np.T.copy())
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    if int(source_sample_rate) != int(target_sample_rate):
        waveform = audio_functional.resample(
            waveform, int(source_sample_rate), int(target_sample_rate)
        )
    return waveform.contiguous()


def collect_audio_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
