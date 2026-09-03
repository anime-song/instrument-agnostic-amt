from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class DecodedAudio:
    """Decoded PCM audio paired with its source sample rate."""

    samples: torch.Tensor
    sample_rate: int

    def __post_init__(self) -> None:
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        if not isinstance(self.samples, torch.Tensor):
            raise TypeError("samples must be a torch.Tensor")
        if self.samples.device.type != "cpu":
            raise ValueError("samples must be on CPU")
        if self.samples.ndim != 2:
            raise ValueError("samples must have shape [channels, audio_frames]")
        if int(self.samples.shape[0]) not in (1, 2):
            raise ValueError("samples must have one or two channels")
        if int(self.samples.shape[1]) == 0:
            raise ValueError("samples must not be empty")
        if self.samples.dtype != torch.float32:
            raise ValueError("samples must have dtype float32")


def load_audio(audio_path: Path, *, target_sample_rate: int) -> torch.Tensor:
    waveform_np, source_sample_rate = sf.read(
        audio_path, dtype="float32", always_2d=True
    )
    if waveform_np.shape[1] > 2:
        waveform_np = waveform_np[:, :2]
    elif waveform_np.shape[1] == 1:
        waveform_np = waveform_np.repeat(2, axis=1)
    waveform = torch.from_numpy(waveform_np.T.copy())
    if int(source_sample_rate) != int(target_sample_rate):
        waveform = audio_functional.resample(
            waveform, int(source_sample_rate), int(target_sample_rate)
        )
    return waveform.contiguous()


def prepare_waveform(
    audio: str | Path | DecodedAudio,
    *,
    target_sample_rate: int,
) -> torch.Tensor:
    """Load or normalize decoded audio for model inference."""

    if not isinstance(audio, DecodedAudio):
        return load_audio(Path(audio), target_sample_rate=target_sample_rate)

    waveform = audio.samples.detach()
    if int(waveform.shape[0]) == 1:
        waveform = waveform.repeat(2, 1)
    if int(audio.sample_rate) != int(target_sample_rate):
        waveform = audio_functional.resample(
            waveform,
            int(audio.sample_rate),
            int(target_sample_rate),
        )
    return waveform.contiguous()


def collect_audio_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )
