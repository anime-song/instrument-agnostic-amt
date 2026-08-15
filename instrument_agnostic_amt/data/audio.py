from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    num_frames: int


def inspect_audio(audio_path: str | Path) -> AudioInfo:
    """音声ファイルのサンプルレートとフレーム数を取得する。"""
    info = sf.info(str(audio_path))
    return AudioInfo(sample_rate=int(info.samplerate), num_frames=int(info.frames))


def read_audio_frames(
    audio_path: str | Path, *, frame_offset: int = 0, num_frames: int = -1
) -> tuple[np.ndarray, int]:
    """音声の指定範囲をチャンネル優先の float32 配列として読む。"""
    audio, sample_rate = sf.read(
        str(audio_path),
        start=int(frame_offset),
        frames=int(num_frames),
        dtype="float32",
        always_2d=True,
    )
    return np.ascontiguousarray(audio.transpose(1, 0)), int(sample_rate)


def load_audio_window(
    audio_path: str, *, sample_rate: int, window_start_ms: int, window_ms: int
) -> np.ndarray:
    """Load a fixed audio window as stereo float32 [2, frames]."""
    start_frame = int(round(window_start_ms * sample_rate / 1000.0))
    window_frames = int(round(window_ms * sample_rate / 1000.0))
    audio, _ = read_audio_frames(
        audio_path,
        frame_offset=start_frame,
        num_frames=window_frames,
    )
    if audio.shape[0] > 2:
        audio = audio[:2]
    elif audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)

    # Zero-pad short reads near the end of a file.
    if audio.shape[1] < window_frames:
        padded = np.zeros((audio.shape[0], window_frames), dtype=np.float32)
        padded[:, : audio.shape[1]] = audio
        audio = padded
    return audio.astype(np.float32, copy=False)


def compute_model_frames(audio_frames: int, n_fft: int, hop_length: int) -> int:
    """Convert audio sample count to model frame count."""
    return math.ceil(audio_frames / hop_length)
