from __future__ import annotations

import math

import numpy as np
import soundfile as sf


def load_audio_window(
    audio_path: str, *, sample_rate: int, window_start_ms: int, window_ms: int
) -> np.ndarray:
    """Load a fixed audio window as stereo float32 [2, frames]."""
    start_frame = int(round(window_start_ms * sample_rate / 1000.0))
    window_frames = int(round(window_ms * sample_rate / 1000.0))
    audio, _ = sf.read(
        audio_path,
        start=start_frame,
        frames=window_frames,
        dtype="float32",
        always_2d=True,
    )
    if audio.shape[1] > 2:
        audio = audio[:, :2]
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    audio = audio.transpose(1, 0)  # [2, frames]
    # Zero-pad short reads near the end of a file.
    if audio.shape[1] < window_frames:
        padded = np.zeros((audio.shape[0], window_frames), dtype=np.float32)
        padded[:, : audio.shape[1]] = audio
        audio = padded
    return audio.astype(np.float32, copy=False)


def compute_model_frames(audio_frames: int, n_fft: int, hop_length: int) -> int:
    """Convert audio sample count to model frame count."""
    return math.ceil(audio_frames / hop_length)
