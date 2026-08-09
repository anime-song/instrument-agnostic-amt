"""
faster-whisper word-level speech recognition on vocals stem,
injecting recognized lyrics as Lyric meta events into the stem's MIDI file.
Each word becomes an independent Lyric event, with a newline (\n) inserted
between whisper segments.
"""

from __future__ import annotations

import os
import shutil
from faster_whisper import WhisperModel
from pathlib import Path


_WHISPER_MODEL_CACHE: dict[str, WhisperModel] = {}


def _ensure_cublas_compat():
    """
    ctranslate2 (faster-whisper backend) requires cublasLt64_11.dll / cublas64_11.dll,
    but PyTorch CUDA 13 ships cublasLt64_13.dll / cublas64_13.dll.
    Copies the 13 versions to 11 if the 11 versions are missing.
    """
    try:
        import torch
        torch_dir = Path(torch.__file__).parent / "lib"
    except ImportError:
        return
    for name in ["cublasLt64_11.dll", "cublas64_11.dll"]:
        dst = torch_dir / name
        if dst.exists():
            continue
        src_name = name.replace("_11.", "_13.")
        src = torch_dir / src_name
        if src.exists():
            print(f"[WhisperLyrics] Linking {src_name} -> {name} for ctranslate2 compatibility")
            try:
                shutil.copy2(str(src), str(dst))
            except Exception:
                pass


def _get_whisper_model(model_size: str = "small") -> WhisperModel:
    """Get (and cache) a faster-whisper model instance."""
    if model_size not in _WHISPER_MODEL_CACHE:
        _ensure_cublas_compat()
        try:
            import torch
            has_cuda = torch.cuda.is_available()
        except ImportError:
            has_cuda = False
        device = "cuda" if has_cuda else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[WhisperLyrics] Loading faster-whisper {model_size} on {device} ({compute_type}) ...")
        _WHISPER_MODEL_CACHE[model_size] = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
    return _WHISPER_MODEL_CACHE[model_size]


def add_whisper_lyrics_to_vocals_midi(
    vocals_wav_path: str | Path,
    stem_midi_path: str | Path,
    *,
    language: str | None = None,
    model_size: str = "small",
    target_track_name: str = "vocals",
) -> None:
    """
    Transcribe a vocals wav with faster-whisper (word-level timestamps) and
    inject the lyrics as Lyric meta events into the stem's MIDI file (in-place).

    Args:
        vocals_wav_path: Path to the vocals stem audio.
        stem_midi_path:  Path to the stem MIDI file (will be modified in-place).
        language:        Audio language (None = auto-detect, "zh", "ja", "en", etc.).
        model_size:      faster-whisper model size ("small", "medium", "large-v3", etc.).
        target_track_name: Track name keyword to find the target track.
    """
    vocals_wav = Path(vocals_wav_path)
    stem_midi = Path(stem_midi_path)

    if not vocals_wav.exists():
        print(f"[WhisperLyrics] WARNING: vocals wav not found: {vocals_wav}, skipping.")
        return
    if not stem_midi.exists():
        print(f"[WhisperLyrics] WARNING: stem midi not found: {stem_midi}, skipping.")
        return

    # 1. Run faster-whisper inference (preserving segment boundaries)
    model = _get_whisper_model(model_size)
    print(f"[WhisperLyrics] Transcribing {vocals_wav.name} with word-level timestamps ...")
    segments, info = model.transcribe(
        str(vocals_wav), language=language, word_timestamps=True, beam_size=5
    )
    if info.language_probability < 0.65:
        print("Language probability is less than 0.65, skipping transcription.")
        return
    print(f"   Detected language: {info.language} (p={info.language_probability:.2f})")

    # 2. Read the stem MIDI via pretty_midi (builds an internal tempo map)
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(stem_midi))
    print(f"[WhisperLyrics] tempo changes: {len(pm.get_tempo_changes()[0])}")

    # 3. Build Lyric events: one per word, plus a newline at each segment boundary
    from pretty_midi import Lyric
    new_lyrics: list[Lyric] = []
    for segment in segments:
        if not segment.words:
            continue
        for word in segment.words:
            clean = word.word.strip().replace("\u3000", "")
            if clean:
                new_lyrics.append(Lyric(text=clean, time=word.start))
        # Insert a newline after each segment to mark sentence boundaries
        new_lyrics.append(Lyric(text="\n", time=segment.end))

    if not new_lyrics:
        print("[WhisperLyrics] No valid lyrics.")
        return

    # 4. Merge with any existing lyrics and sort by time
    pm.lyrics.extend(new_lyrics)
    pm.lyrics.sort(key=lambda l: l.time)

    # 5. Patch mido's encoding to support UTF-8 (required for CJK text), then write
    import mido.midifiles.meta as _mm
    _orig_es = _mm.encode_string
    _mm.encode_string = lambda s: list(bytearray(s.encode('utf-8')))
    try:
        pm.write(str(stem_midi))
    finally:
        _mm.encode_string = _orig_es

    print(f"[WhisperLyrics] ✓ Inserted {len(new_lyrics)} lyric events into {stem_midi.name}")
