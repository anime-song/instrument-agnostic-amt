from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from instrument_agnostic_amt.amt.data.audio import inspect_audio


@dataclass(frozen=True)
class AudioDurationEntry:
    """1つの音声ファイルに対する duration キャッシュ情報。"""

    sample_rate: int
    num_frames: int
    size_bytes: int
    mtime_ns: int

    @property
    def duration_sec(self) -> float:
        return float(self.num_frames) / float(self.sample_rate)

    def matches(self, audio_path: Path) -> bool:
        stat = audio_path.stat()
        return self.size_bytes == int(stat.st_size) and self.mtime_ns == int(
            stat.st_mtime_ns
        )

    def to_json(self) -> dict[str, int]:
        return {
            "sample_rate": int(self.sample_rate),
            "num_frames": int(self.num_frames),
            "size_bytes": int(self.size_bytes),
            "mtime_ns": int(self.mtime_ns),
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "AudioDurationEntry":
        return cls(
            sample_rate=int(data["sample_rate"]),
            num_frames=int(data["num_frames"]),
            size_bytes=int(data["size_bytes"]),
            mtime_ns=int(data["mtime_ns"]),
        )


class AudioDurationCache:
    """音声メタデータを dataset root に保存して初期化時間を短縮する。"""

    def __init__(self, *, audio_dir: Path, cache_path: Path) -> None:
        self.audio_dir = Path(audio_dir)
        self.cache_path = Path(cache_path)
        self._entries: dict[str, AudioDurationEntry] = {}
        self._dirty = False

        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for key, value in raw.get("files", {}).items():
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid audio duration cache entry: {key}")
                self._entries[str(key)] = AudioDurationEntry.from_json(value)

    def get_duration_sec(self, audio_path: Path) -> float:
        audio_path = Path(audio_path)
        key = str(audio_path.relative_to(self.audio_dir))
        entry = self._entries.get(key)
        if entry is not None and entry.matches(audio_path):
            return entry.duration_sec

        entry = read_audio_duration_entry(audio_path)
        self._entries[key] = entry
        self._dirty = True
        return entry.duration_sec

    def save_if_dirty(self) -> None:
        if not self._dirty:
            return

        # 1. 中途半端な cache を残さないよう、一時ファイルへ書いてから置き換える。
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        payload = {
            "version": 1,
            "files": {
                key: entry.to_json() for key, entry in sorted(self._entries.items())
            },
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.cache_path)
        self._dirty = False


def read_audio_duration_entry(audio_path: Path) -> AudioDurationEntry:
    audio_path = Path(audio_path)
    info = inspect_audio(audio_path)
    sample_rate = int(info.sample_rate)
    num_frames = int(info.num_frames)
    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate for {audio_path}: {sample_rate}")
    if num_frames < 0:
        raise ValueError(f"Invalid frame count for {audio_path}: {num_frames}")

    stat = audio_path.stat()
    return AudioDurationEntry(
        sample_rate=sample_rate,
        num_frames=num_frames,
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
    )


def resolve_audio_duration_sec(audio_path: Path) -> float:
    return read_audio_duration_entry(audio_path).duration_sec


def compute_valid_audio_frames(
    *,
    duration_sec: float,
    window_start_sec: float,
    window_sec: float,
    sample_rate: int,
    window_frames: int,
) -> int:
    valid_end_sec = min(
        float(duration_sec), float(window_start_sec) + float(window_sec)
    )
    valid_duration_sec = max(0.0, valid_end_sec - float(window_start_sec))
    return min(int(window_frames), int(round(valid_duration_sec * float(sample_rate))))


def splice_roll_at_frame(
    source_roll: torch.Tensor,
    shifted_roll: torch.Tensor,
    *,
    splice_frame: int,
) -> torch.Tensor:
    """MIDI ロールを splice フレームで hard switch する。"""
    if source_roll.shape != shifted_roll.shape:
        raise ValueError("source_roll and shifted_roll must share the same shape")
    splice_frame = int(max(0, min(splice_frame, source_roll.shape[1])))

    # 1. 境界より前は元ロール、境界以降は転調先ロールをそのまま使う。
    #    MIDI ロールは binary 入力なので、音声のような crossfade は行わない。
    output = source_roll.clone()
    output[:, splice_frame:, :] = shifted_roll[:, splice_frame:, :]
    return output
