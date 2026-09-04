from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from instrument_agnostic_amt.beat_chord.midi_roll import (
    MidiFrameLoader,
    MidiFrameLoaderConfig,
    _build_tempo_seconds,
    _dedupe_tempo_events,
    _tick_to_seconds,
)
from .augment import (
    MidiAugmentConfig,
    compute_stretch_window,
    resize_roll_time,
    shift_roll_pitch,
    transform_segment_times,
    transpose_key_segments,
    warp_roll_time,
)
from .chord import parse_key_to_relative_major


def read_midi_key_segments(
    midi_path: str | Path,
) -> tuple[list[dict[str, Any]], float]:
    """Read key signatures using the same tempo conversion as MIDI rolls."""
    from mido import MidiFile

    midi = MidiFile(str(Path(midi_path)))
    raw_tempos: list[tuple[int, int]] = []
    keys_by_tick: dict[int, str] = {}
    duration_tick = 0

    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            if message.type == "set_tempo":
                raw_tempos.append((tick, int(message.tempo)))
            elif message.type == "key_signature":
                keys_by_tick[tick] = str(message.key)
        duration_tick = max(duration_tick, tick)

    if not keys_by_tick:
        return [], 0.0

    tempo_events = _dedupe_tempo_events(raw_tempos)
    start_ticks, start_seconds, tempos = _build_tempo_seconds(
        tempo_events=tempo_events,
        ticks_per_beat=int(midi.ticks_per_beat),
    )

    def to_seconds(tick: int) -> float:
        return _tick_to_seconds(
            tick=int(tick),
            ticks_per_beat=int(midi.ticks_per_beat),
            start_ticks=start_ticks,
            start_seconds=start_seconds,
            tempos=tempos,
        )

    duration_sec = to_seconds(duration_tick)
    key_events = [
        (to_seconds(tick), key_name) for tick, key_name in sorted(keys_by_tick.items())
    ]
    duration_sec = max(duration_sec, key_events[-1][0])

    segments: list[dict[str, Any]] = []
    for index, (start_time, key_name) in enumerate(key_events):
        end_time = (
            key_events[index + 1][0] if index + 1 < len(key_events) else duration_sec
        )
        if end_time <= start_time:
            continue
        segments.append(
            {
                "start_time": float(start_time),
                "end_time": float(end_time),
                "key": key_name,
            }
        )
    return segments, float(duration_sec)


class MidiKeyOnlyDataset(Dataset):
    """MIDI-note inputs supervised only by corrected key-signature events."""

    def __init__(
        self,
        root: str | Path,
        *,
        window_ms: int,
        sample_rate: int,
        hop_length: int,
        pitch_min: int,
        pitch_max: int,
        num_input_channels: int,
        seed: int = 42,
        augment_config: MidiAugmentConfig | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.midi_dir = self.root / "midis"
        self.window_ms = int(window_ms)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.seed = int(seed)
        self.epoch = 0
        self.augment_config = augment_config or MidiAugmentConfig()

        if not self.midi_dir.exists():
            raise FileNotFoundError(
                f"Key-only dataset must contain midis/: {self.root}"
            )

        self.window_frames = int(round(self.window_ms * self.sample_rate / 1000.0))
        self.window_sec = self.window_frames / self.sample_rate
        self.model_frames = math.ceil(self.window_frames / self.hop_length)
        self.midi_loader = MidiFrameLoader(
            MidiFrameLoaderConfig(
                midi_dir=self.midi_dir,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                pitch_min=int(pitch_min),
                pitch_max=int(pitch_max),
                num_channels=int(num_input_channels),
            )
        )

        midi_paths: list[Path] = []
        for pattern in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
            midi_paths.extend(self.midi_dir.glob(pattern))

        self.items: list[dict[str, Any]] = []
        for midi_path in sorted(set(midi_paths)):
            key_segments, duration_sec = read_midi_key_segments(midi_path)
            if not key_segments or duration_sec <= 0.0:
                continue
            self.items.append(
                {
                    "song_name": midi_path.stem,
                    "midi_path": midi_path,
                    "duration_sec": float(duration_sec),
                    "keys": key_segments,
                }
            )

        if not self.items:
            raise ValueError(
                f"No MIDI files with key_signature events were found in {self.midi_dir}"
            )

        # Long MIDIs contribute roughly one full-song equivalent of crops per epoch.
        self.samples: list[int] = []
        for item_index, item in enumerate(self.items):
            num_windows = max(
                1,
                int(math.ceil(float(item["duration_sec"]) / self.window_sec)),
            )
            self.samples.extend([item_index] * num_windows)

    @property
    def num_songs(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.samples)

    def _render_key_targets(
        self,
        *,
        key_segments: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_boundary = torch.zeros(self.model_frames)
        key_targets = torch.full((self.model_frames,), -100, dtype=torch.long)

        for segment in key_segments:
            start_time = float(segment["start_time"])
            end_time = float(segment["end_time"])
            overlap_start = max(0.0, start_time)
            overlap_end = min(self.window_sec, end_time)
            if overlap_end <= overlap_start:
                continue

            frame_start = max(
                0,
                int(math.floor(overlap_start * self.sample_rate / self.hop_length)),
            )
            frame_end = min(
                self.model_frames,
                int(math.ceil(overlap_end * self.sample_rate / self.hop_length)),
            )
            if frame_end > frame_start:
                key_targets[frame_start:frame_end] = parse_key_to_relative_major(
                    str(segment["key"])
                )

            if 0.0 <= start_time < self.window_sec:
                frame_index = int(
                    round(start_time * self.sample_rate / self.hop_length)
                )
                if 0 <= frame_index < self.model_frames:
                    key_boundary[frame_index] = 1.0

        return key_boundary, key_targets

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[self.samples[idx]]
        rng = random.Random(self.seed + self.epoch * len(self.samples) + idx)
        duration_sec = float(item["duration_sec"])
        augment_params = self.augment_config.sample(rng)
        rubato_time_map = augment_params.build_rubato_time_map(self.window_sec)
        (
            source_window_sec,
            _source_window_frames,
            source_model_frames,
            effective_stretch,
        ) = compute_stretch_window(
            target_window_sec=self.window_sec,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            stretch_factor=augment_params.time_stretch,
        )

        max_start = max(0.0, duration_sec - source_window_sec)
        source_window_start_sec = (
            float(rng.uniform(0.0, max_start)) if max_start > 0.0 else 0.0
        )
        midi_frames = self.midi_loader.load_window(
            song_name=str(item["song_name"]),
            window_start_sec=source_window_start_sec,
            num_frames=int(source_model_frames),
            drop_drum=augment_params.drop_drum,
            drop_note_prob=augment_params.drop_note_prob,
            rng=rng,
        )
        midi_frames = resize_roll_time(midi_frames, self.model_frames)
        midi_frames = warp_roll_time(midi_frames, rubato_time_map)

        key_segments = transform_segment_times(
            item["keys"],
            source_window_start_sec=source_window_start_sec,
            stretch_factor=effective_stretch,
            target_window_sec=self.window_sec,
            rubato_time_map=rubato_time_map,
        )
        midi_frames = shift_roll_pitch(midi_frames, augment_params.pitch_shift)
        key_segments = transpose_key_segments(
            key_segments,
            augment_params.pitch_shift,
        )
        key_boundary, key_targets = self._render_key_targets(key_segments=key_segments)
        key_supervision_mask = (key_targets != -100).to(torch.float32)

        # Predicted chord/beat metadata is ignored by MidiFrameLoader and masked.
        return {
            "midi_frames": midi_frames,
            "chord_boundary": torch.zeros(self.model_frames),
            "root_chord_targets": torch.full(
                (self.model_frames,), -100, dtype=torch.long
            ),
            "bass_targets": torch.full((self.model_frames,), -100, dtype=torch.long),
            "key_boundary": key_boundary,
            "key_targets": key_targets,
            "chord_pitch_targets": torch.zeros(self.model_frames, 25),
            "chord_boundary_mask": torch.zeros(self.model_frames),
            "key_boundary_mask": key_supervision_mask,
            "chord_pitch_mask": torch.zeros(self.model_frames),
            "song_name": item["song_name"],
            "supervision": "key_only",
            "augment_pitch_shift": int(augment_params.pitch_shift),
            "augment_time_stretch": float(effective_stretch),
            "augment_rubato_strength": float(augment_params.rubato_strength),
        }
