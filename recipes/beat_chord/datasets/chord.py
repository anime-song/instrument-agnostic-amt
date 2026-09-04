from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch
from dlchordx import Tone
from dlchordx.const import CHORD_MAP
from torch.utils.data import Dataset

from .modulation import (
    ModulationAugmentConfig,
    WindowHarmonyContext,
    apply_modulation_to_roll,
    choose_modulation_candidate,
    enumerate_modulation_candidates,
    inject_synthetic_boundaries,
)
from instrument_agnostic_amt.beat_chord.midi_roll import (
    MidiFrameLoader,
    MidiFrameLoaderConfig,
)
from .augment import (
    MidiAugmentConfig,
    compute_stretch_window,
    parse_note_name_to_index,
    resize_roll_time,
    shift_roll_pitch,
    transform_segment_times,
    transform_splice_frame,
    transpose_chord_segments,
    transpose_key_segments,
    warp_roll_time,
)
from .common import AudioDurationCache

ROOT_TO_INDEX = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "N": 12,
}


def normalize_chord_quality(quality: str) -> str:
    """コード品質ラベルの空白表記ゆれを、quality.json と同じキーへそろえる。"""
    return "".join(str(quality).split())


def parse_key_to_relative_major(key_str: str) -> int:
    if key_str == "N":
        return 12

    is_minor = False
    root_part = key_str
    if key_str.endswith("m"):
        is_minor = True
        root_part = key_str[:-1]

    try:
        root_idx = int(Tone(root_part).get_interval())
    except Exception:
        root_idx = ROOT_TO_INDEX.get(root_part, 12)

    if root_idx >= 12:
        return 12
    if is_minor:
        return (root_idx + 3) % 12
    return root_idx


class MidiChordDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        midi_dir: str | Path,
        window_ms: int,
        sample_rate: int,
        hop_length: int,
        pitch_min: int,
        pitch_max: int,
        num_input_channels: int,
        seed: int = 42,
        modulation_config: ModulationAugmentConfig | None = None,
        augment_config: MidiAugmentConfig | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        self.chord_label_dir = self.root / "chord_label"
        self.key_label_dir = self.root / "key_label"
        self.midi_dir = Path(midi_dir)
        self.window_ms = int(window_ms)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.seed = int(seed)
        self.epoch = 0
        self.modulation_config = modulation_config or ModulationAugmentConfig()
        self.augment_config = augment_config or MidiAugmentConfig()

        if not self.audio_dir.exists() or not self.chord_label_dir.exists():
            raise FileNotFoundError(
                f"Chord dataset must contain audio/ and chord_label/: {self.root}"
            )
        if not self.key_label_dir.exists():
            raise FileNotFoundError(
                f"Chord dataset must contain key_label/: {self.root}"
            )
        if not self.midi_dir.exists():
            raise FileNotFoundError(f"MIDI directory not found: {self.midi_dir}")

        with open(self.root / "quality.json", "r", encoding="utf-8") as f:
            self.quality_map = json.load(f)
        with open(self.root / "quality_freq_count.json", "r", encoding="utf-8") as f:
            self.quality_freqs = json.load(f)

        self.dl_chord_map = {
            normalize_chord_quality(key): val for key, val in CHORD_MAP.items()
        }

        self.num_qualities = len(self.quality_map)
        self.n_quality_idx = 62
        self.num_root_chord_classes = 12 * (self.num_qualities - 1) + 1
        self.quality_to_idx = {
            normalize_chord_quality(v): int(k) for k, v in self.quality_map.items()
        }

        self.root_chord_counts = torch.zeros(self.num_root_chord_classes)
        for q_idx in range(self.num_qualities):
            count = float(self.quality_freqs[q_idx])
            if q_idx == self.n_quality_idx:
                self.root_chord_counts[-1] = count
            else:
                for root_index in range(12):
                    self.root_chord_counts[
                        root_index * (self.num_qualities - 1) + q_idx
                    ] = count / 12.0

        audio_files = {p.stem: p for p in self.audio_dir.glob("*.wav")}
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

        self.items = []
        duration_cache = AudioDurationCache(
            audio_dir=self.audio_dir,
            cache_path=self.root / "audio_duration_cache.json",
        )
        for label_path in sorted(self.chord_label_dir.glob("*.jsonl")):
            stem = label_path.stem
            audio_path = audio_files.get(stem)
            key_path = self.key_label_dir / f"{stem}.txt"
            midi_path = self.midi_dir / f"{stem}.mid"
            if audio_path is None or not key_path.exists() or not midi_path.exists():
                continue

            chords = []
            with open(label_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        chords.append(json.loads(line))

            keys = []
            with open(key_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(":")
                    if len(parts) == 3:
                        keys.append(
                            {
                                "start_time": float(parts[0]),
                                "end_time": float(parts[1]),
                                "key": parts[2],
                            }
                        )

            if not chords:
                continue

            self.items.append(
                {
                    "song_name": stem,
                    "audio_path": audio_path,
                    "duration_sec": duration_cache.get_duration_sec(audio_path),
                    "midi_path": midi_path,
                    "chords": chords,
                    "keys": keys,
                }
            )

        duration_cache.save_if_dirty()

        if not self.items:
            raise ValueError(
                f"No chord samples with aligned MIDI were found in {self.root}"
            )

        self.window_frames = int(round(self.window_ms * self.sample_rate / 1000.0))
        self.window_sec = self.window_frames / self.sample_rate
        self.model_frames = math.ceil(self.window_frames / self.hop_length)

    @staticmethod
    def _resolve_audio_duration_sec(audio_path: Path) -> float:
        from .common import resolve_audio_duration_sec

        return resolve_audio_duration_sec(audio_path)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _get_root_chord_index(self, root: str, quality_idx: int) -> int:
        if root == "N" or quality_idx == self.n_quality_idx:
            return self.num_root_chord_classes - 1
        root_index = parse_note_name_to_index(root)
        if root_index is None:
            return self.num_root_chord_classes - 1
        return root_index * (self.num_qualities - 1) + quality_idx

    def _create_chord_vector(self, root: str, quality: str, bass: str) -> torch.Tensor:
        vec = torch.zeros(25)
        if root == "N" or quality == "N":
            return vec

        try:
            root_index = int(Tone(root).get_interval())
            quality_key = normalize_chord_quality(quality)
            if quality_key in self.dl_chord_map:
                for interval in self.dl_chord_map[quality_key]:
                    vec[(root_index + int(interval)) % 12] = 1.0

            bass_index = int(Tone(bass).get_interval()) + 1 if bass != "N" else 0
            vec[12 + bass_index] = 1.0
        except Exception:
            pass

        return vec

    def _maybe_apply_modulation(
        self,
        *,
        item: dict[str, Any],
        midi_frames: torch.Tensor,
        window_start_sec: float,
        window_end_sec: float,
        rng: random.Random,
    ) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]], float | None]:
        if float(self.modulation_config.prob) <= 0.0:
            return midi_frames, item["chords"], item["keys"], None
        if rng.random() >= float(self.modulation_config.prob):
            return midi_frames, item["chords"], item["keys"], None

        context = WindowHarmonyContext(
            source_item=item,
            window_start_sec=float(window_start_sec),
            window_end_sec=float(window_end_sec),
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
        )
        candidates = enumerate_modulation_candidates(context, self.modulation_config)
        candidate = choose_modulation_candidate(candidates, rng)
        if candidate is None:
            return midi_frames, item["chords"], item["keys"], None

        modulated_frames, modulated_chords, modulated_keys = apply_modulation_to_roll(
            midi_frames,
            item["chords"],
            item["keys"],
            candidate,
        )
        return (
            modulated_frames,
            modulated_chords,
            modulated_keys,
            float(candidate.splice_time_sec),
        )

    def _render_chord_targets(
        self,
        *,
        chord_segments: list[dict[str, Any]],
        window_start_sec: float,
        window_end_sec: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chord_boundary = torch.zeros(self.model_frames)
        root_chord_targets = torch.full((self.model_frames,), -100, dtype=torch.long)
        bass_targets = torch.full((self.model_frames,), -100, dtype=torch.long)
        chord_pitch_targets = torch.zeros(self.model_frames, 25)

        for chord_segment in chord_segments:
            start_time = float(chord_segment["start_time"])
            end_time = float(chord_segment["end_time"])
            overlap_start = max(start_time, window_start_sec)
            overlap_end = min(end_time, window_end_sec)
            if overlap_end <= overlap_start:
                continue

            frame_start = int(
                math.floor(
                    (overlap_start - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                )
            )
            frame_end = int(
                math.ceil(
                    (overlap_end - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                )
            )
            frame_start = max(0, frame_start)
            frame_end = min(self.model_frames, frame_end)

            if frame_end > frame_start:
                quality_idx = self.quality_to_idx.get(
                    normalize_chord_quality(str(chord_segment["quality"])),
                    self.n_quality_idx,
                )
                root_chord_index = self._get_root_chord_index(
                    str(chord_segment["root"]),
                    quality_idx,
                )
                root_chord_targets[frame_start:frame_end] = root_chord_index
                bass_targets[frame_start:frame_end] = ROOT_TO_INDEX.get(
                    str(chord_segment["bass"]),
                    12,
                )
                chord_pitch_targets[frame_start:frame_end] = self._create_chord_vector(
                    str(chord_segment["root"]),
                    str(chord_segment["quality"]),
                    str(chord_segment["bass"]),
                )

            if window_start_sec <= start_time < window_end_sec:
                frame_index = int(
                    round(
                        (start_time - window_start_sec)
                        * self.sample_rate
                        / self.hop_length
                    )
                )
                if 0 <= frame_index < self.model_frames:
                    chord_boundary[frame_index] = 1.0

        return chord_boundary, root_chord_targets, bass_targets, chord_pitch_targets

    def _render_key_targets(
        self,
        *,
        key_segments: list[dict[str, Any]],
        window_start_sec: float,
        window_end_sec: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_boundary = torch.zeros(self.model_frames)
        key_targets = torch.full((self.model_frames,), -100, dtype=torch.long)

        for key_segment in key_segments:
            start_time = float(key_segment["start_time"])
            end_time = float(key_segment["end_time"])
            overlap_start = max(start_time, window_start_sec)
            overlap_end = min(end_time, window_end_sec)
            if overlap_end <= overlap_start:
                continue

            frame_start = int(
                math.floor(
                    (overlap_start - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                )
            )
            frame_end = int(
                math.ceil(
                    (overlap_end - window_start_sec)
                    * self.sample_rate
                    / self.hop_length
                )
            )
            frame_start = max(0, frame_start)
            frame_end = min(self.model_frames, frame_end)

            if frame_end > frame_start:
                key_targets[frame_start:frame_end] = parse_key_to_relative_major(
                    str(key_segment["key"])
                )

            if window_start_sec <= start_time < window_end_sec:
                frame_index = int(
                    round(
                        (start_time - window_start_sec)
                        * self.sample_rate
                        / self.hop_length
                    )
                )
                if 0 <= frame_index < self.model_frames:
                    key_boundary[frame_index] = 1.0

        return key_boundary, key_targets

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        rng = random.Random(self.seed + self.epoch * len(self.items) + idx)
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
        source_window_end_sec = source_window_start_sec + source_window_sec

        midi_frames = self.midi_loader.load_window(
            song_name=str(item["song_name"]),
            window_start_sec=float(source_window_start_sec),
            num_frames=int(source_model_frames),
            drop_drum=augment_params.drop_drum,
            drop_note_prob=augment_params.drop_note_prob,
            rng=rng,
        )
        (
            midi_frames,
            chord_segments,
            key_segments,
            splice_time_sec,
        ) = self._maybe_apply_modulation(
            item=item,
            midi_frames=midi_frames,
            window_start_sec=source_window_start_sec,
            window_end_sec=source_window_end_sec,
            rng=rng,
        )

        midi_frames = resize_roll_time(midi_frames, self.model_frames)
        midi_frames = warp_roll_time(midi_frames, rubato_time_map)
        chord_segments = transform_segment_times(
            chord_segments,
            source_window_start_sec=source_window_start_sec,
            stretch_factor=effective_stretch,
            target_window_sec=self.window_sec,
            rubato_time_map=rubato_time_map,
        )
        key_segments = transform_segment_times(
            key_segments,
            source_window_start_sec=source_window_start_sec,
            stretch_factor=effective_stretch,
            target_window_sec=self.window_sec,
            rubato_time_map=rubato_time_map,
        )

        midi_frames = shift_roll_pitch(midi_frames, augment_params.pitch_shift)
        chord_segments = transpose_chord_segments(
            chord_segments,
            augment_params.pitch_shift,
        )
        key_segments = transpose_key_segments(
            key_segments,
            augment_params.pitch_shift,
        )
        splice_frame = transform_splice_frame(
            splice_time_sec,
            source_window_start_sec=source_window_start_sec,
            stretch_factor=effective_stretch,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            max_frames=self.model_frames,
            rubato_time_map=rubato_time_map,
        )

        (
            chord_boundary,
            root_chord_targets,
            bass_targets,
            chord_pitch_targets,
        ) = self._render_chord_targets(
            chord_segments=chord_segments,
            window_start_sec=0.0,
            window_end_sec=self.window_sec,
        )
        key_boundary, key_targets = self._render_key_targets(
            key_segments=key_segments,
            window_start_sec=0.0,
            window_end_sec=self.window_sec,
        )
        inject_synthetic_boundaries(
            chord_boundary,
            splice_frame,
            force_boundary=self.modulation_config.force_boundary_for_chord,
        )
        inject_synthetic_boundaries(
            key_boundary,
            splice_frame,
            force_boundary=self.modulation_config.force_boundary_for_key,
        )

        return {
            "midi_frames": midi_frames,
            "chord_boundary": chord_boundary,
            "root_chord_targets": root_chord_targets,
            "bass_targets": bass_targets,
            "key_boundary": key_boundary,
            "key_targets": key_targets,
            "chord_pitch_targets": chord_pitch_targets,
            "song_name": item["song_name"],
            "augment_pitch_shift": int(augment_params.pitch_shift),
            "augment_time_stretch": float(effective_stretch),
            "augment_rubato_strength": float(augment_params.rubato_strength),
        }


def midi_chord_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    collated = {
        "midi_frames": torch.stack([item["midi_frames"] for item in batch]),
        "chord_boundary": torch.stack([item["chord_boundary"] for item in batch]),
        "root_chord_targets": torch.stack(
            [item["root_chord_targets"] for item in batch]
        ),
        "bass_targets": torch.stack([item["bass_targets"] for item in batch]),
        "key_boundary": torch.stack([item["key_boundary"] for item in batch]),
        "key_targets": torch.stack([item["key_targets"] for item in batch]),
        "chord_pitch_targets": torch.stack(
            [item["chord_pitch_targets"] for item in batch]
        ),
        "song_name": [item["song_name"] for item in batch],
        "augment_pitch_shift": torch.tensor(
            [item["augment_pitch_shift"] for item in batch],
            dtype=torch.long,
        ),
        "augment_time_stretch": torch.tensor(
            [item["augment_time_stretch"] for item in batch],
            dtype=torch.float32,
        ),
        "augment_rubato_strength": torch.tensor(
            [item["augment_rubato_strength"] for item in batch],
            dtype=torch.float32,
        ),
    }
    collated["chord_boundary_mask"] = torch.stack(
        [
            item.get(
                "chord_boundary_mask",
                torch.ones_like(item["chord_boundary"]),
            )
            for item in batch
        ]
    )
    collated["key_boundary_mask"] = torch.stack(
        [
            item.get(
                "key_boundary_mask",
                torch.ones_like(item["key_boundary"]),
            )
            for item in batch
        ]
    )
    collated["chord_pitch_mask"] = torch.stack(
        [
            item.get(
                "chord_pitch_mask",
                torch.ones_like(item["chord_boundary"]),
            )
            for item in batch
        ]
    )
    collated["supervision"] = [item.get("supervision", "chord_key") for item in batch]
    return collated
