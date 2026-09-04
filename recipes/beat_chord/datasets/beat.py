from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from instrument_agnostic_amt.beat_chord.midi_roll import (
    MidiFrameLoader,
    MidiFrameLoaderConfig,
)
from .augment import (
    MidiAugmentConfig,
    compute_stretch_window,
    resize_roll_time,
    shift_roll_pitch,
    transform_event_times,
    transform_intervals,
    transform_meter_intervals,
    warp_roll_time,
)
from .common import AudioDurationCache, compute_valid_audio_frames
from .meter_aware_crop import (
    MeterAwareCropConfig,
    MeterAwareCropSelection,
    choose_meter_aware_window_start,
)


class MidiBeatDataset(Dataset):
    @staticmethod
    def _assign_phase_targets(
        *,
        phase_targets: torch.Tensor,
        phase_mask: torch.Tensor,
        intervals: tuple[tuple[float, float], ...] | list[tuple[float, float]],
        window_start_sec: float,
        valid_model_frames: int,
        sample_rate: int,
        hop_length: int,
    ) -> None:
        if valid_model_frames <= 0:
            return

        frame_times = window_start_sec + (
            torch.arange(valid_model_frames, dtype=torch.float32)
            * float(hop_length)
            / float(sample_rate)
        )
        for start_sec, end_sec in intervals:
            interval_duration = float(end_sec - start_sec)
            if interval_duration <= 0.0:
                continue
            active = (frame_times >= float(start_sec)) & (frame_times < float(end_sec))
            if not torch.any(active):
                continue
            phase_targets[:valid_model_frames][active] = (
                frame_times[active] - float(start_sec)
            ) / interval_duration
            phase_mask[:valid_model_frames][active] = 1.0

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
        augment_config: MidiAugmentConfig | None = None,
        meter_aware_crop_config: MeterAwareCropConfig | None = None,
        extra_meter_classes: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.audio_dir = self.root / "audio"
        self.label_dir = self.root / "label"
        self.midi_dir = Path(midi_dir)
        self.window_ms = int(window_ms)
        self.sample_rate = int(sample_rate)
        self.hop_length = int(hop_length)
        self.seed = int(seed)
        self.epoch = 0
        self.augment_config = augment_config or MidiAugmentConfig()
        self.meter_aware_crop_config = meter_aware_crop_config or MeterAwareCropConfig()

        if self.window_ms <= 0:
            raise ValueError("window_ms must be positive")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if not self.audio_dir.exists() or not self.label_dir.exists():
            raise FileNotFoundError(
                f"Beat dataset must contain audio/ and label/: {self.root}"
            )
        if not self.midi_dir.exists():
            raise FileNotFoundError(f"MIDI directory not found: {self.midi_dir}")

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

        label_suffix = ".beat.beats.json"
        audio_by_stem = {
            path.stem: path for path in self.audio_dir.glob("*.wav") if path.is_file()
        }

        raw_items: list[dict[str, Any]] = []
        meter_keys: set[tuple[int, int]] = set()
        # checkpoint 由来の meter class も先に足し、pretrain 済み head と形状を合わせる。
        for meter_num, meter_den in extra_meter_classes or ():
            meter_num = int(meter_num)
            meter_den = int(meter_den)
            if meter_num <= 0 or meter_den <= 0:
                raise ValueError("extra_meter_classes entries must be positive")
            meter_keys.add((meter_num, meter_den))

        duration_cache = AudioDurationCache(
            audio_dir=self.audio_dir,
            cache_path=self.root / "audio_duration_cache.json",
        )
        for label_path in sorted(self.label_dir.glob(f"*{label_suffix}")):
            stem = label_path.name[: -len(label_suffix)]
            audio_path = audio_by_stem.get(stem)
            midi_path = self.midi_dir / f"{stem}.mid"
            if audio_path is None or not midi_path.exists():
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)

            measures: list[dict[str, float | int]] = []
            for raw_measure in label_data.get("measures", []):
                meter_num = int(raw_measure["time_sig_num"])
                meter_den = int(raw_measure["time_sig_den"])
                if meter_num <= 0 or meter_den <= 0:
                    continue
                raw_beat_times = raw_measure.get("beat_times")
                parsed_beat_times = (
                    tuple(float(t) for t in raw_beat_times)
                    if isinstance(raw_beat_times, (list, tuple))
                    else None
                )
                measures.append(
                    {
                        "downbeat_sec": float(raw_measure["downbeat_sec"]),
                        "meter_num": meter_num,
                        "meter_den": meter_den,
                        "tempo_bpm": float(raw_measure.get("tempo_bpm", 0.0)),
                        "beat_times": parsed_beat_times,
                    }
                )
                meter_keys.add((meter_num, meter_den))

            measures.sort(key=lambda measure: float(measure["downbeat_sec"]))
            if not measures:
                continue

            duration_sec = duration_cache.get_duration_sec(audio_path)
            raw_items.append(
                {
                    "song_name": stem,
                    "audio_path": audio_path,
                    "duration_sec": duration_sec,
                    "measures": measures,
                }
            )

        duration_cache.save_if_dirty()

        self.meter_classes = tuple(sorted(meter_keys))
        self.meter_to_index = {
            meter: index for index, meter in enumerate(self.meter_classes)
        }
        self.num_meter_classes = len(self.meter_classes)
        if self.num_meter_classes == 0 or not raw_items:
            raise ValueError(f"No usable beat+MIDI samples found in {self.root}")

        self.window_frames = int(round(self.window_ms * self.sample_rate / 1000.0))
        self.window_sec = float(self.window_frames) / float(self.sample_rate)
        self.model_frames = math.ceil(self.window_frames / self.hop_length)

        items: list[dict[str, Any]] = []
        meter_counts = torch.zeros(self.num_meter_classes, dtype=torch.float32)
        for raw_item in raw_items:
            beat_times: list[float] = []
            downbeat_times: list[float] = []
            meter_intervals: list[tuple[float, float, int]] = []
            beat_intervals: list[tuple[float, float]] = []
            measures = raw_item["measures"]
            for index, measure in enumerate(measures):
                start_sec = float(measure["downbeat_sec"])
                meter_num = int(measure["meter_num"])
                meter_den = int(measure["meter_den"])
                tempo_bpm = float(measure["tempo_bpm"])

                if index + 1 < len(measures):
                    end_sec = float(measures[index + 1]["downbeat_sec"])
                elif tempo_bpm > 0.0:
                    measure_sec = meter_num * (4.0 / meter_den) * 60.0 / tempo_bpm
                    end_sec = start_sec + measure_sec
                else:
                    end_sec = start_sec + 4.0

                if end_sec <= start_sec:
                    continue

                meter_index = self.meter_to_index[(meter_num, meter_den)]
                meter_intervals.append((start_sec, end_sec, meter_index))
                downbeat_times.append(start_sec)

                raw_beat_times = measure.get("beat_times")
                beat_starts: list[float] = []
                # 途中で終わる不完全小節の場合、beat の数は meter_num より少なくなるため、
                # len > 0 であればそのまま信頼して読み込む。
                if raw_beat_times is not None and len(raw_beat_times) > 0:
                    beat_starts = [float(t) for t in raw_beat_times]
                else:
                    # 古いデータセット用フォールバック
                    measure_duration = end_sec - start_sec
                    for beat_index in range(meter_num):
                        beat_starts.append(
                            start_sec + measure_duration * beat_index / meter_num
                        )

                for beat_start in beat_starts:
                    beat_times.append(beat_start)

                for beat_index, beat_start in enumerate(beat_starts):
                    beat_end = (
                        beat_starts[beat_index + 1]
                        if beat_index + 1 < len(beat_starts)
                        else end_sec
                    )
                    # ビート区間が次の小節に食い込まないようにクリップする
                    # (ただし、最後の小節で end_sec が不正確な場合はクリップしない)
                    if index + 1 < len(measures):
                        beat_end = min(beat_end, end_sec)

                    if beat_end > beat_start:
                        beat_intervals.append((beat_start, beat_end))

                start_frame = max(
                    0,
                    math.floor(start_sec * self.sample_rate / self.hop_length),
                )
                end_frame = math.ceil(end_sec * self.sample_rate / self.hop_length)
                if end_frame > start_frame:
                    meter_counts[meter_index] += float(end_frame - start_frame)

            if not meter_intervals:
                continue

            item = dict(raw_item)
            item.pop("measures")
            item.update(
                {
                    "beat_times": tuple(beat_times),
                    "downbeat_times": tuple(downbeat_times),
                    "meter_intervals": tuple(meter_intervals),
                    "beat_intervals": tuple(beat_intervals),
                }
            )
            items.append(item)

        if not items:
            raise ValueError(
                f"No beat samples with aligned MIDI were found in {self.root}"
            )

        self.items = items
        self.meter_class_counts = meter_counts
        self.meter_sampling_counts = tuple(
            float(count) for count in meter_counts.tolist()
        )

    @staticmethod
    def _resolve_audio_duration_sec(audio_path: Path) -> float:
        from .common import resolve_audio_duration_sec

        return resolve_audio_duration_sec(audio_path)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.items)

    def _choose_window_start(
        self,
        *,
        item: dict[str, Any],
        duration_sec: float,
        source_window_sec: float,
        rng: random.Random,
    ) -> MeterAwareCropSelection:
        return choose_meter_aware_window_start(
            meter_intervals=item["meter_intervals"],
            beat_times=item["beat_times"],
            downbeat_times=item["downbeat_times"],
            meter_classes=self.meter_classes,
            meter_class_counts=self.meter_sampling_counts,
            duration_sec=duration_sec,
            source_window_sec=source_window_sec,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            config=self.meter_aware_crop_config,
            rng=rng,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        rng = random.Random(self.seed + self.epoch * len(self.items) + idx)
        duration_sec = float(item["duration_sec"])
        augment_params = self.augment_config.sample(rng)
        rubato_time_map = augment_params.build_rubato_time_map(self.window_sec)
        (
            source_window_sec,
            source_window_frames,
            source_model_frames,
            effective_stretch,
        ) = compute_stretch_window(
            target_window_sec=self.window_sec,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            stretch_factor=augment_params.time_stretch,
        )
        crop_selection = self._choose_window_start(
            item=item,
            duration_sec=duration_sec,
            source_window_sec=source_window_sec,
            rng=rng,
        )
        source_window_start_sec = crop_selection.window_start_sec

        valid_source_frames = compute_valid_audio_frames(
            duration_sec=duration_sec,
            window_start_sec=source_window_start_sec,
            window_sec=source_window_sec,
            sample_rate=self.sample_rate,
            window_frames=source_window_frames,
        )
        valid_output_audio_frames = min(
            self.window_frames,
            int(round(valid_source_frames * effective_stretch)),
        )
        valid_output_sec = valid_output_audio_frames / float(self.sample_rate)
        valid_output_audio_frames = min(
            self.window_frames,
            int(round(rubato_time_map.map_time(valid_output_sec) * self.sample_rate)),
        )
        valid_model_frames = min(
            self.model_frames,
            math.ceil(valid_output_audio_frames / self.hop_length),
        )

        midi_frames = self.midi_loader.load_window(
            song_name=str(item["song_name"]),
            window_start_sec=float(source_window_start_sec),
            num_frames=int(source_model_frames),
            drop_drum=augment_params.drop_drum,
            drop_note_prob=augment_params.drop_note_prob,
            rng=rng,
        )
        midi_frames = resize_roll_time(midi_frames, self.model_frames)
        midi_frames = warp_roll_time(midi_frames, rubato_time_map)
        midi_frames = shift_roll_pitch(midi_frames, augment_params.pitch_shift)

        beat_times = transform_event_times(
            item["beat_times"],
            source_window_start_sec=source_window_start_sec,
            stretch_factor=effective_stretch,
            target_window_sec=self.window_sec,
            rubato_time_map=rubato_time_map,
        )
        downbeat_times = transform_event_times(
            item["downbeat_times"],
            source_window_start_sec=source_window_start_sec,
            stretch_factor=effective_stretch,
            target_window_sec=self.window_sec,
            rubato_time_map=rubato_time_map,
        )
        meter_intervals = transform_meter_intervals(
            item["meter_intervals"],
            source_window_start_sec=source_window_start_sec,
            stretch_factor=effective_stretch,
            target_window_sec=self.window_sec,
            rubato_time_map=rubato_time_map,
        )
        beat_intervals = transform_intervals(
            item["beat_intervals"],
            source_window_start_sec=source_window_start_sec,
            stretch_factor=effective_stretch,
            target_window_sec=self.window_sec,
            rubato_time_map=rubato_time_map,
        )

        beat_targets = torch.zeros(self.model_frames, dtype=torch.float32)
        downbeat_targets = torch.zeros(self.model_frames, dtype=torch.float32)
        meter_targets = torch.full((self.model_frames,), -100, dtype=torch.long)
        beat_mask = torch.zeros(self.model_frames, dtype=torch.float32)
        beat_phase_targets = torch.zeros(self.model_frames, dtype=torch.float32)
        bar_phase_targets = torch.zeros(self.model_frames, dtype=torch.float32)
        phase_mask = torch.zeros(self.model_frames, dtype=torch.float32)

        for start_sec, end_sec, meter_index in meter_intervals:
            start_frame = max(
                0,
                math.floor(start_sec * self.sample_rate / self.hop_length),
            )
            end_frame = min(
                valid_model_frames,
                math.ceil(end_sec * self.sample_rate / self.hop_length),
            )
            if end_frame > start_frame:
                meter_targets[start_frame:end_frame] = int(meter_index)
                beat_mask[start_frame:end_frame] = 1.0

        self._assign_phase_targets(
            phase_targets=beat_phase_targets,
            phase_mask=phase_mask,
            intervals=beat_intervals,
            window_start_sec=0.0,
            valid_model_frames=valid_model_frames,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
        )
        bar_phase_mask = torch.zeros(self.model_frames, dtype=torch.float32)
        self._assign_phase_targets(
            phase_targets=bar_phase_targets,
            phase_mask=bar_phase_mask,
            intervals=[
                (start_sec, end_sec) for start_sec, end_sec, _ in meter_intervals
            ],
            window_start_sec=0.0,
            valid_model_frames=valid_model_frames,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
        )
        phase_mask = phase_mask * bar_phase_mask

        for target, times in (
            (beat_targets, beat_times),
            (downbeat_targets, downbeat_times),
        ):
            for event_sec in times:
                frame_index = int(round(event_sec * self.sample_rate / self.hop_length))
                if 0 <= frame_index < valid_model_frames:
                    target[frame_index] = 1.0

        if valid_model_frames < self.model_frames:
            beat_mask[valid_model_frames:] = 0.0
            meter_targets[valid_model_frames:] = -100
            phase_mask[valid_model_frames:] = 0.0

        return {
            "midi_frames": midi_frames,
            "valid_audio_frames": valid_output_audio_frames,
            "song_name": item["song_name"],
            "window_start_sec": source_window_start_sec,
            "beat_targets": beat_targets,
            "downbeat_targets": downbeat_targets,
            "meter_targets": meter_targets,
            "beat_mask": beat_mask,
            "beat_phase_targets": beat_phase_targets,
            "bar_phase_targets": bar_phase_targets,
            "phase_mask": phase_mask,
            "augment_pitch_shift": int(augment_params.pitch_shift),
            "augment_time_stretch": float(effective_stretch),
            "augment_rubato_strength": float(augment_params.rubato_strength),
            "meter_aware_crop": bool(crop_selection.used_meter_aware),
            "meter_aware_meter_index": int(
                crop_selection.target_meter_index
                if crop_selection.target_meter_index is not None
                else -1
            ),
        }


def midi_beat_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "midi_frames": torch.stack([item["midi_frames"] for item in batch]),
        "valid_audio_frames": torch.tensor(
            [item["valid_audio_frames"] for item in batch],
            dtype=torch.long,
        ),
        "beat_targets": torch.stack([item["beat_targets"] for item in batch]),
        "downbeat_targets": torch.stack([item["downbeat_targets"] for item in batch]),
        "meter_targets": torch.stack([item["meter_targets"] for item in batch]),
        "beat_mask": torch.stack([item["beat_mask"] for item in batch]),
        "beat_phase_targets": torch.stack(
            [item["beat_phase_targets"] for item in batch]
        ),
        "bar_phase_targets": torch.stack([item["bar_phase_targets"] for item in batch]),
        "phase_mask": torch.stack([item["phase_mask"] for item in batch]),
        "song_name": [item["song_name"] for item in batch],
        "window_start_sec": torch.tensor(
            [item["window_start_sec"] for item in batch],
            dtype=torch.float32,
        ),
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
        "meter_aware_crop": torch.tensor(
            [item.get("meter_aware_crop", False) for item in batch],
            dtype=torch.bool,
        ),
        "meter_aware_meter_index": torch.tensor(
            [item.get("meter_aware_meter_index", -1) for item in batch],
            dtype=torch.long,
        ),
    }
