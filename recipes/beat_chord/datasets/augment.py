"""MIDI augmentation used by beat/chord recipes."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

SHARP_NOTE_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
NOTE_TO_INDEX = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
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
    "Cb": 11,
    "N": 12,
}

_RUBATO_BUMP_DERIVATIVE_MAX = 16.0 * math.sqrt(3.0) / 9.0


@dataclass(frozen=True)
class RubatoGesture:
    """A smooth, compensated timing gesture within one phrase."""

    start_sec: float
    end_sec: float
    amplitude_sec: float

    @property
    def duration_sec(self) -> float:
        return float(self.end_sec) - float(self.start_sec)


@dataclass(frozen=True)
class RubatoTimeMap:
    """Monotonic non-uniform time map whose window endpoints remain fixed."""

    duration_sec: float
    gestures: tuple[RubatoGesture, ...] = ()

    @classmethod
    def sample(
        cls,
        *,
        duration_sec: float,
        strength: float,
        period_sec: float,
        seed: int,
    ) -> RubatoTimeMap:
        duration_sec = float(duration_sec)
        strength = float(strength)
        period_sec = float(period_sec)
        if duration_sec <= 0.0 or strength <= 0.0:
            return cls(duration_sec=max(0.0, duration_sec))
        if not 0.0 < strength < 0.8:
            raise ValueError("rubato strength must be between 0 and 0.8")
        if period_sec <= 0.0:
            raise ValueError("rubato period_sec must be positive")

        rng = random.Random(int(seed))
        boundaries = [0.0]
        while boundaries[-1] < duration_sec:
            remaining_sec = duration_sec - boundaries[-1]
            if remaining_sec <= period_sec * 1.25:
                boundaries.append(duration_sec)
                break
            phrase_sec = period_sec * rng.uniform(0.75, 1.25)
            next_boundary = min(duration_sec, boundaries[-1] + phrase_sec)
            if duration_sec - next_boundary < period_sec * 0.5:
                next_boundary = duration_sec
            boundaries.append(next_boundary)

        gestures: list[RubatoGesture] = []
        for start_sec, end_sec in zip(boundaries[:-1], boundaries[1:]):
            phrase_duration_sec = float(end_sec) - float(start_sec)
            if phrase_duration_sec <= 0.0:
                continue
            direction = -1.0 if rng.random() < 0.5 else 1.0
            gesture_strength = strength * rng.uniform(0.6, 1.0)
            amplitude_sec = (
                direction
                * gesture_strength
                * phrase_duration_sec
                / _RUBATO_BUMP_DERIVATIVE_MAX
            )
            gestures.append(
                RubatoGesture(
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    amplitude_sec=float(amplitude_sec),
                )
            )
        return cls(duration_sec=duration_sec, gestures=tuple(gestures))

    @property
    def is_identity(self) -> bool:
        return not self.gestures

    def map_time(self, time_sec: float) -> float:
        """Map one local time while preserving values outside the window."""
        time_sec = float(time_sec)
        if not self.gestures or time_sec <= 0.0 or time_sec >= self.duration_sec:
            return time_sec

        for gesture in self.gestures:
            if gesture.start_sec <= time_sec <= gesture.end_sec:
                if gesture.duration_sec <= 0.0:
                    return time_sec
                position = (time_sec - gesture.start_sec) / gesture.duration_sec
                smooth_bump = 16.0 * position**2 * (1.0 - position) ** 2
                return time_sec + gesture.amplitude_sec * smooth_bump
        return time_sec


@dataclass(frozen=True)
class MidiAugmentParams:
    pitch_shift: int = 0
    time_stretch: float = 1.0
    rubato_strength: float = 0.0
    rubato_period_sec: float = 4.0
    rubato_seed: int = 0
    drop_drum: bool = False
    drop_note_prob: float = 0.0

    def build_rubato_time_map(self, target_window_sec: float) -> RubatoTimeMap:
        return RubatoTimeMap.sample(
            duration_sec=float(target_window_sec),
            strength=float(self.rubato_strength),
            period_sec=float(self.rubato_period_sec),
            seed=int(self.rubato_seed),
        )

    @property
    def is_identity(self) -> bool:
        return (
            self.pitch_shift == 0
            and abs(self.time_stretch - 1.0) < 1e-9
            and self.rubato_strength <= 0.0
            and not self.drop_drum
            and self.drop_note_prob <= 0.0
        )


@dataclass(frozen=True)
class MidiAugmentConfig:
    pitch_shift_min: int = 0
    pitch_shift_max: int = 0
    time_stretch_min: float = 1.0
    time_stretch_max: float = 1.0
    rubato_prob: float = 0.0
    rubato_strength: float = 0.12
    rubato_period_sec: float = 4.0
    drop_drum_prob: float = 0.0
    drop_note_prob: float = 0.0

    def __post_init__(self) -> None:
        if int(self.pitch_shift_min) > int(self.pitch_shift_max):
            raise ValueError("pitch_shift_min must be <= pitch_shift_max")
        if float(self.time_stretch_min) <= 0.0:
            raise ValueError("time_stretch_min must be positive")
        if float(self.time_stretch_max) <= 0.0:
            raise ValueError("time_stretch_max must be positive")
        if float(self.time_stretch_min) > float(self.time_stretch_max):
            raise ValueError("time_stretch_min must be <= time_stretch_max")
        if not 0.0 <= float(self.rubato_prob) <= 1.0:
            raise ValueError("rubato_prob must be between 0 and 1")
        if not 0.0 < float(self.rubato_strength) < 0.8:
            raise ValueError("rubato_strength must be between 0 and 0.8")
        if float(self.rubato_period_sec) <= 0.0:
            raise ValueError("rubato_period_sec must be positive")

    def sample(self, rng: Any) -> MidiAugmentParams:
        pitch_shift_min = int(self.pitch_shift_min)
        pitch_shift_max = int(self.pitch_shift_max)
        if pitch_shift_min == pitch_shift_max:
            pitch_shift = pitch_shift_min
        else:
            pitch_shift = int(rng.randint(pitch_shift_min, pitch_shift_max))

        time_stretch_min = float(self.time_stretch_min)
        time_stretch_max = float(self.time_stretch_max)
        if abs(time_stretch_max - time_stretch_min) < 1e-9:
            time_stretch = time_stretch_min
        else:
            time_stretch = float(rng.uniform(time_stretch_min, time_stretch_max))

        apply_rubato = float(self.rubato_prob) > 0.0 and rng.random() < float(
            self.rubato_prob
        )
        rubato_strength = float(self.rubato_strength) if apply_rubato else 0.0
        rubato_seed = int(rng.randint(0, 2**31 - 1)) if apply_rubato else 0

        drop_drum = False
        if float(self.drop_drum_prob) > 0.0:
            drop_drum = rng.random() < float(self.drop_drum_prob)

        return MidiAugmentParams(
            pitch_shift=pitch_shift,
            time_stretch=time_stretch,
            rubato_strength=rubato_strength,
            rubato_period_sec=float(self.rubato_period_sec),
            rubato_seed=rubato_seed,
            drop_drum=drop_drum,
            drop_note_prob=float(self.drop_note_prob),
        )


def compute_stretch_window(
    *,
    target_window_sec: float,
    sample_rate: int,
    hop_length: int,
    stretch_factor: float,
) -> tuple[float, int, int, float]:
    if stretch_factor <= 0.0:
        raise ValueError("stretch_factor must be positive")

    source_window_sec = float(target_window_sec) / float(stretch_factor)
    source_window_frames = max(1, int(round(source_window_sec * float(sample_rate))))
    source_model_frames = max(
        1, int(math.ceil(source_window_frames / float(hop_length)))
    )
    quantized_source_window_sec = (
        float(source_model_frames) * float(hop_length) / float(sample_rate)
    )
    effective_stretch = float(target_window_sec) / quantized_source_window_sec
    return (
        quantized_source_window_sec,
        int(source_window_frames),
        int(source_model_frames),
        float(effective_stretch),
    )


def shift_roll_pitch(roll: torch.Tensor, semitones: int) -> torch.Tensor:
    if roll.dim() != 3:
        raise ValueError("roll must have shape [C, T, P]")
    semitones = int(semitones)
    if semitones == 0:
        return roll

    shifted = torch.zeros_like(roll)
    pitch_bins = int(roll.shape[-1])
    if abs(semitones) >= pitch_bins:
        return shifted
    if semitones > 0:
        shifted[:, :, semitones:] = roll[:, :, : pitch_bins - semitones]
    else:
        shifted[:, :, : pitch_bins + semitones] = roll[:, :, -semitones:]
    return shifted


def resize_roll_time(roll: torch.Tensor, target_frames: int) -> torch.Tensor:
    """MIDI ロールの時間軸を、sustain/onset の意味を保ったまま伸縮する。"""
    if roll.dim() != 3:
        raise ValueError("roll must have shape [C, T, P]")
    target_frames = int(target_frames)
    if target_frames <= 0:
        raise ValueError("target_frames must be positive")
    if roll.shape[1] == target_frames:
        return roll.contiguous()

    channel_count, source_frames, pitch_bins = roll.shape
    if channel_count % 2 != 0:
        raise ValueError(
            "roll channel count must be even: sustain channels + onset channels"
        )

    resized_roll = roll.new_zeros((channel_count, target_frames, pitch_bins))

    class_channel_count = channel_count // 2
    stretch_ratio = float(target_frames) / float(source_frames)

    # 1. 持続音 (Sustain) の処理:
    #    各ターゲットフレームに対応する元フレーム範囲をまとめて作り、区間 max を一括計算する。
    target_positions = torch.arange(
        target_frames,
        dtype=torch.float64,
        device=roll.device,
    )
    source_starts = torch.floor(target_positions / stretch_ratio).long()
    source_ends = torch.ceil((target_positions + 1.0) / stretch_ratio).long()
    source_starts = torch.clamp(source_starts, 0, source_frames)
    source_ends = torch.clamp(source_ends, 0, source_frames)

    window_widths = source_ends - source_starts
    max_window_width = int(window_widths.max().item())
    offsets = torch.arange(max_window_width, device=roll.device)
    source_indices = source_starts[:, None] + offsets[None, :]
    valid_source_mask = source_indices < source_ends[:, None]
    source_indices = torch.clamp(source_indices, 0, source_frames - 1)

    sustain_windows = roll[:class_channel_count, source_indices, :]
    sustain_windows = sustain_windows.masked_fill(
        ~valid_source_mask.view(1, target_frames, max_window_width, 1),
        0.0,
    )
    resized_roll[:class_channel_count] = sustain_windows.amax(dim=2)

    # 2. 開始音 (Onset) の処理: 元の発音開始フレームをターゲットフレームに丸めて配置する
    onset_indices = (roll[class_channel_count:] > 0.5).nonzero(as_tuple=False)
    if onset_indices.numel() > 0:
        channel_indices = onset_indices[:, 0] + class_channel_count
        source_frame_indices = onset_indices[:, 1]
        pitch_indices = onset_indices[:, 2]

        # ターゲット時間軸上のインデックスに丸めてマッピング
        target_frame_indices = (
            (source_frame_indices.float() * stretch_ratio).round().long()
        )
        target_frame_indices = torch.clamp(target_frame_indices, 0, target_frames - 1)

        resized_roll[channel_indices, target_frame_indices, pitch_indices] = 1.0

    return resized_roll


def warp_roll_time(
    roll: torch.Tensor,
    rubato_time_map: RubatoTimeMap,
) -> torch.Tensor:
    """Apply a non-uniform time map while preserving sustain and onset semantics."""
    if roll.dim() != 3:
        raise ValueError("roll must have shape [C, T, P]")
    if rubato_time_map.is_identity:
        return roll.contiguous()

    channel_count, frame_count, pitch_bins = roll.shape
    if channel_count % 2 != 0:
        raise ValueError(
            "roll channel count must be even: sustain channels + onset channels"
        )
    if frame_count <= 0 or rubato_time_map.duration_sec <= 0.0:
        return roll.contiguous()

    class_channel_count = channel_count // 2
    warped_roll = roll.new_zeros((channel_count, frame_count, pitch_bins))

    source_boundary_times = torch.linspace(
        0.0,
        float(rubato_time_map.duration_sec),
        frame_count + 1,
        dtype=torch.float64,
        device=roll.device,
    )
    mapped_boundary_frames = torch.tensor(
        [
            rubato_time_map.map_time(float(time_sec))
            * frame_count
            / float(rubato_time_map.duration_sec)
            for time_sec in source_boundary_times.detach().cpu().tolist()
        ],
        dtype=torch.float64,
        device=roll.device,
    )
    target_starts = torch.arange(
        frame_count,
        dtype=torch.float64,
        device=roll.device,
    )
    source_starts = (
        torch.searchsorted(mapped_boundary_frames, target_starts, right=True) - 1
    )
    source_ends = torch.searchsorted(
        mapped_boundary_frames,
        target_starts + 1.0,
        right=False,
    )
    source_starts = torch.clamp(source_starts, 0, frame_count - 1)
    source_ends = torch.clamp(source_ends, 1, frame_count)
    source_ends = torch.maximum(source_ends, source_starts + 1)

    window_widths = source_ends - source_starts
    max_window_width = int(window_widths.max().item())
    offsets = torch.arange(max_window_width, device=roll.device)
    source_indices = source_starts[:, None] + offsets[None, :]
    valid_source_mask = source_indices < source_ends[:, None]
    source_indices = torch.clamp(source_indices, 0, frame_count - 1)

    sustain_windows = roll[:class_channel_count, source_indices, :]
    sustain_windows = sustain_windows.masked_fill(
        ~valid_source_mask.view(1, frame_count, max_window_width, 1),
        0.0,
    )
    warped_roll[:class_channel_count] = sustain_windows.amax(dim=2)

    onset_indices = (roll[class_channel_count:] > 0.5).nonzero(as_tuple=False)
    if onset_indices.numel() > 0:
        channel_indices = onset_indices[:, 0] + class_channel_count
        source_frame_indices = onset_indices[:, 1]
        pitch_indices = onset_indices[:, 2]
        source_times = (
            source_frame_indices.detach().cpu().to(torch.float64)
            * float(rubato_time_map.duration_sec)
            / float(frame_count)
        )
        target_frame_indices = torch.tensor(
            [
                round(
                    rubato_time_map.map_time(float(time_sec))
                    * frame_count
                    / float(rubato_time_map.duration_sec)
                )
                for time_sec in source_times.tolist()
            ],
            dtype=torch.long,
            device=roll.device,
        )
        target_frame_indices = torch.clamp(
            target_frame_indices,
            0,
            frame_count - 1,
        )
        warped_roll[channel_indices, target_frame_indices, pitch_indices] = 1.0

    return warped_roll


def _transform_local_time(
    time_sec: float,
    *,
    source_window_start_sec: float,
    stretch_factor: float,
    rubato_time_map: RubatoTimeMap | None = None,
) -> float:
    local_time = (float(time_sec) - float(source_window_start_sec)) * float(
        stretch_factor
    )
    if rubato_time_map is not None:
        local_time = rubato_time_map.map_time(local_time)
    return local_time


def transform_event_times(
    times: Sequence[float],
    *,
    source_window_start_sec: float,
    stretch_factor: float,
    target_window_sec: float,
    rubato_time_map: RubatoTimeMap | None = None,
) -> list[float]:
    transformed: list[float] = []
    for time_sec in times:
        local_time = _transform_local_time(
            float(time_sec),
            source_window_start_sec=source_window_start_sec,
            stretch_factor=stretch_factor,
            rubato_time_map=rubato_time_map,
        )
        if 0.0 <= local_time < float(target_window_sec):
            transformed.append(local_time)
    return transformed


def transform_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    source_window_start_sec: float,
    stretch_factor: float,
    target_window_sec: float,
    rubato_time_map: RubatoTimeMap | None = None,
) -> list[tuple[float, float]]:
    transformed: list[tuple[float, float]] = []
    for start_sec, end_sec in intervals:
        new_start = _transform_local_time(
            float(start_sec),
            source_window_start_sec=source_window_start_sec,
            stretch_factor=stretch_factor,
            rubato_time_map=rubato_time_map,
        )
        new_end = _transform_local_time(
            float(end_sec),
            source_window_start_sec=source_window_start_sec,
            stretch_factor=stretch_factor,
            rubato_time_map=rubato_time_map,
        )
        clipped_start = max(0.0, new_start)
        clipped_end = min(float(target_window_sec), new_end)
        if clipped_end > clipped_start:
            transformed.append((clipped_start, clipped_end))
    return transformed


def transform_meter_intervals(
    intervals: Sequence[tuple[float, float, int]],
    *,
    source_window_start_sec: float,
    stretch_factor: float,
    target_window_sec: float,
    rubato_time_map: RubatoTimeMap | None = None,
) -> list[tuple[float, float, int]]:
    transformed: list[tuple[float, float, int]] = []
    for start_sec, end_sec, meter_index in intervals:
        new_start = _transform_local_time(
            float(start_sec),
            source_window_start_sec=source_window_start_sec,
            stretch_factor=stretch_factor,
            rubato_time_map=rubato_time_map,
        )
        new_end = _transform_local_time(
            float(end_sec),
            source_window_start_sec=source_window_start_sec,
            stretch_factor=stretch_factor,
            rubato_time_map=rubato_time_map,
        )
        clipped_start = max(0.0, new_start)
        clipped_end = min(float(target_window_sec), new_end)
        if clipped_end > clipped_start:
            transformed.append((clipped_start, clipped_end, int(meter_index)))
    return transformed


def parse_note_name_to_index(note_name: str) -> int | None:
    if not note_name or note_name == "N":
        return None

    base_note_pitches = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }

    first_character = note_name[0].upper()
    if first_character not in base_note_pitches:
        return None

    pitch_index = base_note_pitches[first_character]

    accidentals = note_name[1:]
    for character in accidentals:
        if character == "#":
            pitch_index += 1
        elif character == "b":
            pitch_index -= 1
        elif character == "x" or character == "X":
            pitch_index += 2

    return pitch_index % 12


def transpose_note_name(note_name: str, semitones: int) -> str:
    if note_name == "N":
        return "N"
    note_index = parse_note_name_to_index(note_name)
    if note_index is None:
        return note_name
    return SHARP_NOTE_NAMES[(note_index + int(semitones)) % 12]


def transpose_key_name(key_name: str, semitones: int) -> str:
    if key_name == "N":
        return "N"
    is_minor = str(key_name).endswith("m")
    root_name = str(key_name)[:-1] if is_minor else str(key_name)
    shifted_root = transpose_note_name(root_name, semitones)
    if shifted_root == "N":
        return "N"
    return f"{shifted_root}m" if is_minor else shifted_root


def transpose_chord_segments(
    segments: Sequence[dict[str, Any]],
    semitones: int,
) -> list[dict[str, Any]]:
    semitones = int(semitones)
    if semitones == 0:
        return [dict(segment) for segment in segments]

    transposed: list[dict[str, Any]] = []
    for segment in segments:
        new_segment = dict(segment)
        if "root" in new_segment:
            new_segment["root"] = transpose_note_name(
                str(new_segment["root"]), semitones
            )
        if "bass" in new_segment:
            new_segment["bass"] = transpose_note_name(
                str(new_segment["bass"]), semitones
            )
        transposed.append(new_segment)
    return transposed


def transpose_key_segments(
    segments: Sequence[dict[str, Any]],
    semitones: int,
) -> list[dict[str, Any]]:
    semitones = int(semitones)
    if semitones == 0:
        return [dict(segment) for segment in segments]

    transposed: list[dict[str, Any]] = []
    for segment in segments:
        new_segment = dict(segment)
        if "key" in new_segment:
            new_segment["key"] = transpose_key_name(str(new_segment["key"]), semitones)
        transposed.append(new_segment)
    return transposed


def transform_segment_times(
    segments: Sequence[dict[str, Any]],
    *,
    source_window_start_sec: float,
    stretch_factor: float,
    target_window_sec: float,
    rubato_time_map: RubatoTimeMap | None = None,
) -> list[dict[str, Any]]:
    transformed: list[dict[str, Any]] = []
    for segment in segments:
        start_time = _transform_local_time(
            float(segment["start_time"]),
            source_window_start_sec=source_window_start_sec,
            stretch_factor=stretch_factor,
            rubato_time_map=rubato_time_map,
        )
        end_time = _transform_local_time(
            float(segment["end_time"]),
            source_window_start_sec=source_window_start_sec,
            stretch_factor=stretch_factor,
            rubato_time_map=rubato_time_map,
        )
        if end_time <= 0.0 or start_time >= float(target_window_sec):
            continue
        new_segment = dict(segment)
        new_segment["start_time"] = start_time
        new_segment["end_time"] = end_time
        transformed.append(new_segment)
    return transformed


def transform_splice_frame(
    splice_time_sec: float | None,
    *,
    source_window_start_sec: float,
    stretch_factor: float,
    sample_rate: int,
    hop_length: int,
    max_frames: int,
    rubato_time_map: RubatoTimeMap | None = None,
) -> int | None:
    if splice_time_sec is None:
        return None
    local_time = _transform_local_time(
        float(splice_time_sec),
        source_window_start_sec=source_window_start_sec,
        stretch_factor=stretch_factor,
        rubato_time_map=rubato_time_map,
    )
    frame_index = int(round(local_time * float(sample_rate) / float(hop_length)))
    if max_frames <= 0:
        return None
    return max(0, min(frame_index, int(max_frames) - 1))
