from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from instrument_agnostic_amt.taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_instrument_class_id,
)


def _normalize_instrument_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


_CLASS_NAME_TO_ID = {_normalize_instrument_name(class_name): idx for idx, class_name in enumerate(INSTRUMENT_CLASSES)}


class MidiReadError(RuntimeError):
    """MIDI ファイルを note event として読めなかったことを表す例外。"""


def _dedupe_tempo_events(
    raw_tempos: list[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    # 同じ tick に複数 tempo がある場合は、ファイル上で後から出たものを使う。
    by_tick: dict[int, int] = {0: 500000}
    for tick, tempo in raw_tempos:
        tick = int(tick)
        tempo = int(tempo)
        if tick >= 0 and tempo > 0:
            by_tick[tick] = tempo
    return tuple(sorted(by_tick.items()))


def _build_tempo_seconds(
    *,
    tempo_events: tuple[tuple[int, int], ...],
    ticks_per_beat: int,
) -> tuple[tuple[int, ...], tuple[float, ...], tuple[int, ...]]:
    # tempo event の開始秒を積分しておき、note tick を高速に秒へ変換する。
    if ticks_per_beat <= 0:
        raise ValueError("ticks_per_beat must be positive")
    start_ticks: list[int] = []
    start_seconds: list[float] = []
    tempos: list[int] = []
    current_seconds = 0.0
    previous_tick = int(tempo_events[0][0])
    previous_tempo = int(tempo_events[0][1])
    for tick, tempo in tempo_events:
        tick = int(tick)
        tempo = int(tempo)
        current_seconds += float(tick - previous_tick) * float(previous_tempo) / 1_000_000.0 / float(ticks_per_beat)
        start_ticks.append(tick)
        start_seconds.append(current_seconds)
        tempos.append(tempo)
        previous_tick = tick
        previous_tempo = tempo
    return tuple(start_ticks), tuple(start_seconds), tuple(tempos)


def _tick_to_seconds(
    *,
    tick: int,
    ticks_per_beat: int,
    start_ticks: tuple[int, ...],
    start_seconds: tuple[float, ...],
    tempos: tuple[int, ...],
) -> float:
    from bisect import bisect_right

    index = max(0, bisect_right(start_ticks, int(tick)) - 1)
    return float(start_seconds[index]) + (
        float(int(tick) - int(start_ticks[index])) * float(tempos[index]) / 1_000_000.0 / float(ticks_per_beat)
    )


def _load_midi_events_with_mido(
    midi_path_str: str,
) -> tuple[tuple[int, int, float, float], ...]:
    from mido import MidiFile

    midi = MidiFile(midi_path_str)
    raw_tempos: list[tuple[int, int]] = []
    note_ticks: list[tuple[int, int, int, int, bool]] = []

    # 1. mido で各トラックの note と tempo event を絶対 tick へ変換する。
    for track in midi.tracks:
        tick = 0
        program_by_channel = {channel: 0 for channel in range(16)}
        active_notes: dict[tuple[int, int], list[tuple[int, int, bool]]] = {}
        for message in track:
            tick += int(message.time)
            if message.type == "set_tempo":
                raw_tempos.append((tick, int(message.tempo)))
                continue
            if not hasattr(message, "channel"):
                continue

            channel = int(message.channel)
            if message.type == "program_change":
                program_by_channel[channel] = int(message.program)
                continue
            if message.type not in ("note_on", "note_off"):
                continue

            pitch = int(message.note)
            key = (channel, pitch)
            is_note_on = message.type == "note_on" and int(message.velocity) > 0
            if is_note_on:
                active_notes.setdefault(key, []).append((tick, int(program_by_channel.get(channel, 0)), channel == 9))
                continue

            starts = active_notes.get(key)
            if not starts:
                continue
            start_tick, program, is_drum = starts.pop()
            if tick > start_tick:
                note_ticks.append((program, pitch, start_tick, tick, is_drum))

    tempo_events = _dedupe_tempo_events(raw_tempos)
    start_ticks, start_seconds, tempos = _build_tempo_seconds(
        tempo_events=tempo_events,
        ticks_per_beat=int(midi.ticks_per_beat),
    )

    # 2. note の start/end tick を tempo map に沿って秒へ変換する。
    events: list[tuple[int, int, float, float]] = []
    for program, pitch, start_tick, end_tick, is_drum in note_ticks:
        class_id = int(get_instrument_class_id(program, is_drum=bool(is_drum)))
        note_start = _tick_to_seconds(
            tick=start_tick,
            ticks_per_beat=int(midi.ticks_per_beat),
            start_ticks=start_ticks,
            start_seconds=start_seconds,
            tempos=tempos,
        )
        note_end = _tick_to_seconds(
            tick=end_tick,
            ticks_per_beat=int(midi.ticks_per_beat),
            start_ticks=start_ticks,
            start_seconds=start_seconds,
            tempos=tempos,
        )
        if note_end > note_start:
            events.append((class_id, int(pitch), float(note_start), float(note_end)))
    return tuple(events)


def _load_midi_events_with_symusic(
    midi_path_str: str,
) -> tuple[tuple[int, int, float, float], ...]:
    from symusic import Score

    score = Score(midi_path_str)
    tempo_events = _dedupe_tempo_events([(int(tempo.time), int(tempo.mspq)) for tempo in score.tempos])
    start_ticks, start_seconds, tempos = _build_tempo_seconds(
        tempo_events=tempo_events,
        ticks_per_beat=int(score.tpq),
    )

    events: list[tuple[int, int, float, float]] = []
    for track in score.tracks:
        class_id = _CLASS_NAME_TO_ID.get(_normalize_instrument_name(str(track.name)))
        if class_id is None:
            class_id = int(
                get_instrument_class_id(
                    int(track.program),
                    is_drum=bool(track.is_drum),
                )
            )
        for note in track.notes:
            start_tick = int(note.start)
            end_tick = int(note.end)
            if end_tick <= start_tick:
                continue
            note_start = _tick_to_seconds(
                tick=start_tick,
                ticks_per_beat=int(score.tpq),
                start_ticks=start_ticks,
                start_seconds=start_seconds,
                tempos=tempos,
            )
            note_end = _tick_to_seconds(
                tick=end_tick,
                ticks_per_beat=int(score.tpq),
                start_ticks=start_ticks,
                start_seconds=start_seconds,
                tempos=tempos,
            )
            if note_end > note_start:
                events.append((int(class_id), int(note.pitch), float(note_start), float(note_end)))
    return tuple(events)


def _load_midi_events_with_pretty_midi(
    midi_path_str: str,
) -> tuple[tuple[int, int, float, float], ...]:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI(midi_path_str)

    events: list[tuple[int, int, float, float]] = []
    for instrument in midi.instruments:
        class_id = _CLASS_NAME_TO_ID.get(_normalize_instrument_name(instrument.name))
        if class_id is None:
            class_id = int(
                get_instrument_class_id(
                    int(instrument.program),
                    is_drum=bool(instrument.is_drum),
                )
            )
        for note in instrument.notes:
            if note.end <= note.start:
                continue
            events.append(
                (
                    int(class_id),
                    int(note.pitch),
                    float(note.start),
                    float(note.end),
                )
            )
    return tuple(events)


@dataclass(frozen=True)
class MidiFrameLoaderConfig:
    midi_dir: Path
    sample_rate: int
    hop_length: int
    pitch_min: int
    pitch_max: int
    num_channels: int

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.pitch_max < self.pitch_min:
            raise ValueError("pitch_max must be >= pitch_min")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")

    @property
    def num_pitch_bins(self) -> int:
        return int(self.pitch_max - self.pitch_min + 1)


@lru_cache(maxsize=4096)
def _load_midi_events(midi_path_str: str) -> tuple[tuple[int, int, float, float], ...]:
    errors: list[str] = []
    # tempo/拍子イベントが非ゼロ track にある MIDI では pretty_midi が警告を出し、
    # 拍位置がずれる可能性があるため、tick ベースで読める reader を優先する。
    for reader_name, reader in (
        ("symusic", _load_midi_events_with_symusic),
        ("mido", _load_midi_events_with_mido),
        ("pretty_midi", _load_midi_events_with_pretty_midi),
    ):
        try:
            return reader(midi_path_str)
        except ImportError as exc:
            errors.append(f"{reader_name}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            errors.append(f"{reader_name}: {type(exc).__name__}: {exc}")
    raise MidiReadError(f"Failed to read MIDI note events: {midi_path_str}; " + " | ".join(errors))


class MidiFrameLoader:
    def __init__(self, config: MidiFrameLoaderConfig) -> None:
        self.config = config

    def resolve_path(self, song_name: str) -> Path:
        for suffix in (".mid", ".midi", ".MID", ".MIDI"):
            midi_path = self.config.midi_dir / f"{song_name}{suffix}"
            if midi_path.exists():
                return midi_path
        raise FileNotFoundError(f"MIDI not found for {song_name}: {self.config.midi_dir}")

    def load_window(
        self,
        *,
        song_name: str,
        window_start_sec: float,
        num_frames: int,
        drop_drum: bool = False,
        drop_note_prob: float = 0.0,
        rng: random.Random | None = None,
    ) -> torch.Tensor:
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")

        midi_path = self.resolve_path(song_name)
        # ノートごとの書き込みはtorchのslice代入だと1音につき数回のkernel launchに
        # なる。1窓あたり数百音を曲全体で繰り返すため、numpy上で組み立ててから
        # 最後に一度だけTensor化する。
        roll_array = np.zeros(
            (
                self.config.num_channels,
                int(num_frames),
                self.config.num_pitch_bins,
            ),
            dtype=np.float32,
        )
        window_end_sec = float(window_start_sec) + (
            float(num_frames) * float(self.config.hop_length) / float(self.config.sample_rate)
        )

        class_channels = self.config.num_channels // 2
        drum_class_id = get_instrument_class_id(0, is_drum=True)
        for class_id, pitch, note_start, note_end in _load_midi_events(str(midi_path)):
            if drop_drum and class_id == drum_class_id:
                continue

            if drop_note_prob > 0.0 and rng is not None:
                if rng.random() < drop_note_prob:
                    continue

            if class_id < 0 or class_id >= class_channels:
                continue
            if pitch < self.config.pitch_min or pitch > self.config.pitch_max:
                continue
            if note_end <= window_start_sec or note_start >= window_end_sec:
                continue

            clipped_start = max(note_start, float(window_start_sec))
            clipped_end = min(note_end, window_end_sec)
            if clipped_end <= clipped_start:
                continue

            frame_start = max(
                0,
                int(
                    math.floor(
                        (clipped_start - float(window_start_sec))
                        * float(self.config.sample_rate)
                        / float(self.config.hop_length)
                    )
                ),
            )
            frame_end = min(
                int(num_frames),
                int(
                    math.ceil(
                        (clipped_end - float(window_start_sec))
                        * float(self.config.sample_rate)
                        / float(self.config.hop_length)
                    )
                ),
            )
            if frame_end <= frame_start:
                continue

            pitch_index = int(pitch - self.config.pitch_min)
            # AMT 由来 MIDI の velocity は信頼しにくいため、入力は発音有無だけにする。
            # 書き込む値は常に 1.0 で、既存値は 0.0 か 1.0 しか取らないため、
            # 以前の max(既存, 1.0) と代入は同じ結果になる。
            roll_array[class_id, frame_start:frame_end, pitch_index] = 1.0

            # Onset の書き込み (発音開始の瞬間のフレームのみ 1.0 にする)
            if window_start_sec <= note_start < window_end_sec:
                onset_frame = int(
                    round(
                        (note_start - float(window_start_sec))
                        * float(self.config.sample_rate)
                        / float(self.config.hop_length)
                    )
                )
                if 0 <= onset_frame < num_frames:
                    roll_array[class_id + class_channels, onset_frame, pitch_index] = 1.0

        return torch.from_numpy(roll_array)
