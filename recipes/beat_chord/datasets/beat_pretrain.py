from __future__ import annotations

import json
import logging
import math
import random
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from instrument_agnostic_amt.beat_chord.midi_roll import (
    MidiFrameLoader,
    MidiFrameLoaderConfig,
    MidiReadError,
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
from .beat import MidiBeatDataset, midi_beat_collate_fn
from .common import compute_valid_audio_frames
from .meter_aware_crop import (
    MeterAwareCropConfig,
    MeterAwareCropSelection,
    choose_meter_aware_window_start,
)

logger = logging.getLogger(__name__)

MIDI_METADATA_READER_ID = "symusic_mido_v1"


@dataclass(frozen=True)
class TempoChange:
    """MIDI tick 上の tempo 変更点。"""

    tick: int
    tempo: int


@dataclass(frozen=True)
class TimeSignatureChange:
    """MIDI tick 上の拍子変更点。"""

    tick: int
    numerator: int
    denominator: int


@dataclass(frozen=True)
class TickSecondConverter:
    """tempo map を積分して MIDI tick を秒へ変換する。"""

    ticks_per_beat: int
    start_ticks: tuple[int, ...]
    start_seconds: tuple[float, ...]
    tempos: tuple[int, ...]

    @classmethod
    def from_tempo_changes(
        cls,
        *,
        ticks_per_beat: int,
        tempo_changes: Sequence[TempoChange],
    ) -> "TickSecondConverter":
        if ticks_per_beat <= 0:
            raise ValueError("ticks_per_beat must be positive")
        if not tempo_changes:
            raise ValueError("tempo_changes must not be empty")

        start_ticks: list[int] = []
        start_seconds: list[float] = []
        tempos: list[int] = []
        current_seconds = 0.0
        previous_tick = int(tempo_changes[0].tick)
        previous_tempo = int(tempo_changes[0].tempo)

        for change in tempo_changes:
            tick = int(change.tick)
            tempo = int(change.tempo)
            if tick < previous_tick:
                raise ValueError("tempo changes must be sorted")
            current_seconds += (
                float(tick - previous_tick)
                * float(previous_tempo)
                / 1_000_000.0
                / float(ticks_per_beat)
            )
            start_ticks.append(tick)
            start_seconds.append(current_seconds)
            tempos.append(tempo)
            previous_tick = tick
            previous_tempo = tempo

        return cls(
            ticks_per_beat=int(ticks_per_beat),
            start_ticks=tuple(start_ticks),
            start_seconds=tuple(start_seconds),
            tempos=tuple(tempos),
        )

    def tick_to_seconds(self, tick: int | float | Fraction) -> float:
        tick_value = float(tick)
        index = max(0, bisect_right(self.start_ticks, tick_value) - 1)
        return float(self.start_seconds[index]) + (
            tick_value - float(self.start_ticks[index])
        ) * float(self.tempos[index]) / 1_000_000.0 / float(self.ticks_per_beat)


@dataclass(frozen=True)
class MidiBeatPretrainMetadata:
    """1曲分の MIDI 由来 beat/downbeat/meter ラベル。"""

    song_name: str
    midi_path: Path
    duration_sec: float
    beat_times: tuple[float, ...]
    downbeat_times: tuple[float, ...]
    meter_intervals: tuple[tuple[float, float, tuple[int, int]], ...]
    beat_intervals: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class InvalidMidiCacheEntry:
    """壊れた MIDI 判定を再利用するための file 状態。"""

    size_bytes: int
    mtime_ns: int
    error: str

    def matches(self, midi_path: Path) -> bool:
        stat = midi_path.stat()
        return self.size_bytes == int(stat.st_size) and self.mtime_ns == int(
            stat.st_mtime_ns
        )

    def to_json(self) -> dict[str, object]:
        return {
            "size_bytes": int(self.size_bytes),
            "mtime_ns": int(self.mtime_ns),
            "error": str(self.error),
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "InvalidMidiCacheEntry":
        return cls(
            size_bytes=int(data["size_bytes"]),
            mtime_ns=int(data["mtime_ns"]),
            error=str(data.get("error", "")),
        )


class InvalidMidiCache:
    """invalid MIDI の判定結果を JSON に保存する小さな cache。"""

    def __init__(self, *, midi_dir: Path, cache_path: Path) -> None:
        self.midi_dir = Path(midi_dir)
        self.cache_path = Path(cache_path)
        self._entries: dict[str, InvalidMidiCacheEntry] = {}
        self._dirty = False

        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if int(raw.get("version", 0)) != 1:
                raise ValueError(f"Unsupported invalid MIDI cache: {self.cache_path}")
            if raw.get("reader_id") != MIDI_METADATA_READER_ID:
                logger.info(
                    "Ignoring invalid MIDI cache from a different reader: %s",
                    self.cache_path,
                )
                return
            for key, value in raw.get("files", {}).items():
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid MIDI cache entry: {key}")
                self._entries[str(key)] = InvalidMidiCacheEntry.from_json(value)

    def _key(self, midi_path: Path) -> str:
        return str(Path(midi_path).relative_to(self.midi_dir))

    def get_cached_error(self, midi_path: Path) -> str | None:
        key = self._key(midi_path)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.matches(midi_path):
            return entry.error

        # ファイルが更新されている場合は、古い invalid 判定を捨てて再検査する。
        del self._entries[key]
        self._dirty = True
        return None

    def mark_invalid(self, midi_path: Path, error: str) -> None:
        stat = midi_path.stat()
        self._entries[self._key(midi_path)] = InvalidMidiCacheEntry(
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            error=str(error),
        )
        self._dirty = True

    def save_if_dirty(self) -> None:
        if not self._dirty:
            return

        # 中途半端な JSON を残さないように一時ファイル経由で置き換える。
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        payload = {
            "version": 1,
            "reader_id": MIDI_METADATA_READER_ID,
            "files": {
                key: entry.to_json() for key, entry in sorted(self._entries.items())
            },
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.cache_path)
        self._dirty = False


@dataclass(frozen=True)
class MidiMetadataCacheEntry:
    """beat ラベル導出に必要な MIDI meta event だけを保持する cache entry。"""

    size_bytes: int
    mtime_ns: int
    ticks_per_beat: int
    max_tick: int
    tempo_changes: tuple[TempoChange, ...]
    time_signatures: tuple[TimeSignatureChange, ...]

    def matches(self, midi_path: Path) -> bool:
        stat = midi_path.stat()
        return self.size_bytes == int(stat.st_size) and self.mtime_ns == int(
            stat.st_mtime_ns
        )

    def to_metadata(self, midi_path: Path) -> MidiBeatPretrainMetadata:
        return build_midi_beat_pretrain_metadata(
            midi_path=midi_path,
            ticks_per_beat=self.ticks_per_beat,
            max_tick=self.max_tick,
            tempo_changes=self.tempo_changes,
            time_signatures=self.time_signatures,
        )

    def to_json(self) -> dict[str, object]:
        return {
            "size_bytes": int(self.size_bytes),
            "mtime_ns": int(self.mtime_ns),
            "ticks_per_beat": int(self.ticks_per_beat),
            "max_tick": int(self.max_tick),
            "tempo_changes": [
                [int(change.tick), int(change.tempo)] for change in self.tempo_changes
            ],
            "time_signatures": [
                [
                    int(change.tick),
                    int(change.numerator),
                    int(change.denominator),
                ]
                for change in self.time_signatures
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "MidiMetadataCacheEntry":
        tempo_changes = tuple(
            TempoChange(tick=int(raw_change[0]), tempo=int(raw_change[1]))
            for raw_change in data["tempo_changes"]
        )
        time_signatures = tuple(
            TimeSignatureChange(
                tick=int(raw_change[0]),
                numerator=int(raw_change[1]),
                denominator=int(raw_change[2]),
            )
            for raw_change in data["time_signatures"]
        )
        return cls(
            size_bytes=int(data["size_bytes"]),
            mtime_ns=int(data["mtime_ns"]),
            ticks_per_beat=int(data["ticks_per_beat"]),
            max_tick=int(data["max_tick"]),
            tempo_changes=tempo_changes,
            time_signatures=time_signatures,
        )


class MidiMetadataCache:
    """valid MIDI の meta event 読み取り結果を JSON に保存する cache。"""

    def __init__(self, *, midi_dir: Path, cache_path: Path) -> None:
        self.midi_dir = Path(midi_dir)
        self.cache_path = Path(cache_path)
        self._entries: dict[str, MidiMetadataCacheEntry] = {}
        self._dirty = False

        if self.cache_path.exists():
            with open(self.cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if int(raw.get("version", 0)) != 1:
                raise ValueError(f"Unsupported MIDI metadata cache: {self.cache_path}")
            if raw.get("reader_id") != MIDI_METADATA_READER_ID:
                logger.info(
                    "Ignoring MIDI metadata cache from a different reader: %s",
                    self.cache_path,
                )
                return
            for key, value in raw.get("files", {}).items():
                if not isinstance(value, dict):
                    raise ValueError(f"Invalid MIDI metadata cache entry: {key}")
                self._entries[str(key)] = MidiMetadataCacheEntry.from_json(value)

    def _key(self, midi_path: Path) -> str:
        return str(Path(midi_path).relative_to(self.midi_dir))

    def get_metadata(self, midi_path: Path) -> MidiBeatPretrainMetadata | None:
        key = self._key(midi_path)
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.matches(midi_path):
            return entry.to_metadata(midi_path)

        # MIDI が更新されている場合は、古い meta event を捨てて読み直す。
        del self._entries[key]
        self._dirty = True
        return None

    def mark_valid(
        self,
        *,
        midi_path: Path,
        ticks_per_beat: int,
        max_tick: int,
        tempo_changes: tuple[TempoChange, ...],
        time_signatures: tuple[TimeSignatureChange, ...],
    ) -> None:
        stat = midi_path.stat()
        self._entries[self._key(midi_path)] = MidiMetadataCacheEntry(
            size_bytes=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            ticks_per_beat=int(ticks_per_beat),
            max_tick=int(max_tick),
            tempo_changes=tuple(tempo_changes),
            time_signatures=tuple(time_signatures),
        )
        self._dirty = True

    def save_if_dirty(self) -> None:
        if not self._dirty:
            return

        # 中途半端な JSON を残さないように一時ファイル経由で置き換える。
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        payload = {
            "version": 1,
            "reader_id": MIDI_METADATA_READER_ID,
            "files": {
                key: entry.to_json() for key, entry in sorted(self._entries.items())
            },
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp_path.replace(self.cache_path)
        self._dirty = False


def normalize_meter_class_sequence(
    meter_classes: Sequence[tuple[int, int]] | None,
) -> tuple[tuple[int, int], ...]:
    """外部から受け取った meter class 表を検証して固定順にそろえる。"""

    normalized: set[tuple[int, int]] = set()
    for meter_num, meter_den in meter_classes or ():
        meter_num = int(meter_num)
        meter_den = int(meter_den)
        if meter_num <= 0 or meter_den <= 0:
            raise ValueError("meter class values must be positive")
        normalized.add((meter_num, meter_den))
    return tuple(sorted(normalized))


def read_beat_label_meter_classes(root: str | Path) -> tuple[tuple[int, int], ...]:
    """既存 beat dataset の JSON ラベルから meter class だけを読む。"""

    label_dir = Path(root) / "label"
    if not label_dir.exists():
        return ()

    meter_keys: set[tuple[int, int]] = set()
    for label_path in sorted(label_dir.glob("*.beat.beats.json")):
        with open(label_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)
        for raw_measure in label_data.get("measures", []):
            meter_num = int(raw_measure["time_sig_num"])
            meter_den = int(raw_measure["time_sig_den"])
            if meter_num > 0 and meter_den > 0:
                meter_keys.add((meter_num, meter_den))
    return tuple(sorted(meter_keys))


def _dedupe_tempo_changes(
    raw_changes: Sequence[tuple[int, int]],
) -> tuple[TempoChange, ...]:
    # 同じ tick に複数 tempo がある場合は MIDI 上で後に出たものを採用する。
    by_tick: dict[int, int] = {0: 500000}
    for tick, tempo in raw_changes:
        tick = int(tick)
        tempo = int(tempo)
        if tick < 0 or tempo <= 0:
            continue
        by_tick[tick] = tempo
    return tuple(
        TempoChange(tick=tick, tempo=tempo) for tick, tempo in sorted(by_tick.items())
    )


def _dedupe_time_signature_changes(
    raw_changes: Sequence[tuple[int, int, int]],
    ticks_per_beat: int = 480,
) -> tuple[TimeSignatureChange, ...]:
    # time signature が未指定の先頭区間は MIDI 標準の 4/4 として扱う。
    by_tick: dict[int, tuple[int, int]] = {0: (4, 4)}
    for tick, numerator, denominator in raw_changes:
        tick = int(tick)
        numerator = int(numerator)
        denominator = int(denominator)
        if tick < 0 or numerator <= 0 or denominator <= 0:
            continue
        by_tick[tick] = (numerator, denominator)

    deduped: list[TimeSignatureChange] = []
    last_change_tick = 0
    last_signature: tuple[int, int] | None = None

    for tick, (numerator, denominator) in sorted(by_tick.items()):
        if last_signature == (numerator, denominator):
            # 直前と同一拍子記号であり、かつ位置が直前セクションの小節境界上（整数倍）にある場合のみ統合
            num, den = last_signature
            measure_ticks = (ticks_per_beat * 4 * num) // den
            if measure_ticks > 0 and (tick - last_change_tick) % measure_ticks == 0:
                continue

        deduped.append(
            TimeSignatureChange(
                tick=tick,
                numerator=numerator,
                denominator=denominator,
            )
        )
        last_change_tick = tick
        last_signature = (numerator, denominator)

    return tuple(deduped)


def _read_midi_meta_events_with_mido(
    midi_path: Path,
) -> tuple[int, int, tuple[TempoChange, ...], tuple[TimeSignatureChange, ...]]:
    from mido import MidiFile

    midi_file = MidiFile(midi_path)
    max_tick = 0
    raw_tempos: list[tuple[int, int]] = []
    raw_time_signatures: list[tuple[int, int, int]] = []

    # 1. 全トラックの delta time を絶対 tick に直し、meta event だけ拾う。
    for track in midi_file.tracks:
        tick = 0
        for message in track:
            tick += int(message.time)
            max_tick = max(max_tick, tick)
            if message.type == "set_tempo":
                raw_tempos.append((tick, int(message.tempo)))
            elif message.type == "time_signature":
                raw_time_signatures.append(
                    (tick, int(message.numerator), int(message.denominator))
                )

    tpq = int(midi_file.ticks_per_beat)
    return (
        tpq,
        int(max_tick),
        _dedupe_tempo_changes(raw_tempos),
        _dedupe_time_signature_changes(raw_time_signatures, ticks_per_beat=tpq),
    )


def _read_midi_meta_events_with_symusic(
    midi_path: Path,
) -> tuple[int, int, tuple[TempoChange, ...], tuple[TimeSignatureChange, ...]]:
    from symusic import Score

    score = Score(midi_path)
    raw_tempos = [(int(tempo.time), int(tempo.mspq)) for tempo in score.tempos]
    raw_time_signatures = [
        (
            int(time_signature.time),
            int(time_signature.numerator),
            int(time_signature.denominator),
        )
        for time_signature in score.time_signatures
    ]

    tpq = int(score.tpq)
    return (
        tpq,
        int(score.end()),
        _dedupe_tempo_changes(raw_tempos),
        _dedupe_time_signature_changes(raw_time_signatures, ticks_per_beat=tpq),
    )


def _read_midi_meta_events(
    midi_path: Path,
) -> tuple[int, int, tuple[TempoChange, ...], tuple[TimeSignatureChange, ...]]:
    try:
        return _read_midi_meta_events_with_symusic(midi_path)
    except ImportError:
        return _read_midi_meta_events_with_mido(midi_path)
    except Exception:
        # symusic で読めない MIDI でも mido が読める場合があるため、最後に fallback する。
        return _read_midi_meta_events_with_mido(midi_path)


def _iter_meter_sections(
    *,
    time_signatures: Sequence[TimeSignatureChange],
    end_tick: int,
) -> Sequence[tuple[int, int, int, int]]:
    sections: list[tuple[int, int, int, int]] = []
    for index, signature in enumerate(time_signatures):
        start_tick = int(signature.tick)
        next_tick = (
            int(time_signatures[index + 1].tick)
            if index + 1 < len(time_signatures)
            else int(end_tick)
        )
        end_section_tick = min(int(end_tick), next_tick)
        if end_section_tick > start_tick:
            sections.append(
                (
                    start_tick,
                    end_section_tick,
                    int(signature.numerator),
                    int(signature.denominator),
                )
            )
    return tuple(sections)


def build_midi_beat_pretrain_metadata(
    *,
    midi_path: Path,
    ticks_per_beat: int,
    max_tick: int,
    tempo_changes: Sequence[TempoChange],
    time_signatures: Sequence[TimeSignatureChange],
) -> MidiBeatPretrainMetadata:
    """読み取り済み meta event から beat/downbeat/meter ラベルを導出する。"""

    if max_tick <= 0:
        raise ValueError(f"MIDI has no positive duration: {midi_path}")

    converter = TickSecondConverter.from_tempo_changes(
        ticks_per_beat=ticks_per_beat,
        tempo_changes=tempo_changes,
    )
    duration_sec = converter.tick_to_seconds(max_tick)
    if duration_sec <= 0.0:
        raise ValueError(f"MIDI has no positive duration: {midi_path}")

    beat_times: list[float] = []
    downbeat_times: list[float] = []
    meter_intervals: list[tuple[float, float, tuple[int, int]]] = []
    beat_intervals: list[tuple[float, float]] = []
    end_tick_fraction = Fraction(max_tick, 1)

    # 2. 拍子区間ごとに小節 grid を作る。beat 長は分母に合わせた音価で計算する。
    for section_start, section_end, meter_num, meter_den in _iter_meter_sections(
        time_signatures=time_signatures,
        end_tick=max_tick,
    ):
        beat_tick_step = Fraction(ticks_per_beat * 4, meter_den)
        if beat_tick_step <= 0:
            continue
        measure_tick_step = beat_tick_step * int(meter_num)
        measure_tick = Fraction(section_start, 1)
        section_end_fraction = Fraction(section_end, 1)

        while measure_tick < section_end_fraction:
            next_measure_tick = min(
                measure_tick + measure_tick_step,
                section_end_fraction,
                end_tick_fraction,
            )
            measure_start_sec = converter.tick_to_seconds(measure_tick)
            measure_end_sec = converter.tick_to_seconds(next_measure_tick)
            if measure_end_sec <= measure_start_sec:
                measure_tick += measure_tick_step
                continue

            downbeat_times.append(measure_start_sec)
            meter_intervals.append(
                (measure_start_sec, measure_end_sec, (int(meter_num), int(meter_den)))
            )

            for beat_index in range(int(meter_num)):
                beat_tick = measure_tick + beat_tick_step * beat_index
                if beat_tick >= next_measure_tick:
                    break
                next_beat_tick = min(beat_tick + beat_tick_step, next_measure_tick)
                beat_start_sec = converter.tick_to_seconds(beat_tick)
                beat_end_sec = converter.tick_to_seconds(next_beat_tick)
                beat_times.append(beat_start_sec)
                if beat_end_sec > beat_start_sec:
                    beat_intervals.append((beat_start_sec, beat_end_sec))

            measure_tick += measure_tick_step

    if not meter_intervals:
        raise ValueError(f"No usable meter intervals found in {midi_path}")

    return MidiBeatPretrainMetadata(
        song_name=midi_path.stem,
        midi_path=midi_path,
        duration_sec=float(duration_sec),
        beat_times=tuple(beat_times),
        downbeat_times=tuple(downbeat_times),
        meter_intervals=tuple(meter_intervals),
        beat_intervals=tuple(beat_intervals),
    )


def read_midi_beat_pretrain_metadata(
    midi_path: str | Path,
) -> MidiBeatPretrainMetadata:
    """MIDI の tempo/time signature map から beat/downbeat/meter ラベルを導出する。"""

    midi_path = Path(midi_path)
    (
        ticks_per_beat,
        max_tick,
        tempo_changes,
        time_signatures,
    ) = _read_midi_meta_events(midi_path)
    return build_midi_beat_pretrain_metadata(
        midi_path=midi_path,
        ticks_per_beat=ticks_per_beat,
        max_tick=max_tick,
        tempo_changes=tempo_changes,
        time_signatures=time_signatures,
    )


class MidiBeatPretrainDataset(Dataset):
    """MIDI 単体から beat/downbeat/meter の前学習 batch を作る Dataset。"""

    def __init__(
        self,
        midi_dir: str | Path,
        *,
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
        max_files: int = 0,
        skip_invalid: bool = True,
        invalid_midi_cache_path: str | Path | None = None,
        midi_metadata_cache_path: str | Path | None = None,
    ) -> None:
        super().__init__()
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

        # 1. MIDI の tempo/time signature map から全曲のラベル候補を作る。
        midi_paths = sorted(
            {
                *self.midi_dir.glob("*.mid"),
                *self.midi_dir.glob("*.midi"),
                *self.midi_dir.glob("*.MID"),
                *self.midi_dir.glob("*.MIDI"),
            }
        )
        if max_files > 0:
            midi_paths = midi_paths[: int(max_files)]
        raw_items: list[MidiBeatPretrainMetadata] = []
        meter_keys = set(normalize_meter_class_sequence(extra_meter_classes))
        skipped_count = 0
        cached_invalid_count = 0
        cached_metadata_count = 0
        invalid_cache = (
            InvalidMidiCache(
                midi_dir=self.midi_dir,
                cache_path=Path(invalid_midi_cache_path),
            )
            if invalid_midi_cache_path is not None
            else None
        )
        metadata_cache = (
            MidiMetadataCache(
                midi_dir=self.midi_dir,
                cache_path=Path(midi_metadata_cache_path),
            )
            if midi_metadata_cache_path is not None
            else None
        )
        for midi_path in midi_paths:
            metadata = (
                None
                if metadata_cache is None
                else metadata_cache.get_metadata(midi_path)
            )
            if metadata is not None:
                cached_metadata_count += 1
                raw_items.append(metadata)
                for _start_sec, _end_sec, meter in metadata.meter_intervals:
                    meter_keys.add(meter)
                continue

            cached_error = (
                None
                if invalid_cache is None
                else invalid_cache.get_cached_error(midi_path)
            )
            if cached_error is not None:
                if not skip_invalid:
                    raise ValueError(
                        f"Cached invalid MIDI for beat pretrain: {midi_path} "
                        f"({cached_error})"
                    )
                cached_invalid_count += 1
                continue

            try:
                (
                    ticks_per_beat,
                    max_tick,
                    tempo_changes,
                    time_signatures,
                ) = _read_midi_meta_events(midi_path)
                metadata = build_midi_beat_pretrain_metadata(
                    midi_path=midi_path,
                    ticks_per_beat=ticks_per_beat,
                    max_tick=max_tick,
                    tempo_changes=tempo_changes,
                    time_signatures=time_signatures,
                )
                if metadata_cache is not None:
                    metadata_cache.mark_valid(
                        midi_path=midi_path,
                        ticks_per_beat=ticks_per_beat,
                        max_tick=max_tick,
                        tempo_changes=tempo_changes,
                        time_signatures=time_signatures,
                    )
            except Exception as exc:
                if invalid_cache is not None:
                    invalid_cache.mark_invalid(midi_path, str(exc))
                if not skip_invalid:
                    if metadata_cache is not None:
                        metadata_cache.save_if_dirty()
                    if invalid_cache is not None:
                        invalid_cache.save_if_dirty()
                    raise
                skipped_count += 1
                logger.warning(
                    "Skipping invalid MIDI for beat pretrain: %s (%s)",
                    midi_path,
                    exc,
                )
                continue
            raw_items.append(metadata)
            for _start_sec, _end_sec, meter in metadata.meter_intervals:
                meter_keys.add(meter)

        if metadata_cache is not None:
            metadata_cache.save_if_dirty()
        if invalid_cache is not None:
            invalid_cache.save_if_dirty()
        if cached_metadata_count > 0:
            logger.info(
                "Loaded %d cached MIDI metadata entries for beat pretrain.",
                cached_metadata_count,
            )
        if cached_invalid_count > 0:
            logger.info(
                "Skipped %d cached invalid MIDI files for beat pretrain.",
                cached_invalid_count,
            )
        if skipped_count > 0:
            logger.info(
                "Skipped %d invalid MIDI files for beat pretrain.",
                skipped_count,
            )
        if not raw_items:
            raise ValueError(
                f"No usable beat pretrain MIDI files found in {self.midi_dir}"
            )

        # 2. 本学習と pretrain MIDI の拍子を union し、同じ head 形状で保存できるようにする。
        self.meter_classes = tuple(sorted(meter_keys))
        self.meter_to_index = {
            meter: index for index, meter in enumerate(self.meter_classes)
        }
        self.num_meter_classes = len(self.meter_classes)
        if self.num_meter_classes == 0:
            raise ValueError("No meter classes found for beat pretrain")

        self.window_frames = int(round(self.window_ms * self.sample_rate / 1000.0))
        self.window_sec = float(self.window_frames) / float(self.sample_rate)
        self.model_frames = math.ceil(self.window_frames / self.hop_length)

        # 3. meter 区間を class index へ変換し、BalancedSoftmaxLoss 用の出現量を数える。
        self.items: list[dict[str, Any]] = []
        meter_counts = torch.zeros(self.num_meter_classes, dtype=torch.float32)
        for raw_item in raw_items:
            indexed_intervals: list[tuple[float, float, int]] = []
            for start_sec, end_sec, meter in raw_item.meter_intervals:
                meter_index = self.meter_to_index[meter]
                indexed_intervals.append((start_sec, end_sec, meter_index))
                start_frame = max(
                    0,
                    math.floor(float(start_sec) * self.sample_rate / self.hop_length),
                )
                end_frame = math.ceil(
                    float(end_sec) * self.sample_rate / self.hop_length
                )
                if end_frame > start_frame:
                    meter_counts[meter_index] += float(end_frame - start_frame)

            self.items.append(
                {
                    "song_name": raw_item.song_name,
                    "duration_sec": raw_item.duration_sec,
                    "beat_times": raw_item.beat_times,
                    "downbeat_times": raw_item.downbeat_times,
                    "meter_intervals": tuple(indexed_intervals),
                    "beat_intervals": raw_item.beat_intervals,
                }
            )

        self.meter_class_counts = meter_counts
        self.meter_sampling_counts = tuple(
            float(count) for count in meter_counts.tolist()
        )

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
        last_error: MidiReadError | None = None
        for offset in range(len(self.items)):
            sample_idx = (int(idx) + offset) % len(self.items)
            try:
                return self._build_sample(sample_idx)
            except MidiReadError as exc:
                last_error = exc
                logger.warning(
                    "Skipping MIDI with unreadable note events during batch build: "
                    "%s (%s)",
                    self.items[sample_idx]["song_name"],
                    exc,
                )
                continue

        raise MidiReadError(
            "No readable MIDI note events found in beat pretrain dataset"
        ) from last_error

    def _build_sample(self, idx: int) -> dict[str, Any]:
        """1曲から学習窓を切り出し、MIDI roll と beat target を作る。"""

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

        # 4. meter 区間を frame label と loss mask に展開する。
        for start_sec, end_sec, meter_index in meter_intervals:
            start_frame = max(
                0,
                math.floor(float(start_sec) * self.sample_rate / self.hop_length),
            )
            end_frame = min(
                valid_model_frames,
                math.ceil(float(end_sec) * self.sample_rate / self.hop_length),
            )
            if end_frame > start_frame:
                meter_targets[start_frame:end_frame] = int(meter_index)
                beat_mask[start_frame:end_frame] = 1.0

        # 5. phase target は本学習 dataset と同じ形式で作る。
        MidiBeatDataset._assign_phase_targets(
            phase_targets=beat_phase_targets,
            phase_mask=phase_mask,
            intervals=beat_intervals,
            window_start_sec=0.0,
            valid_model_frames=valid_model_frames,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
        )
        bar_phase_mask = torch.zeros(self.model_frames, dtype=torch.float32)
        MidiBeatDataset._assign_phase_targets(
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

        # 6. beat/downbeat のイベントを最近傍 frame に立てる。
        for target, times in (
            (beat_targets, beat_times),
            (downbeat_targets, downbeat_times),
        ):
            for event_sec in times:
                frame_index = int(
                    round(float(event_sec) * self.sample_rate / self.hop_length)
                )
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


midi_beat_pretrain_collate_fn = midi_beat_collate_fn
