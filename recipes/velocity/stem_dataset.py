from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from torch.utils.data import Dataset

from instrument_agnostic_amt.velocity.stems import (
    STEM_CLASS_BY_NAME,
    UNKNOWN_STEM_CLASS,
)
from .split import assign_song_split


SplitName = Literal["train", "validation", "test", "all"]


@dataclass(frozen=True)
class StemAudioRecord:
    stem_name: str
    stem_class_id: int
    audio_path: Path
    label_path: Path
    input_midi_path: Path
    base_relative_level_db: float


@dataclass(frozen=True)
class StemSetExampleRecord:
    example_id: str
    song_id: str
    variation: int
    duration_seconds: float
    stems: tuple[StemAudioRecord, ...]


@dataclass(frozen=True)
class StemSetWindowRecord:
    example_index: int
    start_seconds: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _resolve(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.expanduser().resolve()


def _optional_float(value: str | None, default: float = 0.0) -> float:
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def _parse_duration_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        duration = float(text)
        return duration if duration > 0.0 else None
    except ValueError:
        return None


def _window_starts(
    duration_seconds: float,
    *,
    window_seconds: float,
    hop_seconds: float,
) -> list[float]:
    if duration_seconds <= window_seconds:
        return [0.0]
    last_full_start = max(0.0, duration_seconds - window_seconds)
    starts: list[float] = []
    cursor = 0.0
    while cursor <= last_full_start + 1e-9:
        starts.append(cursor)
        cursor += hop_seconds
    if not starts or last_full_start - starts[-1] > 1e-6:
        starts.append(last_full_start)
    return starts


def _load_audio_window(
    path: Path,
    *,
    start_seconds: float,
    window_seconds: float,
    target_sample_rate: int,
) -> tuple[torch.Tensor, int]:
    info = sf.info(str(path))
    source_rate = int(info.samplerate)
    source_start = max(0, int(round(start_seconds * source_rate)))
    source_frames = max(1, int(round(window_seconds * source_rate)))
    waveform_np, read_rate = sf.read(
        str(path),
        start=source_start,
        frames=source_frames,
        dtype="float32",
        always_2d=True,
    )
    if int(read_rate) != source_rate:
        raise RuntimeError(f"Audio sample-rate changed while reading {path.name}")
    if waveform_np.shape[1] > 2:
        waveform_np = waveform_np[:, :2]
    elif waveform_np.shape[1] == 1:
        waveform_np = np.repeat(waveform_np, 2, axis=1)
    valid_source_frames = int(waveform_np.shape[0])
    waveform = torch.from_numpy(waveform_np.T.copy())
    if source_rate != int(target_sample_rate):
        waveform = audio_functional.resample(
            waveform,
            source_rate,
            int(target_sample_rate),
        )
    target_frames = int(round(window_seconds * int(target_sample_rate)))
    if waveform.shape[1] < target_frames:
        waveform = torch.nn.functional.pad(
            waveform,
            (0, target_frames - int(waveform.shape[1])),
        )
    elif waveform.shape[1] > target_frames:
        waveform = waveform[:, :target_frames]
    valid_seconds = valid_source_frames / float(source_rate)
    valid_target_frames = min(
        target_frames,
        int(round(valid_seconds * int(target_sample_rate))),
    )
    return waveform.contiguous(), valid_target_frames


class SyntheticStemVelocityDataset(Dataset):
    """Aligned separated stems at a fixed render level with MIDI targets."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: SplitName = "train",
        sample_rate: int = 22_050,
        window_seconds: float = 8.0,
        hop_seconds: float | None = None,
        split_seed: int = 42,
        train_fraction: float = 0.9,
        validation_fraction: float = 0.05,
        use_gain_augmentation: bool = False,
        gain_jitter_std_db: float = 2.5,
        gain_clip_db: float = 18.0,
        master_gain_min_db: float = -12.0,
        master_gain_max_db: float = 0.0,
        allow_incomplete: bool = False,
        max_examples: int | None = None,
        label_cache_size: int = 16,
    ) -> None:
        super().__init__()
        if split not in ("train", "validation", "test", "all"):
            raise ValueError(f"Unknown split: {split}")
        if sample_rate <= 0 or window_seconds <= 0.0:
            raise ValueError("sample_rate and window_seconds must be positive")
        if hop_seconds is None:
            hop_seconds = window_seconds
        if hop_seconds <= 0.0:
            raise ValueError("hop_seconds must be positive")
        if gain_jitter_std_db < 0.0 or gain_clip_db <= 0.0:
            raise ValueError("invalid gain augmentation settings")
        if master_gain_max_db < master_gain_min_db:
            raise ValueError("invalid master gain range")
        if max_examples is not None and max_examples < 1:
            raise ValueError("max_examples must be positive")

        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.window_frames = int(round(self.window_seconds * self.sample_rate))
        self.use_gain_augmentation = bool(use_gain_augmentation)
        self.gain_jitter_std_db = float(gain_jitter_std_db)
        self.gain_clip_db = float(gain_clip_db)
        self.master_gain_min_db = float(master_gain_min_db)
        self.master_gain_max_db = float(master_gain_max_db)
        self.label_cache_size = int(label_cache_size)
        self._label_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

        render_manifest = self.root / "render_manifest.csv"
        examples_manifest = self.root / "examples.csv"
        if not render_manifest.is_file() or not examples_manifest.is_file():
            raise FileNotFoundError("render_manifest.csv and examples.csv are required")
        render_rows = _read_csv(render_manifest)
        rows_by_example: dict[str, list[dict[str, str]]] = {}
        for row in render_rows:
            rows_by_example.setdefault(row["example_id"], []).append(row)

        examples: list[StemSetExampleRecord] = []
        missing_audio_count = 0
        audio_duration_cache: dict[Path, float] = {}

        for row in _read_csv(examples_manifest):
            song_id = str(row["song_id"])
            assigned = assign_song_split(
                song_id,
                seed=split_seed,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
            if split != "all" and assigned != split:
                continue
            stem_rows = sorted(
                rows_by_example.get(str(row["example_id"]), []),
                key=lambda value: value["stem_name"],
            )
            if len(stem_rows) != int(row["stem_count"]):
                raise ValueError(f"Stem count mismatch for {row['example_id']}")
            stems: list[StemAudioRecord] = []
            durations: list[float] = []
            example_missing = False
            example_duration = _parse_duration_seconds(row.get("duration_seconds"))

            for stem_row in stem_rows:
                audio_path = _resolve(stem_row["rendered_stem_path"], self.root)
                if not audio_path.is_file():
                    missing_audio_count += 1
                    example_missing = True
                    break
                label_path = _resolve(stem_row["label_path"], self.root)
                if not label_path.is_file():
                    raise FileNotFoundError(f"Velocity label not found: {label_path}")
                if example_duration is None:
                    stem_duration = _parse_duration_seconds(stem_row.get("duration_seconds"))
                    if stem_duration is None:
                        if audio_path in audio_duration_cache:
                            stem_duration = audio_duration_cache[audio_path]
                        else:
                            info = sf.info(str(audio_path))
                            stem_duration = float(info.frames) / float(info.samplerate)
                            audio_duration_cache[audio_path] = stem_duration
                    durations.append(stem_duration)
                stem_name = str(stem_row["stem_name"])
                stems.append(
                    StemAudioRecord(
                        stem_name=stem_name,
                        stem_class_id=STEM_CLASS_BY_NAME.get(
                            stem_name,
                            UNKNOWN_STEM_CLASS,
                        ),
                        audio_path=audio_path,
                        label_path=label_path,
                        input_midi_path=_resolve(
                            stem_row["input_midi_path"],
                            self.root,
                        ),
                        base_relative_level_db=_optional_float(
                            stem_row.get("base_relative_level_db")
                        ),
                    )
                )
            if example_missing:
                if allow_incomplete:
                    continue
                raise FileNotFoundError(
                    f"Rendered stem missing for {row['example_id']}"
                )
            if not stems:
                continue
            final_duration = (
                example_duration
                if example_duration is not None
                else (max(durations) if durations else 0.0)
            )
            examples.append(
                StemSetExampleRecord(
                    example_id=str(row["example_id"]),
                    song_id=song_id,
                    variation=int(row["variation"]),
                    duration_seconds=final_duration,
                    stems=tuple(stems),
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
        if not examples:
            suffix = (
                f"; skipped {missing_audio_count} missing stems"
                if missing_audio_count
                else ""
            )
            raise ValueError(f"No usable examples for split={split}{suffix}")
        self.examples = examples
        self.missing_audio_count = missing_audio_count
        self.windows = [
            StemSetWindowRecord(example_index=example_index, start_seconds=start)
            for example_index, example in enumerate(examples)
            for start in _window_starts(
                example.duration_seconds,
                window_seconds=self.window_seconds,
                hop_seconds=self.hop_seconds,
            )
        ]

    @property
    def song_count(self) -> int:
        return len({example.song_id for example in self.examples})

    def __len__(self) -> int:
        return len(self.windows)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_label_cache"] = OrderedDict()
        return state

    def _load_label(self, path: Path) -> dict[str, np.ndarray]:
        key = str(path)
        cached = self._label_cache.get(key)
        if cached is not None:
            self._label_cache.move_to_end(key)
            return cached
        required = (
            "note_start_seconds",
            "note_end_seconds",
            "note_pitch",
            "note_program",
            "note_is_drum",
            "note_track_index",
            "target_velocity",
            "source_pseudo_confidence",
            "rank_source",
            "independently_randomized",
        )
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(set(required) - set(data.files))
            if missing:
                raise ValueError(
                    f"Missing label arrays in {path.name}: {', '.join(missing)}"
                )
            arrays = {name: np.asarray(data[name]).copy() for name in required}
        note_count = int(arrays["note_pitch"].size)
        for name, value in arrays.items():
            if int(value.size) != note_count:
                raise ValueError(f"Label array length mismatch in {path.name}: {name}")
        if self.label_cache_size > 0:
            self._label_cache[key] = arrays
            self._label_cache.move_to_end(key)
            while len(self._label_cache) > self.label_cache_size:
                self._label_cache.popitem(last=False)
        return arrays

    def _sample_gains(self, stems: tuple[StemAudioRecord, ...]) -> tuple[torch.Tensor, float]:
        if not self.use_gain_augmentation:
            return torch.zeros(len(stems), dtype=torch.float32), 0.0
        base = torch.tensor(
            [stem.base_relative_level_db for stem in stems],
            dtype=torch.float32,
        ).clamp(-self.gain_clip_db, self.gain_clip_db)
        jitter = torch.randn_like(base) * self.gain_jitter_std_db
        gains = (base + jitter).clamp(-self.gain_clip_db, self.gain_clip_db)
        gains = gains - torch.quantile(gains, 0.5)
        gains = gains.clamp(-self.gain_clip_db, self.gain_clip_db)
        master = float(
            torch.empty(1).uniform_(
                self.master_gain_min_db,
                self.master_gain_max_db,
            )[0]
        )
        return gains, master

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        example = self.examples[window.example_index]
        window_start = float(window.start_seconds)
        window_end = window_start + self.window_seconds
        gains, master_gain_db = self._sample_gains(example.stems)

        stem_audio: list[torch.Tensor] = []
        valid_frames: list[int] = []
        stem_active: list[bool] = []
        note_values: dict[str, list[np.ndarray]] = {
            "start": [],
            "end": [],
            "pitch": [],
            "program": [],
            "is_drum": [],
            "track_index": [],
            "stem_index": [],
            "target_velocity": [],
            "pseudo_confidence": [],
            "rank_source": [],
            "independently_randomized": [],
        }
        for stem_index, stem in enumerate(example.stems):
            audio, valid = _load_audio_window(
                stem.audio_path,
                start_seconds=window_start,
                window_seconds=self.window_seconds,
                target_sample_rate=self.sample_rate,
            )
            amplitude = float(10.0 ** ((master_gain_db + float(gains[stem_index])) / 20.0))
            stem_audio.append(audio * amplitude)
            valid_frames.append(valid)

            labels = self._load_label(stem.label_path)
            starts = labels["note_start_seconds"].astype(np.float64, copy=False)
            ends = labels["note_end_seconds"].astype(np.float64, copy=False)
            onset_mask = (starts >= window_start) & (starts < window_end)
            active_mask = (starts < window_end) & (ends > window_start)
            indices = np.flatnonzero(onset_mask)
            stem_active.append(bool(np.any(active_mask)))
            note_values["start"].append((starts[indices] - window_start).astype(np.float32))
            note_values["end"].append((ends[indices] - window_start).astype(np.float32))
            note_values["pitch"].append(labels["note_pitch"][indices].astype(np.int64))
            note_values["program"].append(labels["note_program"][indices].astype(np.int64))
            note_values["is_drum"].append(labels["note_is_drum"][indices].astype(np.bool_))
            note_values["track_index"].append(labels["note_track_index"][indices].astype(np.int64))
            note_values["stem_index"].append(np.full(indices.size, stem_index, dtype=np.int64))
            note_values["target_velocity"].append(labels["target_velocity"][indices].astype(np.int64))
            note_values["pseudo_confidence"].append(labels["source_pseudo_confidence"][indices].astype(np.float32))
            note_values["rank_source"].append(labels["rank_source"][indices].astype(np.int64))
            note_values["independently_randomized"].append(labels["independently_randomized"][indices].astype(np.bool_))

        def concatenate(name: str, dtype: torch.dtype) -> torch.Tensor:
            arrays = note_values[name]
            if not arrays or not any(array.size for array in arrays):
                return torch.zeros(0, dtype=dtype)
            return torch.as_tensor(np.concatenate(arrays), dtype=dtype)

        note_tensors = {
            "note_start_seconds": concatenate("start", torch.float32),
            "note_end_seconds": concatenate("end", torch.float32),
            "note_pitch": concatenate("pitch", torch.long),
            "note_program": concatenate("program", torch.long),
            "note_is_drum": concatenate("is_drum", torch.bool),
            "note_track_index": concatenate("track_index", torch.long),
            "note_stem_index": concatenate("stem_index", torch.long),
            "target_velocity": concatenate("target_velocity", torch.long),
            "source_pseudo_confidence": concatenate("pseudo_confidence", torch.float32),
            "rank_source": concatenate("rank_source", torch.long),
            "independently_randomized": concatenate("independently_randomized", torch.bool),
        }
        if note_tensors["target_velocity"].numel():
            order = sorted(
                range(note_tensors["target_velocity"].numel()),
                key=lambda note_index: (
                    float(note_tensors["note_start_seconds"][note_index]),
                    int(note_tensors["note_stem_index"][note_index]),
                    int(note_tensors["note_pitch"][note_index]),
                ),
            )
            order_tensor = torch.tensor(order, dtype=torch.long)
            note_tensors = {name: value[order_tensor] for name, value in note_tensors.items()}
        target_velocity = note_tensors["target_velocity"]
        return {
            "example_id": example.example_id,
            "song_id": example.song_id,
            "variation": example.variation,
            "window_start_seconds": window_start,
            "audio": torch.stack(stem_audio),
            "valid_audio_frames": torch.tensor(valid_frames, dtype=torch.long),
            **note_tensors,
            "target_velocity_unit": (target_velocity.float() - 1.0) / 126.0,
            "stem_gain_db": gains,
            "stem_class_id": torch.tensor(
                [stem.stem_class_id for stem in example.stems],
                dtype=torch.long,
            ),
            "stem_active": torch.tensor(stem_active, dtype=torch.bool),
            "stem_names": [stem.stem_name for stem in example.stems],
            "master_gain_db": master_gain_db,
            "peak_limiter_gain_db": 0.0,
        }
