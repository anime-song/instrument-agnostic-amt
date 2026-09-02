from __future__ import annotations

import csv
import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from torch.utils.data import Dataset

from ..stems import STEM_CLASS_BY_NAME, STEM_NAMES as STEM_NAMES, UNKNOWN_STEM_CLASS

SplitName = Literal["train", "validation", "test", "all"]


@dataclass(frozen=True)
class VelocityStemRecord:
    stem_name: str
    stem_class_id: int
    label_path: Path
    input_midi_path: Path
    stem_gain_db: float


@dataclass(frozen=True)
class VelocityExampleRecord:
    example_id: str
    song_id: str
    variation: int
    mixture_path: Path
    duration_seconds: float
    source_sample_rate: int
    master_gain_db: float
    peak_limiter_gain_db: float
    stems: tuple[VelocityStemRecord, ...]


@dataclass(frozen=True)
class VelocityWindowRecord:
    example_index: int
    start_seconds: float


def assign_song_split(
    song_id: str,
    *,
    seed: int = 42,
    train_fraction: float = 0.9,
    validation_fraction: float = 0.05,
) -> Literal["train", "validation", "test"]:
    """Assign every variation of one song to the same deterministic split."""

    if not 0.0 <= train_fraction <= 1.0:
        raise ValueError("train_fraction must be within 0..1")
    if not 0.0 <= validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be within 0..1")
    if train_fraction + validation_fraction > 1.0:
        raise ValueError("train and validation fractions must sum to at most 1")
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(int(seed)).encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(song_id).encode("utf-8"))
    value = int.from_bytes(digest.digest(), "little") / float(2**64)
    if value < train_fraction:
        return "train"
    if value < train_fraction + validation_fraction:
        return "validation"
    return "test"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _resolve(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.expanduser().resolve()


def _load_resampled_audio_window(
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
        padded = torch.zeros((2, target_frames), dtype=torch.float32)
        padded[:, : waveform.shape[1]] = waveform
        waveform = padded
    elif waveform.shape[1] > target_frames:
        waveform = waveform[:, :target_frames]
    valid_seconds = valid_source_frames / float(source_rate)
    valid_target_frames = min(
        target_frames,
        int(round(valid_seconds * int(target_sample_rate))),
    )
    return waveform.contiguous(), valid_target_frames


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


class SyntheticVelocityDataset(Dataset):
    """Windowed mixture audio with variable-length MIDI-note and stem targets."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: SplitName = "train",
        sample_rate: int = 22_050,
        window_seconds: float = 10.0,
        hop_seconds: float | None = None,
        split_seed: int = 42,
        train_fraction: float = 0.9,
        validation_fraction: float = 0.05,
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
        if max_examples is not None and max_examples < 1:
            raise ValueError("max_examples must be positive")
        if label_cache_size < 0:
            raise ValueError("label_cache_size must be nonnegative")

        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.sample_rate = int(sample_rate)
        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)
        self.window_frames = int(round(self.window_seconds * self.sample_rate))
        self.label_cache_size = int(label_cache_size)
        self._label_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

        render_manifest = self.root / "render_manifest.csv"
        if not render_manifest.is_file():
            raise FileNotFoundError("render_manifest.csv was not found")
        render_rows = _read_csv(render_manifest)
        render_by_example: dict[str, list[dict[str, str]]] = {}
        for row in render_rows:
            render_by_example.setdefault(row["example_id"], []).append(row)

        dataset_manifest = self.root / "dataset_manifest.csv"
        if dataset_manifest.is_file():
            example_rows = _read_csv(dataset_manifest)
        else:
            examples_path = self.root / "examples.csv"
            if not examples_path.is_file():
                raise FileNotFoundError(
                    "dataset_manifest.csv or examples.csv is required"
                )
            example_rows = _read_csv(examples_path)

        examples: list[VelocityExampleRecord] = []
        missing_audio_count = 0
        for row in example_rows:
            song_id = str(row["song_id"])
            assigned_split = assign_song_split(
                song_id,
                seed=split_seed,
                train_fraction=train_fraction,
                validation_fraction=validation_fraction,
            )
            if split != "all" and assigned_split != split:
                continue
            mixture_path = _resolve(row["mixture_path"], self.root)
            if not mixture_path.is_file():
                missing_audio_count += 1
                if allow_incomplete:
                    continue
                raise FileNotFoundError(f"Mixture audio not found: {mixture_path.name}")
            audio_info = sf.info(str(mixture_path))
            stem_rows = sorted(
                render_by_example.get(str(row["example_id"]), []),
                key=lambda value: value["stem_name"],
            )
            if not stem_rows:
                raise ValueError(f"No stems found for {row['example_id']}")
            expected_stems = int(row.get("stem_count", len(stem_rows)))
            if len(stem_rows) != expected_stems:
                raise ValueError(f"Stem count mismatch for {row['example_id']}")
            stems: list[VelocityStemRecord] = []
            for stem_row in stem_rows:
                label_path = _resolve(stem_row["label_path"], self.root)
                if not label_path.is_file():
                    raise FileNotFoundError(
                        f"Velocity label not found: {label_path.name}"
                    )
                input_midi_path = _resolve(stem_row["input_midi_path"], self.root)
                stem_name = str(stem_row["stem_name"])
                stems.append(
                    VelocityStemRecord(
                        stem_name=stem_name,
                        stem_class_id=STEM_CLASS_BY_NAME.get(
                            stem_name,
                            UNKNOWN_STEM_CLASS,
                        ),
                        label_path=label_path,
                        input_midi_path=input_midi_path,
                        stem_gain_db=float(stem_row["stem_gain_db"]),
                    )
                )
            examples.append(
                VelocityExampleRecord(
                    example_id=str(row["example_id"]),
                    song_id=song_id,
                    variation=int(row["variation"]),
                    mixture_path=mixture_path,
                    duration_seconds=float(audio_info.frames)
                    / float(audio_info.samplerate),
                    source_sample_rate=int(audio_info.samplerate),
                    master_gain_db=float(row.get("master_gain_db", 0.0) or 0.0),
                    peak_limiter_gain_db=float(
                        row.get("peak_limiter_gain_db", 0.0) or 0.0
                    ),
                    stems=tuple(stems),
                )
            )
            if max_examples is not None and len(examples) >= max_examples:
                break
        if not examples:
            suffix = (
                f"; skipped {missing_audio_count} incomplete examples"
                if missing_audio_count
                else ""
            )
            raise ValueError(f"No usable examples for split={split}{suffix}")
        self.examples = examples
        self.missing_audio_count = missing_audio_count
        self.windows = [
            VelocityWindowRecord(example_index=example_index, start_seconds=start)
            for example_index, example in enumerate(self.examples)
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
            "stem_gain_db",
        )
        with np.load(path, allow_pickle=False) as data:
            missing = sorted(set(required) - set(data.files))
            if missing:
                raise ValueError(
                    f"Missing label arrays in {path.name}: {', '.join(missing)}"
                )
            arrays = {name: np.asarray(data[name]).copy() for name in required}
        note_count = int(arrays["note_pitch"].size)
        for name in required:
            if name == "stem_gain_db":
                continue
            if int(arrays[name].size) != note_count:
                raise ValueError(f"Label array length mismatch in {path.name}: {name}")
        if self.label_cache_size > 0:
            self._label_cache[key] = arrays
            self._label_cache.move_to_end(key)
            while len(self._label_cache) > self.label_cache_size:
                self._label_cache.popitem(last=False)
        return arrays

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        example = self.examples[window.example_index]
        window_start = float(window.start_seconds)
        window_end = window_start + self.window_seconds
        audio, valid_audio_frames = _load_resampled_audio_window(
            example.mixture_path,
            start_seconds=window_start,
            window_seconds=self.window_seconds,
            target_sample_rate=self.sample_rate,
        )

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
        stem_gains: list[float] = []
        stem_classes: list[int] = []
        stem_active: list[bool] = []
        stem_names: list[str] = []
        for stem_index, stem in enumerate(example.stems):
            labels = self._load_label(stem.label_path)
            starts = labels["note_start_seconds"].astype(np.float64, copy=False)
            ends = labels["note_end_seconds"].astype(np.float64, copy=False)
            onset_mask = (starts >= window_start) & (starts < window_end)
            active_mask = (starts < window_end) & (ends > window_start)
            indices = np.flatnonzero(onset_mask)
            note_values["start"].append(
                (starts[indices] - window_start).astype(np.float32)
            )
            note_values["end"].append((ends[indices] - window_start).astype(np.float32))
            note_values["pitch"].append(labels["note_pitch"][indices].astype(np.int64))
            note_values["program"].append(
                labels["note_program"][indices].astype(np.int64)
            )
            note_values["is_drum"].append(
                labels["note_is_drum"][indices].astype(np.bool_)
            )
            note_values["track_index"].append(
                labels["note_track_index"][indices].astype(np.int64)
            )
            note_values["stem_index"].append(
                np.full(indices.size, stem_index, dtype=np.int64)
            )
            note_values["target_velocity"].append(
                labels["target_velocity"][indices].astype(np.int64)
            )
            note_values["pseudo_confidence"].append(
                labels["source_pseudo_confidence"][indices].astype(np.float32)
            )
            note_values["rank_source"].append(
                labels["rank_source"][indices].astype(np.int64)
            )
            note_values["independently_randomized"].append(
                labels["independently_randomized"][indices].astype(np.bool_)
            )
            label_gain = float(labels["stem_gain_db"])
            if not np.isclose(label_gain, stem.stem_gain_db, atol=1e-4):
                raise ValueError(
                    f"Stem gain mismatch for {example.example_id}/{stem.stem_name}"
                )
            stem_gains.append(stem.stem_gain_db)
            stem_classes.append(stem.stem_class_id)
            stem_active.append(bool(np.any(active_mask)))
            stem_names.append(stem.stem_name)

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
            "source_pseudo_confidence": concatenate(
                "pseudo_confidence",
                torch.float32,
            ),
            "rank_source": concatenate("rank_source", torch.long),
            "independently_randomized": concatenate(
                "independently_randomized",
                torch.bool,
            ),
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
            note_tensors = {
                name: value[order_tensor] for name, value in note_tensors.items()
            }
        target_velocity = note_tensors["target_velocity"]
        return {
            "example_id": example.example_id,
            "song_id": example.song_id,
            "variation": example.variation,
            "window_start_seconds": window_start,
            "audio": audio,
            "valid_audio_frames": valid_audio_frames,
            **note_tensors,
            "target_velocity_unit": ((target_velocity.to(torch.float32) - 1.0) / 126.0),
            "stem_gain_db": torch.tensor(stem_gains, dtype=torch.float32),
            "stem_class_id": torch.tensor(stem_classes, dtype=torch.long),
            "stem_active": torch.tensor(stem_active, dtype=torch.bool),
            "stem_names": stem_names,
            "master_gain_db": float(example.master_gain_db),
            "peak_limiter_gain_db": float(example.peak_limiter_gain_db),
        }
