from __future__ import annotations

import csv
import logging
import math
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from instrument_agnostic_amt.amt.data.audio import compute_model_frames, load_audio_window
from instrument_agnostic_amt.amt.data.pitch_aliases import (
    parse_pitch_aliases,
    remap_drum_pitch_array,
)
from instrument_agnostic_amt.amt.modeling.model import normalize_semi_crf_version
from instrument_agnostic_amt.taxonomy.instrument_classes import (
    NUM_INSTRUMENT_CLASSES,
    get_instrument_class_id_by_name,
)
from recipes.common.augmentation import AudioAugmentor

from .harmony import (
    HarmonyAugmentationConfig,
    HarmonyAugmentationManager,
    _build_harmony_augmentation_config,
)
from .notes import (
    WindowNotes,
    concat_window_notes,
    split_window_notes,
)
from .sampling import StemWindowSelector
from .targets import (
    build_frame_instrument_targets,
    build_frame_note_targets,
    build_interval_targets,
)

logger = logging.getLogger(__name__)

PITCH_SHIFT_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_pitch_(?P<shift>-?\d+)$")
try:
    DRUM_CLASS_ID = get_instrument_class_id_by_name("drums")
except KeyError:
    DRUM_CLASS_ID = None


def _get_instrument_name(stem_name: str) -> str:
    """Extract an instrument name from a stem name, dropping trailing numeric suffixes."""
    parts = stem_name.split("__")
    inst_part = parts[-1] if len(parts) > 1 else stem_name
    return re.sub(r"_\d+$", "", inst_part)


def _split_pitch_shift_suffix(name: str) -> tuple[str, int]:
    """Split a `_pitch_<semitone>` suffix from an augmented stem name."""
    match = PITCH_SHIFT_SUFFIX_RE.match(name)
    if match is None:
        return name, 0
    return match.group("base"), int(match.group("shift"))


def _valid_instrument_id_set(instrument_ids: np.ndarray) -> frozenset[int]:
    if instrument_ids.size == 0:
        return frozenset()

    values = instrument_ids.astype(np.int64, copy=False)
    valid_mask = (values >= 0) & (values < NUM_INSTRUMENT_CLASSES)
    if not np.any(valid_mask):
        return frozenset()

    return frozenset(int(value) for value in np.unique(values[valid_mask]).tolist())


class StemDataset(Dataset):
    """
    Load stem audio/MIDI pairs and build all-instrument AMT samples.
    Supports weighted multi-dataset sampling from a dataset_config YAML file.
    Dataset entries may share a song-identity group across manifests.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        dataset_config_path: str | Path | None = None,
        window_ms: int = 5000,
        n_fft: int = 1024,
        hop_length: int = 512,
        sample_rate: int = 22050,
        semi_crf_version: str = 'v1',
        num_pitch_slots: int = 1,
        p_intra_drop: float = 0.2,
        p_cross_mix: float = 0.1,
        p_cross_mix_decay: float = 0.3,
        max_cross_stems: int = 5,
        max_cross_aug_positive_pairs: int = 160,
        max_cross_aug_intervals: int = 1200,
        p_augment: float = 0.5,
        ir_folder: str | Path | None = None,
        noise_folder: str | Path | None = None,
        drum_folder: str | Path | None = None,
        p_drum_mix: float = 0.1,
        seed: int = 42,
    ):
        self.window_ms = int(window_ms)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.semi_crf_version = normalize_semi_crf_version(semi_crf_version)
        self.num_pitch_slots = max(1, int(num_pitch_slots))
        # Probability of dropping stems from the same song.
        self.p_intra_drop = float(p_intra_drop)
        # Probability of mixing stems from other songs.
        self.p_cross_mix = float(p_cross_mix)
        # Decay factor when adding multiple cross-song stems.
        self.p_cross_mix_decay = float(p_cross_mix_decay)
        self.max_cross_stems = int(max_cross_stems)
        self.max_cross_aug_positive_pairs = int(max_cross_aug_positive_pairs)
        self.max_cross_aug_intervals = int(max_cross_aug_intervals)
        self.p_augment = float(p_augment)
        self.seed = int(seed)
        self.epoch = 0
        self.drum_pitch_aliases: dict[int, int] = {}
        self.ir_folder = ir_folder
        self.noise_folder = noise_folder
        self.group_augmentors: dict[str, AudioAugmentor | None] = {}
        self.drum_augmentor = self._build_audio_augmentor(distortion_augmentations=None)

        self.p_drum_mix = float(p_drum_mix)
        self.drum_files: list[str] = []
        if drum_folder is not None and Path(drum_folder).exists():
            for p in Path(drum_folder).rglob("*"):
                if p.is_file() and p.suffix.lower() in [".wav", ".flac", ".mp3"]:
                    self.drum_files.append(str(p))
            if self.drum_files:
                logger.info(f"Found {len(self.drum_files)} drum files in {drum_folder}")
            else:
                logger.warning(f"No audio files found in drum_folder: {drum_folder}")

        self.window_frames = int(round(self.window_ms * self.sample_rate / 1000.0))
        self.model_frames = max(
            0, compute_model_frames(self.window_frames, self.n_fft, self.hop_length)
        )

        self.stems_by_song = defaultdict(list)
        self.all_stems = []
        # Index pitch-shift variants by their original stem name.
        self.pitch_shift_stems_by_group: dict[
            tuple[str, str], dict[int, dict[str, Any]]
        ] = defaultdict(dict)

        # Dataset entries: [{name, group, song_names, weight}, ...].
        self.dataset_groups: list[dict] = []

        if dataset_config_path is not None and Path(dataset_config_path).exists():
            self._load_config(dataset_config_path)
        else:
            # Without a YAML config, train from a single manifest.
            source_stems_by_song = self._load_manifest(manifest_path)
            primary_songs = list(source_stems_by_song)
            self.dataset_groups.append(
                {
                    "name": "main",
                    "group": "main",
                    "song_names": primary_songs,
                    "source_stems_by_song": source_stems_by_song,
                    "weight": 1.0,
                    "use_for_cross_aug": True,
                    "active_window_sampling": False,
                    "use_intra_drop": True,
                    "allow_multi_stem_same_song": True,
                    "mask_instrument_loss": False,
                    "distortion_augmentations": (),
                    "harmony_config": HarmonyAugmentationConfig(),
                }
            )
            self.group_augmentors["main"] = self._build_audio_augmentor(
                distortion_augmentations=()
            )

        if not self.dataset_groups:
            raise ValueError(
                f"No usable dataset groups found for Semi-CRF {self.semi_crf_version}"
            )

        self.dataset_groups_by_name = {
            str(group["name"]): group for group in self.dataset_groups
        }
        self.harmony_manager = HarmonyAugmentationManager(
            dataset_groups_by_name=self.dataset_groups_by_name,
            pitch_shift_stems_by_group=self.pitch_shift_stems_by_group,
        )
        self.window_selector = StemWindowSelector(
            dataset_groups_by_name=self.dataset_groups_by_name,
            window_ms=self.window_ms,
            p_intra_drop=self.p_intra_drop,
        )

        # Song list for the primary dataset group.
        self.primary_song_names = self.dataset_groups[0]["song_names"]

        # Convert group weights into cumulative sampling probabilities.
        total_weight = sum(group["weight"] for group in self.dataset_groups)
        self._cumulative_probs: list[float] = []
        cumulative = 0.0
        for group in self.dataset_groups:
            cumulative += group["weight"] / total_weight
            self._cumulative_probs.append(cumulative)

        for group in self.dataset_groups:
            probability = group["weight"] / total_weight * 100
            logger.info(
                f"Dataset '{group['name']}': {len(group['song_names'])} songs, "
                f"group={group.get('group', group['name'])}, "
                f"weight={group['weight']}, prob={probability:.1f}%, "
                f"cross_aug={group.get('use_for_cross_aug', True)}, "
                f"active_window={group.get('active_window_sampling', False)}, "
                f"multi_stem_same_song={group.get('allow_multi_stem_same_song', True)}, "
                f"intra_drop={group.get('use_intra_drop', True)}, "
                f"mask_inst={group.get('mask_instrument_loss', False)}, "
                f"distort={group.get('distortion_augmentations', ()) or 'none'}, "
                f"harmony={group.get('harmony_config', HarmonyAugmentationConfig()).describe()}"
            )

        # Build the cross-augmentation group sampler.
        self.cross_dataset_groups = [
            g
            for g in self.dataset_groups
            if g.get("use_for_cross_aug", True)
            and not g.get("mask_instrument_loss", False)
        ]
        self._cross_cumulative_probs = []
        if self.cross_dataset_groups:
            total_cross_weight = sum(g["weight"] for g in self.cross_dataset_groups)
            cumulative_cross = 0.0
            for g in self.cross_dataset_groups:
                cumulative_cross += g["weight"] / total_cross_weight
                self._cross_cumulative_probs.append(cumulative_cross)

    def _build_audio_augmentor(
        self,
        *,
        distortion_augmentations: list[str] | tuple[str, ...] | None,
    ) -> AudioAugmentor | None:
        """Build an augmentor configured for one dataset group."""
        if self.p_augment <= 0.0:
            return None
        return AudioAugmentor(
            sample_rate=self.sample_rate,
            ir_folder=self.ir_folder,
            noise_folder=self.noise_folder,
            distortion_augmentations=distortion_augmentations,
        )

    def _get_stem_augmentor(self, stem: dict[str, Any]) -> AudioAugmentor | None:
        """Return the augmentor for the dataset group of this stem."""
        group_name = str(stem.get("dataset_group_name", "main"))
        return self.group_augmentors.get(group_name)

    def _load_config(self, config_path: str | Path):
        """Load dataset YAML and register every usable manifest."""
        import yaml

        config_path = Path(config_path)
        config_dir = config_path.parent

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self.drum_pitch_aliases = parse_pitch_aliases(
            config.get("drum_pitch_aliases"),
        )

        dataset_names: set[str] = set()
        for dataset_entry in config.get("datasets", []):
            manifest_rel = dataset_entry["manifest"]
            manifest_full = config_dir / manifest_rel
            if not manifest_full.exists():
                root_relative = Path(manifest_rel)
                if root_relative.exists():
                    manifest_full = root_relative
            if not manifest_full.exists():
                logger.warning(f"Manifest not found, skipping: {manifest_full}")
                continue

            dataset_name = str(dataset_entry.get("name", manifest_rel)).strip()
            if not dataset_name:
                raise ValueError("Dataset name must not be empty")
            if dataset_name in dataset_names:
                raise ValueError(f"Duplicate dataset name: {dataset_name}")
            dataset_names.add(dataset_name)

            # group is a virtual folder for song identity. Entries without it
            # retain the old dataset-name prefix and therefore cannot collide.
            raw_song_group_name = dataset_entry.get("group")
            song_group_name = (
                dataset_name
                if raw_song_group_name is None
                else str(raw_song_group_name).strip()
            )
            if not song_group_name:
                raise ValueError(
                    f"Dataset '{dataset_name}' has an empty group"
                )

            mask_inst = bool(dataset_entry.get("mask_instrument_loss", False))
            if mask_inst and self.semi_crf_version == "v2":
                logger.info(
                    "Skipping dataset group %s for V2 because mask_instrument_loss=true",
                    dataset_entry.get("name", manifest_rel),
                )
                continue
            distortion_augmentations = tuple(
                dataset_entry.get("distortion_augmentations", []) or []
            )
            harmony_config = _build_harmony_augmentation_config(dataset_entry)
            source_stems_by_song = self._load_manifest(
                manifest_full,
                song_name_prefix=song_group_name,
                mask_instrument_loss=mask_inst,
                dataset_group_name=dataset_name,
                song_group_name=song_group_name,
            )
            song_names = list(source_stems_by_song)
            if not song_names:
                logger.warning(
                    "Manifest has no rows, skipping dataset '%s': %s",
                    dataset_name,
                    manifest_full,
                )
                continue

            self.dataset_groups.append(
                {
                    "name": dataset_name,
                    "group": song_group_name,
                    "song_names": song_names,
                    "source_stems_by_song": source_stems_by_song,
                    "weight": float(dataset_entry.get("weight", 1.0)),
                    "use_for_cross_aug": bool(
                        dataset_entry.get("use_for_cross_aug", True)
                    ),
                    "active_window_sampling": bool(
                        dataset_entry.get("active_window_sampling", False)
                    ),
                    "allow_multi_stem_same_song": bool(
                        dataset_entry.get("allow_multi_stem_same_song", True)
                    ),
                    "use_intra_drop": bool(dataset_entry.get("use_intra_drop", True)),
                    "mask_instrument_loss": bool(
                        dataset_entry.get("mask_instrument_loss", False)
                    ),
                    "distortion_augmentations": distortion_augmentations,
                    "harmony_config": harmony_config,
                }
            )
            self.group_augmentors[dataset_name] = self._build_audio_augmentor(
                distortion_augmentations=distortion_augmentations
            )

    def _load_manifest(
        self,
        manifest_path: str | Path,
        song_name_prefix: str = "",
        mask_instrument_loss: bool = False,
        dataset_group_name: str = "main",
        song_group_name: str = "main",
    ) -> dict[str, list[dict[str, Any]]]:
        """Load a manifest and return its own stems indexed by shared song key."""
        manifest_path = Path(manifest_path)
        manifest_dir = manifest_path.parent
        source_stems_by_song: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with open(manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV paths are relative to the manifest file.
                wav_rel_path = row["wav_path"].replace("\\", "/")
                wav_path = str(manifest_dir / wav_rel_path).replace("\\", "/")
                npz_path = str(manifest_dir / row["npz_path"]).replace("\\", "/")
                wav_rel_no_suffix = Path(wav_rel_path).with_suffix("")
                pitch_shift_base_name, pitch_shift_value = _split_pitch_shift_suffix(
                    wav_rel_no_suffix.name
                )
                pitch_shift_group_key = wav_rel_no_suffix.with_name(
                    pitch_shift_base_name
                ).as_posix()
                # A shared group prefix intentionally merges the same song_name
                # across manifests; the dataset name remains separate metadata.
                song_name = row["song_name"]
                if song_name_prefix:
                    song_name = f"{song_name_prefix}/{song_name}"
                stem_info = {
                    "song_name": song_name,
                    "stem_name": row["stem_name"],
                    "wav_path": wav_path,
                    "npz_path": npz_path,
                    "duration_ms": int(row["duration_ms"]),
                    "end_note_ms": int(row["end_note_ms"]),
                    "note_count": int(row["note_count"]),
                    "mask_instrument_loss": mask_instrument_loss,
                    "dataset_group_name": str(dataset_group_name),
                    "song_group_name": str(song_group_name),
                    "pitch_shift_value": pitch_shift_value,
                    "pitch_shift_group_key": pitch_shift_group_key,
                }
                self.stems_by_song[song_name].append(stem_info)
                source_stems_by_song[song_name].append(stem_info)
                self.all_stems.append(stem_info)
                pitch_shift_group = self.pitch_shift_stems_by_group[
                    (str(dataset_group_name), pitch_shift_group_key)
                ]
                pitch_shift_group[pitch_shift_value] = stem_info

        return dict(source_stems_by_song)

    def set_epoch(self, epoch: int):
        """Set epoch for deterministic per-epoch sampling."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.primary_song_names)

    def _select_dataset_group(self, rng: random.Random) -> dict:
        """Select a dataset group by configured weight."""
        roll = rng.random()
        for group, cumulative_prob in zip(self.dataset_groups, self._cumulative_probs):
            if roll < cumulative_prob:
                return group
        return self.dataset_groups[-1]

    def _select_cross_dataset_group(self, rng: random.Random) -> dict | None:
        """Select a cross-augmentation dataset group by weight."""
        if not self.cross_dataset_groups:
            return None
        roll = rng.random()
        for group, cumulative_prob in zip(
            self.cross_dataset_groups, self._cross_cumulative_probs
        ):
            if roll < cumulative_prob:
                return group
        return self.cross_dataset_groups[-1]

    def _get_stem_note_instrument_ids(self, stem: dict[str, Any]) -> frozenset[int]:
        cached_ids = stem.get("note_instrument_ids")
        if cached_ids is not None:
            return frozenset(int(value) for value in cached_ids)

        note_instrument_ids = frozenset()
        try:
            with np.load(stem["npz_path"]) as data:
                if "note_instrument" in data.files:
                    note_instrument_ids = _valid_instrument_id_set(
                        data["note_instrument"]
                    )
        except Exception as e:
            logger.warning(
                "Failed to read note_instrument from %s: %s",
                stem.get("npz_path"),
                e,
            )

        stem["note_instrument_ids"] = note_instrument_ids
        return note_instrument_ids

    def _get_stem_instrument_keys(
        self,
        stem: dict[str, Any],
    ) -> frozenset[tuple[str, int | str]]:
        note_instrument_ids = self._get_stem_note_instrument_ids(stem)
        if note_instrument_ids:
            return frozenset(
                ("note", instrument_id) for instrument_id in note_instrument_ids
            )

        return frozenset({("name", _get_instrument_name(str(stem["stem_name"])))})

    def _load_window_note_groups_for_mix_specs(
        self,
        mix_specs: list[Any],
        *,
        window_start_ms: int,
        window_end_ms: int,
    ) -> list[WindowNotes]:
        note_groups: list[WindowNotes] = []
        for mix_spec in mix_specs:
            mix_stem = mix_spec.stem
            with np.load(mix_stem["npz_path"]) as data:
                start_ms = data["note_start_ms"]
                end_ms = data["note_end_ms"]
                pitch = data["note_pitch"]
                velocity = data["note_velocity"]
                instrument_ids = data.get("note_instrument", np.zeros_like(pitch))
                instrument_ids = mix_spec.override_instrument_ids(instrument_ids)

            pitch = remap_drum_pitch_array(
                pitch,
                instrument_ids,
                self.drum_pitch_aliases,
                drum_class_id=DRUM_CLASS_ID,
            )

            carry_in, body = split_window_notes(
                start_ms=start_ms,
                end_ms=end_ms,
                pitch=pitch,
                velocity=velocity,
                instrument=instrument_ids,
                window_start_ms=int(window_start_ms),
                window_end_ms=int(window_end_ms),
                clip_note_end_to_window=True,
            )
            note_groups.extend([carry_in, body])
        return note_groups

    def _cross_aug_density_allowed(
        self,
        current_note_groups: list[WindowNotes],
        candidate_note_groups: list[WindowNotes],
    ) -> bool:
        max_positive_pairs = int(self.max_cross_aug_positive_pairs)
        max_intervals = int(self.max_cross_aug_intervals)
        if max_positive_pairs <= 0 and max_intervals <= 0:
            return True

        merged_notes = concat_window_notes(*current_note_groups, *candidate_note_groups)
        interval_targets = build_interval_targets(
            semi_crf_version=self.semi_crf_version,
            num_pitch_slots=self.num_pitch_slots,
            active_start_ms=merged_notes.start_ms,
            active_end_ms=merged_notes.end_ms,
            active_pitch=merged_notes.pitch,
            active_instrument=merged_notes.instrument,
            active_has_onset=merged_notes.has_onset,
            active_has_offset=merged_notes.has_offset,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            num_frames=self.model_frames,
        )
        positive_pair_count = (
            len(interval_targets.positive_pair_ids)
            if self.semi_crf_version == "v2"
            else sum(bool(track) for track in interval_targets.intervals)
        )
        interval_count = sum(len(track) for track in interval_targets.intervals)
        if max_positive_pairs > 0 and positive_pair_count > max_positive_pairs:
            return False
        if max_intervals > 0 and interval_count > max_intervals:
            return False
        return True

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rng = random.Random(self.seed + self.epoch * len(self.primary_song_names) + idx)

        # Select the source dataset group for this sample.
        selected_group = self._select_dataset_group(rng)
        if selected_group is self.dataset_groups[0]:
            # Primary dataset: cover songs uniformly by idx.
            song_name = self.primary_song_names[idx]
        else:
            # Extra dataset: draw a random song from that group.
            song_name = rng.choice(selected_group["song_names"])

        base_stems = self.stems_by_song[song_name]

        # 1. Choose base stems from the selected song.
        #    allow_multi_stem_same_song=false forces a single stem.
        selected_base_stems = self.window_selector.select_base_stems(
            base_stems=base_stems,
            selected_group=selected_group,
            rng=rng,
        )

        # Track instruments already present in the base mix.
        base_instrument_keys: set[tuple[str, int | str]] = set()
        for stem in selected_base_stems:
            base_instrument_keys.update(self._get_stem_instrument_keys(stem))

        # 2. Choose a shared base window start.
        #    active_window_sampling=true biases toward windows with active notes.
        #
        window_start_ms = self.window_selector.select_base_window_start_ms(
            stems=selected_base_stems,
            selected_group=selected_group,
            rng=rng,
        )

        active_stems_with_offset = [
            (stem, window_start_ms, False) for stem in selected_base_stems
        ]

        # 3. Cross-song mix augmentation.
        if (
            rng.random() < self.p_cross_mix
            and len(self.all_stems) > 0
            and self.cross_dataset_groups
            and not selected_group.get("mask_instrument_loss", False)
        ):
            for j in range(self.max_cross_stems):
                # Continuation probability for adding the j-th extra stem.
                continue_prob = math.exp(-self.p_cross_mix_decay * j)
                if rng.random() >= continue_prob:
                    break

                max_retry = 10
                for _ in range(max_retry):
                    # Select the cross group by dataset weight.
                    cross_group = self._select_cross_dataset_group(rng)
                    if cross_group is None:
                        break
                    cross_song_name = rng.choice(cross_group["song_names"])
                    source_stems = cross_group.get(
                        "source_stems_by_song", {}
                    ).get(cross_song_name)
                    if not source_stems:
                        source_stems = self.stems_by_song[cross_song_name]
                    extra_stem = rng.choice(source_stems)

                    if extra_stem["song_name"] != song_name:
                        extra_instrument_keys = self._get_stem_instrument_keys(
                            extra_stem
                        )
                        # Do not add the same note-labeled instrument twice.
                        if base_instrument_keys.isdisjoint(extra_instrument_keys):
                            stem_window_start_ms = (
                                self.window_selector.select_stem_window_start_ms(
                                    stem=extra_stem,
                                    rng=rng,
                                )
                            )
                            active_stems_with_offset.append(
                                (extra_stem, stem_window_start_ms, True)
                            )
                            base_instrument_keys.update(extra_instrument_keys)
                            break

        # 4. Load and mix audio plus note labels.
        mixed_audio = np.zeros((2, self.window_frames), dtype=np.float32)
        note_groups: list[WindowNotes] = []
        mixed_stems_with_offset: list[tuple[dict[str, Any], int]] = []
        mixed_instrument_keys: set[tuple[str, int | str]] = set()
        density_guard_closed = False
        cross_aug_density_skipped = 0

        for stem, stem_window_start_ms, is_cross_aug in active_stems_with_offset:
            if is_cross_aug and density_guard_closed:
                cross_aug_density_skipped += 1
                continue

            stem_window_end_ms = stem_window_start_ms + self.window_ms
            # Harmony handling returns a mix plan; the loop below only applies it.
            mix_specs = self.harmony_manager.build_mix_specs(stem, rng)
            stem_note_groups = self._load_window_note_groups_for_mix_specs(
                mix_specs,
                window_start_ms=int(stem_window_start_ms),
                window_end_ms=int(stem_window_end_ms),
            )
            if is_cross_aug and not self._cross_aug_density_allowed(
                note_groups,
                stem_note_groups,
            ):
                density_guard_closed = True
                cross_aug_density_skipped += 1
                continue

            # Use one base gain for the main stem.
            # Harmony stems use gain offsets relative to that main gain.
            base_gain_db = rng.uniform(-6.0, 6.0)

            for mix_spec in mix_specs:
                mix_stem = mix_spec.stem

                # 1. Load and augment audio.
                audio = load_audio_window(
                    mix_stem["wav_path"],
                    sample_rate=self.sample_rate,
                    window_start_ms=stem_window_start_ms,
                    window_ms=self.window_ms,
                )
                stem_augmentor = self._get_stem_augmentor(mix_stem)
                if stem_augmentor is not None and rng.random() < self.p_augment:
                    audio = stem_augmentor(audio)

                # 2. Apply gain and add to the mixture.
                #    Harmony gain is relative to the main stem gain,
                #    not independently randomized.
                gain_db = base_gain_db + float(mix_spec.gain_db_offset)
                gain = 10.0 ** (gain_db / 20.0)
                mixed_audio += audio * gain
                mixed_instrument_keys.update(self._get_stem_instrument_keys(mix_stem))

            note_groups.extend(stem_note_groups)
            mixed_stems_with_offset.append((stem, int(stem_window_start_ms)))

        # 5. Random drum mix-in for drum robustness.
        has_drum = (
            DRUM_CLASS_ID is not None
            and ("note", DRUM_CLASS_ID) in mixed_instrument_keys
            or any(
                key_type == "name" and "drum" in str(value).lower()
                for key_type, value in mixed_instrument_keys
            )
        )
        if not has_drum and self.drum_files and rng.random() < self.p_drum_mix:
            drum_path = rng.choice(self.drum_files)
            try:
                info = sf.info(drum_path)
                duration_ms = int(info.frames / info.samplerate * 1000)
                max_start = max(0, duration_ms - self.window_ms)
                drum_start_ms = rng.randint(0, max_start) if max_start > 0 else 0

                drum_audio = load_audio_window(
                    drum_path,
                    sample_rate=self.sample_rate,
                    window_start_ms=drum_start_ms,
                    window_ms=self.window_ms,
                )

                if self.drum_augmentor is not None and rng.random() < self.p_augment:
                    drum_audio = self.drum_augmentor(drum_audio)

                gain = 10.0 ** (rng.uniform(-6.0, 6.0) / 20.0)
                mixed_audio += drum_audio * gain
            except Exception as e:
                logger.warning(f"Failed to load drum file {drum_path}: {e}")

        # Avoid clipping from additive mixing.
        peak = np.abs(mixed_audio).max()
        if peak > 1.0:
            mixed_audio /= peak

        audio_tensor = torch.from_numpy(mixed_audio).contiguous()

        merged_notes = concat_window_notes(*note_groups)

        frame_active_targets = build_frame_note_targets(
            active_start_ms=merged_notes.start_ms,
            active_end_ms=merged_notes.end_ms,
            active_pitch=merged_notes.pitch,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            num_frames=self.model_frames,
        )

        frame_instrument_targets = (
            build_frame_instrument_targets(
                active_start_ms=merged_notes.start_ms,
                active_end_ms=merged_notes.end_ms,
                active_pitch=merged_notes.pitch,
                active_instrument=merged_notes.instrument,
                sample_rate=self.sample_rate,
                hop_length=self.hop_length,
                num_frames=self.model_frames,
            )
            if self.semi_crf_version == "v1"
            else None
        )

        interval_targets = build_interval_targets(
            semi_crf_version=self.semi_crf_version,
            num_pitch_slots=self.num_pitch_slots,
            active_start_ms=merged_notes.start_ms,
            active_end_ms=merged_notes.end_ms,
            active_pitch=merged_notes.pitch,
            active_instrument=merged_notes.instrument,
            active_has_onset=merged_notes.has_onset,
            active_has_offset=merged_notes.has_offset,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            num_frames=self.model_frames,
        )
        # Compute the valid, non-padded audio length.
        max_valid_audio_ms = 0
        for stem, stem_window_start_ms in mixed_stems_with_offset:
            valid_ms = stem["duration_ms"] - stem_window_start_ms
            if valid_ms > max_valid_audio_ms:
                max_valid_audio_ms = valid_ms

        valid_audio_ms = max_valid_audio_ms
        if valid_audio_ms > self.window_ms:
            valid_audio_ms = self.window_ms
        if valid_audio_ms < 0:
            valid_audio_ms = 0
        valid_audio_frames_val = int(round(valid_audio_ms * self.sample_rate / 1000.0))

        # V1 masks only instrument loss; V2 excludes such groups during config loading.
        mask_instrument_loss = any(
            stem.get("mask_instrument_loss", False)
            for stem, _ in mixed_stems_with_offset
        )

        return {
            "song_name": song_name,
            "window_start_ms": window_start_ms,
            "audio": audio_tensor,
            "frame_active_targets": frame_active_targets,
            "frame_instrument_targets": frame_instrument_targets,
            "interval_targets": interval_targets,
            "valid_audio_frames": valid_audio_frames_val,
            "mask_instrument_loss": mask_instrument_loss,
            "cross_aug_density_skipped": int(cross_aug_density_skipped),
        }
