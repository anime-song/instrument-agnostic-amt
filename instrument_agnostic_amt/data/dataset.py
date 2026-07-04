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

from .audio import compute_model_frames, load_audio_window
from .augmentation import AudioAugmentor
from .harmony import (
    HarmonyAugmentationConfig,
    HarmonyAugmentationManager,
    _build_harmony_augmentation_config,
)
from .notes import (
    WindowNotes,
    choose_condition_instrument_id,
    concat_window_notes,
    filter_window_notes_by_instrument,
    split_window_notes,
)
from .sampling import StemWindowSelector
from .targets import build_frame_note_targets, build_pitch_interval_targets

logger = logging.getLogger(__name__)

PITCH_SHIFT_SUFFIX_RE = re.compile(r"^(?P<base>.+?)_pitch_(?P<shift>-?\d+)$")


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


class StemDataset(Dataset):
    """
    Load stem audio/MIDI pairs and build V2 conditioned AMT samples.
    Supports weighted multi-dataset sampling from a dataset_config YAML file.
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
        num_pitch_slots: int = 1,
        p_intra_drop: float = 0.2,
        p_cross_mix: float = 0.1,
        p_cross_mix_decay: float = 0.3,
        max_cross_stems: int = 5,
        p_augment: float = 0.5,
        ir_folder: str | Path | None = None,
        noise_folder: str | Path | None = None,
        drum_folder: str | Path | None = None,
        p_drum_mix: float = 0.1,
        condition_negative_prob: float = 0.25,
        seed: int = 42,
    ):
        self.window_ms = int(window_ms)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.sample_rate = int(sample_rate)
        self.num_pitch_slots = max(1, int(num_pitch_slots))
        # Probability of dropping stems from the same song.
        self.p_intra_drop = float(p_intra_drop)
        # Probability of mixing stems from other songs.
        self.p_cross_mix = float(p_cross_mix)
        # Decay factor when adding multiple cross-song stems.
        self.p_cross_mix_decay = float(p_cross_mix_decay)
        self.max_cross_stems = int(max_cross_stems)
        self.p_augment = float(p_augment)
        self.seed = int(seed)
        self.epoch = 0
        self.ir_folder = ir_folder
        self.noise_folder = noise_folder
        self.group_augmentors: dict[str, AudioAugmentor | None] = {}
        self.drum_augmentor = self._build_audio_augmentor(distortion_augmentations=None)

        self.p_drum_mix = float(p_drum_mix)
        self.condition_negative_prob = float(condition_negative_prob)
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

        # Dataset groups: [{name, song_names, weight}, ...].
        self.dataset_groups: list[dict] = []

        if dataset_config_path is not None and Path(dataset_config_path).exists():
            self._load_config(dataset_config_path)
        else:
            # Without a YAML config, train from a single manifest.
            self._load_manifest(manifest_path)
            primary_songs = list(self.stems_by_song.keys())
            self.dataset_groups.append(
                {
                    "name": "main",
                    "song_names": primary_songs,
                    "weight": 1.0,
                    "use_for_cross_aug": True,
                    "active_window_sampling": False,
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
                "No usable dataset groups found for V2 conditioned training"
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
                f"weight={group['weight']}, prob={probability:.1f}%, "
                f"cross_aug={group.get('use_for_cross_aug', True)}, "
                f"active_window={group.get('active_window_sampling', False)}, "
                f"multi_stem_same_song={group.get('allow_multi_stem_same_song', True)}, "
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

            # Track existing songs so we can collect only newly loaded ones.
            dataset_name = dataset_entry.get("name", manifest_rel)
            mask_inst = bool(dataset_entry.get("mask_instrument_loss", False))
            if mask_inst:
                logger.info(
                    "Skipping dataset group %s for V2 conditioned training because mask_instrument_loss=true",
                    dataset_entry.get("name", manifest_rel),
                )
                continue
            distortion_augmentations = tuple(
                dataset_entry.get("distortion_augmentations", []) or []
            )
            harmony_config = _build_harmony_augmentation_config(dataset_entry)
            existing_songs = set(self.stems_by_song.keys())
            self._load_manifest(
                manifest_full,
                song_name_prefix=dataset_name,
                mask_instrument_loss=mask_inst,
                dataset_group_name=str(dataset_name),
            )
            new_songs = [
                name for name in self.stems_by_song if name not in existing_songs
            ]

            self.dataset_groups.append(
                {
                    "name": dataset_entry.get("name", manifest_rel),
                    "song_names": new_songs,
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
                    "mask_instrument_loss": bool(
                        dataset_entry.get("mask_instrument_loss", False)
                    ),
                    "distortion_augmentations": distortion_augmentations,
                    "harmony_config": harmony_config,
                }
            )
            self.group_augmentors[str(dataset_name)] = self._build_audio_augmentor(
                distortion_augmentations=distortion_augmentations
            )

    def _load_manifest(
        self,
        manifest_path: str | Path,
        song_name_prefix: str = "",
        mask_instrument_loss: bool = False,
        dataset_group_name: str = "main",
    ):
        """Load a manifest CSV into stems_by_song and all_stems."""
        manifest_path = Path(manifest_path)
        manifest_dir = manifest_path.parent
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
                # Prefix song names to avoid collisions between datasets.
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
                    "pitch_shift_value": pitch_shift_value,
                    "pitch_shift_group_key": pitch_shift_group_key,
                }
                self.stems_by_song[song_name].append(stem_info)
                self.all_stems.append(stem_info)
                pitch_shift_group = self.pitch_shift_stems_by_group[
                    (str(dataset_group_name), pitch_shift_group_key)
                ]
                pitch_shift_group[pitch_shift_value] = stem_info

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
        base_instruments = {
            _get_instrument_name(stem["stem_name"]) for stem in selected_base_stems
        }

        # 2. Choose a shared base window start.
        #    active_window_sampling=true biases toward windows with active notes.
        #
        window_start_ms = self.window_selector.select_base_window_start_ms(
            stems=selected_base_stems,
            selected_group=selected_group,
            rng=rng,
        )

        active_stems_with_offset = [
            (stem, window_start_ms) for stem in selected_base_stems
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
                    extra_stem = rng.choice(self.stems_by_song[cross_song_name])

                    if extra_stem["song_name"] != song_name:
                        extra_inst = _get_instrument_name(extra_stem["stem_name"])
                        # Do not add the same instrument twice.
                        if extra_inst not in base_instruments:
                            stem_window_start_ms = (
                                self.window_selector.select_stem_window_start_ms(
                                    stem=extra_stem,
                                    rng=rng,
                                )
                            )
                            active_stems_with_offset.append(
                                (extra_stem, stem_window_start_ms)
                            )
                            base_instruments.add(extra_inst)
                            break

        # 4. Load and mix audio plus note labels.
        mixed_audio = np.zeros((2, self.window_frames), dtype=np.float32)
        note_groups = []

        for stem, stem_window_start_ms in active_stems_with_offset:
            stem_window_end_ms = stem_window_start_ms + self.window_ms
            # Harmony handling returns a mix plan; the loop below only applies it.
            #
            mix_specs = self.harmony_manager.build_mix_specs(stem, rng)
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

                # 3. Load MIDI-derived labels for the same window.
                with np.load(mix_stem["npz_path"]) as data:
                    start_ms = data["note_start_ms"]
                    end_ms = data["note_end_ms"]
                    pitch = data["note_pitch"]
                    velocity = data["note_velocity"]
                    instrument_ids = data.get("note_instrument", np.zeros_like(pitch))
                    instrument_ids = mix_spec.override_instrument_ids(instrument_ids)

                carry_in, body = split_window_notes(
                    start_ms=start_ms,
                    end_ms=end_ms,
                    pitch=pitch,
                    velocity=velocity,
                    instrument=instrument_ids,
                    window_start_ms=stem_window_start_ms,
                    window_end_ms=stem_window_end_ms,
                    clip_note_end_to_window=True,
                )
                note_groups.extend([carry_in, body])

        # 5. Random drum mix-in for drum robustness.
        has_drum = any("drum" in inst.lower() for inst in base_instruments)
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

        # V2: choose one target instrument condition and build AMT targets only for it.
        merged_notes = concat_window_notes(*note_groups)
        condition_instrument_id = choose_condition_instrument_id(
            merged_notes,
            rng=rng,
            negative_prob=self.condition_negative_prob,
        )
        target_notes = filter_window_notes_by_instrument(
            merged_notes,
            condition_instrument_id,
        )

        frame_active_targets = build_frame_note_targets(
            active_start_ms=target_notes.start_ms,
            active_end_ms=target_notes.end_ms,
            active_pitch=target_notes.pitch,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            num_frames=self.model_frames,
        )

        interval_targets = build_pitch_interval_targets(
            active_start_ms=target_notes.start_ms,
            active_end_ms=target_notes.end_ms,
            active_pitch=target_notes.pitch,
            active_instrument=target_notes.instrument,
            active_has_onset=target_notes.has_onset,
            active_has_offset=target_notes.has_offset,
            sample_rate=self.sample_rate,
            hop_length=self.hop_length,
            num_frames=self.model_frames,
            num_pitch_slots=self.num_pitch_slots,
        )

        # Compute the valid, non-padded audio length.
        max_valid_audio_ms = 0
        for stem, stem_window_start_ms in active_stems_with_offset:
            valid_ms = stem["duration_ms"] - stem_window_start_ms
            if valid_ms > max_valid_audio_ms:
                max_valid_audio_ms = valid_ms

        valid_audio_ms = max_valid_audio_ms
        if valid_audio_ms > self.window_ms:
            valid_audio_ms = self.window_ms
        if valid_audio_ms < 0:
            valid_audio_ms = 0
        valid_audio_frames_val = int(round(valid_audio_ms * self.sample_rate / 1000.0))

        # V2 excludes mask_instrument_loss=true groups while loading configs.
        mask_instrument_loss = any(
            stem.get("mask_instrument_loss", False)
            for stem, _ in active_stems_with_offset
        )

        return {
            "song_name": song_name,
            "window_start_ms": window_start_ms,
            "audio": audio_tensor,
            "frame_active_targets": frame_active_targets,
            "interval_targets": interval_targets,
            "valid_audio_frames": valid_audio_frames_val,
            "condition_instrument_id": int(condition_instrument_id),
            "mask_instrument_loss": mask_instrument_loss,
        }
