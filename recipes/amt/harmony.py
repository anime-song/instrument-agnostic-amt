from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from instrument_agnostic_amt.taxonomy.instrument_classes import (
    get_instrument_class_id_by_name,
)


def _normalize_harmony_pitch_shifts(
    values: list[Any] | tuple[Any, ...],
) -> tuple[int, ...]:
    """Normalize YAML harmony pitch-shift values into an ordered int tuple."""
    normalized: list[int] = []
    for value in values:
        semitone = int(value)
        if semitone == 0 or semitone in normalized:
            continue
        normalized.append(semitone)
    return tuple(normalized)


@dataclass(frozen=True)
class HarmonyAugmentationConfig:
    """Configuration for pseudo-harmony stem augmentation."""

    pitch_shifts: tuple[int, ...] = ()
    instrument_class_name: str | None = None
    instrument_class_id: int | None = None
    gain_db: float = 0.0

    @property
    def enabled(self) -> bool:
        """Whether at least one harmony candidate is configured."""
        return bool(self.pitch_shifts)

    def describe(self) -> str:
        """Return a compact string for logging."""
        if not self.enabled:
            return "none"

        parts = [f"shifts={self.pitch_shifts}"]
        if self.instrument_class_name is not None:
            parts.append(f"class={self.instrument_class_name}")
        if self.gain_db != 0.0:
            parts.append(f"gain_db={self.gain_db:+.1f}")
        return ", ".join(parts)


@dataclass(frozen=True)
class StemMixSpec:
    """How one stem should be mixed into the current sample."""

    stem: dict[str, Any]
    # Gain offset relative to the main stem.
    # Harmony stems are usually mixed quieter than the main stem.
    gain_db_offset: float = 0.0
    instrument_override_id: int | None = None

    def override_instrument_ids(self, instrument_ids: np.ndarray) -> np.ndarray:
        """Override note instrument labels when a harmony class is configured."""
        if self.instrument_override_id is None or instrument_ids.size == 0:
            return instrument_ids
        return np.full_like(
            instrument_ids,
            fill_value=int(self.instrument_override_id),
        )


def _build_harmony_augmentation_config(
    dataset_entry: dict[str, Any] | None,
) -> HarmonyAugmentationConfig:
    """Build harmony settings from one dataset YAML entry."""
    if dataset_entry is None:
        return HarmonyAugmentationConfig()

    instrument_class_name = dataset_entry.get("harmony_instrument_class_name")
    if instrument_class_name is not None:
        instrument_class_name = str(instrument_class_name).strip() or None

    instrument_class_id = None
    if instrument_class_name is not None:
        instrument_class_id = get_instrument_class_id_by_name(instrument_class_name)

    return HarmonyAugmentationConfig(
        pitch_shifts=_normalize_harmony_pitch_shifts(
            dataset_entry.get("harmony_pitch_shifts", []) or []
        ),
        instrument_class_name=instrument_class_name,
        instrument_class_id=instrument_class_id,
        gain_db=float(dataset_entry.get("harmony_gain_db", 0.0)),
    )


class HarmonyAugmentationManager:
    """Select pseudo-harmony stems and their mix metadata."""

    def __init__(
        self,
        *,
        dataset_groups_by_name: dict[str, dict[str, Any]],
        pitch_shift_stems_by_group: dict[tuple[str, str], dict[int, dict[str, Any]]],
    ) -> None:
        self.dataset_groups_by_name = dataset_groups_by_name
        self.pitch_shift_stems_by_group = pitch_shift_stems_by_group

    def _get_config(self, stem: dict[str, Any]) -> HarmonyAugmentationConfig:
        """Return the harmony config for the dataset group of this stem."""
        group_name = str(stem.get("dataset_group_name", "main"))
        group = self.dataset_groups_by_name.get(group_name)
        if group is None:
            return HarmonyAugmentationConfig()
        config = group.get("harmony_config")
        if not isinstance(config, HarmonyAugmentationConfig):
            return HarmonyAugmentationConfig()
        return config

    def _resolve_harmony_stem(
        self,
        stem: dict[str, Any],
        harmony_shift: int,
    ) -> dict[str, Any] | None:
        """Find a pitch-shifted sibling stem that can serve as harmony."""
        group_name = str(stem.get("dataset_group_name", "main"))
        pitch_shift_group_key = str(stem.get("pitch_shift_group_key", ""))
        pitch_shift_group = self.pitch_shift_stems_by_group.get(
            (group_name, pitch_shift_group_key)
        )
        if not pitch_shift_group:
            return None

        current_pitch_shift = int(stem.get("pitch_shift_value", 0))
        for resolved_shift in (int(harmony_shift), -int(harmony_shift)):
            target_pitch_shift = current_pitch_shift + resolved_shift
            if target_pitch_shift == current_pitch_shift:
                continue
            harmony_stem = pitch_shift_group.get(target_pitch_shift)
            if harmony_stem is not None:
                return harmony_stem
        return None

    def _select_harmony_stem(
        self,
        stem: dict[str, Any],
        harmony_config: HarmonyAugmentationConfig,
        rng: random.Random,
    ) -> dict[str, Any] | None:
        """Choose one configured harmony candidate when available."""
        if not harmony_config.enabled:
            return None

        candidate_shifts = list(harmony_config.pitch_shifts)
        rng.shuffle(candidate_shifts)
        for harmony_shift in candidate_shifts:
            harmony_stem = self._resolve_harmony_stem(stem, harmony_shift)
            if harmony_stem is not None:
                return harmony_stem
        return None

    def build_mix_specs(
        self,
        stem: dict[str, Any],
        rng: random.Random,
    ) -> list[StemMixSpec]:
        """
        Return the original stem plus an optional pseudo-harmony stem mix plan.
        """
        mix_specs = [StemMixSpec(stem=stem)]
        harmony_config = self._get_config(stem)
        harmony_stem = self._select_harmony_stem(
            stem,
            harmony_config=harmony_config,
            rng=rng,
        )
        if harmony_stem is None:
            return mix_specs

        mix_specs.append(
            StemMixSpec(
                stem=harmony_stem,
                gain_db_offset=harmony_config.gain_db,
                instrument_override_id=harmony_config.instrument_class_id,
            )
        )
        return mix_specs
