"""Drum pitch canonicalization for core AMT labels and MIDI export.

The model still keeps the full 88-pitch output space.  A pitch alias only
changes the training/export label at the drum boundary, so non-drum MIDI
notes with the same numeric pitch are never rewritten.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .constants import MAX_MIDI_PITCH, MIN_MIDI_PITCH

# These production/sample distinctions are not stable after stem separation,
# so use the acoustic-snare and Crash Cymbal 1 keys as canonical.
DEFAULT_DRUM_PITCH_ALIASES: dict[int, int] = {40: 38, 57: 49}

def parse_pitch_aliases(
    raw_aliases: object,
    *,
    field_name: str = "drum_pitch_aliases",
) -> dict[int, int]:
    """Validate a YAML/CLI pitch-alias mapping and normalize its keys."""
    if raw_aliases is None:
        return {}
    if not isinstance(raw_aliases, Mapping):
        raise ValueError(f"{field_name} must be a mapping of source pitch to target pitch")

    aliases: dict[int, int] = {}
    for raw_source, raw_target in raw_aliases.items():
        try:
            source = int(raw_source)
            target = int(raw_target)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} entries must contain integer MIDI pitches"
            ) from exc
        if not MIN_MIDI_PITCH <= source <= MAX_MIDI_PITCH:
            raise ValueError(
                f"{field_name} source pitch {source} is outside "
                f"{MIN_MIDI_PITCH}..{MAX_MIDI_PITCH}"
            )
        if not MIN_MIDI_PITCH <= target <= MAX_MIDI_PITCH:
            raise ValueError(
                f"{field_name} target pitch {target} is outside "
                f"{MIN_MIDI_PITCH}..{MAX_MIDI_PITCH}"
            )
        if source != target:
            aliases[source] = target

    # Reject cycles so an alias cannot depend on iteration order.
    for source in aliases:
        seen: set[int] = set()
        current = source
        while current in aliases:
            if current in seen:
                raise ValueError(f"{field_name} contains an alias cycle")
            seen.add(current)
            current = aliases[current]
    return aliases


def resolve_pitch_alias(pitch: int, aliases: Mapping[int, int]) -> int:
    """Resolve one pitch through an already validated alias mapping."""
    current = int(pitch)
    seen: set[int] = set()
    while current in aliases:
        if current in seen:
            raise ValueError("pitch alias mapping contains an alias cycle")
        seen.add(current)
        current = int(aliases[current])
    return current


def remap_drum_pitch_array(
    pitch: np.ndarray,
    instrument: np.ndarray,
    aliases: Mapping[int, int],
    *,
    drum_class_id: int | None,
) -> np.ndarray:
    """Return pitches with aliases applied only to notes labeled as drums."""
    remapped = np.asarray(pitch).copy()
    if not aliases or drum_class_id is None or remapped.size == 0:
        return remapped

    drum_mask = np.asarray(instrument).astype(np.int64, copy=False) == int(
        drum_class_id
    )
    if not np.any(drum_mask):
        return remapped

    # Resolve from the original value so chained aliases remain deterministic.
    source_values = remapped[drum_mask].astype(np.int64, copy=False)
    for index, value in enumerate(source_values.tolist()):
        source_values[index] = resolve_pitch_alias(int(value), aliases)
    remapped[drum_mask] = source_values.astype(remapped.dtype, copy=False)
    return remapped
