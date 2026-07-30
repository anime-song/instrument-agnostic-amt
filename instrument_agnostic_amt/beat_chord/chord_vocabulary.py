from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def normalize_chord_quality_map(
    raw_quality_map: object,
    *,
    expected_root_chord_classes: int | None,
) -> dict[str, str]:
    """Validate and order the quality-index vocabulary used by the chord head."""

    if not isinstance(raw_quality_map, Mapping) or not raw_quality_map:
        raise ValueError("Chord quality map must be a non-empty JSON object")
    quality_map = {str(key): str(value) for key, value in raw_quality_map.items()}
    expected_keys = [str(index) for index in range(len(quality_map))]
    if set(quality_map) != set(expected_keys):
        raise ValueError("Chord quality map keys must be contiguous from zero")
    quality_map = {key: quality_map[key] for key in expected_keys}
    if quality_map[expected_keys[-1]] != "N":
        raise ValueError("The final chord quality class must be 'N'")

    root_chord_classes = 12 * (len(quality_map) - 1) + 1
    if (
        expected_root_chord_classes is not None
        and root_chord_classes != int(expected_root_chord_classes)
    ):
        raise ValueError(
            "Chord quality map does not match model_config: "
            f"{root_chord_classes} != {int(expected_root_chord_classes)}"
        )
    return quality_map


def load_chord_quality_map_json(
    path: str | Path,
    *,
    expected_root_chord_classes: int | None,
) -> dict[str, str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Chord quality JSON not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        raw_quality_map = json.load(file)
    return normalize_chord_quality_map(
        raw_quality_map,
        expected_root_chord_classes=expected_root_chord_classes,
    )
