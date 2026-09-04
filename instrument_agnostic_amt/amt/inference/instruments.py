"""Instrument filters used during AMT decoding."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

from ...taxonomy.instrument_classes import INSTRUMENT_CLASSES


STEM_INSTRUMENT_CLASSES: dict[str, tuple[str, ...]] = {
    "drums": ("drums", "wind_chimes", "timpani"),
    "bass": (
        "acoustic_bass",
        "electric_bass",
        "slap_bass",
        "synth_bass",
    ),
    "vocals": ("melody", "vocal_harmony", "choir"),
    "guitar": (
        "acoustic_guitar",
        "electric_guitar_clean",
        "electric_guitar_muted",
        "distorted_guitar",
        "guitar_harmonics",
        "ethnic",
    ),
    "piano": ("piano", "electric_piano", "plucked_keyboard"),
    "other": (
        "accordion_family",
        "brass",
        "chromatic_percussion",
        "ethnic",
        "flute_pipe",
        "harmonica",
        "orchestra_hit",
        "orchestral_harp",
        "orchestral_woodwind",
        "organ",
        "percussive_fx",
        "pizzicato_strings",
        "plucked_keyboard",
        "sax",
        "sound_fx",
        "strings",
        "synth_fx",
        "synth_lead",
        "synth_pad",
        "timpani",
        "wind_chimes",
    ),
}


def normalize_instrument_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def instrument_class_ids_from_names(
    class_names: Iterable[str],
) -> tuple[int, ...]:
    available_names = {
        normalize_instrument_class_name(class_name): class_id
        for class_id, class_name in enumerate(INSTRUMENT_CLASSES)
    }
    class_ids: list[int] = []
    unknown_names: list[str] = []
    for raw_name in class_names:
        normalized_name = normalize_instrument_class_name(raw_name)
        if normalized_name not in available_names:
            unknown_names.append(raw_name)
            continue
        class_id = int(available_names[normalized_name])
        if class_id not in class_ids:
            class_ids.append(class_id)

    if unknown_names:
        available = ", ".join(sorted(available_names))
        raise ValueError(
            f"Unknown instrument class(es): {', '.join(unknown_names)}. "
            f"Available classes: {available}"
        )
    if not class_ids:
        raise ValueError("At least one allowed instrument class is required.")
    return tuple(class_ids)


def parse_allowed_instrument_args(entries: list[str]) -> tuple[int, ...] | None:
    if not entries:
        return None
    class_names = [
        class_name
        for entry in entries
        for class_name in entry.split(",")
        if class_name.strip()
    ]
    return instrument_class_ids_from_names(class_names)


def resolve_stem_instrument_class_ids(stem_name: str) -> tuple[int, ...] | None:
    """Return plausible instrument IDs for a six-stem separator output."""
    normalized_stem_name = normalize_instrument_class_name(stem_name)
    aliases = (
        ("drum", "drums"),
        ("bass", "bass"),
        ("vocal", "vocals"),
        ("guitar", "guitar"),
        ("piano", "piano"),
        ("other", "other"),
    )
    for fragment, canonical_name in aliases:
        if fragment in normalized_stem_name:
            return instrument_class_ids_from_names(
                STEM_INSTRUMENT_CLASSES[canonical_name]
            )
    return None


def filter_supported_instrument_class_ids(
    allowed_instrument_ids: tuple[int, ...] | None,
    *,
    num_model_classes: int,
) -> tuple[int, ...] | None:
    """Drop instrument IDs not represented by the loaded checkpoint head."""
    if allowed_instrument_ids is None:
        return None
    if num_model_classes <= 0:
        raise ValueError("num_model_classes must be positive")

    supported_ids = tuple(
        class_id
        for class_id in allowed_instrument_ids
        if 0 <= class_id < num_model_classes
    )
    unsupported_ids = tuple(
        class_id
        for class_id in allowed_instrument_ids
        if class_id not in supported_ids
    )
    if unsupported_ids:
        unsupported_names = [
            INSTRUMENT_CLASSES[class_id]
            if 0 <= class_id < len(INSTRUMENT_CLASSES)
            else str(class_id)
            for class_id in unsupported_ids
        ]
        warnings.warn(
            f"Checkpoint instrument head has {num_model_classes} classes; "
            "ignoring unsupported candidate(s): "
            f"{', '.join(unsupported_names)}",
            RuntimeWarning,
            stacklevel=2,
        )
    if not supported_ids:
        raise ValueError(
            "None of the allowed instrument IDs are supported by the checkpoint: "
            f"num_model_classes={num_model_classes}, ids={allowed_instrument_ids}"
        )
    return supported_ids


def effective_instrument_ids(
    *,
    num_model_classes: int,
    allowed_instrument_ids: tuple[int, ...] | None,
    instrument_filter_id: int | None,
) -> tuple[int, ...]:
    """Resolve the exact decoder candidate set, with a single filter taking priority."""
    requested = (
        (int(instrument_filter_id),)
        if instrument_filter_id is not None
        else allowed_instrument_ids
    )
    if requested is None:
        return tuple(range(int(num_model_classes)))
    if not requested:
        raise ValueError("allowed_instrument_ids must not be empty")
    if any(
        instrument_id < 0 or instrument_id >= int(num_model_classes)
        for instrument_id in requested
    ):
        raise ValueError(
            "instrument filter contains an ID outside the model output: "
            f"num_classes={num_model_classes}, ids={requested}"
        )
    return tuple(dict.fromkeys(int(item) for item in requested))
