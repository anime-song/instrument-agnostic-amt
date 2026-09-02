"""Canonical stem classes shared by velocity training and inference."""

STEM_NAMES = ("bass", "drums", "guitar", "other", "piano", "vocals")
STEM_CLASS_BY_NAME = {name: index for index, name in enumerate(STEM_NAMES)}
UNKNOWN_STEM_CLASS = len(STEM_NAMES)

__all__ = ("STEM_CLASS_BY_NAME", "STEM_NAMES", "UNKNOWN_STEM_CLASS")
