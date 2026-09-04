"""Stem classes shared by velocity modeling and inference."""

STEM_NAMES = ("bass", "drums", "guitar", "other", "piano", "vocals")
STEM_CLASS_BY_NAME = {name: index for index, name in enumerate(STEM_NAMES)}
UNKNOWN_STEM_CLASS = len(STEM_NAMES)
