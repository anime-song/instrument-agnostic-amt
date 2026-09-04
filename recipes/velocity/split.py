"""Deterministic song-level splits for velocity datasets."""

from __future__ import annotations

import hashlib
from typing import Literal


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
