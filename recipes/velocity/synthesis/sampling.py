"""Deterministic target sampling for synthetic velocity data."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .config import SyntheticDataConfig


@dataclass(frozen=True)
class VelocitySample:
    target_velocity: np.ndarray
    filled_rank: np.ndarray
    rank_source: np.ndarray
    independently_randomized: np.ndarray
    style_center: float
    style_span: float


def derived_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable NumPy seed without relying on Python's randomized hash."""

    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(int(base_seed)).encode("utf-8"))
    for part in parts:
        digest.update(b"\0")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little", signed=False)


def _fill_missing_ranks(
    ranks: np.ndarray,
    valid: np.ndarray,
    track_index: np.ndarray,
    start_seconds: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    output = np.asarray(ranks, dtype=np.float64).copy()
    usable = np.asarray(valid, dtype=np.bool_) & np.isfinite(output)
    source = np.zeros(output.shape, dtype=np.int8)
    source[usable] = 2  # directly observed pseudo rank
    for track in np.unique(track_index):
        indices = np.flatnonzero(track_index == track)
        known = indices[usable[indices]]
        missing = indices[~usable[indices]]
        if not missing.size:
            continue
        if known.size >= 2:
            known_times = start_seconds[known]
            unique_times = np.unique(known_times)
            values_at_time = np.asarray(
                [
                    np.median(output[known[known_times == time]])
                    for time in unique_times
                ],
                dtype=np.float64,
            )
            output[missing] = np.interp(
                start_seconds[missing],
                unique_times,
                values_at_time,
            )
            source[missing] = 1  # interpolated within a track
        elif known.size == 1:
            output[missing] = output[known[0]]
            source[missing] = 1
        else:
            missing_times, time_inverse = np.unique(
                start_seconds[missing],
                return_inverse=True,
            )
            values_at_time = rng.beta(2.0, 2.0, size=missing_times.size)
            output[missing] = values_at_time[time_inverse]
            source[missing] = 0  # no pseudo evidence; stochastic fallback
    unresolved = ~np.isfinite(output)
    if np.any(unresolved):
        output[unresolved] = rng.beta(2.0, 2.0, size=int(unresolved.sum()))
        source[unresolved] = 0
    return np.clip(output, 0.0, 1.0), source


def sample_note_velocities(
    *,
    pseudo_rank: np.ndarray,
    pseudo_valid: np.ndarray,
    track_index: np.ndarray,
    note_start_seconds: np.ndarray,
    rng: np.random.Generator,
    config: SyntheticDataConfig,
) -> VelocitySample:
    ranks = np.asarray(pseudo_rank, dtype=np.float64)
    valid = np.asarray(pseudo_valid, dtype=np.bool_)
    tracks = np.asarray(track_index, dtype=np.int64)
    starts = np.asarray(note_start_seconds, dtype=np.float64)
    if (
        ranks.ndim != 1
        or valid.shape != ranks.shape
        or tracks.shape != ranks.shape
        or starts.shape != ranks.shape
    ):
        raise ValueError(
            "pseudo rank, valid mask, track index and note starts "
            "must have equal 1D shapes"
        )
    filled, rank_source = _fill_missing_ranks(ranks, valid, tracks, starts, rng)
    center = float(rng.uniform(config.velocity_center_min, config.velocity_center_max))
    span = float(rng.uniform(config.velocity_span_min, config.velocity_span_max))
    continuous = center + (filled - 0.5) * span
    if config.velocity_jitter_std > 0.0:
        continuous += rng.normal(0.0, config.velocity_jitter_std, size=filled.size)

    independently_randomized = (
        rng.random(filled.size) < config.independent_velocity_probability
    )
    if np.any(independently_randomized):
        continuous[independently_randomized] = rng.integers(
            config.velocity_min,
            config.velocity_max + 1,
            size=int(independently_randomized.sum()),
        )
    target = np.clip(
        np.rint(continuous),
        config.velocity_min,
        config.velocity_max,
    ).astype(np.int16)
    return VelocitySample(
        target_velocity=target,
        filled_rank=filled.astype(np.float32),
        rank_source=rank_source,
        independently_randomized=independently_randomized.astype(np.bool_),
        style_center=center,
        style_span=span,
    )


def sample_stem_gains(
    base_relative_levels_db: Mapping[str, float | None],
    *,
    rng: np.random.Generator,
    config: SyntheticDataConfig,
) -> dict[str, float]:
    names = sorted(base_relative_levels_db)
    if not names:
        return {}
    if not config.use_gain_augmentation:
        return {name: 0.0 for name in names}
    base = np.asarray(
        [
            0.0
            if base_relative_levels_db[name] is None
            else float(base_relative_levels_db[name])
            for name in names
        ],
        dtype=np.float64,
    )
    base = np.clip(base, -config.gain_clip_db, config.gain_clip_db)
    jitter = rng.normal(0.0, config.gain_jitter_std_db, size=len(names))
    gains = base + jitter
    gains -= float(np.median(gains))
    gains = np.clip(gains, -config.gain_clip_db, config.gain_clip_db)
    return {name: float(value) for name, value in zip(names, gains)}
