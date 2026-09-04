"""Synthetic modulation augmentation for chord training."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

NOTE_TO_INDEX = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}
SHARP_NOTE_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
CHORD_INTERVALS_BY_FAMILY = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "dominant": (0, 4, 7, 10),
    "diminished": (0, 3, 6, 9),
}
FAMILY_PRIORITY_BY_SHIFT = {
    1: {
        "ii_v_i": 0,
        "dominant_to_tonic": 1,
        "secondary_dominant": 2,
        "common_tone_diminished": 3,
        "common_chord_pivot": 4,
    },
    2: {
        "ii_v_i": 0,
        "dominant_to_tonic": 1,
        "secondary_dominant": 2,
        "common_chord_pivot": 3,
        "common_tone_diminished": 4,
    },
    3: {
        "ii_v_i": 0,
        "dominant_to_tonic": 1,
        "secondary_dominant": 2,
        "common_chord_pivot": 2,
        "common_tone_diminished": 4,
    },
    4: {
        "ii_v_i": 0,
        "dominant_to_tonic": 1,
        "secondary_dominant": 2,
        "common_chord_pivot": 2,
        "common_tone_diminished": 4,
    },
    5: {
        "common_chord_pivot": 0,
        "ii_v_i": 1,
        "dominant_to_tonic": 2,
        "secondary_dominant": 3,
        "common_tone_diminished": 4,
    },
    6: {
        "ii_v_i": 0,
        "dominant_to_tonic": 1,
        "secondary_dominant": 2,
        "common_tone_diminished": 3,
        "common_chord_pivot": 4,
    },
}
FAMILY_TIEBREAKER = (
    "ii_v_i",
    "dominant_to_tonic",
    "secondary_dominant",
    "common_chord_pivot",
    "common_tone_diminished",
)
DEGREE_POSITION = {
    "I": 1,
    "i": 1,
    "ii": 2,
    "III": 3,
    "iii": 3,
    "IV": 4,
    "iv": 4,
    "V": 5,
    "vi": 6,
    "VI": 6,
    "VII": 7,
    "vii": 7,
}
CADENTIAL_DEGREES = {"I", "i", "V"}


@dataclass(frozen=True)
class ModulationAugmentConfig:
    """人工転調 augmentation の設定をまとめる。"""

    prob: float = 0.25
    force_boundary_for_key: bool = True
    force_boundary_for_chord: bool = True
    allowed_shifts: tuple[int, ...] = (-5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6)

    def __post_init__(self) -> None:
        # バグを早めに見つけるため、設定の範囲はここで固定する。
        if not 0.0 <= float(self.prob) <= 1.0:
            raise ValueError("prob must be in [0, 1]")
        if not self.allowed_shifts:
            raise ValueError("allowed_shifts must not be empty")
        if any(int(shift) == 0 or abs(int(shift)) > 6 for shift in self.allowed_shifts):
            raise ValueError("allowed_shifts must contain non-zero shifts in [-6, 6]")


@dataclass(frozen=True)
class WindowHarmonyContext:
    """1つの学習窓に対する harmonic 判定入力。"""

    source_item: dict[str, Any]
    window_start_sec: float
    window_end_sec: float
    sample_rate: int
    hop_length: int


@dataclass(frozen=True)
class ModulationCandidate:
    """採用可能な人工転調候補を表す。"""

    shift: int
    splice_time_sec: float
    splice_frame: int
    family: str
    priority: int
    support_duration_sec: float
    source_key: str
    target_key: str


@dataclass(frozen=True)
class KeySpec:
    """機能判定用にキーの tonic と mode を保持する。"""

    tonic_pc: int
    mode: str


def enumerate_modulation_candidates(
    context: WindowHarmonyContext,
    config: ModulationAugmentConfig,
) -> list[ModulationCandidate]:
    """窓内の全コード境界と全 shift を総当たりして候補を列挙する。"""
    source_item = context.source_item
    source_chords = list(source_item.get("chords", []))
    source_keys = list(source_item.get("keys", []))
    if len(source_chords) < 2 or not source_keys:
        return []

    candidates: list[ModulationCandidate] = []
    for boundary_index in range(1, len(source_chords)):
        boundary_time_sec = float(source_chords[boundary_index]["start_time"])
        if not (context.window_start_sec < boundary_time_sec < context.window_end_sec):
            continue

        prev2 = source_chords[boundary_index - 2] if boundary_index >= 2 else None
        prev1 = source_chords[boundary_index - 1]
        source_key = _find_active_key_name(
            source_keys,
            boundary_time_sec,
            side="left",
        )
        if source_key is None:
            continue

        right_key = _find_active_key_name(
            source_keys,
            boundary_time_sec,
            side="right",
        )
        if right_key is None:
            continue

        for shift in sorted({int(value) for value in config.allowed_shifts}):
            next1 = _transpose_chord_segment(source_chords[boundary_index], shift)
            next2 = (
                _transpose_chord_segment(source_chords[boundary_index + 1], shift)
                if boundary_index + 1 < len(source_chords)
                else None
            )
            target_key = transpose_key_name(right_key, shift)
            if _parse_key_to_relative_major_class(
                source_key
            ) == _parse_key_to_relative_major_class(target_key):
                continue

            candidate = _build_modulation_candidate(
                shift=int(shift),
                boundary_time_sec=boundary_time_sec,
                window_start_sec=context.window_start_sec,
                prev2=prev2,
                prev1=prev1,
                next1=next1,
                next2=next2,
                source_key=source_key,
                target_key=target_key,
                sample_rate=context.sample_rate,
                hop_length=context.hop_length,
            )
            if candidate is None:
                splice_frame = int(
                    round(
                        (boundary_time_sec - context.window_start_sec)
                        * float(context.sample_rate)
                        / float(context.hop_length)
                    )
                )
                candidate = ModulationCandidate(
                    shift=shift,
                    splice_time_sec=boundary_time_sec,
                    splice_frame=splice_frame,
                    family="direct_transposition",
                    priority=5,
                    support_duration_sec=_support_duration(prev1, next1),
                    source_key=source_key,
                    target_key=target_key,
                )
            candidates.append(candidate)

    return candidates


def choose_modulation_candidate(
    candidates: Sequence[ModulationCandidate],
    rng: random.Random,
) -> ModulationCandidate | None:
    """優先度順に絞り込み、最後だけランダムに 1 候補選ぶ。"""
    if not candidates:
        return None

    best_priority = min(candidate.priority for candidate in candidates)
    priority_group = [
        candidate for candidate in candidates if candidate.priority == best_priority
    ]
    best_support = max(candidate.support_duration_sec for candidate in priority_group)
    support_group = [
        candidate
        for candidate in priority_group
        if abs(candidate.support_duration_sec - best_support) < 1e-6
    ]
    return rng.choice(support_group)


def transpose_note_name(note_name: str, semitones: int) -> str:
    """Transpose a chord-label note without changing its pitch class meaning."""
    if note_name == "N":
        return "N"
    pitch_class = _note_name_to_index(str(note_name))
    if pitch_class is None:
        return str(note_name)
    return SHARP_NOTE_NAMES[(pitch_class + int(semitones)) % 12]


def transpose_key_name(key_name: str, semitones: int) -> str:
    if key_name == "N":
        return "N"
    is_minor = str(key_name).endswith("m")
    tonic = str(key_name)[:-1] if is_minor else str(key_name)
    transposed = transpose_note_name(tonic, semitones)
    return f"{transposed}m" if is_minor else transposed


def apply_modulation_to_roll(
    roll: torch.Tensor,
    chord_segments: Sequence[dict[str, Any]],
    key_segments: Sequence[dict[str, Any]],
    candidate: ModulationCandidate,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, Any]]]:
    """Transpose the MIDI roll and all harmony labels after one boundary."""
    if roll.dim() != 3:
        raise ValueError("roll must have shape [C, T, P]")

    splice_frame = max(0, min(int(candidate.splice_frame), int(roll.shape[1])))
    output = roll.clone()
    output[:, splice_frame:, :] = _shift_roll_pitch(
        roll[:, splice_frame:, :],
        int(candidate.shift),
    )
    chords = _transpose_segment_tail(
        chord_segments,
        splice_time_sec=float(candidate.splice_time_sec),
        semitones=int(candidate.shift),
        kind="chord",
    )
    keys = _transpose_segment_tail(
        key_segments,
        splice_time_sec=float(candidate.splice_time_sec),
        semitones=int(candidate.shift),
        kind="key",
    )
    return output, chords, keys


def _shift_roll_pitch(roll: torch.Tensor, semitones: int) -> torch.Tensor:
    shifted = torch.zeros_like(roll)
    pitch_bins = int(roll.shape[-1])
    semitones = int(semitones)
    if semitones == 0:
        return roll.clone()
    if abs(semitones) >= pitch_bins:
        return shifted
    if semitones > 0:
        shifted[..., semitones:] = roll[..., : pitch_bins - semitones]
    else:
        shifted[..., : pitch_bins + semitones] = roll[..., -semitones:]
    return shifted


def _transpose_chord_segment(
    segment: dict[str, Any],
    semitones: int,
) -> dict[str, Any]:
    transposed = dict(segment)
    for field in ("root", "bass"):
        if field in transposed:
            transposed[field] = transpose_note_name(
                str(transposed[field]),
                semitones,
            )
    return transposed


def _transpose_segment_tail(
    segments: Sequence[dict[str, Any]],
    *,
    splice_time_sec: float,
    semitones: int,
    kind: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for segment in segments:
        start_time = float(segment["start_time"])
        end_time = float(segment["end_time"])
        if end_time <= splice_time_sec:
            output.append(dict(segment))
            continue

        if start_time < splice_time_sec:
            before = dict(segment)
            before["end_time"] = float(splice_time_sec)
            if float(before["end_time"]) > start_time:
                output.append(before)

        after = dict(segment)
        after["start_time"] = max(start_time, float(splice_time_sec))
        if kind == "chord":
            after = _transpose_chord_segment(after, semitones)
        elif kind == "key" and "key" in after:
            after["key"] = transpose_key_name(str(after["key"]), semitones)
        else:
            if kind != "key":
                raise ValueError(f"Unknown segment kind: {kind}")
        if end_time > float(after["start_time"]):
            output.append(after)
    return output


def inject_synthetic_boundaries(
    boundary_targets: torch.Tensor,
    splice_frame: int | None,
    *,
    force_boundary: bool,
) -> None:
    """synthetic splice 境界を明示的に 1.0 へ上書きする。"""
    if not force_boundary or splice_frame is None:
        return
    if 0 <= int(splice_frame) < int(boundary_targets.shape[-1]):
        boundary_targets[int(splice_frame)] = 1.0


def _build_modulation_candidate(
    *,
    shift: int,
    boundary_time_sec: float,
    window_start_sec: float,
    prev2: dict[str, Any] | None,
    prev1: dict[str, Any],
    next1: dict[str, Any],
    next2: dict[str, Any] | None,
    source_key: str,
    target_key: str,
    sample_rate: int,
    hop_length: int,
) -> ModulationCandidate | None:
    source_key_spec = _parse_key_spec(source_key)
    target_key_spec = _parse_key_spec(target_key)
    if source_key_spec is None or target_key_spec is None:
        return None

    prev1_root_pc = _note_name_to_index(str(prev1.get("root", "N")))
    next1_root_pc = _note_name_to_index(str(next1.get("root", "N")))
    if prev1_root_pc is None or next1_root_pc is None:
        return None

    prev1_family = _simplify_quality_family(str(prev1.get("quality", "")))
    next1_family = _simplify_quality_family(str(next1.get("quality", "")))
    if prev1_family is None or next1_family is None:
        return None

    prev2_root_pc = None
    prev2_family = None
    if prev2 is not None:
        prev2_root_pc = _note_name_to_index(str(prev2.get("root", "N")))
        prev2_family = _simplify_quality_family(str(prev2.get("quality", "")))

    next2_root_pc = None
    next2_family = None
    if next2 is not None:
        next2_root_pc = _note_name_to_index(str(next2.get("root", "N")))
        next2_family = _simplify_quality_family(str(next2.get("quality", "")))

    source_degree_map = _build_degree_map(source_key_spec)
    target_degree_map = _build_degree_map(target_key_spec)
    common_chords = _build_common_chord_set(source_degree_map, target_degree_map)

    next1_degree = _classify_degree(next1_root_pc, next1_family, target_degree_map)
    next2_degree = None
    if next2_root_pc is not None and next2_family is not None:
        next2_degree = _classify_degree(next2_root_pc, next2_family, target_degree_map)

    satisfied_families: dict[str, float] = {}
    if _is_ii_v_i(
        prev2_root_pc=prev2_root_pc,
        prev2_family=prev2_family,
        prev1_root_pc=prev1_root_pc,
        prev1_family=prev1_family,
        next1_root_pc=next1_root_pc,
        next1_family=next1_family,
        target_degree_map=target_degree_map,
    ):
        satisfied_families["ii_v_i"] = _support_duration(prev2, prev1, next1)

    if _is_dominant_to_tonic(
        prev1_root_pc=prev1_root_pc,
        prev1_family=prev1_family,
        next1_root_pc=next1_root_pc,
        next1_family=next1_family,
        target_degree_map=target_degree_map,
    ):
        satisfied_families["dominant_to_tonic"] = _support_duration(prev1, next1)

    if _is_secondary_dominant(
        prev1_root_pc=prev1_root_pc,
        prev1_family=prev1_family,
        next1_root_pc=next1_root_pc,
        next1_family=next1_family,
        next1_degree=next1_degree,
        next2_degree=next2_degree,
        target_degree_map=target_degree_map,
    ):
        satisfied_families["secondary_dominant"] = _support_duration(
            prev1,
            next1,
            next2,
        )

    if _is_common_chord_pivot(
        shift=shift,
        prev1_root_pc=prev1_root_pc,
        prev1_family=prev1_family,
        next1_degree=next1_degree,
        next2_degree=next2_degree,
        common_chords=common_chords,
    ):
        satisfied_families["common_chord_pivot"] = _support_duration(
            prev1,
            next1,
            next2,
        )

    if _is_common_tone_diminished(
        prev1_root_pc=prev1_root_pc,
        prev1_family=prev1_family,
        next1_root_pc=next1_root_pc,
        next1_degree=next1_degree,
        target_key_spec=target_key_spec,
    ):
        satisfied_families["common_tone_diminished"] = _support_duration(prev1, next1)

    if not satisfied_families:
        return None

    family, priority = _choose_best_family(abs(int(shift)), satisfied_families)
    splice_frame = int(
        round(
            (boundary_time_sec - window_start_sec)
            * float(sample_rate)
            / float(hop_length)
        )
    )
    return ModulationCandidate(
        shift=int(shift),
        splice_time_sec=float(boundary_time_sec),
        splice_frame=splice_frame,
        family=family,
        priority=priority,
        support_duration_sec=float(satisfied_families[family]),
        source_key=source_key,
        target_key=target_key,
    )


def _find_active_key_name(
    segments: Sequence[dict[str, Any]],
    time_sec: float,
    *,
    side: str,
) -> str | None:
    if not segments:
        return None

    epsilon = 1e-6
    probe_time = (
        float(time_sec) - epsilon if side == "left" else float(time_sec) + epsilon
    )
    for segment in segments:
        start_time = float(segment["start_time"])
        end_time = float(segment["end_time"])
        if start_time <= probe_time < end_time:
            key_name = str(segment.get("key", "N"))
            return key_name if key_name != "N" else None

    for segment in segments:
        start_time = float(segment["start_time"])
        end_time = float(segment["end_time"])
        if (
            start_time <= float(time_sec) < end_time
            or abs(start_time - float(time_sec)) < 1e-6
        ):
            key_name = str(segment.get("key", "N"))
            return key_name if key_name != "N" else None
    return None


def _parse_key_spec(key_name: str) -> KeySpec | None:
    if key_name == "N":
        return None
    mode = "minor" if key_name.endswith("m") else "major"
    tonic_name = key_name[:-1] if mode == "minor" else key_name
    tonic_pc = _note_name_to_index(tonic_name)
    if tonic_pc is None:
        return None
    return KeySpec(tonic_pc=tonic_pc, mode=mode)


def _parse_key_to_relative_major_class(key_name: str) -> int | None:
    key_spec = _parse_key_spec(key_name)
    if key_spec is None:
        return None
    if key_spec.mode == "minor":
        return (key_spec.tonic_pc + 3) % 12
    return key_spec.tonic_pc


def _note_name_to_index(note_name: str) -> int | None:
    if note_name == "N":
        return None
    return NOTE_TO_INDEX.get(note_name)


def _simplify_quality_family(quality: str) -> str | None:
    normalized = str(quality).replace(" ", "")
    if normalized == "N":
        return None
    if normalized == "":
        return "major"
    if normalized in {"5", "sus4"}:
        return None
    if normalized in {"sus2"}:
        return "major"
    if "aug" in normalized or "#5" in normalized:
        return None
    if normalized.startswith("7sus4"):
        return "dominant"
    if "dim" in normalized or normalized.startswith("m7-5"):
        return "diminished"
    if normalized.startswith("m"):
        return "minor"
    if normalized.startswith("M") or normalized in {"6", "69", "add9"}:
        return "major"
    if (
        normalized.startswith("7")
        or normalized.startswith("9")
        or normalized.startswith("13")
    ):
        return "dominant"
    return None


def _build_degree_map(key_spec: KeySpec) -> dict[str, tuple[int, tuple[str, ...]]]:
    tonic_pc = int(key_spec.tonic_pc)
    if key_spec.mode == "major":
        return {
            "I": (tonic_pc, ("major",)),
            "ii": ((tonic_pc + 2) % 12, ("minor",)),
            "iii": ((tonic_pc + 4) % 12, ("minor",)),
            "IV": ((tonic_pc + 5) % 12, ("major",)),
            "V": ((tonic_pc + 7) % 12, ("major", "dominant")),
            "vi": ((tonic_pc + 9) % 12, ("minor",)),
            "vii": ((tonic_pc + 11) % 12, ("diminished",)),
        }
    return {
        "i": (tonic_pc, ("minor",)),
        "ii": ((tonic_pc + 2) % 12, ("diminished",)),
        "III": ((tonic_pc + 3) % 12, ("major",)),
        "iv": ((tonic_pc + 5) % 12, ("minor",)),
        "V": ((tonic_pc + 7) % 12, ("major", "dominant")),
        "VI": ((tonic_pc + 8) % 12, ("major",)),
        "VII": ((tonic_pc + 10) % 12, ("major",)),
        "vii": ((tonic_pc + 11) % 12, ("diminished",)),
    }


def _build_common_chord_set(
    source_degree_map: dict[str, tuple[int, tuple[str, ...]]],
    target_degree_map: dict[str, tuple[int, tuple[str, ...]]],
) -> set[tuple[int, str]]:
    source_chords = _degree_map_to_chord_set(source_degree_map)
    target_chords = _degree_map_to_chord_set(target_degree_map)
    return source_chords & target_chords


def _degree_map_to_chord_set(
    degree_map: dict[str, tuple[int, tuple[str, ...]]],
) -> set[tuple[int, str]]:
    chord_set: set[tuple[int, str]] = set()
    for root_pc, families in degree_map.values():
        for family in families:
            chord_set.add((int(root_pc), family))
    return chord_set


def _classify_degree(
    root_pc: int,
    family: str,
    degree_map: dict[str, tuple[int, tuple[str, ...]]],
) -> str | None:
    for degree_name, (degree_root_pc, allowed_families) in degree_map.items():
        if int(root_pc) == int(degree_root_pc) and family in allowed_families:
            return degree_name
    return None


def _choose_best_family(
    abs_shift: int,
    satisfied_families: dict[str, float],
) -> tuple[str, int]:
    priority_map = FAMILY_PRIORITY_BY_SHIFT.get(abs_shift, FAMILY_PRIORITY_BY_SHIFT[6])
    sorted_families = sorted(
        satisfied_families.keys(),
        key=lambda family_name: (
            priority_map.get(family_name, 999),
            FAMILY_TIEBREAKER.index(family_name),
        ),
    )
    best_family = sorted_families[0]
    return best_family, int(priority_map.get(best_family, 999))


def _is_ii_v_i(
    *,
    prev2_root_pc: int | None,
    prev2_family: str | None,
    prev1_root_pc: int,
    prev1_family: str,
    next1_root_pc: int,
    next1_family: str,
    target_degree_map: dict[str, tuple[int, tuple[str, ...]]],
) -> bool:
    if prev2_root_pc is None or prev2_family is None:
        return False
    prev2_degree = _classify_degree(prev2_root_pc, prev2_family, target_degree_map)
    prev1_degree = _classify_degree(prev1_root_pc, prev1_family, target_degree_map)
    next1_degree = _classify_degree(next1_root_pc, next1_family, target_degree_map)
    return prev2_degree == "ii" and prev1_degree == "V" and next1_degree in {"I", "i"}


def _is_dominant_to_tonic(
    *,
    prev1_root_pc: int,
    prev1_family: str,
    next1_root_pc: int,
    next1_family: str,
    target_degree_map: dict[str, tuple[int, tuple[str, ...]]],
) -> bool:
    prev1_degree = _classify_degree(prev1_root_pc, prev1_family, target_degree_map)
    next1_degree = _classify_degree(next1_root_pc, next1_family, target_degree_map)
    if next1_degree not in {"I", "i"}:
        return False
    if prev1_degree == "V" and prev1_family in {"major", "dominant"}:
        return True
    return prev1_degree == "vii" and prev1_family == "diminished"


def _is_secondary_dominant(
    *,
    prev1_root_pc: int,
    prev1_family: str,
    next1_root_pc: int,
    next1_family: str,
    next1_degree: str | None,
    next2_degree: str | None,
    target_degree_map: dict[str, tuple[int, tuple[str, ...]]],
) -> bool:
    if next1_degree is None:
        return False

    next1_position = DEGREE_POSITION.get(next1_degree)
    if next1_position not in {2, 3, 4, 5, 6}:
        return False

    is_cadential_follow = (
        next1_degree in CADENTIAL_DEGREES or next2_degree in CADENTIAL_DEGREES
    )
    if not is_cadential_follow and next1_position != 5:
        return False

    target_root_pc = next1_root_pc
    dominant_root_pc = (target_root_pc + 7) % 12
    leading_root_pc = (target_root_pc + 11) % 12
    is_v_of_x = prev1_root_pc == dominant_root_pc and prev1_family in {
        "major",
        "dominant",
    }
    is_vii_of_x = prev1_root_pc == leading_root_pc and prev1_family == "diminished"
    if not (is_v_of_x or is_vii_of_x):
        return False

    # next1 は target key 内の実在コードである必要がある。
    return (
        _classify_degree(next1_root_pc, next1_family, target_degree_map) == next1_degree
    )


def _is_common_chord_pivot(
    *,
    shift: int,
    prev1_root_pc: int,
    prev1_family: str,
    next1_degree: str | None,
    next2_degree: str | None,
    common_chords: set[tuple[int, str]],
) -> bool:
    if (prev1_root_pc, prev1_family) not in common_chords:
        return False
    if next1_degree not in CADENTIAL_DEGREES and next2_degree not in CADENTIAL_DEGREES:
        return False

    common_count = len(common_chords)
    abs_shift = abs(int(shift))
    if abs_shift == 1:
        return common_count <= 2 and next1_degree in CADENTIAL_DEGREES
    if abs_shift == 6:
        return common_count <= 1 and next1_degree in {"I", "i"}
    return True


def _is_common_tone_diminished(
    *,
    prev1_root_pc: int,
    prev1_family: str,
    next1_root_pc: int,
    next1_degree: str | None,
    target_key_spec: KeySpec,
) -> bool:
    if prev1_family != "diminished" or next1_degree not in {"I", "i"}:
        return False

    diminished_notes = _build_chord_pitch_classes(prev1_root_pc, prev1_family)
    tonic_notes = _build_chord_pitch_classes(
        target_key_spec.tonic_pc,
        "major" if target_key_spec.mode == "major" else "minor",
    )
    dominant_notes = _build_chord_pitch_classes(
        (target_key_spec.tonic_pc + 7) % 12, "major"
    )
    if diminished_notes & tonic_notes:
        return True
    return bool(diminished_notes & dominant_notes)


def _build_chord_pitch_classes(root_pc: int, family: str) -> set[int]:
    intervals = CHORD_INTERVALS_BY_FAMILY.get(family, ())
    return {(int(root_pc) + interval) % 12 for interval in intervals}


def _support_duration(*segments: dict[str, Any] | None) -> float:
    duration = 0.0
    for segment in segments:
        if segment is None:
            continue
        duration += max(
            0.0,
            float(segment["end_time"]) - float(segment["start_time"]),
        )
    return duration
