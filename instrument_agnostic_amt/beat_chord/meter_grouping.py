from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal


GroupingMode = Literal["fixed", "latent"]
GroupingPattern = tuple[int, ...]


@dataclass(frozen=True)
class MeterGroupingSpec:
    """Major beat-group candidates for one notated meter."""

    mode: GroupingMode
    patterns: tuple[GroupingPattern, ...]

    def __post_init__(self) -> None:
        if self.mode not in {"fixed", "latent"}:
            raise ValueError(f"Unsupported grouping mode: {self.mode}")
        if not self.patterns:
            raise ValueError("Grouping specs must contain at least one pattern")
        if self.mode == "fixed" and len(self.patterns) != 1:
            raise ValueError("Fixed grouping specs must contain exactly one pattern")

        totals = {sum(pattern) for pattern in self.patterns}
        if len(totals) != 1:
            raise ValueError("All grouping patterns must have the same total")
        for pattern in self.patterns:
            if len(pattern) < 2:
                raise ValueError("Major grouping patterns need an internal boundary")
            if any(group_size <= 0 for group_size in pattern):
                raise ValueError("Grouping sizes must be positive")

    @property
    def numerator(self) -> int:
        return int(sum(self.patterns[0]))


def _dedupe_patterns(
    patterns: tuple[GroupingPattern, ...] | list[GroupingPattern],
) -> tuple[GroupingPattern, ...]:
    return tuple(
        dict.fromkeys(tuple(int(size) for size in pattern) for pattern in patterns)
    )


@lru_cache(maxsize=None)
def ordered_groupings(
    numerator: int,
    *,
    allowed_group_sizes: tuple[int, ...] = (2, 3),
) -> tuple[GroupingPattern, ...]:
    """Return ordered additive partitions of ``numerator``."""

    numerator = int(numerator)
    sizes = tuple(sorted({int(size) for size in allowed_group_sizes if int(size) > 0}))
    if numerator <= 0 or not sizes:
        return ()

    patterns: list[GroupingPattern] = []

    def visit(remaining: int, prefix: GroupingPattern) -> None:
        if remaining == 0:
            if len(prefix) >= 2:
                patterns.append(prefix)
            return
        for size in sizes:
            if size <= remaining:
                visit(remaining - size, (*prefix, size))

    visit(numerator, ())
    return _dedupe_patterns(patterns)


_EXPLICIT_LATENT_GROUPINGS: dict[tuple[int, int], tuple[GroupingPattern, ...]] = {
    (5, 2): ((2, 3), (3, 2)),
    (5, 4): ((2, 3), (3, 2)),
    (5, 8): ((2, 3), (3, 2)),
    (5, 16): ((2, 3), (3, 2)),
    (6, 2): ((3, 3), (2, 2, 2)),
    (6, 4): ((3, 3), (2, 2, 2)),
    # At the major-boundary level, 7/4 is most usefully represented as
    # two large groups. A four-beat group may contain a lower-level 2+2
    # subdivision, but that is intentionally outside this model.
    (7, 2): ((4, 3), (3, 4)),
    (7, 4): ((4, 3), (3, 4)),
    (7, 8): ((2, 2, 3), (2, 3, 2), (3, 2, 2)),
    (7, 16): ((2, 2, 3), (2, 3, 2), (3, 2, 2)),
    (8, 8): ((4, 4), (3, 3, 2), (3, 2, 3), (2, 3, 3)),
    (8, 16): ((4, 4), (3, 3, 2), (3, 2, 3), (2, 3, 3)),
}

_COMPOUND_DENOMINATORS = frozenset({8, 16, 32})
_FIXED_COMPOUND_NUMERATORS = frozenset({6, 9, 12})
_ADDITIVE_DENOMINATORS = frozenset({8, 16, 32})
_LONG_BEAT_DENOMINATORS = frozenset({2, 4})
# Candidate counts grow exponentially. Twenty-one denominator-note units
# covers the practical odd/compound meters in the training corpus while
# keeping latent marginalization and DP edge scoring bounded.
_MAX_GENERATED_NUMERATOR = 21


@lru_cache(maxsize=None)
def grouping_spec_for_meter(
    meter_num: int,
    meter_den: int,
) -> MeterGroupingSpec | None:
    """Return a major-grouping spec only for meters that need one."""

    meter = (int(meter_num), int(meter_den))
    numerator, denominator = meter
    if numerator <= 4 or numerator > _MAX_GENERATED_NUMERATOR:
        return None

    if (
        denominator in _COMPOUND_DENOMINATORS
        and numerator in _FIXED_COMPOUND_NUMERATORS
    ):
        fixed_pattern = (3,) * (numerator // 3)
        return MeterGroupingSpec(mode="fixed", patterns=(fixed_pattern,))

    latent_patterns = _EXPLICIT_LATENT_GROUPINGS.get(meter)
    if latent_patterns is not None:
        return MeterGroupingSpec(
            mode="latent",
            patterns=_dedupe_patterns(latent_patterns),
        )

    if denominator in _ADDITIVE_DENOMINATORS:
        patterns = ordered_groupings(numerator)
        if patterns:
            return MeterGroupingSpec(mode="latent", patterns=patterns)

    if denominator in _LONG_BEAT_DENOMINATORS and numerator >= 8:
        patterns = ordered_groupings(
            numerator,
            allowed_group_sizes=(3, 4, 5),
        )
        if patterns:
            return MeterGroupingSpec(mode="latent", patterns=patterns)
    return None


@lru_cache(maxsize=None)
def grouping_boundary_offsets(pattern: GroupingPattern) -> tuple[int, ...]:
    """Return internal boundary offsets in denominator-note units."""

    offsets: list[int] = []
    position = 0
    for group_size in pattern[:-1]:
        position += int(group_size)
        offsets.append(position)
    return tuple(offsets)
