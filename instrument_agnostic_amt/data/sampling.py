import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


UNIFORM_WINDOW_SAMPLING_PROBABILITY = 0.5


@dataclass(frozen=True)
class ActiveWindowSegment:
    """active 区間サンプリング用の一定密度セグメント。"""

    start_ms: int
    end_ms: int
    note_overlap_count: int
    sample_mass: float


@dataclass(frozen=True)
class ActiveWindowProfile:
    """1本の stem から引ける active window 開始位置の分布。"""

    max_start_ms: int
    segments: tuple[ActiveWindowSegment, ...] = ()
    total_sample_mass: float = 0.0

    @property
    def has_active_segments(self) -> bool:
        """active 区間の候補が1つでもあるかを返す。"""
        return self.total_sample_mass > 0.0 and bool(self.segments)


class StemWindowSelector:
    """stem 選択と window 開始位置サンプリングをまとめる。"""

    def __init__(
        self,
        *,
        dataset_groups_by_name: dict[str, dict[str, Any]],
        window_ms: int,
        p_intra_drop: float,
    ) -> None:
        self.dataset_groups_by_name = dataset_groups_by_name
        self.window_ms = int(window_ms)
        self.p_intra_drop = float(p_intra_drop)
        # active 区間サンプリング用に、stem ごとの window 分布を遅延構築して再利用する。
        self.active_window_profiles_by_npz_path: dict[str, ActiveWindowProfile] = {}

    def select_base_stems(
        self,
        *,
        base_stems: list[dict[str, Any]],
        selected_group: dict[str, Any],
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        """同一楽曲から使う base stem 群を選ぶ。"""
        if not selected_group.get("allow_multi_stem_same_song", True):
            return [rng.choice(base_stems)]
        if not selected_group.get("use_intra_drop", True):
            return list(base_stems)

        selected_base_stems = []
        for stem in base_stems:
            if rng.random() >= self.p_intra_drop:
                selected_base_stems.append(stem)

        if not selected_base_stems:
            selected_base_stems.append(rng.choice(base_stems))
        return selected_base_stems

    def select_base_window_start_ms(
        self,
        *,
        stems: list[dict[str, Any]],
        selected_group: dict[str, Any],
        rng: random.Random,
    ) -> int:
        """base stem 群に対して共通の window 開始位置を決める。"""
        if not stems:
            return 0

        if (
            not selected_group.get("active_window_sampling", False)
            or rng.random() < UNIFORM_WINDOW_SAMPLING_PROBABILITY
        ):
            max_effective_end_ms = max(
                min(int(stem["duration_ms"]), int(stem["end_note_ms"]))
                for stem in stems
            )
            max_start_ms = max(0, max_effective_end_ms - self.window_ms)
            return rng.randint(0, max_start_ms) if max_start_ms > 0 else 0

        anchor_stem = self._select_note_rich_anchor_stem(stems=stems, rng=rng)
        profile = self._get_active_window_profile(anchor_stem)
        return self._sample_window_start_from_profile(profile=profile, rng=rng)

    def select_stem_window_start_ms(
        self,
        *,
        stem: dict[str, Any],
        rng: random.Random,
    ) -> int:
        """1本の stem に対して開始位置を選ぶ。"""
        group_name = str(stem.get("dataset_group_name", "main"))
        group = self.dataset_groups_by_name.get(group_name, {})
        profile = self._get_active_window_profile(stem)

        if (
            not group.get("active_window_sampling", False)
            or rng.random() < UNIFORM_WINDOW_SAMPLING_PROBABILITY
        ):
            if profile.max_start_ms <= 0:
                return 0
            return rng.randint(0, profile.max_start_ms)

        return self._sample_window_start_from_profile(profile=profile, rng=rng)

    def _compute_max_start_ms(self, stem: dict[str, Any]) -> int:
        """stem の有効な window 開始位置の上限を返す。"""
        effective_end_ms = min(int(stem["duration_ms"]), int(stem["end_note_ms"]))
        return max(0, effective_end_ms - self.window_ms)

    def _build_active_window_profile(self, stem: dict[str, Any]) -> ActiveWindowProfile:
        """
        ノート重なり数を重みとして、active 区間を優先サンプリングする分布を構築する。

        window 開始位置 s がノート [start, end) と重なる条件は
        `s < end` かつ `s + window_ms > start` なので、
        各ノートは開始位置軸上の区間 `[start - window_ms + 1, end - 1]`
        へ寄与する。
        """
        max_start_ms = self._compute_max_start_ms(stem)
        if max_start_ms <= 0 or int(stem.get("note_count", 0)) <= 0:
            return ActiveWindowProfile(max_start_ms=max_start_ms)

        with np.load(stem["npz_path"]) as data:
            start_ms = data["note_start_ms"]
            end_ms = data["note_end_ms"]

        effective_end_ms = min(int(stem["duration_ms"]), int(stem["end_note_ms"]))
        events: dict[int, int] = defaultdict(int)

        # 1. 各ノートが「この開始位置なら window 内に入る」という範囲を作る。
        for note_start_ms, note_end_ms in zip(
            start_ms.tolist(),
            end_ms.tolist(),
        ):
            clipped_end_ms = min(int(note_end_ms), effective_end_ms)
            interval_start_ms = max(0, int(note_start_ms) - self.window_ms + 1)
            interval_end_ms = min(max_start_ms, clipped_end_ms - 1)
            if interval_start_ms > interval_end_ms:
                continue
            events[interval_start_ms] += 1
            events[interval_end_ms + 1] -= 1

        if not events:
            return ActiveWindowProfile(max_start_ms=max_start_ms)

        # 2. 開始位置軸上で重なり数が一定の区間へ圧縮する。
        segments: list[ActiveWindowSegment] = []
        total_sample_mass = 0.0
        active_overlap_count = 0
        previous_position = min(events)

        for position in sorted(events):
            if active_overlap_count > 0 and previous_position < position:
                segment_start_ms = previous_position
                segment_end_ms = position - 1
                segment_length = segment_end_ms - segment_start_ms + 1
                sample_mass = float(active_overlap_count * segment_length)
                segments.append(
                    ActiveWindowSegment(
                        start_ms=segment_start_ms,
                        end_ms=segment_end_ms,
                        note_overlap_count=active_overlap_count,
                        sample_mass=sample_mass,
                    )
                )
                total_sample_mass += sample_mass

            active_overlap_count += events[position]
            previous_position = position

        return ActiveWindowProfile(
            max_start_ms=max_start_ms,
            segments=tuple(segments),
            total_sample_mass=total_sample_mass,
        )

    def _get_active_window_profile(self, stem: dict[str, Any]) -> ActiveWindowProfile:
        """stem ごとの active window 分布を遅延構築し、以後は再利用する。"""
        npz_path = str(stem["npz_path"])
        profile = self.active_window_profiles_by_npz_path.get(npz_path)
        if profile is not None:
            return profile

        profile = self._build_active_window_profile(stem)
        self.active_window_profiles_by_npz_path[npz_path] = profile
        return profile

    def _sample_window_start_from_profile(
        self,
        *,
        profile: ActiveWindowProfile,
        rng: random.Random,
    ) -> int:
        """構築済み profile から開始位置を1つサンプリングする。"""
        if profile.max_start_ms <= 0:
            return 0

        if not profile.has_active_segments:
            return rng.randint(0, profile.max_start_ms)

        roll = rng.uniform(0.0, profile.total_sample_mass)
        cumulative_mass = 0.0
        for segment in profile.segments:
            cumulative_mass += segment.sample_mass
            if roll <= cumulative_mass:
                if segment.start_ms >= segment.end_ms:
                    return int(segment.start_ms)
                return rng.randint(int(segment.start_ms), int(segment.end_ms))

        return int(profile.segments[-1].end_ms)

    def _select_note_rich_anchor_stem(
        self,
        stems: list[dict[str, Any]],
        rng: random.Random,
    ) -> dict[str, Any]:
        """複数 stem から、ノート量に応じて window 選択の基準 stem を選ぶ。"""
        if len(stems) == 1:
            return stems[0]

        total_note_count = sum(max(0, int(stem.get("note_count", 0))) for stem in stems)
        if total_note_count <= 0:
            return rng.choice(stems)

        roll = rng.uniform(0.0, float(total_note_count))
        cumulative = 0.0
        for stem in stems:
            cumulative += float(max(0, int(stem.get("note_count", 0))))
            if roll <= cumulative:
                return stem
        return stems[-1]
