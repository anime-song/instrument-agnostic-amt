"""Discovery of source audio and MIDI for velocity training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_STEM_NAMES = ("bass", "drums", "guitar", "other", "piano", "vocals")


@dataclass(frozen=True)
class VelocitySourceItem:
    """One separated stem and the AMT MIDI that will condition velocity prediction."""

    song_id: str
    stem_name: str
    wav_path: Path
    midi_path: Path | None
    merged_midi_path: Path | None

    @property
    def has_midi(self) -> bool:
        return self.midi_path is not None and self.midi_path.is_file()


def _stem_name_from_path(
    path: Path,
    *,
    song_id: str,
    stem_names: Iterable[str],
) -> str | None:
    prefix = f"{song_id}_"
    if not path.stem.startswith(prefix):
        return None
    suffix = path.stem[len(prefix) :]
    return suffix if suffix in set(stem_names) else None


def discover_amt_cbnet_items(
    source_root: str | Path,
    *,
    stem_names: tuple[str, ...] = DEFAULT_STEM_NAMES,
    limit_songs: int | None = None,
) -> list[VelocitySourceItem]:
    """
    Join AMT-CBNet ``stems/``, ``midis/`` and ``merged/`` by song/stem ID.

    Missing per-stem MIDI files are retained in the result so an audit manifest
    can distinguish missing transcription from a genuinely empty MIDI file.
    """

    root = Path(source_root).expanduser().resolve()
    stems_dir = root / "stems"
    midis_dir = root / "midis"
    merged_dir = root / "merged"
    if not stems_dir.is_dir():
        raise FileNotFoundError(f"AMT-CBNet stems directory not found: {stems_dir}")
    if not midis_dir.is_dir():
        raise FileNotFoundError(f"AMT-CBNet MIDI directory not found: {midis_dir}")

    song_dirs = sorted(
        (path for path in stems_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )
    if limit_songs is not None:
        if limit_songs < 1:
            raise ValueError("limit_songs must be positive")
        song_dirs = song_dirs[: int(limit_songs)]

    items: list[VelocitySourceItem] = []
    stem_name_set = set(stem_names)
    for song_dir in song_dirs:
        song_id = song_dir.name
        merged_path = merged_dir / f"{song_id}.mid"
        merged = merged_path if merged_path.is_file() else None
        wav_by_stem: dict[str, Path] = {}
        for wav_path in sorted(song_dir.glob("*.wav")):
            stem_name = _stem_name_from_path(
                wav_path,
                song_id=song_id,
                stem_names=stem_name_set,
            )
            if stem_name is not None:
                wav_by_stem[stem_name] = wav_path.resolve()

        for stem_name in stem_names:
            wav_path = wav_by_stem.get(stem_name)
            if wav_path is None:
                continue
            midi_candidate = midis_dir / f"{song_id}_{stem_name}.mid"
            items.append(
                VelocitySourceItem(
                    song_id=song_id,
                    stem_name=stem_name,
                    wav_path=wav_path,
                    midi_path=(
                        midi_candidate.resolve() if midi_candidate.is_file() else None
                    ),
                    merged_midi_path=(merged.resolve() if merged is not None else None),
                )
            )
    return items
