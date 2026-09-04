from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import soundfile as sf

from .render_soundfont import build_fluidsynth_command, resolve_fluidsynth


VELOCITY_ROOT = (
    Path(__file__).resolve().parents[2] / "instrument_agnostic_amt" / "velocity"
)
DEFAULT_MANIFEST = VELOCITY_ROOT / "artifacts" / "synthetic" / "render_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render synthetic target MIDI stems with FluidSynth."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--soundfont",
        type=Path,
        required=True,
        help="SoundFont used at runtime; its location is not written to manifests.",
    )
    parser.add_argument("--fluidsynth-executable", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    executable = resolve_fluidsynth(args.fluidsynth_executable)
    soundfont_path = args.soundfont.expanduser().resolve()
    if not soundfont_path.is_file():
        raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")
    manifest_path = args.manifest.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit is not None:
        rows = rows[: args.limit]
    rendered = 0
    skipped = 0
    for index, row in enumerate(rows, start=1):
        midi_path = _resolve(row["target_midi_path"], manifest_path.parent)
        wav_path = _resolve(row["rendered_stem_path"], manifest_path.parent)
        wav_is_current = (
            wav_path.is_file()
            and wav_path.stat().st_mtime_ns >= midi_path.stat().st_mtime_ns
            and int(sf.info(str(wav_path)).samplerate) == int(row["sample_rate"])
        )
        if wav_is_current and not args.overwrite:
            skipped += 1
            continue
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_fluidsynth_command(
            executable=executable,
            soundfont_path=soundfont_path,
            midi_path=midi_path,
            wav_path=wav_path,
            sample_rate=int(row["sample_rate"]),
            gain=float(row["render_synth_gain"]),
        )
        print(f"[{index}/{len(rows)}] {row['example_id']} / {row['stem_name']}")
        subprocess.run(command, check=True)
        rendered += 1
    print(f"Rendered {rendered} stems; skipped {skipped} existing stems")


if __name__ == "__main__":
    main()
