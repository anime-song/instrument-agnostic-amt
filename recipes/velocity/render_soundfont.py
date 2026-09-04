"""Render velocity calibration sweeps with FluidSynth."""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
from pathlib import Path


VELOCITY_ROOT = (
    Path(__file__).resolve().parents[2] / "instrument_agnostic_amt" / "velocity"
)
DEFAULT_MANIFEST = (
    VELOCITY_ROOT / "artifacts" / "monalisa_gm_calibration" / "render_manifest.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a prepared velocity-sweep manifest with FluidSynth."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--soundfont",
        type=Path,
        required=True,
        help="SoundFont used for this render. Its location is not saved to artifacts.",
    )
    parser.add_argument(
        "--fluidsynth-executable",
        type=Path,
        default=None,
        help="Path to fluidsynth.exe. If omitted, PATH is searched.",
    )
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_fluidsynth(path: Path | None) -> Path:
    if path is not None:
        candidate = path.expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"FluidSynth executable not found: {candidate}")
        return candidate
    discovered = shutil.which("fluidsynth")
    if discovered is None:
        raise FileNotFoundError(
            "FluidSynth was not found. Install it or pass --fluidsynth-executable."
        )
    return Path(discovered).resolve()


def build_fluidsynth_command(
    *,
    executable: Path,
    soundfont_path: Path,
    midi_path: Path,
    wav_path: Path,
    sample_rate: int,
    gain: float,
) -> list[str]:
    return [
        str(executable),
        "-ni",
        "-g",
        str(float(gain)),
        "-r",
        str(int(sample_rate)),
        "-O",
        "s16",
        "-F",
        str(wav_path),
        str(soundfont_path),
        str(midi_path),
    ]


def main() -> None:
    args = parse_args()
    if args.gain <= 0.0:
        raise ValueError("--gain must be positive")
    executable = resolve_fluidsynth(args.fluidsynth_executable)
    soundfont_path = args.soundfont.expanduser().resolve()
    if not soundfont_path.is_file():
        raise FileNotFoundError(f"SoundFont not found: {soundfont_path}")
    manifest_path = args.manifest.expanduser().resolve()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    for index, row in enumerate(rows, start=1):
        midi_path = Path(row["midi_path"])
        wav_path = Path(row["wav_path"])
        if not midi_path.is_absolute():
            midi_path = manifest_path.parent / midi_path
        if not wav_path.is_absolute():
            wav_path = manifest_path.parent / wav_path
        if wav_path.is_file() and not args.overwrite:
            continue
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_fluidsynth_command(
            executable=executable,
            soundfont_path=soundfont_path,
            midi_path=midi_path,
            wav_path=wav_path,
            sample_rate=int(row["sample_rate"]),
            gain=args.gain,
        )
        print(f"[{index}/{len(rows)}] {midi_path.name}")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
