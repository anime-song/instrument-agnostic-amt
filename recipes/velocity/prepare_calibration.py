from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import load_pipeline_config
from .data.calibration import build_velocity_sweep_midi, write_velocity_sweep_midi


VELOCITY_ROOT = (
    Path(__file__).resolve().parents[2] / "instrument_agnostic_amt" / "velocity"
)
DEFAULT_CONFIG = VELOCITY_ROOT / "configs" / "monalisa_gm.json"
DEFAULT_OUTPUT_ROOT = VELOCITY_ROOT / "artifacts" / "monalisa_gm_calibration"


def parse_programs(value: str) -> tuple[int, ...]:
    programs: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"invalid program range: {part}")
            programs.update(range(start, end + 1))
        else:
            programs.add(int(part))
    if not programs or min(programs) < 0 or max(programs) > 127:
        raise ValueError("programs must be within General MIDI 0..127")
    return tuple(sorted(programs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare per-program MIDI velocity sweeps for the target SoundFont."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--programs",
        type=str,
        default="0-127",
        help="Comma-separated zero-based GM programs and ranges, e.g. 0,24-31,32-39.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-drums", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config(args.config)
    output_root = args.output_root.expanduser().resolve()
    sweep_dir = output_root / "midi"
    programs = parse_programs(args.programs)
    event_rows: list[dict[str, object]] = []
    render_rows: list[dict[str, object]] = []
    for program in programs:
        midi_path = sweep_dir / f"program_{program:03d}.mid"
        if args.overwrite or not midi_path.is_file():
            events = write_velocity_sweep_midi(
                midi_path,
                program=program,
                is_drum=False,
                pitches=config.sweep_pitches,
                velocities=config.sweep_velocities,
                note_seconds=config.sweep_note_seconds,
                gap_seconds=config.sweep_gap_seconds,
            )
        else:
            _, events = build_velocity_sweep_midi(
                program=program,
                is_drum=False,
                pitches=config.sweep_pitches,
                velocities=config.sweep_velocities,
                note_seconds=config.sweep_note_seconds,
                gap_seconds=config.sweep_gap_seconds,
            )
        for event in events:
            event_rows.append(
                {
                    "program": event.program,
                    "is_drum": int(event.is_drum),
                    "pitch": event.pitch,
                    "velocity": event.velocity,
                    "start_seconds": event.start_seconds,
                    "end_seconds": event.end_seconds,
                    "midi_path": midi_path.relative_to(output_root).as_posix(),
                }
            )
        render_rows.append(
            {
                "program": program,
                "is_drum": 0,
                "midi_path": midi_path.relative_to(output_root).as_posix(),
                "wav_path": f"wav/program_{program:03d}.wav",
                "sample_rate": config.render_sample_rate,
            }
        )

    if not args.skip_drums:
        midi_path = sweep_dir / "drums.mid"
        if args.overwrite or not midi_path.is_file():
            drum_events = write_velocity_sweep_midi(
                midi_path,
                program=0,
                is_drum=True,
                pitches=config.sweep_drum_pitches,
                velocities=config.sweep_velocities,
                note_seconds=config.sweep_note_seconds,
                gap_seconds=config.sweep_gap_seconds,
            )
        else:
            _, drum_events = build_velocity_sweep_midi(
                program=0,
                is_drum=True,
                pitches=config.sweep_drum_pitches,
                velocities=config.sweep_velocities,
                note_seconds=config.sweep_note_seconds,
                gap_seconds=config.sweep_gap_seconds,
            )
        for event in drum_events:
            event_rows.append(
                {
                    "program": event.program,
                    "is_drum": 1,
                    "pitch": event.pitch,
                    "velocity": event.velocity,
                    "start_seconds": event.start_seconds,
                    "end_seconds": event.end_seconds,
                    "midi_path": midi_path.relative_to(output_root).as_posix(),
                }
            )
        render_rows.append(
            {
                "program": 0,
                "is_drum": 1,
                "midi_path": midi_path.relative_to(output_root).as_posix(),
                "wav_path": "wav/drums.wav",
                "sample_rate": config.render_sample_rate,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "sweep_events.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=tuple(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)
    with (output_root / "render_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=tuple(render_rows[0]))
        writer.writeheader()
        writer.writerows(render_rows)

    metadata = {
        "schema_version": 2,
        "calibration_id": "monalisa_gm",
        "programs": list(programs),
        "pitches": list(config.sweep_pitches),
        "drum_pitches": list(config.sweep_drum_pitches),
        "velocities": list(config.sweep_velocities),
        "sample_rate": config.render_sample_rate,
        "note_seconds": config.sweep_note_seconds,
        "gap_seconds": config.sweep_gap_seconds,
    }
    (output_root / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    sweep_count = len(programs) + int(not args.skip_drums)
    print(f"Prepared {sweep_count} melodic/drum sweeps in {output_root}")


if __name__ == "__main__":
    main()
