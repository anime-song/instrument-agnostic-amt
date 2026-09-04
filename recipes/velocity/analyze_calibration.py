from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.curve import (
    CalibrationAnalysisConfig,
    analyze_sweep_files,
    write_curve_csv,
    write_curve_npz,
)


VELOCITY_ROOT = (
    Path(__file__).resolve().parents[2] / "instrument_agnostic_amt" / "velocity"
)
DEFAULT_CALIBRATION_ROOT = VELOCITY_ROOT / "artifacts" / "monalisa_gm_calibration"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract monotonic velocity curves from rendered sweep WAV files."
    )
    parser.add_argument(
        "--calibration-root",
        type=Path,
        default=DEFAULT_CALIBRATION_ROOT,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--frame-ms", type=float, default=10.0)
    parser.add_argument("--signal-start-ms", type=float, default=5.0)
    parser.add_argument("--signal-end-ms", type=float, default=350.0)
    parser.add_argument("--frame-percentile", type=float, default=95.0)
    parser.add_argument("--min-signal-dbfs", type=float, default=-90.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calibration_root = args.calibration_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else calibration_root / "analysis"
    )
    config = CalibrationAnalysisConfig(
        frame_ms=args.frame_ms,
        signal_start_ms=args.signal_start_ms,
        signal_end_ms=args.signal_end_ms,
        frame_percentile=args.frame_percentile,
        min_signal_dbfs=args.min_signal_dbfs,
    )
    rows, summary = analyze_sweep_files(
        calibration_root / "sweep_events.csv",
        calibration_root / "render_manifest.csv",
        config=config,
    )
    write_curve_csv(output_dir / "velocity_curves.csv", rows)
    write_curve_npz(output_dir / "velocity_curves.npz", rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        "Analyzed "
        f"{summary['event_count']} events across "
        f"{summary['curve_group_count']} pitch-conditioned curves"
    )


if __name__ == "__main__":
    main()
