from __future__ import annotations

import argparse
from pathlib import Path

from .synthesis.config import load_synthetic_config
from .synthesis.plan import prepare_synthetic_plan


VELOCITY_ROOT = (
    Path(__file__).resolve().parents[2] / "instrument_agnostic_amt" / "velocity"
)
DEFAULT_PSEUDO_MANIFEST = VELOCITY_ROOT / "artifacts" / "amt_cbnet" / "manifest.csv"
DEFAULT_OUTPUT_ROOT = VELOCITY_ROOT / "artifacts" / "synthetic"
DEFAULT_CONFIG = VELOCITY_ROOT / "configs" / "synthetic.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic target MIDI and render jobs for velocity training."
    )
    parser.add_argument("--pseudo-manifest", type=Path, default=DEFAULT_PSEUDO_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--min-stems", type=int, default=2)
    parser.add_argument("--limit-songs", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = prepare_synthetic_plan(
        args.pseudo_manifest,
        args.output_root,
        config=load_synthetic_config(args.config),
        variations=args.variations,
        seed=args.seed,
        min_stems=args.min_stems,
        limit_songs=args.limit_songs,
        overwrite=args.overwrite,
    )
    print(
        f"Prepared {summary['example_count']} examples and "
        f"{summary['render_job_count']} stem render jobs"
    )


if __name__ == "__main__":
    main()
