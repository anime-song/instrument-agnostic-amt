from __future__ import annotations

import argparse
import json
from pathlib import Path

from .synthesis.config import load_synthetic_config
from .synthesis.mix import (
    mix_rendered_stems,
    write_dataset_manifest,
)


VELOCITY_ROOT = (
    Path(__file__).resolve().parents[2] / "instrument_agnostic_amt" / "velocity"
)
DEFAULT_ROOT = VELOCITY_ROOT / "artifacts" / "synthetic"
DEFAULT_CONFIG = VELOCITY_ROOT / "configs" / "synthetic.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply stem gains and assemble rendered synthetic mixtures."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    config = load_synthetic_config(args.config)
    rows, summary = mix_rendered_stems(
        root / "render_manifest.csv",
        root / "examples.csv",
        peak_limit_dbfs=config.mix_peak_limit_dbfs,
        output_sample_rate=config.mixture_sample_rate,
        overwrite=args.overwrite,
        skip_missing=args.skip_missing,
    )
    write_dataset_manifest(root / "dataset_manifest.csv", rows)
    (root / "assembly_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Assembled {summary['mixed_example_count']} mixtures; "
        f"missing stems: {summary['missing_stem_count']}"
    )


if __name__ == "__main__":
    main()
