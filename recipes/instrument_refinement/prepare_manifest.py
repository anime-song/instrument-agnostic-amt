"""学習用 manifest を作る CLI。

データセットの置き場所は人によって違うので、パスは gitignore 済みのローカル設定
（既定: リポジトリ直下の instrument_refinement_datasets.local.json）に書く。
設定ファイルの書き方は data/manifest.py の説明を参照。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import build_refinement_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "instrument_refinement_datasets.local.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an instrument-refinement manifest from a local config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output path declared in the local config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        raise FileNotFoundError(
            f"Local dataset config not found: {args.config}. Keep dataset paths in this gitignored file."
        )
    summary = build_refinement_manifest(args.config, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
