from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from .collate import collate_velocity_batch
from .stem_dataset import SyntheticStemVelocityDataset


VELOCITY_ROOT = (
    Path(__file__).resolve().parents[2] / "instrument_agnostic_amt" / "velocity"
)
DEFAULT_ROOT = VELOCITY_ROOT / "artifacts" / "synthetic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect synthetic velocity training batches."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test", "all"),
        default="all",
    )
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--hop-seconds", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = SyntheticStemVelocityDataset(
        args.root,
        split=args.split,
        sample_rate=args.sample_rate,
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        allow_incomplete=args.allow_incomplete,
        max_examples=args.max_examples,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_velocity_batch,
    )
    batch = next(iter(loader))
    print(
        f"examples={len(dataset.examples)}, songs={dataset.song_count}, "
        f"windows={len(dataset)}, skipped_incomplete={dataset.missing_audio_count}"
    )
    print(
        f"audio={tuple(batch['audio'].shape)}, "
        f"notes={tuple(batch['target_velocity'].shape)}, "
        f"stems={tuple(batch['stem_gain_db'].shape)}, "
        f"valid_notes={int(batch['note_mask'].sum())}"
    )


if __name__ == "__main__":
    main()
