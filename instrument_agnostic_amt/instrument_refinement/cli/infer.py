"""単体推論の CLI。

ステム音声 1 本とその MIDI 1 本を渡し、楽器を振り直した MIDI を書き出す。
複数ステムをまとめて処理したい場合は、リポジトリ直下の infer_stem.py の
パイプライン（refine_instruments オプション）を使う方がモデルを読み直さずに済む。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from ..inference.refine import refine_midi_instruments


HF_CHECKPOINT_BASE_URL = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main"
DEFAULT_REFINEMENT_CHECKPOINT_FILENAME = "best_instrument_refinement.pth"
REFINEMENT_ROOT = Path(__file__).resolve().parent.parent


def ensure_refinement_checkpoint(checkpoint_path: Path | str | None = None) -> Path:
    """Instrument Refinement のチェックポイントを解決し、無ければ Hugging Face から取得する。

    未指定のときの探索順は、配布物の置き場所 -> ローカル学習の出力先 -> ダウンロード。
    """
    if checkpoint_path is not None and str(checkpoint_path).strip() not in (
        "",
        "DEFAULT",
    ):
        target_path = Path(checkpoint_path)
    else:
        candidates = [
            Path("checkpoints") / DEFAULT_REFINEMENT_CHECKPOINT_FILENAME,
            REFINEMENT_ROOT / "artifacts" / "checkpoints" / "best_model.pth",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        target_path = candidates[0]

    if not target_path.is_file():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF_CHECKPOINT_BASE_URL}/{DEFAULT_REFINEMENT_CHECKPOINT_FILENAME}?download=true"
        print(f"Instrument refinement checkpoint not found at {target_path}. Downloading from Hugging Face...")
        torch.hub.download_url_to_file(url, str(target_path))

    return target_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reclassify MIDI instruments from a separated stem and AMT MIDI.")
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--midi", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-midi", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    # ステム名を渡すと楽器候補が絞られる。分からない場合は省略（全クラスが候補）。
    parser.add_argument("--stem-name", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--stride-seconds", type=float, default=4.0)
    parser.add_argument("--window-batch-size", type=int, default=1)
    # cluster: 音色ごとに楽器を分ける / single: ステム全体を 1 楽器とみなす。
    parser.add_argument("--mode", choices=("single", "cluster"), default="cluster")
    parser.add_argument("--cluster-distance", type=float, default=0.25)
    parser.add_argument("--min-cluster-notes", type=int, default=2)
    parser.add_argument("--switch-penalty", type=float, default=2.0)
    parser.add_argument(
        "--global-logit-weight",
        type=float,
        default=0.0,
        help="Optional whole-window bias added to each timbre cluster.",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-stem-mask", action="store_true")
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = refine_midi_instruments(
        args.audio,
        args.midi,
        checkpoint_path=ensure_refinement_checkpoint(args.checkpoint),
        output_midi_path=args.output_midi,
        output_json_path=args.output_json,
        stem_name=args.stem_name,
        device=args.device,
        window_seconds=args.window_seconds,
        stride_seconds=args.stride_seconds,
        window_batch_size=args.window_batch_size,
        mode=args.mode,
        cluster_distance=args.cluster_distance,
        min_cluster_notes=args.min_cluster_notes,
        switch_penalty=args.switch_penalty,
        global_logit_weight=args.global_logit_weight,
        top_k=args.top_k,
        use_stem_mask=not args.no_stem_mask,
        disable_tqdm=args.disable_tqdm,
    )
    # レポート全文は長いので、標準出力には要点だけ出す（全文は --output-json へ）。
    summary = {
        key: report[key]
        for key in (
            "note_count",
            "window_count",
            "cluster_count",
            "window_flip_count_before_smoothing",
            "window_flip_count_after_smoothing",
            "global_candidates",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
