"""Evaluate a sequence of AMT checkpoints from already-rendered MIDI files.

The script evaluates all non-drum tracks. This is intentional for the EGDB
solo-guitar clips: inferred notes may be split across several guitar classes,
while the reference MIDI tracks do not carry compatible instrument names.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mir_eval
import numpy as np
import pretty_midi
import torch


METRICS = ("conpoff", "conp", "con")


def extract_notes(path: Path) -> tuple[np.ndarray, np.ndarray]:
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = [
        note
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]
    notes.sort(key=lambda note: (note.start, note.pitch, note.end))
    if not notes:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    intervals = np.asarray([(note.start, note.end) for note in notes], dtype=float)
    pitches = mir_eval.util.midi_to_hz(np.asarray([note.pitch for note in notes]))
    return intervals, pitches


def _scores(tp: int, predicted: int, reference: int) -> dict[str, float | int]:
    precision = tp / predicted if predicted else float(reference == 0)
    recall = tp / reference if reference else float(predicted == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "precision": precision, "recall": recall, "f1": f1}


def evaluate_pair(ref_path: Path, pred_path: Path) -> dict[str, float | int]:
    ref_intervals, ref_pitches = extract_notes(ref_path)
    pred_intervals, pred_pitches = extract_notes(pred_path)

    conpoff = mir_eval.transcription.match_notes(
        ref_intervals,
        ref_pitches,
        pred_intervals,
        pred_pitches,
        onset_tolerance=0.05,
        pitch_tolerance=50.0,
        offset_ratio=0.2,
        offset_min_tolerance=0.05,
    )
    conp = mir_eval.transcription.match_notes(
        ref_intervals,
        ref_pitches,
        pred_intervals,
        pred_pitches,
        onset_tolerance=0.05,
        pitch_tolerance=50.0,
        offset_ratio=None,
    )
    con = mir_eval.util.match_events(
        ref_intervals[:, 0], pred_intervals[:, 0], window=0.05
    )

    row: dict[str, float | int] = {
        "reference_notes": len(ref_intervals),
        "predicted_notes": len(pred_intervals),
    }
    for name, matches in (("conpoff", conpoff), ("conp", conp), ("con", con)):
        for field, value in _scores(
            len(matches), len(pred_intervals), len(ref_intervals)
        ).items():
            row[f"{name}_{field}"] = value
    return row


def tone_for(stem: str) -> str:
    if stem.startswith("JC_jazz_"):
        return "JC Jazz"
    if stem.startswith("Plexi_"):
        return "Plexi"
    if stem.startswith("Boss_Metal_Zone_"):
        return "Boss Metal Zone"
    if stem.startswith("Rockman_Distortion_"):
        return "Rockman Distortion"
    return "Other"


def find_prediction(directory: Path, stem: str) -> Path:
    candidates = [directory / f"{stem}.mid", directory / f"{stem}.midi"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No prediction for {stem!r} in {directory}")


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "clips": len(rows),
        "reference_notes": sum(int(row["reference_notes"]) for row in rows),
        "predicted_notes": sum(int(row["predicted_notes"]) for row in rows),
    }
    for metric in METRICS:
        tp = sum(int(row[f"{metric}_tp"]) for row in rows)
        micro = _scores(
            tp, int(result["predicted_notes"]), int(result["reference_notes"])
        )
        result[f"{metric}_micro_precision"] = micro["precision"]
        result[f"{metric}_micro_recall"] = micro["recall"]
        result[f"{metric}_micro_f1"] = micro["f1"]
        result[f"{metric}_macro_f1"] = float(
            np.mean([float(row[f"{metric}_f1"]) for row in rows])
        )
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_loss(template: str | None, epoch: int) -> float | None:
    if not template:
        return None
    path = Path(template.format(epoch=epoch))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    value = checkpoint.get("loss")
    return float(value) if value is not None else None


def bootstrap_vs_baseline(
    rows_by_epoch: dict[int, list[dict[str, object]]],
    baseline_epoch: int,
    samples: int,
    seed: int,
) -> list[dict[str, object]]:
    baseline = rows_by_epoch[baseline_epoch]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(baseline), size=(samples, len(baseline)))
    output = []
    for epoch, rows in rows_by_epoch.items():
        values = np.asarray([float(row["conpoff_f1"]) for row in rows])
        baseline_values = np.asarray(
            [float(row["conpoff_f1"]) for row in baseline]
        )
        differences = (values - baseline_values)[indices].mean(axis=1)
        observed = float((values - baseline_values).mean())
        output.append(
            {
                "epoch": epoch,
                "baseline_epoch": baseline_epoch,
                "macro_conpoff_f1_difference": observed,
                "bootstrap_ci_low": float(np.percentile(differences, 2.5)),
                "bootstrap_ci_high": float(np.percentile(differences, 97.5)),
                "bootstrap_probability_above_baseline": float(
                    np.mean(differences > 0) + 0.5 * np.mean(differences == 0)
                ),
                "paired_clips_better": int(np.sum(values > baseline_values)),
                "paired_clips_equal": int(np.sum(values == baseline_values)),
                "paired_clips_worse": int(np.sum(values < baseline_values)),
            }
        )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref-dir", type=Path, required=True)
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--pred-template", default="extended_epoch_{epoch}")
    parser.add_argument("--epochs", default="10,20,30,40,50,60,70,80")
    parser.add_argument("--checkpoint-template")
    parser.add_argument("--baseline-epoch", type=int, default=80)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    epochs = [int(value) for value in args.epochs.split(",")]
    refs = sorted([*args.ref_dir.glob("*.mid"), *args.ref_dir.glob("*.midi")])
    if not refs:
        raise FileNotFoundError(f"No reference MIDI files in {args.ref_dir}")

    rows_by_epoch: dict[int, list[dict[str, object]]] = {}
    file_rows: list[dict[str, object]] = []
    for epoch in epochs:
        pred_dir = args.pred_root / args.pred_template.format(epoch=epoch)
        epoch_rows = []
        for ref_path in refs:
            row: dict[str, object] = {
                "epoch": epoch,
                "file": ref_path.stem,
                "tone": tone_for(ref_path.stem),
            }
            row.update(evaluate_pair(ref_path, find_prediction(pred_dir, ref_path.stem)))
            epoch_rows.append(row)
            file_rows.append(row)
        rows_by_epoch[epoch] = epoch_rows

    summary_rows = []
    tone_rows = []
    for epoch, rows in rows_by_epoch.items():
        summary = {"epoch": epoch, "train_loss": checkpoint_loss(args.checkpoint_template, epoch)}
        summary.update(summarize(rows))
        summary_rows.append(summary)
        for tone in sorted({str(row["tone"]) for row in rows}):
            tone_summary: dict[str, object] = {"epoch": epoch, "tone": tone}
            tone_summary.update(summarize([row for row in rows if row["tone"] == tone]))
            tone_rows.append(tone_summary)

    bootstrap_rows = bootstrap_vs_baseline(
        rows_by_epoch, args.baseline_epoch, args.bootstrap_samples, args.seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "summary.csv", summary_rows)
    write_csv(args.output_dir / "tone_summary.csv", tone_rows)
    write_csv(args.output_dir / "file_metrics.csv", file_rows)
    write_csv(args.output_dir / "bootstrap_vs_baseline.csv", bootstrap_rows)
    metadata = {
        "epochs": epochs,
        "reference_directory": str(args.ref_dir),
        "prediction_template": str(args.pred_root / args.pred_template),
        "baseline_epoch": args.baseline_epoch,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "metric_definitions": {
            "COnPOff": "onset within 50 ms, pitch within 50 cents, offset within max(20%, 50 ms)",
            "COnP": "onset within 50 ms and pitch within 50 cents; offset ignored",
            "COn": "onset within 50 ms; pitch and offset ignored",
        },
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("epoch train_loss COnPOff_micro COnPOff_macro COnP_micro COn_micro")
    for row in summary_rows:
        print(
            f"{row['epoch']:>5} {row['train_loss']:.6f} "
            f"{row['conpoff_micro_f1']:.6f} {row['conpoff_macro_f1']:.6f} "
            f"{row['conp_micro_f1']:.6f} {row['con_micro_f1']:.6f}"
        )


if __name__ == "__main__":
    main()
