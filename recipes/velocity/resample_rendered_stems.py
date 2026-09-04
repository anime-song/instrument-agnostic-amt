"""Resample rendered velocity training stems in place."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import soundfile as sf
from tqdm import tqdm


VELOCITY_ROOT = (
    Path(__file__).resolve().parents[2] / "instrument_agnostic_amt" / "velocity"
)
DEFAULT_MANIFEST = VELOCITY_ROOT / "artifacts" / "synthetic" / "render_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Atomically resample existing rendered stem WAVs in place."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ffmpeg-executable", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _resolve(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _resolve_ffmpeg(value: Path | None) -> Path:
    if value is not None:
        result = value.expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(f"FFmpeg executable not found: {result}")
        return result
    discovered = shutil.which("ffmpeg")
    if discovered is None:
        raise FileNotFoundError("ffmpeg was not found on PATH")
    return Path(discovered).resolve()


def _resample_one(
    path: Path,
    *,
    sample_rate: int,
    ffmpeg: Path,
) -> tuple[str, int, int]:
    source_info = sf.info(str(path))
    source_rate = int(source_info.samplerate)
    if source_rate == sample_rate and source_info.subtype == "PCM_16":
        return "skipped", int(path.stat().st_size), int(path.stat().st_size)
    expected_frames = int(round(int(source_info.frames) * sample_rate / source_rate))
    temporary = path.with_name(
        f"{path.stem}.resample-{uuid.uuid4().hex}.tmp{path.suffix}"
    )
    source_bytes = int(path.stat().st_size)
    try:
        subprocess.run(
            [
                str(ffmpeg),
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(path),
                "-map_metadata",
                "-1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            check=True,
        )
        target_info = sf.info(str(temporary))
        if (
            int(target_info.samplerate) != sample_rate
            or int(target_info.channels) != int(source_info.channels)
            or abs(int(target_info.frames) - expected_frames) > 2
            or target_info.subtype != "PCM_16"
        ):
            raise RuntimeError(f"Resample verification failed for {path.name}")
        target_bytes = int(temporary.stat().st_size)
        os.replace(temporary, path)
        return "converted", source_bytes, target_bytes
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_manifest(
    path: Path,
    rows: list[dict[str, str]],
    *,
    sample_rate: int,
) -> None:
    for row in rows:
        row["sample_rate"] = str(sample_rate)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.sample_rate <= 0 or args.workers < 1:
        raise ValueError("sample rate and workers must be positive")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be positive")
    manifest = args.manifest.expanduser().resolve()
    artifact_root = manifest.parent
    allowed_root = (artifact_root / "rendered_stems").resolve()
    with manifest.open("r", encoding="utf-8-sig", newline="") as file:
        all_rows = list(csv.DictReader(file))
    selected_rows = all_rows if args.limit is None else all_rows[: args.limit]
    paths: list[Path] = []
    for row in selected_rows:
        path = _resolve(row["rendered_stem_path"], artifact_root)
        try:
            path.relative_to(allowed_root)
        except ValueError as error:
            raise ValueError(
                "Refusing to resample a path outside rendered_stems"
            ) from error
        if path.suffix.casefold() != ".wav":
            raise ValueError(f"Rendered stem is not a WAV: {path.name}")
        if not path.is_file():
            raise FileNotFoundError(f"Rendered stem not found: {path}")
        paths.append(path)
    source_bytes = sum(path.stat().st_size for path in paths)
    source_rates: dict[int, int] = {}
    for path in paths:
        rate = int(sf.info(str(path)).samplerate)
        source_rates[rate] = source_rates.get(rate, 0) + 1
    if args.dry_run:
        print(
            f"Would inspect {len(paths)} stems ({source_bytes} bytes); "
            f"rates={source_rates}; target={args.sample_rate}"
        )
        return

    ffmpeg = _resolve_ffmpeg(args.ffmpeg_executable)
    counts = {"converted": 0, "skipped": 0}
    output_bytes = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _resample_one,
                path,
                sample_rate=args.sample_rate,
                ffmpeg=ffmpeg,
            ): path
            for path in paths
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="resample stems",
        ):
            status, _, target_bytes = future.result()
            counts[status] += 1
            output_bytes += target_bytes

    # A limited conversion intentionally leaves the global manifest unchanged.
    if args.limit is None:
        for path in paths:
            if int(sf.info(str(path)).samplerate) != args.sample_rate:
                raise RuntimeError(f"Final sample-rate verification failed: {path}")
        _write_manifest(manifest, all_rows, sample_rate=args.sample_rate)
    summary: dict[str, Any] = {
        "target_sample_rate": args.sample_rate,
        "selected_stem_count": len(paths),
        "converted_stem_count": counts["converted"],
        "skipped_stem_count": counts["skipped"],
        "source_bytes": source_bytes,
        "output_bytes": output_bytes,
        "manifest_updated": args.limit is None,
    }
    summary_path = artifact_root / "stem_resample_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"Converted {counts['converted']} stems; skipped {counts['skipped']}; "
        f"output bytes: {output_bytes}"
    )


if __name__ == "__main__":
    main()
