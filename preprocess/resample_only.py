from __future__ import annotations

import argparse
import os
import stat
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List

import soundfile as sf
import torch
import torchaudio.functional as AF

from instrument_agnostic_amt.data.audio import inspect_audio, read_audio_frames


def _iter_audio_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return

    if not path.exists():
        raise FileNotFoundError(f"入力パスが存在しません: {path}")

    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in {".wav", ".mp3"}:
            yield candidate


def needs_resample(file_path: Path, target_sample_rate: int) -> bool:
    metadata = inspect_audio(file_path)
    return metadata.sample_rate != target_sample_rate


def resample_in_place(file_path: Path, target_sample_rate: int) -> None:
    metadata = sf.info(str(file_path))
    source_sample_rate = int(metadata.samplerate)
    source_mode = stat.S_IMODE(file_path.stat().st_mode)

    waveform_array, _ = read_audio_frames(file_path)
    waveform = torch.from_numpy(waveform_array)
    waveform = AF.resample(
        waveform, orig_freq=source_sample_rate, new_freq=target_sample_rate
    )
    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{file_path.name}.",
        suffix=".tmp",
        dir=file_path.parent,
    )
    os.close(temporary_fd)
    temporary_path = Path(temporary_name)
    try:
        sf.write(
            str(temporary_path),
            waveform.transpose(0, 1).contiguous().numpy(),
            samplerate=target_sample_rate,
            format=metadata.format,
            subtype=metadata.subtype,
        )
        temporary_path.chmod(source_mode)
        temporary_path.replace(file_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="指定したオーディオファイル（単体またはフォルダ配下）を上書きリサンプリングします。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("./stems"),
        help="リサンプリング対象のファイル、またはフォルダ",
    )
    parser.add_argument(
        "--resample-rate",
        type=int,
        default=22050,
        help="上書きリサンプリング先のサンプルレート（Hz）",
    )
    parser.add_argument("--workers", type=int, default=4, help="並列ワーカー数")
    args = parser.parse_args()

    try:
        all_targets: List[Path] = list(_iter_audio_files(args.input))
    except FileNotFoundError as exc:
        print(exc)
        return

    if not all_targets:
        print("リサンプリング対象ファイルが見つかりませんでした。")
        return

    print(f"全ファイル数: {len(all_targets)}")
    print(f"事前チェック中... (target sample_rate={args.resample_rate})")

    targets_to_resample: List[Path] = []
    skipped = 0

    for i, path in enumerate(all_targets, start=1):
        try:
            if needs_resample(path, args.resample_rate):
                targets_to_resample.append(path)
            else:
                skipped += 1
        except Exception as exc:
            print(f"[ERROR] Failed to inspect {path}: {exc}")

        if i % 100 == 0 or i == len(all_targets):
            print(f"Check progress: {i}/{len(all_targets)}")

    if not targets_to_resample:
        print(f"すべて既に {args.resample_rate} Hz でした。 skipped={skipped}")
        return

    print(
        f"リサンプリング対象: {len(targets_to_resample)} / {len(all_targets)} "
        f"(skipped={skipped})"
    )

    worker_count = min(args.workers, len(targets_to_resample)) or 1
    processed = 0
    resampled = 0

    if worker_count == 1:
        for path in targets_to_resample:
            try:
                resample_in_place(path, args.resample_rate)
                resampled += 1
            except Exception as exc:
                print(f"[ERROR] Failed to resample {path}: {exc}")
            finally:
                processed += 1
                if processed % 10 == 0 or processed == len(targets_to_resample):
                    print(f"Resample progress: {processed}/{len(targets_to_resample)}")

        print(
            f"リサンプリングが完了しました。 resampled={resampled}, skipped={skipped}"
        )
        return

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_path = {
            executor.submit(resample_in_place, path, args.resample_rate): path
            for path in targets_to_resample
        }
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                future.result()
                resampled += 1
            except Exception as exc:
                print(f"[ERROR] Failed to resample {path}: {exc}")
            finally:
                processed += 1
                if processed % 10 == 0 or processed == len(targets_to_resample):
                    print(f"Resample progress: {processed}/{len(targets_to_resample)}")

    print(f"リサンプリングが完了しました。 resampled={resampled}, skipped={skipped}")


if __name__ == "__main__":
    main()
