from __future__ import annotations

import argparse
import json
import logging
import math
import wave
from bisect import bisect_right
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from .datasets.beat_pretrain import (
    TickSecondConverter,
    _iter_meter_sections,
    _read_midi_meta_events,
)

logger = logging.getLogger(__name__)


def build_beat_label_measures(
    ticks_per_beat: int,
    max_tick: int,
    tempo_changes: Sequence[object],
    time_signatures: Sequence[object],
) -> list[dict[str, float | int]]:
    """MIDI のメタイベントから小節ごとのビートラベル情報を構築する。"""
    converter = TickSecondConverter.from_tempo_changes(
        ticks_per_beat=ticks_per_beat,
        tempo_changes=tempo_changes,  # type: ignore[arg-type]
    )

    measures: list[dict[str, float | int]] = []
    for start_tick, end_tick, meter_num, meter_den in _iter_meter_sections(
        time_signatures=time_signatures,  # type: ignore[arg-type]
        end_tick=max_tick,
    ):
        beat_step = Fraction(ticks_per_beat * 4, meter_den)
        if beat_step <= 0:
            continue
        measure_step = beat_step * int(meter_num)
        cur_tick = Fraction(start_tick, 1)

        while cur_tick < end_tick:
            second_value = converter.tick_to_seconds(cur_tick)

            # 小節内の各ビート (拍) の厳密な秒数位置を算出 (rit. 等の非線形テンポに対応)
            beat_times: list[float] = []
            for beat_idx in range(meter_num):
                beat_tick = cur_tick + beat_step * beat_idx
                # 小節の終端 (end_tick) を完全に超えるビートは不完全小節のはみ出しなので除外する
                # ただし最初のビート (beat_idx == 0) は必ず含める
                if beat_tick >= end_tick and beat_idx > 0:
                    break
                beat_second = converter.tick_to_seconds(beat_tick)
                beat_times.append(float(beat_second))

            # 現在の tick における tempo (microseconds per quarter note) を取得
            tempo_index = max(
                0, bisect_right(converter.start_ticks, float(cur_tick)) - 1
            )
            tempo_microseconds = converter.tempos[tempo_index]
            tempo_bpm = 60_000_000.0 / float(tempo_microseconds)

            measures.append(
                {
                    "downbeat_sec": float(second_value),
                    "time_sig_num": meter_num,
                    "time_sig_den": meter_den,
                    "tempo_bpm": float(tempo_bpm),
                    "beat_times": beat_times,
                }
            )
            cur_tick += measure_step

    return measures


def validate_beat_measures(measures: list[dict[str, float | int]]) -> list[str]:
    """生成された小節情報（拍子・テンポ・ダウンビート位置・ビート位置）の整合性を検証する。"""
    errors: list[str] = []
    if not measures:
        errors.append("小節データが存在しません。")
        return errors

    previous_downbeat = -1.0
    for index, measure in enumerate(measures):
        downbeat_sec = float(measure["downbeat_sec"])
        meter_num = int(measure["time_sig_num"])
        meter_den = int(measure["time_sig_den"])
        tempo_bpm = float(measure["tempo_bpm"])

        if downbeat_sec < 0.0:
            errors.append(
                f"小節 #{index + 1}: downbeat_sec が負の値です ({downbeat_sec})"
            )
        if downbeat_sec <= previous_downbeat:
            errors.append(
                f"小節 #{index + 1}: downbeat_sec が単調増加していません ({downbeat_sec} <= {previous_downbeat})"
            )
        if meter_num <= 0 or meter_den <= 0:
            errors.append(
                f"小節 #{index + 1}: 不正な拍子記号です ({meter_num}/{meter_den})"
            )
        if tempo_bpm <= 0.0:
            errors.append(f"小節 #{index + 1}: 不正なテンポです ({tempo_bpm} BPM)")

        raw_beat_times = measure.get("beat_times")
        if raw_beat_times is not None:
            if not isinstance(raw_beat_times, (list, tuple)):
                errors.append(f"小節 #{index + 1}: beat_times の形式が不正です")
            elif len(raw_beat_times) == 0:
                errors.append(f"小節 #{index + 1}: beat_times が空です")
            elif len(raw_beat_times) > meter_num:
                errors.append(
                    f"小節 #{index + 1}: beat_times の長さ ({len(raw_beat_times)}) が拍子数 ({meter_num}) を超過しています"
                )

        previous_downbeat = downbeat_sec

    return errors


def export_audacity_labels(
    measures: list[dict[str, float | int]],
    output_txt_path: Path,
) -> None:
    """Audacity で波形や MIDI と重ねて確認するためのラベルテキストファイルを出力する。"""
    output_txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    for index, measure in enumerate(measures):
        start_sec = float(measure["downbeat_sec"])
        meter_num = int(measure["time_sig_num"])
        meter_den = int(measure["time_sig_den"])
        tempo_bpm = float(measure["tempo_bpm"])

        # ダウンビート (ラベル: 1)
        lines.append(f"{start_sec:.6f}\t{start_sec:.6f}\t1")

        # 小節内のビート (ラベル: 2)
        raw_beat_times = measure.get("beat_times")
        if (
            isinstance(raw_beat_times, (list, tuple))
            and len(raw_beat_times) == meter_num
        ):
            for beat_sec in raw_beat_times[1:]:
                lines.append(f"{float(beat_sec):.6f}\t{float(beat_sec):.6f}\t2")
        else:
            if index + 1 < len(measures):
                end_sec = float(measures[index + 1]["downbeat_sec"])
            elif tempo_bpm > 0.0:
                measure_duration = meter_num * (4.0 / meter_den) * (60.0 / tempo_bpm)
                end_sec = start_sec + measure_duration
            else:
                end_sec = start_sec + 4.0

            measure_duration = end_sec - start_sec
            for beat_idx in range(1, meter_num):
                beat_sec = start_sec + (measure_duration * beat_idx / meter_num)
                lines.append(f"{beat_sec:.6f}\t{beat_sec:.6f}\t2")

    with open(output_txt_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def create_silent_wave_file(
    output_path: Path,
    duration_sec: float,
    sample_rate: int = 16000,
) -> None:
    """指定された長さ（秒）の無音 WAV ファイルを出力する。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(round(duration_sec * sample_rate))
    silent_frames = b"\x00\x00" * frame_count

    with wave.open(str(output_path), "wb") as wave_file:
        wave_file.setnchannels(1)
        wave_file.setsampwidth(2)
        wave_file.setframerate(sample_rate)
        wave_file.writeframes(silent_frames)


def convert_midi_dir_to_beat_dataset(
    midi_dir: Path,
    output_dataset_dir: Path,
    *,
    source_audio_dir: Path | None = None,
    sample_rate: int = 16000,
    overwrite: bool = False,
    export_audacity: bool = False,
) -> tuple[int, int]:
    """MIDI ディレクトリを beat_dataset 形式（label/ および audio/）へ変換する。"""
    midi_dir = Path(midi_dir).resolve()
    output_dataset_dir = Path(output_dataset_dir).resolve()

    label_dir = output_dataset_dir / "label"
    audio_dir = output_dataset_dir / "audio"
    audacity_dir = output_dataset_dir / "audacity_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    midi_files = sorted(
        path
        for path in midi_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in (".mid", ".midi")
    )

    success_count = 0
    skip_count = 0

    for midi_path in midi_files:
        stem_name = midi_path.stem
        json_label_path = label_dir / f"{stem_name}.beat.beats.json"
        target_audio_path = audio_dir / f"{stem_name}.wav"

        if not overwrite and json_label_path.exists() and target_audio_path.exists():
            skip_count += 1
            continue

        try:
            (
                ticks_per_beat,
                max_tick,
                tempo_changes,
                time_signatures,
            ) = _read_midi_meta_events(midi_path)

            if max_tick <= 0:
                logger.warning("スキップ (再生時間が0以下): %s", midi_path)
                skip_count += 1
                continue

            measures = build_beat_label_measures(
                ticks_per_beat=ticks_per_beat,
                max_tick=max_tick,
                tempo_changes=tempo_changes,
                time_signatures=time_signatures,
            )

            validation_errors = validate_beat_measures(measures)
            if validation_errors:
                logger.warning(
                    "検証エラーのためスキップ (%s): %s", midi_path, validation_errors
                )
                skip_count += 1
                continue

            # JSON ラベルの書き出し
            label_data = {"measures": measures}
            with open(json_label_path, "w", encoding="utf-8") as json_file:
                json.dump(label_data, json_file, indent=2, ensure_ascii=False)

            # Audacity ラベルの出力 (オプション)
            if export_audacity:
                audacity_txt_path = audacity_dir / f"{stem_name}.beats.txt"
                export_audacity_labels(measures, audacity_txt_path)

            # 音声ファイルの作成またはコピー
            converter = TickSecondConverter.from_tempo_changes(
                ticks_per_beat=ticks_per_beat,
                tempo_changes=tempo_changes,
            )
            duration_sec = converter.tick_to_seconds(max_tick)

            source_audio_file = (
                source_audio_dir / f"{stem_name}.wav" if source_audio_dir else None
            )

            if source_audio_file and source_audio_file.exists():
                import shutil

                shutil.copy2(source_audio_file, target_audio_path)
            elif overwrite or not target_audio_path.exists():
                create_silent_wave_file(
                    output_path=target_audio_path,
                    duration_sec=duration_sec,
                    sample_rate=sample_rate,
                )

            success_count += 1

        except Exception as error:
            logger.error("処理エラー (%s): %s", midi_path, error)
            skip_count += 1

    return success_count, skip_count


def parse_command_line_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIDIファイル群から beat_dataset 用のJSONラベルとダミーWAV音声を自動生成します。"
    )
    parser.add_argument(
        "--midi_dir",
        type=Path,
        required=True,
        help="入力MIDIファイルが格納されているディレクトリのパス",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("beat_chord_dataset/beat_dataset"),
        help="出力先 beat_dataset ディレクトリのパス",
    )
    parser.add_argument(
        "--source_audio_dir",
        type=Path,
        default=None,
        help="実音声をコピーして使用する場合のWAV音声ディレクトリのパス",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=16000,
        help="ダミーWAV生成時のサンプルレート",
    )
    parser.add_argument(
        "--export_audacity",
        action="store_true",
        help="Audacity確認用のビートラベルテキスト(.txt)も出力します",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="既存のラベル・音声ファイルを上書きします",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_command_line_arguments()

    success_count, skip_count = convert_midi_dir_to_beat_dataset(
        midi_dir=args.midi_dir,
        output_dataset_dir=args.output_dir,
        source_audio_dir=args.source_audio_dir,
        sample_rate=args.sample_rate,
        overwrite=args.overwrite,
        export_audacity=args.export_audacity,
    )

    logger.info(
        "変換完了: 成功=%d 曲, スキップ=%d 曲 -> 出力先: %s",
        success_count,
        skip_count,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
