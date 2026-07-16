"""
音声とMIDIのペアデータを同じ速度倍率でtime stretchする共通前処理。

このスクリプトは単体で実行できるようにしてあり、特定データセットのプリセットは持たない。
音声はrubberbandで時間伸縮し、MIDIはtempo mapを同じ比率だけ変換する。
既定では `.wav/.flac` と `.mid/.midi` を扱い、`_pitch_` / `_stretch_` / `_swing_`
付きファイルは再入力しない。
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TEMPO = 500_000
MAX_TEMPO_VALUE = 16_777_215
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac")
DEFAULT_MIDI_EXTENSIONS = (".mid", ".midi")
DEFAULT_EXCLUDE_FILENAME_MARKERS = ("_pitch_", "_stretch_", "_swing_")
SCRIPT_PATH_EXAMPLE = "preprocess/time_stretch_paired_dataset.py"
INPUT_AUDIO_EXAMPLE = "path/to/audio_dir/song.wav"
INPUT_MIDI_EXAMPLE = "path/to/midi_dir/song.mid"


@dataclass(frozen=True)
class RuntimeTimeStretchConfig:
    """CLI解決後の実行設定を保持する。"""

    generate_audio: bool
    generate_midi: bool
    audio_dir: Path
    midi_dir: Path
    output_audio_dir: Path
    output_midi_dir: Path
    audio_extensions: tuple[str, ...]
    midi_extensions: tuple[str, ...]
    exclude_filename_markers: tuple[str, ...]
    speed_rates: tuple[float, ...]
    workers: int
    overwrite: bool

    @classmethod
    def from_arguments(cls, arguments: argparse.Namespace) -> RuntimeTimeStretchConfig:
        """引数を後段の処理が扱いやすい形へ正規化する。"""
        generate_audio = not arguments.midi_only
        generate_midi = not arguments.audio_only

        if generate_audio and arguments.audio_dir is None:
            raise ValueError("--audio_dir は音声処理時に必須です。")
        if generate_midi and arguments.midi_dir is None:
            raise ValueError("--midi_dir はMIDI処理時に必須です。")

        audio_dir = (
            arguments.audio_dir.resolve()
            if arguments.audio_dir is not None
            else Path(".").resolve()
        )
        midi_dir = (
            arguments.midi_dir.resolve()
            if arguments.midi_dir is not None
            else Path(".").resolve()
        )

        runtime_config = cls(
            generate_audio=generate_audio,
            generate_midi=generate_midi,
            audio_dir=audio_dir,
            midi_dir=midi_dir,
            output_audio_dir=(
                arguments.output_audio_dir.resolve()
                if arguments.output_audio_dir is not None
                else audio_dir
            ),
            output_midi_dir=(
                arguments.output_midi_dir.resolve()
                if arguments.output_midi_dir is not None
                else midi_dir
            ),
            audio_extensions=cls.normalize_extensions(
                arguments.audio_ext,
                option_name="--audio-ext",
            ),
            midi_extensions=cls.normalize_extensions(
                arguments.midi_ext,
                option_name="--midi-ext",
            ),
            exclude_filename_markers=DEFAULT_EXCLUDE_FILENAME_MARKERS,
            speed_rates=cls.build_speed_rates(
                arguments.min_speed,
                arguments.max_speed,
                arguments.speed_step,
            ),
            workers=arguments.workers,
            overwrite=arguments.overwrite,
        )
        runtime_config.validate(
            min_speed=arguments.min_speed,
            max_speed=arguments.max_speed,
            speed_step=arguments.speed_step,
        )
        return runtime_config

    @staticmethod
    def normalize_extensions(
        extensions: list[str] | tuple[str, ...],
        *,
        option_name: str,
    ) -> tuple[str, ...]:
        """拡張子指定を `.wav` 形式へ正規化し、重複を除く。"""
        normalized_extensions: list[str] = []
        for extension in extensions:
            normalized = extension.strip().lower()
            if not normalized:
                continue
            if not normalized.startswith("."):
                normalized = f".{normalized}"
            if normalized not in normalized_extensions:
                normalized_extensions.append(normalized)

        if not normalized_extensions:
            raise ValueError(f"{option_name} には1つ以上の拡張子を指定してください。")
        return tuple(normalized_extensions)

    @staticmethod
    def build_speed_rates(
        min_speed: float,
        max_speed: float,
        speed_step: float,
    ) -> tuple[float, ...]:
        """生成対象の速度倍率一覧を作る。1.0 は元データなので除外する。"""
        min_speed_decimal = Decimal(str(min_speed))
        max_speed_decimal = Decimal(str(max_speed))
        speed_step_decimal = Decimal(str(speed_step))
        one_decimal = Decimal("1.0")
        epsilon = Decimal("0.0000001")

        if min_speed_decimal > max_speed_decimal:
            raise ValueError("--min_speed は --max_speed 以下にしてください。")

        speed_rates: list[float] = []
        current = min_speed_decimal
        while current <= max_speed_decimal + epsilon:
            if abs(current - one_decimal) > epsilon:
                speed_rates.append(float(current))
            current += speed_step_decimal

        if not speed_rates:
            raise ValueError("指定範囲から 1.0 以外の速度倍率を生成できません。")
        return tuple(speed_rates)

    def validate(
        self,
        *,
        min_speed: float,
        max_speed: float,
        speed_step: float,
    ) -> None:
        """実行前に最低限の前提を検証する。"""
        if self.workers < 1:
            raise ValueError("--workers は 1 以上にしてください。")
        if min_speed <= 0.0 or max_speed <= 0.0:
            raise ValueError("--min_speed と --max_speed は 0 より大きくしてください。")
        if speed_step <= 0.0:
            raise ValueError("--speed_step は 0 より大きくしてください。")
        if self.generate_audio and not self.audio_dir.exists():
            raise FileNotFoundError(
                f"入力音声ディレクトリが見つかりません: {self.audio_dir}"
            )
        if self.generate_midi and not self.midi_dir.exists():
            raise FileNotFoundError(
                f"入力MIDIディレクトリが見つかりません: {self.midi_dir}"
            )


@dataclass(frozen=True)
class AudioTask:
    """音声time stretch用の 1 タスクを表す。"""

    input_path: Path
    output_dir: Path
    speed_rates: tuple[float, ...]


@dataclass(frozen=True)
class MidiTask:
    """MIDI time stretch用の 1 タスクを表す。"""

    input_path: Path
    output_dir: Path
    speed_rates: tuple[float, ...]


@dataclass
class TaskResult:
    """1ファイル分の処理結果を集約する。"""

    success_count: int = 0
    skipped_count: int = 0
    error_messages: list[str] = field(default_factory=list)


def format_rate_tag(speed_rate: float) -> str:
    """ファイル名に埋め込む速度倍率タグを返す。"""
    return f"{speed_rate:.2f}".replace(".", "p")


def build_example_output_path(example_path: str, speed_rate: float) -> str:
    """ヘルプ表示用の出力パス例を組み立てる。"""
    path = Path(example_path)
    output_path = path.with_name(
        f"{path.stem}_stretch_{format_rate_tag(speed_rate)}x{path.suffix}"
    )
    return output_path.as_posix()


def parse_arguments() -> argparse.Namespace:
    """コマンドライン引数を読み取る。"""
    parser = argparse.ArgumentParser(
        description="音声とMIDIを同じ速度倍率でtime stretchする",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "入力例:\n"
            f"  {INPUT_AUDIO_EXAMPLE}\n"
            f"  {INPUT_MIDI_EXAMPLE}\n\n"
            "出力例:\n"
            f"  {build_example_output_path(INPUT_AUDIO_EXAMPLE, 0.80)}\n"
            f"  {build_example_output_path(INPUT_MIDI_EXAMPLE, 0.80)}\n\n"
            "使い方:\n"
            f"  python {SCRIPT_PATH_EXAMPLE} --audio_dir ./audio --midi_dir ./midi\n"
            f"  python {SCRIPT_PATH_EXAMPLE} --midi_dir ./midi --midi-only\n"
            f"  python {SCRIPT_PATH_EXAMPLE} --audio_dir ./audio --audio-only --workers 4\n\n"
            "倍率の意味:\n"
            "  0.80 = 20% 遅くなる（長さは 1.25 倍）\n"
            "  1.25 = 25% 速くなる（長さは 0.80 倍）"
        ),
    )
    parser.add_argument(
        "--audio_dir", type=Path, default=None, help="入力音声ディレクトリ"
    )
    parser.add_argument(
        "--midi_dir", type=Path, default=None, help="入力MIDIディレクトリ"
    )
    parser.add_argument(
        "--audio-ext",
        nargs="+",
        default=list(DEFAULT_AUDIO_EXTENSIONS),
        metavar="EXTENSIONS",
        help="処理対象に含める音声拡張子。例: .wav .flac",
    )
    parser.add_argument(
        "--midi-ext",
        nargs="+",
        default=list(DEFAULT_MIDI_EXTENSIONS),
        metavar="EXTENSIONS",
        help="処理対象に含めるMIDI拡張子。例: .mid .midi",
    )
    parser.add_argument(
        "--output_audio_dir",
        type=Path,
        default=None,
        help="出力音声ディレクトリ。未指定なら audio_dir に保存",
    )
    parser.add_argument(
        "--output_midi_dir",
        type=Path,
        default=None,
        help="出力MIDIディレクトリ。未指定なら midi_dir に保存",
    )
    parser.add_argument(
        "--min_speed",
        type=float,
        default=0.80,
        help="生成する最小速度倍率。0.80 は 20%% 遅くなる",
    )
    parser.add_argument(
        "--max_speed",
        type=float,
        default=1.25,
        help="生成する最大速度倍率。1.25 は 25%% 速くなる",
    )
    parser.add_argument(
        "--speed_step", type=float, default=0.05, help="速度倍率の刻み幅"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="並列処理のワーカー数",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="既存の生成ファイルを上書きする"
    )
    parser.add_argument("--audio-only", action="store_true", help="音声だけを生成する")
    parser.add_argument("--midi-only", action="store_true", help="MIDIだけを生成する")
    arguments = parser.parse_args()

    if arguments.audio_only and arguments.midi_only:
        raise ValueError("--audio-only と --midi-only は同時に指定できません。")
    return arguments


@lru_cache(maxsize=1)
def get_pyrubberband():
    """pyrubberbandとbackendの利用可否を確認してプロセスごとにキャッシュする。"""
    try:
        import pyrubberband
    except ImportError as exception:
        raise ImportError(
            "pyrubberbandがインストールされていません。"
            " `pip install pyrubberband` を実行してください。"
        ) from exception

    try:
        pyrubberband.time_stretch(np.zeros(64, dtype=np.float32), 16000, 1.0)
    except Exception as exception:
        raise RuntimeError(
            "pyrubberbandはimportできましたが、rubberband backendを利用できません。"
            " rubberband CLIの導入状態を確認してください。"
        ) from exception

    return pyrubberband


def initialize_audio_worker() -> None:
    """各音声ワーカープロセスでpyrubberbandを初期化する。"""
    get_pyrubberband()


def build_output_path(output_dir: Path, input_path: Path, speed_rate: float) -> Path:
    """出力ファイルパスを命名規則どおりに組み立てる。"""
    return (
        output_dir
        / f"{input_path.stem}_stretch_{format_rate_tag(speed_rate)}x{input_path.suffix}"
    )


def collect_pending_speed_rates(
    output_dir: Path,
    input_path: Path,
    speed_rates: tuple[float, ...],
    *,
    overwrite: bool,
) -> tuple[list[float], int]:
    """未生成または上書き対象の速度倍率だけを抽出する。"""
    pending_speed_rates: list[float] = []
    skipped_count = 0

    for speed_rate in speed_rates:
        output_path = build_output_path(output_dir, input_path, speed_rate)
        if overwrite or not output_path.exists():
            pending_speed_rates.append(speed_rate)
        else:
            skipped_count += 1

    return pending_speed_rates, skipped_count


def get_temporary_output_path(output_path: Path) -> Path:
    """拡張子を保った一時ファイルパスを返す。"""
    return output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")


def scale_tempo(tempo: int, speed_rate: float) -> int:
    """速度倍率に合わせてtempo値を変換し、MIDIの表現可能範囲を検証する。"""
    if speed_rate <= 0.0:
        raise ValueError("speed_rate は 0 より大きくしてください。")

    scaled_tempo = int(round(int(tempo) / float(speed_rate)))
    if not 1 <= scaled_tempo <= MAX_TEMPO_VALUE:
        raise ValueError(
            "time stretch後のtempoがMIDIの表現可能範囲外です: "
            f"tempo={tempo}, speed_rate={speed_rate}, scaled_tempo={scaled_tempo}"
        )
    return scaled_tempo


def has_tempo_at_tick_zero(midi_file: mido.MidiFile) -> bool:
    """いずれかのトラックのtick 0にtempoイベントがあるかを返す。"""
    for track in midi_file.tracks:
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            if absolute_tick > 0:
                break
            if message.type == "set_tempo":
                return True
    return False


def build_time_stretched_midi(
    midi_file: mido.MidiFile,
    speed_rate: float,
) -> mido.MidiFile:
    """
    MIDIを指定速度倍率に合わせてtime stretchした新しいMIDIを返す。

    delta tick、ticks_per_beat、tempo mapの位置を保持し、tempo値だけを変換する。
    これによりイベントを再量子化せず、実時間を音声と同じ倍率だけ伸縮する。
    """
    stretched_midi = mido.MidiFile(
        type=midi_file.type,
        ticks_per_beat=midi_file.ticks_per_beat,
        charset=midi_file.charset,
        clip=midi_file.clip,
    )

    for track in midi_file.tracks:
        new_track = mido.MidiTrack()
        for message in track:
            if message.type == "set_tempo":
                new_track.append(
                    message.copy(tempo=scale_tempo(message.tempo, speed_rate))
                )
                continue
            new_track.append(message.copy())
        stretched_midi.tracks.append(new_track)

    if not stretched_midi.tracks:
        stretched_midi.tracks.append(mido.MidiTrack())
    if not has_tempo_at_tick_zero(midi_file):
        stretched_midi.tracks[0].insert(
            0,
            mido.MetaMessage(
                "set_tempo",
                tempo=scale_tempo(DEFAULT_TEMPO, speed_rate),
                time=0,
            ),
        )
    return stretched_midi


def write_stretched_audio(
    output_path: Path,
    stretched_audio: np.ndarray,
    sample_rate: int,
    audio_format: str,
    audio_subtype: str | None,
) -> None:
    """一時ファイル経由で音声を書き込む。"""
    temp_output_path = get_temporary_output_path(output_path)
    try:
        audio_write_arguments = {"format": audio_format}
        if audio_subtype:
            audio_write_arguments["subtype"] = audio_subtype

        sf.write(
            str(temp_output_path),
            stretched_audio,
            sample_rate,
            **audio_write_arguments,
        )
        temp_output_path.replace(output_path)
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise


def write_stretched_midi(output_path: Path, midi_data: mido.MidiFile) -> None:
    """一時ファイル経由でMIDIを書き込む。"""
    temp_output_path = get_temporary_output_path(output_path)
    try:
        midi_data.save(str(temp_output_path))
        temp_output_path.replace(output_path)
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise


def process_audio_task(task: AudioTask, overwrite: bool) -> TaskResult:
    """1つの音声ファイルから複数のtime stretch音声を作る。"""
    result = TaskResult()
    pending_speed_rates, result.skipped_count = collect_pending_speed_rates(
        task.output_dir,
        task.input_path,
        task.speed_rates,
        overwrite=overwrite,
    )
    if not pending_speed_rates:
        return result

    pyrubberband = get_pyrubberband()
    audio_info = sf.info(str(task.input_path))
    audio_data, sample_rate = sf.read(
        str(task.input_path),
        dtype="float32",
        always_2d=True,
    )
    task.output_dir.mkdir(parents=True, exist_ok=True)

    for speed_rate in pending_speed_rates:
        output_path = build_output_path(task.output_dir, task.input_path, speed_rate)
        try:
            stretched_audio = pyrubberband.time_stretch(
                audio_data,
                sample_rate,
                speed_rate,
            ).astype(np.float32, copy=False)
            write_stretched_audio(
                output_path=output_path,
                stretched_audio=stretched_audio,
                sample_rate=sample_rate,
                audio_format=audio_info.format,
                audio_subtype=audio_info.subtype,
            )
            result.success_count += 1
        except (OSError, RuntimeError, ValueError, sf.LibsndfileError) as exception:
            result.error_messages.append(
                f"{task.input_path.name} speed={speed_rate:.2f}: {exception}"
            )

    return result


def process_midi_task(task: MidiTask, overwrite: bool) -> TaskResult:
    """1つのMIDIファイルから複数のtime stretch MIDIを作る。"""
    result = TaskResult()
    pending_speed_rates, result.skipped_count = collect_pending_speed_rates(
        task.output_dir,
        task.input_path,
        task.speed_rates,
        overwrite=overwrite,
    )
    if not pending_speed_rates:
        return result

    try:
        midi_data = mido.MidiFile(str(task.input_path))
    except (OSError, RuntimeError, ValueError) as exception:
        result.error_messages.append(f"{task.input_path.name} load: {exception}")
        return result

    task.output_dir.mkdir(parents=True, exist_ok=True)
    for speed_rate in pending_speed_rates:
        output_path = build_output_path(task.output_dir, task.input_path, speed_rate)
        try:
            stretched_midi = build_time_stretched_midi(
                midi_file=midi_data,
                speed_rate=speed_rate,
            )
            write_stretched_midi(output_path, stretched_midi)
            result.success_count += 1
        except (OSError, RuntimeError, ValueError) as exception:
            result.error_messages.append(
                f"{task.input_path.name} speed={speed_rate:.2f}: {exception}"
            )

    return result


class TimeStretchDatasetRunner:
    """入出力収集と並列実行をまとめて担当する。"""

    def __init__(self, runtime_config: RuntimeTimeStretchConfig) -> None:
        self.runtime_config = runtime_config

    def run(self) -> None:
        """CLIの本体処理を順番に実行する。"""
        self.log_runtime_summary()

        audio_files, midi_files = self.collect_input_files()
        if self.runtime_config.generate_audio and self.runtime_config.generate_midi:
            self.log_pairing_summary(audio_files, midi_files)

        if self.runtime_config.generate_audio:
            self.run_audio(audio_files)
        if self.runtime_config.generate_midi:
            self.run_midi(midi_files)

    def log_runtime_summary(self) -> None:
        """実行範囲と対象拡張子をログに出す。"""
        logger.info(
            "速度倍率: %s",
            ", ".join(
                f"{speed_rate:.2f}" for speed_rate in self.runtime_config.speed_rates
            ),
        )
        logger.info(
            "対象拡張子: audio=%s / midi=%s",
            ", ".join(self.runtime_config.audio_extensions),
            ", ".join(self.runtime_config.midi_extensions),
        )

    def collect_input_files(self) -> tuple[list[Path], list[Path]]:
        """音声とMIDIの入力ファイル一覧をまとめて返す。"""
        audio_files: list[Path] = []
        midi_files: list[Path] = []

        if self.runtime_config.generate_audio:
            audio_files = self.collect_source_files(
                self.runtime_config.audio_dir,
                extensions=self.runtime_config.audio_extensions,
                source_label="音声",
            )
        if self.runtime_config.generate_midi:
            midi_files = self.collect_source_files(
                self.runtime_config.midi_dir,
                extensions=self.runtime_config.midi_extensions,
                source_label="MIDI",
            )
        return audio_files, midi_files

    def collect_source_files(
        self,
        root_dir: Path,
        *,
        extensions: tuple[str, ...],
        source_label: str,
    ) -> list[Path]:
        """
        入力ディレクトリから元データだけを再帰収集する。
        生成済みファイルを除外することで、既存拡張データの再入力を防ぐ。
        """
        source_files = sorted(
            path
            for path in root_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in extensions
            and not self.is_derived_file(path)
        )
        if source_files:
            return source_files

        extension_text = ", ".join(extensions)
        raise FileNotFoundError(
            f"入力{source_label}が見つかりません: {root_dir} (extensions: {extension_text})"
        )

    def is_derived_file(self, path: Path) -> bool:
        """生成済み派生ファイルかどうかをファイル名から判定する。"""
        return any(
            marker in path.stem
            for marker in self.runtime_config.exclude_filename_markers
        )

    def run_audio(self, audio_files: list[Path]) -> None:
        """音声タスク群を構築して並列実行する。"""
        audio_tasks = [
            AudioTask(
                input_path=path,
                output_dir=self.build_task_output_dir(
                    self.runtime_config.audio_dir,
                    self.runtime_config.output_audio_dir,
                    path,
                ),
                speed_rates=self.runtime_config.speed_rates,
            )
            for path in audio_files
        ]
        logger.info("音声処理対象: %d ファイル", len(audio_tasks))
        audio_success, audio_skipped, audio_errors = self.execute_audio_tasks(
            audio_tasks
        )
        logger.info(
            "音声処理完了: success=%d, skipped=%d, errors=%d",
            audio_success,
            audio_skipped,
            len(audio_errors),
        )
        self.log_error_messages(audio_errors)

    def run_midi(self, midi_files: list[Path]) -> None:
        """MIDIタスク群を構築して並列実行する。"""
        midi_tasks = [
            MidiTask(
                input_path=path,
                output_dir=self.build_task_output_dir(
                    self.runtime_config.midi_dir,
                    self.runtime_config.output_midi_dir,
                    path,
                ),
                speed_rates=self.runtime_config.speed_rates,
            )
            for path in midi_files
        ]
        logger.info("MIDI処理対象: %d ファイル", len(midi_tasks))
        midi_success, midi_skipped, midi_errors = self.execute_midi_tasks(
            midi_tasks
        )
        logger.info(
            "MIDI処理完了: success=%d, skipped=%d, errors=%d",
            midi_success,
            midi_skipped,
            len(midi_errors),
        )
        self.log_error_messages(midi_errors)

    def build_task_output_dir(
        self,
        input_root_dir: Path,
        output_root_dir: Path,
        input_path: Path,
    ) -> Path:
        """入力の相対ディレクトリ構造を保ったまま出力先を決める。"""
        return output_root_dir / input_path.relative_to(input_root_dir).parent

    def execute_audio_tasks(self, tasks: list[AudioTask]) -> tuple[int, int, list[str]]:
        """音声タスク群を並列実行して結果を集計する。"""
        get_pyrubberband()
        total_success = 0
        total_skipped = 0
        all_errors: list[str] = []

        with ProcessPoolExecutor(
            max_workers=self.runtime_config.workers,
            initializer=initialize_audio_worker,
        ) as executor:
            futures = {
                executor.submit(
                    process_audio_task,
                    task,
                    self.runtime_config.overwrite,
                ): task.input_path.name
                for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                total_success += result.success_count
                total_skipped += result.skipped_count
                all_errors.extend(result.error_messages)

        return total_success, total_skipped, all_errors

    def execute_midi_tasks(
        self, tasks: list[MidiTask]
    ) -> tuple[int, int, list[str]]:
        """MIDIタスク群を並列実行して結果を集計する。"""
        total_success = 0
        total_skipped = 0
        all_errors: list[str] = []

        with ProcessPoolExecutor(max_workers=self.runtime_config.workers) as executor:
            futures = {
                executor.submit(
                    process_midi_task,
                    task,
                    self.runtime_config.overwrite,
                ): task.input_path.name
                for task in tasks
            }
            for future in as_completed(futures):
                result = future.result()
                total_success += result.success_count
                total_skipped += result.skipped_count
                all_errors.extend(result.error_messages)

        return total_success, total_skipped, all_errors

    def log_pairing_summary(
        self, audio_files: list[Path], midi_files: list[Path]
    ) -> None:
        """音声とMIDIの対応状況をログに出す。"""
        audio_stems = {
            path.relative_to(self.runtime_config.audio_dir).with_suffix("")
            for path in audio_files
        }
        midi_stems = {
            path.relative_to(self.runtime_config.midi_dir).with_suffix("")
            for path in midi_files
        }
        missing_midis = sorted(audio_stems - midi_stems)
        missing_audios = sorted(midi_stems - audio_stems)

        if not missing_midis and not missing_audios:
            logger.info(
                "ペア対応確認: 音声 %d 件 / MIDI %d 件は stem ベースで一致",
                len(audio_files),
                len(midi_files),
            )
            return

        logger.warning(
            "ペア対応に不一致があります: audio_only=%d, midi_only=%d",
            len(missing_midis),
            len(missing_audios),
        )
        if missing_midis:
            logger.warning(
                "MIDIが見つからない音声例: %s", self.format_path_preview(missing_midis)
            )
        if missing_audios:
            logger.warning(
                "音声が見つからないMIDI例: %s", self.format_path_preview(missing_audios)
            )

    @staticmethod
    def format_path_preview(relative_paths: list[Path], *, limit: int = 5) -> str:
        """ログ用に相対パスの先頭数件だけを整形する。"""
        preview = ", ".join(path.as_posix() for path in relative_paths[:limit])
        if len(relative_paths) > limit:
            preview += ", ..."
        return preview

    @staticmethod
    def log_error_messages(error_messages: list[str]) -> None:
        """集約したエラーメッセージを順に出力する。"""
        for error_message in error_messages:
            logger.error(error_message)


def run_cli() -> None:
    """CLIエントリポイントとして共通処理を実行する。"""
    runtime_config = RuntimeTimeStretchConfig.from_arguments(parse_arguments())
    TimeStretchDatasetRunner(runtime_config).run()


def main() -> None:
    """単体実行用のメイン関数。"""
    run_cli()


if __name__ == "__main__":
    main()
