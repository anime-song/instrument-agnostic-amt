from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Iterable

import os
import pretty_midi
import soundfile as sf
import torch
import torch.hub
import torchaudio.functional as audio_functional
from tqdm.auto import tqdm

from models.model import AudioSemiCRFTransformer, SemiCRFModelConfig
from models.semi_crf import decode_pitch_intervals
from dataset import MIN_MIDI_PITCH
import numpy as np
from inference_postprocess import MidiMetadataEmbedder


DEFAULT_CHECKPOINT_URL = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main/best_model.pth?download=true"
DEFAULT_CHECKPOINT_PATH = Path("checkpoints/best_model.pth")


@dataclass
class PredictedNote:
    """推論中のノートと、後段で楽器を確定するための集約情報を保持する。"""

    pitch: int
    start_frame: int
    end_frame: int
    velocity: int
    slot_index: int = 0
    has_onset: bool = True
    has_offset: bool = True
    instrument_id: int = 0
    instrument_candidates: tuple[int, ...] = ()
    instrument_prob_sum: np.ndarray | None = None
    instrument_frame_count: int = 0


@dataclass(frozen=True)
class InferenceSettings:
    window_ms: int
    stride_ms: int
    track_batch_size: int
    length_scaling: str
    length_penalty: float
    instrument_probability_mode: str
    note_bias: float = 0.0


def resolve_amp_dtype(device: torch.device, dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float16


def _normalize_instrument_probability_mode(mode: str | None) -> str:
    normalized = "sigmoid" if mode is None else str(mode).strip().lower()
    if normalized in {"bce", "sigmoid"}:
        return "sigmoid"
    if normalized in {"ce", "softmax"}:
        return "softmax"
    raise ValueError(f"Unsupported instrument probability mode: {mode}")


def _instrument_probabilities(logits: torch.Tensor, *, mode: str) -> torch.Tensor:
    probability_mode = _normalize_instrument_probability_mode(mode)
    if probability_mode == "softmax":
        return torch.softmax(logits, dim=-1)
    return torch.sigmoid(logits)


SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".opus",
    ".m4a",
    ".aac",
    ".wma",
    ".aiff",
    ".aif",
}

DEFAULT_INSTRUMENT_VOLUMES: dict[str, int] = {
    "piano": 105,
    "electric_piano": 96,
    "acoustic_guitar": 88,
    "electric_guitar_clean": 84,
    "distorted_guitar": 78,
    "electric_guitar_muted": 68,
    "guitar_harmonics": 72,
    "acoustic_bass": 92,
    "electric_bass": 90,
    "synth_bass": 86,
    "drums": 100,
    "strings": 82,
    "brass": 84,
    "sax": 86,
    "flute_pipe": 84,
    "choir": 80,
    "organ": 88,
    "synth_pad": 76,
    "synth_lead": 84,
    "melody": 110,
    "vocal_harmony": 60
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Semi-CRF AMT inference on an audio file and export MIDI.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=f"Path to the trained model checkpoint (.pth). If not provided, downloads from Hugging Face.",
    )
    parser.add_argument(
        "--type",
        choices=["default", "bass", "vocal", "guitar", "vocal_harmony"],
        default="default",
        help="Type of the model to download from Hugging Face if checkpoint is not provided. 'default' for multi-instrument, 'bass' for bass-focused model, 'vocal' for vocal-focused model, 'guitar' for guitar-focused model, 'vocal_harmony' for vocal harmony model.",
    )

    # 単一ファイルモード
    parser.add_argument(
        "--audio", type=Path, default=None, help="Path to the input audio file"
    )
    parser.add_argument(
        "--output-midi",
        type=Path,
        default=None,
        help="Output MIDI path. Defaults to <audio_stem>.mid next to the input audio.",
    )

    # ディレクトリバッチモード
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=None,
        help="Directory containing audio files for batch inference.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for MIDI files when using --audio-dir. Defaults to --audio-dir.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--amp", action="store_true", help="Enable Mixed Precision inference"
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default="bf16" if torch.cuda.is_bf16_supported() else "fp16",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=None,
        help="Inference window size in milliseconds. Defaults to the training window.",
    )
    parser.add_argument(
        "--stride-ms",
        type=int,
        default=None,
        help="Inference stride in milliseconds. Defaults to half of window-ms.",
    )
    parser.add_argument(
        "--semi-crf-track-batch-size",
        type=int,
        default=None,
        help="Chunk size for CRF decoding. Defaults to the training setting.",
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=1,
        help="Number of windows to process at once. Increase only if VRAM allows it.",
    )
    parser.add_argument(
        "--merge-gap-ms",
        type=float,
        default=None,
        help="Merge same-pitch notes when the gap is below this threshold. Defaults to one hop.",
    )
    parser.add_argument(
        "--merge-onset-ms",
        type=float,
        default=20.0,
        help="Merge same-pitch notes when their onset difference is below this threshold.",
    )
    parser.add_argument(
        "--velocity",
        type=int,
        default=100,
        help="Constant MIDI velocity for predicted notes.",
    )
    parser.add_argument(
        "--instrument-volume",
        action="append",
        default=[],
        metavar="CLASS=VALUE",
        help=(
            "Set MIDI CC7 volume (0-127) for a specific instrument class. "
            "Repeatable, e.g. --instrument-volume piano=90 --instrument-volume guitar=70. "
            "If omitted, a built-in per-instrument default mix is used."
        ),
    )
    parser.add_argument(
        "--max-note-seconds",
        type=float,
        default=15.0,
        help=(
            "Delete notes whose duration is longer than or equal to this threshold. "
            "Use 0 or a negative value to disable."
        ),
    )
    parser.add_argument(
        "--min-midi-note-ms",
        type=float,
        default=5.0,
        help=(
            "Extend very short exported MIDI notes up to this duration when possible. "
            "Useful for preserving muted guitar notes without tick-quantization collapse. "
            "Use 0 or a negative value to disable."
        ),
    )
    parser.add_argument(
        "--max-midi-melodic-instruments",
        type=int,
        default=15,
        help=(
            "Maximum number of non-drum instrument tracks in exported MIDI. "
            "Extra low-note-count instruments are reassigned by predicted instrument confidence rank. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--silence-gate-rms-dbfs",
        type=float,
        default=-72,
        help="Skip fully silent windows before model forward pass.",
    )
    parser.add_argument("--disable-tqdm", action="store_true")

    parser.add_argument(
        "--disable-beat", action="store_true", help="Disable beat embedding"
    )
    parser.add_argument(
        "--disable-chord", action="store_true", help="Disable chord embedding"
    )
    parser.add_argument(
        "--note-bias",
        type=float,
        default=0.0,
        help="Add a constant bias to the semi-CRF interval score to increase recall.",
    )

    args = parser.parse_args()

    # --audio と --audio-dir は排他
    if args.audio is None and args.audio_dir is None:
        parser.error("--audio or --audio-dir is required.")
    if args.audio is not None and args.audio_dir is not None:
        parser.error("--audio and --audio-dir are mutually exclusive.")
    if args.output_midi is not None and args.audio_dir is not None:
        parser.error(
            "--output-midi cannot be used with --audio-dir. Use --output-dir instead."
        )

    try:
        args.instrument_volume_map = _parse_instrument_volume_args(
            args.instrument_volume
        )
    except ValueError as exc:
        parser.error(str(exc))

    return args


def _normalize_instrument_class_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _parse_instrument_volume_args(entries: list[str]) -> dict[str, int]:
    if not entries:
        return dict(DEFAULT_INSTRUMENT_VOLUMES)

    from instrument_classes import INSTRUMENT_CLASSES

    available_names = {
        _normalize_instrument_class_name(class_name): class_name
        for class_name in INSTRUMENT_CLASSES
    }
    volume_map: dict[str, int] = {}

    for entry in entries:
        if "=" not in entry:
            raise ValueError(
                f"Invalid --instrument-volume '{entry}'. Expected CLASS=VALUE."
            )
        raw_name, raw_value = entry.split("=", 1)
        class_key = _normalize_instrument_class_name(raw_name)
        if class_key not in available_names:
            available = ", ".join(sorted(available_names.keys()))
            raise ValueError(
                f"Unknown instrument class '{raw_name}'. Available classes: {available}"
            )
        try:
            volume = int(raw_value)
        except ValueError as exc:
            raise ValueError(
                f"Invalid volume '{raw_value}' for instrument '{raw_name}'."
            ) from exc
        if not 0 <= volume <= 127:
            raise ValueError(
                f"Volume for instrument '{raw_name}' must be in the range 0-127."
            )
        volume_map[class_key] = volume

    return volume_map


def _ensure_checkpoint(
    checkpoint_path: Path | None, model_type: str = "default"
) -> Path:
    if checkpoint_path is None:
        if model_type == "bass":
            checkpoint_path = Path("checkpoints/best_model_bass.pth")
            url = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main/best_model_bass.pth?download=true"
        elif model_type == "vocal":
            checkpoint_path = Path("checkpoints/best_model_vocal.pth")
            url = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main/best_model_vocal.pth?download=true"
        elif model_type == "guitar":
            checkpoint_path = Path("checkpoints/best_model_guitar.pth")
            url = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main/best_model_guitar.pth?download=true"
        elif model_type == "vocal_harmony":
            checkpoint_path = Path("checkpoints/best_model_vocal_harmony.pth")
            url = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main/best_model_vocal_harmony.pth?download=true"
        else:
            checkpoint_path = DEFAULT_CHECKPOINT_PATH
            url = DEFAULT_CHECKPOINT_URL
    else:
        if model_type == "bass":
            url = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main/best_model_bass.pth?download=true"
        elif model_type == "vocal":
            url = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main/best_model_vocal.pth?download=true"
        elif model_type == "guitar":
            url = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main/best_model_guitar.pth?download=true"
        elif model_type == "vocal_harmony":
            url = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main/best_model_vocal_harmony.pth?download=true"
        else:
            url = DEFAULT_CHECKPOINT_URL

    if not checkpoint_path.exists():
        print(
            f"Checkpoint not found at {checkpoint_path}. Downloading from Hugging Face..."
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(url, str(checkpoint_path))

    return checkpoint_path


def _iter_batches(values: list[int], batch_size: int) -> Iterable[list[int]]:
    chunk_size = max(1, int(batch_size))
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _coerce_model_config(raw_model_config: dict[str, Any]) -> SemiCRFModelConfig:
    allowed_fields = {field.name for field in fields(SemiCRFModelConfig)}
    kwargs = {
        key: value for key, value in raw_model_config.items() if key in allowed_fields
    }
    # 推論時に augmentation や gradient checkpoint は不要なので無効化する。
    kwargs["spec_augment_params"] = None
    kwargs["use_gradient_checkpoint"] = False
    return SemiCRFModelConfig(**kwargs)


def _load_model_and_settings(
    checkpoint_path: Path,
    *,
    device: torch.device,
    window_ms_override: int | None,
    stride_ms_override: int | None,
    track_batch_size_override: int | None,
    note_bias: float = 0.0,
) -> tuple[AudioSemiCRFTransformer, SemiCRFModelConfig, InferenceSettings]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    raw_model_config = None
    raw_run_config = checkpoint.get("config")
    if isinstance(raw_run_config, dict):
        raw_model_config = raw_run_config.get("model_config")
    if raw_model_config is None:
        raw_model_config = checkpoint.get("model_config")

    if not isinstance(raw_model_config, dict):
        import logging

        logging.warning(
            "Checkpoint does not contain model_config. Using fallback default config (sample_rate=22050, hop_length=512)."
        )
        raw_model_config = {
            "sample_rate": 22050,
            "hop_length": 512,
            "n_fft": 2048,
            "cqt_n_bins": 312,
        }

    model_config = _coerce_model_config(raw_model_config)
    model = AudioSemiCRFTransformer(model_config)

    # EMA の重みがあれば優先的に使用する
    ema_state_dict = checkpoint.get("ema_state_dict")
    if ema_state_dict is not None:
        state_dict = ema_state_dict
        print("Using EMA weights from checkpoint.")
    else:
        raw_state_dict = checkpoint.get("model_state_dict")
        if raw_state_dict is not None:
            state_dict = raw_state_dict
        elif all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
            state_dict = checkpoint
        else:
            state_dict = None
    if not isinstance(state_dict, dict):
        raise ValueError(f"Checkpoint does not contain a state_dict: {checkpoint_path}")

    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Missing keys while loading checkpoint: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(
            f"Unexpected keys while loading checkpoint: {incompatible.unexpected_keys}"
        )
    missing_interval_head_keys = [
        key
        for key in incompatible.missing_keys
        if key.startswith("interval_instrument_predictor.")
    ]
    model._use_interval_instrument_head = len(missing_interval_head_keys) == 0
    if not model._use_interval_instrument_head:
        print(
            "Interval instrument head weights were not found in the checkpoint. "
            "Falling back to the legacy framewise instrument classifier."
        )
    model.to(device)
    model.eval()

    raw_args = (
        raw_run_config.get("args", {}) if isinstance(raw_run_config, dict) else {}
    )
    default_window_ms = int(raw_args.get("window_ms", 5000))
    window_ms = (
        int(window_ms_override) if window_ms_override is not None else default_window_ms
    )
    stride_ms = (
        int(stride_ms_override)
        if stride_ms_override is not None
        else max(1, window_ms // 2)
    )
    track_batch_size = (
        int(track_batch_size_override) if track_batch_size_override is not None else 128
    )
    instrument_probability_mode = _normalize_instrument_probability_mode(
        raw_args.get("instrument_loss_type", "bce")
    )
    settings = InferenceSettings(
        window_ms=window_ms,
        stride_ms=stride_ms,
        track_batch_size=track_batch_size,
        length_scaling=str(model_config.semi_crf_length_scaling),
        length_penalty=float(model_config.semi_crf_length_penalty),
        instrument_probability_mode=instrument_probability_mode,
        note_bias=float(note_bias),
    )
    return model, model_config, settings


def _load_audio(
    audio_path: Path,
    *,
    target_sample_rate: int,
) -> tuple[torch.Tensor, int, int]:
    waveform_np, source_sample_rate = sf.read(
        audio_path,
        dtype="float32",
        always_2d=True,
    )
    waveform = torch.from_numpy(waveform_np.T.copy())
    source_channels = int(waveform.shape[0])

    if source_channels <= 0:
        raise ValueError(f"Audio file has no channels: {audio_path}")
    if source_channels == 1:
        waveform = waveform.repeat(2, 1)
    elif source_channels > 2:
        waveform = waveform[:2]

    if int(source_sample_rate) != int(target_sample_rate):
        waveform = audio_functional.resample(
            waveform,
            int(source_sample_rate),
            int(target_sample_rate),
        )
    return waveform.contiguous(), int(source_sample_rate), source_channels


def _build_window_starts(
    *,
    total_audio_frames: int,
    window_audio_frames: int,
    stride_audio_frames: int,
) -> list[int]:
    if total_audio_frames <= window_audio_frames:
        return [0]
    starts = list(
        range(
            0,
            max(1, total_audio_frames - window_audio_frames + 1),
            stride_audio_frames,
        )
    )
    last_start = total_audio_frames - window_audio_frames
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _slice_window(
    waveform: torch.Tensor,
    *,
    start_frame: int,
    window_audio_frames: int,
) -> tuple[torch.Tensor, int]:
    end_frame = min(int(waveform.shape[-1]), start_frame + window_audio_frames)
    window = waveform[:, start_frame:end_frame]
    valid_audio_frames = int(window.shape[-1])
    if valid_audio_frames < window_audio_frames:
        padded = torch.zeros(
            (int(waveform.shape[0]), window_audio_frames),
            dtype=waveform.dtype,
        )
        padded[:, :valid_audio_frames] = window
        window = padded
    return window.contiguous(), valid_audio_frames


def _silence_gate_rms_linear(silence_gate_rms_dbfs: float | None) -> float | None:
    if silence_gate_rms_dbfs is None:
        return None
    threshold_dbfs = float(silence_gate_rms_dbfs)
    if threshold_dbfs > 0.0:
        raise ValueError("silence_gate_rms_dbfs must be <= 0 dBFS")
    return float(10.0 ** (threshold_dbfs / 20.0))


def _compute_silent_window_mask(
    batch_waveform: torch.Tensor,
    *,
    silence_gate_rms_linear: float | None,
) -> torch.Tensor | None:
    if silence_gate_rms_linear is None:
        return None
    if batch_waveform.dim() != 3:
        raise ValueError("batch_waveform must have shape [B, C, T]")
    # convert to mono for RMS calculation
    gate_input = batch_waveform.mean(dim=1, keepdim=True)
    rms = gate_input.float().square().mean(dim=-1).sqrt()
    return torch.all(rms < float(silence_gate_rms_linear), dim=1)


class PitchFirstNoteStitcher:
    """窓またぎのノート継続を pitch 優先で管理し、最後に楽器を確定する。"""

    def __init__(
        self,
        *,
        hop_length: int,
        total_audio_frames: int,
        num_pitch_slots: int,
        velocity: int,
        merge_gap_frames: int,
        merge_onset_frames: int,
        instrument_probability_mode: str,
    ) -> None:
        # 1. 推論設定
        self.hop_length = int(hop_length)
        self.total_audio_frames = int(total_audio_frames)
        self.num_pitch_slots = max(1, int(num_pitch_slots))
        self.velocity = int(velocity)
        self.merge_gap_frames = int(merge_gap_frames)
        self.merge_onset_frames = int(merge_onset_frames)
        self.instrument_probability_mode = _normalize_instrument_probability_mode(
            instrument_probability_mode
        )

        # 2. 窓またぎ状態
        self.notes_by_track: dict[int, list[PredictedNote]] = defaultdict(list)
        self.last_closed_global_frames: list[int] | None = None

    def get_forced_start_positions(
        self,
        *,
        window_start_frame: int,
        num_tracks: int,
        valid_model_frames: int,
    ) -> list[int]:
        """各 track の継続状態から、次の decode 開始位置制約を返す。"""
        self._ensure_track_state(num_tracks)
        if valid_model_frames <= 0:
            return [0] * int(num_tracks)

        window_model_start = int(
            round(float(window_start_frame) / float(self.hop_length))
        )
        return [
            max(
                0,
                min(
                    int(last_closed_frame) - window_model_start,
                    int(valid_model_frames) - 1,
                ),
            )
            for last_closed_frame in self.last_closed_global_frames
        ]

    def consume_window(
        self,
        *,
        intervals_by_track: list[list[tuple[int, int]]],
        boundary_flags_by_track: list[list[tuple[bool, bool, float, float]]] | None,
        interval_instrument_stats_by_track: (list[list[tuple[np.ndarray, int]]] | None),
        instrument_logits: torch.Tensor | None,
        window_start_frame: int,
        valid_audio_frames: int,
        valid_model_frames: int,
    ) -> None:
        """1ウィンドウ分の区間を取り込み、継続ノート状態を更新する。"""
        self._ensure_track_state(len(intervals_by_track))
        window_model_start = int(
            round(float(window_start_frame) / float(self.hop_length))
        )

        for track_index, track_intervals in enumerate(intervals_by_track):
            track_notes = self.notes_by_track[int(track_index)]
            track_boundary_flags = (
                boundary_flags_by_track[track_index]
                if boundary_flags_by_track is not None
                else []
            )
            track_interval_instrument_stats = (
                interval_instrument_stats_by_track[track_index]
                if interval_instrument_stats_by_track is not None
                else []
            )
            local_last_closed_frame: int | None = None

            for interval_index, (begin_frame, end_frame) in enumerate(track_intervals):
                boundary_flag = (
                    track_boundary_flags[interval_index]
                    if interval_index < len(track_boundary_flags)
                    else None
                )
                interval_instrument_stats = (
                    track_interval_instrument_stats[interval_index]
                    if interval_index < len(track_interval_instrument_stats)
                    else None
                )
                note = self._build_interval_note(
                    track_index=int(track_index),
                    begin_frame=int(begin_frame),
                    end_frame=int(end_frame),
                    boundary_flag=boundary_flag,
                    interval_instrument_stats=interval_instrument_stats,
                    instrument_logits=instrument_logits,
                    window_start_frame=int(window_start_frame),
                    valid_audio_frames=int(valid_audio_frames),
                    valid_model_frames=int(valid_model_frames),
                )
                if note is None:
                    continue

                if track_notes and int(note.start_frame) < int(
                    track_notes[-1].end_frame
                ):
                    if note.has_onset:
                        track_notes[-1] = note
                    else:
                        self._merge_note_segments(
                            track_notes[-1],
                            note,
                            overwrite_offset=True,
                        )
                    if note.has_offset:
                        local_last_closed_frame = int(end_frame)
                    continue

                if note.has_onset:
                    track_notes.append(note)
                if note.has_offset:
                    local_last_closed_frame = int(end_frame)

            if local_last_closed_frame is not None:
                self.last_closed_global_frames[track_index] = window_model_start + int(
                    local_last_closed_frame
                )

    def finalize(self) -> list[PredictedNote]:
        """窓ごとの継続状態を閉じ、最終的なノート列へ整形する。"""
        for track_notes in self.notes_by_track.values():
            if track_notes:
                track_notes[-1].has_offset = True

        stitched_notes = sorted(
            [
                note
                for track_notes in self.notes_by_track.values()
                for note in track_notes
                if note.has_offset
            ],
            key=lambda note: (
                note.start_frame,
                note.pitch,
                note.slot_index,
                note.end_frame,
            ),
        )
        merged_notes = self._merge_nearby_notes(stitched_notes)
        return self._assign_note_instruments(merged_notes)

    def _ensure_track_state(self, num_tracks: int) -> None:
        """track 数に応じた継続状態バッファを初期化する。"""
        if self.last_closed_global_frames is None:
            self.last_closed_global_frames = [0] * int(num_tracks)
            return
        if len(self.last_closed_global_frames) != int(num_tracks):
            raise ValueError(
                "num_tracks changed during note stitching: "
                f"{len(self.last_closed_global_frames)} -> {num_tracks}"
            )

    def _track_to_pitch_slot(self, track_index: int) -> tuple[int, int]:
        slot_count = max(1, int(self.num_pitch_slots))
        return int(track_index) // slot_count, int(track_index) % slot_count

    def _resolve_interval_boundary_flags(
        self,
        *,
        begin_frame: int,
        end_frame: int,
        valid_model_frames: int,
        boundary_flag: tuple[bool, bool, float, float] | None,
    ) -> tuple[bool, bool, float, float]:
        """境界ヘッドが無い場合でも、窓端の継続フラグを補完する。"""
        if boundary_flag is not None:
            return boundary_flag

        last_valid_frame = max(0, int(valid_model_frames) - 1)
        return (
            bool(int(begin_frame) > 0),
            bool(int(end_frame) < last_valid_frame),
            0.0,
            1.0,
        )

    def _extract_interval_instrument_stats(
        self,
        *,
        interval_instrument_stats: tuple[np.ndarray, int] | None,
        instrument_logits: torch.Tensor | None,
        begin_frame: int,
        end_frame: int,
        track_index: int,
    ) -> tuple[np.ndarray | None, int]:
        """区間内の楽器確率をフレーム和として集約する。"""
        if interval_instrument_stats is not None:
            instrument_prob_sum, instrument_frame_count = interval_instrument_stats
            return (
                instrument_prob_sum.astype(np.float32, copy=False),
                int(instrument_frame_count),
            )
        if instrument_logits is None:
            return None, 0

        interval_start = max(0, int(begin_frame))
        interval_end = min(int(end_frame) + 1, int(instrument_logits.shape[0]))
        if interval_end <= interval_start:
            return None, 0

        pitch_index, _ = self._track_to_pitch_slot(int(track_index))
        note_logits = instrument_logits[interval_start:interval_end, pitch_index, :]
        note_prob_sum = (
            _instrument_probabilities(
                note_logits,
                mode=self.instrument_probability_mode,
            )
            .sum(dim=0)
            .float()
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )
        return note_prob_sum, int(note_logits.shape[0])

    def _build_interval_note(
        self,
        *,
        track_index: int,
        begin_frame: int,
        end_frame: int,
        boundary_flag: tuple[bool, bool, float, float] | None,
        interval_instrument_stats: tuple[np.ndarray, int] | None,
        instrument_logits: torch.Tensor | None,
        window_start_frame: int,
        valid_audio_frames: int,
        valid_model_frames: int,
    ) -> PredictedNote | None:
        """1つの decode 区間を、stitch 用ノート候補へ変換する。"""
        has_onset, has_offset, onset_off, offset_off = (
            self._resolve_interval_boundary_flags(
                begin_frame=int(begin_frame),
                end_frame=int(end_frame),
                valid_model_frames=int(valid_model_frames),
                boundary_flag=boundary_flag,
            )
        )

        start_frame = window_start_frame + int(
            round((float(begin_frame) + float(onset_off)) * float(self.hop_length))
        )
        end_frame_exclusive = window_start_frame + min(
            int(valid_audio_frames),
            int(round((float(end_frame) + float(offset_off)) * float(self.hop_length))),
        )

        start_frame = max(0, min(start_frame, self.total_audio_frames))
        end_frame_exclusive = max(
            start_frame + 1,
            min(end_frame_exclusive, self.total_audio_frames),
        )
        window_valid_end = min(
            self.total_audio_frames,
            int(window_start_frame) + int(valid_audio_frames),
        )
        if start_frame >= window_valid_end:
            return None

        instrument_prob_sum, instrument_frame_count = (
            self._extract_interval_instrument_stats(
                interval_instrument_stats=interval_instrument_stats,
                instrument_logits=instrument_logits,
                begin_frame=int(begin_frame),
                end_frame=int(end_frame),
                track_index=int(track_index),
            )
        )
        pitch_index, slot_index = self._track_to_pitch_slot(int(track_index))
        return PredictedNote(
            pitch=int(MIN_MIDI_PITCH + pitch_index),
            start_frame=int(start_frame),
            end_frame=int(end_frame_exclusive),
            velocity=int(self.velocity),
            slot_index=int(slot_index),
            has_onset=bool(has_onset),
            has_offset=bool(has_offset),
            instrument_id=0,
            instrument_candidates=(),
            instrument_prob_sum=instrument_prob_sum,
            instrument_frame_count=int(instrument_frame_count),
        )

    def _accumulate_note_instrument_stats(
        self,
        target: PredictedNote,
        source: PredictedNote,
    ) -> None:
        """窓またぎで分割された区間の楽器確率を加算する。"""
        if source.instrument_prob_sum is None or source.instrument_frame_count <= 0:
            return
        if target.instrument_prob_sum is None or target.instrument_frame_count <= 0:
            target.instrument_prob_sum = source.instrument_prob_sum.copy()
            target.instrument_frame_count = int(source.instrument_frame_count)
            return
        target.instrument_prob_sum = (
            target.instrument_prob_sum + source.instrument_prob_sum
        ).astype(np.float32, copy=False)
        target.instrument_frame_count += int(source.instrument_frame_count)

    def _merge_note_segments(
        self,
        target: PredictedNote,
        source: PredictedNote,
        *,
        overwrite_offset: bool,
    ) -> None:
        """同一 pitch-slot track の分割区間を1ノートへ統合する。"""
        target.end_frame = max(int(target.end_frame), int(source.end_frame))
        target.velocity = max(int(target.velocity), int(source.velocity))
        target.has_onset = bool(target.has_onset or source.has_onset)
        if overwrite_offset:
            target.has_offset = bool(source.has_offset)
        else:
            target.has_offset = bool(target.has_offset or source.has_offset)
        self._accumulate_note_instrument_stats(target, source)

    def _merge_nearby_notes(self, notes: list[PredictedNote]) -> list[PredictedNote]:
        """最終段の軽いクレンジングとして、近接した同一 pitch を再統合する。"""
        if not notes:
            return []

        ordered_notes = sorted(
            notes,
            key=lambda note: (
                note.pitch,
                note.slot_index,
                note.start_frame,
                note.end_frame,
            ),
        )
        merged: list[PredictedNote] = []
        current = replace(ordered_notes[0])
        for note in ordered_notes[1:]:
            can_merge_by_gap = (
                note.pitch == current.pitch
                and note.slot_index == current.slot_index
                and note.start_frame <= current.end_frame + self.merge_gap_frames
                and not note.has_onset
            )
            can_merge_by_onset = (
                note.pitch == current.pitch
                and note.slot_index == current.slot_index
                and abs(note.start_frame - current.start_frame)
                <= self.merge_onset_frames
            )

            if can_merge_by_gap:
                self._merge_note_segments(current, note, overwrite_offset=True)
                continue
            if can_merge_by_onset:
                self._merge_note_segments(current, note, overwrite_offset=False)
                continue
            merged.append(current)
            current = replace(note)
        merged.append(current)
        return sorted(
            merged,
            key=lambda note: (
                note.start_frame,
                note.pitch,
                note.slot_index,
                note.end_frame,
            ),
        )

    def _assign_note_instruments(
        self, notes: list[PredictedNote]
    ) -> list[PredictedNote]:
        """集約済みの楽器確率から、最終的な楽器IDと候補順を確定する。"""
        finalized_notes: list[PredictedNote] = []
        for note in notes:
            if note.instrument_prob_sum is None or note.instrument_frame_count <= 0:
                finalized_notes.append(
                    replace(
                        note,
                        instrument_id=0,
                        instrument_candidates=(0,),
                        instrument_prob_sum=None,
                        instrument_frame_count=0,
                    )
                )
                continue

            note_probs = note.instrument_prob_sum / float(note.instrument_frame_count)
            order = np.argsort(-note_probs.astype(np.float64, copy=False))
            instrument_candidates = tuple(int(index) for index in order.tolist())
            instrument_id = (
                int(instrument_candidates[0]) if instrument_candidates else 0
            )
            finalized_notes.append(
                replace(
                    note,
                    instrument_id=instrument_id,
                    instrument_candidates=instrument_candidates or (0,),
                    instrument_prob_sum=None,
                    instrument_frame_count=0,
                )
            )
        return finalized_notes


def _decode_boundary_features(
    boundary_logits: torch.Tensor,
    entries: list[tuple[int, int, int, int, int]],
    *,
    batch_size: int,
    num_tracks: int,
) -> list[list[list[tuple[bool, bool, float, float]]]]:
    flags: list[list[list[tuple[bool, bool, float, float]]]] = [
        [[] for _ in range(num_tracks)] for _ in range(batch_size)
    ]
    if not entries:
        return flags

    # 分布計算は低精度のままだと不安定になりやすいため、ここだけ FP32 に戻す。
    boundary_logits = boundary_logits.float()
    presence_logits, offset_logits = boundary_logits.chunk(2, dim=-1)
    boundary_presence = presence_logits > 0.0

    offset_dist = torch.distributions.ContinuousBernoulli(logits=offset_logits)
    offset_values = (offset_dist.mean - 0.005) / 0.99
    offset_values = torch.clamp(offset_values, 0.0, 1.0)

    for row_index, entry in enumerate(entries):
        batch_index, track_index, _, _, _ = entry
        flags[batch_index][track_index].append(
            (
                bool(boundary_presence[row_index, 0].item()),
                bool(boundary_presence[row_index, 1].item()),
                float(offset_values[row_index, 0].item()),
                float(offset_values[row_index, 1].item()),
            )
        )
    return flags


def _decode_interval_instrument_logits(
    interval_instrument_logits: torch.Tensor,
    entries: list[tuple[int, int, int, int, int]],
    *,
    batch_size: int,
    num_tracks: int,
    probability_mode: str,
) -> list[list[list[tuple[np.ndarray, int]]]]:
    stats: list[list[list[tuple[np.ndarray, int]]]] = [
        [[] for _ in range(num_tracks)] for _ in range(batch_size)
    ]
    if not entries:
        return stats

    interval_probs = (
        _instrument_probabilities(
            interval_instrument_logits.float(),
            mode=probability_mode,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    for row_index, entry in enumerate(entries):
        batch_index, track_index, _, begin_frame, end_frame = entry
        interval_frame_count = max(1, int(end_frame) - int(begin_frame) + 1)
        prob_sum = (interval_probs[row_index] * float(interval_frame_count)).astype(
            np.float32, copy=False
        )
        stats[batch_index][track_index].append((prob_sum, interval_frame_count))
    return stats


def _truncate_overlapping_notes(
    notes: list[PredictedNote],
    *,
    separate_adjacent_same_pitch: bool = False,
    min_separation_frames: int = 0,
) -> list[PredictedNote]:
    if not notes:
        return []

    ordered_notes = sorted(
        notes,
        key=lambda note: (
            note.pitch,
            note.instrument_id,
            note.start_frame,
            note.end_frame,
        ),
    )

    by_pitch: dict[tuple[int, int], list[PredictedNote]] = {}
    for note in ordered_notes:
        key = (int(note.pitch), int(note.instrument_id))
        pitch_notes = by_pitch.setdefault(key, [])
        if pitch_notes:
            previous_note = pitch_notes[-1]
            separation_frames = max(
                1 if separate_adjacent_same_pitch else 0,
                int(min_separation_frames),
            )
            has_conflicting_boundary = (
                previous_note.end_frame > note.start_frame - separation_frames
                if separate_adjacent_same_pitch
                else previous_note.end_frame > note.start_frame
            )
            if has_conflicting_boundary:
                # Keep same-pitch note-off safely before the next note-on in the
                # exported MIDI track. A one-sample gap can quantize back to the
                # same MIDI tick, which some consumers read as a sustaining note.
                new_end_frame = int(note.start_frame) - separation_frames
                # 短縮後の長さが極端に短い場合は残さずに削除する
                if new_end_frame - previous_note.start_frame >= separation_frames:
                    pitch_notes[-1] = replace(
                        previous_note,
                        end_frame=new_end_frame,
                        has_offset=True,
                    )
                else:
                    pitch_notes.pop()
        pitch_notes.append(note)

    valid_notes = [
        note
        for pitch_notes in by_pitch.values()
        for note in pitch_notes
        if note.start_frame < note.end_frame
    ]
    return sorted(
        valid_notes,
        key=lambda note: (
            note.start_frame,
            note.pitch,
            note.instrument_id,
            note.end_frame,
        ),
    )


def _enforce_minimum_note_duration(
    notes: list[PredictedNote],
    *,
    min_duration_frames: int,
    min_separation_frames: int = 0,
) -> list[PredictedNote]:
    if not notes or min_duration_frames <= 1:
        return notes

    ordered_notes = sorted(
        notes,
        key=lambda note: (
            note.pitch,
            note.start_frame,
            note.end_frame,
        ),
    )
    by_pitch: dict[int, list[PredictedNote]] = defaultdict(list)
    for note in ordered_notes:
        by_pitch[int(note.pitch)].append(note)

    adjusted_notes: list[PredictedNote] = []
    for pitch_notes in by_pitch.values():
        for note_index, note in enumerate(pitch_notes):
            target_end_frame = max(
                int(note.end_frame),
                int(note.start_frame) + int(min_duration_frames),
            )
            if note_index + 1 < len(pitch_notes):
                next_note = pitch_notes[note_index + 1]
                latest_safe_end_frame = int(next_note.start_frame) - int(
                    min_separation_frames
                )
                target_end_frame = min(target_end_frame, latest_safe_end_frame)
            if target_end_frame <= int(note.start_frame):
                adjusted_notes.append(note)
                continue
            adjusted_notes.append(replace(note, end_frame=int(target_end_frame)))

    return sorted(
        adjusted_notes,
        key=lambda note: (
            note.start_frame,
            note.pitch,
            note.end_frame,
        ),
    )


def _filter_long_notes(
    notes: list[PredictedNote],
    *,
    max_duration_frames: int,
) -> tuple[list[PredictedNote], int]:
    """一定時間以上の長すぎるノートを除外する。"""
    if max_duration_frames <= 0:
        return notes, 0

    kept_notes: list[PredictedNote] = []
    removed_count = 0
    for note in notes:
        # 15秒以上を削除したいので、閾値ちょうども除外対象にする。
        duration_frames = int(note.end_frame) - int(note.start_frame)
        if duration_frames >= int(max_duration_frames):
            removed_count += 1
            continue
        kept_notes.append(note)

    return kept_notes, removed_count


def _remap_overflow_midi_instruments(
    notes: list[PredictedNote],
    *,
    max_melodic_instruments: int,
) -> tuple[list[PredictedNote], dict[str, int]]:
    if not notes or max_melodic_instruments <= 0:
        return notes, {
            "midi_instrument_count_before_remap": len(
                {int(note.instrument_id) for note in notes}
            ),
            "midi_instrument_count_after_remap": len(
                {int(note.instrument_id) for note in notes}
            ),
            "remapped_instrument_count": 0,
            "remapped_note_count": 0,
        }

    from instrument_classes import INSTRUMENT_CLASSES

    drum_ids = {
        idx
        for idx, class_name in enumerate(INSTRUMENT_CLASSES)
        if class_name.lower() == "drums"
    }
    counts = Counter(int(note.instrument_id) for note in notes)
    melodic_ids = [inst_id for inst_id in counts if inst_id not in drum_ids]
    if len(melodic_ids) <= max_melodic_instruments:
        return notes, {
            "midi_instrument_count_before_remap": len(counts),
            "midi_instrument_count_after_remap": len(counts),
            "remapped_instrument_count": 0,
            "remapped_note_count": 0,
        }

    kept_melodic_ids = set(
        sorted(melodic_ids, key=lambda inst_id: (-counts[inst_id], inst_id))[
            :max_melodic_instruments
        ]
    )
    overflow_ids = set(melodic_ids) - kept_melodic_ids
    fallback_inst_id = max(kept_melodic_ids, key=lambda inst_id: counts[inst_id])

    remapped_notes: list[PredictedNote] = []
    remapped_note_count = 0
    for note in notes:
        inst_id = int(note.instrument_id)
        if inst_id not in overflow_ids:
            remapped_notes.append(note)
            continue

        replacement_inst_id = None
        for candidate_id in note.instrument_candidates:
            candidate_id = int(candidate_id)
            if candidate_id in kept_melodic_ids:
                replacement_inst_id = candidate_id
                break
        if replacement_inst_id is None:
            replacement_inst_id = fallback_inst_id

        remapped_notes.append(replace(note, instrument_id=int(replacement_inst_id)))
        remapped_note_count += 1

    return sorted(
        remapped_notes,
        key=lambda note: (
            note.start_frame,
            note.pitch,
            note.instrument_id,
            note.end_frame,
        ),
    ), {
        "midi_instrument_count_before_remap": len(counts),
        "midi_instrument_count_after_remap": len(
            {int(note.instrument_id) for note in remapped_notes}
        ),
        "remapped_instrument_count": len(overflow_ids),
        "remapped_note_count": remapped_note_count,
    }


def _build_midi(
    notes: list[PredictedNote],
    *,
    sample_rate: int,
    min_midi_note_ms: float = 5.0,
    instrument_volumes: dict[str, int] | None = None,
) -> pretty_midi.PrettyMIDI:
    from instrument_classes import INSTRUMENT_CLASSES, get_program_number_from_class_id

    midi = pretty_midi.PrettyMIDI(resolution=1920)
    short_note_velocity_threshold_ms = 30.0
    short_note_velocity_scale = 0.9
    min_midi_note_frames = max(
        0,
        int(round(float(min_midi_note_ms) * float(sample_rate) / 1000.0)),
    )
    min_separation_frames = max(1, int(round(float(sample_rate) * 0.005)))

    notes_by_inst: dict[int, list[PredictedNote]] = {}
    for note in notes:
        notes_by_inst.setdefault(note.instrument_id, []).append(note)

    for inst_id, inst_notes in notes_by_inst.items():
        class_name = (
            INSTRUMENT_CLASSES[inst_id]
            if 0 <= inst_id < len(INSTRUMENT_CLASSES)
            else "Piano"
        )
        prog_num = get_program_number_from_class_id(inst_id)
        is_drum = class_name.lower() == "drums"

        instrument = pretty_midi.Instrument(
            program=prog_num, is_drum=is_drum, name=class_name
        )
        class_key = _normalize_instrument_class_name(class_name)
        if instrument_volumes and class_key in instrument_volumes:
            instrument.control_changes.append(
                pretty_midi.ControlChange(
                    number=7,
                    value=int(instrument_volumes[class_key]),
                    time=0.0,
                )
            )
        inst_notes = _truncate_overlapping_notes(
            inst_notes,
            separate_adjacent_same_pitch=True,
            min_separation_frames=min_separation_frames,
        )
        inst_notes = _enforce_minimum_note_duration(
            inst_notes,
            min_duration_frames=min_midi_note_frames,
            min_separation_frames=min_separation_frames,
        )
        for note in inst_notes:
            duration_ms = (
                float(int(note.end_frame) - int(note.start_frame)) * 1000.0
            ) / float(sample_rate)
            velocity = int(note.velocity)
            if duration_ms <= short_note_velocity_threshold_ms:
                velocity = max(
                    1,
                    min(127, int(round(float(velocity) * short_note_velocity_scale))),
                )
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=int(note.pitch),
                    start=float(note.start_frame) / float(sample_rate),
                    end=float(note.end_frame) / float(sample_rate),
                )
            )
        midi.instruments.append(instrument)

    return midi


@torch.inference_mode()
def run_inference(
    *,
    model: AudioSemiCRFTransformer,
    waveform: torch.Tensor,
    model_config: SemiCRFModelConfig,
    settings: InferenceSettings,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    velocity: int,
    merge_gap_ms: float | None,
    merge_onset_ms: float,
    max_note_seconds: float,
    silence_gate_rms_dbfs: float | None,
    window_batch_size: int,
    max_midi_melodic_instruments: int,
    disable_tqdm: bool,
) -> tuple[list[PredictedNote], dict[str, int], dict[str, np.ndarray]]:
    if waveform.dim() != 2 or int(waveform.shape[0]) != 2:
        raise ValueError("waveform must have shape [2, audio_frames]")

    sample_rate = int(model_config.sample_rate)
    total_audio_frames = int(waveform.shape[-1])
    window_audio_frames = int(round(settings.window_ms * sample_rate / 1000.0))
    stride_audio_frames = int(round(settings.stride_ms * sample_rate / 1000.0))
    if window_audio_frames < int(model_config.n_fft):
        raise ValueError(
            f"window_ms={settings.window_ms} is too short for n_fft={model_config.n_fft}"
        )
    if total_audio_frames < int(model_config.n_fft):
        raise ValueError(
            f"Audio is too short for n_fft={model_config.n_fft}: {total_audio_frames} frames"
        )
    if stride_audio_frames <= 0:
        raise ValueError("stride_ms must be positive")

    # 1. 推論全体で共有する設定と状態を準備する。
    window_starts = _build_window_starts(
        total_audio_frames=total_audio_frames,
        window_audio_frames=window_audio_frames,
        stride_audio_frames=stride_audio_frames,
    )
    silence_gate_linear = _silence_gate_rms_linear(silence_gate_rms_dbfs)
    skipped_silent_window_count = 0
    decoded_window_count = 0
    progress = tqdm(
        _iter_batches(window_starts, window_batch_size),
        total=math.ceil(len(window_starts) / max(1, int(window_batch_size))),
        desc="infer",
        dynamic_ncols=True,
        disable=bool(disable_tqdm),
    )

    hop_length = int(model_config.hop_length)
    total_model_frames = math.ceil(total_audio_frames / hop_length)
    merge_gap_frames = (
        int(model_config.hop_length)
        if merge_gap_ms is None
        else max(0, int(round(float(merge_gap_ms) * sample_rate / 1000.0)))
    )
    merge_onset_frames = max(
        0, int(round(float(merge_onset_ms) * sample_rate / 1000.0))
    )
    max_note_frames = max(0, int(round(float(max_note_seconds) * sample_rate)))
    note_stitcher = PitchFirstNoteStitcher(
        hop_length=hop_length,
        total_audio_frames=total_audio_frames,
        num_pitch_slots=int(model_config.num_pitch_slots),
        velocity=int(velocity),
        merge_gap_frames=merge_gap_frames,
        merge_onset_frames=merge_onset_frames,
        instrument_probability_mode=settings.instrument_probability_mode,
    )

    # 2. ノート処理とは独立に、補助タスク用ロジットの集約バッファを用意する。
    # chord
    agg_chord_boundary = np.zeros(total_model_frames, dtype=np.float32)
    agg_root_chord = None  # 後で次元がわかってから初期化
    agg_bass = None
    agg_key_boundary = np.zeros(total_model_frames, dtype=np.float32)
    agg_key = None
    # beat
    agg_beat = np.zeros(total_model_frames, dtype=np.float32)
    agg_downbeat = np.zeros(total_model_frames, dtype=np.float32)
    agg_meter = None

    agg_weights = np.zeros(total_model_frames, dtype=np.float32)

    # 三角形の重み窓
    model_frames_per_window = math.ceil(window_audio_frames / hop_length)
    window_weight = np.bartlett(model_frames_per_window).astype(np.float32)

    use_boundary_head = bool(model.supports_interval_boundaries())
    for batch_starts in progress:
        # 3. 窓ごとの audio batch を作る。
        window_tensors = []
        valid_audio_frames = []
        for start_frame in batch_starts:
            window, valid_frames = _slice_window(
                waveform,
                start_frame=int(start_frame),
                window_audio_frames=window_audio_frames,
            )
            window_tensors.append(window)
            valid_audio_frames.append(int(valid_frames))

        batch_waveform_cpu = torch.stack(window_tensors, dim=0)
        silent_window_mask = _compute_silent_window_mask(
            batch_waveform_cpu,
            silence_gate_rms_linear=silence_gate_linear,
        )
        if silent_window_mask is not None:
            skipped_silent_window_count += int(silent_window_mask.sum().item())
            active_indices = [
                index
                for index, is_silent in enumerate(silent_window_mask.tolist())
                if not is_silent
            ]
        else:
            active_indices = list(range(len(batch_starts)))

        if not active_indices:
            continue

        active_batch_starts = [int(batch_starts[index]) for index in active_indices]
        active_valid_audio_frames = [
            int(valid_audio_frames[index]) for index in active_indices
        ]
        decoded_window_count += len(active_indices)

        batch_waveform = batch_waveform_cpu[active_indices].to(device)
        valid_audio_frames_tensor = torch.tensor(
            active_valid_audio_frames,
            dtype=torch.long,
            device=device,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_enabled and device.type == "cuda"),
        ):
            outputs = model(
                batch_waveform,
                valid_audio_frames=valid_audio_frames_tensor,
            )

        # 4. ノート decode は状態を持つので、窓順に stitcher へ流し込む。
        valid_lengths = outputs["frame_valid_mask"].to(dtype=torch.long).sum(dim=-1)
        num_tracks = int(outputs["interval_query"].shape[2])

        # 1. forward はまとめて計算しつつ、decode は時間順状態を保つため 1 窓ずつ行う。
        decoded_intervals_batch: list[list[list[tuple[int, int]]]] = []
        for sample_index, start_frame in enumerate(active_batch_starts):
            sample_valid_length = int(valid_lengths[sample_index].item())
            if sample_valid_length <= 0:
                decoded_intervals_batch.append([[] for _ in range(num_tracks)])
                continue

            forced_start_pos = note_stitcher.get_forced_start_positions(
                window_start_frame=int(start_frame),
                num_tracks=num_tracks,
                valid_model_frames=int(sample_valid_length),
            )
            decoded_intervals = decode_pitch_intervals(
                outputs["interval_query"][sample_index : sample_index + 1],
                outputs["interval_key"][sample_index : sample_index + 1],
                outputs["interval_diag"][sample_index : sample_index + 1],
                valid_lengths[sample_index : sample_index + 1],
                length_scaling=settings.length_scaling,
                length_penalty=settings.length_penalty,
                note_bias=settings.note_bias,
                track_batch_size=settings.track_batch_size,
                forced_start_pos=[forced_start_pos],
            )
            decoded_intervals_batch.append(decoded_intervals[0])

        boundary_flags_batch: (
            list[list[list[tuple[bool, bool, float, float]]]] | None
        ) = None
        if use_boundary_head:
            # boundary head は autocast の外で呼ぶので、入力も明示的に FP32 に揃える。
            boundary_logits, boundary_entries = model.predict_interval_boundaries(
                outputs.get(
                    "interval_features", outputs["pitch_query_features"]
                ).float(),
                decoded_intervals_batch,
            )
            boundary_flags_batch = _decode_boundary_features(
                boundary_logits,
                boundary_entries,
                batch_size=len(active_batch_starts),
                num_tracks=num_tracks,
            )

        interval_instrument_stats_batch: (
            list[list[list[tuple[np.ndarray, int]]]] | None
        ) = None
        if (
            hasattr(model, "supports_interval_instruments")
            and model.supports_interval_instruments()
            and getattr(model, "_use_interval_instrument_head", True)
            and outputs.get("instrument_features") is not None
        ):
            interval_instrument_logits, interval_instrument_entries = (
                model.predict_interval_instruments(
                    outputs["instrument_features"].float(),
                    decoded_intervals_batch,
                )
            )
            interval_instrument_stats_batch = _decode_interval_instrument_logits(
                interval_instrument_logits,
                interval_instrument_entries,
                batch_size=len(active_batch_starts),
                num_tracks=num_tracks,
                probability_mode=settings.instrument_probability_mode,
            )

        sample_logits_batch = outputs.get("instrument_logits")

        for sample_index, (start_frame, valid_frames) in enumerate(
            zip(active_batch_starts, active_valid_audio_frames)
        ):
            sample_valid_length = int(valid_lengths[sample_index].item())
            intervals_by_track = decoded_intervals_batch[sample_index]
            sample_boundary_flags = (
                boundary_flags_batch[sample_index]
                if boundary_flags_batch is not None
                else None
            )
            sample_interval_instrument_stats = (
                interval_instrument_stats_batch[sample_index]
                if interval_instrument_stats_batch is not None
                else None
            )
            sample_logits = (
                sample_logits_batch[sample_index, :sample_valid_length]
                if sample_logits_batch is not None
                else None
            )
            note_stitcher.consume_window(
                intervals_by_track=intervals_by_track,
                boundary_flags_by_track=sample_boundary_flags,
                interval_instrument_stats_by_track=sample_interval_instrument_stats,
                instrument_logits=sample_logits,
                window_start_frame=int(start_frame),
                valid_audio_frames=int(valid_frames),
                valid_model_frames=int(sample_valid_length),
            )

            # 5. Chord / Beat のロジットはノート処理と分けて平均集約する。
            f_start = int(round(start_frame / hop_length))
            f_end = f_start + int(sample_valid_length)
            f_end = min(f_end, total_model_frames)
            num_valid_f = f_end - f_start

            # 重みの適用（端を減衰させる）
            w = window_weight[:num_valid_f]
            agg_weights[f_start:f_end] += w

            # Chord
            if "root_chord_logits" in outputs:
                if agg_root_chord is None:
                    agg_root_chord = np.zeros(
                        (total_model_frames, outputs["root_chord_logits"].shape[-1]),
                        dtype=np.float32,
                    )
                    agg_bass = np.zeros(
                        (total_model_frames, outputs["bass_logits"].shape[-1]),
                        dtype=np.float32,
                    )
                    agg_key = np.zeros(
                        (total_model_frames, outputs["key_logits"].shape[-1]),
                        dtype=np.float32,
                    )

                # numpy に変換して加算
                cb = (
                    outputs["chord_boundary_logits"][sample_index, :num_valid_f]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                rc = (
                    outputs["root_chord_logits"][sample_index, :num_valid_f]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                ba = (
                    outputs["bass_logits"][sample_index, :num_valid_f]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                kb = (
                    outputs["key_boundary_logits"][sample_index, :num_valid_f]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                ke = (
                    outputs["key_logits"][sample_index, :num_valid_f]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )

                agg_chord_boundary[f_start:f_end] += cb * w
                agg_root_chord[f_start:f_end] += rc * w[:, None]
                agg_bass[f_start:f_end] += ba * w[:, None]
                agg_key_boundary[f_start:f_end] += kb * w
                agg_key[f_start:f_end] += ke * w[:, None]

            # Beat
            if "beat_logits" in outputs:
                if agg_meter is None:
                    agg_meter = np.zeros(
                        (total_model_frames, outputs["meter_logits"].shape[-1]),
                        dtype=np.float32,
                    )

                bt = (
                    outputs["beat_logits"][sample_index, :num_valid_f]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                db = (
                    outputs["downbeat_logits"][sample_index, :num_valid_f]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                mt = (
                    outputs["meter_logits"][sample_index, :num_valid_f]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )

                agg_beat[f_start:f_end] += bt * w
                agg_downbeat[f_start:f_end] += db * w
                agg_meter[f_start:f_end] += mt * w[:, None]

    # 6. stitcher から最終ノート系列を受け取り、MIDI 出力向け後処理を行う。
    merged_notes = note_stitcher.finalize()

    remapped_notes, remap_stats = _remap_overflow_midi_instruments(
        merged_notes,
        max_melodic_instruments=int(max_midi_melodic_instruments),
    )
    if remap_stats["remapped_note_count"] > 0:
        merged_notes = _truncate_overlapping_notes(
            remapped_notes,
            separate_adjacent_same_pitch=True,
            min_separation_frames=max(1, int(round(float(sample_rate) * 0.005))),
        )
        remap_stats["midi_instrument_count_after_remap"] = len(
            {int(note.instrument_id) for note in merged_notes}
        )
    else:
        merged_notes = _truncate_overlapping_notes(
            remapped_notes,
            separate_adjacent_same_pitch=True,
            min_separation_frames=max(1, int(round(float(sample_rate) * 0.005))),
        )

    # 4. 明らかに長すぎる持続ノートを最終段で除外する。
    merged_notes, removed_long_note_count = _filter_long_notes(
        merged_notes,
        max_duration_frames=max_note_frames,
    )
    remap_stats["midi_instrument_count_after_remap"] = len(
        {int(note.instrument_id) for note in merged_notes}
    )

    # ロジットの正規化 (除算)
    # 重みが 0 の箇所（末尾など）での 0 除算を避ける
    safe_weights = np.where(agg_weights > 1e-6, agg_weights, 1.0)

    aggregated_logits = {}
    if agg_root_chord is not None:
        aggregated_logits["chord_boundary_logits"] = agg_chord_boundary / safe_weights
        aggregated_logits["root_chord_logits"] = agg_root_chord / safe_weights[:, None]
        aggregated_logits["bass_logits"] = agg_bass / safe_weights[:, None]
        aggregated_logits["key_boundary_logits"] = agg_key_boundary / safe_weights
        aggregated_logits["key_logits"] = agg_key / safe_weights[:, None]

    if agg_meter is not None:
        aggregated_logits["beat_logits"] = agg_beat / safe_weights
        aggregated_logits["downbeat_logits"] = agg_downbeat / safe_weights
        aggregated_logits["meter_logits"] = agg_meter / safe_weights[:, None]

    return (
        merged_notes,
        {
            "window_count": len(window_starts),
            "decoded_window_count": int(decoded_window_count),
            "skipped_silent_window_count": int(skipped_silent_window_count),
            "window_audio_frames": int(window_audio_frames),
            "stride_audio_frames": int(stride_audio_frames),
            "merge_gap_frames": int(merge_gap_frames),
            "merge_onset_frames": int(merge_onset_frames),
            "removed_long_note_count": int(removed_long_note_count),
            "max_note_frames": int(max_note_frames),
            **remap_stats,
        },
        aggregated_logits,
    )


def _collect_audio_files(directory: Path) -> list[Path]:
    """ディレクトリ内の対応音声ファイルを再帰的に収集する。"""
    files = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    return files


def _format_decoded_notes_by_pitch_slot(
    notes: list[PredictedNote],
    *,
    num_pitch_slots: int,
) -> str:
    slot_count = max(1, int(num_pitch_slots))
    slot_counts = Counter(int(note.slot_index) for note in notes)
    return ",".join(
        f"slot{slot_index}:{slot_counts.get(slot_index, 0)}"
        for slot_index in range(slot_count)
    )


def _process_single_file(
    audio_path: Path,
    output_midi_path: Path,
    *,
    model: AudioSemiCRFTransformer,
    model_config: SemiCRFModelConfig,
    settings: InferenceSettings,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
) -> None:
    """1ファイルの音声読み込み→推論→MIDI出力を行う。"""
    print(f"Loading audio from {audio_path}...")
    waveform, source_sample_rate, source_channels = _load_audio(
        audio_path,
        target_sample_rate=int(model_config.sample_rate),
    )

    print("Running inference...")
    notes, stats, aggregated_logits = run_inference(
        model=model,
        waveform=waveform,
        model_config=model_config,
        settings=settings,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        velocity=int(args.velocity),
        merge_gap_ms=args.merge_gap_ms,
        merge_onset_ms=args.merge_onset_ms,
        max_note_seconds=float(args.max_note_seconds),
        silence_gate_rms_dbfs=args.silence_gate_rms_dbfs,
        window_batch_size=int(args.window_batch_size),
        max_midi_melodic_instruments=int(args.max_midi_melodic_instruments),
        disable_tqdm=bool(args.disable_tqdm),
    )

    midi = _build_midi(
        notes,
        sample_rate=int(model_config.sample_rate),
        min_midi_note_ms=float(args.min_midi_note_ms),
        instrument_volumes=dict(args.instrument_volume_map),
    )

    # メタデータの埋め込み
    if aggregated_logits and not (args.disable_beat and args.disable_chord):
        print("Embedding chords, keys, and beats into MIDI...")
        embedder = MidiMetadataEmbedder(
            sample_rate=int(model_config.sample_rate),
            hop_length=int(model_config.hop_length),
        )
        # miditoolkit オブジェクトが返る
        midi = embedder.embed_all(
            midi,
            aggregated_logits,
            disable_beat=args.disable_beat,
            disable_chord=args.disable_chord,
        )

    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(midi, "dump"):
        midi.dump(str(output_midi_path))
    else:
        midi.write(str(output_midi_path))

    duration_seconds = float(waveform.shape[-1]) / float(model_config.sample_rate)
    decoded_notes_by_pitch_slot = _format_decoded_notes_by_pitch_slot(
        notes,
        num_pitch_slots=int(model_config.num_pitch_slots),
    )
    print(
        f"audio_source_sr={source_sample_rate} "
        f"model_sr={model_config.sample_rate} "
        f"source_channels={source_channels} "
        f"used_channels=2"
    )
    print(
        f"duration_seconds={duration_seconds:.2f} "
        f"window_ms={settings.window_ms} "
        f"stride_ms={settings.stride_ms} "
        f"windows={stats['window_count']} "
        f"decoded_windows={stats['decoded_window_count']} "
        f"skipped_silent_windows={stats['skipped_silent_window_count']}"
    )
    print(f"decoded_notes_by_pitch_slot={decoded_notes_by_pitch_slot}")
    print(
        f"decoded_notes={len(notes)} "
        f"removed_long_notes={stats['removed_long_note_count']} "
        f"max_note_seconds={args.max_note_seconds} "
        f"midi_instruments_before_remap={stats['midi_instrument_count_before_remap']} "
        f"midi_instruments_after_remap={stats['midi_instrument_count_after_remap']} "
        f"remapped_instruments={stats['remapped_instrument_count']} "
        f"remapped_notes={stats['remapped_note_count']} "
        f"silence_gate_rms_dbfs={args.silence_gate_rms_dbfs} "
        f"output_midi={output_midi_path}"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)

    checkpoint_path = _ensure_checkpoint(args.checkpoint, model_type=args.type)
    print(f"Loading checkpoint from {checkpoint_path}...")
    model, model_config, settings = _load_model_and_settings(
        checkpoint_path.resolve(),
        device=device,
        window_ms_override=args.window_ms,
        stride_ms_override=args.stride_ms,
        track_batch_size_override=args.semi_crf_track_batch_size,
        note_bias=args.note_bias,
    )

    # 処理対象ファイルリストの構築
    if args.audio is not None:
        # 単一ファイルモード
        audio_path = args.audio.resolve()
        output_midi_path = (
            args.output_midi.resolve()
            if args.output_midi is not None
            else audio_path.with_suffix(".mid")
        )
        file_pairs = [(audio_path, output_midi_path)]
    else:
        # ディレクトリバッチモード
        audio_dir = args.audio_dir.resolve()
        output_dir = (
            args.output_dir.resolve() if args.output_dir is not None else audio_dir
        )
        audio_files = _collect_audio_files(audio_dir)
        if not audio_files:
            print(f"No audio files found in {audio_dir}")
            return
        file_pairs = [
            (path, output_dir / path.relative_to(audio_dir).with_suffix(".mid"))
            for path in audio_files
        ]
        print(f"Found {len(file_pairs)} audio file(s) in {audio_dir}")

    shared_kwargs = dict(
        model=model,
        model_config=model_config,
        settings=settings,
        device=device,
        amp_enabled=bool(args.amp),
        amp_dtype=amp_dtype,
        args=args,
    )

    for file_index, (audio_path, output_midi_path) in enumerate(file_pairs):
        if len(file_pairs) > 1:
            print(f"\n[{file_index + 1}/{len(file_pairs)}] {audio_path.name}")
        _process_single_file(audio_path, output_midi_path, **shared_kwargs)

    if len(file_pairs) > 1:
        print(f"\nDone. Processed {len(file_pairs)} file(s).")


if __name__ == "__main__":
    main()
