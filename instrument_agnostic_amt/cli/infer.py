from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..inference.audio import collect_audio_files, load_audio
from ..inference.midi import build_midi
from ..inference.types import InferenceSettings
from ..inference.windowed import decode_notes
from ..modeling.model import AudioSemiCRFTransformer, SemiCRFModelConfig
from ..taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_instrument_class_id_by_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V2 conditioned AMT inference.")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="Path to a V2 checkpoint"
    )
    parser.add_argument(
        "--instrument", type=str, required=True, help="Target instrument class name"
    )
    parser.add_argument("--audio", type=Path, default=None, help="Input audio file")
    parser.add_argument(
        "--output-midi", type=Path, default=None, help="Output MIDI for --audio"
    )
    parser.add_argument(
        "--audio-dir", type=Path, default=None, help="Directory for batch inference"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for --audio-dir"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--amp", action="store_true", help="Enable CUDA autocast")
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default="bf16" if torch.cuda.is_bf16_supported() else "fp16",
    )
    parser.add_argument(
        "--window-ms",
        type=int,
        default=None,
        help=(
            "Inference window size in milliseconds. Defaults to checkpoint "
            "training window, or 8000."
        ),
    )
    parser.add_argument(
        "--stride-ms",
        type=int,
        default=None,
        help="Inference stride in milliseconds. Defaults to half of window-ms.",
    )
    parser.add_argument(
        "--window-batch-size",
        type=int,
        default=1,
        help="Number of windows to forward at once. Keep at 1 unless VRAM allows more.",
    )
    parser.add_argument(
        "--semi-crf-track-batch-size",
        type=int,
        default=None,
        help="Chunk size for Semi-CRF track decoding. Defaults to checkpoint value, or 128.",
    )
    parser.add_argument("--merge-gap-ms", type=float, default=None)
    parser.add_argument("--merge-onset-ms", type=float, default=20.0)
    parser.add_argument(
        "--silence-gate-rms-dbfs",
        type=float,
        default=-72.0,
        help=(
            "Skip fully silent windows below this RMS level. Set very low "
            "to effectively disable."
        ),
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--note-bias", type=float, default=0.0)
    parser.add_argument("--velocity", type=int, default=100)
    parser.add_argument("--min-midi-note-ms", type=float, default=5.0)
    args = parser.parse_args()
    if args.audio is None and args.audio_dir is None:
        parser.error("one of --audio or --audio-dir is required")
    if args.audio is not None and args.audio_dir is not None:
        parser.error("--audio and --audio-dir cannot be used together")
    if args.output_midi is not None and args.audio_dir is not None:
        parser.error("--output-midi cannot be used with --audio-dir")
    if args.window_ms is not None and args.window_ms <= 0:
        parser.error("--window-ms must be positive")
    if args.stride_ms is not None and args.stride_ms <= 0:
        parser.error("--stride-ms must be positive")
    if args.window_batch_size <= 0:
        parser.error("--window-batch-size must be positive")
    if (
        args.semi_crf_track_batch_size is not None
        and args.semi_crf_track_batch_size <= 0
    ):
        parser.error("--semi-crf-track-batch-size must be positive")
    if args.merge_gap_ms is not None and args.merge_gap_ms < 0:
        parser.error("--merge-gap-ms must be non-negative")
    if args.merge_onset_ms < 0:
        parser.error("--merge-onset-ms must be non-negative")
    if args.silence_gate_rms_dbfs is not None and args.silence_gate_rms_dbfs > 0:
        parser.error("--silence-gate-rms-dbfs must be <= 0")
    return args


def resolve_amp_dtype(device: torch.device, dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float16


def _coerce_model_config(raw_model_config: dict[str, Any]) -> SemiCRFModelConfig:
    allowed = {field.name for field in fields(SemiCRFModelConfig)}
    kwargs = {key: value for key, value in raw_model_config.items() if key in allowed}
    if int(kwargs.get("architecture_version", 1)) != 2:
        raise ValueError("This inference entrypoint only supports V2 checkpoints")
    if "encoder_head_dim" not in kwargs:
        kwargs["encoder_head_dim"] = int(kwargs.get("hidden_size", 256)) // int(
            kwargs.get("encoder_num_heads", 8)
        )
    kwargs["use_gradient_checkpoint"] = False
    kwargs["spec_augment_params"] = None
    return SemiCRFModelConfig(**kwargs)


def load_model(
    checkpoint_path: Path, *, device: torch.device
) -> tuple[AudioSemiCRFTransformer, SemiCRFModelConfig, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    raw_config = checkpoint.get("model_config")
    raw_run_config = checkpoint.get("config")
    if raw_config is None and isinstance(raw_run_config, dict):
        raw_config = raw_run_config.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint does not contain model_config")
    config = _coerce_model_config(raw_config)
    model = AudioSemiCRFTransformer(config)
    state_dict = checkpoint.get("ema_state_dict") or checkpoint.get("model_state_dict")
    if state_dict is None:
        state_dict = checkpoint
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys:
        print(f"Missing keys while loading checkpoint: {incompatible.missing_keys}")
    if incompatible.unexpected_keys:
        print(
            f"Unexpected keys while loading checkpoint: {incompatible.unexpected_keys}"
        )
    model.to(device)
    model.eval()
    training_args = (
        raw_run_config.get("args", {}) if isinstance(raw_run_config, dict) else {}
    )
    if not isinstance(training_args, dict):
        training_args = {}
    return model, config, training_args


def resolve_inference_settings(
    config: SemiCRFModelConfig,
    training_args: dict[str, Any],
    args: argparse.Namespace,
) -> InferenceSettings:
    default_window_ms = int(training_args.get("window_ms") or 8000)
    window_ms = int(args.window_ms) if args.window_ms is not None else default_window_ms
    stride_ms = (
        int(args.stride_ms) if args.stride_ms is not None else max(1, window_ms // 2)
    )
    track_batch_size = (
        int(args.semi_crf_track_batch_size)
        if args.semi_crf_track_batch_size is not None
        else int(training_args.get("semi_crf_track_batch_size") or 128)
    )
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if stride_ms <= 0:
        raise ValueError("stride_ms must be positive")
    if track_batch_size <= 0:
        raise ValueError("semi_crf_track_batch_size must be positive")
    if int(round(window_ms * int(config.sample_rate) / 1000.0)) < int(config.n_fft):
        raise ValueError(f"window_ms={window_ms} is too short for n_fft={config.n_fft}")
    return InferenceSettings(
        window_ms=window_ms,
        stride_ms=stride_ms,
        track_batch_size=track_batch_size,
        window_batch_size=int(args.window_batch_size),
        merge_gap_ms=args.merge_gap_ms,
        merge_onset_ms=float(args.merge_onset_ms),
        silence_gate_rms_dbfs=args.silence_gate_rms_dbfs,
        note_bias=float(args.note_bias),
        disable_tqdm=bool(args.disable_tqdm),
    )


def resolve_instrument_id(name: str) -> int:
    normalized = name.strip()
    try:
        return get_instrument_class_id_by_name(normalized)
    except KeyError:
        available = ", ".join(INSTRUMENT_CLASSES)
        raise ValueError(
            f"Unknown instrument class '{name}'. Available classes: {available}"
        ) from None


def process_file(
    audio_path: Path,
    output_midi_path: Path,
    *,
    model: AudioSemiCRFTransformer,
    config: SemiCRFModelConfig,
    instrument_id: int,
    settings: InferenceSettings,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
) -> None:
    waveform = load_audio(audio_path, target_sample_rate=int(config.sample_rate))
    notes, stats = decode_notes(
        model,
        config,
        waveform,
        condition_instrument_id=instrument_id,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        settings=settings,
        velocity=int(args.velocity),
    )
    midi = build_midi(
        notes,
        sample_rate=int(config.sample_rate),
        instrument_id=instrument_id,
        min_midi_note_ms=float(args.min_midi_note_ms),
    )
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_midi_path))
    print(
        f"wrote {len(notes)} notes: {output_midi_path} "
        f"window_ms={settings.window_ms} stride_ms={settings.stride_ms} "
        f"windows={stats['window_count']} "
        f"decoded_windows={stats['decoded_window_count']} "
        f"skipped_silent_windows={stats['skipped_silent_window_count']}"
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    amp_enabled = bool(args.amp and device.type == "cuda")
    instrument_id = resolve_instrument_id(args.instrument)
    model, config, training_args = load_model(args.checkpoint.resolve(), device=device)
    settings = resolve_inference_settings(config, training_args, args)

    if args.audio_dir is not None:
        audio_dir = args.audio_dir.resolve()
        output_dir = (
            args.output_dir.resolve() if args.output_dir is not None else audio_dir
        )
        files = collect_audio_files(audio_dir)
        for audio_path in tqdm(files, desc="Files"):
            output_path = output_dir / audio_path.relative_to(audio_dir).with_suffix(
                ".mid"
            )
            process_file(
                audio_path,
                output_path,
                model=model,
                config=config,
                instrument_id=instrument_id,
                settings=settings,
                device=device,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                args=args,
            )
        return

    audio_path = args.audio.resolve()
    output_path = (
        args.output_midi.resolve()
        if args.output_midi is not None
        else audio_path.with_suffix(".mid")
    )
    process_file(
        audio_path,
        output_path,
        model=model,
        config=config,
        instrument_id=instrument_id,
        settings=settings,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        args=args,
    )


if __name__ == "__main__":
    main()
