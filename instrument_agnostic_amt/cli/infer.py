from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ..data.pitch_aliases import DEFAULT_DRUM_PITCH_ALIASES
from ..inference.audio import collect_audio_files, load_audio
from ..inference.instruments import (
    normalize_instrument_class_name,
    parse_allowed_instrument_args,
)
from ..inference.loading import load_inference_model
from ..inference.midi import build_midi
from ..inference.settings import (
    resolve_inference_settings as resolve_typed_inference_settings,
)
from ..inference.types import InferenceSettings
from ..inference.windowed import decode_notes
from ..modeling.checkpoints import CheckpointLoadReport
from ..modeling.model import AudioSemiCRFTransformer, SemiCRFModelConfig
from ..runtime import (
    resolve_device,
)
from ..taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_instrument_class_id_by_name,
)
from ..transcription import (
    MidiExportOptions,
    Transcriber,
    TranscriptionOptions,
    TranscriptionResult,
)

HF_CHECKPOINT_BASE_URL = (
    "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main"
)
MODEL_CHECKPOINT_FILENAMES = {
    "default": "best_model.pth",
    "bass": "best_model_bass.pth",
    "bass_v2": "best_model_bass_v2.pth",
    "vocal": "best_model_vocal.pth",
    "guitar": "best_model_guitar.pth",
    "guitar_v1_5": "best_model_guitar_v1_5.pth",
    "vocal_harmony": "best_model_vocal_harmony.pth",
    "vocal_harmony_v1_5": "best_model_vocal_harmony_v1_5.pth",
    "drums": "best_model_drums.pth",
    "drums_v1_5": "best_model_drums_v1_5.pth",
    "other": "best_model_other.pth",
    "other_v1_5": "best_model_other_v1_5.pth",
}


def _ensure_checkpoint(checkpoint_path: Path | None, model_type: str = "default") -> Path:
    filename = MODEL_CHECKPOINT_FILENAMES[model_type]
    if checkpoint_path is None:
        checkpoint_path = Path("checkpoints") / filename
    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF_CHECKPOINT_BASE_URL}/{filename}?download=true"
        print(f"Checkpoint not found at {checkpoint_path}. Downloading from Hugging Face...")
        torch.hub.download_url_to_file(url, str(checkpoint_path))
    return checkpoint_path


DEFAULT_INSTRUMENT_VOLUMES: dict[str, int] = {
    "piano": 105,
    "electric_piano": 96,
    "acoustic_guitar": 88,
    "electric_guitar_clean": 84,
    "distorted_guitar": 78,
    "electric_guitar_muted": 68,
    "guitar_harmonics": 72,
    "acoustic_bass": 100,
    "electric_bass": 100,
    "synth_bass": 100,
    "drums": 100,
    "strings": 82,
    "brass": 100,
    "sax": 100,
    "flute_pipe": 84,
    "choir": 80,
    "organ": 88,
    "synth_pad": 76,
    "synth_lead": 76,
    "melody": 120,
    "vocal_harmony": 60,
}


def _normalize_instrument_class_name(name: str) -> str:
    return normalize_instrument_class_name(name)


def _parse_instrument_volume_args(entries: list[str]) -> dict[str, int]:
    if not entries:
        return dict(DEFAULT_INSTRUMENT_VOLUMES)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run V1-compatible or overlap-capable V2 AMT inference."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. The selected --type is downloaded when omitted.",
    )
    parser.add_argument(
        "--type",
        choices=tuple(MODEL_CHECKPOINT_FILENAMES),
        default="default",
        help="Pretrained V1 model type used when --checkpoint is omitted.",
    )

    parser.add_argument(
        "--instrument",
        type=str,
        default=None,
        help="Optional single instrument class. When omitted, all allowed instruments are decoded.",
    )
    parser.add_argument(
        "--allowed-instruments",
        action="append",
        default=[],
        metavar="CLASS[,CLASS...]",
        help=(
            "Restrict instrument classification to these classes. Repeatable "
            "and/or comma-separated. Softmax is recalculated over this subset."
        ),
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
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cuda, mps, or cpu",
    )
    parser.add_argument("--amp", action="store_true", help="Enable accelerator autocast")
    parser.add_argument(
        "--amp-dtype",
        choices=("fp16", "bf16"),
        default=None,
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile shared AMT Transformer regions with TorchInductor",
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
        help="torch.compile mode used when --compile is enabled",
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
    parser.add_argument(
        "--semi-crf-backend",
        choices=("torch", "triton"),
        default="torch",
        help=(
            "Dense Semi-CRF decoder backend. Triton requires CUDA and performs "
            "a one-time JIT compilation for each frame length."
        ),
    )
    parser.add_argument(
        "--semi-crf-sparse-decode",
        action="store_true",
        help="Decode Semi-CRF paths from a sparse interval candidate set.",
    )
    parser.add_argument(
        "--semi-crf-sparse-topk-per-start",
        type=int,
        default=16,
        help="Sparse decode candidates kept per start frame before Viterbi.",
    )
    parser.add_argument(
        "--semi-crf-sparse-score-threshold",
        type=float,
        default=None,
        help="Drop sparse decode interval candidates below this score.",
    )
    parser.add_argument(
        "--semi-crf-sparse-max-span-ms",
        type=float,
        default=None,
        help="Required for sparse decode; only score intervals up to this duration in milliseconds.",
    )
    parser.add_argument(
        "--instrument-pair-infer-topk",
        type=int,
        default=256,
        help="Gate top-k instrument-pitch pairs decoded per window.",
    )
    parser.add_argument(
        "--instrument-pair-gate-threshold",
        type=float,
        default=-3.0,
        help="Decode instrument-pitch pairs with gate logits above this threshold.",
    )
    parser.add_argument(
        "--instrument-pair-max-pairs",
        type=int,
        default=512,
        help="Maximum instrument-pitch pairs decoded per window after threshold/top-k union.",
    )
    parser.add_argument("--merge-gap-ms", type=float, default=None)
    parser.add_argument("--merge-onset-ms", type=float, default=50.0)
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
    parser.add_argument(
        "--no-boundary-head",
        action="store_true",
        help=(
            "Ignore the interval boundary head during stitching. Useful for "
            "checking raw Semi-CRF decoded intervals."
        ),
    )
    parser.add_argument("--velocity", type=int, default=100)
    parser.add_argument(
        "--collapse-crash-cymbals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Map drum pitch 57 (Crash Cymbal 2) to the canonical pitch 49 "
            "(Crash Cymbal 1) in exported MIDI. Enabled by default; use "
            "--no-collapse-crash-cymbals to preserve both pitches."
        ),
    )
    parser.add_argument(
        "--instrument-volume",
        action="append",
        default=[],
        metavar="CLASS=VALUE",
        help=(
            "Set MIDI CC7 volume (0-127) for a specific instrument class. "
            "Repeatable, e.g. --instrument-volume piano=90 "
            "--instrument-volume acoustic_guitar=70. If omitted, a built-in "
            "per-instrument default mix is used."
        ),
    )
    parser.add_argument("--min-midi-note-ms", type=float, default=5.0)
    parser.add_argument(
        "--max-midi-melodic-instruments",
        type=int,
        default=15,
        help=(
            "Maximum number of non-drum instrument tracks in exported MIDI. "
            "Extra low-note-count instruments are reassigned by predicted "
            "instrument confidence rank. Use 0 to disable."
        ),
    )
    args = parser.parse_args()
    if args.audio is None and args.audio_dir is None:
        parser.error("one of --audio or --audio-dir is required")
    if args.audio is not None and args.audio_dir is not None:
        parser.error("--audio and --audio-dir cannot be used together")
    if args.output_midi is not None and args.audio_dir is not None:
        parser.error("--output-midi cannot be used with --audio-dir")
    if args.instrument is not None and args.allowed_instruments:
        parser.error("--instrument and --allowed-instruments cannot be used together")
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
    if args.semi_crf_sparse_topk_per_start <= 0:
        parser.error("--semi-crf-sparse-topk-per-start must be positive")
    if (
        args.semi_crf_sparse_max_span_ms is not None
        and args.semi_crf_sparse_max_span_ms <= 0
    ):
        parser.error("--semi-crf-sparse-max-span-ms must be positive")
    if args.semi_crf_sparse_decode and args.semi_crf_sparse_max_span_ms is None:
        parser.error(
            "--semi-crf-sparse-decode requires --semi-crf-sparse-max-span-ms "
            "to avoid dense score construction"
        )
    if args.semi_crf_sparse_decode and args.semi_crf_backend != "torch":
        parser.error(
            "--semi-crf-backend triton cannot be combined with "
            "--semi-crf-sparse-decode"
        )
    if args.instrument_pair_infer_topk < 0:
        parser.error("--instrument-pair-infer-topk must be non-negative")
    if args.instrument_pair_max_pairs <= 0:
        parser.error("--instrument-pair-max-pairs must be positive")
    if args.merge_gap_ms is not None and args.merge_gap_ms < 0:
        parser.error("--merge-gap-ms must be non-negative")
    if args.merge_onset_ms < 0:
        parser.error("--merge-onset-ms must be non-negative")
    if args.silence_gate_rms_dbfs is not None and args.silence_gate_rms_dbfs > 0:
        parser.error("--silence-gate-rms-dbfs must be <= 0")
    if args.max_midi_melodic_instruments < 0:
        parser.error("--max-midi-melodic-instruments must be non-negative")
    try:
        args.instrument_volume_map = _parse_instrument_volume_args(
            args.instrument_volume
        )
        args.allowed_instrument_ids = parse_allowed_instrument_args(
            args.allowed_instruments
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args


def load_model(
    checkpoint_path: Path, *, device: torch.device
) -> tuple[AudioSemiCRFTransformer, SemiCRFModelConfig, dict[str, Any]]:
    bundle = load_inference_model(checkpoint_path, device=device)
    _print_checkpoint_load_report(bundle.report)
    return bundle.model, bundle.config, dict(bundle.checkpoint_args)


def _print_checkpoint_load_report(report: CheckpointLoadReport) -> None:
    if report.missing_keys:
        print(f"Missing keys while loading checkpoint: {report.missing_keys}")
    if report.unexpected_keys:
        print(f"Skipped checkpoint keys: {report.unexpected_keys}")
    if report.shape_mismatches:
        print(f"Skipped shape-mismatched keys: {report.shape_mismatches}")


def _transcription_options_from_args(args: argparse.Namespace) -> TranscriptionOptions:
    allowed_instrument_ids = getattr(args, "allowed_instrument_ids", None)
    allowed_instruments = (
        None
        if allowed_instrument_ids is None
        else tuple(
            INSTRUMENT_CLASSES[instrument_id]
            for instrument_id in allowed_instrument_ids
        )
    )
    return TranscriptionOptions(
        instrument=getattr(args, "instrument", None),
        allowed_instruments=allowed_instruments,
        window_ms=args.window_ms,
        stride_ms=args.stride_ms,
        window_batch_size=int(args.window_batch_size),
        semi_crf_track_batch_size=args.semi_crf_track_batch_size,
        semi_crf_backend=str(args.semi_crf_backend),
        semi_crf_sparse_decode=bool(args.semi_crf_sparse_decode),
        semi_crf_sparse_topk_per_start=int(args.semi_crf_sparse_topk_per_start),
        semi_crf_sparse_score_threshold=args.semi_crf_sparse_score_threshold,
        semi_crf_sparse_max_span_ms=args.semi_crf_sparse_max_span_ms,
        instrument_pair_infer_topk=int(args.instrument_pair_infer_topk),
        instrument_pair_gate_threshold=float(args.instrument_pair_gate_threshold),
        instrument_pair_max_pairs=int(args.instrument_pair_max_pairs),
        merge_gap_ms=args.merge_gap_ms,
        merge_onset_ms=float(args.merge_onset_ms),
        silence_gate_rms_dbfs=args.silence_gate_rms_dbfs,
        note_bias=float(args.note_bias),
        use_boundary_head=not bool(args.no_boundary_head),
        midi_velocity=int(getattr(args, "velocity", 100)),
        show_progress=not bool(args.disable_tqdm),
    )


def _midi_export_options_from_args(args: argparse.Namespace) -> MidiExportOptions:
    return MidiExportOptions(
        min_midi_note_ms=float(args.min_midi_note_ms),
        max_midi_melodic_instruments=int(args.max_midi_melodic_instruments),
        instrument_volumes=dict(args.instrument_volume_map),
        drum_pitch_aliases=(
            DEFAULT_DRUM_PITCH_ALIASES
            if bool(getattr(args, "collapse_crash_cymbals", True))
            else None
        ),
    )


def resolve_inference_settings(
    config: SemiCRFModelConfig,
    training_args: dict[str, Any],
    args: argparse.Namespace,
) -> InferenceSettings:
    options = _transcription_options_from_args(args)
    return resolve_typed_inference_settings(
        config,
        training_args,
        options,
        allowed_instrument_ids=None,
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


def _instrument_label(instrument_id: int | None) -> str:
    if instrument_id is None:
        return "all-instruments"
    if 0 <= int(instrument_id) < len(INSTRUMENT_CLASSES):
        return INSTRUMENT_CLASSES[int(instrument_id)]
    return str(instrument_id)


def process_file(
    audio_path: Path,
    output_midi_path: Path,
    *,
    model: AudioSemiCRFTransformer,
    forward_model: torch.nn.Module | None = None,
    config: SemiCRFModelConfig,
    instrument_id: int | None,
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
        instrument_filter_id=instrument_id,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        settings=settings,
        velocity=int(args.velocity),
        forward_model=forward_model,
    )
    midi, midi_stats = build_midi(
        notes,
        sample_rate=int(config.sample_rate),
        instrument_id=instrument_id,
        min_midi_note_ms=float(args.min_midi_note_ms),
        max_midi_melodic_instruments=int(args.max_midi_melodic_instruments),
        instrument_volumes=dict(args.instrument_volume_map),
        drum_pitch_aliases=(
            DEFAULT_DRUM_PITCH_ALIASES
            if bool(getattr(args, "collapse_crash_cymbals", True))
            else None
        ),
        return_stats=True,
    )
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_midi_path))
    _print_transcription_summary(
        note_count=len(notes),
        output_midi_path=output_midi_path,
        instrument_id=instrument_id,
        settings=settings,
        inference_stats=stats,
        midi_stats=midi_stats,
    )


def _process_file_with_transcriber(
    audio_path: Path,
    output_midi_path: Path,
    *,
    transcriber: Transcriber,
    options: TranscriptionOptions,
    midi_options: MidiExportOptions,
    instrument_id: int | None,
) -> None:
    result = transcriber.transcribe(
        audio_path,
        options=options,
        midi_options=midi_options,
    )
    output_midi_path.parent.mkdir(parents=True, exist_ok=True)
    output_midi_path.write_bytes(result.midi_bytes)
    _print_transcription_result(
        result,
        output_midi_path=output_midi_path,
        instrument_id=instrument_id,
    )


def _print_transcription_result(
    result: TranscriptionResult,
    *,
    output_midi_path: Path,
    instrument_id: int | None,
) -> None:
    _print_transcription_summary(
        note_count=len(result.notes),
        output_midi_path=output_midi_path,
        instrument_id=instrument_id,
        settings=result.settings,
        inference_stats=result.inference_stats,
        midi_stats=result.midi_stats,
    )


def _print_transcription_summary(
    *,
    note_count: int,
    output_midi_path: Path,
    instrument_id: int | None,
    settings: InferenceSettings,
    inference_stats: Mapping[str, int],
    midi_stats: Mapping[str, int],
) -> None:
    print(
        f"wrote {note_count} notes: {output_midi_path} "
        f"instrument={_instrument_label(instrument_id)} "
        f"window_ms={settings.window_ms} stride_ms={settings.stride_ms} "
        f"semi_crf_backend={settings.semi_crf_backend} "
        f"windows={inference_stats['window_count']} "
        f"decoded_windows={inference_stats['decoded_window_count']} "
        f"skipped_silent_windows={inference_stats['skipped_silent_window_count']} "
        f"selected_pairs={inference_stats['selected_pair_count']} "
        f"decoded_intervals={inference_stats['decoded_interval_count']} "
        f"boundary_no_onset={inference_stats['boundary_no_onset_count']} "
        f"boundary_no_offset={inference_stats['boundary_no_offset_count']} "
        f"midi_instruments_before_remap={midi_stats['midi_instrument_count_before_remap']} "
        f"midi_instruments_after_remap={midi_stats['midi_instrument_count_after_remap']} "
        f"remapped_instruments={midi_stats['remapped_instrument_count']} "
        f"remapped_notes={midi_stats['remapped_note_count']} "
        f"remapped_drum_pitch_notes={midi_stats['remapped_drum_pitch_note_count']}"
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if args.semi_crf_backend == "triton" and device.type != "cuda":
        raise ValueError("--semi-crf-backend triton requires a CUDA device")
    instrument_id = (
        resolve_instrument_id(args.instrument) if args.instrument is not None else None
    )
    checkpoint_path = _ensure_checkpoint(args.checkpoint, model_type=args.type)
    transcriber = Transcriber.from_checkpoint(
        checkpoint_path.resolve(),
        device=device,
        amp=bool(args.amp),
        amp_dtype=args.amp_dtype,
        compile=bool(args.compile),
        compile_mode=str(args.compile_mode),
    )
    _print_checkpoint_load_report(transcriber.load_report)
    transcription_options = _transcription_options_from_args(args)
    midi_options = _midi_export_options_from_args(args)

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
            _process_file_with_transcriber(
                audio_path,
                output_path,
                transcriber=transcriber,
                options=transcription_options,
                midi_options=midi_options,
                instrument_id=instrument_id,
            )
        return

    audio_path = args.audio.resolve()
    output_path = (
        args.output_midi.resolve()
        if args.output_midi is not None
        else audio_path.with_suffix(".mid")
    )
    _process_file_with_transcriber(
        audio_path,
        output_path,
        transcriber=transcriber,
        options=transcription_options,
        midi_options=midi_options,
        instrument_id=instrument_id,
    )


if __name__ == "__main__":
    main()
