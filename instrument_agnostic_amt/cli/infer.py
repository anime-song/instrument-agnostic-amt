from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pretty_midi
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from tqdm.auto import tqdm

from ..modeling.heads.semi_crf import decode_pitch_intervals
from ..modeling.model import MIN_MIDI_PITCH, AudioSemiCRFTransformer, SemiCRFModelConfig
from ..taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_instrument_class_id_by_name,
    get_program_number_from_class_id,
)

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


@dataclass(frozen=True)
class PredictedNote:
    pitch: int
    start_sample: int
    end_sample: int
    velocity: int
    slot_index: int = 0


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
    parser.add_argument("--semi-crf-track-batch-size", type=int, default=128)
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
) -> tuple[AudioSemiCRFTransformer, SemiCRFModelConfig]:
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
    return model, config


def resolve_instrument_id(name: str) -> int:
    normalized = name.strip()
    try:
        return get_instrument_class_id_by_name(normalized)
    except KeyError:
        available = ", ".join(INSTRUMENT_CLASSES)
        raise ValueError(
            f"Unknown instrument class '{name}'. Available classes: {available}"
        ) from None


def load_audio(audio_path: Path, *, target_sample_rate: int) -> torch.Tensor:
    waveform_np, source_sample_rate = sf.read(
        audio_path, dtype="float32", always_2d=True
    )
    waveform = torch.from_numpy(waveform_np.T.copy())
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    if int(source_sample_rate) != int(target_sample_rate):
        waveform = audio_functional.resample(
            waveform, int(source_sample_rate), int(target_sample_rate)
        )
    return waveform.contiguous()


def decode_notes(
    model: AudioSemiCRFTransformer,
    config: SemiCRFModelConfig,
    waveform: torch.Tensor,
    *,
    condition_instrument_id: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    track_batch_size: int,
    note_bias: float,
    velocity: int,
) -> list[PredictedNote]:
    audio = waveform.unsqueeze(0).to(device)
    valid_audio_frames = torch.tensor(
        [waveform.shape[-1]], dtype=torch.long, device=device
    )
    condition_ids = torch.tensor(
        [condition_instrument_id], dtype=torch.long, device=device
    )
    with (
        torch.no_grad(),
        torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_enabled and device.type == "cuda"),
        ),
    ):
        outputs = model(
            audio,
            condition_instrument_ids=condition_ids,
            valid_audio_frames=valid_audio_frames,
        )
    valid_lengths = outputs["frame_valid_mask"].to(dtype=torch.long).sum(dim=-1)
    decoded = decode_pitch_intervals(
        outputs["interval_query"].float(),
        outputs["interval_key"].float(),
        outputs["interval_diag"].float(),
        valid_lengths,
        length_scaling=str(config.semi_crf_length_scaling),
        length_penalty=float(config.semi_crf_length_penalty),
        note_bias=float(note_bias),
        track_batch_size=int(track_batch_size),
    )[0]

    notes: list[PredictedNote] = []
    num_pitch_slots = int(config.num_pitch_slots)
    min_duration_samples = max(1, int(round(float(config.sample_rate) * 0.005)))
    for track_index, intervals in enumerate(decoded):
        pitch_index = int(track_index) // num_pitch_slots
        slot_index = int(track_index) % num_pitch_slots
        midi_pitch = MIN_MIDI_PITCH + pitch_index
        for begin_frame, end_frame in intervals:
            start_sample = int(begin_frame) * int(config.hop_length)
            end_sample = (int(end_frame) + 1) * int(config.hop_length)
            end_sample = max(end_sample, start_sample + min_duration_samples)
            if start_sample >= int(waveform.shape[-1]):
                continue
            notes.append(
                PredictedNote(
                    pitch=midi_pitch,
                    start_sample=start_sample,
                    end_sample=min(end_sample, int(waveform.shape[-1])),
                    velocity=max(1, min(127, int(velocity))),
                    slot_index=slot_index,
                )
            )
    return sorted(
        notes, key=lambda note: (note.start_sample, note.pitch, note.slot_index)
    )


def build_midi(
    notes: list[PredictedNote],
    *,
    sample_rate: int,
    instrument_id: int,
    min_midi_note_ms: float,
) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(resolution=1920)
    class_name = (
        INSTRUMENT_CLASSES[instrument_id]
        if 0 <= instrument_id < len(INSTRUMENT_CLASSES)
        else "Piano"
    )
    instrument = pretty_midi.Instrument(
        program=get_program_number_from_class_id(instrument_id),
        is_drum=class_name.lower() == "drums",
        name=class_name,
    )
    min_duration_sec = float(min_midi_note_ms) / 1000.0
    for note in notes:
        start = float(note.start_sample) / float(sample_rate)
        end = float(note.end_sample) / float(sample_rate)
        end = max(end, start + min_duration_sec)
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=start,
                end=end,
            )
        )
    midi.instruments.append(instrument)
    return midi


def collect_audio_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    )


def process_file(
    audio_path: Path,
    output_midi_path: Path,
    *,
    model: AudioSemiCRFTransformer,
    config: SemiCRFModelConfig,
    instrument_id: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    args: argparse.Namespace,
) -> None:
    waveform = load_audio(audio_path, target_sample_rate=int(config.sample_rate))
    notes = decode_notes(
        model,
        config,
        waveform,
        condition_instrument_id=instrument_id,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        track_batch_size=int(args.semi_crf_track_batch_size),
        note_bias=float(args.note_bias),
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
    print(f"wrote {len(notes)} notes: {output_midi_path}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    amp_dtype = resolve_amp_dtype(device, args.amp_dtype)
    amp_enabled = bool(args.amp and device.type == "cuda")
    instrument_id = resolve_instrument_id(args.instrument)
    model, config = load_model(args.checkpoint.resolve(), device=device)

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
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        args=args,
    )


if __name__ == "__main__":
    main()
