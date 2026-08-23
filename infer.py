"""V1-compatible inference entrypoint backed by the packaged implementation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import soundfile as sf
import torch

from instrument_agnostic_amt.cli.infer import (
    DEFAULT_INSTRUMENT_VOLUMES,
    MODEL_CHECKPOINT_FILENAMES,
    _ensure_checkpoint,
    load_model,
    main,
    resolve_inference_settings,
)
from instrument_agnostic_amt.inference.audio import load_audio
from instrument_agnostic_amt.inference.instruments import (
    STEM_INSTRUMENT_CLASSES,
    filter_supported_instrument_class_ids,
    resolve_stem_instrument_class_ids,
)
from instrument_agnostic_amt.inference.midi import build_midi
from instrument_agnostic_amt.inference.windowed import decode_notes


def _load_model_and_settings(
    checkpoint_path: Path,
    *,
    device: torch.device,
    window_ms_override: int | None,
    stride_ms_override: int | None,
    track_batch_size_override: int | None,
    note_bias: float = 0.0,
    semi_crf_backend: str = "torch",
):
    model, config, training_args = load_model(Path(checkpoint_path), device=device)
    args = SimpleNamespace(
        window_ms=window_ms_override,
        stride_ms=stride_ms_override,
        semi_crf_track_batch_size=track_batch_size_override,
        window_batch_size=1,
        merge_gap_ms=None,
        merge_onset_ms=50.0,
        silence_gate_rms_dbfs=-72.0,
        note_bias=note_bias,
        disable_tqdm=False,
        no_boundary_head=False,
        semi_crf_backend=semi_crf_backend,
        semi_crf_sparse_decode=False,
        semi_crf_sparse_topk_per_start=16,
        semi_crf_sparse_score_threshold=None,
        semi_crf_sparse_max_span_ms=None,
        instrument_pair_infer_topk=256,
        instrument_pair_gate_threshold=-3.0,
        instrument_pair_max_pairs=512,
    )
    settings = resolve_inference_settings(config, training_args, args)
    return model, config, settings


def _load_audio(audio_path: Path, *, target_sample_rate: int):
    info = sf.info(str(audio_path))
    waveform = load_audio(Path(audio_path), target_sample_rate=target_sample_rate)
    return waveform, int(info.samplerate), int(info.channels)


def run_inference(
    *,
    model,
    waveform,
    model_config,
    settings,
    device,
    amp_enabled,
    amp_dtype,
    forward_model=None,
    velocity=100,
    merge_gap_ms=None,
    merge_onset_ms=50.0,
    silence_gate_rms_dbfs=-72.0,
    window_batch_size=1,
    disable_tqdm=False,
    allowed_instrument_ids=None,
    **_,
):
    settings = replace(
        settings,
        merge_gap_ms=merge_gap_ms,
        merge_onset_ms=float(merge_onset_ms),
        silence_gate_rms_dbfs=silence_gate_rms_dbfs,
        window_batch_size=int(window_batch_size),
        disable_tqdm=bool(disable_tqdm),
        allowed_instrument_ids=filter_supported_instrument_class_ids(
            allowed_instrument_ids,
            num_model_classes=int(model_config.num_instrument_classes),
        ),
    )
    notes, stats = decode_notes(
        model,
        model_config,
        waveform,
        instrument_filter_id=None,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        settings=settings,
        velocity=velocity,
        forward_model=forward_model,
    )
    return notes, stats, {}


def _build_midi(
    notes,
    *,
    sample_rate: int,
    instrument_volumes=None,
    min_midi_note_ms: float = 5.0,
):
    return build_midi(
        notes,
        sample_rate=sample_rate,
        instrument_id=None,
        min_midi_note_ms=min_midi_note_ms,
        max_midi_melodic_instruments=15,
        instrument_volumes=instrument_volumes,
    )


if __name__ == "__main__":
    main()
