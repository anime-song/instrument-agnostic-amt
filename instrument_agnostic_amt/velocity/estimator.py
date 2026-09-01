from __future__ import annotations

import gc
import io
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import TracebackType
from typing import Literal

import mido
import torch
import torchaudio.functional as audio_functional

from ..inference.audio import load_audio
from ..runtime import empty_device_cache, maybe_compile_forward, resolve_device
from ..transcription import DecodedAudio
from .inference import VelocityNoteQuery, predict_velocity_values
from .midi import (
    apply_velocities_to_template,
    note_records_from_mido,
)
from .modeling.checkpoints import (
    VelocityCheckpointLoadReport,
    load_velocity_checkpoint,
)
from .modeling.model import VelocityModelConfig, VelocityPredictionModel
from .training.dataset import STEM_CLASS_BY_NAME, UNKNOWN_STEM_CLASS


class VelocityEstimatorBusyError(RuntimeError):
    """Raised when one VelocityEstimator is called concurrently."""


@dataclass(frozen=True, slots=True)
class VelocityOptions:
    """Options for one note-velocity estimation call."""

    window_seconds: float = 8.0
    loudness_controls: Literal["velocity_only", "preserve", "strip"] = (
        "velocity_only"
    )
    show_progress: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if self.loudness_controls not in {"velocity_only", "preserve", "strip"}:
            raise ValueError(
                "loudness_controls must be one of: velocity_only, preserve, strip"
            )


@dataclass(frozen=True, slots=True)
class VelocityModelInfo:
    """Checkpoint identity and effective runtime settings."""

    checkpoint_path: Path
    sample_rate: int
    device: str
    compile_enabled: bool
    compile_mode: str | None


@dataclass(frozen=True, slots=True)
class VelocityResult:
    """In-memory MIDI with estimated note velocities."""

    midi_bytes: bytes
    velocity_applied: bool
    note_count: int
    model_info: VelocityModelInfo


class VelocityEstimator:
    """Long-lived note-velocity estimator for one checkpoint."""

    def __init__(
        self,
        *,
        model: VelocityPredictionModel,
        config: VelocityModelConfig,
        forward_model: torch.nn.Module,
        model_info: VelocityModelInfo,
        load_report: VelocityCheckpointLoadReport,
    ) -> None:
        self._model = model
        self._config = config
        self._forward_model = forward_model
        self._model_info = model_info
        self._load_report = load_report
        self._call_lock = Lock()
        self._closed = False

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device | None = "auto",
        compile: bool = False,
        compile_mode: str = "default",
    ) -> VelocityEstimator:
        """Load one trusted local velocity checkpoint without downloading files."""

        resolved_checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
        if not resolved_checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint is not a file: {resolved_checkpoint}")
        target_device = resolve_device(device)
        model, config, report = load_velocity_checkpoint(
            resolved_checkpoint,
            device=target_device,
        )
        forward_model = maybe_compile_forward(
            model,
            enabled=bool(compile),
            mode=str(compile_mode),
        )
        model_info = VelocityModelInfo(
            checkpoint_path=resolved_checkpoint,
            sample_rate=int(config.sample_rate),
            device=str(target_device),
            compile_enabled=bool(compile),
            compile_mode=str(compile_mode) if compile else None,
        )
        return cls(
            model=model,
            config=config,
            forward_model=forward_model,
            model_info=model_info,
            load_report=report,
        )

    @property
    def model_info(self) -> VelocityModelInfo:
        return self._model_info

    @property
    def load_report(self) -> VelocityCheckpointLoadReport:
        return self._load_report

    @property
    def closed(self) -> bool:
        """Whether this estimator has released its model."""

        return self._closed

    def __enter__(self) -> VelocityEstimator:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Wait for an active call and release the loaded model references."""

        with self._call_lock:
            if self._closed:
                return
            self._model = None
            self._forward_model = None
            self._closed = True
            gc.collect()
            empty_device_cache(torch.device(self._model_info.device))

    def estimate(
        self,
        *,
        midi: str | Path | bytes,
        audio: str | Path | DecodedAudio,
        stem_kind: Literal[
            "bass",
            "drums",
            "guitar",
            "other",
            "piano",
            "vocals",
            "unknown",
        ],
        options: VelocityOptions | None = None,
    ) -> VelocityResult:
        """Estimate velocities for one logical stem without inferring its producer."""

        if not self._call_lock.acquire(blocking=False):
            raise VelocityEstimatorBusyError(
                "This VelocityEstimator is already processing another call"
            )
        try:
            if self._closed:
                raise RuntimeError("This VelocityEstimator is closed")
            return self._estimate(
                midi=midi,
                audio=audio,
                stem_kind=stem_kind,
                options=options,
            )
        finally:
            self._call_lock.release()

    def _estimate(
        self,
        *,
        midi: str | Path | bytes,
        audio: str | Path | DecodedAudio,
        stem_kind: str,
        options: VelocityOptions | None,
    ) -> VelocityResult:
        if stem_kind not in {*STEM_CLASS_BY_NAME, "unknown"}:
            available = ", ".join((*STEM_CLASS_BY_NAME, "unknown"))
            raise ValueError(f"stem_kind must be one of: {available}")
        call_options = options or VelocityOptions()
        source_bytes = self._read_midi_bytes(midi)
        template_midi = mido.MidiFile(file=io.BytesIO(source_bytes), clip=True)
        note_records = note_records_from_mido(
            template_midi,
            stem_name=stem_kind,
            stem_index=0,
        )
        if not note_records:
            return VelocityResult(
                midi_bytes=source_bytes,
                velocity_applied=False,
                note_count=0,
                model_info=self._model_info,
            )
        waveform = self._prepare_audio(audio)
        stem_class_id = STEM_CLASS_BY_NAME.get(stem_kind, UNKNOWN_STEM_CLASS)
        predictions = predict_velocity_values(
            model=self._model,
            forward_model=self._forward_model,
            config=self._config,
            device=torch.device(self._model_info.device),
            stem_waveforms=(waveform,),
            stem_class_ids=(stem_class_id,),
            notes=tuple(
                VelocityNoteQuery(
                    start_seconds=float(record.start_seconds),
                    end_seconds=float(record.end_seconds),
                    pitch=int(record.pitch),
                    program=int(record.program),
                    is_drum=bool(record.is_drum),
                    stem_index=0,
                )
                for record in note_records
            ),
            window_seconds=float(call_options.window_seconds),
            show_progress=bool(call_options.show_progress),
            include_stem_gain=False,
            configure_stem_gain=(
                isinstance(self._model, VelocityPredictionModel)
                and self._forward_model is self._model
            ),
            select_note_stems_only=True,
            reject_out_of_range=True,
        )
        if len(predictions.velocities) != len(note_records):
            raise ValueError("Velocity prediction count does not match MIDI notes")
        for record, velocity in zip(note_records, predictions.velocities):
            record.velocity = int(velocity)
        apply_velocities_to_template(
            template_midi=template_midi,
            note_records=note_records,
            loudness_controls=call_options.loudness_controls,
        )
        output = io.BytesIO()
        template_midi.save(file=output)
        return VelocityResult(
            midi_bytes=output.getvalue(),
            velocity_applied=True,
            note_count=len(note_records),
            model_info=self._model_info,
        )

    def _prepare_audio(self, audio: str | Path | DecodedAudio):
        if isinstance(audio, DecodedAudio):
            waveform = audio.samples.detach()
            if int(waveform.shape[0]) == 1:
                waveform = waveform.repeat(2, 1)
            if int(audio.sample_rate) != int(self._config.sample_rate):
                waveform = audio_functional.resample(
                    waveform,
                    int(audio.sample_rate),
                    int(self._config.sample_rate),
                )
        else:
            waveform = load_audio(
                Path(audio),
                target_sample_rate=int(self._config.sample_rate),
            )
        return waveform.contiguous().numpy()

    @staticmethod
    def _read_midi_bytes(midi: str | Path | bytes) -> bytes:
        if isinstance(midi, bytes):
            if not midi:
                raise ValueError("MIDI bytes must not be empty")
            return midi
        midi_path = Path(midi).expanduser().resolve(strict=True)
        if not midi_path.is_file():
            raise FileNotFoundError(f"MIDI is not a file: {midi_path}")
        return midi_path.read_bytes()
