from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional
from tqdm.auto import tqdm

from .modeling.model import VelocityModelConfig, VelocityPredictionModel


@dataclass(frozen=True, slots=True)
class VelocityNoteQuery:
    start_seconds: float
    end_seconds: float
    pitch: int
    program: int
    is_drum: bool
    stem_index: int


@dataclass(frozen=True, slots=True)
class VelocityPredictions:
    velocities: np.ndarray
    stem_gains_db: np.ndarray | None


@dataclass(frozen=True, slots=True)
class _VelocityWindowPlan:
    index: int
    owner_start_sample: int
    owner_end_sample: int
    audio_start_sample: int
    audio_end_sample: int


def _plan_velocity_windows(
    *,
    total_samples: int,
    window_samples: int,
) -> tuple[_VelocityWindowPlan, ...]:
    window_count = max(1, (total_samples + window_samples - 1) // window_samples)
    plans: list[_VelocityWindowPlan] = []
    for index in range(window_count):
        owner_start = index * window_samples
        owner_end = owner_start + window_samples
        if (
            total_samples - owner_start < window_samples
            and total_samples >= window_samples
        ):
            audio_start = total_samples - window_samples
            audio_end = total_samples
        else:
            audio_start = owner_start
            audio_end = min(owner_end, total_samples)
        plans.append(
            _VelocityWindowPlan(
                index=index,
                owner_start_sample=owner_start,
                owner_end_sample=owner_end,
                audio_start_sample=audio_start,
                audio_end_sample=audio_end,
            )
        )
    return tuple(plans)


def _minimum_model_input_samples(model: object) -> int | None:
    backbone = getattr(model, "backbone", None)
    feature_extractor = getattr(backbone, "feature_extractor", None)
    cqt = getattr(feature_extractor, "cqt", None)
    minimum = getattr(cqt, "minimum_input_samples", None)
    if minimum is None:
        return None
    value = int(minimum)
    if value <= 0:
        raise ValueError("model minimum_input_samples must be positive")
    return value


def _accepts_keyword_argument(callable_object: object, keyword: str) -> bool:
    target = getattr(callable_object, "forward", callable_object)
    try:
        parameters = inspect.signature(target).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == keyword or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def predict_velocity_values(
    *,
    model: VelocityPredictionModel,
    forward_model: Any,
    config: VelocityModelConfig,
    device: torch.device,
    stem_waveforms: Sequence[np.ndarray],
    stem_class_ids: Sequence[int],
    notes: Sequence[VelocityNoteQuery],
    window_seconds: float,
    show_progress: bool,
    include_stem_gain: bool,
    configure_stem_gain: bool,
    select_note_stems_only: bool,
    reject_out_of_range: bool,
) -> VelocityPredictions:
    """Predict aligned MIDI velocities from normalized stereo stem waveforms."""

    if len(stem_waveforms) != len(stem_class_ids) or not stem_waveforms:
        raise ValueError(
            "stem_waveforms and stem_class_ids must be non-empty and aligned"
        )
    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    for waveform in stem_waveforms:
        if waveform.ndim != 2 or int(waveform.shape[0]) != 2:
            raise ValueError("stem waveforms must have shape [2, T]")
        if waveform.dtype != np.float32:
            raise ValueError("stem waveforms must have dtype float32")

    window_samples = int(window_seconds * config.sample_rate)
    if window_samples <= 0:
        raise ValueError("window_seconds is too small for the model sample rate")
    minimum_input_samples = _minimum_model_input_samples(model)
    if minimum_input_samples is not None and window_samples < minimum_input_samples:
        minimum_seconds = minimum_input_samples / float(config.sample_rate)
        raise ValueError(
            "window_seconds is too short for this velocity model: "
            f"requires at least {minimum_input_samples:,} samples "
            f"(about {minimum_seconds:.4f} seconds at "
            f"{int(config.sample_rate):,} Hz)"
        )

    max_samples = max(waveform.shape[1] for waveform in stem_waveforms)
    stem_sample_counts = np.asarray(
        [waveform.shape[1] for waveform in stem_waveforms],
        dtype=np.int64,
    )
    padded_waveforms = [
        np.pad(waveform, ((0, 0), (0, max_samples - waveform.shape[1])))
        if waveform.shape[1] < max_samples
        else waveform
        for waveform in stem_waveforms
    ]
    audio_tensor = (
        torch.from_numpy(np.stack(padded_waveforms, axis=0))
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )
    stem_class_tensor = (
        torch.tensor(stem_class_ids, dtype=torch.long).unsqueeze(0).to(device=device)
    )

    starts = np.asarray([note.start_seconds for note in notes], dtype=np.float64)
    ends = np.asarray([note.end_seconds for note in notes], dtype=np.float64)
    pitches = np.asarray([note.pitch for note in notes], dtype=np.int64)
    programs = np.asarray([note.program for note in notes], dtype=np.int64)
    is_drums = np.asarray([note.is_drum for note in notes], dtype=np.int64)
    stem_indices = np.asarray([note.stem_index for note in notes], dtype=np.int64)
    if np.any(stem_indices < 0) or np.any(stem_indices >= len(stem_waveforms)):
        raise ValueError("note stem_index is outside the supplied stem waveforms")

    onset_samples = np.floor(starts * float(config.sample_rate)).astype(np.int64)
    if reject_out_of_range and (
        np.any(onset_samples < 0)
        or np.any(onset_samples >= stem_sample_counts[stem_indices])
    ):
        raise ValueError("MIDI contains a note onset outside the supplied audio")
    window_plans = _plan_velocity_windows(
        total_samples=max_samples,
        window_samples=window_samples,
    )
    note_window_indices = onset_samples // window_samples

    predicted_velocities = np.full(len(notes), 80, dtype=np.int32)
    predicted_stem_gains: list[np.ndarray] = []
    assignment_counts = np.zeros(len(notes), dtype=np.uint8)
    accepts_valid_audio_frames = _accepts_keyword_argument(
        forward_model,
        "valid_audio_frames",
    )
    with torch.inference_mode():
        for window_plan in tqdm(
            window_plans,
            desc="Predicting velocity",
            disable=not show_progress,
        ):
            indices_in_window = np.flatnonzero(
                (note_window_indices == window_plan.index)
                & (onset_samples >= window_plan.owner_start_sample)
                & (onset_samples < window_plan.owner_end_sample)
            )
            if len(indices_in_window) == 0:
                continue
            assignment_counts[indices_in_window] += 1

            sample_start = window_plan.audio_start_sample
            sample_end = window_plan.audio_end_sample
            audio_start_seconds = float(sample_start) / float(config.sample_rate)
            window_stem_indices = stem_indices[indices_in_window]
            valid_audio_frames = np.clip(
                stem_sample_counts - sample_start,
                0,
                window_samples,
            )
            if select_note_stems_only:
                selected_stem_indices = np.unique(window_stem_indices)
                sub_audio = audio_tensor[
                    :, selected_stem_indices.tolist(), :, sample_start:sample_end
                ]
                local_stem_indices = np.searchsorted(
                    selected_stem_indices,
                    window_stem_indices,
                )
                window_stem_class_tensor = stem_class_tensor[
                    :, selected_stem_indices.tolist()
                ]
                window_valid_audio_frames = valid_audio_frames[selected_stem_indices]
            else:
                sub_audio = audio_tensor[:, :, :, sample_start:sample_end]
                local_stem_indices = window_stem_indices
                window_stem_class_tensor = stem_class_tensor
                window_valid_audio_frames = valid_audio_frames
            if int(sub_audio.shape[-1]) < window_samples:
                sub_audio = functional.pad(
                    sub_audio,
                    (0, window_samples - int(sub_audio.shape[-1])),
                )

            def tensor(values: np.ndarray, *, dtype: torch.dtype) -> torch.Tensor:
                return (
                    torch.from_numpy(values).unsqueeze(0).to(device=device, dtype=dtype)
                )

            forward_kwargs: dict[str, Any] = {
                "note_start_seconds": tensor(
                    (starts[indices_in_window] - audio_start_seconds).astype(
                        np.float32
                    ),
                    dtype=torch.float32,
                ),
                "note_end_seconds": tensor(
                    (ends[indices_in_window] - audio_start_seconds).astype(np.float32),
                    dtype=torch.float32,
                ),
                "note_pitch": tensor(pitches[indices_in_window], dtype=torch.long),
                "note_program": tensor(programs[indices_in_window], dtype=torch.long),
                "note_is_drum": tensor(is_drums[indices_in_window], dtype=torch.long),
                "note_stem_index": tensor(local_stem_indices, dtype=torch.long),
                "stem_class_id": window_stem_class_tensor,
            }
            if accepts_valid_audio_frames and np.any(
                window_valid_audio_frames < int(sub_audio.shape[-1])
            ):
                forward_kwargs["valid_audio_frames"] = tensor(
                    window_valid_audio_frames,
                    dtype=torch.long,
                )
            if configure_stem_gain:
                forward_kwargs["include_stem_gain"] = bool(include_stem_gain)
            outputs = forward_model(sub_audio, **forward_kwargs)
            expected = outputs["velocity_expected"].squeeze(0).cpu().numpy()
            predicted_velocities[indices_in_window] = np.clip(
                np.round(expected), 1, 127
            ).astype(np.int32)
            if "stem_gain_db" in outputs:
                predicted_stem_gains.append(
                    outputs["stem_gain_db"].squeeze(0).cpu().numpy()
                )
            del outputs

    if np.any(assignment_counts > 1):
        raise RuntimeError("A velocity note was assigned to multiple windows")
    if reject_out_of_range and np.any(assignment_counts != 1):
        raise ValueError("Some MIDI notes were not assigned to an audio window")

    mean_stem_gains = (
        np.mean(np.stack(predicted_stem_gains, axis=0), axis=0)
        if predicted_stem_gains
        else None
    )
    return VelocityPredictions(
        velocities=predicted_velocities,
        stem_gains_db=mean_stem_gains,
    )
