from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

import torch
from tqdm.auto import tqdm

from ..modeling.heads.semi_crf import decode_pitch_intervals
from ..modeling.model import MIN_MIDI_PITCH, AudioSemiCRFTransformer, SemiCRFModelConfig
from .types import InferenceSettings, PredictedNote


def _iter_batches(values: list[int], batch_size: int) -> Iterable[list[int]]:
    chunk_size = max(1, int(batch_size))
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


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
    end_frame = min(
        int(waveform.shape[-1]), int(start_frame) + int(window_audio_frames)
    )
    window = waveform[:, int(start_frame) : end_frame]
    valid_audio_frames = int(window.shape[-1])
    if valid_audio_frames < int(window_audio_frames):
        padded = torch.zeros(
            (int(waveform.shape[0]), int(window_audio_frames)),
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
    gate_input = batch_waveform.mean(dim=1, keepdim=True)
    rms = gate_input.float().square().mean(dim=-1).sqrt()
    return torch.all(rms < float(silence_gate_rms_linear), dim=1)


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

    boundary_logits = boundary_logits.float()
    presence_logits, offset_logits = boundary_logits.chunk(2, dim=-1)
    boundary_presence = presence_logits > 0.0
    offset_dist = torch.distributions.ContinuousBernoulli(logits=offset_logits)
    offset_values = torch.clamp((offset_dist.mean - 0.005) / 0.99, 0.0, 1.0)

    for row_index, entry in enumerate(entries):
        batch_index, track_index, _, _, _ = entry
        flags[int(batch_index)][int(track_index)].append(
            (
                bool(boundary_presence[row_index, 0].item()),
                bool(boundary_presence[row_index, 1].item()),
                float(offset_values[row_index, 0].item()),
                float(offset_values[row_index, 1].item()),
            )
        )
    return flags


class WindowNoteStitcher:
    def __init__(
        self,
        *,
        hop_length: int,
        total_audio_frames: int,
        num_pitch_slots: int,
        velocity: int,
        merge_gap_samples: int,
        merge_onset_samples: int,
    ) -> None:
        self.hop_length = int(hop_length)
        self.total_audio_frames = int(total_audio_frames)
        self.num_pitch_slots = max(1, int(num_pitch_slots))
        self.velocity = max(1, min(127, int(velocity)))
        self.merge_gap_samples = int(merge_gap_samples)
        self.merge_onset_samples = int(merge_onset_samples)
        self.notes_by_track: dict[int, list[PredictedNote]] = defaultdict(list)
        self.last_closed_global_model_frames: list[int] | None = None

    def get_forced_start_positions(
        self,
        *,
        window_start_frame: int,
        num_tracks: int,
        valid_model_frames: int,
    ) -> list[int]:
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
            for last_closed_frame in self.last_closed_global_model_frames
        ]

    def consume_window(
        self,
        *,
        intervals_by_track: list[list[tuple[int, int]]],
        boundary_flags_by_track: list[list[tuple[bool, bool, float, float]]] | None,
        window_start_frame: int,
        valid_audio_frames: int,
        valid_model_frames: int,
    ) -> None:
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
            local_last_closed_model_frame: int | None = None

            for interval_index, (begin_frame, end_frame) in enumerate(track_intervals):
                boundary_flag = (
                    track_boundary_flags[interval_index]
                    if interval_index < len(track_boundary_flags)
                    else None
                )
                note = self._build_interval_note(
                    track_index=int(track_index),
                    begin_frame=int(begin_frame),
                    end_frame=int(end_frame),
                    boundary_flag=boundary_flag,
                    window_start_frame=int(window_start_frame),
                    valid_audio_frames=int(valid_audio_frames),
                    valid_model_frames=int(valid_model_frames),
                )
                if note is None:
                    continue

                if track_notes and int(note.start_sample) < int(
                    track_notes[-1].end_sample
                ):
                    if note.has_onset:
                        track_notes[-1] = note
                    else:
                        self._merge_note_segments(
                            track_notes[-1], note, overwrite_offset=True
                        )
                    if note.has_offset:
                        local_last_closed_model_frame = int(end_frame)
                    continue

                if note.has_onset:
                    track_notes.append(note)
                if note.has_offset:
                    local_last_closed_model_frame = int(end_frame)

            if local_last_closed_model_frame is not None:
                self.last_closed_global_model_frames[track_index] = (
                    window_model_start + int(local_last_closed_model_frame)
                )

    def finalize(self) -> list[PredictedNote]:
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
                note.start_sample,
                note.pitch,
                note.slot_index,
                note.end_sample,
            ),
        )
        return self._merge_nearby_notes(stitched_notes)

    def _ensure_track_state(self, num_tracks: int) -> None:
        if self.last_closed_global_model_frames is None:
            self.last_closed_global_model_frames = [0] * int(num_tracks)
            return
        if len(self.last_closed_global_model_frames) != int(num_tracks):
            raise ValueError(
                "num_tracks changed during note stitching: "
                f"{len(self.last_closed_global_model_frames)} -> {num_tracks}"
            )

    def _track_to_pitch_slot(self, track_index: int) -> tuple[int, int]:
        slot_count = max(1, int(self.num_pitch_slots))
        return int(track_index) // slot_count, int(track_index) % slot_count

    @staticmethod
    def _resolve_interval_boundary_flags(
        *,
        begin_frame: int,
        end_frame: int,
        valid_model_frames: int,
        boundary_flag: tuple[bool, bool, float, float] | None,
    ) -> tuple[bool, bool, float, float]:
        if boundary_flag is not None:
            return boundary_flag

        last_valid_frame = max(0, int(valid_model_frames) - 1)
        return (
            bool(int(begin_frame) > 0),
            bool(int(end_frame) < last_valid_frame),
            0.0,
            1.0,
        )

    def _build_interval_note(
        self,
        *,
        track_index: int,
        begin_frame: int,
        end_frame: int,
        boundary_flag: tuple[bool, bool, float, float] | None,
        window_start_frame: int,
        valid_audio_frames: int,
        valid_model_frames: int,
    ) -> PredictedNote | None:
        has_onset, has_offset, onset_off, offset_off = (
            self._resolve_interval_boundary_flags(
                begin_frame=int(begin_frame),
                end_frame=int(end_frame),
                valid_model_frames=int(valid_model_frames),
                boundary_flag=boundary_flag,
            )
        )
        if (
            boundary_flag is None
            and int(window_start_frame) == 0
            and int(begin_frame) == 0
        ):
            has_onset = True
        start_sample = int(window_start_frame) + int(
            round((float(begin_frame) + float(onset_off)) * float(self.hop_length))
        )
        end_sample = int(window_start_frame) + min(
            int(valid_audio_frames),
            int(round((float(end_frame) + float(offset_off)) * float(self.hop_length))),
        )
        start_sample = max(0, min(start_sample, self.total_audio_frames))
        end_sample = max(start_sample + 1, min(end_sample, self.total_audio_frames))
        window_valid_end = min(
            self.total_audio_frames,
            int(window_start_frame) + int(valid_audio_frames),
        )
        if start_sample >= window_valid_end:
            return None

        pitch_index, slot_index = self._track_to_pitch_slot(int(track_index))
        return PredictedNote(
            pitch=int(MIN_MIDI_PITCH + pitch_index),
            start_sample=int(start_sample),
            end_sample=int(end_sample),
            velocity=int(self.velocity),
            slot_index=int(slot_index),
            has_onset=bool(has_onset),
            has_offset=bool(has_offset),
        )

    @staticmethod
    def _merge_note_segments(
        target: PredictedNote,
        source: PredictedNote,
        *,
        overwrite_offset: bool,
    ) -> None:
        target.end_sample = max(int(target.end_sample), int(source.end_sample))
        target.velocity = max(int(target.velocity), int(source.velocity))
        target.has_onset = bool(target.has_onset or source.has_onset)
        if overwrite_offset:
            target.has_offset = bool(source.has_offset)
        else:
            target.has_offset = bool(target.has_offset or source.has_offset)

    def _merge_nearby_notes(self, notes: list[PredictedNote]) -> list[PredictedNote]:
        if not notes:
            return []

        ordered_notes = sorted(
            notes,
            key=lambda note: (
                note.pitch,
                note.slot_index,
                note.start_sample,
                note.end_sample,
            ),
        )
        merged: list[PredictedNote] = []
        current = replace(ordered_notes[0])
        for note in ordered_notes[1:]:
            can_merge_by_gap = (
                note.pitch == current.pitch
                and note.slot_index == current.slot_index
                and note.start_sample <= current.end_sample + self.merge_gap_samples
                and not note.has_onset
            )
            can_merge_by_onset = (
                note.pitch == current.pitch
                and note.slot_index == current.slot_index
                and abs(note.start_sample - current.start_sample)
                <= self.merge_onset_samples
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
                note.start_sample,
                note.pitch,
                note.slot_index,
                note.end_sample,
            ),
        )


@torch.inference_mode()
def decode_notes(
    model: AudioSemiCRFTransformer,
    config: SemiCRFModelConfig,
    waveform: torch.Tensor,
    *,
    condition_instrument_id: int,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    settings: InferenceSettings,
    velocity: int,
) -> tuple[list[PredictedNote], dict[str, int]]:
    if waveform.dim() != 2 or int(waveform.shape[0]) != 2:
        raise ValueError("waveform must have shape [2, audio_frames]")

    sample_rate = int(config.sample_rate)
    total_audio_frames = int(waveform.shape[-1])
    if total_audio_frames <= 0:
        raise ValueError("audio is empty")
    window_audio_frames = int(round(float(settings.window_ms) * sample_rate / 1000.0))
    stride_audio_frames = int(round(float(settings.stride_ms) * sample_rate / 1000.0))
    if window_audio_frames < int(config.n_fft):
        raise ValueError(
            f"window_ms={settings.window_ms} is too short for n_fft={config.n_fft}"
        )
    if stride_audio_frames <= 0:
        raise ValueError("stride_ms must be positive")

    window_starts = _build_window_starts(
        total_audio_frames=total_audio_frames,
        window_audio_frames=window_audio_frames,
        stride_audio_frames=stride_audio_frames,
    )
    merge_gap_samples = (
        int(config.hop_length)
        if settings.merge_gap_ms is None
        else max(0, int(round(float(settings.merge_gap_ms) * sample_rate / 1000.0)))
    )
    merge_onset_samples = max(
        0, int(round(float(settings.merge_onset_ms) * sample_rate / 1000.0))
    )
    note_stitcher = WindowNoteStitcher(
        hop_length=int(config.hop_length),
        total_audio_frames=total_audio_frames,
        num_pitch_slots=int(config.num_pitch_slots),
        velocity=int(velocity),
        merge_gap_samples=merge_gap_samples,
        merge_onset_samples=merge_onset_samples,
    )
    silence_gate_linear = _silence_gate_rms_linear(settings.silence_gate_rms_dbfs)
    skipped_silent_window_count = 0
    decoded_window_count = 0
    use_boundary_head = bool(model.supports_interval_boundaries())
    progress = tqdm(
        _iter_batches(window_starts, int(settings.window_batch_size)),
        total=math.ceil(len(window_starts) / max(1, int(settings.window_batch_size))),
        desc="infer",
        dynamic_ncols=True,
        disable=bool(settings.disable_tqdm),
    )

    for batch_starts in progress:
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
        condition_ids = torch.full(
            (len(active_indices),),
            int(condition_instrument_id),
            dtype=torch.long,
            device=device,
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_enabled and device.type == "cuda"),
        ):
            outputs = model(
                batch_waveform,
                condition_instrument_ids=condition_ids,
                valid_audio_frames=valid_audio_frames_tensor,
            )

        valid_lengths = outputs["frame_valid_mask"].to(dtype=torch.long).sum(dim=-1)
        num_tracks = int(outputs["interval_query"].shape[2])
        boundary_features = outputs.get("interval_features")
        if boundary_features is None:
            boundary_features = outputs["pitch_query_features"]

        for sample_index, (start_frame, valid_frames) in enumerate(
            zip(active_batch_starts, active_valid_audio_frames)
        ):
            sample_valid_length = int(valid_lengths[sample_index].item())
            if sample_valid_length <= 0:
                continue

            forced_start_pos = note_stitcher.get_forced_start_positions(
                window_start_frame=int(start_frame),
                num_tracks=num_tracks,
                valid_model_frames=sample_valid_length,
            )
            decoded_intervals = decode_pitch_intervals(
                outputs["interval_query"][sample_index : sample_index + 1],
                outputs["interval_key"][sample_index : sample_index + 1],
                outputs["interval_diag"][sample_index : sample_index + 1],
                valid_lengths[sample_index : sample_index + 1],
                length_scaling=str(config.semi_crf_length_scaling),
                length_penalty=float(config.semi_crf_length_penalty),
                note_bias=float(settings.note_bias),
                track_batch_size=int(settings.track_batch_size),
                forced_start_pos=[forced_start_pos],
            )[0]

            boundary_flags_by_track = None
            if use_boundary_head:
                boundary_logits, boundary_entries = model.predict_interval_boundaries(
                    boundary_features[sample_index : sample_index + 1].float(),
                    [decoded_intervals],
                )
                boundary_flags_by_track = _decode_boundary_features(
                    boundary_logits,
                    boundary_entries,
                    batch_size=1,
                    num_tracks=num_tracks,
                )[0]

            note_stitcher.consume_window(
                intervals_by_track=decoded_intervals,
                boundary_flags_by_track=boundary_flags_by_track,
                window_start_frame=int(start_frame),
                valid_audio_frames=int(valid_frames),
                valid_model_frames=sample_valid_length,
            )

        del outputs, batch_waveform

    notes = note_stitcher.finalize()
    return notes, {
        "window_count": len(window_starts),
        "decoded_window_count": int(decoded_window_count),
        "skipped_silent_window_count": int(skipped_silent_window_count),
        "window_audio_frames": int(window_audio_frames),
        "stride_audio_frames": int(stride_audio_frames),
    }
