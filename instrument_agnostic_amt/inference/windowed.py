from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

import torch
from tqdm.auto import tqdm

from ..runtime import is_amp_supported
from ..modeling.heads.semi_crf import (
    decode_factorized_pair_intervals,
    decode_factorized_pair_intervals_sparse,
)
from ..modeling.model import (
    MIN_MIDI_PITCH,
    NUM_PITCHES,
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)
from .instruments import effective_instrument_ids
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


def _select_pair_candidates(
    pair_gate_logits: torch.Tensor,
    *,
    settings: InferenceSettings,
    instrument_filter_id: int | None,
) -> list[int]:
    if pair_gate_logits.dim() != 2:
        raise ValueError("pair_gate_logits must have shape [I, P]")
    num_instruments, num_pitches = pair_gate_logits.shape
    if int(num_pitches) != NUM_PITCHES:
        raise ValueError(f"expected {NUM_PITCHES} pitches, got {int(num_pitches)}")
    candidate_instrument_ids = effective_instrument_ids(
        num_model_classes=int(num_instruments),
        allowed_instrument_ids=settings.allowed_instrument_ids,
        instrument_filter_id=instrument_filter_id,
    )

    scores = pair_gate_logits.float().reshape(-1)
    finite_mask = torch.isfinite(scores)
    allowed = torch.zeros_like(finite_mask)
    for instrument_id in candidate_instrument_ids:
        start = int(instrument_id) * NUM_PITCHES
        allowed[start : start + NUM_PITCHES] = True
    finite_mask = finite_mask & allowed
    if not bool(torch.any(finite_mask).item()):
        return []

    candidate_ids: set[int] = set()
    threshold = float(settings.instrument_pair_gate_threshold)
    threshold_mask = finite_mask & (scores >= threshold)
    if bool(torch.any(threshold_mask).item()):
        candidate_ids.update(
            int(item) for item in threshold_mask.nonzero().flatten().tolist()
        )

    topk = max(0, int(settings.instrument_pair_infer_topk))
    if topk > 0:
        masked_scores = scores.masked_fill(~finite_mask, float("-inf"))
        finite_count = int(torch.isfinite(masked_scores).sum().item())
        if finite_count > 0:
            top_ids = torch.topk(masked_scores, k=min(topk, finite_count)).indices
            candidate_ids.update(int(item) for item in top_ids.tolist())

    if not candidate_ids:
        return []
    ranked = sorted(
        candidate_ids,
        key=lambda pair_id: (-float(scores[int(pair_id)].item()), int(pair_id)),
    )
    max_pairs = int(settings.instrument_pair_max_pairs)
    if max_pairs > 0:
        ranked = ranked[:max_pairs]
    return ranked


def _rank_instrument_candidates_by_pitch(
    pair_gate_logits: torch.Tensor,
    *,
    settings: InferenceSettings,
    instrument_filter_id: int | None,
) -> list[tuple[int, ...]]:
    if pair_gate_logits.dim() != 2:
        raise ValueError("pair_gate_logits must have shape [I, P]")
    num_instruments, num_pitches = pair_gate_logits.shape
    if int(num_pitches) != NUM_PITCHES:
        raise ValueError(f"expected {NUM_PITCHES} pitches, got {int(num_pitches)}")
    candidate_instrument_ids = effective_instrument_ids(
        num_model_classes=int(num_instruments),
        allowed_instrument_ids=settings.allowed_instrument_ids,
        instrument_filter_id=instrument_filter_id,
    )
    if len(candidate_instrument_ids) == 1:
        return [candidate_instrument_ids for _ in range(NUM_PITCHES)]

    scores = pair_gate_logits.detach().float().cpu()
    ranked_by_pitch: list[tuple[int, ...]] = []
    for pitch_index in range(NUM_PITCHES):
        score_values = {
            instrument_index: float(scores[instrument_index, pitch_index].item())
            for instrument_index in candidate_instrument_ids
        }
        finite_ids = [
            instrument_index
            for instrument_index in candidate_instrument_ids
            if math.isfinite(score_values[instrument_index])
        ]
        ranked = sorted(
            finite_ids,
            key=lambda instrument_index: (
                -score_values[int(instrument_index)],
                int(instrument_index),
            ),
        )
        seen = set(ranked)
        ranked.extend(
            instrument_index
            for instrument_index in candidate_instrument_ids
            if instrument_index not in seen
        )
        ranked_by_pitch.append(tuple(int(item) for item in ranked))
    return ranked_by_pitch


def _merge_instrument_candidates(
    primary: tuple[int, ...], secondary: tuple[int, ...]
) -> tuple[int, ...]:
    merged: list[int] = []
    seen: set[int] = set()
    for candidate in (*primary, *secondary):
        candidate_id = int(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        merged.append(candidate_id)
    return tuple(merged)


def _decode_flat_boundary_features(
    boundary_logits: torch.Tensor,
    entries: list[tuple[int, int, int, int]],
    *,
    num_tracks: int,
) -> list[list[tuple[bool, bool, float, float]]]:
    flags: list[list[tuple[bool, bool, float, float]]] = [
        [] for _ in range(int(num_tracks))
    ]
    if not entries:
        return flags

    boundary_logits = boundary_logits.float()
    presence_logits, offset_logits = boundary_logits.chunk(2, dim=-1)
    boundary_presence = presence_logits > 0.0
    offset_dist = torch.distributions.ContinuousBernoulli(logits=offset_logits)
    offset_values = torch.clamp((offset_dist.mean - 0.005) / 0.99, 0.0, 1.0)

    for row_index, entry in enumerate(entries):
        track_index, _, _, _ = entry
        flags[int(track_index)].append(
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
        velocity: int,
        merge_gap_samples: int,
        merge_onset_samples: int,
    ) -> None:
        self.hop_length = int(hop_length)
        self.total_audio_frames = int(total_audio_frames)
        self.velocity = max(1, min(127, int(velocity)))
        self.merge_gap_samples = int(merge_gap_samples)
        self.merge_onset_samples = int(merge_onset_samples)
        self.notes_by_pair: dict[int, list[PredictedNote]] = defaultdict(list)
        self.last_closed_global_model_frames: dict[int, int] = {}

    def get_forced_start_positions(
        self,
        *,
        pair_ids: list[int],
        window_start_frame: int,
        valid_model_frames: int,
    ) -> list[int]:
        if valid_model_frames <= 0:
            return [0] * len(pair_ids)

        window_model_start = int(
            round(float(window_start_frame) / float(self.hop_length))
        )
        positions: list[int] = []
        for pair_id in pair_ids:
            last_closed_frame = self.last_closed_global_model_frames.get(
                int(pair_id), 0
            )
            positions.append(
                max(
                    0,
                    min(
                        int(last_closed_frame) - window_model_start,
                        int(valid_model_frames) - 1,
                    ),
                )
            )
        return positions

    def consume_window(
        self,
        *,
        pair_ids: list[int],
        intervals_by_track: list[list[tuple[int, int]]],
        boundary_flags_by_track: list[list[tuple[bool, bool, float, float]]] | None,
        instrument_candidates_by_pair: dict[int, tuple[int, ...]] | None,
        window_start_frame: int,
        valid_audio_frames: int,
        valid_model_frames: int,
    ) -> None:
        if len(pair_ids) != len(intervals_by_track):
            raise ValueError("pair_ids length must match decoded interval tracks")
        window_model_start = int(
            round(float(window_start_frame) / float(self.hop_length))
        )

        for track_index, (pair_id, track_intervals) in enumerate(
            zip(pair_ids, intervals_by_track)
        ):
            pair_id = int(pair_id)
            track_notes = self.notes_by_pair[pair_id]
            track_boundary_flags = (
                boundary_flags_by_track[track_index]
                if boundary_flags_by_track is not None
                else []
            )
            instrument_candidates = (
                instrument_candidates_by_pair.get(pair_id, ())
                if instrument_candidates_by_pair is not None
                else ()
            )
            local_last_closed_model_frame: int | None = None

            for interval_index, (begin_frame, end_frame) in enumerate(track_intervals):
                boundary_flag = (
                    track_boundary_flags[interval_index]
                    if interval_index < len(track_boundary_flags)
                    else None
                )
                note = self._build_interval_note(
                    pair_id=pair_id,
                    begin_frame=int(begin_frame),
                    end_frame=int(end_frame),
                    boundary_flag=boundary_flag,
                    instrument_candidates=instrument_candidates,
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
                self.last_closed_global_model_frames[pair_id] = (
                    window_model_start + int(local_last_closed_model_frame)
                )

    def finalize(self) -> list[PredictedNote]:
        for track_notes in self.notes_by_pair.values():
            if track_notes:
                track_notes[-1].has_offset = True

        # V1 decoding appends overlapping-window segments directly.  A sustained
        # note therefore commonly has several intermediate segments whose offset
        # flag is false.  Merge those continuations before filtering on offset;
        # filtering first discards the true onset and keeps only the last window.
        merged_notes = self._merge_nearby_notes(
            [
                note
                for track_notes in self.notes_by_pair.values()
                for note in track_notes
            ]
        )
        return sorted(
            [note for note in merged_notes if note.has_offset],
            key=lambda note: (
                int(note.start_sample),
                int(note.instrument_id),
                int(note.pitch),
                int(note.slot_index),
                int(note.end_sample),
            ),
        )

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
        pair_id: int,
        begin_frame: int,
        end_frame: int,
        boundary_flag: tuple[bool, bool, float, float] | None,
        instrument_candidates: tuple[int, ...],
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

        instrument_id = int(pair_id) // NUM_PITCHES
        pitch_index = int(pair_id) % NUM_PITCHES
        candidates = tuple(int(candidate) for candidate in instrument_candidates)
        if int(instrument_id) not in candidates:
            candidates = (int(instrument_id),) + candidates
        return PredictedNote(
            instrument_id=int(instrument_id),
            pitch=int(MIN_MIDI_PITCH + pitch_index),
            start_sample=int(start_sample),
            end_sample=int(end_sample),
            velocity=int(self.velocity),
            slot_index=0,
            has_onset=bool(has_onset),
            has_offset=bool(has_offset),
            instrument_candidates=candidates,
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
        target.instrument_candidates = _merge_instrument_candidates(
            target.instrument_candidates,
            source.instrument_candidates,
        )
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
                int(note.instrument_id),
                int(note.pitch),
                int(note.slot_index),
                int(note.start_sample),
                int(note.end_sample),
            ),
        )
        merged: list[PredictedNote] = []
        current = replace(ordered_notes[0])
        for note in ordered_notes[1:]:
            same_track = (
                int(note.instrument_id) == int(current.instrument_id)
                and int(note.pitch) == int(current.pitch)
                and int(note.slot_index) == int(current.slot_index)
            )
            can_merge_by_gap = (
                same_track
                and int(note.start_sample)
                <= int(current.end_sample) + self.merge_gap_samples
                and not note.has_onset
            )
            can_merge_by_onset = (
                same_track
                and abs(int(note.start_sample) - int(current.start_sample))
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
                int(note.start_sample),
                int(note.instrument_id),
                int(note.pitch),
                int(note.slot_index),
                int(note.end_sample),
            ),
        )


@torch.inference_mode()
def decode_notes(
    model: AudioSemiCRFTransformer,
    config: SemiCRFModelConfig,
    waveform: torch.Tensor,
    *,
    instrument_filter_id: int | None,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    settings: InferenceSettings,
    velocity: int,
) -> tuple[list[PredictedNote], dict[str, int]]:
    if config.semi_crf_version == "v1":
        from .v1_windowed import decode_v1_notes

        return decode_v1_notes(
            model,
            config,
            waveform,
            instrument_filter_id=instrument_filter_id,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            settings=settings,
            velocity=velocity,
        )
    if waveform.dim() != 2:
        raise ValueError("waveform must have shape [channels, audio_frames]")
    expected_channels = int(getattr(config, "input_audio_channels", 2))
    if int(waveform.shape[0]) != expected_channels:
        if expected_channels == 1 and int(waveform.shape[0]) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        elif expected_channels == 2 and int(waveform.shape[0]) == 1:
            waveform = waveform.expand(2, -1).contiguous()
        else:
            raise ValueError(
                f"waveform has {int(waveform.shape[0])} channels, "
                f"expected {expected_channels}"
            )

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
        velocity=int(velocity),
        merge_gap_samples=merge_gap_samples,
        merge_onset_samples=merge_onset_samples,
    )
    silence_gate_linear = _silence_gate_rms_linear(settings.silence_gate_rms_dbfs)
    skipped_silent_window_count = 0
    decoded_window_count = 0
    selected_pair_count = 0
    decoded_interval_count = 0
    boundary_interval_count = 0
    boundary_no_onset_count = 0
    boundary_no_offset_count = 0
    use_boundary_head = bool(
        settings.use_boundary_head and model.supports_interval_boundaries()
    )
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

        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=bool(amp_enabled and is_amp_supported(device)),
        ):
            outputs = model(
                batch_waveform,
                valid_audio_frames=valid_audio_frames_tensor,
                include_aux_outputs=False,
            )

        interval_features = outputs.get("interval_features")
        pair_gate_logits = outputs.get("pair_gate_logits")
        pitch_interval_query = outputs.get("pitch_interval_query")
        pitch_interval_key = outputs.get("pitch_interval_key")
        pitch_interval_diag = outputs.get("pitch_interval_diag")
        instrument_interval_query = outputs.get("instrument_interval_query")
        instrument_interval_key = outputs.get("instrument_interval_key")
        instrument_interval_diag = outputs.get("instrument_interval_diag")
        frame_valid_mask = outputs.get("frame_valid_mask")
        if (
            interval_features is None
            or pair_gate_logits is None
            or pitch_interval_query is None
            or pitch_interval_key is None
            or pitch_interval_diag is None
            or instrument_interval_query is None
            or instrument_interval_key is None
            or instrument_interval_diag is None
            or frame_valid_mask is None
        ):
            raise ValueError(
                "Factorized V2 inference requires interval projections and pair gate logits"
            )
        valid_lengths = frame_valid_mask.to(dtype=torch.long).sum(dim=-1)

        for sample_index, (start_frame, valid_frames) in enumerate(
            zip(active_batch_starts, active_valid_audio_frames)
        ):
            sample_valid_length = int(valid_lengths[sample_index].item())
            if sample_valid_length <= 0:
                continue

            pair_ids = _select_pair_candidates(
                pair_gate_logits[sample_index],
                settings=settings,
                instrument_filter_id=instrument_filter_id,
            )
            if not pair_ids:
                continue
            selected_pair_count += len(pair_ids)
            candidate_orders_by_pitch = _rank_instrument_candidates_by_pitch(
                pair_gate_logits[sample_index],
                settings=settings,
                instrument_filter_id=instrument_filter_id,
            )
            instrument_candidates_by_pair = {
                int(pair_id): candidate_orders_by_pitch[int(pair_id) % NUM_PITCHES]
                for pair_id in pair_ids
            }

            forced_start_pos = note_stitcher.get_forced_start_positions(
                pair_ids=pair_ids,
                window_start_frame=int(start_frame),
                valid_model_frames=sample_valid_length,
            )
            selected_pairs = model.build_selected_pair_indices(
                [pair_ids],
            )

            if settings.semi_crf_sparse_decode:
                decoded_intervals = decode_factorized_pair_intervals_sparse(
                    pitch_interval_query[sample_index : sample_index + 1],
                    pitch_interval_key[sample_index : sample_index + 1],
                    pitch_interval_diag[sample_index : sample_index + 1],
                    instrument_interval_query,
                    instrument_interval_key,
                    instrument_interval_diag,
                    selected_pairs.batch_indices,
                    selected_pairs.instrument_indices,
                    selected_pairs.pitch_indices,
                    valid_lengths[sample_index : sample_index + 1],
                    length_scaling=str(config.semi_crf_length_scaling),
                    length_penalty=float(config.semi_crf_length_penalty),
                    note_bias=float(settings.note_bias),
                    track_batch_size=int(settings.track_batch_size),
                    forced_start_pos=forced_start_pos,
                    sparse_topk_per_start=int(settings.semi_crf_sparse_topk_per_start),
                    sparse_score_threshold=settings.semi_crf_sparse_score_threshold,
                    sparse_max_span_frames=settings.semi_crf_sparse_max_span_frames,
                )
            else:
                decoded_intervals = decode_factorized_pair_intervals(
                    pitch_interval_query[sample_index : sample_index + 1],
                    pitch_interval_key[sample_index : sample_index + 1],
                    pitch_interval_diag[sample_index : sample_index + 1],
                    instrument_interval_query,
                    instrument_interval_key,
                    instrument_interval_diag,
                    selected_pairs.batch_indices,
                    selected_pairs.instrument_indices,
                    selected_pairs.pitch_indices,
                    valid_lengths[sample_index : sample_index + 1],
                    length_scaling=str(config.semi_crf_length_scaling),
                    length_penalty=float(config.semi_crf_length_penalty),
                    note_bias=float(settings.note_bias),
                    track_batch_size=int(settings.track_batch_size),
                    forced_start_pos=forced_start_pos,
                )

            decoded_interval_count += sum(len(track) for track in decoded_intervals)

            boundary_flags_by_track = None
            if use_boundary_head:
                boundary_logits, boundary_entries = (
                    model.predict_flat_interval_boundaries(
                        interval_features[sample_index : sample_index + 1].float(),
                        selected_pairs,
                        decoded_intervals,
                    )
                )
                boundary_interval_count += len(boundary_entries)
                boundary_flags_by_track = _decode_flat_boundary_features(
                    boundary_logits,
                    boundary_entries,
                    num_tracks=len(pair_ids),
                )
                for track_flags in boundary_flags_by_track:
                    for has_onset, has_offset, _, _ in track_flags:
                        if not has_onset:
                            boundary_no_onset_count += 1
                        if not has_offset:
                            boundary_no_offset_count += 1

            note_stitcher.consume_window(
                pair_ids=pair_ids,
                intervals_by_track=decoded_intervals,
                boundary_flags_by_track=boundary_flags_by_track,
                instrument_candidates_by_pair=instrument_candidates_by_pair,
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
        "selected_pair_count": int(selected_pair_count),
        "decoded_interval_count": int(decoded_interval_count),
        "boundary_interval_count": int(boundary_interval_count),
        "boundary_no_onset_count": int(boundary_no_onset_count),
        "boundary_no_offset_count": int(boundary_no_offset_count),
        "note_count": len(notes),
    }
