from __future__ import annotations

import math
from collections.abc import Callable

import torch
from tqdm.auto import tqdm

from ..modeling.heads.semi_crf import decode_pitch_intervals
from ..modeling.model import (
    MIN_MIDI_PITCH,
    NUM_PITCHES,
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)
from ..runtime import is_amp_supported
from .instruments import effective_instrument_ids
from .types import InferenceSettings, PredictedNote
from .windowed import (
    WindowNoteStitcher,
    _build_window_starts,
    _compute_silent_window_mask,
    _iter_batches,
    _silence_gate_rms_linear,
    _slice_window,
)


def _decode_boundary_map(
    logits: torch.Tensor,
    entries: list[tuple[int, int, int, int, int]],
) -> dict[tuple[int, int, int], tuple[bool, bool, float, float]]:
    if not entries:
        return {}
    presence_logits, offset_logits = logits.float().chunk(2, dim=-1)
    presence = presence_logits > 0.0
    offset_dist = torch.distributions.ContinuousBernoulli(logits=offset_logits)
    offsets = torch.clamp((offset_dist.mean - 0.005) / 0.99, 0.0, 1.0)
    decoded_rows = torch.cat(
        (presence.to(dtype=offsets.dtype), offsets),
        dim=-1,
    ).detach().cpu().tolist()
    return {
        (entry[0], entry[1], entry[2]): (
            bool(decoded_rows[index][0]),
            bool(decoded_rows[index][1]),
            float(decoded_rows[index][2]),
            float(decoded_rows[index][3]),
        )
        for index, entry in enumerate(entries)
    }


def _decode_instrument_map(
    logits: torch.Tensor,
    entries: list[tuple[int, int, int, int, int]],
    *,
    probability_mode: str,
    allowed_instrument_ids: tuple[int, ...],
) -> dict[tuple[int, int, int], tuple[int, tuple[int, ...]]]:
    if not entries:
        return {}
    selected_logits = logits.float()[..., list(allowed_instrument_ids)]
    probabilities = (
        torch.softmax(selected_logits, dim=-1)
        if probability_mode == "softmax"
        else torch.sigmoid(selected_logits)
    )
    order = probabilities.argsort(dim=-1, descending=True)
    order_rows = order.detach().cpu().tolist()
    decoded = {}
    for index, entry in enumerate(entries):
        candidates = tuple(
            int(allowed_instrument_ids[int(value)])
            for value in order_rows[index]
        )
        decoded[(entry[0], entry[1], entry[2])] = (candidates[0], candidates)
    return decoded


def _decode_frame_instrument_map(
    frame_logits: torch.Tensor,
    decoded_batch: list[list[list[tuple[int, int]]]],
    *,
    num_pitch_slots: int,
    allowed_instrument_ids: tuple[int, ...],
    excluded_keys: set[tuple[int, int, int]],
) -> dict[tuple[int, int, int], tuple[int, tuple[int, ...]]]:
    keys: list[tuple[int, int, int]] = []
    orders: list[torch.Tensor] = []
    for batch_index, sample_intervals in enumerate(decoded_batch):
        for track, intervals in enumerate(sample_intervals):
            pitch_index = track // int(num_pitch_slots)
            for interval_index, (begin, end) in enumerate(intervals):
                key = (batch_index, track, interval_index)
                if key in excluded_keys:
                    continue
                logits = frame_logits[
                    batch_index, begin : end + 1, pitch_index
                ].float().mean(dim=0)
                selected_logits = logits[list(allowed_instrument_ids)]
                keys.append(key)
                orders.append(selected_logits.argsort(descending=True))

    if not orders:
        return {}
    order_rows = torch.stack(orders).detach().cpu().tolist()
    decoded = {}
    for key, order_row in zip(keys, order_rows):
        candidates = tuple(
            int(allowed_instrument_ids[int(value)]) for value in order_row
        )
        decoded[key] = (candidates[0], candidates)
    return decoded


@torch.inference_mode()
def decode_v1_notes(
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
    forward_model: Callable[..., dict[str, torch.Tensor | None]] | None = None,
) -> tuple[list[PredictedNote], dict[str, int]]:
    if waveform.dim() != 2:
        raise ValueError("waveform must have shape [channels, audio_frames]")
    if int(waveform.shape[0]) == 1:
        waveform = waveform.expand(2, -1).contiguous()
    if int(waveform.shape[0]) != 2:
        raise ValueError("the V1 backbone requires mono or stereo input")

    candidate_instrument_ids = effective_instrument_ids(
        num_model_classes=int(config.num_instrument_classes),
        allowed_instrument_ids=settings.allowed_instrument_ids,
        instrument_filter_id=instrument_filter_id,
    )
    sample_rate = int(config.sample_rate)
    total_frames = int(waveform.shape[-1])
    window_frames = int(round(settings.window_ms * sample_rate / 1000.0))
    stride_frames = int(round(settings.stride_ms * sample_rate / 1000.0))
    starts = _build_window_starts(
        total_audio_frames=total_frames,
        window_audio_frames=window_frames,
        stride_audio_frames=stride_frames,
    )
    stitcher = WindowNoteStitcher(
        hop_length=config.hop_length,
        total_audio_frames=total_frames,
        velocity=velocity,
        merge_gap_samples=(
            config.hop_length
            if settings.merge_gap_ms is None
            else int(round(settings.merge_gap_ms * sample_rate / 1000.0))
        ),
        merge_onset_samples=int(round(settings.merge_onset_ms * sample_rate / 1000.0)),
    )
    silence_gate = _silence_gate_rms_linear(settings.silence_gate_rms_dbfs)
    decoded_windows = 0
    skipped_windows = 0
    decoded_intervals = 0
    boundary_count = 0
    boundary_no_onset = 0
    boundary_no_offset = 0
    track_count = NUM_PITCHES * int(config.num_pitch_slots)
    last_closed_global_model_frames = [0] * track_count

    progress = tqdm(
        _iter_batches(starts, settings.window_batch_size),
        total=math.ceil(len(starts) / max(1, settings.window_batch_size)),
        desc="infer",
        dynamic_ncols=True,
        disable=settings.disable_tqdm,
    )
    inference_model = model if forward_model is None else forward_model
    for batch_starts in progress:
        windows: list[torch.Tensor] = []
        valid_audio: list[int] = []
        for start in batch_starts:
            window, valid = _slice_window(
                waveform, start_frame=start, window_audio_frames=window_frames
            )
            windows.append(window)
            valid_audio.append(valid)
        cpu_batch = torch.stack(windows)
        silent = _compute_silent_window_mask(
            cpu_batch, silence_gate_rms_linear=silence_gate
        )
        active = (
            [index for index, value in enumerate(silent.tolist()) if not value]
            if silent is not None
            else list(range(len(batch_starts)))
        )
        skipped_windows += len(batch_starts) - len(active)
        if not active:
            continue
        active_starts = [batch_starts[index] for index in active]
        active_valid = [valid_audio[index] for index in active]
        decoded_windows += len(active)
        batch = cpu_batch[active].to(device)
        valid_tensor = torch.tensor(active_valid, device=device)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled and is_amp_supported(device),
        ):
            outputs = inference_model(
                batch,
                valid_audio_frames=valid_tensor,
                include_aux_outputs=False,
                include_frame_instrument_logits=(
                    not model._use_interval_instrument_head
                ),
            )
        valid_mask = outputs.get("frame_valid_mask")
        if valid_mask is None:
            raise ValueError("V1 inference requires frame_valid_mask")
        valid_lengths = valid_mask.long().sum(dim=-1)
        valid_length_values = tuple(
            int(value) for value in valid_lengths.detach().cpu().tolist()
        )
        decoded_batch: list[list[list[tuple[int, int]]]] = []
        boundary_map: dict[tuple[int, int, int], tuple[bool, bool, float, float]] = {}
        interval_features = outputs.get("interval_features")
        # model forwardは窓バッチのまま、状態に依存する復号だけを窓順に進める。
        for batch_index, window_start in enumerate(active_starts):
            valid_model_frames = valid_length_values[batch_index]
            window_model_start = int(
                round(float(window_start) / float(config.hop_length))
            )
            forced_start_positions = [
                [
                    max(
                        0,
                        min(
                            int(last_closed_global_model_frames[track])
                            - window_model_start,
                            valid_model_frames - 1,
                        ),
                    )
                    for track in range(track_count)
                ]
            ]
            decoded_sample = decode_pitch_intervals(
                outputs["interval_query"][batch_index : batch_index + 1],
                outputs["interval_key"][batch_index : batch_index + 1],
                outputs["interval_diag"][batch_index : batch_index + 1],
                valid_lengths[batch_index : batch_index + 1],
                length_scaling=config.semi_crf_length_scaling,
                length_penalty=config.semi_crf_length_penalty,
                note_bias=settings.note_bias,
                track_batch_size=settings.track_batch_size,
                forced_start_pos=forced_start_positions,
                backend=settings.semi_crf_backend,
            )
            decoded_batch.append(decoded_sample[0])
            decoded_intervals += sum(len(intervals) for intervals in decoded_sample[0])

            local_boundary_map = {}
            if (
                settings.use_boundary_head
                and model.supports_interval_boundaries()
                and interval_features is not None
            ):
                logits, entries = model.predict_interval_boundaries(
                    interval_features[batch_index : batch_index + 1],
                    decoded_sample,
                    compute_dtype=torch.float32,
                )
                local_boundary_map = _decode_boundary_map(logits, entries)
                boundary_count += len(entries)
                for (
                    _,
                    track,
                    interval_index,
                ), boundary_flag in local_boundary_map.items():
                    batch_key = (batch_index, track, interval_index)
                    boundary_map[batch_key] = boundary_flag
                    has_onset, has_offset, _, _ = boundary_flag
                    boundary_no_onset += int(not has_onset)
                    boundary_no_offset += int(not has_offset)

            for track, intervals in enumerate(decoded_sample[0]):
                for interval_index, (_, end) in enumerate(intervals):
                    boundary_flag = local_boundary_map.get((0, track, interval_index))
                    has_offset = (
                        bool(boundary_flag[1])
                        if boundary_flag is not None
                        else int(end) < valid_model_frames - 1
                    )
                    if has_offset:
                        last_closed_global_model_frames[track] = (
                            window_model_start + int(end)
                        )

        batch_decoded_interval_count = sum(
            len(intervals) for sample in decoded_batch for intervals in sample
        )
        instrument_map = {}
        instrument_features = outputs.get("instrument_features")
        if (
            model.supports_interval_instruments()
            and model._use_interval_instrument_head
            and instrument_features is not None
        ):
            logits, entries = model.predict_interval_instruments(
                instrument_features,
                decoded_batch,
                compute_dtype=torch.float32,
            )
            instrument_map = _decode_instrument_map(
                logits,
                entries,
                probability_mode=settings.instrument_probability_mode,
                allowed_instrument_ids=candidate_instrument_ids,
            )

        frame_logits = outputs.get("instrument_logits")
        fallback_instrument_map = (
            _decode_frame_instrument_map(
                frame_logits,
                decoded_batch,
                num_pitch_slots=int(config.num_pitch_slots),
                allowed_instrument_ids=candidate_instrument_ids,
                excluded_keys=set(instrument_map),
            )
            if (
                frame_logits is not None
                and len(instrument_map) < batch_decoded_interval_count
            )
            else {}
        )
        for batch_index, (window_start, valid_frames) in enumerate(
            zip(active_starts, active_valid)
        ):
            valid_model_frames = valid_length_values[batch_index]
            for track, intervals in enumerate(decoded_batch[batch_index]):
                pitch_index = track // int(config.num_pitch_slots)
                slot_index = track % int(config.num_pitch_slots)
                for interval_index, (begin, end) in enumerate(intervals):
                    key = (batch_index, track, interval_index)
                    boundary_flag = boundary_map.get(key)
                    predicted = instrument_map.get(key)
                    if predicted is None:
                        predicted = fallback_instrument_map.get(key)
                    if predicted is None:
                        predicted = (
                            candidate_instrument_ids[0],
                            candidate_instrument_ids,
                        )
                    instrument, candidates = predicted
                    pair_id = instrument * NUM_PITCHES + pitch_index
                    note = stitcher._build_interval_note(
                        pair_id=pair_id,
                        begin_frame=begin,
                        end_frame=end,
                        boundary_flag=boundary_flag,
                        instrument_candidates=candidates,
                        window_start_frame=window_start,
                        valid_audio_frames=valid_frames,
                        valid_model_frames=valid_model_frames,
                    )
                    if note is not None:
                        note.slot_index = slot_index
                        # V1 assigns instruments after pitch-slot stitching. Keep the
                        # continuation track independent of per-window class jitter.
                        note.instrument_id = 0
                        stitcher.notes_by_pair[track].append(note)

        del (
            outputs,
            batch,
            valid_mask,
            valid_lengths,
            interval_features,
            instrument_features,
            frame_logits,
        )

    notes = stitcher.finalize()
    for note in notes:
        note.instrument_id = (
            int(note.instrument_candidates[0])
            if note.instrument_candidates
            else 0
        )
    if instrument_filter_id is not None:
        notes = [
            note
            for note in notes
            if int(note.instrument_id) == int(instrument_filter_id)
        ]
    return notes, {
        "window_count": len(starts),
        "decoded_window_count": decoded_windows,
        "skipped_silent_window_count": skipped_windows,
        "window_audio_frames": window_frames,
        "stride_audio_frames": stride_frames,
        "selected_pair_count": track_count,
        "decoded_interval_count": decoded_intervals,
        "boundary_interval_count": boundary_count,
        "boundary_no_onset_count": boundary_no_onset,
        "boundary_no_offset_count": boundary_no_offset,
        "note_count": len(notes),
    }
