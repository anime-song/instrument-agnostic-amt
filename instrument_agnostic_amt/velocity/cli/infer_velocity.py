from __future__ import annotations

import argparse
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import mido
import numpy as np
import pretty_midi
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from tqdm.auto import tqdm

from ...runtime import (
    is_amp_supported,
    maybe_compile_forward,
    resolve_amp_dtype,
    resolve_device,
)
from ..modeling.checkpoints import load_checkpoint, select_state_dict
from ..modeling.model import VelocityModelConfig, VelocityPredictionModel
from ..stems import STEM_CLASS_BY_NAME, STEM_NAMES, UNKNOWN_STEM_CLASS

HF_CHECKPOINT_BASE_URL = "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main"
DEFAULT_VELOCITY_CHECKPOINT_FILENAME = "best_velocity_model.pth"


@dataclass
class _VelocityNoteRecord:
    note: pretty_midi.Note
    program: int
    is_drum: bool
    stem_index: int
    stem_name: str
    instrument_name: str


def _normalized_instrument_name(name: str) -> str:
    return str(name).strip().casefold()


def _pop_unused_assignment(
    queue: deque[int] | None,
    *,
    used: set[int],
) -> int | None:
    if queue is None:
        return None
    while queue and queue[0] in used:
        queue.popleft()
    if not queue:
        return None
    assignment_index = queue.popleft()
    used.add(assignment_index)
    return assignment_index


def _replace_fixed_loudness_controls_in_mido(midi: mido.MidiFile) -> None:
    """CC7/CC11を各演奏チャンネルのtick 0へ1件ずつ設定する。"""

    for track in midi.tracks:
        channels = sorted(
            {
                int(message.channel)
                for message in track
                if hasattr(message, "channel") and message.type in {"program_change", "note_on", "note_off"}
            }
        )
        if not channels:
            continue

        rebuilt_track = mido.MidiTrack()
        carried_delta = 0
        for message in track:
            if message.type == "control_change" and int(message.control) in (7, 11):
                carried_delta += int(message.time)
                continue
            rebuilt_track.append(message.copy(time=int(message.time) + carried_delta))
            carried_delta = 0

        insert_at = 0
        while insert_at < len(rebuilt_track) and rebuilt_track[insert_at].time == 0:
            insert_at += 1
        fixed_controls = [
            mido.Message(
                "control_change",
                channel=channel,
                control=control,
                value=127,
                time=0,
            )
            for channel in channels
            for control in (7, 11)
        ]
        rebuilt_track[insert_at:insert_at] = fixed_controls
        track[:] = rebuilt_track


def _write_velocity_template_midi(
    template_midi_path: Path | str,
    output_midi_path: Path | str,
    note_records: list[_VelocityNoteRecord],
    *,
    set_fixed_loudness_controls: bool,
) -> Path:
    """テンプレートのNote On velocityだけを更新し、ノート構造を維持する。"""

    template_path = Path(template_midi_path)
    if not template_path.exists():
        raise FileNotFoundError(f"Template MIDI file not found: {template_path}")

    template_pm = pretty_midi.PrettyMIDI(str(template_path))
    template_midi = mido.MidiFile(str(template_path))
    exact_assignments: dict[tuple[str, int, bool, int, int], deque[int]] = defaultdict(deque)
    program_assignments: dict[tuple[int, bool, int, int], deque[int]] = defaultdict(deque)
    global_assignments: dict[tuple[int, int], deque[int]] = defaultdict(deque)

    for assignment_index, record in enumerate(note_records):
        start_tick = int(template_pm.time_to_tick(float(record.note.start)))
        instrument_name = _normalized_instrument_name(record.instrument_name)
        exact_assignments[
            (
                instrument_name,
                int(record.program),
                bool(record.is_drum),
                int(record.note.pitch),
                start_tick,
            )
        ].append(assignment_index)
        program_assignments[
            (
                int(record.program),
                bool(record.is_drum),
                int(record.note.pitch),
                start_tick,
            )
        ].append(assignment_index)
        global_assignments[(int(record.note.pitch), start_tick)].append(assignment_index)

    used_assignments: set[int] = set()
    unmatched_note_ons: list[tuple[int, str, int, int]] = []

    for track_index, track in enumerate(template_midi.tracks):
        track_name = next(
            (message.name for message in track if message.type == "track_name"),
            "",
        )
        normalized_track_name = _normalized_instrument_name(track_name)
        program_by_channel: dict[int, int] = {}
        absolute_tick = 0

        for message in track:
            absolute_tick += int(message.time)
            if message.type == "program_change":
                program_by_channel[int(message.channel)] = int(message.program)
                continue
            if message.type != "note_on" or int(message.velocity) <= 0:
                continue

            channel = int(message.channel)
            program = int(program_by_channel.get(channel, 0))
            is_drum = channel == 9
            pitch = int(message.note)
            assignment_index: int | None = None

            for tick_offset in (0, -1, 1, -2, 2):
                candidate_tick = absolute_tick + tick_offset
                assignment_index = _pop_unused_assignment(
                    exact_assignments.get(
                        (
                            normalized_track_name,
                            program,
                            is_drum,
                            pitch,
                            candidate_tick,
                        )
                    ),
                    used=used_assignments,
                )
                if assignment_index is not None:
                    break
                assignment_index = _pop_unused_assignment(
                    program_assignments.get((program, is_drum, pitch, candidate_tick)),
                    used=used_assignments,
                )
                if assignment_index is not None:
                    break
                assignment_index = _pop_unused_assignment(
                    global_assignments.get((pitch, candidate_tick)),
                    used=used_assignments,
                )
                if assignment_index is not None:
                    break

            if assignment_index is None:
                unmatched_note_ons.append((track_index, track_name, pitch, absolute_tick))
                continue
            message.velocity = int(note_records[assignment_index].note.velocity)

    if unmatched_note_ons:
        preview = ", ".join(
            f"track={track_index} name={track_name!r} pitch={pitch} tick={tick}"
            for track_index, track_name, pitch, tick in unmatched_note_ons[:5]
        )
        raise ValueError(
            f"Velocity predictions could not be matched to {len(unmatched_note_ons)} template Note On events. {preview}"
        )

    if set_fixed_loudness_controls:
        _replace_fixed_loudness_controls_in_mido(template_midi)

    destination_path = Path(output_midi_path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    template_midi.save(str(destination_path))
    return destination_path


def ensure_velocity_checkpoint(checkpoint_path: Path | str | None = None) -> Path:
    """Velocity予測モデルのチェックポイントが存在しない場合にHugging Faceから自動ダウンロードする。"""
    if checkpoint_path is None or str(checkpoint_path).strip() in ("", "DEFAULT"):
        checkpoint_path = Path("checkpoints") / DEFAULT_VELOCITY_CHECKPOINT_FILENAME
    else:
        checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF_CHECKPOINT_BASE_URL}/{DEFAULT_VELOCITY_CHECKPOINT_FILENAME}?download=true"
        print(f"Velocity checkpoint not found at {checkpoint_path}. Downloading from Hugging Face...")
        torch.hub.download_url_to_file(url, str(checkpoint_path))

    return checkpoint_path


def load_velocity_model(
    checkpoint_path: Path | str | None = None,
    *,
    device: torch.device | str = "cpu",
) -> tuple[VelocityPredictionModel, VelocityModelConfig]:
    """Velocity予測モデルと設定をチェックポイントから読み込む。"""
    target_device = torch.device(device)
    resolved_checkpoint_path = ensure_velocity_checkpoint(checkpoint_path)
    checkpoint = load_checkpoint(resolved_checkpoint_path)

    raw_config = checkpoint.get("config") or checkpoint.get("model_config")
    if raw_config is not None:
        if isinstance(raw_config, VelocityModelConfig):
            config = raw_config
        elif isinstance(raw_config, dict):
            config_values = dict(raw_config)
            for key in ("harmonics", "local_frame_offsets"):
                if key in config_values and isinstance(config_values[key], list):
                    config_values[key] = tuple(config_values[key])
            config = VelocityModelConfig(**config_values)
        else:
            config = VelocityModelConfig()
    else:
        config = VelocityModelConfig()

    model = VelocityPredictionModel(config)
    state_dict = select_state_dict(checkpoint, prefer_ema=True)
    model.load_state_dict(state_dict, strict=False)
    model.to(target_device)
    model.eval()

    return model, config


def _load_and_preprocess_audio(
    audio_path: Path | str,
    *,
    target_sample_rate: int,
) -> np.ndarray:
    """音声を指定サンプルレートのステレオ (2, num_samples) float32 配列として読み込む。"""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    waveform_np, source_sample_rate = sf.read(str(audio_file), dtype="float32", always_2d=True)
    if waveform_np.shape[1] > 2:
        waveform_np = waveform_np[:, :2]
    elif waveform_np.shape[1] == 1:
        waveform_np = np.repeat(waveform_np, 2, axis=1)

    waveform = torch.from_numpy(waveform_np.T.copy())
    if int(source_sample_rate) != int(target_sample_rate):
        waveform = audio_functional.resample(waveform, int(source_sample_rate), int(target_sample_rate))

    return waveform.numpy().astype(np.float32)


def _resolve_stem_files(
    stems_input: Mapping[str, Path | str] | Path | str | list[Path | str],
) -> dict[str, Path]:
    """多様な入力形式からステム名 -> ファイルパスの辞書を解決する。"""
    resolved_stems: dict[str, Path] = {}

    if isinstance(stems_input, Mapping):
        for name, path in stems_input.items():
            file_path = Path(path)
            if file_path.exists():
                resolved_stems[name.lower()] = file_path
    elif isinstance(stems_input, (str, Path)):
        stem_dir = Path(stems_input)
        if stem_dir.is_dir():
            for wav_file in sorted(stem_dir.glob("*.wav")):
                stem_name = wav_file.stem.lower()
                for known_name in STEM_NAMES:
                    if known_name in stem_name or stem_name.endswith(f"_{known_name}"):
                        resolved_stems[known_name] = wav_file
                        break
                else:
                    resolved_stems[stem_name] = wav_file
        elif stem_dir.is_file():
            resolved_stems["other"] = stem_dir
    elif isinstance(stems_input, (list, tuple)):
        for path_item in stems_input:
            file_path = Path(path_item)
            if file_path.exists():
                stem_name = file_path.stem.lower()
                for known_name in STEM_NAMES:
                    if known_name in stem_name or stem_name.endswith(f"_{known_name}"):
                        resolved_stems[known_name] = file_path
                        break
                else:
                    resolved_stems[stem_name] = file_path

    if not resolved_stems:
        raise ValueError(f"No valid stem audio files found from input: {stems_input}")

    return resolved_stems


def _set_fixed_loudness_controls(
    instruments: list[pretty_midi.Instrument],
) -> None:
    for instrument in instruments:
        instrument.control_changes = [
            control for control in instrument.control_changes if int(control.number) not in (7, 11)
        ]
        instrument.control_changes.extend(
            [
                pretty_midi.ControlChange(number=7, value=127, time=0.0),
                pretty_midi.ControlChange(number=11, value=127, time=0.0),
            ]
        )


def _merge_and_limit_instruments(
    midi_objects: Mapping[str, pretty_midi.PrettyMIDI],
    *,
    max_melodic_instruments: int,
) -> list[pretty_midi.Instrument]:
    """Merge duplicate instruments and reserve the last melodic slot for overflow."""

    grouped: dict[tuple[int, bool, str], pretty_midi.Instrument] = {}
    for midi in midi_objects.values():
        for source in midi.instruments:
            key = (int(source.program), bool(source.is_drum), str(source.name))
            target = grouped.get(key)
            if target is None:
                target = pretty_midi.Instrument(
                    program=key[0],
                    is_drum=key[1],
                    name=key[2],
                )
                grouped[key] = target
            target.notes.extend(source.notes)
            target.control_changes.extend(source.control_changes)
            target.pitch_bends.extend(source.pitch_bends)

    melodic = [instrument for instrument in grouped.values() if not instrument.is_drum]
    drums = [instrument for instrument in grouped.values() if instrument.is_drum]
    sort_key = lambda instrument: (
        -len(instrument.notes),
        int(instrument.program),
        str(instrument.name),
    )
    melodic.sort(key=sort_key)
    drums.sort(key=sort_key)

    limit = int(max_melodic_instruments)
    if limit > 0 and len(melodic) > limit:
        kept = melodic[: max(0, limit - 1)]
        overflow = melodic[max(0, limit - 1) :]
        merged_other = pretty_midi.Instrument(
            program=int(overflow[0].program),
            is_drum=False,
            name="Other / Merged",
        )
        for instrument in overflow:
            merged_other.notes.extend(instrument.notes)
            merged_other.control_changes.extend(instrument.control_changes)
            merged_other.pitch_bends.extend(instrument.pitch_bends)
        melodic = [*kept, merged_other]

    final_instruments = [*melodic, *drums]
    for instrument in final_instruments:
        instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        instrument.control_changes.sort(key=lambda control: control.time)
        instrument.pitch_bends.sort(key=lambda bend: bend.time)
    return final_instruments


def predict_velocity_for_stem_midis(
    stem_midis: Mapping[str, Path | str],
    stem_audios: Mapping[str, Path | str] | Path | str,
    *,
    output_midi_path: Path | str | None = None,
    template_midi_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str | None = None,
    window_seconds: float = 8.0,
    max_melodic_instruments: int = 15,
    apply_stem_gain_to_cc7: bool = False,
    disable_tqdm: bool = False,
    compile_velocity: bool = False,
    compile_mode: str = "default",
    preloaded_model: VelocityPredictionModel | None = None,
    preloaded_config: VelocityModelConfig | None = None,
    preloaded_forward: Any | None = None,
    preloaded_waveforms: Mapping[str, torch.Tensor] | None = None,
    amp: bool = False,
    amp_dtype: str | torch.dtype | None = None,
) -> Path | dict[str, Path]:
    """各ステムの音声とMIDIを対応づけ、ノートVelocityを予測して反映する。

    Velocity-only推論ではCC7 VolumeとCC11 Expressionを127へ固定し、学習renderと
    同じ基準でノートVelocityを唯一の可変音量表現にする。template_midi_pathを
    指定した場合は、そのMIDIのNote Onイベントへ予測値だけを書き戻す。
    preloaded_waveforms にはconfig.sample_rateへresample済みのCPU float32ステレオ波形を渡せる。
    ampを立てると窓ごとのbackboneをautocastで走らせる。予測値はfloat32へ戻してから
    丸めるため、AMP由来の差はvelocity数段階に収まる。
    """
    target_device = resolve_device(device)
    amp_enabled = bool(amp and is_amp_supported(target_device))
    resolved_amp_dtype = (
        amp_dtype if isinstance(amp_dtype, torch.dtype) else resolve_amp_dtype(target_device, amp_dtype)
    )
    if max_melodic_instruments < 0:
        raise ValueError("max_melodic_instruments must be nonnegative")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    if template_midi_path is not None and output_midi_path is None:
        raise ValueError("template_midi_path requires output_midi_path")
    if template_midi_path is not None and apply_stem_gain_to_cc7:
        raise ValueError("template_midi_path is not compatible with legacy stem-gain CC7 output")

    if (preloaded_model is None) != (preloaded_config is None):
        raise ValueError("preloaded_model and preloaded_config must be provided together")
    if preloaded_forward is not None and preloaded_model is None:
        raise ValueError("preloaded_forward requires preloaded_model and preloaded_config")
    if preloaded_model is None or preloaded_config is None:
        model, config = load_velocity_model(checkpoint_path, device=target_device)
    else:
        model = preloaded_model
        config = preloaded_config
        model.to(target_device)
        model.eval()
    forward_model = (
        preloaded_forward
        if preloaded_forward is not None
        else maybe_compile_forward(
            model,
            enabled=bool(compile_velocity),
            mode=str(compile_mode),
        )
    )
    configure_stem_gain = isinstance(model, VelocityPredictionModel) and forward_model is model
    if apply_stem_gain_to_cc7 and not config.predict_stem_gain:
        raise ValueError("This velocity checkpoint does not contain a stem-gain prediction head")
    resolved_audios = _resolve_stem_files(stem_audios)
    resolved_midis: dict[str, Path] = {}
    for key, val in stem_midis.items():
        midi_path = Path(val)
        if midi_path.exists():
            resolved_midis[key.lower()] = midi_path

    if not resolved_midis:
        raise ValueError("No valid stem MIDI files provided.")

    # MIDIを先に読み、velocityを割り当てる必要があるステムを確定する。既定の
    # velocity-only headでは各stemのbackboneが独立なので、ノートがないstemの音声を
    # GPUへ載せても予測値には寄与せず、曲全体にわたって計算量だけが増える。
    flat_notes: list[_VelocityNoteRecord] = []
    loaded_midi_objs: dict[str, pretty_midi.PrettyMIDI] = {}

    for stem_name, midi_path in resolved_midis.items():
        pm_obj = pretty_midi.PrettyMIDI(str(midi_path))
        loaded_midi_objs[stem_name] = pm_obj

        for inst in pm_obj.instruments:
            for note in inst.notes:
                flat_notes.append(
                    _VelocityNoteRecord(
                        note=note,
                        program=int(inst.program),
                        is_drum=bool(inst.is_drum),
                        stem_index=-1,
                        stem_name=stem_name,
                        instrument_name=str(inst.name),
                    )
                )

    if not flat_notes:
        print("Warning: No notes found across stem MIDI files.")
        return list(resolved_midis.values())[0]

    select_note_stems_only = bool(
        not apply_stem_gain_to_cc7 and isinstance(model, VelocityPredictionModel) and forward_model is model
    )
    if select_note_stems_only:
        active_stem_names = sorted({record.stem_name for record in flat_notes})
        skipped_stem_names = sorted((set(resolved_audios) | set(resolved_midis)) - set(active_stem_names))
        if skipped_stem_names:
            print(f"Skipping velocity backbone for stems without notes: {skipped_stem_names}")
    else:
        # Legacy stem-gain予測はstem間の相対レベルを使うため、全stemが必要。
        active_stem_names = sorted(set(resolved_audios.keys()) | set(resolved_midis.keys()))

    stem_index_by_name = {stem_name: stem_index for stem_index, stem_name in enumerate(active_stem_names)}
    for record in flat_notes:
        record.stem_index = stem_index_by_name[record.stem_name]

    cached_waveforms = {str(name).lower(): waveform for name, waveform in (preloaded_waveforms or {}).items()}

    def resolve_waveform(name: str, path: Path) -> np.ndarray:
        cached = cached_waveforms.get(name.lower())
        if cached is None:
            return _load_and_preprocess_audio(
                path,
                target_sample_rate=config.sample_rate,
            )
        if cached.device.type != "cpu":
            raise ValueError("preloaded_waveforms must be on CPU")
        if cached.dtype != torch.float32:
            raise ValueError("preloaded_waveforms must have dtype float32")
        if cached.ndim != 2 or int(cached.shape[0]) != 2:
            raise ValueError("preloaded_waveforms must have shape [2, T]")
        waveform = cached.detach().numpy()
        waveform.flags.writeable = False
        return waveform

    stem_waveforms: list[np.ndarray] = []
    stem_class_ids: list[int] = []
    for name in active_stem_names:
        if name in resolved_audios:
            waveform = resolve_waveform(name, resolved_audios[name])
        else:
            first_name, first_wave = next(iter(resolved_audios.items()))
            waveform = np.zeros_like(resolve_waveform(first_name, first_wave))
        stem_waveforms.append(waveform)
        stem_class_ids.append(STEM_CLASS_BY_NAME.get(name, UNKNOWN_STEM_CLASS))

    max_samples = max(wave.shape[1] for wave in stem_waveforms)
    padded_waveforms: list[np.ndarray] = []
    for wave in stem_waveforms:
        pad_width = max_samples - wave.shape[1]
        if pad_width > 0:
            wave = np.pad(wave, ((0, 0), (0, pad_width)))
        padded_waveforms.append(wave)

    audio_tensor = (
        torch.from_numpy(np.stack(padded_waveforms, axis=0)).unsqueeze(0).to(device=target_device, dtype=torch.float32)
    )
    stem_class_tensor = torch.tensor(stem_class_ids, dtype=torch.long).unsqueeze(0).to(device=target_device)

    starts = np.array([item.note.start for item in flat_notes], dtype=np.float32)
    ends = np.array([item.note.end for item in flat_notes], dtype=np.float32)
    pitches = np.array([item.note.pitch for item in flat_notes], dtype=np.int64)
    programs = np.array([item.program for item in flat_notes], dtype=np.int64)
    is_drums = np.array([item.is_drum for item in flat_notes], dtype=np.int64)
    stem_indices = np.array([item.stem_index for item in flat_notes], dtype=np.int64)

    total_duration_seconds = float(max_samples) / float(config.sample_rate)
    window_samples = int(window_seconds * config.sample_rate)
    if window_samples <= 0:
        raise ValueError("window_seconds is too small for the model sample rate")

    if total_duration_seconds <= window_seconds:
        window_starts_seconds = [0.0]
    else:
        window_starts_seconds = list(np.arange(0.0, total_duration_seconds, window_seconds))

    predicted_velocities = np.full(len(flat_notes), 80, dtype=np.int32)
    predicted_stem_gains_db_list: list[np.ndarray] = []

    with torch.inference_mode():
        for win_start in tqdm(
            window_starts_seconds,
            desc="Predicting velocity",
            disable=disable_tqdm,
        ):
            win_end = win_start + window_seconds
            note_mask_in_win = (starts >= win_start) & (starts < win_end)

            indices_in_win = np.where(note_mask_in_win)[0]
            if len(indices_in_win) == 0:
                continue

            sample_start = int(win_start * config.sample_rate)
            sample_end = min(sample_start + window_samples, max_samples)
            window_stem_indices = stem_indices[indices_in_win]
            if select_note_stems_only:
                # stem-gainを使わない経路では、当該窓にノートがあるstemだけをbackboneへ
                # 通す。stem indexは選択後の局所indexへ詰め直す。
                selected_stem_indices = np.unique(window_stem_indices)
                sub_audio = audio_tensor[:, selected_stem_indices.tolist(), :, sample_start:sample_end]
                local_stem_indices = np.searchsorted(
                    selected_stem_indices,
                    window_stem_indices,
                )
                window_stem_class_tensor = stem_class_tensor[:, selected_stem_indices.tolist()]
            else:
                sub_audio = audio_tensor[:, :, :, sample_start:sample_end]
                local_stem_indices = window_stem_indices
                window_stem_class_tensor = stem_class_tensor

            win_starts = (starts[indices_in_win] - win_start).astype(np.float32)
            win_ends = (ends[indices_in_win] - win_start).astype(np.float32)

            note_start_tensor = torch.from_numpy(win_starts).unsqueeze(0).to(device=target_device, dtype=torch.float32)
            note_end_tensor = torch.from_numpy(win_ends).unsqueeze(0).to(device=target_device, dtype=torch.float32)
            note_pitch_tensor = torch.from_numpy(pitches[indices_in_win]).unsqueeze(0).to(device=target_device)
            note_program_tensor = torch.from_numpy(programs[indices_in_win]).unsqueeze(0).to(device=target_device)
            note_is_drum_tensor = torch.from_numpy(is_drums[indices_in_win]).unsqueeze(0).to(device=target_device)
            note_stem_index_tensor = torch.from_numpy(local_stem_indices).unsqueeze(0).to(device=target_device)

            forward_kwargs: dict[str, Any] = {
                "note_start_seconds": note_start_tensor,
                "note_end_seconds": note_end_tensor,
                "note_pitch": note_pitch_tensor,
                "note_program": note_program_tensor,
                "note_is_drum": note_is_drum_tensor,
                "note_stem_index": note_stem_index_tensor,
                "stem_class_id": window_stem_class_tensor,
            }
            if configure_stem_gain:
                forward_kwargs["include_stem_gain"] = bool(apply_stem_gain_to_cc7)
            with torch.amp.autocast(
                device_type=target_device.type,
                dtype=resolved_amp_dtype,
                enabled=amp_enabled,
            ):
                outputs = forward_model(sub_audio, **forward_kwargs)

            # autocast下ではheadの出力もbf16/fp16になる。numpyはbf16を扱えず、
            # 丸め位置もdtypeで変わるため、float32へ戻してから量子化する。
            velocity_expected = outputs["velocity_expected"].squeeze(0).float().cpu().numpy()
            velocity_clamped = np.clip(np.round(velocity_expected), 1, 127).astype(np.int32)
            predicted_velocities[indices_in_win] = velocity_clamped

            if "stem_gain_db" in outputs:
                stem_gain_np = outputs["stem_gain_db"].squeeze(0).float().cpu().numpy()
                predicted_stem_gains_db_list.append(stem_gain_np)
            del outputs

    for idx, record in enumerate(flat_notes):
        record.note.velocity = int(predicted_velocities[idx])

    if not apply_stem_gain_to_cc7:
        for pm_obj in loaded_midi_objs.values():
            _set_fixed_loudness_controls(pm_obj.instruments)

    # 旧joint checkpointを明示的に指定した場合だけ予測gainをCC#7へ反映する。
    stem_gain_by_name: dict[str, float] = {}
    if predicted_stem_gains_db_list:
        mean_stem_gains = np.mean(np.stack(predicted_stem_gains_db_list, axis=0), axis=0)
        for i, s_name in enumerate(active_stem_names):
            stem_gain_by_name[s_name] = float(mean_stem_gains[i])

    if apply_stem_gain_to_cc7 and stem_gain_by_name:
        for s_name, pm_obj in loaded_midi_objs.items():
            gain_db = stem_gain_by_name.get(s_name, 0.0)
            linear_scale = float(10.0 ** (gain_db / 20.0))

            for inst in pm_obj.instruments:
                cc7_events = [cc for cc in inst.control_changes if cc.number == 7]
                if cc7_events:
                    for cc in cc7_events:
                        updated_val = int(np.clip(round(cc.value * linear_scale), 1, 127))
                        cc.value = updated_val
                else:
                    base_val = 100
                    updated_val = int(np.clip(round(base_val * linear_scale), 1, 127))
                    inst.control_changes.append(pretty_midi.ControlChange(number=7, value=updated_val, time=0.0))

    if output_midi_path is not None and template_midi_path is not None:
        return _write_velocity_template_midi(
            template_midi_path=template_midi_path,
            output_midi_path=output_midi_path,
            note_records=flat_notes,
            set_fixed_loudness_controls=not apply_stem_gain_to_cc7,
        )

    if output_midi_path is not None:
        # 最初のステムMIDIファイルをベースにしてテンポマップ・解像度を保持する
        first_midi_path = next(iter(resolved_midis.values()))
        merged_midi = pretty_midi.PrettyMIDI(str(first_midi_path))
        merged_midi.instruments = _merge_and_limit_instruments(
            loaded_midi_objs,
            max_melodic_instruments=max_melodic_instruments,
        )
        if not apply_stem_gain_to_cc7:
            _set_fixed_loudness_controls(merged_midi.instruments)

        destination_path = Path(output_midi_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        merged_midi.write(str(destination_path))
        return destination_path
    else:
        updated_stem_paths: dict[str, Path] = {}
        for stem_name, pm_obj in loaded_midi_objs.items():
            original_path = resolved_midis[stem_name]
            out_path = original_path.parent / f"{original_path.stem}_velocity.mid"
            pm_obj.write(str(out_path))
            updated_stem_paths[stem_name] = out_path
        return updated_stem_paths


def predict_velocity_for_midi(
    midi_path: Path | str,
    stems: Mapping[str, Path | str] | Path | str | list[Path | str],
    *,
    output_midi_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str | None = None,
    window_seconds: float = 8.0,
    max_melodic_instruments: int = 15,
    apply_stem_gain_to_cc7: bool = False,
    disable_tqdm: bool = False,
    compile_velocity: bool = False,
    compile_mode: str = "default",
) -> Path:
    """単一のMIDIファイルを受け取った場合の互換用エントリポイント。"""
    midi_file_path = Path(midi_path)
    if not midi_file_path.exists():
        raise FileNotFoundError(f"MIDI file not found: {midi_file_path}")

    pm_obj = pretty_midi.PrettyMIDI(str(midi_file_path))
    resolved_audios = _resolve_stem_files(stems)

    stem_midis: dict[str, Path] = {}

    for inst in pm_obj.instruments:
        stem_name_candidates = [name for name in STEM_NAMES if name in inst.name.lower()]
        stem_name = stem_name_candidates[0] if stem_name_candidates else "other"

        if stem_name not in stem_midis:
            temp_pm = pretty_midi.PrettyMIDI(str(midi_file_path))
            temp_pm.instruments = [inst]
            temp_path = midi_file_path.parent / f"_temp_{stem_name}.mid"
            temp_pm.write(str(temp_path))
            stem_midis[stem_name] = temp_path
        else:
            temp_pm = pretty_midi.PrettyMIDI(str(stem_midis[stem_name]))
            temp_pm.instruments.append(inst)
            temp_pm.write(str(stem_midis[stem_name]))

    res = predict_velocity_for_stem_midis(
        stem_midis=stem_midis,
        stem_audios=resolved_audios,
        output_midi_path=output_midi_path or (midi_file_path.parent / f"{midi_file_path.stem}_velocity.mid"),
        template_midi_path=(midi_file_path if not apply_stem_gain_to_cc7 else None),
        checkpoint_path=checkpoint_path,
        device=device,
        window_seconds=window_seconds,
        max_melodic_instruments=max_melodic_instruments,
        apply_stem_gain_to_cc7=apply_stem_gain_to_cc7,
        disable_tqdm=disable_tqdm,
        compile_velocity=compile_velocity,
        compile_mode=compile_mode,
    )

    for temp_p in stem_midis.values():
        if temp_p.exists() and temp_p.name.startswith("_temp_"):
            temp_p.unlink(missing_ok=True)

    return res if isinstance(res, Path) else Path(output_midi_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict MIDI note velocity from separated stem audio files.")
    parser.add_argument("--midi", type=Path, required=True, help="Input MIDI file path")
    parser.add_argument(
        "--stems-dir",
        type=Path,
        help="Directory containing separated stem audio files (e.g. bass.wav, drums.wav)",
    )
    parser.add_argument(
        "--stem-files",
        nargs="+",
        help="List of stem audio file paths",
    )
    parser.add_argument(
        "--output-midi",
        type=Path,
        help="Output MIDI file path with updated velocity",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Path to velocity model checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference (auto, cuda, mps, or cpu)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=8.0,
        help="Window size in seconds for long audio inference",
    )
    parser.add_argument(
        "--compile-velocity",
        dest="compile_velocity",
        action="store_true",
        help="Compile shared velocity-model Transformer regions with torch.compile",
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
        help="torch.compile mode",
    )
    parser.add_argument(
        "--max-midi-melodic-instruments",
        type=int,
        default=15,
        help="Maximum melodic tracks; overflow is merged into Other / Merged",
    )
    gain_group = parser.add_mutually_exclusive_group()
    gain_group.add_argument(
        "--apply-cc7-gain",
        dest="apply_cc7_gain",
        action="store_true",
        help="Opt in to legacy stem-gain prediction for a compatible checkpoint",
    )
    gain_group.add_argument(
        "--no-cc7-gain",
        dest="apply_cc7_gain",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(apply_cc7_gain=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stems_input = args.stems_dir if args.stems_dir is not None else args.stem_files
    if stems_input is None:
        raise ValueError("Either --stems-dir or --stem-files must be specified.")

    output_path = predict_velocity_for_midi(
        midi_path=args.midi,
        stems=stems_input,
        output_midi_path=args.output_midi,
        checkpoint_path=args.checkpoint,
        device=args.device,
        window_seconds=args.window_seconds,
        max_melodic_instruments=args.max_midi_melodic_instruments,
        apply_stem_gain_to_cc7=args.apply_cc7_gain,
        compile_velocity=args.compile_velocity,
        compile_mode=args.compile_mode,
    )
    print(f"Successfully generated velocity-predicted MIDI: {output_path}")


if __name__ == "__main__":
    main()
