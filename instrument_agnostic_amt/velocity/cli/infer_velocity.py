from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pretty_midi
import soundfile as sf
import torch
import torchaudio.functional as audio_functional
from tqdm.auto import tqdm

from ..modeling.checkpoints import load_checkpoint, select_state_dict
from ..modeling.model import VelocityModelConfig, VelocityPredictionModel
from ..training.dataset import STEM_CLASS_BY_NAME, STEM_NAMES, UNKNOWN_STEM_CLASS


HF_CHECKPOINT_BASE_URL = (
    "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main"
)
DEFAULT_VELOCITY_CHECKPOINT_FILENAME = "best_velocity_model.pth"


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

    waveform_np, source_sample_rate = sf.read(
        str(audio_file), dtype="float32", always_2d=True
    )
    if waveform_np.shape[1] > 2:
        waveform_np = waveform_np[:, :2]
    elif waveform_np.shape[1] == 1:
        waveform_np = np.repeat(waveform_np, 2, axis=1)

    waveform = torch.from_numpy(waveform_np.T.copy())
    if int(source_sample_rate) != int(target_sample_rate):
        waveform = audio_functional.resample(
            waveform, int(source_sample_rate), int(target_sample_rate)
        )

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


def predict_velocity_for_stem_midis(
    stem_midis: Mapping[str, Path | str],
    stem_audios: Mapping[str, Path | str] | Path | str,
    *,
    output_midi_path: Path | str | None = None,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str | None = None,
    window_seconds: float = 16.0,
    disable_tqdm: bool = False,
) -> Path | dict[str, Path]:
    """各ステムの音声と、そのステムから採譜された個別のMIDIを直接対応づけてVelocityを予測する。"""
    if device is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        target_device = torch.device(device)

    model, config = load_velocity_model(checkpoint_path, device=target_device)
    resolved_audios = _resolve_stem_files(stem_audios)

    resolved_midis: dict[str, Path] = {}
    for key, val in stem_midis.items():
        midi_path = Path(val)
        if midi_path.exists():
            resolved_midis[key.lower()] = midi_path

    if not resolved_midis:
        raise ValueError("No valid stem MIDI files provided.")

    active_stem_names = sorted(set(resolved_audios.keys()) | set(resolved_midis.keys()))
    stem_waveforms: list[np.ndarray] = []
    stem_class_ids: list[int] = []

    for name in active_stem_names:
        if name in resolved_audios:
            waveform = _load_and_preprocess_audio(
                resolved_audios[name],
                target_sample_rate=config.sample_rate,
            )
        else:
            first_wave = next(iter(resolved_audios.values()))
            ref_wave = _load_and_preprocess_audio(first_wave, target_sample_rate=config.sample_rate)
            waveform = np.zeros_like(ref_wave)

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
        torch.from_numpy(np.stack(padded_waveforms, axis=0))
        .unsqueeze(0)
        .to(device=target_device, dtype=torch.float32)
    )
    stem_class_tensor = (
        torch.tensor(stem_class_ids, dtype=torch.long)
        .unsqueeze(0)
        .to(device=target_device)
    )

    flat_notes: list[tuple[pretty_midi.Note, int, int, bool, int, pretty_midi.PrettyMIDI]] = []
    loaded_midi_objs: dict[str, pretty_midi.PrettyMIDI] = {}

    for stem_name, midi_path in resolved_midis.items():
        if stem_name in active_stem_names:
            stem_index = active_stem_names.index(stem_name)
        else:
            stem_index = 0

        pm_obj = pretty_midi.PrettyMIDI(str(midi_path))
        loaded_midi_objs[stem_name] = pm_obj

        for inst in pm_obj.instruments:
            for note in inst.notes:
                flat_notes.append((note, inst.program, int(inst.is_drum), False, stem_index, pm_obj))

    if not flat_notes:
        print("Warning: No notes found across stem MIDI files.")
        return list(resolved_midis.values())[0]

    starts = np.array([item[0].start for item in flat_notes], dtype=np.float32)
    ends = np.array([item[0].end for item in flat_notes], dtype=np.float32)
    pitches = np.array([item[0].pitch for item in flat_notes], dtype=np.int64)
    programs = np.array([item[1] for item in flat_notes], dtype=np.int64)
    is_drums = np.array([item[2] for item in flat_notes], dtype=np.int64)
    stem_indices = np.array([item[4] for item in flat_notes], dtype=np.int64)

    total_duration_seconds = float(max_samples) / float(config.sample_rate)
    window_samples = int(window_seconds * config.sample_rate)

    if total_duration_seconds <= window_seconds:
        window_starts_seconds = [0.0]
    else:
        window_starts_seconds = list(np.arange(0.0, total_duration_seconds, window_seconds))

    predicted_velocities = np.full(len(flat_notes), 80, dtype=np.int32)

    with torch.no_grad():
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
            sub_audio = audio_tensor[:, :, :, sample_start:sample_end]

            win_starts = starts[indices_in_win] - win_start
            win_ends = ends[indices_in_win] - win_start

            note_start_tensor = (
                torch.from_numpy(win_starts).unsqueeze(0).to(device=target_device)
            )
            note_end_tensor = (
                torch.from_numpy(win_ends).unsqueeze(0).to(device=target_device)
            )
            note_pitch_tensor = (
                torch.from_numpy(pitches[indices_in_win]).unsqueeze(0).to(device=target_device)
            )
            note_program_tensor = (
                torch.from_numpy(programs[indices_in_win]).unsqueeze(0).to(device=target_device)
            )
            note_is_drum_tensor = (
                torch.from_numpy(is_drums[indices_in_win]).unsqueeze(0).to(device=target_device)
            )
            note_stem_index_tensor = (
                torch.from_numpy(stem_indices[indices_in_win]).unsqueeze(0).to(device=target_device)
            )

            outputs = model(
                sub_audio,
                note_start_seconds=note_start_tensor,
                note_end_seconds=note_end_tensor,
                note_pitch=note_pitch_tensor,
                note_program=note_program_tensor,
                note_is_drum=note_is_drum_tensor,
                note_stem_index=note_stem_index_tensor,
                stem_class_id=stem_class_tensor,
            )

            velocity_expected = outputs["velocity_expected"].squeeze(0).cpu().numpy()
            velocity_clamped = np.clip(np.round(velocity_expected), 1, 127).astype(np.int32)
            predicted_velocities[indices_in_win] = velocity_clamped

    for idx, (note, _, _, _, _, _) in enumerate(flat_notes):
        note.velocity = int(predicted_velocities[idx])

    if output_midi_path is not None:
        merged_midi = pretty_midi.PrettyMIDI()
        for stem_name, pm_obj in loaded_midi_objs.items():
            for inst in pm_obj.instruments:
                merged_midi.instruments.append(inst)

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
    window_seconds: float = 16.0,
    disable_tqdm: bool = False,
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
            temp_pm = pretty_midi.PrettyMIDI()
            temp_pm.instruments.append(inst)
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
        checkpoint_path=checkpoint_path,
        device=device,
        window_seconds=window_seconds,
        disable_tqdm=disable_tqdm,
    )

    for temp_p in stem_midis.values():
        if temp_p.exists() and temp_p.name.startswith("_temp_"):
            temp_p.unlink(missing_ok=True)

    return res if isinstance(res, Path) else Path(output_midi_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict MIDI note velocity from separated stem audio files."
    )
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
        default=None,
        help="Device for inference (cuda or cpu)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=16.0,
        help="Window size in seconds for long audio inference",
    )
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
    )
    print(f"Successfully generated velocity-predicted MIDI: {output_path}")


if __name__ == "__main__":
    main()
