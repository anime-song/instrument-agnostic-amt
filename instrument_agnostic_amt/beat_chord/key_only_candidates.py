from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from ..runtime import (
    empty_device_cache,
    is_amp_supported,
    maybe_compile_forward,
    resolve_amp_dtype,
    resolve_device,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path("beat_chord_dataset/source_audio")
DEFAULT_OUTPUT_DIR = Path("beat_chord_dataset/key_only_candidates")
DEFAULT_AMT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_SEPARATION_CHECKPOINT = Path("checkpoints/stem_splitter.pt")
DEFAULT_VELOCITY_CHECKPOINT = Path("checkpoints/best_velocity_model.pth")
DEFAULT_BEAT_CHORD_CHECKPOINT_DIR = Path("beat_chord_checkpoints/midi_frame")
DEFAULT_QUALITY_JSON = Path("beat_chord_dataset/chord_dataset/quality.json")

AUDIO_EXTENSIONS = frozenset(
    {
        ".aac",
        ".aif",
        ".aiff",
        ".flac",
        ".m4a",
        ".mp3",
        ".ogg",
        ".opus",
        ".wav",
        ".wma",
    }
)
DEFAULT_STEM_NAMES = (
    "bass",
    "drums",
    "other",
    "vocals",
    "guitar",
    "piano",
)


@dataclass(frozen=True)
class CandidatePaths:
    run_root: Path
    stem_dir: Path
    stem_midi_dir: Path
    merged_dir: Path
    prediction_dir: Path


@dataclass
class CandidateResult:
    audio_path: str
    song_name: str
    run_root: str
    transcription_status: str = "pending"
    prediction_status: str = "pending"
    source_midi_path: str | None = None
    candidate_midi_path: str | None = None
    prediction_json_path: str | None = None
    error_phase: str | None = None
    error: str | None = None


def is_valid_audio_file(path: str | Path) -> bool:
    audio_path = Path(path)
    if not audio_path.is_file() or audio_path.stat().st_size <= 0:
        return False
    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
    except Exception:
        return False
    return int(info.frames) > 0 and int(info.channels) > 0


def is_valid_midi_file(path: str | Path) -> bool:
    midi_path = Path(path)
    if not midi_path.is_file() or midi_path.stat().st_size <= 0:
        return False
    try:
        import pretty_midi

        pretty_midi.PrettyMIDI(str(midi_path))
    except Exception:
        return False
    return True


def is_valid_prediction_json(path: str | Path) -> bool:
    prediction_path = Path(path)
    if not prediction_path.is_file() or prediction_path.stat().st_size <= 0:
        return False
    try:
        with open(prediction_path, encoding="utf-8") as file:
            payload = json.load(file)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    required_lists = ("beats", "downbeats", "meters", "chords", "keys")
    return all(isinstance(payload.get(name), list) for name in required_lists)


def prediction_outputs_complete(
    *,
    candidate_midi_path: str | Path,
    prediction_json_path: str | Path,
) -> bool:
    return is_valid_midi_file(candidate_midi_path) and is_valid_prediction_json(
        prediction_json_path
    )


def discover_audio_files(
    input_dir: str | Path,
    *,
    recursive: bool = False,
) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Input audio directory not found: {root}")

    iterator: Iterable[Path] = root.rglob("*") if recursive else root.iterdir()
    audio_paths = sorted(
        (
            path.resolve()
            for path in iterator
            if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS
        ),
        key=lambda path: str(path).casefold(),
    )
    if not audio_paths:
        raise ValueError(f"No supported audio files were found in {root}")

    by_song_name: dict[str, Path] = {}
    for path in audio_paths:
        normalized_stem = path.stem.casefold()
        previous = by_song_name.get(normalized_stem)
        if previous is not None:
            raise ValueError(
                "Audio files must have unique stems because the stem becomes the "
                f"output song ID: {previous} and {path}"
            )
        by_song_name[normalized_stem] = path
    return audio_paths


def resolve_beat_chord_checkpoint(
    checkpoint: str | Path | None,
    *,
    checkpoint_dir: str | Path = DEFAULT_BEAT_CHORD_CHECKPOINT_DIR,
) -> Path:
    if checkpoint is not None:
        resolved = Path(checkpoint).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Beat/chord checkpoint not found: {resolved}")
        return resolved

    root = Path(checkpoint_dir)
    candidates = list(root.glob("checkpoint_epoch_*.pth"))
    if not candidates:
        candidates = list(root.glob("*.pth"))
    if not candidates:
        raise FileNotFoundError(
            "No beat/chord checkpoint was found. Pass --beat-chord-checkpoint "
            f"or place checkpoints under {root}"
        )
    # The directory can contain an older run with a larger epoch number. The
    # most recently written checkpoint is the best proxy for the current model.
    return max(
        candidates,
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    ).resolve()


def candidate_paths(output_root: str | Path, song_name: str) -> CandidatePaths:
    run_root = Path(output_root).resolve() / song_name
    return CandidatePaths(
        run_root=run_root,
        stem_dir=run_root / "stems",
        stem_midi_dir=run_root / "stem_midis",
        merged_dir=run_root / "merged",
        prediction_dir=run_root / "prediction",
    )


def resolve_stem_paths(
    *,
    song_path: str | Path,
    stem_dir: str | Path,
    stem_names: Sequence[str],
) -> dict[str, Path]:
    song_file = Path(song_path)
    stem_root = Path(stem_dir) / song_file.stem
    resolved: dict[str, Path] = {}

    for stem_name in stem_names:
        expected_path = stem_root / f"{song_file.stem}_{stem_name}.wav"
        if expected_path.exists():
            resolved[stem_name] = expected_path
    if resolved:
        return resolved
    if not stem_root.exists():
        return resolved

    for wav_path in sorted(stem_root.glob("*.wav")):
        for stem_name in stem_names:
            if wav_path.stem.endswith(f"_{stem_name}"):
                resolved.setdefault(stem_name, wav_path)
                break
    return resolved


def prepare_audio_for_stem_separation(
    audio_path: str | Path,
    *,
    temp_dir: str | Path,
) -> Path:
    import librosa
    import numpy as np
    import soundfile as sf

    audio_file = Path(audio_path)
    waveform, sample_rate = librosa.load(str(audio_file), sr=None, mono=False)
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.T

    source_channels = int(waveform.shape[0])
    if source_channels <= 0:
        raise ValueError(f"Audio file has no channels: {audio_file}")
    if source_channels == 2:
        return audio_file

    if source_channels == 1:
        waveform = np.repeat(waveform, 2, axis=0)
        channel_mode = "pseudo-stereo"
    else:
        waveform = waveform[:2]
        channel_mode = "first-two-channels"

    destination_dir = Path(temp_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = destination_dir / f"{audio_file.stem}.wav"
    sf.write(str(prepared_path), waveform.T, samplerate=int(sample_rate))
    LOGGER.info(
        "Prepared %s separation input (%d -> 2 channels): %s",
        channel_mode,
        source_channels,
        prepared_path,
    )
    return prepared_path


def resolve_stem_model_type(stem_name: str) -> str:
    normalized = stem_name.casefold()
    if "drum" in normalized:
        return "drums"
    if "bass" in normalized:
        return "bass_v2"
    if "vocal" in normalized:
        return "vocal_harmony"
    if "guitar" in normalized:
        return "guitar_v1_5"
    if "other" in normalized:
        return "other_v1_5"
    return "default"


def merge_stem_midis(
    midi_paths: Sequence[str | Path],
    output_file: str | Path,
    *,
    max_melodic_instruments: int = 15,
) -> Path:
    import pretty_midi

    source_paths = [Path(path) for path in midi_paths]
    if not source_paths:
        raise ValueError("No MIDI files to merge")
    if max_melodic_instruments <= 0:
        raise ValueError("max_melodic_instruments must be positive")

    master_midi = pretty_midi.PrettyMIDI(str(source_paths[0]))
    all_notes: dict[tuple[int, bool, str], list[Any]] = defaultdict(list)
    all_controls: dict[tuple[int, bool, str], list[Any]] = defaultdict(list)
    all_bends: dict[tuple[int, bool, str], list[Any]] = defaultdict(list)

    for path in source_paths:
        midi = pretty_midi.PrettyMIDI(str(path))
        for instrument in midi.instruments:
            key = (
                int(instrument.program),
                bool(instrument.is_drum),
                str(instrument.name),
            )
            all_notes[key].extend(
                note
                for note in instrument.notes
                if float(note.end) - float(note.start) < 15.0
            )
            all_controls[key].extend(instrument.control_changes)
            all_bends[key].extend(instrument.pitch_bends)

    melodic_keys = sorted(
        (key for key in all_notes if not key[1]),
        key=lambda key: (-len(all_notes[key]), key),
    )
    drum_keys = sorted(
        (key for key in all_notes if key[1]),
        key=lambda key: (-len(all_notes[key]), key),
    )

    if len(melodic_keys) > max_melodic_instruments:
        keep_count = max(0, max_melodic_instruments - 1)
        kept_keys = melodic_keys[:keep_count]
        overflow_keys = melodic_keys[keep_count:]
    else:
        kept_keys = melodic_keys
        overflow_keys = []

    final_instruments: list[Any] = []

    def make_instrument(key: tuple[int, bool, str]) -> Any:
        instrument = pretty_midi.Instrument(
            program=key[0],
            is_drum=key[1],
            name=key[2],
        )
        instrument.notes = all_notes[key]
        instrument.control_changes = all_controls[key]
        instrument.pitch_bends = all_bends[key]
        return instrument

    final_instruments.extend(make_instrument(key) for key in kept_keys)
    if overflow_keys:
        base_key = overflow_keys[0]
        overflow = pretty_midi.Instrument(
            program=base_key[0],
            is_drum=False,
            name="Other / Merged",
        )
        for key in overflow_keys:
            overflow.notes.extend(all_notes[key])
            overflow.control_changes.extend(all_controls[key])
            overflow.pitch_bends.extend(all_bends[key])
        final_instruments.append(overflow)
    final_instruments.extend(make_instrument(key) for key in drum_keys)

    for instrument in final_instruments:
        instrument.notes.sort(key=lambda note: (note.start, note.pitch, note.end))
        instrument.control_changes.sort(key=lambda event: event.time)
        instrument.pitch_bends.sort(key=lambda event: event.time)
    master_midi.instruments = final_instruments

    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    master_midi.write(str(destination))
    return destination


class StemTranscriptionRunner:
    def __init__(
        self,
        *,
        device: str,
        amt_checkpoint_dir: str | Path,
        separation_checkpoint: str | Path,
        velocity_checkpoint: str | Path,
        window_batch_size: int,
        max_melodic_instruments: int,
        merge_onset_ms: float,
        transcribe_drums: bool,
        predict_velocity: bool,
        strict_velocity: bool,
        force: bool,
        cleanup_stems: bool,
        amp: bool = False,
        amp_dtype: str | None = None,
        compile_model: bool = False,
        compile_velocity: bool = False,
        compile_mode: str = "default",
    ) -> None:
        self.device = resolve_device(device)
        self.amt_checkpoint_dir = Path(amt_checkpoint_dir).resolve()
        self.separation_checkpoint = Path(separation_checkpoint).resolve()
        self.velocity_checkpoint = Path(velocity_checkpoint).resolve()
        self.window_batch_size = int(window_batch_size)
        self.max_melodic_instruments = int(max_melodic_instruments)
        self.merge_onset_ms = float(merge_onset_ms)
        self.transcribe_drums = bool(transcribe_drums)
        self.predict_velocity = bool(predict_velocity)
        self.strict_velocity = bool(strict_velocity)
        self.force = bool(force)
        self.cleanup_stems = bool(cleanup_stems)
        self.amp_enabled = bool(amp and is_amp_supported(self.device))
        self.amp_dtype = resolve_amp_dtype(self.device, amp_dtype)
        self.compile_model = bool(compile_model)
        self.compile_velocity = bool(compile_velocity)
        self.compile_mode = str(compile_mode)
        self._separation_bundle: tuple[Any, Any, torch.dtype] | None = None
        self._amt_bundles: dict[str, tuple[Any, Any, Any, Any]] = {}
        self._velocity_bundle: tuple[Any, Any, Any] | None = None

    def _get_separation_bundle(self) -> tuple[Any, Any, torch.dtype]:
        if self._separation_bundle is not None:
            _, model, _ = self._separation_bundle
            model.to(self.device)
            model.eval()
            return self._separation_bundle
        try:
            from stem_splitter.inference import SeparationConfig, load_mss_model
        except ImportError as exc:
            raise ImportError(
                "stem-splitter is required. Install it with "
                "`pip install stem-splitter`."
            ) from exc

        config = SeparationConfig(
            skip_existing=not self.force,
            cache_dir=self.separation_checkpoint.parent,
            hf_filename=self.separation_checkpoint.name,
        )
        LOGGER.info("Loading stem-separation model on %s", self.device)
        model = load_mss_model(config, device=self.device)
        dtype = (
            torch.float16
            if config.use_half_precision and self.device.type == "cuda"
            else torch.float32
        )
        self._separation_bundle = (config, model, dtype)
        return self._separation_bundle

    def _get_amt_bundle(self, model_type: str) -> tuple[Any, Any, Any, Any]:
        cached = self._amt_bundles.get(model_type)
        if cached is not None:
            cached[0].to(self.device)
            cached[0].eval()
            return cached

        from instrument_agnostic_amt.amt.cli import infer as amt_infer

        checkpoint_name = amt_infer.MODEL_CHECKPOINT_FILENAMES[model_type]
        checkpoint_path = self.amt_checkpoint_dir / checkpoint_name
        checkpoint_path = amt_infer._ensure_checkpoint(
            checkpoint_path,
            model_type=model_type,
        )
        LOGGER.info("Loading AMT model %s on %s", model_type, self.device)
        model, config, settings = amt_infer._load_model_and_settings(
            checkpoint_path,
            device=self.device,
            window_ms_override=None,
            stride_ms_override=None,
            track_batch_size_override=None,
        )
        forward_model = maybe_compile_forward(
            model,
            enabled=self.compile_model,
            mode=self.compile_mode,
        )
        self._amt_bundles[model_type] = (model, forward_model, config, settings)
        return self._amt_bundles[model_type]

    def _get_velocity_bundle(self) -> tuple[Any, Any, Any]:
        if self._velocity_bundle is not None:
            self._velocity_bundle[0].to(self.device)
            self._velocity_bundle[0].eval()
        else:
            from instrument_agnostic_amt.velocity.cli.infer_velocity import (
                load_velocity_model,
            )

            LOGGER.info("Loading velocity model on %s", self.device)
            model, config = load_velocity_model(
                self.velocity_checkpoint,
                device=self.device,
            )
            forward_model = maybe_compile_forward(
                model,
                enabled=self.compile_velocity,
                mode=self.compile_mode,
            )
            self._velocity_bundle = (model, forward_model, config)
        return self._velocity_bundle

    def _separate(
        self,
        audio_path: Path,
        paths: CandidatePaths,
    ) -> dict[str, Path]:
        from stem_splitter.inference import _separate_one_file

        existing = resolve_stem_paths(
            song_path=audio_path,
            stem_dir=paths.stem_dir,
            stem_names=DEFAULT_STEM_NAMES,
        )
        if (
            not self.force
            and all(name in existing for name in DEFAULT_STEM_NAMES)
            and all(is_valid_audio_file(existing[name]) for name in DEFAULT_STEM_NAMES)
        ):
            LOGGER.info("Reusing all separated stems: %s", audio_path.name)
            return {name: existing[name] for name in DEFAULT_STEM_NAMES}

        config, model, dtype = self._get_separation_bundle()
        separation_input = prepare_audio_for_stem_separation(
            audio_path,
            temp_dir=paths.run_root / "prepared_inputs",
        )
        LOGGER.info("Separating stems: %s", audio_path.name)
        # stem-splitter skips the whole song when even one output exists. For
        # interrupted runs we must rebuild the set when one or more stems are
        # missing or invalid.
        inference_config = replace(config, skip_existing=False)
        stems = _separate_one_file(
            separation_input,
            paths.stem_dir,
            inference_config,
            model,
            self.device,
            dtype,
        )
        resolved = {str(name): Path(path) for name, path in stems.items()}
        if not resolved:
            resolved = resolve_stem_paths(
                song_path=audio_path,
                stem_dir=paths.stem_dir,
                stem_names=config.stem_names,
            )
        missing = [
            name
            for name in config.stem_names
            if name not in resolved or not is_valid_audio_file(resolved[name])
        ]
        if missing:
            raise RuntimeError(
                f"Missing or invalid separated stems for {audio_path.name}: "
                + ", ".join(missing)
            )
        return {name: resolved[name] for name in config.stem_names}

    def _transcribe_stem(
        self,
        *,
        stem_name: str,
        stem_path: Path,
        output_midi: Path,
    ) -> Path:
        if is_valid_midi_file(output_midi) and not self.force:
            LOGGER.info("Reusing stem MIDI: %s", output_midi)
            return output_midi

        from instrument_agnostic_amt.amt.cli import infer as amt_infer

        model_type = resolve_stem_model_type(stem_name)
        model, forward_model, config, settings = self._get_amt_bundle(model_type)
        allowed_ids = amt_infer.resolve_stem_instrument_class_ids(stem_name)
        allowed_ids = amt_infer.filter_supported_instrument_class_ids(
            allowed_ids,
            num_model_classes=config.num_instrument_classes,
        )
        LOGGER.info("Transcribing %s with %s", stem_name, model_type)
        waveform, _, _ = amt_infer._load_audio(
            stem_path,
            target_sample_rate=config.sample_rate,
        )
        notes, _, _ = amt_infer.run_inference(
            model=model,
            forward_model=forward_model,
            waveform=waveform.to(self.device),
            model_config=config,
            settings=settings,
            device=self.device,
            amp_enabled=self.amp_enabled,
            amp_dtype=self.amp_dtype,
            velocity=100,
            merge_gap_ms=None,
            merge_onset_ms=self.merge_onset_ms,
            silence_gate_rms_dbfs=-72,
            window_batch_size=self.window_batch_size,
            max_midi_melodic_instruments=self.max_melodic_instruments,
            disable_tqdm=True,
            max_note_seconds=15.0,
            allowed_instrument_ids=allowed_ids,
        )
        instrument_volumes = (
            None
            if self.predict_velocity
            else dict(amt_infer.DEFAULT_INSTRUMENT_VOLUMES)
        )
        midi = amt_infer._build_midi(
            notes,
            sample_rate=config.sample_rate,
            instrument_volumes=instrument_volumes,
        )
        output_midi.parent.mkdir(parents=True, exist_ok=True)
        partial_midi = output_midi.with_suffix(".partial.mid")
        partial_midi.unlink(missing_ok=True)
        midi.write(str(partial_midi))
        if not is_valid_midi_file(partial_midi):
            raise RuntimeError(f"AMT wrote an invalid MIDI: {partial_midi}")
        partial_midi.replace(output_midi)
        return output_midi

    def _apply_velocity(
        self,
        *,
        stems: Mapping[str, Path],
        stem_midis: Mapping[str, Path],
        template_midi: Path,
        output_midi: Path,
    ) -> Path:
        if is_valid_midi_file(output_midi) and not self.force:
            LOGGER.info("Reusing velocity MIDI: %s", output_midi)
            return output_midi

        from instrument_agnostic_amt.velocity.cli.infer_velocity import (
            predict_velocity_for_stem_midis,
        )

        model, forward_model, config = self._get_velocity_bundle()
        partial_midi = output_midi.with_suffix(".partial.mid")
        partial_midi.unlink(missing_ok=True)
        generated = Path(
            predict_velocity_for_stem_midis(
                stem_midis=stem_midis,
                stem_audios=stems,
                output_midi_path=partial_midi,
                template_midi_path=template_midi,
                checkpoint_path=self.velocity_checkpoint,
                device=self.device,
                window_seconds=8.0,
                max_melodic_instruments=self.max_melodic_instruments,
                disable_tqdm=True,
                compile_velocity=self.compile_velocity,
                compile_mode=self.compile_mode,
                preloaded_model=model,
                preloaded_forward=forward_model,
                preloaded_config=config,
            )
        )
        if generated != partial_midi:
            raise RuntimeError(
                f"Velocity inference wrote an unexpected path: {generated}"
            )
        if not is_valid_midi_file(partial_midi):
            raise RuntimeError(
                f"Velocity inference wrote an invalid MIDI: {partial_midi}"
            )
        partial_midi.replace(output_midi)
        return output_midi

    def run_song(
        self,
        audio_path: str | Path,
        *,
        output_root: str | Path,
    ) -> Path:
        audio_file = Path(audio_path)
        paths = candidate_paths(output_root, audio_file.stem)
        merged_midi = paths.merged_dir / f"{audio_file.stem}.mid"
        velocity_midi = paths.merged_dir / f"{audio_file.stem}_velocity.mid"
        expected_result = velocity_midi if self.predict_velocity else merged_midi
        if not self.force and is_valid_midi_file(expected_result):
            LOGGER.info("Reusing completed source MIDI: %s", expected_result)
            return expected_result

        for directory in (
            paths.stem_dir,
            paths.stem_midi_dir,
            paths.merged_dir,
            paths.prediction_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        selected_stems = [
            name
            for name in DEFAULT_STEM_NAMES
            if self.transcribe_drums or "drum" not in name.casefold()
        ]
        expected_stem_midis = {
            name: paths.stem_midi_dir / f"{audio_file.stem}_{name}.mid"
            for name in selected_stems
        }
        merged_ready = not self.force and is_valid_midi_file(merged_midi)
        velocity_ready = not self.force and is_valid_midi_file(velocity_midi)
        need_velocity = self.predict_velocity and not velocity_ready
        need_stem_midis = not merged_ready or need_velocity

        stems: dict[str, Path] = {}
        stem_midis: dict[str, Path] = {}
        if need_stem_midis:
            stem_midis = {
                name: path
                for name, path in expected_stem_midis.items()
                if not self.force and is_valid_midi_file(path)
            }
            missing_stem_midis = [
                name for name in selected_stems if name not in stem_midis
            ]
            if missing_stem_midis or need_velocity:
                stems = self._separate(audio_file, paths)
            for stem_name in missing_stem_midis:
                stem_path = stems.get(stem_name)
                if stem_path is None:
                    raise RuntimeError(
                        f"Separated stem is missing for {stem_name}: {audio_file}"
                    )
                stem_midis[stem_name] = self._transcribe_stem(
                    stem_name=stem_name,
                    stem_path=stem_path,
                    output_midi=expected_stem_midis[stem_name],
                )

        if not merged_ready:
            missing_stem_midis = [
                name
                for name in selected_stems
                if name not in stem_midis or not is_valid_midi_file(stem_midis[name])
            ]
            if missing_stem_midis:
                raise RuntimeError(
                    f"Missing or invalid stem MIDIs for {audio_file.name}: "
                    + ", ".join(missing_stem_midis)
                )
            partial_merged = merged_midi.with_suffix(".partial.mid")
            partial_merged.unlink(missing_ok=True)
            merge_stem_midis(
                [stem_midis[name] for name in selected_stems],
                partial_merged,
                max_melodic_instruments=self.max_melodic_instruments,
            )
            if not is_valid_midi_file(partial_merged):
                raise RuntimeError(f"Merge wrote an invalid MIDI: {partial_merged}")
            partial_merged.replace(merged_midi)
        else:
            LOGGER.info("Reusing merged MIDI: %s", merged_midi)

        result_midi = merged_midi
        if self.predict_velocity:
            try:
                result_midi = self._apply_velocity(
                    stems=stems,
                    stem_midis=stem_midis,
                    template_midi=merged_midi,
                    output_midi=velocity_midi,
                )
            except Exception:
                if self.strict_velocity:
                    raise
                LOGGER.exception(
                    "Velocity prediction failed for %s; using fixed velocities",
                    audio_file.name,
                )

        if self.cleanup_stems:
            shutil.rmtree(paths.stem_dir, ignore_errors=True)
        return result_midi

    def park(self) -> None:
        """MIDI-frame推論の前に音声モデルをアクセラレータから退避する。"""
        if self.device.type not in {"cuda", "mps"}:
            return
        if self._separation_bundle is not None:
            self._separation_bundle[1].to("cpu")
        for model, _, _, _ in self._amt_bundles.values():
            model.to("cpu")
        if self._velocity_bundle is not None:
            self._velocity_bundle[0].to("cpu")
        gc.collect()
        empty_device_cache(self.device)

    def release(self) -> None:
        self._separation_bundle = None
        self._amt_bundles.clear()
        self._velocity_bundle = None
        gc.collect()
        empty_device_cache(self.device)


def build_midi_frame_infer_command(
    *,
    checkpoint: str | Path,
    midi_path: str | Path,
    prediction_json_path: str | Path,
    candidate_midi_path: str | Path,
    quality_json: str | Path,
    device: str,
    beat_decode_mode: str = "grid",
    chord_boundary_threshold: float = 0.5,
    key_boundary_threshold: float = 0.5,
    key_boundary_js_weight: float = 0.0,
    extra_args: Sequence[str] = (),
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "instrument_agnostic_amt.beat_chord.cli.infer",
        "--checkpoint",
        str(Path(checkpoint).resolve()),
        "--midi_path",
        str(Path(midi_path).resolve()),
        "--output_path",
        str(Path(prediction_json_path).resolve()),
        "--beat_mapped_midi_path",
        str(Path(candidate_midi_path).resolve()),
        "--quality_json",
        str(Path(quality_json).resolve()),
        "--device",
        str(device),
        "--beat_decode_mode",
        str(beat_decode_mode),
        "--chord_boundary_threshold",
        str(float(chord_boundary_threshold)),
        "--key_boundary_threshold",
        str(float(key_boundary_threshold)),
        "--key_boundary_js_weight",
        str(float(key_boundary_js_weight)),
    ]
    command.extend(str(arg) for arg in extra_args)
    return command


def write_batch_summary(
    *,
    output_root: str | Path,
    input_dir: str | Path,
    beat_chord_checkpoint: str | Path,
    results: Sequence[CandidateResult],
) -> Path:
    destination = Path(output_root).resolve() / "batch_summary.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_dir": str(Path(input_dir).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "beat_chord_checkpoint": str(Path(beat_chord_checkpoint).resolve()),
        "songs": [asdict(result) for result in results],
    }
    temporary = destination.with_suffix(".json.tmp")
    with open(temporary, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary.replace(destination)
    return destination


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-create uncorrected key-only candidate MIDIs from source audio: "
            "stem separation -> per-stem AMT -> merge/velocity -> beat/chord/key."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16"), default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-velocity", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="default",
    )
    parser.add_argument(
        "--amt-checkpoint-dir",
        type=Path,
        default=DEFAULT_AMT_CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--separation-checkpoint",
        type=Path,
        default=DEFAULT_SEPARATION_CHECKPOINT,
    )
    parser.add_argument(
        "--velocity-checkpoint",
        type=Path,
        default=DEFAULT_VELOCITY_CHECKPOINT,
    )
    parser.add_argument("--beat-chord-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--beat-chord-checkpoint-dir",
        type=Path,
        default=DEFAULT_BEAT_CHORD_CHECKPOINT_DIR,
    )
    parser.add_argument("--quality-json", type=Path, default=DEFAULT_QUALITY_JSON)
    parser.add_argument("--window-batch-size", type=int, default=4)
    parser.add_argument("--max-midi-melodic-instruments", type=int, default=15)
    parser.add_argument("--merge-onset-ms", type=float, default=50.0)
    parser.add_argument("--skip-drums", action="store_true")
    parser.add_argument("--skip-velocity", action="store_true")
    parser.add_argument("--strict-velocity", action="store_true")
    parser.add_argument("--cleanup-stems", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--beat-decode-mode",
        choices=("grid", "peaks"),
        default="grid",
    )
    parser.add_argument("--chord-boundary-threshold", type=float, default=0.5)
    parser.add_argument("--key-boundary-threshold", type=float, default=0.5)
    parser.add_argument("--key-boundary-js-weight", type=float, default=0.0)
    parser.add_argument(
        "--midi-frame-extra-arg",
        action="append",
        default=[],
        help=(
            "Extra argument forwarded to midi_frame_infer. Repeat for each token; "
            "use --midi-frame-extra-arg=--name for option tokens."
        ),
    )
    return parser.parse_args(argv)


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.window_batch_size <= 0:
        raise ValueError("--window-batch-size must be positive")
    if args.max_midi_melodic_instruments <= 0:
        raise ValueError("--max-midi-melodic-instruments must be positive")
    if args.merge_onset_ms < 0.0:
        raise ValueError("--merge-onset-ms must be non-negative")
    if not 0.0 <= args.chord_boundary_threshold <= 1.0:
        raise ValueError("--chord-boundary-threshold must be between 0 and 1")
    if not 0.0 <= args.key_boundary_threshold <= 1.0:
        raise ValueError("--key-boundary-threshold must be between 0 and 1")
    if not 0.0 <= args.key_boundary_js_weight <= 1.0:
        raise ValueError("--key-boundary-js-weight must be between 0 and 1")
    if not args.quality_json.is_file():
        raise FileNotFoundError(f"quality.json not found: {args.quality_json}")


def run_batch(args: argparse.Namespace) -> list[CandidateResult]:
    _validate_arguments(args)
    audio_paths = discover_audio_files(args.input_dir, recursive=args.recursive)
    beat_chord_checkpoint = resolve_beat_chord_checkpoint(
        args.beat_chord_checkpoint,
        checkpoint_dir=args.beat_chord_checkpoint_dir,
    )
    output_root = args.output_dir.resolve()
    final_midi_dir = output_root / "midis"

    results = [
        CandidateResult(
            audio_path=str(audio_path),
            song_name=audio_path.stem,
            run_root=str(candidate_paths(output_root, audio_path.stem).run_root),
        )
        for audio_path in audio_paths
    ]
    LOGGER.info("Found %d source audio files", len(audio_paths))
    LOGGER.info("Beat/chord checkpoint: %s", beat_chord_checkpoint)

    if args.dry_run:
        velocity_suffix = "_velocity" if not args.skip_velocity else ""
        for result in results:
            result.transcription_status = "planned"
            result.prediction_status = "planned"
            result.source_midi_path = str(
                Path(result.run_root)
                / "merged"
                / f"{result.song_name}{velocity_suffix}.mid"
            )
            result.candidate_midi_path = str(
                final_midi_dir / f"{result.song_name}{velocity_suffix}.beat_mapped.mid"
            )
            LOGGER.info(
                "[dry-run] %s -> %s",
                result.audio_path,
                result.candidate_midi_path,
            )
        return results

    output_root.mkdir(parents=True, exist_ok=True)
    final_midi_dir.mkdir(parents=True, exist_ok=True)
    runner = StemTranscriptionRunner(
        device=args.device,
        amt_checkpoint_dir=args.amt_checkpoint_dir,
        separation_checkpoint=args.separation_checkpoint,
        velocity_checkpoint=args.velocity_checkpoint,
        window_batch_size=args.window_batch_size,
        max_melodic_instruments=args.max_midi_melodic_instruments,
        merge_onset_ms=args.merge_onset_ms,
        transcribe_drums=not args.skip_drums,
        predict_velocity=not args.skip_velocity,
        strict_velocity=args.strict_velocity,
        force=args.force,
        cleanup_stems=args.cleanup_stems,
        amp=args.amp,
        amp_dtype=args.amp_dtype,
        compile_model=args.compile,
        compile_velocity=args.compile_velocity,
        compile_mode=args.compile_mode,
    )

    repository_root = Path(__file__).resolve().parents[2]
    for result in results:
        try:
            source_midi = runner.run_song(
                result.audio_path,
                output_root=output_root,
            )
            result.source_midi_path = str(source_midi)
            result.transcription_status = "completed"
        except Exception as exc:
            result.transcription_status = "failed"
            result.prediction_status = "skipped"
            result.error_phase = "stem_transcription"
            result.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("Stem transcription failed: %s", result.audio_path)
            if args.fail_fast:
                runner.release()
                write_batch_summary(
                    output_root=output_root,
                    input_dir=args.input_dir,
                    beat_chord_checkpoint=beat_chord_checkpoint,
                    results=results,
                )
                raise
            write_batch_summary(
                output_root=output_root,
                input_dir=args.input_dir,
                beat_chord_checkpoint=beat_chord_checkpoint,
                results=results,
            )
            continue

        source_midi = Path(str(result.source_midi_path))
        candidate_midi = final_midi_dir / f"{source_midi.stem}.beat_mapped.mid"
        prediction_json = (
            Path(result.run_root) / "prediction" / f"{source_midi.stem}.prediction.json"
        )
        result.candidate_midi_path = str(candidate_midi)
        result.prediction_json_path = str(prediction_json)
        if (
            prediction_outputs_complete(
                candidate_midi_path=candidate_midi,
                prediction_json_path=prediction_json,
            )
            and not args.force
        ):
            result.prediction_status = "reused"
            LOGGER.info("Reusing beat/chord/key prediction: %s", result.song_name)
            write_batch_summary(
                output_root=output_root,
                input_dir=args.input_dir,
                beat_chord_checkpoint=beat_chord_checkpoint,
                results=results,
            )
            continue

        candidate_midi.parent.mkdir(parents=True, exist_ok=True)
        prediction_json.parent.mkdir(parents=True, exist_ok=True)
        partial_candidate_midi = candidate_midi.with_suffix(".partial.mid")
        partial_prediction_json = prediction_json.with_suffix(".partial.json")
        partial_candidate_midi.unlink(missing_ok=True)
        partial_prediction_json.unlink(missing_ok=True)
        command = build_midi_frame_infer_command(
            checkpoint=beat_chord_checkpoint,
            midi_path=source_midi,
            prediction_json_path=partial_prediction_json,
            candidate_midi_path=partial_candidate_midi,
            quality_json=args.quality_json,
            device=args.device,
            beat_decode_mode=args.beat_decode_mode,
            chord_boundary_threshold=args.chord_boundary_threshold,
            key_boundary_threshold=args.key_boundary_threshold,
            key_boundary_js_weight=args.key_boundary_js_weight,
            extra_args=args.midi_frame_extra_arg,
        )
        try:
            # Complete every song before moving to the next one. Cached audio
            # models are kept in CPU RAM so midi_frame_infer can use the GPU.
            runner.park()
            LOGGER.info("Predicting beat/chord/key: %s", result.song_name)
            subprocess.run(command, cwd=repository_root, check=True)
            if not prediction_outputs_complete(
                candidate_midi_path=partial_candidate_midi,
                prediction_json_path=partial_prediction_json,
            ):
                raise RuntimeError(
                    "midi_frame_infer did not produce a valid prediction JSON "
                    "and beat-mapped MIDI"
                )
            partial_candidate_midi.replace(candidate_midi)
            partial_prediction_json.replace(prediction_json)
            result.prediction_status = "completed"
        except Exception as exc:
            result.prediction_status = "failed"
            result.error_phase = "midi_frame_infer"
            result.error = f"{type(exc).__name__}: {exc}"
            LOGGER.exception("midi_frame_infer failed: %s", result.song_name)
            if args.fail_fast:
                write_batch_summary(
                    output_root=output_root,
                    input_dir=args.input_dir,
                    beat_chord_checkpoint=beat_chord_checkpoint,
                    results=results,
                )
                raise
        write_batch_summary(
            output_root=output_root,
            input_dir=args.input_dir,
            beat_chord_checkpoint=beat_chord_checkpoint,
            results=results,
        )
    runner.release()
    return results


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    args = parse_arguments(argv)
    results = run_batch(args)
    if args.dry_run:
        return 0

    failed = [
        result
        for result in results
        if result.transcription_status == "failed"
        or result.prediction_status == "failed"
    ]
    completed = [
        result
        for result in results
        if result.prediction_status in {"completed", "reused"}
    ]
    LOGGER.info(
        "Batch finished: %d completed, %d failed, %d total",
        len(completed),
        len(failed),
        len(results),
    )
    LOGGER.info("Candidate MIDIs: %s", args.output_dir.resolve() / "midis")
    return 1 if failed else 0


__all__ = [
    "AUDIO_EXTENSIONS",
    "CandidatePaths",
    "CandidateResult",
    "DEFAULT_STEM_NAMES",
    "StemTranscriptionRunner",
    "build_midi_frame_infer_command",
    "candidate_paths",
    "discover_audio_files",
    "is_valid_audio_file",
    "is_valid_midi_file",
    "is_valid_prediction_json",
    "main",
    "merge_stem_midis",
    "prediction_outputs_complete",
    "resolve_beat_chord_checkpoint",
    "resolve_stem_model_type",
    "run_batch",
]
