"""Stem separation and transcription pipeline entrypoint."""

from __future__ import annotations

import shutil
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf
import torch
from stem_splitter.inference import SeparationConfig, _separate_one_file, load_mss_model

import infer
from infer_beat_chord import predict_beat_chord_for_midi
from infer_instrument_refinement import (
    ensure_refinement_checkpoint,
    refine_midi_instruments,
)
from infer_velocity import predict_velocity_for_stem_midis
from instrument_agnostic_amt.whisper_lyrics import add_whisper_lyrics_to_vocals_midi
from instrument_agnostic_amt.instrument_refinement.data.labels import (
    inference_stem_group,
)
from instrument_agnostic_amt.instrument_refinement.modeling.checkpoints import (
    load_refinement_model,
)
from instrument_agnostic_amt.taxonomy.instrument_classes import INSTRUMENT_CLASSES

# セッション中にモデルを使い回して、再実行時の待ち時間を減らす。
STEM_PIPELINE_CACHE: dict[tuple[str, ...], tuple[object, ...]] = {}

# Instrument Refinement を適用しないステム。
# drums:  候補がドラムだけになり、ドラムを除外すると候補が空になるため refine できない。
# vocals: melody / vocal_harmony / choir の違いは音色ではなく役割（主旋律か副次声部か）で、
#         音色の埋め込みで判断する refinement モデルでは原理的に判別できない。実際に
#         リード全体がコーラス側へ倒れる例が出たため、AMT の判定をそのまま採用する。
REFINEMENT_EXCLUDED_STEM_GROUPS = ("drums", "vocals")


def _current_model_device(model: torch.nn.Module) -> torch.device:
    """モデルの先頭パラメータから現在のデバイスを取得する。"""
    first_param = next(model.parameters(), None)
    if first_param is None:
        return torch.device("cpu")
    return first_param.device


def _move_model(model: torch.nn.Module, target: torch.device | str) -> None:
    """モデルを target へ移動し、CUDA 側の不要な確保を解放する。"""
    target_device = torch.device(target)
    current_device = _current_model_device(model)
    if current_device.type != target_device.type or current_device.index != target_device.index:
        model.to(target_device)
    if target_device.type == "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def merge_midis_logic(
    midi_paths: list[Path | str],
    output_file: Path | str,
    max_melodic_instruments: int = 15,
) -> None:
    """ステムごとの MIDI を 1 本にまとめる。"""
    if not midi_paths:
        raise ValueError("No MIDI files to merge")

    output_path = Path(output_file)
    master_midi = pretty_midi.PrettyMIDI(str(midi_paths[0]))

    all_notes: dict[tuple[int, bool, str], list[pretty_midi.Note]] = defaultdict(list)
    all_control_changes: dict[tuple[int, bool, str], list[pretty_midi.ControlChange]] = defaultdict(list)
    all_pitch_bends: dict[tuple[int, bool, str], list[pretty_midi.PitchBend]] = defaultdict(list)
    instrument_names: dict[tuple[int, bool, str], str] = {}

    for path in midi_paths:
        midi_obj = pretty_midi.PrettyMIDI(str(path))
        for instrument in midi_obj.instruments:
            key = (instrument.program, instrument.is_drum, instrument.name)
            filtered_notes = [note for note in instrument.notes if (note.end - note.start) < 15.0]
            all_notes[key].extend(filtered_notes)
            all_control_changes[key].extend(instrument.control_changes)
            all_pitch_bends[key].extend(instrument.pitch_bends)
            if key not in instrument_names:
                instrument_names[key] = instrument.name

    melodic_keys = [k for k in all_notes.keys() if not k[1]]
    drum_keys = [k for k in all_notes.keys() if k[1]]
    melodic_keys.sort(key=lambda key_tuple: len(all_notes[key_tuple]), reverse=True)

    final_instruments: list[pretty_midi.Instrument] = []
    if len(melodic_keys) > max_melodic_instruments:
        kept_keys = melodic_keys[: max_melodic_instruments - 1]
        overflow_keys = melodic_keys[max_melodic_instruments - 1 :]
        for key in kept_keys:
            instrument = pretty_midi.Instrument(
                program=key[0],
                is_drum=key[1],
                name=instrument_names[key],
            )
            instrument.notes = all_notes[key]
            instrument.control_changes = all_control_changes[key]
            instrument.pitch_bends = all_pitch_bends[key]
            final_instruments.append(instrument)

        base_key = overflow_keys[0]
        overflow_instrument = pretty_midi.Instrument(
            program=base_key[0],
            is_drum=base_key[1],
            name="Other / Merged",
        )
        for key in overflow_keys:
            overflow_instrument.notes.extend(all_notes[key])
            overflow_instrument.control_changes.extend(all_control_changes[key])
            overflow_instrument.pitch_bends.extend(all_pitch_bends[key])
        final_instruments.append(overflow_instrument)
    else:
        for key in melodic_keys:
            instrument = pretty_midi.Instrument(
                program=key[0],
                is_drum=key[1],
                name=instrument_names[key],
            )
            instrument.notes = all_notes[key]
            instrument.control_changes = all_control_changes[key]
            instrument.pitch_bends = all_pitch_bends[key]
            final_instruments.append(instrument)

    for key in drum_keys:
        instrument = pretty_midi.Instrument(
            program=key[0],
            is_drum=key[1],
            name=instrument_names[key],
        )
        instrument.notes = all_notes[key]
        instrument.control_changes = all_control_changes[key]
        instrument.pitch_bends = all_pitch_bends[key]
        final_instruments.append(instrument)

    master_midi.instruments = final_instruments
    for instrument in master_midi.instruments:
        instrument.notes.sort(key=lambda note: note.start)
        instrument.control_changes.sort(key=lambda change: change.time)
        instrument.pitch_bends.sort(key=lambda pitch_bend: pitch_bend.time)

    # 收集 lyrics（来自 vocals stem 的歌词）
    merged_lyrics: list[tuple[str, float]] = []
    for path in midi_paths:
        lmidi = pretty_midi.PrettyMIDI(str(path))
        for l in lmidi.lyrics:
            merged_lyrics.append((l.text, l.time))
    if merged_lyrics:
        merged_lyrics.sort(key=lambda x: x[1])
        from pretty_midi import Lyric
        master_midi.lyrics = [Lyric(text=t, time=s) for t, s in merged_lyrics]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    master_midi.write(str(output_path))


def resolve_stem_paths(
    *,
    song_path: Path | str,
    stem_dir: Path | str,
    stem_names: list[str],
) -> dict[str, Path]:
    """既存 stem 出力の標準パスを組み立て、存在するものだけ返す。"""
    song_file = Path(song_path)
    song_id = song_file.stem
    stem_root = Path(stem_dir) / song_id
    resolved: dict[str, Path] = {}

    for stem_name in stem_names:
        expected_path = stem_root / f"{song_id}_{stem_name}.wav"
        if expected_path.exists():
            resolved[stem_name] = expected_path

    if resolved:
        return resolved

    if not stem_root.exists():
        return resolved

    for wav_path in sorted(stem_root.glob("*.wav")):
        stem_key = wav_path.stem
        for stem_name in stem_names:
            if stem_key.endswith(f"_{stem_name}"):
                resolved.setdefault(stem_name, wav_path)
                break

    return resolved


def _load_audio_channels_first(audio_file: Path) -> tuple[np.ndarray, int]:
    """音声を (チャンネル, サンプル) の配列とサンプルレートで読む。

    wav/flac/ogg/mp3 は soundfile で読み、それが扱えない形式（m4a など）だけ
    librosa へフォールバックする。librosa は resampler の都合で環境によって
    読み込みに失敗することがあるため、対応形式では通らないようにしている。
    """
    try:
        data, sample_rate = sf.read(str(audio_file), dtype="float32", always_2d=True)
        return data.T, int(sample_rate)
    except sf.SoundFileError:
        waveform, sample_rate = librosa.load(str(audio_file), sr=None, mono=False)
        if waveform.ndim == 1:
            waveform = waveform[None, :]
        elif waveform.ndim == 2 and waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T
        return np.asarray(waveform, dtype=np.float32), int(sample_rate)


def prepare_audio_for_stem_separation(
    audio_path: Path | str,
    *,
    temp_dir: Path | str,
) -> Path:
    """Stem Separation 向けに入力音声を一時的に 2ch WAV へそろえる。"""
    audio_file = Path(audio_path)
    waveform, sample_rate = _load_audio_channels_first(audio_file)

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

    temp_directory = Path(temp_dir)
    temp_directory.mkdir(parents=True, exist_ok=True)
    prepared_path = temp_directory / f"{audio_file.stem}.wav"
    sf.write(str(prepared_path), waveform.T, samplerate=int(sample_rate))
    print(
        f"Prepared {channel_mode} input for stem separation: "
        f"source_channels={source_channels} -> {prepared_path.name}"
    )
    return prepared_path


def get_stem_pipeline_models(
    checkpoint_path: Path | str | None = None,
    device_preference: torch.device | str | None = None,
    model_type: str = "default",
    low_vram_mode: bool = False,
    no_half: bool = False,
) -> dict[str, object]:
    """AMT とステム分離モデルを読み込み、セッション中は再利用する。

    low_vram_mode=True の場合、モデルは CPU メモリに常駐させ、推論時にだけ
    対象モデルを推論デバイス（通常 CUDA）へ移動する。これにより GPU 上に
    置かれるモデルは常に 1 つだけになる。
    """
    if device_preference is None:
        compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        compute_device = torch.device(device_preference)

    # 低显存模式：模型常驻 CPU 内存（RAM），仅在推理时搬运到 compute_device。
    storage_device = torch.device("cpu") if low_vram_mode else compute_device

    resolved_checkpoint = infer._ensure_checkpoint(
        None if checkpoint_path in (None, "", "DEFAULT") else Path(checkpoint_path),
        model_type=model_type,
    )
    amt_cache_key = ("amt", str(resolved_checkpoint.resolve()), storage_device.type)
    if low_vram_mode:
        sep_variant = "fp32_half_chunk" if no_half else "fp16"
    else:
        sep_variant = "fp32"
    sep_cache_key = ("sep", storage_device.type, sep_variant)

    if sep_cache_key not in STEM_PIPELINE_CACHE:
        print(f"Loading Separation model on {storage_device} ...")
        if low_vram_mode:
            if no_half:
                # fp16 非対応/低速な GPU 向け: fp32 のまま、チャンクを半分にして VRAM を抑える。
                print("[LowVRAM] Separation uses fp32 with half chunks ...")
                sep_config = SeparationConfig(
                    skip_existing=True,
                    use_half_precision=False,
                    chunk_size=294_400,
                    hop_size=147_200,
                )
            else:
                # 低显存モードでは分離を fp16 autocast で実行し、ピーク VRAM を抑える。
                print("[LowVRAM] Separation uses fp16 autocast ...")
                sep_config = SeparationConfig(
                    skip_existing=True,
                    use_half_precision=True,
                )
        else:
            sep_config = SeparationConfig(skip_existing=True)
        sep_model = load_mss_model(sep_config, device=storage_device)
        sep_dtype = (
            torch.float16
            if sep_config.use_half_precision and compute_device.type == "cuda"
            else torch.float32
        )
        STEM_PIPELINE_CACHE[sep_cache_key] = (sep_config, sep_model, sep_dtype)
    else:
        sep_config, sep_model, sep_dtype = STEM_PIPELINE_CACHE[sep_cache_key]

    if amt_cache_key not in STEM_PIPELINE_CACHE:
        print(f"Loading AMT model ({model_type}) on {storage_device} ...")
        amt_model, amt_config, amt_settings = infer._load_model_and_settings(
            resolved_checkpoint,
            device=storage_device,
            window_ms_override=None,
            stride_ms_override=None,
            track_batch_size_override=None,
        )
        STEM_PIPELINE_CACHE[amt_cache_key] = (amt_model, amt_config, amt_settings)
    else:
        print(f"Reusing cached AMT model ({model_type}) on {storage_device} ...")
        amt_model, amt_config, amt_settings = STEM_PIPELINE_CACHE[amt_cache_key]

    return {
        "device": compute_device,
        "checkpoint": resolved_checkpoint,
        "amt_model": amt_model,
        "amt_config": amt_config,
        "amt_settings": amt_settings,
        "sep_config": sep_config,
        "sep_model": sep_model,
        "sep_dtype": sep_dtype,
    }


def get_refinement_models(
    checkpoint_path: Path | str | None = None,
    device_preference: torch.device | str | None = None,
) -> dict[str, object]:
    """Instrument Refinement モデルを読み込み、セッション中は再利用する。"""
    if device_preference is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_preference)

    resolved_checkpoint = ensure_refinement_checkpoint(checkpoint_path)
    cache_key = ("refine", str(resolved_checkpoint), device.type)

    if cache_key not in STEM_PIPELINE_CACHE:
        print(f"Loading Instrument Refinement model on {device} ...")
        refinement_model, refinement_config, _ = load_refinement_model(
            resolved_checkpoint,
            device=device,
        )
        STEM_PIPELINE_CACHE[cache_key] = (refinement_model, refinement_config)
    else:
        print(f"Reusing cached Instrument Refinement model on {device} ...")
        refinement_model, refinement_config = STEM_PIPELINE_CACHE[cache_key]

    return {
        "device": device,
        "checkpoint": resolved_checkpoint,
        "refinement_model": refinement_model,
        "refinement_config": refinement_config,
    }


def refine_stem_instrument_midis(
    stem_midis: Mapping[str, Path | str],
    stem_audios: Mapping[str, Path | str],
    *,
    output_dir: Path | str,
    refinement_model: object,
    refinement_config: object,
    device: torch.device | str | None = None,
    stem_names: Sequence[str] | None = None,
    mode: str = "cluster",
    window_seconds: float = 8.0,
    stride_seconds: float = 4.0,
    disable_tqdm: bool = True,
) -> dict[str, Path]:
    """ステム音声を使って各ステム MIDI の楽器ラベルを付け直す。

    REFINEMENT_EXCLUDED_STEM_GROUPS のステムは stem_names で明示しても常にスキップする。
    再ラベリングできたステムだけを stem 名 -> 新しい MIDI パスで返す。
    """
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    selected_stem_names = (
        {str(name).lower() for name in stem_names} if stem_names is not None else None
    )
    refined_midi_paths: dict[str, Path] = {}

    for stem_name, midi_path in sorted(stem_midis.items()):
        if (
            selected_stem_names is not None
            and stem_name.lower() not in selected_stem_names
        ):
            continue
        stem_group = inference_stem_group(stem_name)
        if stem_group in REFINEMENT_EXCLUDED_STEM_GROUPS:
            print(f"Skipping instrument refinement for {stem_group} stem: {stem_name}")
            continue

        stem_audio_path = stem_audios.get(stem_name)
        if stem_audio_path is None or not Path(stem_audio_path).exists():
            print(f"Skipping instrument refinement for {stem_name}: stem audio not found")
            continue

        source_midi = Path(midi_path)
        refined_midi = output_directory / f"{source_midi.stem}_refined.mid"
        report = refine_midi_instruments(
            stem_audio_path,
            source_midi,
            output_midi_path=refined_midi,
            stem_name=stem_name,
            device=device,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            mode=mode,
            disable_tqdm=disable_tqdm,
            preloaded_model=refinement_model,
            preloaded_config=refinement_config,
        )
        refined_midi_paths[stem_name] = refined_midi

        refined_class_names = sorted(
            {
                cluster["candidates"][0]["class_name"]
                for cluster in report["clusters"]
                if cluster["candidates"]
            }
        )
        print(
            f"Refined stem: {stem_name} (notes={report['note_count']}, "
            f"clusters={report['cluster_count']}, "
            f"instruments={','.join(refined_class_names) or 'none'})"
        )

    return refined_midi_paths


def resolve_stem_model_type(stem_name: str) -> str:
    """ステム名に対応する AMT モデルタイプを選択する。"""
    stem_name_lower = stem_name.lower()
    if "drum" in stem_name_lower:
        return "drums"
    if "bass" in stem_name_lower:
        return "bass_v2"
    if "vocal" in stem_name_lower:
        return "vocal_harmony"
    if "guitar" in stem_name_lower:
        return "guitar_v1_5"
    if "other" in stem_name_lower:
        return "other"
    return "default"


def run_stem_separated_transcription(
    audio_path: Path | str,
    *,
    checkpoint_path: Path | str | None = None,
    output_root: Path | str = "colab_outputs",
    window_batch_size: int = 4,
    max_midi_melodic_instruments: int = 15,
    transcribe_drum_stems: bool = True,
    cleanup_separated_stems: bool = False,
    refine_instruments: bool = True,
    refinement_checkpoint_path: Path | str | None = None,
    refinement_mode: str = "cluster",
    refinement_stem_names: Sequence[str] | None = ["piano","bass","guitar"],
    predict_velocity: bool = True,
    velocity_checkpoint_path: Path | str | None = None,
    predict_beat_chord: bool = True,
    beat_chord_checkpoint_path: Path | str | None = None,
    merge_onset_ms: float = 50.0,
    transcribe_lyrics: bool = True,
    lyrics_confidence_threshold: float = 0.65,
    whisper_lyrics_language: str | None = None,
    low_vram_mode: bool = False,
    no_half: bool = False,
) -> dict[str, object]:
    """ステム分離 -> 各ステム採譜 -> MIDI マージ -> Velocity予測 -> Beat/Chord予測を一括実行する。

    low_vram_mode=True にすると、モデルをすべて CPU メモリに常駐させ、
    各ステムの採譜を行う直前に対象 AMT モデルだけを GPU へ移動し、
    終わったらすぐ CPU へ戻す。GPU に置かれるモデルは常に 1 つだけになる。
    """
    """ステム分離 -> 各ステム採譜 -> 楽器再ラベリング -> MIDI マージ -> Velocity予測 -> Beat/Chord予測を一括実行する。"""
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    bundle = get_stem_pipeline_models(
        checkpoint_path=checkpoint_path,
        model_type="default",
        low_vram_mode=low_vram_mode,
        no_half=no_half,
    )
    device = bundle["device"]
    sep_config = bundle["sep_config"]
    sep_model = bundle["sep_model"]
    sep_dtype = bundle["sep_dtype"]

    amt_bundles: dict[str, dict[str, object]] = {"default": bundle}

    def get_amt_bundle(model_type_key: str) -> dict[str, object]:
        if model_type_key not in amt_bundles:
            amt_bundles[model_type_key] = get_stem_pipeline_models(
                checkpoint_path=checkpoint_path,
                model_type=model_type_key,
                low_vram_mode=low_vram_mode,
                no_half=no_half,
            )
        return amt_bundles[model_type_key]

    # 1. 出力先を曲ごとに分ける。
    run_root = Path(output_root) / audio_file.stem
    stem_dir = run_root / "stems"
    stem_midi_dir = run_root / "stem_midis"
    refined_stem_midi_dir = run_root / "refined_stem_midis"
    merged_dir = run_root / "merged"
    for directory in (stem_dir, stem_midi_dir, merged_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 2. 元音源をステム分離する。
    separation_input = prepare_audio_for_stem_separation(
        audio_file,
        temp_dir=run_root / "prepared_inputs",
    )
    print(f"Separating stems for: {audio_file.name}")
    if low_vram_mode:
        print(f"[LowVRAM] Moving separation model to {device} ...")
        _move_model(sep_model, device)
    try:
        stems = _separate_one_file(
            separation_input,
            stem_dir,
            sep_config,
            sep_model,
            device,
            sep_dtype,
        )
    finally:
        if low_vram_mode:
            print("[LowVRAM] Moving separation model back to CPU ...")
            _move_model(sep_model, "cpu")

    # 3. 再実行時に分離が省略された場合は既存 stem のパスを復元。
    if not stems:
        stems = resolve_stem_paths(
            song_path=audio_file,
            stem_dir=stem_dir,
            stem_names=sep_config.stem_names,
        )
        if stems:
            print(f"Reusing existing stems: {sorted(stems)}")
        else:
            raise RuntimeError(f"No stems found for {audio_file.stem}")

    # 4. 各ステムを採譜。
    stem_midi_paths: dict[str, Path] = {}
    for stem_name, stem_path in sorted(stems.items()):
        if not transcribe_drum_stems and "drum" in stem_name.lower():
            print(f"Skipping drum stem: {stem_name}")
            continue

        output_midi = stem_midi_dir / f"{audio_file.stem}_{stem_name}.mid"
        model_type = resolve_stem_model_type(stem_name)
        current_bundle = get_amt_bundle(model_type)
        current_amt_model = current_bundle["amt_model"]
        current_amt_config = current_bundle["amt_config"]
        current_amt_settings = current_bundle["amt_settings"]

        allowed_instrument_ids = infer.resolve_stem_instrument_class_ids(stem_name)
        allowed_instrument_ids = infer.filter_supported_instrument_class_ids(
            allowed_instrument_ids,
            num_model_classes=current_amt_config.num_instrument_classes,
        )
        allowed_instrument_names = (
            [INSTRUMENT_CLASSES[class_id] for class_id in allowed_instrument_ids]
            if allowed_instrument_ids is not None
            else ["all"]
        )
        print(
            f"Transcribing stem: {stem_name} (model={model_type}, "
            f"instruments={','.join(allowed_instrument_names)})"
        )

        if low_vram_mode:
            print(f"[LowVRAM] Moving AMT model ({model_type}) to {device} ...")
            _move_model(current_amt_model, device)
        try:
            waveform, _, _ = infer._load_audio(
                Path(stem_path),
                target_sample_rate=current_amt_config.sample_rate,
            )
            notes, _, _ = infer.run_inference(
                model=current_amt_model,
                waveform=waveform.to(device),
                model_config=current_amt_config,
                settings=current_amt_settings,
                device=device,
                amp_enabled=False,
                amp_dtype=torch.float16 if device.type == "cuda" else torch.float32,
                velocity=100,
                merge_gap_ms=None,
                merge_onset_ms=merge_onset_ms,
                silence_gate_rms_dbfs=-72,
                window_batch_size=window_batch_size,
                max_midi_melodic_instruments=max_midi_melodic_instruments,
                disable_tqdm=True,
                max_note_seconds=15.0,
                allowed_instrument_ids=allowed_instrument_ids,
            )

            instrument_volumes = None if predict_velocity else dict(infer.DEFAULT_INSTRUMENT_VOLUMES)
            midi = infer._build_midi(
                notes,
                sample_rate=current_amt_config.sample_rate,
                instrument_volumes=instrument_volumes,
            )
            midi.write(str(output_midi))
            stem_midi_paths[stem_name]=output_midi
        finally:
            if low_vram_mode:
                print(f"[LowVRAM] Moving AMT model ({model_type}) back to CPU ...")
                _move_model(current_amt_model, "cpu")

        # ── Whisper word-level lyrics for vocals stem ──
        if "vocal" in stem_name.lower() and transcribe_lyrics:
            try:
                print(f"[WhisperLyrics] Running whisper on {stem_name} ...")
                add_whisper_lyrics_to_vocals_midi(
                    vocals_wav_path=stem_path,
                    stem_midi_path=output_midi,
                    language=whisper_lyrics_language,
                    target_track_name="vocals",
                    lyrics_confidence_threshold=lyrics_confidence_threshold
                )
            except Exception as e:
                print(f"[WhisperLyrics] WARNING: lyrics insertion failed: {e}")

    if not stem_midi_paths:
        raise RuntimeError("No stem MIDI files were generated")

    # 4.5 Instrument Refinement でステム内の楽器ラベルを付け直す。
    instruments_refined = False
    if refine_instruments:
        try:
            print("Refining stem instrument labels...")
            refinement_bundle = get_refinement_models(
                checkpoint_path=refinement_checkpoint_path,
                device_preference=device,
            )
            refined_midi_paths = refine_stem_instrument_midis(
                stem_midis=stem_midi_paths,
                stem_audios=stems,
                output_dir=refined_stem_midi_dir,
                refinement_model=refinement_bundle["refinement_model"],
                refinement_config=refinement_bundle["refinement_config"],
                device=device,
                stem_names=refinement_stem_names,
                mode=refinement_mode,
                disable_tqdm=True,
            )
            if refined_midi_paths:
                stem_midi_paths.update(refined_midi_paths)
                instruments_refined = True
                print(f"Refined stem MIDI files: {sorted(refined_midi_paths)}")
            else:
                print("Warning: No stem was eligible for instrument refinement")
        except Exception as err:
            print(f"Warning: Instrument refinement skipped due to error: {err}")

    song_midi_paths = [stem_midi_paths[stem_name] for stem_name in sorted(stem_midi_paths)]

    # 5. ステムごとの MIDI を 1 本にまとめる。
    merged_midi_path = merged_dir / f"{audio_file.stem}.mid"
    merge_midis_logic(
        song_midi_paths,
        merged_midi_path,
        max_melodic_instruments=max_midi_melodic_instruments,
    )

    # 6. Velocity予測を実行し、MIDIノートの強弱を補正する。
    if predict_velocity:
        velocity_model = None
        velocity_config = None
        try:
            print("Predicting note velocities from separated stems...")
            stem_midis_map = {
                stem_name: midi_path
                for stem_name, midi_path in stem_midi_paths.items()
                if Path(midi_path).exists()
            }
            velocity_midi_path = merged_dir / f"{audio_file.stem}_velocity.mid"
            if low_vram_mode:
                from infer_velocity import load_velocity_model

                print("[LowVRAM] Loading velocity model on CPU ...")
                velocity_model, velocity_config = load_velocity_model(
                    velocity_checkpoint_path,
                    device="cpu",
                )
                print(f"[LowVRAM] Moving velocity model to {device} ...")
                _move_model(velocity_model, device)
            predict_velocity_for_stem_midis(
                stem_midis=stem_midis_map,
                stem_audios=stems,
                output_midi_path=velocity_midi_path,
                template_midi_path=merged_midi_path,
                checkpoint_path=velocity_checkpoint_path,
                device=device,
                window_seconds=8.0,
                max_melodic_instruments=max_midi_melodic_instruments,
                disable_tqdm=True,
                preloaded_model=velocity_model,
                preloaded_config=velocity_config,
            )
            merged_midi_path = velocity_midi_path
            print("Updated merged MIDI with predicted velocities:", merged_midi_path)
        except Exception as err:
            print(f"Warning: Velocity prediction skipped due to error: {err}")
        finally:
            if low_vram_mode and velocity_model is not None:
                print("[LowVRAM] Moving velocity model back to CPU ...")
                _move_model(velocity_model, "cpu")

    # 6.5 Beat, Chord, Key 予測を実行し、ビート・コード情報を MIDI に書き込む。
    if predict_beat_chord:
        try:
            print("Predicting beat, chord, key from MIDI...")
            beat_chord_midi_path = merged_dir / f"{audio_file.stem}_beat_chord.mid"
            predict_beat_chord_for_midi(
                input_midi_path=merged_midi_path,
                output_midi_path=beat_chord_midi_path,
                checkpoint_path=beat_chord_checkpoint_path,
                device=device,
            )
            merged_midi_path = beat_chord_midi_path
            print("Updated merged MIDI with predicted beat/chord/key:", merged_midi_path)
        except Exception as err:
            print(f"Warning: Beat/chord prediction skipped due to error: {err}")
        finally:
            if low_vram_mode and device.type == "cuda":
                # beat_chord モデルは関数内でローカルに読み込まれるため、
                # 戻り後に空きメモリを解放するだけで GPU 上から消える。
                torch.cuda.empty_cache()

    # 7. 中間の分離 wav を削除。
    if cleanup_separated_stems:
        shutil.rmtree(stem_dir, ignore_errors=True)

    result = {
        "audio_path": str(audio_file),
        "output_root": str(run_root),
        "stem_dir": str(stem_dir),
        "stem_midi_dir": str(stem_midi_dir),
        "merged_midi_path": str(merged_midi_path),
        "stem_count": len(stems),
        "transcribed_stem_count": len(song_midi_paths),
        "instruments_refined": instruments_refined,
        "velocity_predicted": predict_velocity,
        "beat_chord_predicted": predict_beat_chord,
    }
    if instruments_refined:
        result["refined_stem_midi_dir"] = str(refined_stem_midi_dir)
    print("Final Merged MIDI:", merged_midi_path)
    return result
