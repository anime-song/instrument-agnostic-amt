import dataclasses
import argparse
import json
import math
from collections.abc import Mapping

# Windowsでの PosixPath アンピクル対策
# Linux環境等で保存された PosixPath を含むチェックポイントを Windows で読み込めるようにする
import pathlib
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import torch

if sys.platform == "win32":
    pathlib.PosixPath = pathlib.WindowsPath

from .. import MidiFrameBeatChordModel, MidiFrameModelConfig
from ..chord_vocabulary import (
    load_chord_quality_map_json,
    normalize_chord_quality_map,
)
from ..decoding.beat_grid import (
    BeatGridDPConfig,
    MeterGridSegment,
    decode_beats_with_meter_grid_dp,
    result_to_diagnostics,
)
from ..decoding.legacy_grid import (
    decode_beats_with_meter_grid,
    detect_peaks,
)
from ..midi_roll import MidiFrameLoader, MidiFrameLoaderConfig
from ..tempo_map_export import MeterSegmentSpec, export_tempo_mapped_midi
from ...runtime import copy_tensors_to_cpu_once, resolve_device


HF_CHECKPOINT_BASE_URL = (
    "https://huggingface.co/anime-song/instrument_agnostic_amt/resolve/main"
)
DEFAULT_BEAT_CHORD_CHECKPOINT_FILENAME = "best_beat_chord_key.pth"


def ensure_beat_chord_checkpoint(
    checkpoint_path: Path | str | None = None,
) -> Path:
    """指定された checkpoint パスを確保し、存在しない場合は Hugging Face からダウンロードする。"""
    if checkpoint_path is not None:
        target_path = Path(checkpoint_path).resolve()
    else:
        candidates = [
            Path("beat_chord_checkpoints") / DEFAULT_BEAT_CHORD_CHECKPOINT_FILENAME,
            Path("checkpoints") / DEFAULT_BEAT_CHORD_CHECKPOINT_FILENAME,
            Path("beat_chord_checkpoints") / "midi_frame" / DEFAULT_BEAT_CHORD_CHECKPOINT_FILENAME,
        ]
        target_path = candidates[0]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

    if not target_path.exists():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{HF_CHECKPOINT_BASE_URL}/{DEFAULT_BEAT_CHORD_CHECKPOINT_FILENAME}?download=true"
        print(f"Beat/chord checkpoint not found at {target_path}. Downloading from Hugging Face...")
        torch.hub.download_url_to_file(url, str(target_path))

    return target_path.resolve()


def load_beat_chord_model(
    checkpoint_path: Path | str,
    device: torch.device | str = "cpu",
    meter_classes_json: Path | str | None = None,
) -> tuple[MidiFrameBeatChordModel, MidiFrameModelConfig, dict[str, object]]:
    """チェックポイントから beat_chord モデルと設定、関連メタデータをロードする。"""
    device_obj = torch.device(device)
    checkpoint_file = Path(checkpoint_path).resolve()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Beat/chord checkpoint not found: {checkpoint_file}")

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
    model_config_dict = checkpoint["model_config"]
    model_config = MidiFrameModelConfig(**model_config_dict)

    meter_json_path = Path(meter_classes_json).resolve() if meter_classes_json else None
    meter_classes = load_meter_classes(
        checkpoint=checkpoint,
        model_config=model_config,
        meter_classes_json=meter_json_path,
    )

    quality_map = load_chord_quality_map(
        checkpoint=checkpoint,
        model_config=model_config,
        quality_json=None,
    )

    model = MidiFrameBeatChordModel(model_config).to(device_obj)
    state_dict = checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict"))
    if state_dict is None:
        state_dict = checkpoint
    has_major_grouping_head = state_dict_has_major_grouping_head(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    legacy_config = checkpoint.get("config", {})
    legacy_args = (
        legacy_config.get("args", {}) if isinstance(legacy_config, Mapping) else {}
    )
    checkpoint_args = dict(legacy_args) if isinstance(legacy_args, Mapping) else {}
    inference_config = checkpoint.get("inference_config", {})
    if isinstance(inference_config, Mapping):
        checkpoint_args.update(inference_config)

    metadata = {
        "checkpoint": checkpoint,
        "meter_classes": meter_classes,
        "quality_map": quality_map,
        "has_major_grouping_head": has_major_grouping_head,
        "checkpoint_args": checkpoint_args,
    }
    return model, model_config, metadata



@dataclasses.dataclass
class BeatChordInferenceConfig:
    device: torch.device
    window_ms_override: int | None = None
    stride_ms_override: int | None = None
    window_batch_size: int = 1
    beat_decode_mode: str = "grid"
    beat_threshold: float = 0.5
    downbeat_threshold: float = 0.5
    grid_tolerance_frames: int = 2
    meter_score_weight: float = 1.0
    beat_grid_score_weight: float = 1.0
    grid_downbeat_candidate_threshold: float = 0.15
    grid_beat_candidate_threshold: float = 0.35
    grid_max_bar_count: int = 4
    grid_beam_size: int = 24
    grid_jit: bool = False
    group_boundary_score_weight: float = 0.5
    grid_false_group_boundary_weight: float = 0.25
    grid_downbeat_score_weight: float = 1.5
    grid_false_downbeat_weight: float = 0.75
    grid_segment_penalty: float = 0.25
    grid_additive_meter_penalty: float = 0.35
    grid_tempo_transition_weight: float = 2.0
    grid_meter_change_penalty: float = 12.0
    grid_short_meter_run_penalty: float = 8.5
    grid_minimum_meter_run_quarter_notes: float = 4.0
    grid_octave_jump_penalty: float = 2.0
    grid_min_bpm: float = 30.0
    grid_max_bpm: float = 300.0
    chord_boundary_threshold: float = 0.5
    chord_boundary_min_distance_frames: int = 5
    key_boundary_threshold: float = 0.5
    key_boundary_min_distance_frames: int = 5
    key_boundary_js_weight: float = 0.0
    key_boundary_js_context_seconds: float = 3.0
    key_boundary_js_gap_seconds: float = 0.25
    disable_tqdm: bool = False
    meter_classes_json: Path | None = None
    quality_json: Path | None = None
    beat_decoder_cache_path: Path | None = None
    beat_mapped_ticks_per_beat: int = 480

@dataclasses.dataclass
class BeatChordInferenceResult:
    beat_times: list[float]
    downbeat_times: list[float]
    mapped_downbeat_times: list[float]
    meter_segments: list[MeterGridSegment] | list[MeterSegmentSpec]
    chord_segments: list[dict[str, object]]
    key_segments: list[dict[str, object]]
    chord_romanizer_status: str
    duration_seconds: float
    decoder_diagnostics: dict[str, object]
    beat_probabilities: np.ndarray | None = None
    downbeat_probabilities: np.ndarray | None = None
    group_boundary_probabilities: np.ndarray | None = None
    meter_logits: np.ndarray | None = None
    hop_length: int = 0
    sample_rate: int = 0
    meter_classes: list[tuple[int, int]] | None = None


def run_beat_chord_inference(
    midi_path: Path | str,
    checkpoint: dict,
    model: torch.nn.Module,
    model_config: MidiFrameModelConfig,
    metadata: dict,
    config: BeatChordInferenceConfig,
) -> BeatChordInferenceResult:
    midi_file = Path(midi_path).resolve()
    device = config.device
    
    meter_classes = metadata["meter_classes"]
    quality_map = metadata["quality_map"]
    has_major_grouping_head = metadata["has_major_grouping_head"]
    checkpoint_args = metadata["checkpoint_args"]
    
    beat_decode_mode = config.beat_decode_mode
    if beat_decode_mode in {"grid", "grid_legacy"} and meter_classes is None:
        beat_decode_mode = "peaks"

    window_ms, stride_ms = resolve_inference_window_settings(
        window_ms_override=config.window_ms_override,
        stride_ms_override=config.stride_ms_override,
        checkpoint_args=checkpoint_args,
    )

    sample_rate = model_config.sample_rate
    hop_length = model_config.hop_length
    window_frames = int(round(window_ms * sample_rate / 1000.0))
    model_frames = math.ceil(window_frames / hop_length)

    midi_data = pretty_midi.PrettyMIDI(str(midi_file))
    duration_seconds = midi_data.get_end_time()

    loader_config = MidiFrameLoaderConfig(
        midi_dir=midi_file.parent,
        sample_rate=sample_rate,
        hop_length=hop_length,
        pitch_min=model_config.pitch_min,
        pitch_max=model_config.pitch_max,
        num_channels=model_config.num_input_channels,
    )
    loader = SingleMidiFrameLoader(loader_config, midi_file)

    total_frames = math.ceil(duration_seconds * sample_rate / hop_length)
    buffer_size = max(total_frames, model_frames) + model_frames

    beat_probabilities_accum = torch.zeros(buffer_size)
    downbeat_probabilities_accum = torch.zeros(buffer_size)
    group_boundary_probabilities_accum = torch.zeros(buffer_size)
    meter_logits_accum = torch.zeros(buffer_size, model_config.num_meter_classes)
    chord_logits_accum = torch.zeros(buffer_size, model_config.num_root_chord_classes)
    chord_boundary_probabilities_accum = torch.zeros(buffer_size)
    bass_logits_accum = torch.zeros(buffer_size, 13)
    key_boundary_probabilities_accum = torch.zeros(buffer_size)
    key_logits_accum = torch.zeros(buffer_size, 13)
    weight_accum = torch.zeros(buffer_size)

    start_seconds = 0.0
    stride_seconds = stride_ms / 1000.0
    window_batch_size = int(config.window_batch_size)
    if window_batch_size <= 0:
        raise ValueError("window_batch_size must be positive")
    pending_rolls: list[torch.Tensor] = []
    pending_start_frames: list[int] = []
    omit_aux_outputs = isinstance(model, MidiFrameBeatChordModel)

    def flush_pending_windows() -> None:
        if not pending_rolls:
            return
        roll_batch = torch.stack(pending_rolls).to(device)
        with torch.inference_mode():
            model_kwargs: dict[str, object] = {
                "include_beat": True,
                "include_chord": True,
            }
            if omit_aux_outputs:
                model_kwargs["include_aux_outputs"] = False
            outputs = model(roll_batch, **model_kwargs)
            (
                beat_probs,
                downbeat_probs,
                group_boundary_probs,
                meter_logits_batch,
                chord_logits_batch,
                chord_boundary_probs,
                bass_logits_batch,
                key_boundary_probs,
                key_logits_batch,
            ) = copy_tensors_to_cpu_once(
                (
                    torch.sigmoid(outputs["beat_logits"]),
                    torch.sigmoid(outputs["downbeat_logits"]),
                    torch.sigmoid(outputs["group_boundary_logits"]),
                    outputs["meter_logits"],
                    outputs["root_chord_logits"],
                    torch.sigmoid(outputs["chord_boundary_logits"]),
                    outputs["bass_logits"],
                    torch.sigmoid(outputs["key_boundary_logits"]),
                    outputs["key_logits"],
                )
            )
            del outputs

        for batch_index, batch_start_frame in enumerate(pending_start_frames):
            target = slice(batch_start_frame, batch_start_frame + model_frames)
            chord_boundary_probabilities_accum[target] += chord_boundary_probs[
                batch_index
            ]
            bass_logits_accum[target] += bass_logits_batch[batch_index]
            key_boundary_probabilities_accum[target] += key_boundary_probs[
                batch_index
            ]
            key_logits_accum[target] += key_logits_batch[batch_index]
            beat_probabilities_accum[target] += beat_probs[batch_index]
            downbeat_probabilities_accum[target] += downbeat_probs[batch_index]
            group_boundary_probabilities_accum[target] += group_boundary_probs[
                batch_index
            ]
            meter_logits_accum[target] += meter_logits_batch[batch_index]
            chord_logits_accum[target] += chord_logits_batch[batch_index]
            weight_accum[target] += 1.0

        pending_rolls.clear()
        pending_start_frames.clear()

    while start_seconds < duration_seconds:
        start_frame = int(round(start_seconds * sample_rate / hop_length))
        try:
            roll = loader.load_window(
                song_name="", window_start_sec=start_seconds, num_frames=model_frames
            )
        except Exception:
            pass
        else:
            pending_rolls.append(roll)
            pending_start_frames.append(start_frame)

        start_seconds += stride_seconds
        if (
            len(pending_rolls) >= window_batch_size
            or start_seconds >= duration_seconds
        ):
            flush_pending_windows()

    active_mask = weight_accum > 0
    beat_probabilities_accum[active_mask] /= weight_accum[active_mask]
    downbeat_probabilities_accum[active_mask] /= weight_accum[active_mask]
    group_boundary_probabilities_accum[active_mask] /= weight_accum[active_mask]
    meter_logits_accum[active_mask] /= weight_accum[active_mask].unsqueeze(-1)
    chord_logits_accum[active_mask] /= weight_accum[active_mask].unsqueeze(-1)
    chord_boundary_probabilities_accum[active_mask] /= weight_accum[active_mask]
    bass_logits_accum[active_mask] /= weight_accum[active_mask].unsqueeze(-1)
    key_boundary_probabilities_accum[active_mask] /= weight_accum[active_mask]
    key_logits_accum[active_mask] /= weight_accum[active_mask].unsqueeze(-1)

    beat_probabilities_accum = beat_probabilities_accum[:total_frames]
    downbeat_probabilities_accum = downbeat_probabilities_accum[:total_frames]
    group_boundary_probabilities_accum = group_boundary_probabilities_accum[:total_frames]
    meter_logits_accum = meter_logits_accum[:total_frames]
    chord_logits_accum = chord_logits_accum[:total_frames]
    chord_boundary_probabilities_accum = chord_boundary_probabilities_accum[:total_frames]
    bass_logits_accum = bass_logits_accum[:total_frames]
    key_boundary_probabilities_accum = key_boundary_probabilities_accum[:total_frames]
    key_logits_accum = key_logits_accum[:total_frames]

    beat_probabilities_numpy = beat_probabilities_accum.numpy()
    downbeat_probabilities_numpy = downbeat_probabilities_accum.numpy()
    group_boundary_probabilities_numpy = (
        group_boundary_probabilities_accum.numpy() if has_major_grouping_head else None
    )
    meter_logits_numpy = meter_logits_accum.numpy()

    raw_downbeat_frame_indices = detect_peaks(downbeat_probabilities_numpy, threshold=config.downbeat_threshold)
    downbeat_frame_indices = list(raw_downbeat_frame_indices)

    meter_segments = []
    decoder_diagnostics = {}
    
    if beat_decode_mode == "grid" and meter_classes is not None:
        dp_config = BeatGridDPConfig(
            hop_length=hop_length,
            sample_rate=sample_rate,
            tolerance_frames=config.grid_tolerance_frames,
            downbeat_candidate_threshold=config.grid_downbeat_candidate_threshold,
            beat_candidate_threshold=config.grid_beat_candidate_threshold,
            max_bar_count=config.grid_max_bar_count,
            beam_size=config.grid_beam_size,
            use_jit_grid=config.grid_jit,
            min_quarter_bpm=config.grid_min_bpm,
            max_quarter_bpm=config.grid_max_bpm,
            beat_score_weight=config.beat_grid_score_weight,
            downbeat_score_weight=config.grid_downbeat_score_weight,
            false_downbeat_weight=config.grid_false_downbeat_weight,
            meter_score_weight=config.meter_score_weight,
            group_boundary_score_weight=config.group_boundary_score_weight,
            false_group_boundary_weight=config.grid_false_group_boundary_weight,
            additive_meter_penalty=config.grid_additive_meter_penalty,
            segment_penalty=config.grid_segment_penalty,
            tempo_transition_weight=config.grid_tempo_transition_weight,
            meter_change_penalty=config.grid_meter_change_penalty,
            short_meter_run_penalty=config.grid_short_meter_run_penalty,
            minimum_meter_run_quarter_notes=config.grid_minimum_meter_run_quarter_notes,
            octave_jump_penalty=config.grid_octave_jump_penalty,
        )
        beat_grid_result = decode_beats_with_meter_grid_dp(
            beat_probabilities=beat_probabilities_numpy,
            downbeat_probabilities=downbeat_probabilities_numpy,
            group_boundary_probabilities=group_boundary_probabilities_numpy,
            meter_logits=meter_logits_numpy,
            meter_classes=meter_classes,
            config=dp_config,
        )
        decoder_diagnostics = result_to_diagnostics(beat_grid_result)
        if beat_grid_result.beat_frames:
            beat_frame_indices = list(beat_grid_result.beat_frames)
            downbeat_frame_indices = list(beat_grid_result.downbeat_frames)
            meter_segments = list(beat_grid_result.meter_segments)
        else:
            beat_decode_mode = "grid_legacy"
            
    if beat_decode_mode == "grid_legacy" and meter_classes is not None:
        beat_frame_indices, meter_segments = decode_beats_with_meter_grid(
            beat_probabilities=beat_probabilities_numpy,
            downbeat_frames=downbeat_frame_indices,
            meter_logits=meter_logits_numpy,
            meter_classes=meter_classes,
            tolerance_frames=config.grid_tolerance_frames,
            meter_score_weight=config.meter_score_weight,
            beat_grid_score_weight=config.beat_grid_score_weight,
        )
        if not beat_frame_indices:
            beat_decode_mode = "peaks"
            
    if beat_decode_mode == "peaks":
        beat_frame_indices = detect_peaks(beat_probabilities_numpy, threshold=config.beat_threshold)
        meter_segments = []

    beat_times = [float(f * hop_length / sample_rate) for f in beat_frame_indices]
    downbeat_times = [float(f * hop_length / sample_rate) for f in downbeat_frame_indices]
    
    mapped_downbeat_frame_indices = sorted(
        {
            frame
            for segment in meter_segments
            for frame in segment.mapped_downbeat_frames
        }
    )
    if meter_segments:
        mapped_downbeat_frame_indices.append(int(meter_segments[-1].end_frame))
        mapped_downbeat_frame_indices = sorted(set(mapped_downbeat_frame_indices))
    elif downbeat_frame_indices:
        mapped_downbeat_frame_indices = list(downbeat_frame_indices)
    mapped_downbeat_times = [
        float(frame * hop_length / sample_rate)
        for frame in mapped_downbeat_frame_indices
    ]

    seconds_per_frame = hop_length / sample_rate
    chord_segments = decode_chord_segments(
        chord_logits=chord_logits_accum.numpy(),
        bass_logits=bass_logits_accum.numpy(),
        boundary_probabilities=chord_boundary_probabilities_accum.numpy(),
        quality_map=quality_map,
        seconds_per_frame=seconds_per_frame,
        duration_seconds=duration_seconds,
        boundary_threshold=config.chord_boundary_threshold,
        minimum_boundary_distance_frames=config.chord_boundary_min_distance_frames,
    )
    key_segments = decode_key_segments(
        key_logits=key_logits_accum.numpy(),
        boundary_probabilities=key_boundary_probabilities_accum.numpy(),
        seconds_per_frame=seconds_per_frame,
        duration_seconds=duration_seconds,
        boundary_threshold=config.key_boundary_threshold,
        minimum_boundary_distance_frames=config.key_boundary_min_distance_frames,
        js_weight=config.key_boundary_js_weight,
        js_context_seconds=config.key_boundary_js_context_seconds,
        js_gap_seconds=config.key_boundary_js_gap_seconds,
    )
    
    chord_segments, chord_romanizer_status = respell_chord_segments_with_romanizer(
        chord_segments,
        key_segments,
    )

    return BeatChordInferenceResult(
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        mapped_downbeat_times=mapped_downbeat_times,
        meter_segments=meter_segments,
        chord_segments=chord_segments,
        key_segments=key_segments,
        chord_romanizer_status=chord_romanizer_status,
        duration_seconds=duration_seconds,
        decoder_diagnostics=decoder_diagnostics,
        beat_probabilities=beat_probabilities_numpy,
        downbeat_probabilities=downbeat_probabilities_numpy,
        group_boundary_probabilities=group_boundary_probabilities_numpy,
        meter_logits=meter_logits_numpy,
        hop_length=hop_length,
        sample_rate=sample_rate,
        meter_classes=meter_classes,
    )


def predict_beat_chord_for_midi(
    input_midi_path: Path | str,
    output_midi_path: Path | str | None = None,
    *,
    checkpoint_path: Path | str | None = None,
    device: torch.device | str | None = None,
    beat_decode_mode: str = "grid",
    window_ms_override: int | None = None,
    stride_ms_override: int | None = None,
    window_batch_size: int = 1,
    beat_mapped_ticks_per_beat: int = 480,
    disable_tqdm: bool = False,
) -> Path:
    """指定された MIDI 入力に対して beat/chord/key を推論し、テンポマップ書き込み済み MIDI を出力する。"""
    input_midi_file = Path(input_midi_path).resolve()
    if not input_midi_file.exists():
        raise FileNotFoundError(f"Input MIDI file not found: {input_midi_file}")

    if output_midi_path is None:
        output_midi_file = input_midi_file.parent / f"{input_midi_file.stem}.beat_mapped.mid"
    else:
        output_midi_file = Path(output_midi_path).resolve()

    device_obj = resolve_device(device)

    resolved_checkpoint_path = ensure_beat_chord_checkpoint(checkpoint_path)
    model, model_config, metadata = load_beat_chord_model(
        resolved_checkpoint_path,
        device=device_obj,
    )
    
    # Defaults in original function
    config = BeatChordInferenceConfig(
        device=device_obj,
        beat_decode_mode=beat_decode_mode,
        window_ms_override=window_ms_override,
        stride_ms_override=stride_ms_override,
        window_batch_size=int(window_batch_size),
        beat_mapped_ticks_per_beat=beat_mapped_ticks_per_beat,
        disable_tqdm=disable_tqdm,
        beat_threshold=0.3,
        downbeat_threshold=0.3,
        chord_boundary_threshold=0.5,
        chord_boundary_min_distance_frames=5,
        key_boundary_threshold=0.5,
        key_boundary_min_distance_frames=5,
    )

    result = run_beat_chord_inference(
        midi_path=input_midi_file,
        checkpoint=metadata["checkpoint"],
        model=model,
        model_config=model_config,
        metadata=metadata,
        config=config,
    )

    output_midi_file.parent.mkdir(parents=True, exist_ok=True)
    export_tempo_mapped_midi(
        source_midi_path=input_midi_file,
        output_midi_path=output_midi_file,
        beat_times=result.beat_times,
        meter_segments=[
            MeterSegmentSpec(
                start_seconds=segment.start_frame * result.hop_length / result.sample_rate,
                end_seconds=segment.end_frame * result.hop_length / result.sample_rate,
                numerator=segment.meter_num,
                denominator=segment.meter_den,
                bar_count=segment.bar_count,
                score=segment.score,
            )
            for segment in result.meter_segments
        ],
        chord_segments=result.chord_segments,
        key_segments=result.key_segments,
        duration_seconds=result.duration_seconds,
        ticks_per_beat=config.beat_mapped_ticks_per_beat,
    )
    return output_midi_file


class SingleMidiFrameLoader(MidiFrameLoader):
    """特定の1つのMIDIファイルを対象として、load_windowを呼べるようにするラッパー"""

    def __init__(self, config: MidiFrameLoaderConfig, midi_path: Path) -> None:
        super().__init__(config)
        self.midi_path = midi_path

    def resolve_path(self, song_name: str) -> Path:
        return self.midi_path


def normalize_meter_classes(
    raw_meter_classes: object, expected_count: int
) -> list[tuple[int, int]] | None:
    """checkpoint/json 由来の meter class 表を `(num, den)` の list にそろえる。"""

    if raw_meter_classes is None:
        return None
    meter_classes: list[tuple[int, int]] = []
    for raw_meter in raw_meter_classes:
        if not isinstance(raw_meter, (list, tuple)) or len(raw_meter) != 2:
            raise ValueError("meter class entries must be [num, den]")
        meter_num = int(raw_meter[0])
        meter_den = int(raw_meter[1])
        if meter_num <= 0 or meter_den <= 0:
            raise ValueError("meter class values must be positive")
        meter_classes.append((meter_num, meter_den))
    if len(meter_classes) != int(expected_count):
        raise ValueError(
            f"meter class count mismatch: expected {expected_count}, got {len(meter_classes)}"
        )
    return meter_classes


def load_meter_classes(
    *,
    checkpoint: dict,
    model_config: MidiFrameModelConfig,
    meter_classes_json: Path | None,
) -> list[tuple[int, int]] | None:
    """推論に必要な meter class 表を checkpoint か外部 JSON から読み込む。"""

    if meter_classes_json is not None:
        with open(meter_classes_json, "r", encoding="utf-8") as f:
            return normalize_meter_classes(json.load(f), model_config.num_meter_classes)

    checkpoint_config = checkpoint.get("config", {})
    raw_meter_classes = checkpoint.get("beat_meter_classes")
    if raw_meter_classes is None and isinstance(checkpoint_config, dict):
        raw_meter_classes = checkpoint_config.get("beat_meter_classes")
    return normalize_meter_classes(raw_meter_classes, model_config.num_meter_classes)


def state_dict_has_major_grouping_head(
    state_dict: Mapping[str, object],
) -> bool:
    """Return whether a checkpoint contains a trained final grouping head."""

    return any(
        str(key).endswith("beat_head.group_boundary_proj.weight") for key in state_dict
    )


def load_chord_quality_map(
    *,
    checkpoint: Mapping[str, object],
    model_config: MidiFrameModelConfig,
    quality_json: Path | None,
) -> dict[str, str]:
    """Load the chord vocabulary from an override, checkpoint, or legacy path."""

    expected_count = int(model_config.num_root_chord_classes)
    if quality_json is not None:
        return load_chord_quality_map_json(
            quality_json,
            expected_root_chord_classes=expected_count,
        )
    embedded = checkpoint.get("chord_quality_map")
    if embedded is not None:
        return normalize_chord_quality_map(
            embedded,
            expected_root_chord_classes=expected_count,
        )

    legacy_path = Path("beat_chord_dataset/chord_dataset/quality.json")
    if legacy_path.is_file():
        return load_chord_quality_map_json(
            legacy_path,
            expected_root_chord_classes=expected_count,
        )
    raise FileNotFoundError(
        "Chord quality vocabulary is missing. Use a distilled checkpoint that "
        "contains chord_quality_map or pass --quality_json."
    )


def resolve_inference_window_settings(
    *,
    window_ms_override: int | None,
    stride_ms_override: int | None,
    checkpoint_args: dict[str, object],
) -> tuple[int, int]:
    """Resolve inference window and stride, defaulting stride to 50% overlap."""

    checkpoint_window_ms = checkpoint_args.get("window_ms", 8000)
    window_ms = int(
        window_ms_override if window_ms_override is not None else checkpoint_window_ms
    )
    stride_ms = int(
        stride_ms_override if stride_ms_override is not None else max(1, window_ms // 2)
    )
    if window_ms <= 0:
        raise ValueError("window_ms must be positive")
    if stride_ms <= 0:
        raise ValueError("stride_ms must be positive")
    return window_ms, stride_ms


def decode_index_to_chord(index: int, quality_map: dict[str, str]) -> str:
    """予測されたクラスインデックスからコード名を復元する"""
    number_of_qualities = len(quality_map)
    number_of_root_chord_classes = 12 * (number_of_qualities - 1) + 1

    if index == number_of_root_chord_classes - 1 or index < 0:
        return "N"

    root_index = index // (number_of_qualities - 1)
    quality_index = index % (number_of_qualities - 1)

    roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    root_string = roots[root_index]

    quality_string = quality_map.get(str(quality_index), "N")
    if quality_string == "":
        return root_string
    return f"{root_string}:{quality_string}"


PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)
KEY_SIGNATURE_NAMES = (
    "C",
    "Db",
    "D",
    "Eb",
    "E",
    "F",
    "Gb",
    "G",
    "Ab",
    "A",
    "Bb",
    "B",
)


def decode_pitch_class(index: int) -> str:
    """Decode the 13-way pitch-class targets used by bass and key heads."""
    return PITCH_CLASS_NAMES[int(index)] if 0 <= int(index) < 12 else "N"


def decode_key_class(index: int) -> str:
    """Decode a relative-major class to a valid Standard MIDI key name."""
    return KEY_SIGNATURE_NAMES[int(index)] if 0 <= int(index) < 12 else "N"


def format_chord_with_bass(chord_name: str, bass_name: str) -> str:
    """Append slash-bass notation only when it represents an inversion."""
    root = chord_name.split(":", maxsplit=1)[0]
    if chord_name == "N" or bass_name in {"N", root}:
        return chord_name
    return f"{chord_name}/{bass_name}"


def _boundary_frames(
    probabilities: np.ndarray,
    *,
    threshold: float,
    minimum_distance_frames: int,
    center_plateaus: bool = False,
) -> list[int]:
    """Return interval starts, always anchoring the first interval at frame zero."""
    if probabilities.ndim != 1:
        raise ValueError("boundary probabilities must be one-dimensional")
    if len(probabilities) == 0:
        return []
    active_frames = np.flatnonzero(probabilities > float(threshold))
    candidates: list[int] = []
    if len(active_frames) > 0:
        run_start = 0
        for run_end in range(1, len(active_frames) + 1):
            run_finished = (
                run_end == len(active_frames)
                or active_frames[run_end] > active_frames[run_end - 1] + 1
            )
            if not run_finished:
                continue
            run = active_frames[run_start:run_end]
            run_probabilities = probabilities[run]
            maximum = float(np.max(run_probabilities))
            maximum_frames = run[
                np.isclose(run_probabilities, maximum, rtol=1e-6, atol=1e-8)
            ]
            if center_plateaus and len(maximum_frames) > 1:
                candidates.append(int(maximum_frames[len(maximum_frames) // 2]))
            else:
                candidates.append(int(run[np.argmax(run_probabilities)]))
            run_start = run_end

    starts = [0]
    minimum_distance_frames = max(1, int(minimum_distance_frames))
    for candidate in candidates:
        if candidate <= 0:
            continue
        if candidate - starts[-1] >= minimum_distance_frames:
            starts.append(candidate)
        elif len(starts) > 1 and probabilities[candidate] > probabilities[starts[-1]]:
            starts[-1] = candidate
    return starts


def decode_chord_segments(
    *,
    chord_logits: np.ndarray,
    bass_logits: np.ndarray,
    boundary_probabilities: np.ndarray,
    quality_map: dict[str, str],
    seconds_per_frame: float,
    duration_seconds: float,
    boundary_threshold: float,
    minimum_boundary_distance_frames: int,
) -> list[dict[str, object]]:
    """Pool chord and bass evidence inside intervals selected by the boundary head."""
    if chord_logits.ndim != 2 or bass_logits.ndim != 2:
        raise ValueError("chord and bass logits must be two-dimensional")
    frame_count = int(chord_logits.shape[0])
    if (
        bass_logits.shape[0] != frame_count
        or len(boundary_probabilities) != frame_count
    ):
        raise ValueError(
            "chord, bass, and boundary predictions must share a frame axis"
        )
    starts = _boundary_frames(
        boundary_probabilities,
        threshold=boundary_threshold,
        minimum_distance_frames=minimum_boundary_distance_frames,
    )
    segments: list[dict[str, object]] = []
    for start_frame, end_frame in zip(starts, [*starts[1:], frame_count]):
        chord_index = int(np.argmax(chord_logits[start_frame:end_frame].mean(axis=0)))
        bass_index = int(np.argmax(bass_logits[start_frame:end_frame].mean(axis=0)))
        root_chord = decode_index_to_chord(chord_index, quality_map)
        bass = decode_pitch_class(bass_index)
        if root_chord == "N":
            bass = "N"
        chord = format_chord_with_bass(root_chord, bass)
        start_seconds = start_frame * seconds_per_frame
        end_seconds = (
            duration_seconds
            if end_frame == frame_count
            else end_frame * seconds_per_frame
        )
        if segments and segments[-1]["chord"] == chord:
            segments[-1]["end"] = round(end_seconds, 3)
        else:
            segments.append(
                {
                    "start": round(start_seconds, 3),
                    "end": round(end_seconds, 3),
                    "chord": chord,
                    "root_chord": root_chord,
                    "bass": bass,
                }
            )
    return segments


def compute_key_js_divergence(
    key_logits: np.ndarray,
    *,
    seconds_per_frame: float,
    context_seconds: float = 3.0,
    gap_seconds: float = 0.25,
) -> np.ndarray:
    """Measure the key-distribution change before and after every frame."""
    if key_logits.ndim != 2:
        raise ValueError("key logits must be two-dimensional")
    if seconds_per_frame <= 0.0:
        raise ValueError("seconds_per_frame must be positive")
    if context_seconds <= 0.0:
        raise ValueError("key JS context must be positive")
    if gap_seconds < 0.0:
        raise ValueError("key JS gap must be non-negative")

    frame_count = int(key_logits.shape[0])
    if frame_count == 0:
        return np.zeros(0, dtype=np.float32)

    logits = np.asarray(key_logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
    cumulative = np.concatenate(
        [
            np.zeros((1, probabilities.shape[1]), dtype=np.float64),
            np.cumsum(probabilities, axis=0),
        ],
        axis=0,
    )

    context_frames = max(1, int(round(context_seconds / seconds_per_frame)))
    gap_frames = max(0, int(round(gap_seconds / seconds_per_frame)))
    minimum_context_frames = max(1, context_frames // 2)
    divergence = np.zeros(frame_count, dtype=np.float64)
    epsilon = 1e-12
    for frame in range(frame_count):
        left_end = max(0, frame - gap_frames)
        left_start = max(0, left_end - context_frames)
        right_start = min(frame_count, frame + gap_frames + 1)
        right_end = min(frame_count, right_start + context_frames)
        left_count = left_end - left_start
        right_count = right_end - right_start
        if left_count < minimum_context_frames or right_count < minimum_context_frames:
            continue

        left = (cumulative[left_end] - cumulative[left_start]) / left_count
        right = (cumulative[right_end] - cumulative[right_start]) / right_count
        midpoint = 0.5 * (left + right)
        left_kl = np.sum(left * np.log2((left + epsilon) / (midpoint + epsilon)))
        right_kl = np.sum(right * np.log2((right + epsilon) / (midpoint + epsilon)))
        divergence[frame] = 0.5 * (left_kl + right_kl)

    return np.clip(divergence, 0.0, 1.0).astype(np.float32)


def enhance_key_boundary_probabilities(
    *,
    key_logits: np.ndarray,
    boundary_probabilities: np.ndarray,
    seconds_per_frame: float,
    js_weight: float = 0.4,
    js_context_seconds: float = 3.0,
    js_gap_seconds: float = 0.25,
) -> np.ndarray:
    """Use a sustained key-distribution change to reinforce boundary scores."""
    if not 0.0 <= float(js_weight) <= 1.0:
        raise ValueError("key JS weight must be between zero and one")
    raw = np.asarray(boundary_probabilities, dtype=np.float32)
    if raw.ndim != 1 or len(raw) != int(key_logits.shape[0]):
        raise ValueError("key and boundary predictions must share a frame axis")
    if float(js_weight) == 0.0:
        return raw.copy()

    divergence = compute_key_js_divergence(
        key_logits,
        seconds_per_frame=seconds_per_frame,
        context_seconds=js_context_seconds,
        gap_seconds=js_gap_seconds,
    )
    enhanced = raw + float(js_weight) * divergence * (1.0 - raw)
    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)


def decode_key_segments(
    *,
    key_logits: np.ndarray,
    boundary_probabilities: np.ndarray,
    seconds_per_frame: float,
    duration_seconds: float,
    boundary_threshold: float,
    minimum_boundary_distance_frames: int,
    js_weight: float = 0.4,
    js_context_seconds: float = 3.0,
    js_gap_seconds: float = 0.25,
) -> list[dict[str, object]]:
    """Decode key regions using boundary and sustained distribution changes."""
    if key_logits.ndim != 2:
        raise ValueError("key logits must be two-dimensional")
    frame_count = int(key_logits.shape[0])
    if len(boundary_probabilities) != frame_count:
        raise ValueError("key and boundary predictions must share a frame axis")
    enhanced_boundary_probabilities = enhance_key_boundary_probabilities(
        key_logits=key_logits,
        boundary_probabilities=boundary_probabilities,
        seconds_per_frame=seconds_per_frame,
        js_weight=js_weight,
        js_context_seconds=js_context_seconds,
        js_gap_seconds=js_gap_seconds,
    )
    starts = _boundary_frames(
        enhanced_boundary_probabilities,
        threshold=boundary_threshold,
        minimum_distance_frames=minimum_boundary_distance_frames,
        center_plateaus=True,
    )
    segments: list[dict[str, object]] = []
    for start_frame, end_frame in zip(starts, [*starts[1:], frame_count]):
        key_index = int(np.argmax(key_logits[start_frame:end_frame].mean(axis=0)))
        key_name = decode_key_class(key_index)
        start_seconds = start_frame * seconds_per_frame
        end_seconds = (
            duration_seconds
            if end_frame == frame_count
            else end_frame * seconds_per_frame
        )
        if segments and segments[-1]["key"] == key_name:
            segments[-1]["end"] = round(end_seconds, 3)
        else:
            segments.append(
                {
                    "start": round(start_seconds, 3),
                    "end": round(end_seconds, 3),
                    "key": key_name,
                }
            )
    return segments


def _key_at_time(key_segments: list[dict[str, object]], time_seconds: float) -> str:
    for segment in key_segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        if start <= time_seconds < end:
            return str(segment.get("key", "N"))
    return "N"


def _romanize_chord_symbols(
    symbols: list[str], tonic: str
) -> tuple[list[tuple[str, str]], str]:
    """Return display symbols and functional labels from chord-romanizer."""
    original_labels = [(symbol, symbol) for symbol in symbols]
    try:
        from chord_romanizer import Romanizer
    except ImportError:
        return original_labels, "unavailable"

    try:
        strict_factory = getattr(Romanizer, "strict", None)
        if callable(strict_factory):
            romanizer = strict_factory(
                default_tonic=tonic,
                simplify_accidentals=True,
            )
            display_progression = getattr(romanizer, "display_progression", None)
            if callable(display_progression):
                results = display_progression(symbols)
                results_by_index = {
                    int(getattr(result, "event_index")): result for result in results
                }
                if len(results) != len(symbols) or set(results_by_index) != set(
                    range(len(symbols))
                ):
                    raise ValueError(
                        "display_progression did not return every chord event"
                    )
                labels = []
                for event_index, original in enumerate(symbols):
                    result = results_by_index[event_index]
                    fixed_symbol = str(
                        getattr(result, "symbol", None)
                        or getattr(result, "symbol_fixed", None)
                        or original
                    )
                    labels.append((fixed_symbol, fixed_symbol))
                api_status = "applied"
            else:
                romanize_progression = getattr(romanizer, "romanize_progression", None)
                if not callable(romanize_progression):
                    raise AttributeError(
                        "strict Romanizer has neither display_progression "
                        "nor romanize_progression"
                    )
                results = romanize_progression(symbols)
                if len(results) != len(symbols):
                    raise ValueError(
                        "romanize_progression did not return every chord event"
                    )
                labels = [
                    (
                        str(
                            getattr(result, "normalized_symbol", None)
                            or getattr(result, "symbol_fixed", None)
                            or original
                        ),
                        str(
                            getattr(result, "normalized_symbol", None)
                            or getattr(result, "symbol_fixed", None)
                            or original
                        ),
                    )
                    for result, original in zip(results, symbols)
                ]
                api_status = "applied_symbol_only"
        else:
            from chord_romanizer import ChordParser

            parsed = [ChordParser.parse(symbol) for symbol in symbols]
            if any(chord is None for chord in parsed):
                return original_labels, "error"
            romanizer = Romanizer(
                default_tonic=tonic,
                simplify_accidentals=True,
            )
            results = romanizer.annotate_progression(parsed)
            if len(results) != len(symbols):
                raise ValueError(
                    "annotate_progression did not return every chord event"
                )
            labels = [
                (
                    str(
                        getattr(result, "normalized_symbol", None)
                        or getattr(result, "symbol_fixed", None)
                        or original
                    ),
                    str(
                        getattr(result, "normalized_symbol", None)
                        or getattr(result, "symbol_fixed", None)
                        or original
                    ),
                )
                for result, original in zip(results, symbols)
            ]
            api_status = "applied_legacy_api"
    except Exception:
        return original_labels, "error"

    return labels, api_status


def _symbol_root(symbol: str) -> str:
    if not symbol or symbol[0] not in "ABCDEFG":
        return "N"
    end = 1
    while end < len(symbol) and symbol[end] in "#b":
        end += 1
    return symbol[:end]


def _apply_fixed_chord_symbol(
    segment: dict[str, object], fixed_symbol: str, combined_label: str
) -> dict[str, object]:
    updated = dict(segment)
    old_root_chord = str(segment.get("root_chord", "N"))
    old_root = _symbol_root(old_root_chord)
    old_bass = str(segment.get("bass", "N"))
    fixed_root_chord, separator, fixed_bass = fixed_symbol.partition("/")

    updated["chord"] = fixed_symbol
    updated["combined_label"] = combined_label
    updated["root_chord"] = fixed_root_chord
    if separator:
        updated["bass"] = fixed_bass
    elif old_bass == old_root:
        updated["bass"] = _symbol_root(fixed_root_chord)
    return updated


def respell_chord_segments_with_romanizer(
    chord_segments: list[dict[str, object]],
    key_segments: list[dict[str, object]],
) -> tuple[list[dict[str, object]], str]:
    """Use chord-romanizer when installed, without making it a hard dependency."""
    updated = [dict(segment) for segment in chord_segments]
    processed = False
    had_error = False
    api_status = "skipped"
    index = 0
    while index < len(updated):
        symbol = str(updated[index].get("chord", "N"))
        tonic = _key_at_time(key_segments, float(updated[index].get("start", 0.0)))
        if symbol == "N" or tonic == "N":
            index += 1
            continue

        group_end = index + 1
        while group_end < len(updated):
            next_symbol = str(updated[group_end].get("chord", "N"))
            next_tonic = _key_at_time(
                key_segments, float(updated[group_end].get("start", 0.0))
            )
            if next_symbol == "N" or next_tonic != tonic:
                break
            group_end += 1

        symbols = [
            str(segment.get("chord", "N")) for segment in updated[index:group_end]
        ]
        labels, status = _romanize_chord_symbols(symbols, tonic)
        if status == "unavailable":
            return updated, status
        if status == "error":
            had_error = True
        else:
            processed = True
            api_status = status
            for offset, (fixed_symbol, combined_label) in enumerate(labels):
                target = index + offset
                updated[target] = _apply_fixed_chord_symbol(
                    updated[target], fixed_symbol, combined_label
                )
        index = group_end

    if had_error:
        return updated, "partial" if processed else "error"
    return updated, api_status if processed else "skipped"



def main() -> None:
    parser = argparse.ArgumentParser(description="MIDIファイルからビートとコードを予測します。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="学習済み重みのパス (.pth)")
    parser.add_argument("--midi_path", type=Path, required=True, help="推論対象のMIDIファイルのパス")
    parser.add_argument("--output_path", type=Path, default=None, help="予測結果を出力するJSONファイルのパス")
    parser.add_argument("--quality_json", type=Path, default=None, help="quality.json override.")
    parser.add_argument("--beat_decoder_cache_path", type=Path, default=None, help="Optional NPZ cache.")
    parser.add_argument("--window_ms", type=int, default=None, help="推論時の窓幅 (ms)。")
    parser.add_argument(
        "--window_batch_size",
        type=int,
        default=1,
        help="一度に推論する窓数。",
    )
    parser.add_argument("--beat_mapped_midi_path", type=Path, default=None, help="Output MIDI")
    parser.add_argument("--disable_beat_mapped_midi", action="store_true", help="Disable automatic export.")
    parser.add_argument("--beat_mapped_ticks_per_beat", type=int, default=960, help="Ticks per quarter note.")
    parser.add_argument("--stride_ms", type=int, default=None, help="ストライド (ms)。")
    parser.add_argument("--beat_threshold", type=float, default=0.5, help="ビート検出の閾値")
    parser.add_argument("--downbeat_threshold", type=float, default=0.5, help="ダウンビート検出の閾値")
    parser.add_argument("--beat_decode_mode", choices=("grid", "grid_legacy", "peaks"), default="grid", help="ビートのデコード方式。")
    parser.add_argument("--meter_classes_json", type=Path, default=None, help="古い checkpoint 用の meter class JSON。")
    parser.add_argument("--grid_tolerance_frames", type=int, default=2, help="grid score 許容フレーム幅。")
    parser.add_argument("--meter_score_weight", type=float, default=1.0)
    parser.add_argument("--beat_grid_score_weight", type=float, default=1.0)
    parser.add_argument("--grid_downbeat_candidate_threshold", type=float, default=0.15)
    parser.add_argument("--grid_beat_candidate_threshold", type=float, default=0.35)
    parser.add_argument("--grid_max_bar_count", type=int, default=4)
    parser.add_argument("--grid_beam_size", type=int, default=24)
    parser.add_argument("--grid_jit", action="store_true")
    parser.add_argument("--group_boundary_score_weight", type=float, default=0.5)
    parser.add_argument("--grid_false_group_boundary_weight", type=float, default=0.25)
    parser.add_argument("--grid_downbeat_score_weight", type=float, default=1.5)
    parser.add_argument("--grid_false_downbeat_weight", type=float, default=0.75)
    parser.add_argument("--grid_segment_penalty", type=float, default=0.25)
    parser.add_argument("--grid_additive_meter_penalty", type=float, default=0.35)
    parser.add_argument("--grid_tempo_transition_weight", type=float, default=2.0)
    parser.add_argument("--grid_meter_change_penalty", type=float, default=12.0)
    parser.add_argument("--grid_short_meter_run_penalty", type=float, default=8.5)
    parser.add_argument("--grid_minimum_meter_run_quarter_notes", type=float, default=4.0)
    parser.add_argument("--grid_octave_jump_penalty", type=float, default=2.0)
    parser.add_argument("--grid_min_bpm", type=float, default=30.0)
    parser.add_argument("--grid_max_bpm", type=float, default=300.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--chord_boundary_threshold", type=float, default=0.5)
    parser.add_argument("--chord_boundary_min_distance_frames", type=int, default=5)
    parser.add_argument("--key_boundary_threshold", type=float, default=0.5)
    parser.add_argument("--key_boundary_min_distance_frames", type=int, default=5)
    parser.add_argument("--key_boundary_js_weight", type=float, default=0.0)
    parser.add_argument("--key_boundary_js_context_seconds", type=float, default=3.0)
    parser.add_argument("--key_boundary_js_gap_seconds", type=float, default=0.25)
    args = parser.parse_args()

    if not 0.0 <= args.chord_boundary_threshold <= 1.0: raise ValueError("chord boundary threshold must be between zero and one")
    if not 0.0 <= args.key_boundary_threshold <= 1.0: raise ValueError("key boundary threshold must be between zero and one")
    if args.chord_boundary_min_distance_frames < 1: raise ValueError("chord boundary minimum distance must be positive")
    if args.key_boundary_min_distance_frames < 1: raise ValueError("key boundary minimum distance must be positive")
    if not 0.0 <= args.key_boundary_js_weight <= 1.0: raise ValueError("--key_boundary_js_weight must be between zero and one")
    if args.key_boundary_js_context_seconds <= 0.0: raise ValueError("--key_boundary_js_context_seconds must be positive")
    if args.key_boundary_js_gap_seconds < 0.0: raise ValueError("--key_boundary_js_gap_seconds must be non-negative")
    if args.grid_tolerance_frames < 0: raise ValueError("--grid_tolerance_frames must be non-negative")
    for threshold_name in ("grid_downbeat_candidate_threshold", "grid_beat_candidate_threshold"):
        threshold = float(getattr(args, threshold_name))
        if not 0.0 <= threshold <= 1.0: raise ValueError(f"--{threshold_name} must be between zero and one")
    if args.grid_max_bar_count <= 0: raise ValueError("--grid_max_bar_count must be positive")
    if args.grid_beam_size <= 0: raise ValueError("--grid_beam_size must be positive")
    if args.group_boundary_score_weight < 0.0: raise ValueError("--group_boundary_score_weight must be non-negative")
    if args.grid_false_group_boundary_weight < 0.0: raise ValueError("--grid_false_group_boundary_weight must be non-negative")
    if args.grid_minimum_meter_run_quarter_notes <= 0.0: raise ValueError("--grid_minimum_meter_run_quarter_notes must be positive")
    if args.grid_min_bpm <= 0.0 or args.grid_max_bpm <= args.grid_min_bpm: raise ValueError("grid BPM range must be positive and increasing")
    if args.beat_mapped_ticks_per_beat <= 0: raise ValueError("--beat_mapped_ticks_per_beat must be positive")
    if args.window_batch_size <= 0: raise ValueError("--window_batch_size must be positive")

    device = resolve_device(args.device)
    print(f"使用デバイス: {device}")

    print(f"チェックポイントをロード中: {args.checkpoint}")
    resolved_checkpoint_path = ensure_beat_chord_checkpoint(args.checkpoint)
    model, model_config, metadata = load_beat_chord_model(resolved_checkpoint_path, device=device)

    # Note: main originally overrides meter classes with meter_classes_json if passed.
    if args.meter_classes_json:
        metadata["meter_classes"] = load_meter_classes(
            checkpoint=metadata["checkpoint"],
            model_config=model_config,
            meter_classes_json=args.meter_classes_json,
        )

    if args.quality_json:
        metadata["quality_map"] = load_chord_quality_map(
            checkpoint=metadata["checkpoint"],
            model_config=model_config,
            quality_json=args.quality_json,
        )

    config = BeatChordInferenceConfig(
        device=device,
        window_ms_override=args.window_ms,
        stride_ms_override=args.stride_ms,
        window_batch_size=args.window_batch_size,
        beat_decode_mode=args.beat_decode_mode,
        beat_threshold=args.beat_threshold,
        downbeat_threshold=args.downbeat_threshold,
        grid_tolerance_frames=args.grid_tolerance_frames,
        meter_score_weight=args.meter_score_weight,
        beat_grid_score_weight=args.beat_grid_score_weight,
        grid_downbeat_candidate_threshold=args.grid_downbeat_candidate_threshold,
        grid_beat_candidate_threshold=args.grid_beat_candidate_threshold,
        grid_max_bar_count=args.grid_max_bar_count,
        grid_beam_size=args.grid_beam_size,
        grid_jit=args.grid_jit,
        group_boundary_score_weight=args.group_boundary_score_weight,
        grid_false_group_boundary_weight=args.grid_false_group_boundary_weight,
        grid_downbeat_score_weight=args.grid_downbeat_score_weight,
        grid_false_downbeat_weight=args.grid_false_downbeat_weight,
        grid_segment_penalty=args.grid_segment_penalty,
        grid_additive_meter_penalty=args.grid_additive_meter_penalty,
        grid_tempo_transition_weight=args.grid_tempo_transition_weight,
        grid_meter_change_penalty=args.grid_meter_change_penalty,
        grid_short_meter_run_penalty=args.grid_short_meter_run_penalty,
        grid_minimum_meter_run_quarter_notes=args.grid_minimum_meter_run_quarter_notes,
        grid_octave_jump_penalty=args.grid_octave_jump_penalty,
        grid_min_bpm=args.grid_min_bpm,
        grid_max_bpm=args.grid_max_bpm,
        chord_boundary_threshold=args.chord_boundary_threshold,
        chord_boundary_min_distance_frames=args.chord_boundary_min_distance_frames,
        key_boundary_threshold=args.key_boundary_threshold,
        key_boundary_min_distance_frames=args.key_boundary_min_distance_frames,
        key_boundary_js_weight=args.key_boundary_js_weight,
        key_boundary_js_context_seconds=args.key_boundary_js_context_seconds,
        key_boundary_js_gap_seconds=args.key_boundary_js_gap_seconds,
        disable_tqdm=False,
        meter_classes_json=args.meter_classes_json,
        quality_json=args.quality_json,
        beat_decoder_cache_path=args.beat_decoder_cache_path,
        beat_mapped_ticks_per_beat=args.beat_mapped_ticks_per_beat,
    )

    print("推論を実行中...")
    result = run_beat_chord_inference(
        midi_path=args.midi_path,
        checkpoint=metadata["checkpoint"],
        model=model,
        model_config=model_config,
        metadata=metadata,
        config=config,
    )

    if args.beat_decoder_cache_path is not None:
        args.beat_decoder_cache_path.parent.mkdir(parents=True, exist_ok=True)
        decoder_cache = {
            "beat_probabilities": result.beat_probabilities,
            "downbeat_probabilities": result.downbeat_probabilities,
            "meter_logits": result.meter_logits,
            "meter_classes": np.asarray(result.meter_classes or [], dtype=np.int64),
            "sample_rate": np.asarray(result.sample_rate, dtype=np.int64),
            "hop_length": np.asarray(result.hop_length, dtype=np.int64),
        }
        if result.group_boundary_probabilities is not None:
            decoder_cache["group_boundary_probabilities"] = result.group_boundary_probabilities
        np.savez_compressed(args.beat_decoder_cache_path, **decoder_cache)

    if result.chord_romanizer_status == "unavailable":
        print("chord-romanizer is not installed; keeping the decoded spellings.")
    elif result.chord_romanizer_status in {"error", "partial"}:
        print("Warning: chord-romanizer could not respell every chord; keeping the original spelling where needed.")
    elif result.chord_romanizer_status in {"applied_symbol_only", "applied_legacy_api"}:
        print("Warning: this chord-romanizer version does not provide display_progression(); functional labels are unavailable.")

    print("\n--- 予測結果サマリー ---")
    print(f"ビートデコード方式: {config.beat_decode_mode}")
    print(f"検出ビート数: {len(result.beat_times)}")
    print(f"検出ダウンビート数: {len(result.downbeat_times)}")
    if result.meter_segments:
        meter_counter: dict[str, int] = {}
        for segment in result.meter_segments:
            meter_name = f"{segment.meter_num}/{segment.meter_den}"
            meter_counter[meter_name] = meter_counter.get(meter_name, 0) + 1
        meter_summary = ", ".join(f"{name}: {count}" for name, count in sorted(meter_counter.items()))
        print(f"選択meter: {meter_summary}")
    print(f"コード区間数: {len(result.chord_segments)}")
    print(f"キー区間数: {len(result.key_segments)}")

    print("\n予測コード（最初の10区間）:")
    for segment in result.chord_segments[:10]:
        print(f"  {segment['start']:6.2f}s - {segment['end']:6.2f}s : {segment.get('combined_label', segment['chord'])}")
    if len(result.chord_segments) > 10:
        print("  ...")

    if args.output_path is None:
        output_dir = Path("beat_chord_predictions")
        output_dir.mkdir(exist_ok=True)
        args.output_path = output_dir / f"{args.midi_path.stem}.prediction.json"

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    tempo_mapped_result = None
    if not args.disable_beat_mapped_midi:
        beat_mapped_midi_path = args.beat_mapped_midi_path
        if beat_mapped_midi_path is None:
            beat_mapped_midi_path = args.output_path.parent / f"{args.midi_path.stem}.beat_mapped.mid"
        tempo_mapped_result = export_tempo_mapped_midi(
            source_midi_path=args.midi_path,
            output_midi_path=beat_mapped_midi_path,
            beat_times=result.beat_times,
            meter_segments=[
                MeterSegmentSpec(
                    start_seconds=segment.start_frame * result.hop_length / result.sample_rate,
                    end_seconds=segment.end_frame * result.hop_length / result.sample_rate,
                    numerator=segment.meter_num,
                    denominator=segment.meter_den,
                    bar_count=segment.bar_count,
                    score=segment.score,
                )
                for segment in result.meter_segments
            ],
            chord_segments=result.chord_segments,
            key_segments=result.key_segments,
            duration_seconds=result.duration_seconds,
            ticks_per_beat=args.beat_mapped_ticks_per_beat,
        )
        print(f"Saved tempo-mapped MIDI: {beat_mapped_midi_path}")
        print(f"Maximum absolute note timing drift: {tempo_mapped_result.max_note_drift_seconds * 1000.0:.3f}ms")
        if not tempo_mapped_result.used_predicted_tempo:
            print("Warning: no valid meter interval was decoded; the original tempo map was retained and chords/keys were added.")

    output_data = {
        "song_name": args.midi_path.stem,
        "beat_decode_mode": config.beat_decode_mode,
        "decoder_diagnostics": {
            **result.decoder_diagnostics,
            "seconds_per_frame": float(result.hop_length / result.sample_rate),
        },
        "beats": result.beat_times,
        "downbeats": result.downbeat_times,
        "mapped_downbeats": result.mapped_downbeat_times,
        "meters": [
            {
                "start": round(segment.start_frame * result.hop_length / result.sample_rate, 3),
                "end": round(segment.end_frame * result.hop_length / result.sample_rate, 3),
                "meter_index": int(segment.meter_index),
                "meter": f"{segment.meter_num}/{segment.meter_den}",
                "bar_count": int(segment.bar_count),
                "source": "interpolated" if segment.bar_count > 1 else "detected",
                "score": float(segment.score),
                "tempo_bpm": (None if getattr(segment, "quarter_note_bpm", None) is None else float(segment.quarter_note_bpm)),
                "score_components": getattr(segment, "score_components", None),
                "confidence_margin": getattr(segment, "confidence_margin", None),
                "meter_evidence_source": getattr(segment, "meter_evidence_source", "direct"),
                "major_grouping": (None if getattr(segment, "major_grouping", None) is None else list(segment.major_grouping)),
            }
            for segment in result.meter_segments
        ],
        "chords": result.chord_segments,
        "chord_romanizer": result.chord_romanizer_status,
        "keys": result.key_segments,
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved predictions to: {args.output_path}")

if __name__ == '__main__':
    main()
