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
        str(key).endswith("beat_head.group_boundary_proj.weight")
        for key in state_dict
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
                    combined_label = str(
                        getattr(result, "combined_label", None) or fixed_symbol
                    )
                    labels.append((fixed_symbol, combined_label))
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
    parser = argparse.ArgumentParser(
        description="MIDIファイルからビートとコードを予測します。"
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="学習済み重みのパス (.pth)"
    )
    parser.add_argument(
        "--midi_path", type=Path, required=True, help="推論対象のMIDIファイルのパス"
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="予測結果を出力するJSONファイルのパス（未指定の場合は predictions/ の下に自動生成）",
    )
    parser.add_argument(
        "--quality_json",
        type=Path,
        default=None,
        help=(
            "quality.json override. By default the vocabulary embedded in the "
            "checkpoint is used."
        ),
    )
    parser.add_argument(
        "--beat_decoder_cache_path",
        type=Path,
        default=None,
        help="Optional NPZ cache for beat/downbeat probabilities and meter logits.",
    )
    parser.add_argument(
        "--window_ms",
        type=int,
        default=None,
        help="推論時の窓幅 (ms)。未指定の場合はチェックポイントの設定を使用",
    )
    parser.add_argument(
        "--beat_mapped_midi_path",
        type=Path,
        default=None,
        help="Output MIDI containing the predicted tempo, meter, and chord map.",
    )
    parser.add_argument(
        "--disable_beat_mapped_midi",
        action="store_true",
        help="Disable automatic tempo-mapped MIDI export.",
    )
    parser.add_argument(
        "--beat_mapped_ticks_per_beat",
        type=int,
        default=960,
        help="Ticks per quarter note used by the tempo-mapped MIDI.",
    )
    parser.add_argument(
        "--stride_ms",
        type=int,
        default=None,
        help="スライディングウィンドウのストライド (ms)。未指定時は窓幅の半分",
    )
    parser.add_argument(
        "--beat_threshold", type=float, default=0.5, help="ビート検出の閾値"
    )
    parser.add_argument(
        "--downbeat_threshold", type=float, default=0.5, help="ダウンビート検出の閾値"
    )
    parser.add_argument(
        "--beat_decode_mode",
        choices=("grid", "grid_legacy", "peaks"),
        default="grid",
        help="ビートのデコード方式。grid は downbeat 間で meter grid を1つ選ぶ。",
    )
    parser.add_argument(
        "--meter_classes_json",
        type=Path,
        default=None,
        help="古い checkpoint 用の meter class JSON。例: [[4, 4], [12, 8]]",
    )
    parser.add_argument(
        "--grid_tolerance_frames",
        type=int,
        default=2,
        help="grid score と snap に使う許容フレーム幅。",
    )
    parser.add_argument(
        "--meter_score_weight",
        type=float,
        default=1.0,
        help="grid decoder で meter logits をどれだけ重視するか。",
    )
    parser.add_argument(
        "--beat_grid_score_weight",
        type=float,
        default=1.0,
        help="grid decoder で beat の on/off grid score をどれだけ重視するか。",
    )
    parser.add_argument(
        "--grid_downbeat_candidate_threshold",
        type=float,
        default=0.15,
        help="Low threshold used to propose downbeat candidates for the DP lattice.",
    )
    parser.add_argument(
        "--grid_beat_candidate_threshold",
        type=float,
        default=0.35,
        help="Threshold used to propose beat-aligned bar boundaries.",
    )
    parser.add_argument(
        "--grid_max_bar_count",
        type=int,
        default=4,
        help="Maximum number of bars represented by one lattice edge.",
    )
    parser.add_argument(
        "--grid_beam_size",
        type=int,
        default=24,
        help="Number of tempo/meter path states retained per boundary.",
    )
    parser.add_argument(
        "--grid_jit",
        action="store_true",
        help=("Use the optional cached CPU Numba kernel for beat-grid snapping."),
    )
    parser.add_argument(
        "--group_boundary_score_weight",
        type=float,
        default=0.5,
        help="Weight for learned major beat-group boundaries in grid decoding.",
    )
    parser.add_argument(
        "--grid_false_group_boundary_weight",
        type=float,
        default=0.25,
        help="Penalty for unsupported positive major-boundary evidence.",
    )
    parser.add_argument("--grid_downbeat_score_weight", type=float, default=1.5)
    parser.add_argument("--grid_false_downbeat_weight", type=float, default=0.75)
    parser.add_argument("--grid_segment_penalty", type=float, default=0.25)
    parser.add_argument("--grid_additive_meter_penalty", type=float, default=0.35)
    parser.add_argument("--grid_tempo_transition_weight", type=float, default=2.0)
    parser.add_argument("--grid_meter_change_penalty", type=float, default=12.0)
    parser.add_argument("--grid_short_meter_run_penalty", type=float, default=8.5)
    parser.add_argument(
        "--grid_minimum_meter_run_quarter_notes", type=float, default=4.0
    )
    parser.add_argument("--grid_octave_jump_penalty", type=float, default=2.0)
    parser.add_argument("--grid_min_bpm", type=float, default=30.0)
    parser.add_argument("--grid_max_bpm", type=float, default=300.0)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--chord_boundary_threshold",
        type=float,
        default=0.5,
        help="Minimum chord-boundary probability used to start a new chord region.",
    )
    parser.add_argument(
        "--chord_boundary_min_distance_frames",
        type=int,
        default=5,
        help="Minimum distance between decoded chord boundaries, in model frames.",
    )
    parser.add_argument(
        "--key_boundary_threshold",
        type=float,
        default=0.5,
        help="Minimum key-boundary probability used to start a new key region.",
    )
    parser.add_argument(
        "--key_boundary_min_distance_frames",
        type=int,
        default=5,
        help="Minimum distance between decoded key boundaries, in model frames.",
    )
    parser.add_argument(
        "--key_boundary_js_weight",
        type=float,
        default=0.0,
        help="Weight used to reinforce key boundaries with JS divergence.",
    )
    parser.add_argument(
        "--key_boundary_js_context_seconds",
        type=float,
        default=3.0,
        help="Seconds of key evidence averaged on each side of a boundary.",
    )
    parser.add_argument(
        "--key_boundary_js_gap_seconds",
        type=float,
        default=0.25,
        help="Evidence gap excluded on each side of a candidate boundary.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.chord_boundary_threshold <= 1.0:
        raise ValueError("chord boundary threshold must be between zero and one")
    if not 0.0 <= args.key_boundary_threshold <= 1.0:
        raise ValueError("key boundary threshold must be between zero and one")
    if args.chord_boundary_min_distance_frames < 1:
        raise ValueError("chord boundary minimum distance must be positive")
    if args.key_boundary_min_distance_frames < 1:
        raise ValueError("key boundary minimum distance must be positive")
    if not 0.0 <= args.key_boundary_js_weight <= 1.0:
        raise ValueError("--key_boundary_js_weight must be between zero and one")
    if args.key_boundary_js_context_seconds <= 0.0:
        raise ValueError("--key_boundary_js_context_seconds must be positive")
    if args.key_boundary_js_gap_seconds < 0.0:
        raise ValueError("--key_boundary_js_gap_seconds must be non-negative")
    if args.grid_tolerance_frames < 0:
        raise ValueError("--grid_tolerance_frames must be non-negative")
    for threshold_name in (
        "grid_downbeat_candidate_threshold",
        "grid_beat_candidate_threshold",
    ):
        threshold = float(getattr(args, threshold_name))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"--{threshold_name} must be between zero and one")
    if args.grid_max_bar_count <= 0:
        raise ValueError("--grid_max_bar_count must be positive")
    if args.grid_beam_size <= 0:
        raise ValueError("--grid_beam_size must be positive")
    if args.group_boundary_score_weight < 0.0:
        raise ValueError("--group_boundary_score_weight must be non-negative")
    if args.grid_false_group_boundary_weight < 0.0:
        raise ValueError(
            "--grid_false_group_boundary_weight must be non-negative"
        )
    if args.grid_minimum_meter_run_quarter_notes <= 0.0:
        raise ValueError("--grid_minimum_meter_run_quarter_notes must be positive")
    if args.grid_min_bpm <= 0.0 or args.grid_max_bpm <= args.grid_min_bpm:
        raise ValueError("grid BPM range must be positive and increasing")
    if args.beat_mapped_ticks_per_beat <= 0:
        raise ValueError("--beat_mapped_ticks_per_beat must be positive")

    # デバイス設定
    device = torch.device(args.device)
    print(f"使用デバイス: {device}")

    # チェックポイントロード
    print(f"チェックポイントをロード中: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # モデル設定の復元
    model_config_dict = checkpoint["model_config"]
    model_config = MidiFrameModelConfig(**model_config_dict)
    meter_classes = load_meter_classes(
        checkpoint=checkpoint,
        model_config=model_config,
        meter_classes_json=args.meter_classes_json,
    )
    if args.beat_decode_mode in {"grid", "grid_legacy"} and meter_classes is None:
        print(
            "警告: meter class 表が checkpoint にないため、beat_decode_mode=peaks に戻します。"
            " 新しい checkpoint を使うか --meter_classes_json を指定してください。"
        )
        beat_decode_mode = "peaks"
    else:
        beat_decode_mode = args.beat_decode_mode

    # モデル構築と重み適用
    model = MidiFrameBeatChordModel(model_config).to(device)
    state_dict = checkpoint.get("ema_state_dict", checkpoint.get("model_state_dict"))
    if state_dict is None:
        state_dict = checkpoint
    has_major_grouping_head = state_dict_has_major_grouping_head(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print("モデルのロードが完了しました。")

    # 推論窓サイズの設定
    legacy_config = checkpoint.get("config", {})
    legacy_args = (
        legacy_config.get("args", {}) if isinstance(legacy_config, Mapping) else {}
    )
    checkpoint_args = dict(legacy_args) if isinstance(legacy_args, Mapping) else {}
    inference_config = checkpoint.get("inference_config", {})
    if isinstance(inference_config, Mapping):
        checkpoint_args.update(inference_config)
    window_ms, stride_ms = resolve_inference_window_settings(
        window_ms_override=args.window_ms,
        stride_ms_override=args.stride_ms,
        checkpoint_args=checkpoint_args,
    )

    sample_rate = model_config.sample_rate
    hop_length = model_config.hop_length

    window_frames = int(round(window_ms * sample_rate / 1000.0))
    model_frames = math.ceil(window_frames / hop_length)

    print(
        f"推論設定 - 窓幅: {window_ms}ms ({model_frames}フレーム), ストライド: {stride_ms}ms"
    )

    # MIDIファイルのロード
    if not args.midi_path.exists():
        print(
            f"エラー: MIDIファイルが見つかりません: {args.midi_path}", file=sys.stderr
        )
        sys.exit(1)

    print(f"MIDIファイルを解析中: {args.midi_path}")
    try:
        midi_data = pretty_midi.PrettyMIDI(str(args.midi_path))
        duration_seconds = midi_data.get_end_time()
    except Exception as e:
        print(f"エラー: MIDIファイルの読み込みに失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"MIDIの長さ: {duration_seconds:.2f}秒")

    # SingleMidiFrameLoaderの設定
    loader_config = MidiFrameLoaderConfig(
        midi_dir=args.midi_path.parent,  # ダミー。resolve_pathで上書きされるため不使用
        sample_rate=sample_rate,
        hop_length=hop_length,
        pitch_min=model_config.pitch_min,
        pitch_max=model_config.pitch_max,
        num_channels=model_config.num_input_channels,
    )
    loader = SingleMidiFrameLoader(loader_config, args.midi_path)

    # 累積用バッファの初期化
    total_frames = math.ceil(duration_seconds * sample_rate / hop_length)
    # 推論時に余白が出る可能性を考慮してバッファサイズを大きめに確保
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

    # スライディングウィンドウによる推論ループ
    start_seconds = 0.0
    stride_seconds = stride_ms / 1000.0

    print("推論を実行中...")
    while start_seconds < duration_seconds:
        start_frame = int(round(start_seconds * sample_rate / hop_length))

        # 窓のデータを取得
        try:
            roll = loader.load_window(
                song_name="", window_start_sec=start_seconds, num_frames=model_frames
            )
        except Exception as e:
            print(f"警告: 窓のロードに失敗しました (開始時間: {start_seconds}s): {e}")
            start_seconds += stride_seconds
            continue

        roll = roll.unsqueeze(0).to(device)  # バッチ次元の追加

        with torch.no_grad():
            outputs = model(roll, include_beat=True, include_chord=True)

            beat_prob = torch.sigmoid(outputs["beat_logits"]).squeeze(0).cpu()
            downbeat_prob = torch.sigmoid(outputs["downbeat_logits"]).squeeze(0).cpu()
            group_boundary_prob = (
                torch.sigmoid(outputs["group_boundary_logits"]).squeeze(0).cpu()
            )
            meter_logits = outputs["meter_logits"].squeeze(0).cpu()
            chord_logits = outputs["root_chord_logits"].squeeze(0).cpu()
            chord_boundary_prob = (
                torch.sigmoid(outputs["chord_boundary_logits"]).squeeze(0).cpu()
            )
            bass_logits = outputs["bass_logits"].squeeze(0).cpu()
            key_boundary_prob = (
                torch.sigmoid(outputs["key_boundary_logits"]).squeeze(0).cpu()
            )
            key_logits = outputs["key_logits"].squeeze(0).cpu()

        # バッファへの累積
        chord_boundary_probabilities_accum[
            start_frame : start_frame + model_frames
        ] += chord_boundary_prob
        bass_logits_accum[start_frame : start_frame + model_frames] += bass_logits
        key_boundary_probabilities_accum[start_frame : start_frame + model_frames] += (
            key_boundary_prob
        )
        key_logits_accum[start_frame : start_frame + model_frames] += key_logits
        beat_probabilities_accum[start_frame : start_frame + model_frames] += beat_prob
        downbeat_probabilities_accum[start_frame : start_frame + model_frames] += (
            downbeat_prob
        )
        group_boundary_probabilities_accum[
            start_frame : start_frame + model_frames
        ] += group_boundary_prob
        meter_logits_accum[start_frame : start_frame + model_frames] += meter_logits
        chord_logits_accum[start_frame : start_frame + model_frames] += chord_logits
        weight_accum[start_frame : start_frame + model_frames] += 1.0

        start_seconds += stride_seconds

    # 平均化
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

    # 実際の長さにトリミング
    beat_probabilities_accum = beat_probabilities_accum[:total_frames]
    downbeat_probabilities_accum = downbeat_probabilities_accum[:total_frames]
    group_boundary_probabilities_accum = group_boundary_probabilities_accum[
        :total_frames
    ]
    meter_logits_accum = meter_logits_accum[:total_frames]
    chord_logits_accum = chord_logits_accum[:total_frames]
    chord_boundary_probabilities_accum = chord_boundary_probabilities_accum[
        :total_frames
    ]
    bass_logits_accum = bass_logits_accum[:total_frames]
    key_boundary_probabilities_accum = key_boundary_probabilities_accum[:total_frames]
    key_logits_accum = key_logits_accum[:total_frames]

    # 1. downbeat は小節境界としてピーク検出する。
    beat_probabilities_numpy = beat_probabilities_accum.numpy()
    downbeat_probabilities_numpy = downbeat_probabilities_accum.numpy()
    group_boundary_probabilities_numpy = (
        group_boundary_probabilities_accum.numpy()
        if has_major_grouping_head
        else None
    )
    meter_logits_numpy = meter_logits_accum.numpy()
    raw_downbeat_frame_indices = detect_peaks(
        downbeat_probabilities_numpy, threshold=args.downbeat_threshold
    )
    downbeat_frame_indices = list(raw_downbeat_frame_indices)
    if args.beat_decoder_cache_path is not None:
        args.beat_decoder_cache_path.parent.mkdir(parents=True, exist_ok=True)
        decoder_cache = {
            "beat_probabilities": beat_probabilities_numpy,
            "downbeat_probabilities": downbeat_probabilities_numpy,
            "meter_logits": meter_logits_numpy,
            "meter_classes": np.asarray(meter_classes or [], dtype=np.int64),
            "sample_rate": np.asarray(sample_rate, dtype=np.int64),
            "hop_length": np.asarray(hop_length, dtype=np.int64),
        }
        if group_boundary_probabilities_numpy is not None:
            decoder_cache["group_boundary_probabilities"] = (
                group_boundary_probabilities_numpy
            )
        np.savez_compressed(args.beat_decoder_cache_path, **decoder_cache)

    # 2. beat は従来のピーク検出か、meter grid による構造化 decode で決める。
    meter_segments: list[MeterGridSegment] = []
    meter_segments: list[MeterGridSegment] = []
    decoder_diagnostics: dict[str, object] = {}
    if beat_decode_mode == "grid" and meter_classes is not None:
        grid_result = decode_beats_with_meter_grid_dp(
            beat_probabilities=beat_probabilities_numpy,
            downbeat_probabilities=downbeat_probabilities_numpy,
            group_boundary_probabilities=group_boundary_probabilities_numpy,
            meter_logits=meter_logits_numpy,
            meter_classes=meter_classes,
            config=BeatGridDPConfig(
                sample_rate=sample_rate,
                hop_length=hop_length,
                tolerance_frames=args.grid_tolerance_frames,
                downbeat_candidate_threshold=(args.grid_downbeat_candidate_threshold),
                beat_candidate_threshold=args.grid_beat_candidate_threshold,
                max_bar_count=args.grid_max_bar_count,
                beam_size=args.grid_beam_size,
                use_jit_grid=args.grid_jit,
                min_quarter_bpm=args.grid_min_bpm,
                max_quarter_bpm=args.grid_max_bpm,
                beat_score_weight=args.beat_grid_score_weight,
                downbeat_score_weight=args.grid_downbeat_score_weight,
                false_downbeat_weight=args.grid_false_downbeat_weight,
                meter_score_weight=args.meter_score_weight,
                group_boundary_score_weight=args.group_boundary_score_weight,
                false_group_boundary_weight=(
                    args.grid_false_group_boundary_weight
                ),
                additive_meter_penalty=args.grid_additive_meter_penalty,
                segment_penalty=args.grid_segment_penalty,
                tempo_transition_weight=args.grid_tempo_transition_weight,
                meter_change_penalty=args.grid_meter_change_penalty,
                short_meter_run_penalty=args.grid_short_meter_run_penalty,
                minimum_meter_run_quarter_notes=(
                    args.grid_minimum_meter_run_quarter_notes
                ),
                octave_jump_penalty=args.grid_octave_jump_penalty,
            ),
        )
        decoder_diagnostics = result_to_diagnostics(grid_result)
        if grid_result.beat_frames:
            beat_frame_indices = list(grid_result.beat_frames)
            downbeat_frame_indices = list(grid_result.downbeat_frames)
            meter_segments = list(grid_result.meter_segments)
        else:
            print(
                "Warning: DP grid decoder produced no path; "
                "falling back to the legacy grid decoder."
            )
            beat_decode_mode = "grid_legacy"

    if beat_decode_mode == "grid_legacy" and meter_classes is not None:
        beat_frame_indices, meter_segments = decode_beats_with_meter_grid(
            beat_probabilities=beat_probabilities_numpy,
            downbeat_frames=downbeat_frame_indices,
            meter_logits=meter_logits_numpy,
            meter_classes=meter_classes,
            tolerance_frames=args.grid_tolerance_frames,
            meter_score_weight=args.meter_score_weight,
            beat_grid_score_weight=args.beat_grid_score_weight,
        )
        if not beat_frame_indices:
            print(
                "警告: grid decoder が beat を生成できなかったため、ピーク検出に戻します。"
            )
            beat_decode_mode = "peaks"
    if beat_decode_mode == "peaks":
        beat_frame_indices = detect_peaks(
            beat_probabilities_numpy,
            threshold=args.beat_threshold,
        )

    beat_times = [float(f * hop_length / sample_rate) for f in beat_frame_indices]
    downbeat_times = [
        float(f * hop_length / sample_rate) for f in downbeat_frame_indices
    ]
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

    # コード予測のデコード
    quality_map = load_chord_quality_map(
        checkpoint=checkpoint,
        model_config=model_config,
        quality_json=args.quality_json,
    )

    # Boundary heads decide region starts; class heads are pooled within each region.
    seconds_per_frame = hop_length / sample_rate
    chord_segments = decode_chord_segments(
        chord_logits=chord_logits_accum.numpy(),
        bass_logits=bass_logits_accum.numpy(),
        boundary_probabilities=chord_boundary_probabilities_accum.numpy(),
        quality_map=quality_map,
        seconds_per_frame=seconds_per_frame,
        duration_seconds=duration_seconds,
        boundary_threshold=args.chord_boundary_threshold,
        minimum_boundary_distance_frames=args.chord_boundary_min_distance_frames,
    )
    key_segments = decode_key_segments(
        key_logits=key_logits_accum.numpy(),
        boundary_probabilities=key_boundary_probabilities_accum.numpy(),
        seconds_per_frame=seconds_per_frame,
        duration_seconds=duration_seconds,
        boundary_threshold=args.key_boundary_threshold,
        minimum_boundary_distance_frames=args.key_boundary_min_distance_frames,
        js_weight=args.key_boundary_js_weight,
        js_context_seconds=args.key_boundary_js_context_seconds,
        js_gap_seconds=args.key_boundary_js_gap_seconds,
    )
    chord_segments, chord_romanizer_status = respell_chord_segments_with_romanizer(
        chord_segments,
        key_segments,
    )
    if chord_romanizer_status == "unavailable":
        print("chord-romanizer is not installed; keeping the decoded spellings.")
    elif chord_romanizer_status in {"error", "partial"}:
        print(
            "Warning: chord-romanizer could not respell every chord; "
            "keeping the original spelling where needed."
        )
    elif chord_romanizer_status in {
        "applied_symbol_only",
        "applied_legacy_api",
    }:
        print(
            "Warning: this chord-romanizer version does not provide "
            "display_progression(); functional labels are unavailable."
        )
    # 結果表示
    print("\n--- 予測結果サマリー ---")
    print(f"ビートデコード方式: {beat_decode_mode}")
    print(f"検出ビート数: {len(beat_times)}")
    print(f"検出ダウンビート数: {len(downbeat_times)}")
    if meter_segments:
        meter_counter: dict[str, int] = {}
        for segment in meter_segments:
            meter_name = f"{segment.meter_num}/{segment.meter_den}"
            meter_counter[meter_name] = meter_counter.get(meter_name, 0) + 1
        meter_summary = ", ".join(
            f"{meter_name}: {count}"
            for meter_name, count in sorted(meter_counter.items())
        )
        print(f"選択meter: {meter_summary}")
    print(f"コード区間数: {len(chord_segments)}")
    print(f"キー区間数: {len(key_segments)}")

    # 最初の10個のコード進行を表示
    print("\n予測コード（最初の10区間）:")
    for segment in chord_segments[:10]:
        print(
            f"  {segment['start']:6.2f}s - {segment['end']:6.2f}s : "
            f"{segment.get('combined_label', segment['chord'])}"
        )
    if len(chord_segments) > 10:
        print("  ...")

    # 結果の保存
    if args.output_path is None:
        output_dir = Path("beat_chord_predictions")
        output_dir.mkdir(exist_ok=True)
        # 入力MIDI名に基づき、拡張子を変更して保存
        args.output_path = output_dir / f"{args.midi_path.stem}.prediction.json"

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    tempo_mapped_result = None
    if not args.disable_beat_mapped_midi:
        beat_mapped_midi_path = args.beat_mapped_midi_path
        if beat_mapped_midi_path is None:
            beat_mapped_midi_path = (
                args.output_path.parent / f"{args.midi_path.stem}.beat_mapped.mid"
            )
        tempo_mapped_result = export_tempo_mapped_midi(
            source_midi_path=args.midi_path,
            output_midi_path=beat_mapped_midi_path,
            beat_times=beat_times,
            meter_segments=[
                MeterSegmentSpec(
                    start_seconds=segment.start_frame * hop_length / sample_rate,
                    end_seconds=segment.end_frame * hop_length / sample_rate,
                    numerator=segment.meter_num,
                    denominator=segment.meter_den,
                    bar_count=segment.bar_count,
                    score=segment.score,
                )
                for segment in meter_segments
            ],
            chord_segments=chord_segments,
            key_segments=key_segments,
            duration_seconds=duration_seconds,
            ticks_per_beat=args.beat_mapped_ticks_per_beat,
        )
        print(f"Saved tempo-mapped MIDI: {beat_mapped_midi_path}")
        print(
            "Maximum absolute note timing drift: "
            f"{tempo_mapped_result.max_note_drift_seconds * 1000.0:.3f}ms"
        )
        if not tempo_mapped_result.used_predicted_tempo:
            print(
                "Warning: no valid meter interval was decoded; "
                "the original tempo map was retained and chords/keys were added."
            )

    output_data = {
        "song_name": args.midi_path.stem,
        "beat_decode_mode": beat_decode_mode,
        "decoder_diagnostics": {
            **decoder_diagnostics,
            "seconds_per_frame": float(hop_length / sample_rate),
        },
        "beats": beat_times,
        "downbeats": downbeat_times,
        "mapped_downbeats": mapped_downbeat_times,
        "meters": [
            {
                "start": round(segment.start_frame * hop_length / sample_rate, 3),
                "end": round(segment.end_frame * hop_length / sample_rate, 3),
                "meter_index": int(segment.meter_index),
                "meter": f"{segment.meter_num}/{segment.meter_den}",
                "bar_count": int(segment.bar_count),
                "source": "interpolated" if segment.bar_count > 1 else "detected",
                "score": float(segment.score),
                "tempo_bpm": (
                    None
                    if getattr(segment, "quarter_note_bpm", None) is None
                    else float(segment.quarter_note_bpm)
                ),
                "score_components": getattr(segment, "score_components", None),
                "confidence_margin": getattr(segment, "confidence_margin", None),
                "meter_evidence_source": getattr(
                    segment, "meter_evidence_source", "direct"
                ),
                "major_grouping": (
                    None
                    if getattr(segment, "major_grouping", None) is None
                    else list(segment.major_grouping)
                ),
            }
            for segment in meter_segments
        ],
        "chords": chord_segments,
        "chord_romanizer": chord_romanizer_status,
        "keys": key_segments,
        "key_representation": "relative_major",
        "tempo_mapped_midi": (
            None if tempo_mapped_result is None else tempo_mapped_result.to_json()
        ),
    }

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n予測結果を保存しました: {args.output_path}")

    # Audacity用ラベル形式の保存
    audacity_dir = args.output_path.parent / "audacity"
    audacity_dir.mkdir(exist_ok=True)
    audacity_path = audacity_dir / f"{args.midi_path.stem}.txt"
    with open(audacity_path, "w", encoding="utf-8") as f:
        for segment in chord_segments:
            f.write(
                f"{segment['start']:.6f}\t{segment['end']:.6f}\t"
                f"{segment.get('combined_label', segment['chord'])}\n"
            )
    print(f"Audacity用ラベルを保存しました: {audacity_path}")


if __name__ == "__main__":
    main()
