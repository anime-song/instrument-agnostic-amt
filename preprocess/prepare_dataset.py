import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi
import soundfile as sf
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from instrument_classes import get_instrument_class_id, get_instrument_class_id_by_name

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MELODY_CLASS_ID = get_instrument_class_id_by_name("melody")
VOCAL_HARMONY_CLASS_ID = get_instrument_class_id_by_name("vocal_harmony")
MELODY_TRACK_KEYWORDS = ("vocal", "melody")
VOCAL_HARMONY_TRACK_KEYWORDS = ("vocal_harmony", "harmony")
DRUM_TRACK_KEYWORDS = ("drum",)
DRUM_TRACK_NAMES = ("percussion",)


def is_vocal_harmony_track(instrument: pretty_midi.Instrument) -> bool:
    """MIDIトラック名に vocal_harmony または harmony が含まれるかを判定する。"""
    name = (instrument.name or "").lower()
    return any(keyword in name for keyword in VOCAL_HARMONY_TRACK_KEYWORDS)


def is_melody_track(instrument: pretty_midi.Instrument) -> bool:
    """MIDIトラック名に vocal / melody が含まれるかを判定する。ただし vocal_harmony は除外。"""
    if is_vocal_harmony_track(instrument):
        return False
    name = (instrument.name or "").lower()
    return any(keyword in name for keyword in MELODY_TRACK_KEYWORDS)


def is_named_drum_track(instrument: pretty_midi.Instrument) -> bool:
    """MIDIトラック名からドラム/打楽器トラックを判定する。"""
    name = (instrument.name or "").strip().lower()
    return name in DRUM_TRACK_NAMES or any(
        keyword in name for keyword in DRUM_TRACK_KEYWORDS
    )


def is_excluded_instrument(instrument: pretty_midi.Instrument) -> bool:
    """
    ドラム、または音高ラベルとして扱いにくいGM音色を除外する。

    除外対象:
      - is_drum == True (通常MIDI Channel 10)
      - トラック名が percussion
      - トラック名に drum を含む
      - Timpani (47)
      - Synth Effects Family (96-103)
      - Percussive Family (112-119)
      - Sound Effects Family (120-127)
    """
    if instrument.is_drum or is_named_drum_track(instrument):
        return True

    prog = instrument.program
    if (
        prog == 47
        or (96 <= prog <= 103)
        or (112 <= prog <= 119)
        or (120 <= prog <= 127)
    ):
        return True

    return False


def process_stem(
    mid_path: Path | None, wav_path: Path, npz_dir: Path, manifest_dir: Path
) -> dict[str, Any] | None:
    """
    1つの音声ステムを処理し、MIDI由来のノート配列をnpzに保存する。

    MIDIがない場合、または有効なノートがない場合も、空のラベルnpzを出力する。
    トラック名に vocal / melody が含まれるトラックは、GM programに関係なく
    追加クラス melody として保存する。
    """
    midi_data = None
    if mid_path is not None and mid_path.exists():
        try:
            midi_data = pretty_midi.PrettyMIDI(str(mid_path))
        except Exception as e:
            logger.warning(f"Failed to parse MIDI {mid_path}: {e}")

    all_start_ms = []
    all_end_ms = []
    all_pitch = []
    all_velocity = []
    all_instrument_id = []

    if midi_data is not None:
        for instrument in midi_data.instruments:
            is_harmony = is_vocal_harmony_track(instrument)
            is_named_melody = is_melody_track(instrument)
            if not is_named_melody and not is_harmony and is_excluded_instrument(instrument):
                continue

            # vocal/melody/vocal_harmony と明示されたトラックは、元のGM音色よりトラック名を優先する。
            # これにより、Voice/Flute/Synthなどに分散していたメロディを1クラスに集約できる。
            if is_harmony:
                inst_id = VOCAL_HARMONY_CLASS_ID
            elif is_named_melody:
                inst_id = MELODY_CLASS_ID
            else:
                inst_id = get_instrument_class_id(
                    instrument.program, instrument.is_drum
                )

            # CC64 sustain pedal をノート終端へ反映する。
            pedal_events = [cc for cc in instrument.control_changes if cc.number == 64]
            pedal_intervals = []
            current_pedal_on = None
            for cc in sorted(pedal_events, key=lambda x: x.time):
                if cc.value >= 64 and current_pedal_on is None:
                    current_pedal_on = cc.time
                elif cc.value < 64 and current_pedal_on is not None:
                    pedal_intervals.append((current_pedal_on, cc.time))
                    current_pedal_on = None

            if current_pedal_on is not None:
                pedal_intervals.append((current_pedal_on, float("inf")))

            notes = sorted(instrument.notes, key=lambda x: x.start)
            max_original_end = max((n.end for n in notes), default=0.0)

            extended_ends = []
            for note in notes:
                new_end = note.end
                for p_start, p_end in pedal_intervals:
                    if p_start <= note.end < p_end:
                        new_end = p_end
                        break
                extended_ends.append(new_end)

            # 同じ pitch の重複ノートは、次の onset 直前で前ノートを切る。
            for pitch in range(128):
                pitch_indices = [
                    i for i, note in enumerate(notes) if note.pitch == pitch
                ]
                for i in range(len(pitch_indices) - 1):
                    idx = pitch_indices[i]
                    next_idx = pitch_indices[i + 1]
                    if extended_ends[idx] > notes[next_idx].start:
                        extended_ends[idx] = notes[next_idx].start

            for note, new_end in zip(notes, extended_ends):
                if new_end == float("inf"):
                    new_end = max_original_end
                new_end = max(new_end, note.start)

                all_start_ms.append(int(round(note.start * 1000.0)))
                all_end_ms.append(int(round(new_end * 1000.0)))
                all_pitch.append(note.pitch)
                all_velocity.append(note.velocity)
                all_instrument_id.append(inst_id)

    if not all_start_ms:
        start_ms = np.array([], dtype=np.int64)
        end_ms = np.array([], dtype=np.int64)
        pitch = np.array([], dtype=np.int16)
        velocity = np.array([], dtype=np.int16)
        instrument_ids = np.array([], dtype=np.int16)
        note_count = 0
        end_note_ms = 0
    else:
        # Numpy array に変換し、開始時刻でソートする。
        start_ms = np.array(all_start_ms, dtype=np.int64)
        end_ms = np.array(all_end_ms, dtype=np.int64)
        pitch = np.array(all_pitch, dtype=np.int16)
        velocity = np.array(all_velocity, dtype=np.int16)
        instrument_ids = np.array(all_instrument_id, dtype=np.int16)

        sort_idx = np.argsort(start_ms)
        start_ms = start_ms[sort_idx]
        end_ms = end_ms[sort_idx]
        pitch = pitch[sort_idx]
        velocity = velocity[sort_idx]
        instrument_ids = instrument_ids[sort_idx]

        note_count = len(start_ms)
        end_note_ms = int(np.max(end_ms))

    # NPZに保存する。
    npz_path = npz_dir / f"{wav_path.stem}.npz"
    np.savez_compressed(
        npz_path,
        note_start_ms=start_ms,
        note_end_ms=end_ms,
        note_pitch=pitch,
        note_velocity=velocity,
        note_instrument=instrument_ids,
    )

    # 音声の長さを取得する。
    try:
        info = sf.info(str(wav_path))
        sample_rate = int(info.samplerate)
        duration_ms = int(round((info.frames / sample_rate) * 1000.0))
    except Exception as e:
        logger.warning(f"Failed to read audio info for {wav_path}: {e}")
        return None

    # song_name は "__" より前の部分を使う。
    # ピッチシフトやタイムストレッチのサフィックスがある場合は別曲扱いとして付加する。
    import re
    stem_name = wav_path.stem
    match = re.search(r"((?:_(?:pitch|stretch)_[^_]+)+)$", stem_name)
    suffix = match.group(1) if match else ""
    base_name = stem_name[:match.start()] if match else stem_name

    song_name = base_name.split("__")[0] if "__" in base_name else base_name
    song_name += suffix

    return {
        "song_name": song_name,
        "stem_name": stem_name,
        "wav_path": os.path.relpath(wav_path, manifest_dir).replace("\\", "/"),
        "npz_path": os.path.relpath(npz_path, manifest_dir).replace("\\", "/"),
        "duration_ms": duration_ms,
        "end_note_ms": end_note_ms,
        "note_count": note_count,
        "sample_rate": sample_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from midis and stems")
    parser.add_argument(
        "--midis_dir",
        type=Path,
        default=Path("./stem_midis"),
        help="Path to stem midis directory",
    )
    parser.add_argument(
        "--stems_dir",
        type=Path,
        default=Path("./stems"),
        help="Path to audio stems directory",
    )
    parser.add_argument(
        "--npz_dir",
        type=Path,
        default=Path("./stem_npz"),
        help="Path to save processed npz files",
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=Path("./manifest.csv"),
        help="Path to save the manifest CSV",
    )
    parser.add_argument(
        "--require-midi",
        action="store_true",
        help="MIDIファイルが存在するステムのみをマニフェストに含める",
    )
    args = parser.parse_args()

    midis_dir = args.midis_dir.resolve()
    stems_dir = args.stems_dir.resolve()
    npz_dir = args.npz_dir.resolve()
    manifest_path = args.manifest_path.resolve()

    npz_dir.mkdir(parents=True, exist_ok=True)

    if not midis_dir.exists() or not stems_dir.exists():
        logger.error(
            f"Directories not found. Check midis_dir={midis_dir} and stems_dir={stems_dir}."
        )
        return

    wav_files = list(stems_dir.glob("*.wav")) + list(stems_dir.glob("*.flac"))
    logger.info(f"Found {len(wav_files)} audio files.")

    rows = []
    skipped_no_midi = 0
    for wav_path in tqdm(wav_files, desc="Processing stems"):
        mid_path = midis_dir / f"{wav_path.stem}.mid"
        if not mid_path.exists():
            mid_path = midis_dir / f"{wav_path.stem}.midi"
        target_mid_path = mid_path if mid_path.exists() else None

        # --require-midi: MIDIがないステムはスキップする
        if args.require_midi and target_mid_path is None:
            skipped_no_midi += 1
            continue

        row = process_stem(target_mid_path, wav_path, npz_dir, manifest_path.parent)
        if row:
            rows.append(row)

    if skipped_no_midi > 0:
        logger.info(f"Skipped {skipped_no_midi} stems without MIDI (--require-midi)")

    if not rows:
        logger.warning("No valid stems were processed.")
        return

    # マニフェストCSVを書き出す。
    fieldnames = [
        "song_name",
        "stem_name",
        "wav_path",
        "npz_path",
        "duration_ms",
        "end_note_ms",
        "note_count",
        "sample_rate",
    ]
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} entries to {manifest_path}")


if __name__ == "__main__":
    main()
