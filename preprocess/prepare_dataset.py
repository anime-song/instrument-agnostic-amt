import argparse
import csv
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
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
from instrument_agnostic_amt.taxonomy.instrument_classes import (
    get_instrument_class_id,
    get_instrument_class_id_by_name,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MELODY_CLASS_ID = get_instrument_class_id_by_name("melody")
VOCAL_HARMONY_CLASS_ID = get_instrument_class_id_by_name("vocal_harmony")
DRUM_CLASS_ID = get_instrument_class_id_by_name("drums")
TIMPANI_CLASS_ID = get_instrument_class_id_by_name("timpani")
WIND_CHIMES_CLASS_ID = get_instrument_class_id_by_name("wind_chimes")
LABEL_MODE_MELODIC = "melodic"
LABEL_MODE_DRUM = "drum"
LABEL_MODE_ALL = "all"
MELODY_TRACK_KEYWORDS = ("vocal", "melody")
VOCAL_HARMONY_TRACK_KEYWORDS = ("vocal_harmony", "harmony")
DRUM_TRACK_KEYWORDS = ("drum",)
DRUM_TRACK_NAMES = ("percussion",)
TIMPANI_TRACK_KEYWORDS = ("timpani", "timpany")
WIND_CHIMES_TRACK_KEYWORDS = (
    "windchime",
    "wind chime",
    "wind-chime",
    "windbell",
    "wind bell",
    "w.chime",
)


def is_vocal_harmony_track(instrument: pretty_midi.Instrument) -> bool:
    """Return True when the MIDI track name looks like vocal harmony."""
    name = (instrument.name or "").lower()
    return any(keyword in name for keyword in VOCAL_HARMONY_TRACK_KEYWORDS)


def is_melody_track(instrument: pretty_midi.Instrument) -> bool:
    """Return True for vocal/melody tracks, excluding vocal_harmony tracks."""
    if is_vocal_harmony_track(instrument):
        return False
    name = (instrument.name or "").lower()
    return any(keyword in name for keyword in MELODY_TRACK_KEYWORDS)


def is_named_drum_track(instrument: pretty_midi.Instrument) -> bool:
    """Return True when a non-drum MIDI track is named like a drum/percussion track."""
    name = (instrument.name or "").strip().lower()
    return name in DRUM_TRACK_NAMES or any(
        keyword in name for keyword in DRUM_TRACK_KEYWORDS
    )


def is_timpani_track(instrument: pretty_midi.Instrument) -> bool:
    name = (instrument.name or "").strip().lower()
    return instrument.program == 47 or any(
        keyword in name for keyword in TIMPANI_TRACK_KEYWORDS
    )


def is_wind_chimes_track(instrument: pretty_midi.Instrument) -> bool:
    name = (instrument.name or "").strip().lower()
    return any(keyword in name for keyword in WIND_CHIMES_TRACK_KEYWORDS)


def get_drum_mode_class_id(instrument: pretty_midi.Instrument) -> int | None:
    if instrument.is_drum or is_named_drum_track(instrument):
        return DRUM_CLASS_ID
    if is_timpani_track(instrument):
        return TIMPANI_CLASS_ID
    if is_wind_chimes_track(instrument):
        return WIND_CHIMES_CLASS_ID
    return None


def is_excluded_instrument(instrument: pretty_midi.Instrument) -> bool:
    """
    Exclude drum-like tracks and GM programs that should not be used as melodic labels.

    Excluded:
      - is_drum == True (usually MIDI channel 10)
      - track name is exactly percussion
      - track name contains drum
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
    mid_path: Path | None,
    wav_path: Path,
    npz_dir: Path,
    manifest_dir: Path,
    *,
    label_mode: str = LABEL_MODE_MELODIC,
    drum_note_duration_ms: int = 80,
) -> dict[str, Any] | None:
    """
    Process one audio stem and save MIDI-derived note arrays to an NPZ file.

    If MIDI is missing, or no usable notes are found, an empty label NPZ is still
    written. Tracks whose names contain vocal/melody are assigned to the special
    melody class regardless of their GM program.
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
            if label_mode in (LABEL_MODE_DRUM, LABEL_MODE_ALL):
                inst_id = get_drum_mode_class_id(instrument)
                if inst_id is not None:
                    use_fixed_duration = instrument.is_drum or is_named_drum_track(
                        instrument
                    )
                    for note in sorted(instrument.notes, key=lambda x: x.start):
                        start_ms = int(round(note.start * 1000.0))
                        if use_fixed_duration:
                            end_ms = start_ms + int(drum_note_duration_ms)
                        else:
                            end_ms = int(round(note.end * 1000.0))
                        all_start_ms.append(start_ms)
                        all_end_ms.append(max(end_ms, start_ms + 1))
                        all_pitch.append(note.pitch)
                        all_velocity.append(note.velocity)
                        all_instrument_id.append(inst_id)
                    continue

                if label_mode == LABEL_MODE_DRUM:
                    continue

            is_harmony = is_vocal_harmony_track(instrument)
            is_named_melody = is_melody_track(instrument)
            if (
                not is_named_melody
                and not is_harmony
                and is_excluded_instrument(instrument)
            ):
                continue

            # Track-name hints take precedence over GM programs for
            # vocal/melody/vocal_harmony labels.
            if is_harmony:
                inst_id = VOCAL_HARMONY_CLASS_ID
            elif is_named_melody:
                inst_id = MELODY_CLASS_ID
            else:
                inst_id = get_instrument_class_id(
                    instrument.program, instrument.is_drum
                )

            # Extend note ends through CC64 sustain pedal intervals.
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

            pedal_starts = [start for start, _ in pedal_intervals]
            extended_ends = []
            for note in notes:
                new_end = note.end
                pedal_index = bisect_right(pedal_starts, note.end) - 1
                if pedal_index >= 0:
                    p_start, p_end = pedal_intervals[pedal_index]
                    if p_start <= note.end < p_end:
                        new_end = p_end
                extended_ends.append(new_end)

            # For repeated notes of the same pitch, trim each note at the next onset.
            pitch_indices_by_pitch: dict[int, list[int]] = {}
            for note_index, note in enumerate(notes):
                pitch_indices_by_pitch.setdefault(int(note.pitch), []).append(
                    note_index
                )
            for pitch_indices in pitch_indices_by_pitch.values():
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
        # Convert to numpy arrays and sort by onset time.
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

    # Save note labels to NPZ.
    npz_path = npz_dir / f"{wav_path.stem}.npz"
    np.savez_compressed(
        npz_path,
        note_start_ms=start_ms,
        note_end_ms=end_ms,
        note_pitch=pitch,
        note_velocity=velocity,
        note_instrument=instrument_ids,
    )

    # Read audio duration.
    try:
        info = sf.info(str(wav_path))
        sample_rate = int(info.samplerate)
        duration_ms = int(round((info.frames / sample_rate) * 1000.0))
    except Exception as e:
        logger.warning(f"Failed to read audio info for {wav_path}: {e}")
        return None

    # Use the part before "__" as song_name.
    # Keep pitch/stretch/swing suffixes in song_name so augmented variants are
    # sampled independently from their original stems.
    import re

    stem_name = wav_path.stem
    match = re.search(r"((?:_(?:pitch|stretch|swing)_[^_]+)+)$", stem_name)
    suffix = match.group(1) if match else ""
    base_name = stem_name[: match.start()] if match else stem_name

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


def process_stem_task(
    task: tuple[Path | None, Path, Path, Path, str, int],
) -> dict[str, Any] | None:
    mid_path, wav_path, npz_dir, manifest_dir, label_mode, drum_note_duration_ms = task
    return process_stem(
        mid_path,
        wav_path,
        npz_dir,
        manifest_dir,
        label_mode=label_mode,
        drum_note_duration_ms=drum_note_duration_ms,
    )


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
        help="Only include stems that have a matching MIDI file.",
    )
    parser.add_argument(
        "--label-mode",
        choices=(LABEL_MODE_MELODIC, LABEL_MODE_DRUM, LABEL_MODE_ALL),
        default=LABEL_MODE_MELODIC,
        help=(
            "Label extraction mode. Use 'all' for melodic plus drums/percussion; "
            "use 'drum' for drums/percussion only."
        ),
    )
    parser.add_argument(
        "--drum-note-duration-ms",
        type=int,
        default=80,
        help="Fixed label duration for drum notes in --label-mode drum/all.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker processes for MIDI/audio metadata extraction.",
    )
    args = parser.parse_args()
    if args.drum_note_duration_ms <= 0:
        raise ValueError("--drum-note-duration-ms must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")

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

    tasks: list[tuple[Path | None, Path, Path, Path, str, int]] = []
    skipped_no_midi = 0
    for wav_path in wav_files:
        mid_path = midis_dir / f"{wav_path.stem}.mid"
        if not mid_path.exists():
            mid_path = midis_dir / f"{wav_path.stem}.midi"
        target_mid_path = mid_path if mid_path.exists() else None

        # --require-midi skips stems without MIDI.
        if args.require_midi and target_mid_path is None:
            skipped_no_midi += 1
            continue

        tasks.append(
            (
                target_mid_path,
                wav_path,
                npz_dir,
                manifest_path.parent,
                args.label_mode,
                args.drum_note_duration_ms,
            )
        )

    rows = []
    if args.workers == 1 or len(tasks) <= 1:
        row_iter = (process_stem_task(task) for task in tasks)
        for row in tqdm(row_iter, total=len(tasks), desc="Processing stems"):
            if row:
                rows.append(row)
    else:
        worker_count = min(args.workers, len(tasks))
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            row_iter = executor.map(process_stem_task, tasks, chunksize=8)
            for row in tqdm(
                row_iter,
                total=len(tasks),
                desc=f"Processing stems ({worker_count} workers)",
            ):
                if row:
                    rows.append(row)

    if skipped_no_midi > 0:
        logger.info(f"Skipped {skipped_no_midi} stems without MIDI (--require-midi)")

    if not rows:
        logger.warning("No valid stems were processed.")
        return

    # Write manifest CSV.
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
