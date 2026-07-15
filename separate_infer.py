# @title Run stem-separated transcription
OUTPUT_ROOT = "U:"  # @param {type:"string"}
WINDOW_BATCH_SIZE = 4  # @param {type:"integer"}
MAX_MIDI_MELODIC_INSTRUMENTS = 15  # @param {type:"integer"}
CLEANUP_SEPARATED_STEMS = True  # @param {type:"boolean"}
MERGE_ONSET_MS = 50.0  # @param {type:"number"}

from pathlib import Path
import shutil
import glob
from tqdm import tqdm

from separate_helper import run_stem_separated_transcription

import argparse
import os


def safe_glob(pattern):
    dir_part, file_pattern = os.path.split(pattern)
    escaped_dir = glob.escape(dir_part)
    full_pattern = os.path.join(escaped_dir, file_pattern)
    return glob.glob(full_pattern)

parser = argparse.ArgumentParser(description="Run stem-separated transcription")
parser.add_argument(
    "--audio_paths", '-a', 
    type=str, 
    nargs='+',         
    required=True, 
    help="One or more glob patterns for input audio files"
)
args = parser.parse_args()

# Collect audio files from the provided glob patterns
audio_files = set()
for pattern in args.audio_paths:
    for file_path in safe_glob(pattern):
        audio_files.add(file_path)

if not audio_files:
    raise RuntimeError("No audio files found matching the provided glob patterns.")

# Convert to a sorted list for consistent processing order
audio_files = sorted(audio_files) 

# Process each audio file
for audio_path in tqdm(audio_files, desc="Processing audio files"):
    print(f"Processing {audio_path}...")

    try:
        # Check if the audio file exists
        if not Path(audio_path).exists():
            print(f"Audio file not found: {audio_path}")
            continue
        # Check if the output MIDI file already exists
        output_midi_path = Path(audio_path).with_suffix('.mid')
        if output_midi_path.exists():
            print(f"Output MIDI file already exists, skipping: {output_midi_path}")
            continue

        stem_pipeline_result = run_stem_separated_transcription(
        audio_path,
        checkpoint_path=None,
        output_root=OUTPUT_ROOT,
        window_batch_size=WINDOW_BATCH_SIZE,
        max_midi_melodic_instruments=MAX_MIDI_MELODIC_INSTRUMENTS,
        cleanup_separated_stems=CLEANUP_SEPARATED_STEMS,
        merge_onset_ms=MERGE_ONSET_MS,
    )
        stem_pipeline_result

        merged_midi_path = Path(stem_pipeline_result["merged_midi_path"])

        # Move the merged MIDI file to the audio file's directory
        audio_dir = Path(audio_path).parent
        new_midi_path = audio_dir / merged_midi_path.name
        shutil.move(merged_midi_path, new_midi_path)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")    