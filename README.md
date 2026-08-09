# Instrument-Agnostic Automatic Music Transcription

**Transcribe any instrument to MIDI** — Neural Semi-CRF based AMT

[日本語版 README はこちら](README_ja.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anime-song/instrument-agnostic-amt/blob/main/Colab_Inference.ipynb)

<table>
  <tr>
    <td align="center">
      <a href="https://youtu.be/3pCAjQuhzDA">
        <img src="https://img.youtube.com/vi/3pCAjQuhzDA/0.jpg" alt="Transcription example" width="480">
      </a>
      <br>
      <strong>Transcription example</strong>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=JuVu-AoC5M0">
        <img src="https://img.youtube.com/vi/JuVu-AoC5M0/0.jpg" alt="Original source video" width="480">
      </a>
      <br>
      <strong>Original source video</strong>
    </td>
  </tr>
</table>

> **Video note**: The images above are clickable thumbnails. Click either one to watch the videos on YouTube.

> **Colab tip**: [`Colab_Inference.ipynb`](Colab_Inference.ipynb) also includes an optional **stem-separated transcription** workflow: separate the song into stems, transcribe each stem, merge the MIDI files, then predict the velocity of each note from the separated audio. This often gives better results than transcribing the full mix directly, especially for dense arrangements with overlapping instruments. Velocity prediction is enabled by default in this workflow.

---

## What is this?

This project is an **instrument-agnostic Automatic Music Transcription (AMT)** model that converts audio into MIDI.
Like [Basic Pitch](https://github.com/spotify/basic-pitch), it doesn't distinguish between instruments — piano, guitar, bass, vocals, strings, brass — if it has pitch, the model will transcribe it. One model handles everything.

The architecture builds on [**Transkun**](https://github.com/Yujia-Yan/Transkun) (Yujia Yan et al.) and its Neural Semi-CRF approach, originally designed for piano transcription. This project extends it into a general-purpose model that works across all pitched instruments.

> **Note**: There's also an experimental multi-track MIDI output with instrument classification, but classification accuracy is still limited. The core feature is instrument-agnostic pitch detection.

> **Note**: A dedicated drum model is available via `--type drums`, but it is still **Experimental**. Accuracy and behavior may change as the model evolves.

> **Warning**: Generalization to electric guitar (especially with distortion) is still weak, and transcription accuracy tends to be lower. The same applies to ethnic instruments (e.g. shamisen, sitar) that are underrepresented in the training data.

### Changelog

| Date | Update |
|---|---|
| 2026-08-09 | 🎹 Added the Instrument Refinement model, which re-assigns instrument classes to AMT notes using the separated stem audio. It pulls timbrally close sounds onto a single class, so the instrument stops flickering within a piece and manual clean-up gets easier. Overall top-1 on the held-out RWC-I benchmark improves from 71.3% to 74.5%, but **individual instruments both gain and lose** — see [RWC-I benchmark](instrument_agnostic_amt/instrument_refinement/RWC_BENCHMARK.md) for the per-instrument breakdown and what each instrument is confused with. |
| 2026-07-30 | Added beat, chord, and key inference models to the pipeline. Disabled by default. |
| 2026-07-24 | Added a dedicated velocity prediction model that estimates per-note dynamics from separated stem audio. The Colab stem-separated workflow now enables velocity prediction by default and automatically downloads the velocity checkpoint from Hugging Face. |
| 2026-07-22 | Added guitar model v1.5 (`--type guitar_v1_5`). The Colab stem-separated workflow now uses it by default for guitar stems. |
| 2026-07-16 | 🐛 Fixed a bug in the data augmentation pipeline that caused note timing misalignment, then retrained the bass v2 model (`--type bass_v2`) after the fix. The retrained `bass_v2` improves note detection accuracy and fixes the issue where notes were over-segmented into multiple short notes. These improvements apply only to `bass_v2`. |
| 2026-07-15 | 🎸 Added the updated bass model (`--type bass_v2`), with improved slap bass classification in the instrument classification output. |
| 2026-07-12 | 🎯 Added per-stem instrument class selection. Excluding implausible instruments before probability calculation is expected to reduce instrument misclassification in the stem-separated workflow. |
| 2026-06-24 | 🥁 Added experimental drum-focused inference model (`--type drums`) |
| 2026-06-05 | 🎻 Added other-instrument-focused model (`--type other`) |
| 2026-05-31 | 🎤 Added vocal harmony model (`--type vocal_harmony`). Added `vocal_harmony` class to the instrument taxonomy to identify harmony.<br>🧩 Added Pitch Slot feature to predict overlapping note intervals simultaneously. |
| 2026-05-20 | 🎸 Added guitar-focused model (`--type guitar`) |
| 2026-05-18 | 📦 Added pitch-shift / time-stretch preprocessing scripts |
| 2026-05-17 | 🎤 Added vocal-focused model (`--type vocal`) |
| 2026-05-16 | 🎸 Added bass-focused model (`--type bass`) |
| 2026-05-09 | 🔧 Fixed cross-window note stitching / Added beat & chord training |
| 2026-05-09 | 🎼 Added stem-separated transcription workflow to Colab |
| 2026-05-06 | 🥁 Improved drum detection / Added new augmentations |
| 2026-05-05 | ✨ Added EMA, instrument loss masking, batch directory inference |
| 2026-05-03 | 🚀 Initial release — multi-instrument AMT model & Colab notebook |

### Features

- 🎹 **Works with any instrument** — Piano, guitar, bass, vocals, strings, wind instruments, and more
- 🧠 **Neural Semi-CRF + Pitch Slot** — Viterbi decoding finds globally optimal note intervals for each pitch, while Pitch Slots allow predicting overlapping notes of the same pitch.
- 🎼 **HCQT features** — 5 harmonics × stereo 2ch Harmonic CQT captures rich pitch information
- 🎚️ **Per-note velocity prediction** — A dedicated post-processing model estimates MIDI note dynamics from separated stem audio
- 🔧 **Extensive data augmentation** — Stem mixing, IR reverb, EQ, noise injection, drum addition, and more
- 🧪 **[Experimental] Instrument classification & multi-track output** — 33+ instrument class head for per-instrument MIDI tracks (accuracy still improving)

## Known Limitations

This project is still under active development. Depending on the source audio and performance, the following issues may occur.

### Instrument classification

Instrument classification and multi-track MIDI output are experimental features.

- During bass transcription, slap bass and synth bass may not be recognized as distinct instrument classes and may instead be classified as `electric bass`.
- During piano transcription, electric and acoustic pianos are frequently confused due to limited classification accuracy between the two classes.
- Instruments such as sitar and banjo that bleed into a separated guitar stem may not be assigned to the correct instrument class.

### Transcription accuracy

- Fast vocal passages, swing phrasing, and other complex phrases may produce inaccurate pitches, note boundaries, or durations.
- In some cases, a single sustained note may be split into multiple short notes (note over-segmentation).

### Stem-separated workflow

- Transcribing separated stems independently may introduce small timing offsets between the resulting MIDI files, causing synchronization issues after they are merged.

---

## Architecture

```
Audio Waveform [B, 2, T]
        │
        ▼
┌─────────────────────────────┐
│  AudioFeatureExtractor      │
│  (Harmonic CQT × 5)        │   → [B, 10, F=312, T]
│  + SpecAugment (training)   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  StemConv                   │
│  (2D CNN downsampling)      │   → [B, D, T/8, F/4]
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Backbone (Dual-Axis        │
│  Transformer × N layers)    │
│  + Pitch Query Embedding    │   → Band features + Pitch-wise features
│  + Transposed ConvUpsample  │
└─────────────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌───────────────────┐   ┌───────────────────────┐
│ Interval Adapter  │   │ Instrument Adapter    │
│ + IntervalScorer  │   │ + Classifier (33 cls) │
│   (Q, K, Diag)    │   └───────────────────────┘
└───────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ Neural Semi-CRF               │
│ (per-pitch Viterbi decoding)  │  → Note intervals [begin, end] per pitch
│ + Boundary Predictor          │  → Onset/Offset presence & sub-frame offsets
└───────────────────────────────┘
        │
        ▼
    MIDI Output
```

### Dual-Axis Transformer

The backbone processes two types of tokens together:

- **Band tokens** — frequency band features from the CNN stem
- **Pitch query tokens** — learnable embeddings for MIDI pitches 21–108

Each layer alternates between a **band-axis Transformer** (attends across all tokens at each time step) and a **time-axis Transformer** (attends across time for each token). This lets frequency and pitch information mix effectively.

### Neural Semi-CRF

Each of the 88 pitch tracks is modeled as an independent semi-CRF:

- **Pitch Slots** — Processes multiple slots in parallel to predict overlapping notes of the same pitch.
- **Interval score** — bilinear attention between query and key projections
- **Diagonal score** — additive bias for single-frame notes
- **Viterbi decoding** — finds the globally optimal set of non-overlapping note intervals
- **Boundary head** — predicts onset/offset presence and sub-frame timing corrections

---

## Project Structure

```
instrument_agnostic_amt/
├── train.py                    # Training loop (AMP, W&B, warmup)
├── infer.py                    # Inference: audio → MIDI
├── dataset.py                  # StemDataset with stem mixing augmentation
├── losses.py                   # Loss: Semi-CRF NLL + boundary + instrument classification
├── augmentation.py             # AudioAugmentor (EQ, pitch shift, reverb, noise, etc.)
├── instrument_classes.py       # Instrument class mapping (GM program ↔ class ID)
├── instrument_merge.json       # Instrument taxonomy definition
├── gm_instrument_classes.json  # General MIDI metadata
├── dataset_config.yaml         # Multi-dataset weighted sampling config
├── requirements.txt            # Dependencies
│
├── models/
│   ├── model.py                # AudioSemiCRFTransformer (top-level model)
│   ├── transcription_model.py  # Feature extraction, StemConv, Backbone
│   ├── transformer.py          # RoPE Transformer with gated attention
│   ├── cqt.py                  # RecursiveCQT (fast octave-recursive CQT)
│   ├── semi_crf.py             # Neural Semi-CRF (forward-backward, Viterbi, loss)
│   ├── interval_boundaries.py  # Interval boundary feature gathering
│   └── spec_augment.py         # SpecAugment & MiniBatch Mixture Masking
│
└── preprocess/
    ├── prepare_dataset.py      # Generate manifest.csv from audio/MIDI pairs
    ├── resample_only.py        # Batch resampling
    └── apply_ir_augmentation.py # Offline IR convolution for reverb augmentation
```

---

## Installation

### Requirements

- Python 3.10+
- CUDA GPU (12GB+ VRAM recommended)

```bash
# Clone
git clone https://github.com/anime-song/instrument-agnostic-amt.git
cd instrument-agnostic-amt

# Virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Dependencies
pip install -r requirements.txt
```

> `audiomentations` is needed for training augmentation. You can skip it if you only need inference.

---

## Data Preparation

### 1. Organize your files

Put your stem audio and matching MIDI files in the following structure:

```
stems/          # Audio (.wav / .flac)
  ├── song1__piano.wav
  ├── song1__guitar.wav
  ├── song2__vocal.wav
  └── ...

stem_midis/     # Matching MIDI files
  ├── song1__piano.mid
  ├── song1__guitar.mid
  ├── song2__vocal.mid
  └── ...
```

**Naming convention**: `<song_name>__<instrument_name>.wav`
- `__` (double underscore) separates the song name from the instrument
- Stems with the same song name are treated as parts of the same song

### 2. Generate manifest

```bash
python preprocess/prepare_dataset.py \
  --stems_dir ./stems \
  --midis_dir ./stem_midis \
  --npz_dir ./stem_npz \
  --manifest_path ./manifest.csv
```

This creates:
- **`stem_npz/`** — preprocessed note arrays (start/end times, pitch, velocity, instrument ID)
- **`manifest.csv`** — dataset index

### 3. (Optional) Resample audio

If your audio files are not at 22050 Hz:

```bash
python preprocess/resample_only.py \
  --input_dir ./raw_stems \
  --output_dir ./stems \
  --target_sr 22050
```

## Training

### Quick start

```bash
python train.py \
  --manifest_path manifest.csv \
  --batch_size 8 \
  --lr 5e-4 \
  --epochs 3000 \
  --save_dir checkpoints \
  --wandb
```

### Full augmentation

```bash
python train.py \
  --dataset_config dataset_config.yaml \
  --batch_size 8 \
  --lr 5e-4 \
  --warmup_steps 1000 \
  --epochs 3000 \
  --ir_folder ./IRs \
  --noise_folder ./noise \
  --drum_folder ./drum_stems \
  --p_augment 1.0 \
  --p_intra_drop 0.3 \
  --p_cross_mix 0.5 \
  --p_drum_mix 0.1 \
  --sa_p 0.5 --sa_freq_max 10 --sa_time_max 20 --sa_num_freq 2 --sa_num_time 2 \
  --wandb --project_name instrument_agnostic_amt
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_config` | `dataset_config.yaml` | Weighted multi-dataset config |
| `--batch_size` | `8` | Batch size |
| `--lr` | `5e-4` | Learning rate (AdamW) |
| `--warmup_steps` | `1000` | LR warmup steps |
| `--window_ms` | `8000` | Input window length (ms) |
| `--p_intra_drop` | `0.3` | Probability of dropping stems from the same song |
| `--p_cross_mix` | `0.5` | Probability of mixing in stems from other songs |
| `--p_augment` | `1.0` | Probability of applying audio augmentation |
| `--init-from` | `None` | Checkpoint for weight initialization |
| `--no_amp` | `false` | Disable mixed precision |

### Multi-dataset config

`dataset_config.yaml` lets you mix multiple datasets with different weights:

```yaml
datasets:
  - name: main
    manifest: manifest.csv
    weight: 0.2
    use_for_cross_aug: true

  - name: maestro
    manifest: other_db/maestro_manifest.csv
    weight: 0.05
    use_for_cross_aug: true

  - name: musicnet
    manifest: other_db/musicnet_manifest.csv
    weight: 0.5
    use_for_cross_aug: false  # Don't use for cross-stem mixing
```

Use the optional `group` key when separate manifests contain stems rendered
from the same songs:

```yaml
datasets:
  - name: rendered_piano
    group: single_stems
    manifest: piano_stem_manifest.csv
    allow_multi_stem_same_song: true

  - name: rendered_strings
    group: single_stems
    manifest: strings_stem_manifest.csv
    allow_multi_stem_same_song: true
```

Entries with the same `group` and CSV `song_name` share one virtual song.
Consequently, `allow_multi_stem_same_song: true` can select stems across those
manifests. Dataset weights, augmentation settings, and cross-augmentation
eligibility remain per entry. Omitting `group` preserves the previous isolated
behavior by using `name` as the group.

---

## MIDI-frame beat/chord learning

The MIDI-frame beat/chord model is kept independent from the regular audio AMT
training stack under [`instrument_agnostic_amt/beat_chord`](instrument_agnostic_amt/beat_chord/README.md).
It supports beat pretraining from MIDI tempo/signature maps, joint beat/chord
training from AMT-generated merged MIDI, and beat/chord inference from MIDI.

```bash
# Beat pretraining from MIDI
python pretrain_midi_frame_beat.py --pretrain_midi_dir beat_chord_dataset/beat_pretrain_dataset/midis

# Joint beat/chord training
python train_midi_frame_beat_chord.py --midi_dir midi_dataset/merged

# Beat/chord inference
python midi_frame_infer.py --checkpoint path/to/checkpoint.pth --midi_path song.mid
```

Corrected prediction MIDIs placed under
`beat_chord_dataset/key_only_dataset/midis/` are also used during chord/key
training by default. Only `key_signature` events are treated as labels; chord
markers, tempo/meter metadata, and beat information are never supervised from
these files. Chord losses are masked and chord self-refinement feedback is
detached for this auxiliary stream. Use `--skip_key_only` to disable it or
`--key_only_loss_scale` to change its contribution. To avoid concentrating this
small dataset into consecutive updates, one key-only batch is used every four
training steps by default. Change the spacing with
`--key_only_step_interval N`; setting it to `1` restores the previous
every-step behavior. The loader still makes at most one pass through the
key-only dataset per epoch. Minor signatures are mapped to the model's existing
relative-major key classes.

### Batch creation of uncorrected key-only candidates

To reproduce the Colab stem workflow for every audio file in a directory and
then add predicted beat, meter, chord, and key metadata, install the optional
stem separator and run:

```bash
pip install stem-splitter
python create_key_only_candidates.py \
  --input-dir beat_chord_dataset/source_audio \
  --output-dir beat_chord_dataset/key_only_candidates
```

The script uses the same per-stem model routing as the Colab notebook, merges
the stem MIDIs, predicts note velocities, and runs `midi_frame_infer` before
moving on to the next song.
The current beat/chord checkpoint is selected by modification time from
`beat_chord_checkpoints/midi_frame`; use `--beat-chord-checkpoint` to pin one.
Final uncorrected files are written under `key_only_candidates/midis/`, separate
from the supervised `key_only_dataset/midis/`. Intermediate stems, per-stem
MIDIs, prediction JSON, Audacity labels, and `batch_summary.json` are retained.
Interrupted runs can be resumed with the same command. Each valid separated
stem, per-stem MIDI, merged MIDI, velocity MIDI, and beat/chord/key result is
reused independently. A beat/chord/key result is considered complete only when
both its MIDI and prediction JSON are readable; otherwise that inference is run
again. Pass `--force` to rebuild everything or `--dry-run` to inspect the plan
without loading models.

Beat/chord inference also writes
`beat_chord_predictions/<song>.beat_mapped.mid` by default. This Type 1 MIDI
contains a conductor track with the predicted continuous tempo/time-signature
map, a separate chord-marker track, and retimed copies of the original
performance tracks. Event times are converted through absolute seconds so the
notes remain aligned with paired audio; the saved file is reloaded and checked
for at most 1 ms of note timing drift. Use `--beat_mapped_midi_path` to choose
the destination or `--disable_beat_mapped_midi` to skip this export. Small
within-bar beat timing differences are regularized into stable bar-level tempo
regions while decoded downbeats remain aligned within 1 ms.
A beat-level tempo curve is emitted only for a clear monotonic ritardando or
accelerando, keeping the map readable without quantizing performance events.

The model, datasets, losses, checkpoints, and CLIs are separate from `train.py`.
See the [beat/chord pipeline guide](instrument_agnostic_amt/beat_chord/README.md)
for the dataset layout and full commands.

---

## Inference

### Basic

```bash
python infer.py --audio input_song.wav
```

> **Note**: If `--checkpoint` is not provided, the model will be automatically downloaded from Hugging Face.

### Stem-separated workflow in Google Colab

The Google Colab notebook [`Colab_Inference.ipynb`](Colab_Inference.ipynb) includes an optional workflow that:

1. separates the uploaded song into stems,
2. transcribes the separated stems individually,
3. optionally relabels the instrument of each note with the instrument refinement model,
4. merges the per-stem MIDI files,
5. predicts the velocity of each MIDI note from the corresponding separated stem audio.

This is slower than single-pass inference on the mixed song, but in many cases it improves transcription accuracy because each stem is acoustically simpler and overlapping instruments are reduced. It is especially useful for busy mixes, band recordings, and arrangements with sustained chords plus melody lines.

The stem workflow restricts instrument classification to classes that are plausible for each stem and excludes the remaining classes before calculating instrument probabilities. Standalone `infer.py` runs can use the same filtering by passing comma-separated class names to `--allowed-instruments`.

Velocity prediction is enabled by default (`PREDICT_VELOCITY = True`). The velocity checkpoint, `best_velocity_model.pth`, is downloaded automatically from Hugging Face when needed, and the final file is written with a `_velocity.mid` suffix. Set `PREDICT_VELOCITY = False` in the notebook to skip this step.

On GPUs with limited VRAM, set `LOW_VRAM_MODE = True` in the "Run stem-separated transcription" cell (or pass `low_vram_mode=True` to `run_stem_separated_transcription`). All models then stay resident in CPU RAM, and each stem's AMT model is moved to the GPU only while that stem is being transcribed, then moved back immediately, so at most one model occupies VRAM at a time. The same applies to `infer_list.py` runs.

In low VRAM mode the stem separation step also runs with fp16 autocast, reducing its peak VRAM so it fits on a 6 GB GPU.

On GPUs where fp16 is slow (e.g. GTX 16-series), pass `no_half=True` to run separation in fp32 with halved chunks instead.
Instrument refinement is disabled by default (`REFINE_INSTRUMENTS = False`). When it is enabled, each separated stem is listened to again and the instrument class of its notes is reassigned before the per-stem MIDI files are merged, so velocity prediction and the merged result both use the corrected instruments. Refined stem MIDI files are written to `refined_stem_midis/` next to the original `stem_midis/`. Set `REFINEMENT_CHECKPOINT` to use a specific checkpoint; leaving it empty resolves `checkpoints/best_instrument_refinement.pth` or a locally trained `instrument_agnostic_amt/instrument_refinement/artifacts/checkpoints/best_model.pth`, and otherwise downloads the checkpoint from Hugging Face.

Drum and vocal stems are always skipped. Drums have no non-drum candidate classes, so there is nothing to reassign. Vocals are excluded for a different reason: telling `melody` from `vocal_harmony` is a question of musical role — lead line versus a part layered under it — rather than of timbre, and the refinement model decides from timbre embeddings. In practice it collapses an entire vocal stem onto one of the two, so the AMT model's own vocal labels are kept instead.

### Standalone instrument refinement

The instrument refinement model reclassifies the instrument of every note in an existing MIDI file using the separated stem audio it was transcribed from. Note timing and pitch are preserved; only the instrument assignment (track program and name) changes.

```bash
python infer_instrument_refinement.py \
  --audio separated_stems/song_other.wav \
  --midi stem_midis/song_other.mid \
  --stem-name other \
  --output-midi song_other_refined.mid
```

`--stem-name` restricts the candidate instruments to the classes that are plausible for that separated stem. `--mode cluster` (the default) groups notes with a similar timbre embedding and labels each group, while `--mode single` assigns one instrument to the whole stem. If `--checkpoint` is omitted, the checkpoint is resolved from the local paths described above or downloaded from Hugging Face.

### Standalone velocity prediction

The velocity model is a separate post-processing model from the AMT note-detection model. Given an existing MIDI file and its separated stem audio, it replaces fixed note velocities with dynamics predicted for each note. The original tracks, pitches, and Note On/Off timing are preserved.

```bash
python infer_velocity.py \
  --midi output.mid \
  --stems-dir separated_stems \
  --output-midi output_velocity.mid
```

If `--checkpoint` is omitted, `best_velocity_model.pth` is downloaded automatically from Hugging Face. The stem directory should contain separated audio files whose names identify the stem, such as `vocals.wav`, `bass.wav`, `drums.wav`, and `other.wav`. See [`instrument_agnostic_amt/velocity/README.md`](instrument_agnostic_amt/velocity/README.md) for velocity-model training and dataset preparation.

### Additional options

```bash
python infer.py \
  --checkpoint checkpoints/checkpoint_epoch_100.pth \
  --audio input_song.wav \
  --output-midi output.mid \
  --amp \
  --window-ms 8000 \
  --stride-ms 4000 \
  --window-batch-size 4 \
  --velocity 100 \
  --max-midi-melodic-instruments 15
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | (auto) | Path to the trained model. Automatically downloaded from HF if not provided |
| `--type` | `default` | Type of the model to download. `default`: for all instruments. `bass`: original bass model. `bass_v2`: updated bass model. `vocal`: fine-tuned for vocal. `guitar`: original guitar model. `guitar_v1_5`: updated guitar model. `vocal_harmony`: fine-tuned for vocal harmony. `drums`: **Experimental** drum-focused model. `other`: fine-tuned for other instruments. |
| `--audio` | (required) | Input audio path |
| `--output-midi` | `<audio>.mid` | Output MIDI path |
| `--amp` | `false` | Enable mixed precision inference |
| `--window-ms` | training value | Inference window size (ms) |
| `--stride-ms` | `window-ms / 2` | Window stride |
| `--window-batch-size` | `1` | Windows to process at once |
| `--merge-gap-ms` | 1 hop | Merge threshold for small note gaps |
| `--merge-onset-ms` | `20.0` | Merge threshold for near-simultaneous onsets |
| `--max-midi-melodic-instruments` | `15` | Max instrument tracks |
| `--allowed-instruments` | all classes | Instrument classification candidates. Accepts comma-separated names or repeated arguments; softmax probabilities are renormalized within the selected classes |
| `--silence-gate-rms-dbfs` | `-72` | RMS threshold to skip silent windows |

---

## Data Augmentation

Training uses multiple augmentation layers to improve generalization:

### Stem level
- **Intra-song stem dropping** — randomly drop stems from the same song to simulate sparse arrangements
- **Cross-song stem mixing** — mix in stems from different songs to create novel combinations
- **Random drum addition** — add drum tracks to drumless mixtures

### Audio level
- **7-band EQ** — simulate different recording setups and mix styles
- **Micro pitch shift** — ±0.2 semitones for subtle tuning variation
- **IR reverb** — real impulse responses for room ambience
- **Noise** — Gaussian noise and background sounds
- **Stereo manipulation** — channel swap, random panning
- **Gain randomization** — ±6 dB per stem

### Spectrogram level
- **SpecAugment** — time and frequency masking on CQT features
- **Harmonic dropout** — randomly drop harmonic channels (fundamental is always kept)

---

## License

[MIT License](LICENSE)
