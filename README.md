# Instrument-Agnostic Automatic Music Transcription

**Transcribe any instrument to MIDI** — Neural Semi-CRF based AMT

[日本語版 README はこちら](README_ja.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anime-song/instrument-agnostic-amt/blob/main/Colab_Inference.ipynb)

[![Campus mode!! new model](https://img.youtube.com/vi/IXSfVcErRro/0.jpg)](https://www.youtube.com/watch?v=IXSfVcErRro)

> **Colab tip**: [`Colab_Inference.ipynb`](Colab_Inference.ipynb) also includes an optional **stem-separated transcription** workflow: separate the song into stems, transcribe each stem, then merge the MIDI files. This often gives better results than transcribing the full mix directly, especially for dense arrangements with overlapping instruments.

---

## What is this?

This project is an **instrument-agnostic Automatic Music Transcription (AMT)** model that converts audio into MIDI.
Like [Basic Pitch](https://github.com/spotify/basic-pitch), it doesn't distinguish between instruments — piano, guitar, bass, vocals, strings, brass — if it has pitch, the model will transcribe it. One model handles everything.

The architecture builds on [**Transkun**](https://github.com/Yujia-Yan/Transkun) (Yujia Yan et al.) and its Neural Semi-CRF approach, originally designed for piano transcription. This project extends it into a general-purpose model that works across all pitched instruments.

> **Note**: There's also an experimental multi-track MIDI output with instrument classification, but classification accuracy is still limited. The core feature is instrument-agnostic pitch detection.

> **Warning**: Generalization to electric guitar (especially with distortion) is still weak, and transcription accuracy tends to be lower. The same applies to ethnic instruments (e.g. shamisen, sitar) that are underrepresented in the training data.

### Changelog

| Date | Update |
|---|---|
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
- 🔧 **Extensive data augmentation** — Stem mixing, IR reverb, EQ, noise injection, drum addition, and more
- 🧪 **[Experimental] Instrument classification & multi-track output** — 33+ instrument class head for per-instrument MIDI tracks (accuracy still improving)

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
2. transcribes each non-drum stem individually,
3. merges the per-stem MIDI files into one final MIDI.

This is slower than single-pass inference on the mixed song, but in many cases it improves transcription accuracy because each stem is acoustically simpler and overlapping instruments are reduced. It is especially useful for busy mixes, band recordings, and arrangements with sustained chords plus melody lines.

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
| `--type` | `default` | Type of the model to download. `default`: for all instruments. `bass`: fine-tuned for bass. `vocal`: fine-tuned for vocal. `guitar`: fine-tuned for guitar. `vocal_harmony`: fine-tuned for vocal harmony. |
| `--audio` | (required) | Input audio path |
| `--output-midi` | `<audio>.mid` | Output MIDI path |
| `--amp` | `false` | Enable mixed precision inference |
| `--window-ms` | training value | Inference window size (ms) |
| `--stride-ms` | `window-ms / 2` | Window stride |
| `--window-batch-size` | `1` | Windows to process at once |
| `--merge-gap-ms` | 1 hop | Merge threshold for small note gaps |
| `--merge-onset-ms` | `20.0` | Merge threshold for near-simultaneous onsets |
| `--max-midi-melodic-instruments` | `15` | Max instrument tracks |
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
