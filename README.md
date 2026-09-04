# tsumugi

![tsumugi — Transcribe any instrument to MIDI](resources/tsumugi.svg)

**Transcribe any instrument to MIDI.** One instrument-agnostic model powered by a Neural Semi-CRF.

[日本語版 README](README_ja.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anime-song/tsumugi/blob/main/Colab_Inference.ipynb) | [MIT License](LICENSE)

| [![Transcription example](https://img.youtube.com/vi/3pCAjQuhzDA/0.jpg)](https://youtu.be/3pCAjQuhzDA)<br>**Transcription example** | [![Original video](https://img.youtube.com/vi/JuVu-AoC5M0/0.jpg)](https://www.youtube.com/watch?v=JuVu-AoC5M0)<br>**Original video** |
| --- | --- |

Click either image to open the video on YouTube.

---

## Quick start

The easiest way to get started is [`Colab_Inference.ipynb`](Colab_Inference.ipynb): upload a song, run the cells, and download the MIDI. No local setup or checkpoint search required.

To run locally:

```bash
git clone https://github.com/anime-song/tsumugi.git
cd tsumugi
uv sync --locked
uv run python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav
```

The checkpoint is downloaded from Hugging Face on the first run.

---

## Overview

tsumugi is an **instrument-agnostic Automatic Music Transcription (AMT)** model that converts audio to MIDI.

Like [Basic Pitch](https://github.com/spotify/basic-pitch), it works across instruments. Piano, guitar, bass, vocals, strings, brass—if it has pitch, tsumugi can transcribe it with the same model.

The architecture extends the Neural Semi-CRF approach introduced by [**Transkun**](https://github.com/Yujia-Yan/Transkun) (Yujia Yan et al.), originally designed for piano transcription, to instruments of all kinds.

### Strengths and limitations

tsumugi works best on clean, sustained material where each stem is monophonic. Accuracy falls in areas underrepresented in the training data.

- **Electric guitar**, especially with distortion, does not generalize well.
- **Traditional instruments** such as shamisen and sitar are underrepresented in the training data.
- **Fast vocal runs and swing phrasing** can produce unstable pitches and note boundaries.
- A sustained note may be **split into several shorter notes** (over-segmentation).
- The **stem-separated workflow** can introduce small timing offsets between stems, which may cause drift after the MIDI files are merged.

**Instrument classification and multi-track output** are available, but classification accuracy remains limited. Slap and synth bass often collapse into `electric bass`; acoustic and electric piano are frequently confused; and instruments such as sitar or banjo that bleed into a separated guitar stem may be assigned to the wrong class. The **drum models** (`--type drums_v1_5` and the legacy `--type drums`) are experimental and may change. The core instrument-agnostic pitch transcription is not experimental.

---

## Features

- 🎹 **Any pitched instrument** — transcribe piano, guitar, bass, vocals, strings, winds, and more with one model
- 🎚️ **Per-note velocity** — a dedicated post-processing model estimates dynamics from separated stems
- 🎯 **Beat, chord, and key** — optional MIDI-frame models, disabled by default
- 🧪 **[Experimental] Instrument classification** — a 36-class head for per-instrument MIDI tracks

<details>
<summary><b>Changelog</b></summary>

| Date | Update |
| --- | --- |
| 2026-09-01 | 🥁 Added Drum model v1.5 (`--type drums_v1_5`) and made it the Colab default for drum stems. On the real-audio evaluation set (50 ms tolerance, audio-aligned, canonical drum pitches: 40→38 and 57→49), exact F1 improved from 0.6157 to 0.6890 for Drum Kit (+7.3 points) and from 0.1879 to 0.4044 for All Percussions (+21.7 points). |
| 2026-08-25 | 🎤 Added Vocal Harmony model v1.5 (`--type vocal_harmony_v1_5`) and made it the Colab default for `vocals` stems. COnP improved from 0.6052 to 0.6814 on the held-out MIR-ST500 split. |
| 2026-08-20 | ⚡ Migrated to uv and PyTorch 2.13. Added MPS inference and AMP/regional compile controls; reduced device synchronization, temporary copies, and repeated stem loading. Two changes intentionally affect output: CUDA attention no longer downcasts implicitly (FP32 is now the default), and V1 window batching propagates decode state in window order. |
| 2026-08-19 | 🎻 Added Other-Instrument model v1.5 (`--type other_v1_5`) and made it the Colab default for `other` stems. COnP improved from 0.7318 to 0.7701 on an in-house real-recording set. |
| 2026-08-09 | 🎹 Added the Instrument Refinement model. It reassigns instrument classes from separated stems to reduce label flicker within a piece. Overall top-1 accuracy on RWC-I improved from 71.3% to 74.5%, although results vary by instrument; see the [RWC-I benchmark](instrument_agnostic_amt/instrument_refinement/RWC_BENCHMARK.md). |
| 2026-07-30 | Added beat, chord, and key inference models to the pipeline. Disabled by default. |
| 2026-07-24 | Added a dedicated velocity model and enabled it by default in the Colab stem workflow. |
| 2026-07-22 | 🎸 Added Guitar model v1.5 (`--type guitar_v1_5`) and made it the Colab default for guitar stems. |
| 2026-07-16 | 🐛 Fixed a data-augmentation bug that misaligned note timing, then retrained `bass_v2`. Note detection improved and sustained-note over-segmentation was fixed for `bass_v2`. |
| 2026-07-15 | 🎸 Added the updated bass model (`--type bass_v2`) with improved slap-bass classification. |
| 2026-07-12 | 🎯 Added per-stem instrument class selection, excluding implausible instruments before probability calculation. |
| 2026-06-24 | 🥁 Added the experimental drum-focused model (`--type drums`). |
| 2026-06-05 | 🎻 Added the other-instrument model (`--type other`). |
| 2026-05-31 | 🎤 Added the Vocal Harmony model and `vocal_harmony` class. 🧩 Added Pitch Slots for overlapping note intervals. |
| 2026-05-20 | 🎸 Added the guitar-focused model (`--type guitar`). |
| 2026-05-18 | 📦 Added pitch-shift and time-stretch preprocessing scripts. |
| 2026-05-17 | 🎤 Added the vocal-focused model (`--type vocal`). |
| 2026-05-16 | 🎸 Added the bass-focused model (`--type bass`). |
| 2026-05-09 | 🔧 Fixed cross-window note stitching. Added beat/chord training and the Colab stem workflow. |
| 2026-05-06 | 🥁 Improved drum detection and added new augmentations. |
| 2026-05-05 | ✨ Added EMA, instrument loss masking, and batch directory inference. |
| 2026-05-03 | 🚀 Initial release. |

</details>

---

## Installation

**Requirements**

- Python 3.10–3.14
- [uv](https://docs.astral.sh/uv/)
- PyTorch 2.13.0 / torchaudio 2.11.0, installed automatically from `uv.lock`
- An NVIDIA GPU (12 GB+ VRAM recommended), an Apple Silicon Mac, or CPU

Linux and Windows resolve to the CUDA 13.0 wheels. Apple Silicon macOS resolves to the platform wheels. PyTorch 2.13 does not provide Intel macOS wheels.

```bash
git clone https://github.com/anime-song/tsumugi.git
cd tsumugi

# Core inference dependencies pinned by uv.lock
uv sync --locked

# Optional workflows
uv sync --locked --extra stem        # stem-separated inference
uv sync --locked --extra evaluation  # evaluation scripts
uv sync --locked --extra training    # training
```

`uv sync` creates `.venv/`. Activate it with `source .venv/bin/activate`, or prefix commands with `uv run`.

`.python-version` selects Python 3.12 as the development default without narrowing the supported 3.10–3.14 range. If 3.12 is not installed, uv downloads a managed CPython unless downloads are disabled or the machine is offline.

**Tested on:** Apple Silicon (M4 Pro, macOS / MPS) and CUDA (Colab Tesla T4).

**Tests:**

```bash
uv sync --locked --all-extras
uv run pytest
```

MPS-specific tests are skipped when MPS is unavailable. The compile regression test is opt-in:

```bash
RUN_ACCELERATOR_COMPILE_TEST=1 uv run pytest tests/test_mps_inference.py
```

---

## Inference

```bash
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav
```

If `--checkpoint` is omitted, the model is downloaded from Hugging Face.

### Device selection

`--device` defaults to `auto` and selects the first available backend in this order: **CUDA → MPS → CPU**. If an explicitly requested backend is unavailable, inference fails instead of silently falling back.

```bash
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav                # auto
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav --device mps   # Apple Silicon GPU
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav --device cpu
```

### Optional regional compilation

```bash
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav --compile
python -m instrument_agnostic_amt.amt.cli.infer --audio input_song.wav --compile --compile-mode reduce-overhead
```

The first window pays the compilation cost, so a short one-off transcription may be slower. Compilation is most useful when the loaded model processes many songs. `--compile-mode` also accepts `default`, `reduce-overhead`, and `max-autotune-no-cudagraphs`.

### Stem-separated workflow

[`Colab_Inference.ipynb`](Colab_Inference.ipynb) includes a workflow that separates a song into stems, transcribes them, reclassifies each note's instrument, and predicts velocity.

It is slower than a single pass over the mix, but usually more accurate because each stem is acoustically simpler and contains fewer overlapping instruments. The benefit is greatest for dense mixes, band recordings, and arrangements with sustained chords under a melody.

### Standalone instrument refinement

Reclassifies each note in an existing MIDI file using the stem it was transcribed from. Timing and pitch are preserved; only the track program and name change.

```bash
python -m instrument_agnostic_amt.instrument_refinement.cli.infer \
  --audio separated_stems/song_other.wav \
  --midi stem_midis/song_other.mid \
  --stem-name other \
  --output-midi song_other_refined.mid
```

`--stem-name` limits candidates to instruments plausible for that stem. `--mode cluster` (the default) groups notes by timbre embedding and labels each group; `--mode single` assigns one instrument to the entire stem.

### Standalone velocity prediction

This is a separate post-processing model. Given a MIDI file and its stems, it replaces fixed velocities with per-note dynamics while preserving tracks, pitches, and Note On/Off timing.

```bash
python -m instrument_agnostic_amt.velocity.cli.infer_velocity \
  --midi output.mid \
  --stems-dir separated_stems \
  --output-midi output_velocity.mid
```

Name files in the stem directory after their stems, such as `vocals.wav`, `bass.wav`, `drums.wav`, and `other.wav`. `--compile-velocity` regionally compiles the velocity backbone independently of the core AMT `--compile`. See [`instrument_agnostic_amt/velocity/README.md`](instrument_agnostic_amt/velocity/README.md) for training and data preparation.

### Key arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--audio` | (required) | Input audio path |
| `--output-midi` | `<audio>.mid` | Output MIDI path |
| `--checkpoint` | (auto) | Trained model; downloaded from Hugging Face if omitted |
| `--type` | `default` | Model variant; see the table below |
| `--device` | `auto` | `auto` (CUDA → MPS → CPU), `cuda`, `mps`, or `cpu` |
| `--amp` | `false` | Enable mixed precision on CUDA or MPS |
| `--amp-dtype` | device default | `fp16` or `bf16` |
| `--compile` | `false` | Regional `torch.compile` for the backbone Transformer blocks |
| `--compile-mode` | `default` | `default`, `reduce-overhead`, `max-autotune`, or `max-autotune-no-cudagraphs` |
| `--window-ms` | training value | Inference window size (ms) |
| `--stride-ms` | `window-ms / 2` | Window stride |
| `--window-batch-size` | `1` | Windows per batch. Lower values reduce peak memory; byte-identical output across batch sizes is not guaranteed |
| `--merge-gap-ms` | 1 hop | Threshold for merging small gaps between notes |
| `--merge-onset-ms` | `50.0` | Threshold for merging near-simultaneous onsets |
| `--max-midi-melodic-instruments` | `15` | Maximum number of instrument tracks |
| `--allowed-instruments` | all classes | Classification candidates; softmax is renormalized within the selection |
| `--silence-gate-rms-dbfs` | `-72` | RMS threshold for skipping silent windows |

**Model variants (`--type`)**

| Value | Notes |
| --- | --- |
| `default` | All instruments |
| `bass` / `bass_v2` | `bass_v2` is the updated model |
| `guitar` / `guitar_v1_5` | `guitar_v1_5` is the Colab default |
| `vocal` | Fine-tuned for lead vocals |
| `vocal_harmony` / `vocal_harmony_v1_5` | `v1_5` uses one Pitch Slot, so it does not predict simultaneous notes within a part |
| `other` / `other_v1_5` | `other_v1_5` is the Colab default |
| `drums` / `drums_v1_5` | `drums_v1_5` is the Colab default; `drums` remains the previous checkpoint (**Experimental**) |

---

## Data preparation

Place stems and their matching MIDI files in this structure:

```
stems/          # Audio (.wav / .flac)
  ├── song1__piano.wav
  ├── song1__guitar.wav
  ├── song2__vocal.wav
stem_midis/     # Matching MIDI
  ├── song1__piano.mid
  ├── song1__guitar.mid
  ├── song2__vocal.mid
```

Use the naming convention `<song_name>__<instrument_name>.wav`. A double underscore separates the song and instrument names. Stems with the same song name are treated as one song.

```bash
python preprocess/prepare_dataset.py \
  --stems_dir ./stems \
  --midis_dir ./stem_midis \
  --npz_dir ./stem_npz \
  --manifest_path ./manifest.csv
```

This creates `stem_npz/`, containing preprocessed note arrays (start, end, pitch, velocity, and instrument ID), and `manifest.csv`, the dataset index.

If the audio is not 22050 Hz, resample it in place while preserving the container format and sample subtype:

```bash
python -m preprocess.resample_only --input ./stems --resample-rate 22050
```

---

## Training

```bash
python -m recipes.amt.train \
  --manifest_path manifest.csv \
  --batch_size 8 \
  --lr 5e-4 \
  --epochs 3000 \
  --save_dir checkpoints \
  --wandb
```

With all augmentations enabled:

```bash
python -m recipes.amt.train \
  --dataset_config dataset_config.yaml \
  --batch_size 8 --lr 5e-4 --warmup_steps 1000 --epochs 3000 \
  --ir_folder ./IRs --noise_folder ./noise --drum_folder ./drum_stems \
  --p_augment 1.0 --p_intra_drop 0.3 --p_cross_mix 0.5 --p_drum_mix 0.1 \
  --sa_p 0.5 --sa_freq_max 10 --sa_time_max 20 --sa_num_freq 2 --sa_num_time 2 \
  --wandb --project_name tsumugi
```

### Key arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--dataset_config` | `dataset_config.yaml` | Weighted multi-dataset configuration |
| `--batch_size` | `8` | Batch size |
| `--lr` | `5e-4` | Learning rate (AdamW) |
| `--warmup_steps` | `1000` | Learning-rate warmup steps |
| `--window_ms` | `8000` | Input window length (ms) |
| `--p_intra_drop` | `0.3` | Probability of dropping stems from the same song |
| `--p_cross_mix` | `0.5` | Probability of mixing in stems from other songs |
| `--p_augment` | `1.0` | Probability of applying audio augmentation |
| `--init-from` | `None` | Checkpoint used to initialize weights |
| `--no_amp` | `false` | Disable mixed precision |

### Multi-dataset configuration

`dataset_config.yaml` mixes datasets with different weights:

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
    use_for_cross_aug: false  # Do not use for cross-stem mixing
```

Use `group` when stems rendered from the same songs are stored in separate manifests:

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

Entries with the same `group` and CSV `song_name` form one virtual song, allowing `allow_multi_stem_same_song: true` to select stems across manifests. Weights, augmentation settings, and cross-augmentation eligibility remain per entry. If `group` is omitted, `name` is used and each entry remains isolated.

### Data augmentation

**Stem level** — drop stems from the same song to simulate sparse arrangements, mix stems from different songs into new combinations, and add drums to drumless mixes.

**Audio level** — 7-band EQ, ±0.2-semitone micro pitch shift, IR reverb from real impulse responses, Gaussian noise and background sounds, channel swapping and random panning, and ±6 dB gain per stem.

**Spectrogram level** — SpecAugment time/frequency masking on CQT features, plus harmonic dropout that randomly removes harmonic channels while always preserving the fundamental.

---

## Beat, chord, and key

```bash
# Pretrain beat detection from MIDI
python -m recipes.beat_chord.pretrain_beat \
  --pretrain_midi_dir beat_chord_dataset/beat_pretrain_dataset/midis

# Joint beat/chord training
python -m recipes.beat_chord.train --midi_dir midi_dataset/merged

# Beat/chord inference
python -m instrument_agnostic_amt.beat_chord.cli.infer --checkpoint path/to/checkpoint.pth --midi_path song.mid
```

---

## Project structure

```
instrument_agnostic_amt/
├── amt/                        # Core AMT model, inference, data, and CLI
├── taxonomy/                   # Instrument classes and GM mappings
├── beat_chord/                 # Public beat/chord models and inference
├── instrument_refinement/      # Public refinement model and inference
├── velocity/                   # Public velocity model and inference
└── cli/                        # Composed stem inference pipeline

recipes/
├── amt/                        # AMT train, dataset, targets, and losses
├── beat_chord/                 # Beat/chord datasets and training
├── instrument_refinement/      # Refinement dataset preparation and training
├── velocity/                   # Velocity data preparation and training
└── common/                     # Training-only shared augmentation

preprocess/                     # Standalone dataset preparation tools
```

---

## Acknowledgements

The Neural Semi-CRF formulation and much of the decoding design come from [Transkun](https://github.com/Yujia-Yan/Transkun) by Yujia Yan et al. This project extends that work from piano to arbitrary instruments.

## License

[MIT](LICENSE)
