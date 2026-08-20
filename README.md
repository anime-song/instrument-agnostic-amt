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
| 2026-08-20 | ⚡ Moved dependency management to uv and PyTorch 2.13, added MPS inference and opt-in AMP/regional compile, and reduced device synchronization, temporary copies, and repeated stem-audio loading across the inference pipeline. Two changes deliberately alter output: CUDA inference no longer downcasts attention implicitly, so it now runs in full FP32 by default, and V1 window batching now propagates decode state in the same window order as sequential decoding, which changes the output of the stem workflow's default `--window-batch-size 4`. |
| 2026-08-19 | 🎻 Added other-instrument model v1.5 (`--type other_v1_5`). The Colab stem-separated workflow now uses it by default for `other` stems. On an in-house real-recording evaluation set, COnP improves from 0.7318 (`other`) to 0.7701 (`other_v1_5`). |
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
├── pyproject.toml              # Project metadata & dependencies (uv)
├── uv.lock                     # Locked dependency versions
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

- Python 3.10 – 3.14
- [uv](https://docs.astral.sh/uv/) for dependency management
- PyTorch 2.13.0 / torchaudio 2.11.0 — installed automatically from the committed `uv.lock`
- One of the following devices:
  - NVIDIA GPU (12GB+ VRAM recommended; on Linux and Windows the lockfile installs CUDA 13.0 wheels)
  - Apple Silicon Mac on macOS 14+ (MPS backend; PyTorch 2.13 has no wheels for Intel Macs)
  - CPU (works, but slow)

> The Windows CUDA 13.0 wheel resolution is covered by the lockfile and configuration tests, but has not been run on a Windows CUDA host in this PR.

```bash
# Clone
git clone https://github.com/anime-song/instrument-agnostic-amt.git
cd instrument-agnostic-amt

# Core dependencies for inference (exact versions from uv.lock)
uv sync --locked

# Optional extras, depending on what you want to run
uv sync --locked --extra stem        # stem-separated workflow (infer_stem.py)
uv sync --locked --extra evaluation  # evaluation scripts under evaluation/
uv sync --locked --all-extras        # everything, including training dependencies
```

`uv sync` creates `.venv/`. The commands in this README assume that environment is active (`source .venv/bin/activate`); alternatively, prefix each command with `uv run`, e.g. `uv run python infer.py --audio input_song.wav`.

The repository's `.python-version` selects Python 3.12 as the default development interpreter without narrowing the supported Python 3.10–3.14 range. If Python 3.12 is not already installed, uv downloads a managed CPython 3.12 automatically unless Python downloads are disabled or the machine is offline.

### Verified environments

- **Apple Silicon (M4 Pro, macOS / MPS)** — smoke-tested: core AMT V1/V2 forward and decoding, CQT, velocity, instrument refinement, beat/chord, stem-splitter separation, MPS AMP (fp16/bf16), and the regional `--compile` / `--compile-velocity` paths. The full transcription pipeline with the released checkpoints — AMT across six checkpoints, instrument refinement, MIDI merge, velocity, and beat/chord/key — has been run on MPS starting from pre-separated stems. An end-to-end run that also includes the stem-separation step has **not** been executed on MPS yet.
- **CUDA (Colab Tesla T4)** — verified on an actual free-tier Colab T4 runtime with the locked PyTorch 2.13.0 / CUDA 13.0 wheels: `scripts/colab_t4_regression.py` (see below) has run the full test suite plus the opt-in CUDA compile regression to completion. The full transcription pipeline with the released checkpoints has also been run on the T4 starting from pre-separated stems.

### Running the tests

```bash
uv sync --locked --all-extras   # test collection imports the stem extra
uv run pytest
```

CUDA- and MPS-specific tests are skipped automatically when the accelerator is unavailable. The `torch.compile` regression tests are opt-in because compilation is slow:

```bash
RUN_ACCELERATOR_COMPILE_TEST=1 uv run pytest tests/test_mps_inference.py tests/test_cuda_inference.py
```

### CUDA regression on Colab (Tesla T4)

[`scripts/colab_t4_regression.py`](scripts/colab_t4_regression.py) reproduces the CUDA regression suite inside a Colab Linux GPU runtime: it clones the requested branch, verifies that the checked-out HEAD matches the required `--expected-commit` (a full 40-character SHA) before installing anything, installs the locked dependencies with `uv sync --locked --all-extras`, asserts that the runtime GPU is a Tesla T4, and runs the full test suite plus the CUDA compile regression. It is not intended to run directly on a non-CUDA local machine. `--repo-url` defaults to the upstream repository, so a run is always tied to a known, pinned commit. The repository must be reachable over public HTTPS; do not put credentials in `--repo-url`, because executed commands are printed to the log.

In the browser version of Colab, enable a T4 runtime and paste the following into one code cell. `%%bash` makes the cell run as a shell script, and the runner itself is downloaded from the same pinned commit that it tests:

```bash
%%bash
set -euo pipefail
# Resolve and pin the commit you intend to test (full 40-character SHA)
EXPECTED_COMMIT=$(git ls-remote https://github.com/anime-song/instrument-agnostic-amt.git refs/heads/main | cut -f1)
curl -fL -o colab_t4_regression.py \
  "https://raw.githubusercontent.com/anime-song/instrument-agnostic-amt/${EXPECTED_COMMIT}/scripts/colab_t4_regression.py"
python colab_t4_regression.py --branch main --expected-commit "$EXPECTED_COMMIT"
```

To verify a public fork branch before it is merged, use the same one-cell pattern. Fetch the script from the pinned commit rather than from the moving branch name, and pass the fork explicitly via `--repo-url`:

```bash
%%bash
set -euo pipefail
FORK_OWNER=YOUR_GITHUB_USER
FORK_REPO=YOUR_FORK_REPO
BRANCH=your-branch
# Paste the 40-character SHA announced for review; do not derive it from the live branch head.
EXPECTED_COMMIT=FULL_40_CHARACTER_SHA
curl -fL -o colab_t4_regression.py \
  "https://raw.githubusercontent.com/${FORK_OWNER}/${FORK_REPO}/${EXPECTED_COMMIT}/scripts/colab_t4_regression.py"
python colab_t4_regression.py \
  --repo-url "https://github.com/${FORK_OWNER}/${FORK_REPO}.git" \
  --branch "$BRANCH" \
  --expected-commit "$EXPECTED_COMMIT"
```

The same runner can be launched from any local OS with the Colab CLI installed and OAuth-authenticated. `colab run` sends the script to a fresh T4 VM and releases the VM on success or failure:

```bash
FORK_OWNER=YOUR_GITHUB_USER
FORK_REPO=YOUR_FORK_REPO
BRANCH=your-branch
EXPECTED_COMMIT=FULL_40_CHARACTER_SHA
curl -fL -o /tmp/colab_t4_regression.py \
  "https://raw.githubusercontent.com/${FORK_OWNER}/${FORK_REPO}/${EXPECTED_COMMIT}/scripts/colab_t4_regression.py"
colab run --gpu T4 --timeout 3600 /tmp/colab_t4_regression.py \
  --repo-url "https://github.com/${FORK_OWNER}/${FORK_REPO}.git" \
  --branch "$BRANCH" \
  --expected-commit "$EXPECTED_COMMIT"
```

This script has been run to completion on a free-tier Colab Tesla T4 with the locked PyTorch 2.13.0 / CUDA 13.0 wheels: the full test suite and the opt-in CUDA compile regression both passed.

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

If your audio files are not at 22050 Hz, resample them **in place** (the original container format and sample subtype are preserved):

```bash
python preprocess/resample_only.py \
  --input ./stems \
  --resample-rate 22050
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
python -m instrument_agnostic_amt.beat_chord.cli.pretrain_beat --pretrain_midi_dir beat_chord_dataset/beat_pretrain_dataset/midis

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
uv sync --locked --extra stem
uv run python -c "from instrument_agnostic_amt.beat_chord.key_only_candidates import main; main()" \
  --input-dir beat_chord_dataset/source_audio \
  --output-dir beat_chord_dataset/key_only_candidates
```

The script uses the same per-stem model routing as the Colab notebook, merges
the stem MIDIs, predicts note velocities, and runs `midi_frame_infer` before
moving on to the next song. The batch runner accepts the same `--device`,
`--amp`, `--amp-dtype`, `--compile`, and `--compile-mode` options as `infer.py`,
plus `--compile-velocity` to apply the same regional compilation to the
velocity model. `--amp` affects only the AMT transcription stage; velocity,
instrument refinement, and beat/chord/key always run in FP32.
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

### Device selection and performance options

`--device` defaults to `auto`, which picks the first available backend in the order **CUDA → MPS → CPU**. A device can also be pinned explicitly; requesting an unavailable device fails with an error instead of silently falling back.

```bash
python infer.py --audio input_song.wav                # auto: CUDA → MPS → CPU
python infer.py --audio input_song.wav --device mps   # Apple Silicon GPU
python infer.py --audio input_song.wav --device cpu
```

**Mixed precision (`--amp`)** is strictly opt-in — it is never enabled implicitly — and works on both CUDA and MPS. It applies autocast to the forward pass only; it does **not** convert the model weights to half precision. `--amp-dtype` selects `fp16` or `bf16`. When omitted, CUDA uses bf16 if the GPU supports it natively (fp16 otherwise — a Tesla T4, for example, gets fp16), and MPS defaults to fp16 (`--amp-dtype bf16` is also available on MPS). Under fp16 inference autocast, the `StemConv` CNN stem always runs as an FP32 island, unconditionally and with no flag to disable it, because its activations can leave the fp16 range on the released checkpoints (see the note below). bf16, fp32, and training behavior are unchanged. In the stem-separated pipeline and the key-only batch runner, AMP applies to the AMT transcription stage only — velocity, instrument refinement, and beat/chord/key always run in FP32.

AMP is a trade-off you opt into explicitly: it can change the output, and it is not guaranteed to be faster on every device or song. Without `--amp`, inference keeps the whole forward pass — attention included — in FP32; nothing is lowered to half precision implicitly. With `--amp`, the output is an approximation of the FP32 result — close in practice, but not identical — so measure the speed and check the output on representative material before adopting it.

> **Why the `StemConv` FP32 island exists.** Under fp16 autocast, the activations of the first CNN stage can leave fp16's numeric range on the released checkpoints, and when that happens the transcription degrades severely rather than gracefully. Keeping `StemConv` in FP32 removes that failure mode, which is why the island is unconditional and has no flag to disable it.

```bash
python infer.py --audio input_song.wav --device mps --amp                  # fp16 autocast on MPS
python infer.py --audio input_song.wav --device mps --amp --amp-dtype bf16
```

**`torch.compile` (`--compile`)** is also opt-in and uses a *regional* strategy: only the time-axis and band-axis Transformer blocks inside the shared backbone (12 modules in the released models) are compiled, each via `nn.Module.compile()` with TorchInductor. The complex-valued CQT/STFT feature stage, the `StemConv` CNN stem, the prediction heads, Semi-CRF decoding, and MIDI processing stay eager, which keeps compilation quick and sidesteps Inductor's complex-op limitations on MPS. The instrument refinement and beat/chord models are not compiled. `--compile-mode` accepts `default`, `reduce-overhead`, `max-autotune`, and `max-autotune-no-cudagraphs`. The first processed window still includes the compilation time, so a fresh process transcribing one short song may not benefit; the payoff is largest when a loaded model keeps processing windows or songs. The six downloadable AMT checkpoints and the velocity model share the same Transformer shape, so compiled region code is reused rather than recompiled per checkpoint.

**Velocity compilation (`--compile-velocity`)** is a separate opt-in, independent of `--compile`, available on the velocity CLI (`infer_velocity.py`), the key-only batch runner, and the Colab notebook; it shares `--compile-mode`. It compiles the same Transformer regions inside the velocity model's backbone — MIDI parsing, windowing, and the note-level velocity head stay eager. Because the variable per-window note count sits after the compiled regions, every window runs through the same compiled model: full windows, the trailing partial window, and songs shorter than one window alike. There is no eager fallback path, and in our measurements the trailing partial window triggered no additional graph compilation.

```bash
python infer.py --audio input_song.wav --compile
python infer.py --audio input_song.wav --compile --compile-mode max-autotune
python infer_velocity.py --midi song.mid --stems-dir stems/ --compile-velocity   # velocity model
```

#### Which options to use

Every performance option above stays opt-in and off by default; when in doubt, run the defaults.

- **Quality first** — FP32 eager (no extra flags). Its output carries none of the approximation caveats above.
- **Speed** — how much `--amp` and `--compile` help depends on the device, the material, and the workload; neither is a guaranteed win. `--compile` pays its compilation cost inside the first run, so it suits repeated or model-resident processing more than a single fresh song. Benchmark both on representative material before adopting them.
- **Exactness with `--compile`** — regional compile is not a bit-exact transformation, not even in FP32, and how far the beat and tempo output drifts is device-dependent. Prefer FP32 eager when byte-stable output matters.
- **When memory is tight** — lower `--window-batch-size` (the standalone AMT CLI defaults to 1; the stem-separated workflow defaults to 4). Smaller batches reduce peak memory, but byte-identical output across different batch widths is not guaranteed.

### Stem-separated workflow in Google Colab

The Google Colab notebook [`Colab_Inference.ipynb`](Colab_Inference.ipynb) includes an optional workflow that:

1. separates the uploaded song into stems,
2. transcribes the separated stems individually,
3. optionally relabels the instrument of each note with the instrument refinement model,
4. merges the per-stem MIDI files,
5. predicts the velocity of each MIDI note from the corresponding separated stem audio.

This is slower than single-pass inference on the mixed song, but in many cases it improves transcription accuracy because each stem is acoustically simpler and overlapping instruments are reduced. It is especially useful for busy mixes, band recordings, and arrangements with sustained chords plus melody lines.

The notebook also exposes `DEVICE`, `AMP`, `AMP_DTYPE`, `COMPILE_MODEL`, `COMPILE_VELOCITY`, and `COMPILE_MODE` parameters that are passed straight to `run_stem_separated_transcription`. `AMP`, `COMPILE_MODEL`, and `COMPILE_VELOCITY` all default to off, and on Colab GPU runtimes `DEVICE = "auto"` selects CUDA. `COMPILE_MODEL` regionally compiles the Transformer blocks of the AMT backbone, the independent `COMPILE_VELOCITY` does the same for the velocity model, and `AMP` applies mixed precision to the AMT stage only — see "Device selection and performance options" above for what is compiled and for the AMP quality trade-off.

The stem workflow restricts instrument classification to classes that are plausible for each stem and excludes the remaining classes before calculating instrument probabilities. Standalone `infer.py` runs can use the same filtering by passing comma-separated class names to `--allowed-instruments`.

Velocity prediction is enabled by default (`PREDICT_VELOCITY = True`). The velocity checkpoint, `best_velocity_model.pth`, is downloaded automatically from Hugging Face when needed, and the final file is written with a `_velocity.mid` suffix. Set `PREDICT_VELOCITY = False` in the notebook to skip this step.

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

If `--checkpoint` is omitted, `best_velocity_model.pth` is downloaded automatically from Hugging Face. Opt-in `--compile-velocity` (sharing `--compile-mode`) regionally compiles the Transformer blocks of the velocity model; every window, including the trailing partial one, runs through the same compiled model (see "Device selection and performance options" above). The stem directory should contain separated audio files whose names identify the stem, such as `vocals.wav`, `bass.wav`, `drums.wav`, and `other.wav`. See [`instrument_agnostic_amt/velocity/README.md`](instrument_agnostic_amt/velocity/README.md) for velocity-model training and dataset preparation.

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
| `--type` | `default` | Type of the model to download. `default`: for all instruments. `bass`: original bass model. `bass_v2`: updated bass model. `vocal`: fine-tuned for vocal. `guitar`: original guitar model. `guitar_v1_5`: updated guitar model. `vocal_harmony`: fine-tuned for vocal harmony. `drums`: **Experimental** drum-focused model. `other`: original other-instrument model. `other_v1_5`: updated other-instrument model. |
| `--audio` | (required) | Input audio path |
| `--output-midi` | `<audio>.mid` | Output MIDI path |
| `--device` | `auto` | Inference device. `auto` picks CUDA → MPS → CPU; `cuda` / `mps` / `cpu` can be set explicitly |
| `--amp` | `false` | Opt-in mixed precision (autocast) on CUDA/MPS |
| `--amp-dtype` | device default | `fp16` / `bf16`. Defaults to bf16 on CUDA (when natively supported) and fp16 on MPS |
| `--compile` | `false` | Opt-in regional `torch.compile` of the AMT backbone Transformer blocks |
| `--compile-mode` | `default` | Also accepts `reduce-overhead` / `max-autotune` / `max-autotune-no-cudagraphs` |
| `--window-ms` | training value | Inference window size (ms) |
| `--stride-ms` | `window-ms / 2` | Window stride |
| `--window-batch-size` | `1` | Windows to process at once (the stem-separated workflow defaults to 4). Lower values reduce peak memory; byte-identical output across batch widths is not guaranteed |
| `--merge-gap-ms` | 1 hop | Merge threshold for small note gaps |
| `--merge-onset-ms` | `50.0` | Merge threshold for near-simultaneous onsets |
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
