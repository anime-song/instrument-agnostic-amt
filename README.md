# Instrument-Agnostic AMT V2

V2 is a conditioned Automatic Music Transcription model. It predicts notes for one requested instrument class from a mixed audio input.

The current architecture is:

```text
waveform
  -> STFTFeatureExtractor
  -> BandSplit
  -> Dual-axis Transformer blocks
       time axis -> band axis
       LWR-ALL layer-wise time resampling, ratio 4 by default
  -> instrument-conditioned pitch query tokens
  -> Semi-CRF interval decoding + boundary head
  -> single-instrument MIDI
```

V2 intentionally removes the V1 auxiliary tasks and heads:

- Beat and chord training
- Global token auxiliary routing
- Framewise instrument classifier
- Interval instrument predictor
- CQT / HCQT / StemConv feature stack

V1 checkpoints are not compatible with the V2 entrypoints.

## Project Layout

```text
instrument_agnostic_amt/
  cli/
    train.py
    infer.py
  data/
    augmentation.py
    dataset.py
    sampling.py
  modeling/
    features/stft.py
    bands/split.py
    blocks/lwr.py
    blocks/axis_transformer.py
    blocks/transformer.py
    conditioning.py
    heads/semi_crf.py
    heads/interval_boundaries.py
    model.py
  taxonomy/
    instrument_classes.py
    instrument_merge.json
    gm_instrument_classes.json
  training/
    losses.py
configs/
  datasets/
    dataset_config.yaml
    dataset_real_config_vocal.yaml
    ...
preprocess/
  prepare_dataset.py
  resample_only.py
  ...
```

Top-level `train.py` and `infer.py` wrappers are no longer used.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data Preparation

Prepare stem audio and matching MIDI files, then generate a manifest:

```bash
python preprocess/prepare_dataset.py \
  --stems_dir ./stems \
  --midis_dir ./stem_midis \
  --npz_dir ./stem_npz \
  --manifest_path ./manifest.csv
```

V2 training expects reliable instrument labels. Dataset groups with `mask_instrument_loss: true` are skipped because the model is trained with an explicit `condition_instrument_id`.

## Training

```bash
python -m instrument_agnostic_amt.cli.train \
  --dataset_config configs/datasets/dataset_config.yaml \
  --batch_size 8 \
  --lr 5e-4 \
  --epochs 3000 \
  --save_dir checkpoints \
  --wandb
```

Useful architecture arguments:

| Argument | Default | Meaning |
|---|---:|---|
| `--n_fft` | `1024` | STFT FFT size |
| `--hop_length` | `512` | STFT hop length |
| `--hidden_size` | `256` | Transformer token dimension |
| `--encoder_num_layers` | `6` | Dual-axis block count |
| `--encoder_num_heads` | `8` | Attention heads |
| `--encoder_head_dim` | `64` | Per-head attention dimension |
| `--lwr_ratio` | `4` | LWR-ALL time downsample ratio |
| `--condition_negative_prob` | `0.25` | Probability of sampling an absent instrument as a negative target |

Checkpoints save `architecture_version=2`, `feature_extractor="stft"`, `band_split_type="bs"`, `lwr_mode="all"`, and `lwr_ratio`.

## Inference

Inference requires both a V2 checkpoint and the target instrument class:

```bash
python -m instrument_agnostic_amt.cli.infer \
  --checkpoint checkpoints/checkpoint_epoch_100.pth \
  --instrument piano \
  --audio input_song.wav \
  --output-midi output.mid
```

Batch inference:

```bash
python -m instrument_agnostic_amt.cli.infer \
  --checkpoint checkpoints/checkpoint_epoch_100.pth \
  --instrument guitar \
  --audio-dir ./audio \
  --output-dir ./midi
```

The MIDI writer creates one track/program for the requested instrument.

## Smoke Tests

```bash
python -m compileall instrument_agnostic_amt
python -m instrument_agnostic_amt.cli.train --help
python -m instrument_agnostic_amt.cli.infer --help
```
