# Instrument-Agnostic AMT

This branch uses a simplified CQT/StemConv AMT model: the input stack and dual-axis Transformer are back to the V1 style, while the output side keeps the current pair-gated Semi-CRF interval decoder so overlapping instrument-pitch intervals can be represented.

The current architecture is:

```text
waveform
  -> HCQT / RecursiveCQT
  -> StemConv (keeps time resolution)
  -> V1 dual-axis Transformer blocks
       band axis -> time axis
       LWR layer-wise time resampling, ratio 4 by default
  -> pitch query features
  -> instrument-pitch pair gate
  -> flat Semi-CRF interval decoding + boundary head
  -> MIDI tracks
```

Removed from this simplified model:

- Source-separation BS-RoFormer stem-splitter backbone
- stem splitter and V1 partial checkpoint initialization helpers
- previous spectral band-splitting input path
- beat / chord auxiliary heads
- framewise instrument classifier and interval instrument predictor

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
    features/cqt.py
    features/spec_augment.py
    blocks/lwr.py
    blocks/axis_transformer.py
    blocks/transformer.py
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
    ...
preprocess/
  prepare_dataset.py
  resample_only.py
  ...
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

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

Training expects reliable instrument labels because the Semi-CRF tracks are selected by instrument-pitch pair targets.

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
| `--hop_length` | `512` | CQT/model frame hop |
| `--n_fft` | `2048` | Window-size guard saved for compatibility |
| `--cqt_n_bins` | `312` | CQT bins before StemConv |
| `--cqt_bins_per_octave` | `36` | CQT resolution |
| `--harmonics` | `1,2,3,4,5` | HCQT harmonic channels |
| `--hidden_size` | `384` | Transformer attention hidden size |
| `--base_ch` | `64` | StemConv base channels; token dim is `4 * base_ch` |
| `--encoder_num_layers` | `6` | V1 dual-axis block count |
| `--encoder_num_heads` | `12` | Attention heads |
| `--lwr_layers` | `last3` | LWR-enabled Transformer layers: `last3`, `all`, `none`, `0,2,4`, `0-2,5`, or mask `101001` |
| `--lwr_ratio` | `8` | LWR time downsample ratio for enabled layers |
| `--lwr_resampling_mode` | `mean` | LWR resampling operator: `mean` or learnable depthwise `conv1d` |

Checkpoints save `architecture_version=2`, `lwr_layers`, `lwr_ratio`, and `lwr_resampling_mode`.

## Inference

Decode all predicted instruments:

```bash
python -m instrument_agnostic_amt.cli.infer \
  --checkpoint checkpoints/checkpoint_epoch_100.pth \
  --audio input_song.wav \
  --output-midi output.mid
```

Decode a single instrument class:

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
  --audio-dir ./audio \
  --output-dir ./midi
```

Inference runs in overlapping windows by default. `--window-ms` uses the checkpoint training window when available, otherwise 8000 ms, and `--stride-ms` defaults to half the window. Use `--window-batch-size` only when VRAM allows it; `--semi-crf-track-batch-size` controls Semi-CRF track decoding memory inside each window.

## Smoke Tests

```bash
python -m compileall instrument_agnostic_amt
python -m instrument_agnostic_amt.cli.train --help
python -m instrument_agnostic_amt.cli.infer --help
```
