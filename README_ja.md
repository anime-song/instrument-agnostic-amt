# Instrument-Agnostic AMT

このブランチは、入力スタックとdual-axis TransformerをV1相当のCQT/StemConvへ戻しつつ、出力側は現在のinstrument-pitch pair gate + Semi-CRF interval decoderを残したシンプルなAMTモデルです。重複するinstrument-pitch区間を表現できます。

現在の構成:

```text
waveform
  -> HCQT / RecursiveCQT
  -> StemConv (時間解像度は維持)
  -> V1 dual-axis Transformer blocks
       band axis -> time axis
       LWR layer-wise time resampling, default ratio 4
  -> pitch query features
  -> instrument-pitch pair gate
  -> flat Semi-CRF interval decoding + boundary head
  -> MIDI tracks
```

削除したもの:

- Source-separation BS-RoFormer stem-splitter backbone
- stem splitter / V1 partial checkpoint initialization helper
- previous spectral band-splitting input path
- beat / chord auxiliary head
- framewise instrument classifier / interval instrument predictor

## ディレクトリ構成

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

## インストール

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

## データ準備

stem音声と対応するMIDIを用意し、manifestを作ります。

```bash
python preprocess/prepare_dataset.py \
  --stems_dir ./stems \
  --midis_dir ./stem_midis \
  --npz_dir ./stem_npz \
  --manifest_path ./manifest.csv
```

Semi-CRFのtrackはinstrument-pitch pair targetから選ぶため、学習には信頼できる楽器ラベルが必要です。

## 学習

```bash
python -m instrument_agnostic_amt.cli.train \
  --dataset_config configs/datasets/dataset_config.yaml \
  --batch_size 8 \
  --lr 5e-4 \
  --epochs 3000 \
  --save_dir checkpoints \
  --wandb
```

主なアーキテクチャ引数:

| Argument | Default | Meaning |
|---|---:|---|
| `--hop_length` | `512` | CQT/model frame hop |
| `--n_fft` | `2048` | 互換用に保存するwindow-size guard |
| `--cqt_n_bins` | `312` | StemConv前のCQT bins |
| `--cqt_bins_per_octave` | `36` | CQT resolution |
| `--harmonics` | `1,2,3,4,5` | HCQT harmonic channels |
| `--hidden_size` | `384` | Transformer attention hidden size |
| `--base_ch` | `64` | StemConv base channels。token dimは`4 * base_ch` |
| `--encoder_num_layers` | `6` | V1 dual-axis block count |
| `--encoder_num_heads` | `12` | Attention heads |
| `--lwr_layers` | `last3` | LWRを入れるTransformer layer: `last3`, `all`, `none`, `0,2,4`, `0-2,5`, mask `101001` |
| `--lwr_ratio` | `8` | 有効化された層のLWR time downsample ratio |
| `--lwr_resampling_mode` | `mean` | LWRのresampling operator: `mean` または学習可能なdepthwise `conv1d` |

checkpointには`architecture_version=2`, `lwr_layers`, `lwr_ratio`, `lwr_resampling_mode`が保存されます。

## 推論

全楽器をdecode:

```bash
python -m instrument_agnostic_amt.cli.infer \
  --checkpoint checkpoints/checkpoint_epoch_100.pth \
  --audio input_song.wav \
  --output-midi output.mid
```

単一楽器クラスだけdecode:

```bash
python -m instrument_agnostic_amt.cli.infer \
  --checkpoint checkpoints/checkpoint_epoch_100.pth \
  --instrument piano \
  --audio input_song.wav \
  --output-midi output.mid
```

フォルダ一括推論:

```bash
python -m instrument_agnostic_amt.cli.infer \
  --checkpoint checkpoints/checkpoint_epoch_100.pth \
  --audio-dir ./audio \
  --output-dir ./midi
```

推論はデフォルトでoverlapありのwindow推論です。`--window-ms`はcheckpointに保存された学習windowを優先し、無い場合は8000 msです。`--stride-ms`は未指定ならwindowの半分になります。`--window-batch-size`はVRAMに余裕がある場合だけ上げてください。`--semi-crf-track-batch-size`は各window内のSemi-CRF decodeメモリを調整します。

## Smoke Test

```bash
python -m compileall instrument_agnostic_amt
python -m instrument_agnostic_amt.cli.train --help
python -m instrument_agnostic_amt.cli.infer --help
```
