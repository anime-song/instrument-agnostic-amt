# Instrument-Agnostic AMT V2

V2は、混合音声から指定した1つの楽器クラスのノートだけを出力する、条件付きAMTモデルです。

現在の構成は次の通りです。

```text
waveform
  -> STFTFeatureExtractor
  -> BandSplit
  -> Dual-axis Transformer blocks
       time axis -> band axis
       LWR-ALL layer-wise time resampling, default ratio 4
  -> instrument-conditioned pitch query tokens
  -> Semi-CRF interval decoding + boundary head
  -> single-instrument MIDI
```

V2では次のV1要素を外しています。

- beat / chord 学習
- global token による補助タスク分岐
- framewise instrument classifier
- interval instrument predictor
- CQT / HCQT / StemConv

V1 checkpointとの互換性はありません。

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

top直下の`train.py` / `infer.py` wrapperは使いません。

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

V2では`condition_instrument_id`を使うため、信頼できる楽器ラベルが必要です。`mask_instrument_loss: true`のdataset groupは学習対象から除外されます。

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
| `--n_fft` | `1024` | STFT FFT size |
| `--hop_length` | `512` | STFT hop length |
| `--hidden_size` | `256` | Transformer token dimension |
| `--encoder_num_layers` | `6` | Dual-axis block count |
| `--encoder_num_heads` | `8` | Attention heads |
| `--encoder_head_dim` | `64` | Attentionのheadあたりの内部次元 |
| `--lwr_ratio` | `4` | LWR-ALL time downsample ratio |
| `--condition_negative_prob` | `0.25` | 存在しない楽器をnegative targetとして選ぶ確率 |

checkpointには`architecture_version=2`, `feature_extractor="stft"`, `band_split_type="bs"`, `lwr_mode="all"`, `lwr_ratio`が保存されます。

## 推論

推論ではV2 checkpointと対象楽器を必ず指定します。

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
  --instrument guitar \
  --audio-dir ./audio \
  --output-dir ./midi
```

生成されるMIDIは、指定楽器の1トラックだけです。

## Smoke Test

```bash
python -m compileall instrument_agnostic_amt
python -m instrument_agnostic_amt.cli.train --help
python -m instrument_agnostic_amt.cli.infer --help
```
