# Instrument-Agnostic Automatic Music Transcription

**楽器を問わない自動採譜モデル** — Neural Semi-CRF ベース

[English README](README.md) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anime-song/instrument-agnostic-amt/blob/main/Colab_Inference.ipynb)

<table>
  <tr>
    <td align="center">
      <a href="https://youtu.be/3pCAjQuhzDA">
        <img src="https://img.youtube.com/vi/3pCAjQuhzDA/0.jpg" alt="採譜結果例" width="480">
      </a>
      <br>
      <strong>採譜結果例</strong>
    </td>
    <td align="center">
      <a href="https://www.youtube.com/watch?v=JuVu-AoC5M0">
        <img src="https://img.youtube.com/vi/JuVu-AoC5M0/0.jpg" alt="元動画" width="480">
      </a>
      <br>
      <strong>元動画</strong>
    </td>
  </tr>
</table>

> **動画について**: 上の画像はクリックできるサムネイルです。クリックすると YouTube で動画を見られます。

> **Colab 補足**: [`Colab_Inference.ipynb`](Colab_Inference.ipynb) には、**ステム分離してから採譜し、MIDI をマージした後、分離音声から各ノートの velocity を予測する**オプションのワークフローも入っています。曲全体をそのまま 1 回で採譜するより高精度になることが多く、特に音が重なりやすい密なアレンジで有効です。このワークフローでは velocity 予測がデフォルトで有効です。

---

## 概要

このプロジェクトは、オーディオファイルから MIDI を生成する**楽器非依存の自動採譜 (AMT)** モデルです。
[Basic Pitch](https://github.com/spotify/basic-pitch) と同じように、楽器の種類を区別せず、ピアノでもギターでもボーカルでも音高があればひとつのモデルでまとめて採譜します。

アーキテクチャは [**Transkun**](https://github.com/Yujia-Yan/Transkun)（Yujia Yan 氏）の Neural Semi-CRF がベースです。
もともとピアノ採譜用だったこの仕組みを、楽器を問わず使える汎用モデルに拡張しています。

> **Note**: 楽器を識別してマルチトラック MIDI として出力する機能もありますが、これは**実験的 (Experimental)** な追加機能です。分類精度はまだ十分ではなく、メインの機能はあくまで「楽器を区別しないピッチ検出」です。

> **Note**: ドラム専用モデル（`--type drums`）は**実験的 (Experimental)** です。精度や挙動は今後変わる可能性があります。

> **Warning**: エレキギター（特に歪みサウンド）への汎化はまだ弱く、採譜精度が低くなる傾向があります。また、学習データの少ないエスニック楽器（三味線、シタール等）についても同様です。

### 更新履歴

| 日付 | 内容 |
|---|---|
| 2026-07-30 | ビート・コード・キーの推論モデルをパイプラインに追加しました。デフォルトでは無効になっています。 |
| 2026-07-24 | 分離ステム音声からノートごとの強弱を推定する velocity 予測専用モデルを追加。Colab のステム分離ワークフローでは velocity 予測をデフォルトで有効にし、velocity チェックポイントを Hugging Face から自動取得します。 |
| 2026-07-22 | ギター専用モデル v1.5（`--type guitar_v1_5`）を追加。Colab のステム分離ワークフローでは、ギターステムの既定モデルとして使用します。 |
| 2026-07-16 | 🐛 データ拡張処理の不具合によりノートのタイミングにずれが生じていた問題を修正し、修正後にベースモデル v2（`--type bass_v2`）を再学習しました。再学習した `bass_v2` ではノート検出精度が向上し、1つのノートが複数の短いノートに過剰分割される問題も修正されています。これらの改善は `bass_v2` のみに適用されます。 |
| 2026-07-15 | 🎸 ベースモデル v2（`--type bass_v2`）を追加。楽器分類におけるスラップベースの分類精度が向上しました。 |
| 2026-07-12 | 🎯 ステムごとに推論対象の楽器クラスを指定できるようにしました。候補外の楽器を除外して確率を計算することで、ステム分離ワークフローにおける楽器の誤分類軽減が期待できます。 |
| 2026-06-24 | 🥁 推論用のドラム専用モデルを追加（`--type drums`、実験的 / Experimental） |
| 2026-06-05 | 🎻 その他楽器専用モデルを追加（`--type other`） |
| 2026-05-31 | 🎤 ボーカルハモリモデルを追加（`--type vocal_harmony`）。ハモリを識別できるように楽器一覧に `vocal_harmony` クラスを追加。<br>🧩 重複するノート区間を同時に予測できる Pitch Slot 機能を追加。 |
| 2026-05-20 | 🎸 ギター専用モデルを追加（`--type guitar`） |
| 2026-05-18 | 📦 ピッチシフト・タイムストレッチ用の前処理スクリプトを追加 |
| 2026-05-17 | 🎤 ボーカル専用モデルを追加（`--type vocal`） |
| 2026-05-16 | 🎸 ベース専用モデルを追加（`--type bass`） |
| 2026-05-09 | 🔧 ウィンドウ間のノート結合処理を修正 / ビート・コード学習を追加 |
| 2026-05-09 | 🎼 ステム分離→採譜→マージのワークフローを Colab に追加 |
| 2026-05-06 | 🥁 ドラム判定を強化 / 新しいオーグメンテーションを追加 |
| 2026-05-05 | ✨ EMA・楽器ロスマスク・フォルダ一括推論を追加 |
| 2026-05-03 | 🚀 初回リリース — マルチ楽器 AMT モデル & Colab ノートブック公開 |

### 特徴

- 🎹 **楽器を問わない採譜** — ピアノ、ギター、ベース、ボーカル、ストリングス、管楽器など
- 🧠 **Neural Semi-CRF + Pitch Slot** — ピッチごとに最適なノート区間を Viterbi で一括デコード。Pitch Slot により同じ音程の重複ノートも同時に予測可能
- 🎼 **HCQT 特徴量** — 5つの倍音 × ステレオ 2ch の Harmonic CQT で音高情報をしっかり捉える
- 🎚️ **ノート単位の velocity 予測** — 専用の後処理モデルが分離ステム音声から MIDI ノートの強弱を推定
- 🔧 **豊富なデータ拡張** — ステムの混ぜ合わせ、IR リバーブ、EQ、ノイズ、ドラム追加など
- 🧪 **[実験的] 楽器識別 & マルチトラック出力** — 33+ 楽器クラスの分類ヘッド付き（精度は改善中）

## 既知の制約

本モデルは現在も開発中であり、入力音源や演奏内容によって以下のような問題が発生することがあります。

### 楽器分類

楽器分類およびマルチトラック MIDI 出力は実験的な機能です。

- ベース採譜では、スラップベースやシンセベースを独立した楽器クラスとして識別できず、`electric bass` に分類することがあります。
- ピアノ採譜では、エレクトリックピアノとアコースティックピアノの分類精度が低く、両者を頻繁に取り違えることがあります。
- ステム分離時にギターステムへ混入したシタールやバンジョーなどの楽器を、正しい楽器クラスへ分類できないことがあります。

### 採譜精度

- ボーカル採譜では、速いフレーズやスウィングを含む複雑なフレーズで、ノートの開始位置、長さ、音高を正確に推定できないことがあります。
- 演奏内容によっては、ひとつの音が複数の短いノートに分割される「過分割」が発生することがあります。

### ステム分離ワークフロー

- 分離したステムを個別に採譜すると、ステムごとの MIDI にわずかなタイミングのずれが生じ、マージ後にパート間の同期が合わなくなることがあります。

---

## アーキテクチャ

```
オーディオ波形 [B, 2, T]
        │
        ▼
┌─────────────────────────────┐
│  AudioFeatureExtractor      │
│  (Harmonic CQT × 5倍音)    │   → [B, 10, F=312, T]
│  + SpecAugment (学習時)     │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  StemConv                   │
│  (2D CNN ダウンサンプリング) │   → [B, D, T/8, F/4]
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│  Backbone (Dual-Axis        │
│  Transformer × N層)         │
│  + Pitch Query Embedding    │   → バンド特徴量 + ピッチ別特徴量
│  + Transposed ConvUpsample  │
└─────────────────────────────┘
        │
        ├──────────────────────────┐
        ▼                          ▼
┌───────────────────┐   ┌───────────────────────┐
│ Interval Adapter  │   │ Instrument Adapter    │
│ + IntervalScorer  │   │ + 楽器分類ヘッド(33cls)│
│   (Q, K, Diag)    │   └───────────────────────┘
└───────────────────┘
        │
        ▼
┌───────────────────────────────┐
│ Neural Semi-CRF               │
│ (ピッチ別 Viterbi デコード)   │  → ピッチ毎のノート区間 [begin, end]
│ + Boundary Predictor          │  → Onset/Offset の有無 & サブフレーム補正
└───────────────────────────────┘
        │
        ▼
    MIDI 出力
```

### Dual-Axis Transformer

バックボーンでは 2 種類のトークンを同時に処理します:

- **バンドトークン**: CNN stem が出力した周波数帯域の特徴量
- **ピッチクエリトークン**: MIDI ピッチ（21〜108）に対応する学習可能な埋め込み

各レイヤーで **バンド軸 Transformer**（各タイムステップ内で全トークンにアテンド）と **時間軸 Transformer**（各トークンの時系列にアテンド）を交互に適用し、周波数情報とピッチ情報を効率よく統合します。

### Neural Semi-CRF

88 本のピッチトラックをそれぞれ独立した Semi-CRF としてモデル化します:

- **Pitch Slot** — 同じピッチで音が重なる区間（ユニゾン等）を予測できるよう、複数のスロットを並列処理
- **インターバルスコア** — Query と Key のバイリニアアテンションで算出
- **対角スコア** — 1フレームだけのノート用の加算バイアス
- **Viterbi デコード** — 重複しないノート区間の最適解をグローバルに探索
- **境界予測ヘッド** — Onset/Offset の有無とサブフレームレベルのタイミング補正を予測

---

## プロジェクト構成

```
instrument_agnostic_amt/
├── train.py                    # 学習ループ（AMP、W&B、ウォームアップ対応）
├── infer.py                    # 推論: オーディオ → MIDI
├── dataset.py                  # StemDataset（ステムの混ぜ合わせ等のオーグメンテーション）
├── losses.py                   # ロス計算: Semi-CRF NLL + 境界 + 楽器分類
├── augmentation.py             # AudioAugmentor（EQ、ピッチシフト、リバーブ、ノイズ等）
├── instrument_classes.py       # 楽器クラスのマッピング（GM program ↔ クラスID）
├── instrument_merge.json       # 楽器分類の定義
├── gm_instrument_classes.json  # General MIDI メタデータ
├── dataset_config.yaml         # データセットの重み付け設定
├── requirements.txt            # 依存パッケージ
│
├── models/
│   ├── model.py                # AudioSemiCRFTransformer（モデル本体）
│   ├── transcription_model.py  # 特徴抽出、StemConv、Backbone
│   ├── transformer.py          # RoPE 付き Transformer
│   ├── cqt.py                  # RecursiveCQT（再帰ダウンサンプリングによる高速 CQT）
│   ├── semi_crf.py             # Neural Semi-CRF（前向き-後ろ向き、Viterbi、ロス）
│   ├── interval_boundaries.py  # インターバル境界の特徴量収集
│   └── spec_augment.py         # SpecAugment & MiniBatch Mixture Masking
│
└── preprocess/
    ├── prepare_dataset.py      # オーディオ/MIDI ペアから manifest.csv を生成
    ├── resample_only.py        # まとめてリサンプリング
    └── apply_ir_augmentation.py # IR コンボリューションでリバーブ付きステムを事前生成
```

---

## インストール

### 必要なもの

- Python 3.10+
- CUDA 対応 GPU（VRAM 12GB 以上推奨）

```bash
# クローン
git clone https://github.com/anime-song/instrument-agnostic-amt.git
cd instrument-agnostic-amt

# 仮想環境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 依存パッケージ
pip install -r requirements.txt
```

> `audiomentations` は学習時のオーグメンテーションで使います。推論だけなら入れなくても動きます。

---

## データ準備

### 1. ファイルの配置

ステムオーディオと対応する MIDI ファイルを以下のように配置します:

```
stems/          # オーディオファイル (.wav / .flac)
  ├── song1__piano.wav
  ├── song1__guitar.wav
  ├── song2__vocal.wav
  └── ...

stem_midis/     # 対応する MIDI
  ├── song1__piano.mid
  ├── song1__guitar.mid
  ├── song2__vocal.mid
  └── ...
```

**命名規則**: `<曲名>__<楽器名>.wav`
- `__`（アンダースコア 2 つ）が曲名と楽器名の区切り
- 同じ曲名を持つステムは同一曲のパートとして扱われます

### 2. マニフェスト生成

```bash
python preprocess/prepare_dataset.py \
  --stems_dir ./stems \
  --midis_dir ./stem_midis \
  --npz_dir ./stem_npz \
  --manifest_path ./manifest.csv
```

これで以下が生成されます:
- **`stem_npz/`**: ノート情報の前処理済みファイル（開始/終了時刻、ピッチ、ベロシティ、楽器ID）
- **`manifest.csv`**: データセットのインデックス

### 3. （任意）リサンプリング

オーディオファイルが 22050 Hz でない場合:

```bash
python preprocess/resample_only.py \
  --input_dir ./raw_stems \
  --output_dir ./stems \
  --target_sr 22050
```

## 学習

### 基本

```bash
python train.py \
  --manifest_path manifest.csv \
  --batch_size 8 \
  --lr 5e-4 \
  --epochs 3000 \
  --save_dir checkpoints \
  --wandb
```

### フルオーグメンテーション

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

### 主な引数

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--dataset_config` | `dataset_config.yaml` | 重み付きマルチデータセット設定 |
| `--batch_size` | `8` | バッチサイズ |
| `--lr` | `5e-4` | 学習率 (AdamW) |
| `--warmup_steps` | `1000` | LR ウォームアップのステップ数 |
| `--window_ms` | `8000` | 入力ウィンドウの長さ (ms) |
| `--p_intra_drop` | `0.3` | 曲内のステムをランダムに落とす確率 |
| `--p_cross_mix` | `0.5` | 別の曲からステムを混ぜる確率 |
| `--p_augment` | `1.0` | オーディオ拡張を適用する確率 |
| `--init-from` | `None` | 重み初期化用のチェックポイント |
| `--no_amp` | `false` | 混合精度を無効化 |

### マルチデータセット設定

`dataset_config.yaml` で複数データセットを重み付きで混合できます:

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
    use_for_cross_aug: false  # cross-stem ミキシングには使わない
```

別々のmanifestに同じ曲からレンダリングしたstemが入っている場合は、
オプションの `group` を指定します:

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

`group` とCSVの `song_name` が同じentryは、1つの仮想的な曲として扱われます。
そのため `allow_multi_stem_same_song: true` なら、manifestをまたいでstemを
選択できます。weight、augmentation設定、cross-augmentationへの使用可否は
従来どおり各entry単位です。`group` を省略した場合は `name` が使われ、
既存の分離された挙動を維持します。

---

## MIDIフレームによるビート・コード学習

MIDIフレームのビート・コードモデルは、通常の音声AMT学習から独立させて
[`instrument_agnostic_amt/beat_chord`](instrument_agnostic_amt/beat_chord/README.md)
に配置しています。MIDIのtempo・拍子情報を使うbeat事前学習、AMTで生成したmerged MIDIを
使うbeat/chord joint学習、MIDIからのbeat/chord推論に対応します。

```bash
# MIDIからbeatを事前学習
python pretrain_midi_frame_beat.py --pretrain_midi_dir beat_chord_dataset/beat_pretrain_dataset/midis

# beat/chordをjoint学習
python train_midi_frame_beat_chord.py --midi_dir midi_dataset/merged

# beat/chordを推論
python midi_frame_infer.py --checkpoint path/to/checkpoint.pth --midi_path song.mid
```

beat/chord推論では、既定で
`beat_chord_predictions/<song>.beat_mapped.mid` も出力します。このType 1 MIDIには、
予測した連続tempo・拍子mapのconductor track、独立したコードmarker track、
および元の演奏trackを新しいtempo mapへ再配置したコピーが入ります。全イベントを一度
絶対秒へ変換してから再配置するため、ペアの音声との同期は維持されます。保存後に再読込し、
note時刻の誤差が1ms以内であることも検証します。保存先は
`--beat_mapped_midi_path`、出力の無効化は `--disable_beat_mapped_midi` で指定できます。
小節内の拍位置の小さな揺れは安定した小節単位tempo区間へ正規化し、検出したダウンビートは
1ms以内で維持します。明確で一方向に
持続するritardandoまたはaccelerandoだけは拍単位の滑らかなtempo curveとして残すため、
演奏eventを量子化せず、編集しやすいtempo mapを出力できます。

モデル、dataset、loss、checkpoint、CLIは通常学習の `train.py` とは別管理です。
データ配置と全コマンドは[beat/chordパイプラインガイド](instrument_agnostic_amt/beat_chord/README.md)
を参照してください。

---

## 推論

### 基本

```bash
python infer.py --audio input_song.wav
```

> **Note**: `--checkpoint` を指定しない場合、自動的に Hugging Face から最新のモデルがダウンロードされます。

### Google Colab のステム分離ワークフロー

Google Colab 用ノートブック [`Colab_Inference.ipynb`](Colab_Inference.ipynb) には、以下のオプション機能があります。

1. 入力した曲をステム分離する
2. 各ステムを個別に採譜する
3. ステムごとの MIDI を 1 本へマージする
4. 対応する分離ステム音声から MIDI ノートごとの velocity を予測する

この方法は、ミックス全体をそのまま単発で採譜するより時間はかかりますが、各ステムの音響的な複雑さが下がり、楽器同士の重なりも減るため、採譜精度が上がることが多いです。特に、バンド音源、密な伴奏、和音とメロディが強く重なる曲で有効です。

ステム分離ワークフローでは、ステムごとに妥当な楽器クラスだけを候補にし、候補外のクラスを除いて楽器確率を計算します。単体の `infer.py` でも `--allowed-instruments` にカンマ区切りのクラス名を渡すと同じ制限を利用できます。

velocity 予測はデフォルトで有効です（`PREDICT_VELOCITY = True`）。必要な velocity チェックポイント `best_velocity_model.pth` は Hugging Face から自動取得され、最終結果は `_velocity.mid` という接尾辞付きで保存されます。この処理を省略する場合は、ノートブック内で `PREDICT_VELOCITY = False` に設定してください。

### velocity 予測の単体実行

velocity モデルは、AMT のノート検出モデルとは別の後処理モデルです。既存の MIDI と分離ステム音声を入力し、固定されていた velocity をノートごとの予測値に置き換えます。元のトラック、ピッチ、Note On/Off のタイミングは維持されます。

```bash
python infer_velocity.py \
  --midi output.mid \
  --stems-dir separated_stems \
  --output-midi output_velocity.mid
```

`--checkpoint` を省略すると、`best_velocity_model.pth` を Hugging Face から自動取得します。ステム用ディレクトリには、`vocals.wav`、`bass.wav`、`drums.wav`、`other.wav` のようにステム名を識別できる分離音声を配置してください。velocity モデルの学習とデータ準備については [`instrument_agnostic_amt/velocity/README.md`](instrument_agnostic_amt/velocity/README.md) を参照してください。

### その他のオプション

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

### 主な引数

| 引数 | デフォルト | 説明 |
|---|---|---|
| `--checkpoint` | (自動) | 学習済みモデルのパス。指定しない場合は HF から自動取得 |
| `--type` | `default` | ダウンロードするモデルの種類。`default`: 全楽器用、`bass`: 従来のベース専用モデル、`bass_v2`: 新しいベース専用モデル、`vocal`: ボーカル専用モデル、`guitar`: 従来のギター専用モデル、`guitar_v1_5`: 新しいギター専用モデル、`vocal_harmony`: ボーカルハモリモデル、`drums`: **実験的 (Experimental)** なドラム専用モデル、`other`: その他楽器専用モデル |
| `--audio` | （必須） | 入力オーディオのパス |
| `--output-midi` | `<audio>.mid` | 出力 MIDI のパス |
| `--amp` | `false` | 混合精度推論を有効化 |
| `--window-ms` | 学習時の値 | 推論ウィンドウサイズ (ms) |
| `--stride-ms` | `window-ms / 2` | ウィンドウのストライド |
| `--window-batch-size` | `1` | まとめて処理するウィンドウ数 |
| `--merge-gap-ms` | 1 hop 分 | ノート間ギャップのマージ閾値 |
| `--merge-onset-ms` | `20.0` | 近いオンセットのマージ閾値 |
| `--max-midi-melodic-instruments` | `15` | 楽器トラックの上限 |
| `--allowed-instruments` | 全クラス | 楽器分類の候補。カンマ区切りまたは引数を繰り返して指定。softmax 使用時は指定候補内で確率を再正規化 |
| `--silence-gate-rms-dbfs` | `-72` | 無音スキップの RMS 閾値 |

---

## データ拡張

学習時には複数のオーグメンテーションを組み合わせて汎化性能を高めています:

### ステムレベル
- **イントラステムドロップ** — 同じ曲のステムをランダムに落とし、パートが少ない状況をシミュレート
- **クロスステムミキシング** — 別の曲から異なる楽器のステムを混合
- **ドラム追加** — ドラムがない曲にドラムトラックをランダムに追加

### オーディオレベル
- **7 バンド EQ** — 録音環境やミックスの違いをシミュレート
- **マイクロピッチシフト** — ±0.2 半音のチューニング変動
- **IR リバーブ** — 実際のインパルスレスポンスによる部屋鳴りの付加
- **ノイズ注入** — ガウスノイズや環境音
- **ステレオ操作** — チャンネルスワップ、ランダムパンニング
- **ゲインランダム化** — ステムごと ±6 dB

### スペクトログラムレベル
- **SpecAugment** — CQT 特徴量に対する時間・周波数マスキング
- **ハーモニックドロップアウト** — 倍音チャンネルをランダムにドロップ（基本波は保持）

---

## ライセンス

[MIT License](LICENSE)
