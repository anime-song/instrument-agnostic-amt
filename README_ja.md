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
| 2026-08-20 | ⚡ 依存管理を uv と PyTorch 2.13 へ移行し、MPS 推論とオプトインの AMP/regional compile を追加しました。あわせて推論パイプラインの device 同期・一時コピー・分離ステム音声の再読込を削減しました。出力が意図的に変わる変更が 2 件あります。CUDA 推論で Attention を暗黙に低精度へ落とすのをやめたため既定が完全な FP32 になった点と、V1 の窓バッチがデコード状態を逐次実行と同じ窓順で伝播するようになり、ステム分離ワークフローの既定値 `--window-batch-size 4` の出力が変わる点です。 |
| 2026-08-19 | 🎻 その他楽器モデル v1.5（`--type other_v1_5`）を追加。Colab のステム分離ワークフローでは、`other` ステムの既定モデルとして使用します。独自の実音源評価データセットでは COnP が 0.7318（`other`）から 0.7701（`other_v1_5`）に向上しました。 |
| 2026-08-09 | 🎹 分離ステムを使って AMT のノートに楽器クラスを振り直す Instrument Refinement モデルを追加しました。音色の近いものを同じクラスにまとめるので、1 曲の中で楽器がころころ入れ替わることが減り、手作業で修正する際の一貫性が高くなります。学習に使っていない RWC-I ベンチマークでは全体の top-1 が 71.3% から 74.5% に向上しましたが、**楽器単体で見ると上がったものと下がったものがあります**。楽器ごとの増減と、どの楽器がどの楽器に間違われるかは [RWC-I ベンチマーク](instrument_agnostic_amt/instrument_refinement/RWC_BENCHMARK_ja.md) を参照してください。 |
| 2026-07-30 | ビート・コード・キーの推論モデルをパイプラインに追加しました。デフォルトでは無効になっています。 |
| 2026-07-24 | 分離ステムからノートごとの強弱を推定する velocity 予測専用モデルを追加。Colab のステム分離ワークフローでは velocity 予測をデフォルトで有効にし、velocity チェックポイントを Hugging Face から自動取得します。 |
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
- 🎚️ **ノート単位の velocity 予測** — 専用の後処理モデルが分離ステムから MIDI ノートの強弱を推定
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
├── pyproject.toml              # プロジェクト定義と依存パッケージ（uv）
├── uv.lock                     # 依存バージョンのロックファイル
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

- Python 3.10 〜 3.14
- [uv](https://docs.astral.sh/uv/)（依存パッケージ管理）
- PyTorch 2.13.0 / torchaudio 2.11.0 — コミット済みの `uv.lock` から自動でインストールされます
- 以下のいずれかのデバイス:
  - NVIDIA GPU（VRAM 12GB 以上推奨。Linux / Windows ではロックファイルが CUDA 13.0 の wheel をインストールします）
  - Apple Silicon Mac（macOS 14 以降、MPS バックエンド。PyTorch 2.13 は Intel Mac 向け wheel を提供していません）
  - CPU（動きますが遅いです）

> Windows 向け CUDA 13.0 wheel の解決はロックファイルと設定テストで確認していますが、この PR では Windows CUDA 実機での実行は未検証です。

```bash
# クローン
git clone https://github.com/anime-song/instrument-agnostic-amt.git
cd instrument-agnostic-amt

# 推論用のコア依存パッケージ（uv.lock のバージョンに固定）
uv sync --locked

# 用途に応じたオプション依存
uv sync --locked --extra stem        # ステム分離ワークフロー（infer_stem.py）
uv sync --locked --extra evaluation  # evaluation/ 以下の評価スクリプト
uv sync --locked --all-extras        # 学習用も含めてすべて
```

`uv sync` を実行すると `.venv/` が作られます。この README のコマンド例は、この環境を有効化した状態（`source .venv/bin/activate`）を想定しています。有効化しない場合は、各コマンドの先頭に `uv run` を付けてください（例: `uv run python infer.py --audio input_song.wav`）。

### 動作確認済みの環境

- **Apple Silicon（M4 Pro、macOS / MPS）** — スモークテスト済み: コア AMT の V1/V2 forward とデコード、CQT、velocity、instrument refinement、beat/chord、stem-splitter による分離、MPS の AMP（fp16/bf16）、regional 方式の `--compile` / `--compile-velocity`。公開チェックポイントを使った採譜パイプライン全体（AMT 6 チェックポイント、instrument refinement、MIDI マージ、velocity、beat/chord/key）は、事前に分離済みのステムを入力として MPS で実行済みです。ただし、ステム分離の工程まで含めたエンドツーエンド実行は MPS では**未実施**です。
- **CUDA（Colab Tesla T4）** — Colab 無料枠の実機 Tesla T4 ランタイム（ロック済みの PyTorch 2.13.0 / CUDA 13.0 wheel）で検証済みです。`scripts/colab_t4_regression.py`（後述）により、テストスイート全体とオプトインの CUDA compile 回帰が完走しています。公開チェックポイントを使った採譜パイプライン全体も、事前分離済みステムを入力として T4 で実行済みです。

### テストの実行

```bash
uv sync --locked --all-extras   # テスト収集に stem extra が必要です
uv run pytest
```

CUDA / MPS 専用のテストは、アクセラレータが無い環境では自動的にスキップされます。`torch.compile` の回帰テストはコンパイルに時間がかかるためオプトインです:

```bash
RUN_ACCELERATOR_COMPILE_TEST=1 uv run pytest tests/test_mps_inference.py tests/test_cuda_inference.py
```

### Colab（Tesla T4）での CUDA 回帰テスト

[`scripts/colab_t4_regression.py`](scripts/colab_t4_regression.py) は、Colab の Linux GPU ランタイム内で CUDA 回帰テスト一式を再現します。指定ブランチをクローンし、必須の `--expected-commit`（完全な 40 桁 commit SHA）とクローン直後の HEAD が一致することを検証してから（不一致なら依存関係のインストールへ進まず停止します）、`uv sync --locked --all-extras` でロック済み依存をインストールし、ランタイムの GPU が Tesla T4 であることを確認したうえで、テストスイート全体と CUDA compile 回帰を実行します。CUDA のないローカルPC上で直接実行する用途ではありません。`--repo-url` の既定値は upstream リポジトリなので、実行結果は必ず既知のコミットに紐づきます。対象リポジトリは公開HTTPSで取得できる必要があります。実行コマンドはログに表示されるため、`--repo-url`へ認証情報を埋め込まないでください。

ブラウザ版ColabではT4ランタイムを有効にし、次の内容を1つのコードセルへ貼り付けます。`%%bash`によりセル全体をシェルとして実行し、runner本体もテスト対象と同じ固定コミットから取得します:

```bash
%%bash
set -euo pipefail
# テスト対象のコミットを完全な40桁SHAとして固定する
EXPECTED_COMMIT=$(git ls-remote https://github.com/anime-song/instrument-agnostic-amt.git refs/heads/main | cut -f1)
curl -fL -o colab_t4_regression.py \
  "https://raw.githubusercontent.com/anime-song/instrument-agnostic-amt/${EXPECTED_COMMIT}/scripts/colab_t4_regression.py"
python colab_t4_regression.py --branch main --expected-commit "$EXPECTED_COMMIT"
```

マージ前の公開フォークブランチを検証する場合も、同じ1セル形式を使います。移動するブランチ名ではなく固定コミットからスクリプトを取得し、`--repo-url`でフォークURLを明示します:

```bash
%%bash
set -euo pipefail
FORK_OWNER=YOUR_GITHUB_USER
FORK_REPO=YOUR_FORK_REPO
BRANCH=your-branch
# レビュー対象として告知された 40 桁 SHA を確認して貼り付けます（branch HEAD から自動取得しないでください）。
EXPECTED_COMMIT=FULL_40_CHARACTER_SHA
curl -fL -o colab_t4_regression.py \
  "https://raw.githubusercontent.com/${FORK_OWNER}/${FORK_REPO}/${EXPECTED_COMMIT}/scripts/colab_t4_regression.py"
python colab_t4_regression.py \
  --repo-url "https://github.com/${FORK_OWNER}/${FORK_REPO}.git" \
  --branch "$BRANCH" \
  --expected-commit "$EXPECTED_COMMIT"
```

Colab CLIをインストールしてOAuth認証済みであれば、ローカルOSを問わず同じrunnerを起動できます。`colab run`はスクリプトを新しいT4 VMへ送り、成功時・失敗時のどちらでもVMを解放します:

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

このスクリプトは Colab 無料枠の Tesla T4（ロック済みの PyTorch 2.13.0 / CUDA 13.0 wheel）で完走済みで、テストスイート全体とオプトインの CUDA compile 回帰の両方がパスしています。

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

オーディオファイルが 22050 Hz でない場合は、**その場で上書き**リサンプリングします（元のファイル形式とサンプルフォーマットは維持されます）:

```bash
python preprocess/resample_only.py \
  --input ./stems \
  --resample-rate 22050
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

MIDIフレームのビート・コードモデルは、通常のAMT学習から独立させて
[`instrument_agnostic_amt/beat_chord`](instrument_agnostic_amt/beat_chord/README.md)
に配置しています。MIDIのtempo・拍子情報を使うbeat事前学習、AMTで生成したmerged MIDIを
使うbeat/chord joint学習、MIDIからのbeat/chord推論に対応します。

```bash
# MIDIからbeatを事前学習
python -m instrument_agnostic_amt.beat_chord.cli.pretrain_beat --pretrain_midi_dir beat_chord_dataset/beat_pretrain_dataset/midis

# beat/chordをjoint学習
python train_midi_frame_beat_chord.py --midi_dir midi_dataset/merged

# beat/chordを推論
python midi_frame_infer.py --checkpoint path/to/checkpoint.pth --midi_path song.mid
```

`beat_chord_dataset/key_only_dataset/midis/` に置いた修正済み予測MIDIは、既定で
chord/key学習にも利用されます。ラベルとして扱うのは `key_signature` イベントだけで、
コードマーカー、tempo/拍子メタデータ、ビート情報をこれらのファイルから学習することは
ありません。この補助データに対してはchord lossをマスクし、chordの自己refinement
フィードバックもdetachします。`--skip_key_only` で無効化、`--key_only_loss_scale`
で寄与率を変更できます。この小さなデータセットが連続した更新に集中しないよう、
既定では4学習ステップに1回だけkey-onlyバッチを使います。間隔は
`--key_only_step_interval N` で変更でき、`1` にすると従来の毎ステップ動作に戻ります。
エポックあたりのkey-onlyデータセットの走査は最大1回のままです。マイナーキーは、
モデルが持つ平行長調のキークラスへマップされます。

### 未修正key-only候補の一括生成

ディレクトリ内のすべてのオーディオに対して Colab のステムワークフローを再現し、
予測したビート・拍子・コード・キーのメタデータを付与するには、オプションのステム
分離器をインストールしてから一括実行します:

```bash
uv sync --locked --extra stem
uv run python -c "from instrument_agnostic_amt.beat_chord.key_only_candidates import main; main()" \
  --input-dir beat_chord_dataset/source_audio \
  --output-dir beat_chord_dataset/key_only_candidates
```

このバッチ処理は Colab ノートブックと同じステム別のモデル振り分けを使い、ステムMIDIの
マージ、ノートvelocityの予測、`midi_frame_infer` の実行までを済ませてから次の曲へ
進みます。`infer.py` と同じ `--device` / `--amp` / `--amp-dtype` / `--compile` /
`--compile-mode` オプションに加えて、velocity モデルへ同じ regional コンパイルを
適用する `--compile-velocity` も指定できます。`--amp` が適用されるのは AMT 採譜
ステージだけで、velocity・instrument refinement・beat/chord/key は常に FP32 で
実行されます。
beat/chordチェックポイントは `beat_chord_checkpoints/midi_frame` から更新時刻で
選択されます。固定したい場合は `--beat-chord-checkpoint` を指定してください。
最終的な未修正ファイルは、教師データ用の `key_only_dataset/midis/` とは別に
`key_only_candidates/midis/` へ書き出されます。中間生成物（分離ステム、ステム別MIDI、
予測JSON、Audacityラベル、`batch_summary.json`）は保持されます。中断しても同じ
コマンドで再開でき、有効な分離ステム、ステム別MIDI、マージMIDI、velocity MIDI、
beat/chord/key結果はそれぞれ独立に再利用されます。beat/chord/key結果は、MIDIと
予測JSONの両方が読める場合のみ完了とみなし、揃っていなければその推論だけをやり直します。
すべて作り直す場合は `--force`、モデルを読み込まずに実行計画だけ確認する場合は
`--dry-run` を指定してください。

beat/chord推論では、既定で
`beat_chord_predictions/<song>.beat_mapped.mid` も出力します。このType 1 MIDIには、
予測した連続tempo・拍子mapのconductor track、独立したコードmarker track、
および元の演奏trackを新しいtempo mapへ再配置したコピーが入ります。全イベントを一度
絶対秒へ変換してから再配置するため、ペアのオーディオとの同期は維持されます。保存後に再読込し、
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

### デバイス選択とパフォーマンスオプション

`--device` の既定値は `auto` で、**CUDA → MPS → CPU** の順に利用可能なバックエンドを選びます。デバイスを明示することもでき、利用できないデバイスを指定した場合は暗黙のフォールバックはせずエラーで停止します。

```bash
python infer.py --audio input_song.wav                # auto: CUDA → MPS → CPU
python infer.py --audio input_song.wav --device mps   # Apple Silicon GPU
python infer.py --audio input_song.wav --device cpu
```

**混合精度（`--amp`）** は明示的なオプトインで、暗黙に有効化されることはありません。CUDA と MPS の両方で利用できます。forward パスに autocast を適用するだけで、モデルの重み自体を half 精度へ変換する機能では**ありません**。`--amp-dtype` では `fp16` / `bf16` を選べます。省略した場合、CUDA では GPU が bf16 にネイティブ対応していれば bf16（非対応なら fp16。たとえば Tesla T4 は fp16 になります）、MPS では fp16 が既定です（MPS でも `--amp-dtype bf16` を明示できます）。fp16 の推論時 autocast では、CNN ステム（`StemConv`）全体を常に FP32 island として実行します。無効化するフラグはありません。公開チェックポイントでは活性値が fp16 の範囲を超えうるためです（下の注記を参照）。bf16・fp32・学習時の挙動は変わりません。また、ステム分離パイプラインと key-only バッチ処理では、AMP が適用されるのは AMT 採譜ステージだけです。velocity・instrument refinement・beat/chord/key は常に FP32 で実行されます。

AMP は出力の変化を受け入れて明示的に選ぶトレードオフであり、どのデバイス・楽曲でも速くなるという保証もありません。`--amp` を付けない推論は、Attention も含めて forward パス全体が FP32 のまま実行され、暗黙に低精度へ下がることはありません。`--amp` を付けた場合の出力は FP32 の結果の近似です。実用上は近い出力になりますが同一ではないため、採用する前に代表的な素材で速度と出力を確認してください。

> **`StemConv` の FP32 island がある理由**。fp16 の autocast では、最初の CNN ステージの活性値が公開チェックポイントで fp16 の数値範囲を超えることがあり、そうなると採譜は緩やかな近似ではなく大きく崩れます。`StemConv` を FP32 に保つことでこの故障モードを取り除けるため、island は無条件で有効になっており、無効化するフラグもありません。

```bash
python infer.py --audio input_song.wav --device mps --amp                  # MPS で fp16 autocast
python infer.py --audio input_song.wav --device mps --amp --amp-dtype bf16
```

**`torch.compile`（`--compile`）** も明示的なオプトインで、regional 方式を採用しています。コンパイルされるのは共有バックボーン内の time 軸 / band 軸 Transformer ブロック（公開モデルでは 12 モジュール）だけで、それぞれを TorchInductor の `nn.Module.compile()` でコンパイルします。複素数を扱う CQT/STFT の特徴抽出、CNN ステム（`StemConv`）、各予測ヘッド、Semi-CRF デコード、MIDI 処理は eager のままです。このためコンパイルが短時間で済み、MPS の Inductor が複素演算を扱えない問題にも当たりません。instrument refinement と beat/chord のモデルはコンパイルされません。`--compile-mode` には `default` / `reduce-overhead` / `max-autotune` / `max-autotune-no-cudagraphs` を指定できます。最初のウィンドウ処理には引き続きコンパイル時間が含まれるため、新しいプロセスで短い 1 曲だけを処理する場合は効果が出ないことがあります。最も効くのは、ロード済みのモデルでウィンドウや曲を処理し続ける使い方です。ダウンロードできる 6 種類の AMT チェックポイントと velocity モデルは同じ Transformer 形状を共有しているため、コンパイル済みコードはチェックポイントごとに作り直されず再利用されます。

**Velocity のコンパイル（`--compile-velocity`）** は `--compile` とは独立した別のオプトインで、velocity CLI（`infer_velocity.py`）、key-only バッチ処理、Colab ノートブックで使えます（`--compile-mode` は共用）。コンパイル対象は velocity モデルのバックボーン内にある同じ Transformer 領域だけで、MIDI 解析、窓分割、ノート単位の velocity ヘッドは eager のままです。窓ごとに変わるノート数はコンパイル領域より後段にあるため、フル窓・末尾の端数窓・1 窓に満たない短い曲のすべてが同じコンパイル済みモデルを通ります。eager へのフォールバック経路はなく、実測でも末尾の端数窓による追加のグラフコンパイルは発生しませんでした。

```bash
python infer.py --audio input_song.wav --compile
python infer.py --audio input_song.wav --compile --compile-mode max-autotune
python infer_velocity.py --midi song.mid --stems-dir stems/ --compile-velocity   # velocity モデル
```

#### オプションの使い分け

上記のパフォーマンス系オプションはすべてオプトインで、既定では無効のままです。迷ったら既定のまま実行してください。

- **品質優先** — FP32 eager（追加フラグなし）。上記のような近似の注意が付かない構成です。
- **速度** — `--amp` と `--compile` の効果はデバイス・素材・ワークロードに依存し、どちらも常に速くなるとは限りません。`--compile` はコンパイル時間が最初の実行の内側に含まれるため、新規プロセスの 1 曲だけよりも、繰り返し処理やモデル常駐の使い方に向きます。どちらも代表的な素材で計測してから採用してください。
- **`--compile` と厳密一致** — regional compile は FP32 であってもビット単位で同一の変換ではなく、beat・tempo の出力がどこまでずれるかはデバイス依存です。バイト単位で安定した出力が必要な場合は FP32 eager を選んでください。
- **メモリが厳しいとき** — `--window-batch-size` を下げてください（単体の AMT CLI の既定は 1、ステム分離ワークフローの既定は 4 です）。バッチを小さくするとピークメモリを抑えられますが、バッチ幅を跨いだバイト単位の出力一致は保証されません。

### Google Colab のステム分離ワークフロー

Google Colab 用ノートブック [`Colab_Inference.ipynb`](Colab_Inference.ipynb) には、以下のオプション機能があります。

1. 入力した曲をステム分離する
2. 各ステムを個別に採譜する
3. （任意）instrument refinement モデルで各ノートの楽器を付け直す
4. ステムごとの MIDI を 1 本へマージする
5. 対応する分離ステムから MIDI ノートごとの velocity を予測する

この方法は、ミックス全体をそのまま単発で採譜するより時間はかかりますが、各ステムの音響的な複雑さが下がり、楽器同士の重なりも減るため、採譜精度が上がることが多いです。特に、バンド音源、密な伴奏、和音とメロディが強く重なる曲で有効です。

ノートブックには `DEVICE`、`AMP`、`AMP_DTYPE`、`COMPILE_MODEL`、`COMPILE_VELOCITY`、`COMPILE_MODE` のパラメータもあり、そのまま `run_stem_separated_transcription` へ渡されます。`AMP`、`COMPILE_MODEL`、`COMPILE_VELOCITY` はいずれも既定で無効で、Colab の GPU ランタイムでは `DEVICE = "auto"` で CUDA が選択されます。`COMPILE_MODEL` は AMT バックボーン内の Transformer ブロックを regional 方式でコンパイルし、独立した `COMPILE_VELOCITY` は velocity モデルの同じ領域をコンパイルします。`AMP` の混合精度が適用されるのは AMT ステージだけです。コンパイル対象の詳細と AMP の品質トレードオフは、上の「デバイス選択とパフォーマンスオプション」を参照してください。

ステム分離ワークフローでは、ステムごとに妥当な楽器クラスだけを候補にし、候補外のクラスを除いて楽器確率を計算します。単体の `infer.py` でも `--allowed-instruments` にカンマ区切りのクラス名を渡すと同じ制限を利用できます。

velocity 予測はデフォルトで有効です（`PREDICT_VELOCITY = True`）。必要な velocity チェックポイント `best_velocity_model.pth` は Hugging Face から自動取得され、最終結果は `_velocity.mid` という接尾辞付きで保存されます。この処理を省略する場合は、ノートブック内で `PREDICT_VELOCITY = False` に設定してください。

楽器の再判定（instrument refinement）はデフォルトで無効です（`REFINE_INSTRUMENTS = False`）。有効にすると、ステムごとの MIDI をマージする前に各分離ステムをもう一度聴き直し、ノートの楽器クラスを割り当て直します。そのため velocity 予測とマージ結果の両方が修正後の楽器を使います。再判定後のステム MIDI は、元の `stem_midis/` と並べて `refined_stem_midis/` に書き出されます。特定のチェックポイントを使う場合は `REFINEMENT_CHECKPOINT` を設定してください。空のままにすると `checkpoints/best_instrument_refinement.pth`、またはローカルで学習した `instrument_agnostic_amt/instrument_refinement/artifacts/checkpoints/best_model.pth` を解決し、どちらも無ければ Hugging Face からチェックポイントを取得します。

ドラムとボーカルのステムは常にスキップされます。ドラムはドラム以外の候補クラスを持たないため、割り当て直す対象がありません。ボーカルを除外する理由はそれとは別で、`melody` と `vocal_harmony` の区別は音色ではなく音楽的な役割（主旋律か、その下に重ねるパートか）の問題であるのに対し、refinement モデルは音色の埋め込みから判断するためです。実際にはボーカルステム全体がどちらか一方に潰れてしまうため、AMT モデル自身のボーカルラベルをそのまま使います。

### 楽器再判定（instrument refinement）の単体実行

instrument refinement モデルは、採譜元となった分離ステムを使って、既存の MIDI にあるすべてのノートの楽器を判定し直します。ノートのタイミングとピッチは維持され、楽器の割り当て（トラックのプログラム番号と名前）だけが変わります。

```bash
python infer_instrument_refinement.py \
  --audio separated_stems/song_other.wav \
  --midi stem_midis/song_other.mid \
  --stem-name other \
  --output-midi song_other_refined.mid
```

`--stem-name` を指定すると、その分離ステムで妥当な楽器クラスだけに候補を絞ります。`--mode cluster`（既定）は音色の埋め込みが近いノートをグループにまとめ、グループごとに楽器を割り当てます。`--mode single` はステム全体に 1 つの楽器を割り当てます。`--checkpoint` を省略した場合は、上記のローカルパスから解決するか、Hugging Face から取得します。

### velocity 予測の単体実行

velocity モデルは、AMT のノート検出モデルとは別の後処理モデルです。既存の MIDI と分離ステムを入力し、固定されていた velocity をノートごとの予測値に置き換えます。元のトラック、ピッチ、Note On/Off のタイミングは維持されます。

```bash
python infer_velocity.py \
  --midi output.mid \
  --stems-dir separated_stems \
  --output-midi output_velocity.mid
```

`--checkpoint` を省略すると、`best_velocity_model.pth` を Hugging Face から自動取得します。`--compile-velocity`（`--compile-mode` 共用）は、velocity モデル内の Transformer ブロックを regional 方式でコンパイルするオプトインです。末尾の端数窓を含むすべての窓が同じコンパイル済みモデルを通ります（詳細は上の「デバイス選択とパフォーマンスオプション」を参照）。ステム用ディレクトリには、`vocals.wav`、`bass.wav`、`drums.wav`、`other.wav` のようにステム名を識別できる分離ステムを配置してください。velocity モデルの学習とデータ準備については [`instrument_agnostic_amt/velocity/README.md`](instrument_agnostic_amt/velocity/README.md) を参照してください。

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
| `--type` | `default` | ダウンロードするモデルの種類。`default`: 全楽器用、`bass`: 従来のベース専用モデル、`bass_v2`: 新しいベース専用モデル、`vocal`: ボーカル専用モデル、`guitar`: 従来のギター専用モデル、`guitar_v1_5`: 新しいギター専用モデル、`vocal_harmony`: ボーカルハモリモデル、`drums`: **実験的 (Experimental)** なドラム専用モデル、`other`: 従来のその他楽器専用モデル、`other_v1_5`: 新しいその他楽器専用モデル |
| `--audio` | （必須） | 入力オーディオのパス |
| `--output-midi` | `<audio>.mid` | 出力 MIDI のパス |
| `--device` | `auto` | 推論デバイス。`auto` は CUDA → MPS → CPU の順に選択。`cuda` / `mps` / `cpu` の明示も可 |
| `--amp` | `false` | CUDA/MPS での混合精度（autocast）をオプトインで有効化 |
| `--amp-dtype` | デバイス既定 | `fp16` / `bf16`。既定は CUDA では bf16（ネイティブ対応時）、MPS では fp16 |
| `--compile` | `false` | AMT バックボーン内 Transformer ブロックへの regional `torch.compile` をオプトインで有効化 |
| `--compile-mode` | `default` | `reduce-overhead` / `max-autotune` / `max-autotune-no-cudagraphs` も指定可 |
| `--window-ms` | 学習時の値 | 推論ウィンドウサイズ (ms) |
| `--stride-ms` | `window-ms / 2` | ウィンドウのストライド |
| `--window-batch-size` | `1` | まとめて処理するウィンドウ数（ステム分離ワークフローの既定値は 4）。小さくするとピークメモリを抑えられますが、バッチ幅を跨いだバイト単位の出力一致は保証されません |
| `--merge-gap-ms` | 1 hop 分 | ノート間ギャップのマージ閾値 |
| `--merge-onset-ms` | `50.0` | 近いオンセットのマージ閾値 |
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
