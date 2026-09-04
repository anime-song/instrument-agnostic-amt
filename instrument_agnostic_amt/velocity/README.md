# Velocity pipeline

このディレクトリは、AMT本体のノート検出学習とは独立した後段処理です。
入力はステム音声とAMT出力MIDI、出力はノート単位のvelocityです。

```text
velocity/
├── data/
│   └── midi.py           # MIDI読込、固定velocity/CC7/CC11の除去
├── modeling/             # 公開モデルとcheckpoint読込
├── cli/
│   └── infer_velocity.py
└── stems.py              # stem class定義
```

## 教師の扱い

- `pseudo_velocity_rank`: 実ステム内の相対的な強さ。主教師候補。
- `pseudo_velocity`: rankを32〜112へ写した便宜的な値。正解velocityではない。
- `pseudo_confidence`: onset SNR、短音、同時onset数を反映した重み。
- `relative_active_level_db`: 同一曲の有効ステム中央値を0 dBとした相対レベル。
- `input_velocity`: AMT出力に元から入っていた値。監査用であり教師には使わない。

擬似ノート強度は低コストな波形エネルギー版です。pitch-conditioned CQT教師は、
この形式を基準に後から追加できます。

## 1. 入力ファイルの索引を作る

```powershell
python -m recipes.velocity.prepare_dataset `
  --source-root '<AMT dataset root>' `
  --mode index
```

`artifacts/amt_cbnet/manifest.csv` と `summary.json` を作ります。

## 2. 実ステムの擬似教師を作る

まず少数曲で確認します。

```powershell
python -m recipes.velocity.prepare_dataset `
  --source-root '<AMT dataset root>' `
  --mode pseudo `
  --limit-songs 10 `
  --workers 2 `
  --write-canonical-midi
```

確認後に `--limit-songs` を外すと全件を処理します。既存NPZは再利用され、
作り直す場合だけ `--overwrite` を指定します。

出力NPZにはノート時刻、pitch、元velocity、onset/pre-onset level、SNR、rank、
pseudo velocity、confidence、valid maskが入ります。canonical MIDIはAMTのノート・
タイミング誤りを残したまま、Note Onをvelocity 80にし、CC7/CC11を除去します。

## 3. 対象SoundFontのvelocity curveを作る

0-based GM programごとのsweep MIDIを作ります。

```powershell
python -m recipes.velocity.prepare_calibration
```

初期設定では128 melodic programそれぞれについて5 pitch × 8 velocityの40音と、
GM drumの代表9 pitch × 8 velocityを作ります。`sweep_events.csv` がノート時刻を、
`render_manifest.csv` が相対的なMIDI/WAV名を保持します。

WAVを作り直す場合だけ、SoundFontは実行時引数として渡します。場所は設定や生成物へ
保存しません。

```powershell
python -m recipes.velocity.render_soundfont `
  --soundfont '<path-to-sf2>' `
  --fluidsynth-executable '<path-to-fluidsynth>'
```

WAV生成後は、次の解析を実行します。

```powershell
python -m recipes.velocity.analyze_calibration
```

`analysis/velocity_curves.csv` と `velocity_curves.npz` に、各
`program × pitch × velocity` の実測dBFS、単調回帰後のdBFS、velocity 127基準の
相対dBを保存します。前音のリリースが混ざった境界点は `tail_contaminated=1`、
実測値をfitに使った点は `fit_observed=1` です。`analysis/summary.json` は無音、
クリップ、非単調補正量などの品質情報です。

## 4. 合成学習データを作る

擬似教師のmanifestから、まず音声を必要としない合成計画を作ります。

```powershell
python -m recipes.velocity.prepare_synthetic_dataset `
  --variations 2 `
  --limit-songs 10 `
  --overwrite
```

各variationについて次を生成します。

- velocity 80へ正規化したモデル入力MIDI
- 擬似rankからサンプルした正解velocity入りtarget MIDI
- ノート教師・stem gainを保持したNPZ
- SoundFontレンダリング用の相対パスmanifest
- 複数ステムをまとめる曲単位のexample manifest

擬似教師NPZが更新された場合はfingerprintを確認し、対応するtarget MIDIとlabelを
自動的に作り直します。`--overwrite` は全件を明示的に作り直す場合に使います。

target MIDIをレンダリングします。SoundFontの場所は実行時にだけ渡し、manifestには
保存しません。

```powershell
python -m recipes.velocity.render_synthetic `
  --soundfont '<path-to-sf2>' `
  --fluidsynth-executable '<path-to-fluidsynth>'
```

rendered stemは最初から22.05 kHzのPCM 16-bit WAVとして保存します。過去に44.1 kHzで
作ったWAVは、次のコマンドで個別に検証しながら同じパスへ変換できます。

```powershell
python -m recipes.velocity.resample_rendered_stems `
  --workers 4
```

学習は `render_manifest.csv` のrendered stemとlabel NPZを直接読みます。混合音声の
作成は不要です。`assemble_synthetic_dataset` は音声比較などで明示的にmixtureが
必要な場合だけ使う任意工程です。rendered stemは学習入力なので削除しません。

rendered stemはすべて `render_synth_gain`（既定0.5）の同じ固定値で生成され、stemごとの
ランダムgainはrender時には入りません。Datasetも既定ではrendered stemを0 dBのまま読み、
stem別gainやmaster gainを追加しません。そのため音声レベルとvelocity教師の基準が固定されます。
旧joint学習を再現するときだけ `use_gain_augmentation: true` を設定します。
mixtureとpeak limiterを経由しないため、学習と推論のどちらも分離stem音声が入力に
なります。sample rate不一致時のDataset fallbackは残していますが、通常は変換済み
22.05 kHz WAVをそのまま読みます。

## 5. 学習Datasetを確認する

```powershell
python -m recipes.velocity.inspect_dataset `
  --split all `
  --max-examples 4
```

Datasetは `song_id` の安定hashでtrain/validation/testを分けるため、同じ曲の
variationが別splitへ漏れません。長い曲は固定長windowとして読み、各windowから
次を返します。

- stemごとのstereo音声と有効フレーム数
- onset順のMIDI note query（時刻、pitch、program、drum、stem）
- ノート単位の正解velocityと生成経路
- stem単位のclass、active mask（gain教師は旧joint checkpointとの互換用）
- 固定0 dBのstem gainとmaster gain

`collate_velocity_batch` はnote数とstem数をbatch内最大値へpadし、`note_mask`、
`stem_mask`、`stem_gain_mask` を作ります。

## 6. モデルをdry-runする

```powershell
python -m recipes.velocity.train `
  --root instrument_agnostic_amt/velocity/artifacts/synthetic `
  --init-amt '<AMT checkpoint>' `
  --freeze-backbone `
  --dry-run
```

`--init-amt` はAMT checkpointからHCQT/pitch-query backboneだけを読みます。
velocity modelのcheckpointとAMT modelのcheckpointは別管理です。まずheadだけを
確認するときは `--freeze-backbone` を使い、その後は外して全体をfine-tuneできます。

モデルは各ノートが属する分離stemからpitch/onset位置の局所音響特徴を読み、
1～127のvelocityを分類します。分類分布の期待値にも補助lossを掛けます。
HCQT backboneで消える全体レベルを補うため、velocity headには局所的な絶対dBFS特徴も
渡します。既定model configではstem gain headを作らず、lossもvelocityだけです。
推論MIDIは学習renderと同じくCC7/CC11を127固定にします。旧checkpointとの読み込み
互換性は維持していますが、予測stem gainのCC7反映は明示指定時だけです。

通常学習では `--dry-run` を外します。checkpointは既定では
`velocity/artifacts/checkpoints_velocity_only/` に保存されます。旧joint checkpointとは
別ディレクトリなので、最初は `--resume` ではなくAMT checkpointから開始してください。

## 実装の境界

公開モデル・MIDI前処理・推論CLIは `velocity/`、データ準備・合成データ生成・
損失・Dataset・学習補助・学習CLIは `recipes/velocity/` にあります。
MIDIへの書き戻しは次段階として `velocity/inference/` に追加します。AMT本体の
`modeling/` と `training/` には混在させません。
