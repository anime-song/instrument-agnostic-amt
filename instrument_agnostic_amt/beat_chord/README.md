# MIDI-frame beat/chord pipeline

このディレクトリは、通常の音声AMT学習とは独立したビート・コード用パイプラインです。
AMTで生成したmerged MIDIをフレーム化し、ビート、ダウンビート、拍子、コード、ベース音、
キーを推定する別モデルを学習します。通常AMTの `modeling/` と `training/` には
ビート・コード固有コードを混在させていません。

## 構成

```text
instrument_agnostic_amt/beat_chord/
|-- cli/                 # 推論の入口
|-- decoding/            # beat/chord decoder
|-- heads/               # beat/chord model head
|-- models/              # MIDI-frame stem、backbone、joint model
|-- config.py            # モデル設定
`-- midi_roll.py         # merged MIDIからonset/sustain rollを生成

recipes/beat_chord/
|-- datasets/            # beat、chord、MIDI事前学習dataset
|-- beat.py              # beat loss
|-- chord.py             # chord loss
|-- train.py             # joint学習
|-- pretrain_beat.py     # MIDI事前学習
`-- training_utils.py    # checkpoint、EMA、AMP、W&B
```

共有する既存実装は楽器タクソノミーとTransformer blockだけです。チェックポイント、
optimizer、scheduler、EMAは通常AMTとは別管理です。

## データ

既定では次の配置を使います。

```text
beat_chord_dataset/
|-- beat_pretrain_dataset/
|   `-- midis/                 # tempo・拍子情報を持つ事前学習MIDI
|-- beat_dataset/
|   |-- audio/                 # 曲長と対応曲の確認に使用
|   `-- label/                 # *.beat.beats.json
`-- chord_dataset/
    |-- audio/                 # 曲長と対応曲の確認に使用
    |-- chord_label/           # JSONL
    |-- key_label/             # start:end:key
    |-- quality.json
    `-- quality_freq_count.json

midi_dataset/
`-- merged/                    # AMT出力の <song>.mid
```

beat/chordの教師ラベルとmerged MIDIは同じ曲名にしてください。joint学習のモデル入力は
音声ではなく `midi_dataset/merged` のMIDIロールです。

## 1. MIDIからbeatを事前学習

```bash
python -m recipes.beat_chord.pretrain_beat \
  --pretrain_midi_dir beat_chord_dataset/beat_pretrain_dataset/midis \
  --epochs 10 \
  --batch_size 4
```

tempo mapとtime-signature mapからbeat、downbeat、meter教師を作ります。壊れたMIDIの
スキップとmetadata cacheにも対応しています。既定の保存先は
`beat_chord_checkpoints/midi_frame_beat_pretrain` です。

## 2. beat/chordをjoint学習

事前学習checkpointを初期値に使う場合:

```bash
python -m recipes.beat_chord.train \
  --beat_dataset_path beat_chord_dataset/beat_dataset \
  --chord_dataset_path beat_chord_dataset/chord_dataset \
  --midi_dir midi_dataset/merged \
  --init-from beat_chord_checkpoints/midi_frame_beat_pretrain/checkpoint_epoch_10.pth \
  --epochs 20 \
  --batch_size 4
```

`--skip_beat` または `--skip_chord` で片方だけ学習できます。再開は `--resume-from`、
AMP無効化は `--no_amp`、短い動作確認は `--max_steps_per_epoch 1` を使います。
既定の保存先は `beat_chord_checkpoints/midi_frame` です。

### MIDIのtempo rubato拡張

学習時の一定倍率time stretchに加えて、演奏記号のtempo rubatoを模した非線形な
時間拡張を利用できます。約4秒ごとに「少し溜めてから取り戻す」またはその逆の
滑らかなカーブをランダム生成し、MIDI rollとbeat/downbeat、拍子、コード、キーの
教師ラベルを同じカーブで変形します。各区間と学習窓の両端は元の時刻へ戻るため、
窓全体の長さは変わりません。

```bash
python -m recipes.beat_chord.train \
  --midi_rubato_prob 0.5 \
  --midi_rubato_strength 0.12 \
  --midi_rubato_period_sec 4.0
```

`--midi_rubato_prob 0` で無効化できます。`strength=0.12` は局所的な時間軸の
傾きが最大でおよそ±12%変化する設定です。一定time stretchとは独立しており、
両方が有効な場合は一定伸縮の後にrubatoカーブを合成します。beat事前学習でも
同じ3オプションを使用できます。

## 3. MIDIからbeat/chordを推論

```bash
python -m instrument_agnostic_amt.beat_chord.cli.infer \
  --checkpoint beat_chord_checkpoints/midi_frame/checkpoint_epoch_20.pth \
  --midi_path midi_dataset/merged/song.mid \
  --quality_json beat_chord_dataset/chord_dataset/quality.json
```

既定では `beat_chord_predictions/<song>.prediction.json` とAudacity用ラベルを出力します。
`chord-romanizer` がインストールされている場合は、予測キーごとに
`Romanizer.strict(...).display_progression()` を適用します。JSON の `chord` には
臨時記号を修正したコード記号、`combined_label` にはローマ数字と機能を含む表示用ラベルを
保存します。コンソール、Audacityラベル、tempo-mapped MIDIのコードマーカーでは
`combined_label` を表示します。

### 公開用チェックポイント

学習チェックポイントにはoptimizerやローカルのデータパスが含まれるため、そのまま配布せず、
公開用のallowlist形式へ変換します。

```bash
python scripts/distill_model.py \
  --input beat_chord_checkpoints/midi_frame/checkpoint_epoch_20.pth \
  --output beat_chord_checkpoints/public/midi_frame_beat_chord.pth \
  --quality-json beat_chord_dataset/chord_dataset/quality.json
```

公開用チェックポイントには推論重み、モデル構成、meter class、推論窓幅、コードquality語彙だけが
保存されます。`quality.json`の内容は `chord_quality_map` として内包されるため、このチェックポイントを
使う推論では `--quality_json` は不要です。`quality_freq_count.json` は学習時のloss重み専用なので、
公開推論には不要です。

## 4. MIDIディレクトリから beat_dataset へ変換（本学習用ラベル生成）

手元の MIDI ディレクトリ（テンポ・拍子情報を含む）から、本学習用の `beat_dataset` (JSONラベル・WAV音声) を自動作成します。

```bash
python -m recipes.beat_chord.convert_beat_dataset \
  --midi_dir beat_dataset_midi \
  --output_dir beat_chord_dataset/beat_dataset
```

各入口はmoduleとしても実行できます。

```bash
python -m recipes.beat_chord.convert_beat_dataset --help
python -m recipes.beat_chord.pretrain_beat --help
python -m recipes.beat_chord.train --help
python -m instrument_agnostic_amt.beat_chord.cli.infer --help
```

### Major beat-group boundaries

An independent one-channel head learns internal major boundaries without
changing the beat or downbeat targets. Only complete bars are supervised, and
meters with no required grouping (numerators 1 through 4) are excluded.

- `6/8`, `9/8`, and `12/8` (plus the equivalent `/16` and `/32` meters) use
  fixed groups of three denominator-note units.
- Ambiguous meters such as `5/4`, `6/4`, `7/4`, `5/8`, `7/8`, and `8/8` use
  latent grouping candidates.
- Candidate patterns are generated for additive `/8`, `/16`, and `/32` meters
  up to 21 units, and for long `/2` and `/4` meters.

Latent candidates are trained by valid-set marginal likelihood. Detached MIDI
onset salience supplies a soft target for choosing between candidates, such as
`(4, 3)` versus `(3, 4)` in `7/4`. Training controls are
`--major_grouping_loss_weight`, `--major_grouping_accent_loss_weight`, and
`--major_grouping_accent_temperature`.

Grid inference adds the learned evidence to the meter/beat score and writes the
selected pattern to `meter_segments[].major_grouping`. Use
`--group_boundary_score_weight` and `--grid_false_group_boundary_weight` to
tune it. Legacy checkpoints still load strictly, but grouping is automatically
disabled when the trained head is absent.

When starting training from a checkpoint created before this head existed, use
`--init-from old_checkpoint.pth` instead of `--resume-from`; the old optimizer
state does not contain the newly added parameters.

Meter-aware crop sampling is enabled for both beat pretraining and joint beat
training. With probability `0.75`, it selects a grouping meter using
inverse-frequency weights, then chooses a source crop that contains a complete
bar with both downbeats, the expected number of beat annotations, and two
frames of boundary context. If no eligible grouping bar exists, sampling falls
back to the previous labeled random crop. Configure it with:

```bash
--meter_aware_crop_probability 0.75 \
--meter_aware_crop_rarity_power 0.5 \
--meter_aware_crop_margin_frames 2
```

Set `--meter_aware_crop_probability 0` to disable it. W&B receives
`train/beat/meter_aware_crop_rate` together with
`train/beat/major_grouping_bar_count`, so sampling and effective supervision
can be checked independently.

## スモークテスト

実データを使う最小joint学習:

```bash
python -m recipes.beat_chord.train \
  --epochs 1 \
  --max_steps_per_epoch 1 \
  --batch_size 1 \
  --num_workers 0 \
  --save_dir tmp/midi_frame_smoke \
  --no_amp \
  --chord_modulation_prob 0.0
```

コード単体の回帰テストは `python -m pytest tests/test_beat_chord.py -q` で実行できます。
