# Instrument Refinement — RWC-I ベンチマーク

[RWC 楽器音データベース](https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-i.html) を使った、
楽器分類の評価結果です。RWC-I は学習には一切使っておらず、評価専用です。

## 指標について

2 つの数字を並べています。意味が違うので混同しないでください。

| 指標 | 定義 |
|---|---|
| **top-1** | 1 ファイルの中で最も多くのノートが付いたクラスが正解と一致した割合。AMT と refine を同じ土俵で比べられるのはこちら |
| **share** | 1 ファイル内で正解クラスに付いたノートの割合を、ファイル間で平均したもの。「どのくらい混ざっているか」を見る細かい指標 |

誤り先の百分率は、そのクラスの全ノートに占める割合です。

## 評価条件

- 対象 554 ファイル（強弱記号が `M` のもの。ノートが 1 つも出なかったファイルは除外）
- 各ファイルの最もエネルギーの大きい 40 秒を切り出して使用
- ステムごとに楽器候補を制限（例: `other` ステムに `electric_bass` は出さない）
- ピチカート奏法（奏法コード `PZ`）の弦は `pizzicato_strings` を正解として採点

## 総合

| | top-1 |
|---|---|
| AMT のみ | 71.3% |
| AMT + Instrument Refinement | **74.5%** |

全体では refine の方が上ですが、**楽器単体で見ると上がったものと下がったものがあります**。
下の表で確認してください。

## クラス別

楽器クラスは taxonomy の粒度です（期待クラスの成績が低い順）。

| クラス | n | AMT top-1 | refine top-1 | 差 | refine share | refine の主な誤り先 |
|---|---:|---:|---:|---:|---:|---|
| `plucked_keyboard` | 12 | 0% | 0% | +0 | 0% | `piano` 63%, `electric_piano` 37% |
| `pizzicato_strings` | 11 | 55% | 9% | **-45** | 8% | `timpani` 31%, `ethnic` 17%, `orchestral_harp` 16% |
| `sax` | 51 | 39% | 31% | **-8** | 31% | `brass` 23%, `orchestral_woodwind` 17%, `flute_pipe` 8% |
| `accordion_family` | 15 | 47% | 47% | +0 | 47% | `sax` 23%, `harmonica` 22% |
| `harmonica` | 8 | 38% | 50% | **+12** | 50% | `sax` 25%, `synth_lead` 19%, `accordion_family` 18% |
| `electric_guitar_clean` | 19 | 68% | 63% | **-5** | 63% | `electric_guitar_muted` 37%, `distorted_guitar` 2% |
| `electric_bass` | 21 | 29% | 71% | **+43** | 73% | `slap_bass` 21%, `synth_bass` 5% |
| `orchestral_woodwind` | 36 | 44% | 72% | **+28** | 68% | `flute_pipe` 25%, `brass` 7%, `ethnic` 2% |
| `strings` | 101 | 80% | 73% | **-7** | 71% | `brass` 5%, `timpani` 5%, `flute_pipe` 5% |
| `organ` | 20 | 65% | 80% | **+15** | 79% | `brass` 15%, `synth_lead` 1%, `synth_pad` 1% |
| `ethnic` | 33 | 79% | 82% | **+3** | 83% | `acoustic_guitar` 8%, `electric_guitar_clean` 3%, `plucked_keyboard` 3% |
| `piano` | 12 | 100% | 83% | **-17** | 83% | `electric_piano` 26% |
| `brass` | 89 | 97% | 92% | **-4** | 91% | `sax` 7%, `orchestral_woodwind` 2%, `ethnic` 1% |
| `flute_pipe` | 47 | 81% | 94% | **+13** | 88% | `sax` 1%, `orchestral_woodwind` 1%, `synth_lead` 1% |
| `electric_piano` | 6 | 67% | 100% | **+33** | 100% | — |
| `chromatic_percussion` | 22 | 100% | 100% | +0 | 100% | — |
| `acoustic_guitar` | 33 | 100% | 100% | +0 | 99% | `electric_guitar_clean` 1% |
| `orchestral_harp` | 18 | 50% | 100% | **+50** | 99% | `synth_pad` 1%, `timpani` 0% |

## 楽器別

RWC-I の楽器 43 種を、期待クラスごとにまとめています。

| 楽器 | 期待クラス | n | AMT top-1 | refine top-1 | 差 | refine の主な誤り先 |
|---|---|---:|---:|---:|---:|---|
| Clavinet | `plucked_keyboard` | 2 | 0% | 0% | +0 | `electric_piano` 88%, `piano` 12% |
| Harpsichord | `plucked_keyboard` | 10 | 0% | 0% | +0 | `piano` 100% |
| Soprano Sax | `sax` | 13 | 23% | 15% | **-8** | `orchestral_woodwind` 34%, `brass` 20%, `flute_pipe` 19% |
| Alto Sax | `sax` | 13 | 46% | 31% | **-15** | `brass` 26%, `ethnic` 10%, `flute_pipe` 9% |
| Tenor Sax | `sax` | 13 | 46% | 38% | **-8** | `orchestral_woodwind` 20%, `brass` 19%, `synth_lead` 9% |
| Baritone Sax | `sax` | 12 | 42% | 42% | +0 | `brass` 29%, `strings` 9%, `orchestral_woodwind` 4% |
| Accordion | `accordion_family` | 15 | 47% | 47% | +0 | `sax` 23%, `harmonica` 22% |
| Harmonica | `harmonica` | 8 | 38% | 50% | **+12** | `sax` 25%, `synth_lead` 19%, `accordion_family` 18% |
| Electric Guitar | `electric_guitar_clean` | 19 | 68% | 63% | **-5** | `electric_guitar_muted` 37%, `distorted_guitar` 2% |
| Electric Bass | `electric_bass` | 21 | 29% | 71% | **+43** | `slap_bass` 21%, `synth_bass` 5% |
| Clarinet | `orchestral_woodwind` | 12 | 83% | 58% | **-25** | `flute_pipe` 76%, `sax` 4%, `percussive_fx` 1% |
| Bassoon | `orchestral_woodwind` | 12 | 25% | 67% | **+42** | `brass` 16%, `ethnic` 5% |
| English Horn | `orchestral_woodwind` | 4 | 25% | 75% | **+50** | `brass` 5% |
| Oboe | `orchestral_woodwind` | 8 | 25% | 100% | **+75** | `flute_pipe` 10%, `brass` 1% |
| Contrabass | `strings` | 31 | 61% | 61% | +0 | `timpani` 32%, `brass` 3%, `synth_pad` 0% |
| Viola | `strings` | 27 | 78% | 63% | **-15** | `brass` 9%, `harmonica` 9%, `flute_pipe` 8% |
| Violin | `strings` | 27 | 78% | 67% | **-11** | `flute_pipe` 7%, `brass` 5%, `ethnic` 3% |
| Cello | `strings` | 27 | 96% | 78% | **-19** | `timpani` 4%, `brass` 3%, `orchestral_harp` 2% |
| Pipe Organ | `organ` | 8 | 25% | 50% | **+25** | `brass` 31% |
| Hammond Organ | `organ` | 12 | 92% | 100% | **+8** | `synth_lead` 2%, `synth_pad` 1% |
| Banjo | `ethnic` | 6 | 0% | 50% | **+50** | `acoustic_guitar` 35%, `electric_guitar_clean` 13%, `electric_guitar_muted` 2% |
| Koto | `ethnic` | 15 | 93% | 80% | **-13** | `plucked_keyboard` 9%, `brass` 5%, `orchestral_woodwind` 3% |
| Shamisen | `ethnic` | 12 | 100% | 100% | +0 | `brass` 1% |
| Pianoforte | `piano` | 12 | 100% | 83% | **-17** | `electric_piano` 26% |
| Horn | `brass` | 18 | 100% | 83% | **-17** | `sax` 21%, `harmonica` 2% |
| Trumpet | `brass` | 21 | 86% | 86% | +0 | `sax` 11%, `ethnic` 5%, `flute_pipe` 2% |
| Cornet | `brass` | 9 | 100% | 89% | **-11** | `orchestral_woodwind` 18%, `chromatic_percussion` 2% |
| Trombone | `brass` | 35 | 100% | 100% | +0 | — |
| Tuba | `brass` | 6 | 100% | 100% | +0 | — |
| Pan Flute | `flute_pipe` | 4 | 100% | 75% | **-25** | `synth_lead` 33%, `orchestral_woodwind` 18% |
| Shakuhachi | `flute_pipe` | 8 | 75% | 88% | **+12** | `sax` 5%, `brass` 1%, `harmonica` 1% |
| Recorder | `flute_pipe` | 12 | 42% | 92% | **+50** | `orchestral_woodwind` 2%, `strings` 0% |
| Flute | `flute_pipe` | 9 | 100% | 100% | +0 | `sax` 3% |
| Piccolo | `flute_pipe` | 14 | 100% | 100% | +0 | `percussive_fx` 1%, `sound_fx` 0% |
| Acoustic Guitar | `acoustic_guitar` | 12 | 100% | 100% | +0 | `electric_guitar_clean` 2% |
| Classic Guitar | `acoustic_guitar` | 12 | 100% | 100% | +0 | — |
| Ukulele | `acoustic_guitar` | 9 | 100% | 100% | +0 | — |
| Glockenspiel | `chromatic_percussion` | 2 | 100% | 100% | +0 | — |
| Marimba | `chromatic_percussion` | 6 | 100% | 100% | +0 | — |
| Vibraphone | `chromatic_percussion` | 12 | 100% | 100% | +0 | — |
| Xylophone | `chromatic_percussion` | 2 | 100% | 100% | +0 | — |
| Electric Piano | `electric_piano` | 6 | 67% | 100% | **+33** | — |
| Harp | `orchestral_harp` | 18 | 50% | 100% | **+50** | `synth_pad` 1%, `timpani` 0% |

## 読み取れること

- **同じグループにまとまる方向に働きます。** 音色の近いものを 1 つのクラスへ寄せるので、1 曲の中で楽器がころころ入れ替わることが減り、手作業で直すときの当たりが付けやすくなります。
- **楽器単体では上下します。** 上の表の差の列を見てください。
- **撥弦鍵盤（ハープシコード / クラビネット）は 0% のままです。** ピアノやエレピに吸われます。
- **ピチカートの弦は苦手です。** 実録音の学習データが無く、撥弦や打楽器へ散ります。
- **サックスは金管と混ざります。** RWC-I のサックスは無伴奏の単音で、学習側の実録音がサックス四重奏に偏っていることが影響している可能性があります。

## 再現方法

RWC-I は産業技術総合研究所から別途入手する必要があります。
データが手元にあれば、各ファイルに対して `instrument_agnostic_amt/instrument_refinement/cli/infer.py` を回し、ファイル単位でノート数の多数決を取ることで同じ数字が得られます。
