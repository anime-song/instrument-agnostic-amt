# Instrument Refinement — RWC-I Benchmark

Instrument-classification results on the
[RWC Musical Instrument Sound Database](https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-i.html).
RWC-I is never used for training — it is held out purely for evaluation.

## Metrics

Two different numbers are reported. They are not interchangeable.

| Metric | Definition |
|---|---|
| **top-1** | The class holding the most notes in a file matches the ground truth. This is the like-for-like comparison between AMT and refinement. |
| **share** | Fraction of notes in a file assigned to the correct class, averaged over files. A finer measure of how much the output is contaminated. |

Confusion percentages are shares of all notes belonging to that class.

## Setup

- 554 files (dynamics marker `M`; files where the AMT produced no notes are excluded)
- The loudest 40-second excerpt of each file is used
- Instrument candidates are restricted per stem (e.g. `electric_bass` is never proposed for an `other` stem)
- Pizzicato strings (playing-style code `PZ`) are scored against `pizzicato_strings`

## Overall

| | top-1 |
|---|---|
| AMT only | 71.3% |
| AMT + Instrument Refinement | **74.5%** |

Refinement wins overall, but **per instrument there are both gains and losses**.
See the tables below.

## By instrument class

Classes follow the taxonomy granularity, worst first.

| Class | n | AMT top-1 | refine top-1 | Δ | refine share | Main refine confusions |
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

## By instrument

All 43 RWC-I instruments, grouped by the class they are expected to map to.

| Instrument | Expected class | n | AMT top-1 | refine top-1 | Δ | Main refine confusions |
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

## What the numbers say

- **It pulls sounds into consistent groups.** Timbrally close instruments collapse onto one class, so the instrument stops flickering within a piece and manual clean-up gets easier.
- **Individual instruments go both ways.** See the Δ columns above.
- **Plucked keyboards (harpsichord, clavinet) stay at 0%** — they are absorbed by piano / electric piano.
- **Pizzicato strings are weak.** There is no real-recording training data for the class, so the notes scatter into plucked and percussive classes.
- **Saxophone blends into brass.** RWC-I saxophones are unaccompanied single lines, while the real-recording training data is skewed toward saxophone quartets.

## Reproducing

RWC-I has to be obtained separately from AIST. Once you have it, running `instrument_agnostic_amt/instrument_refinement/cli/infer.py` over each file and taking the per-file majority vote by note count reproduces these numbers.
