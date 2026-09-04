# Instrument refinement

This module refines instrument classes for AMT note queries using separated-stem
audio. Drums are excluded from training and refinement candidates.

The public code contains no dataset names or private paths. Dataset discovery is
controlled by a local JSON file that is ignored by Git:

```text
instrument_refinement_datasets.local.json
```

## Dataset configuration

Two generic adapters are available:

- `class_manifests`: a collection of per-class source manifests
- `flat`: a flat audio collection with matching MIDI and optional NPZ labels

Example with placeholder paths:

```json
{
  "output": "instrument_agnostic_amt/instrument_refinement/artifacts/datasets/manifest.csv",
  "seed": 42,
  "train_fraction": 0.8,
  "validation_fraction": 0.1,
  "datasets": [
    {
      "name": "private_collection_a",
      "type": "class_manifests",
      "root": "/private/path/class_sources",
      "manifest_glob": "manifests/*.csv",
      "class_name_regex": "(?P<class>.+)_manifest",
      "midi_dir_template": "{class_name}_midi",
      "song_key_mode": "prefix_before_last_double_underscore"
    },
    {
      "name": "private_collection_b",
      "type": "flat",
      "root": "/private/path",
      "audio_glob": "audio/*.wav",
      "midi_dir": "midi",
      "npz_dir": "npz",
      "class_source": "npz",
      "allowed_stem_groups": ["other"]
    }
  ]
}
```

The local config may additionally declare:

- `class_remap`: remap a collection-specific label into the public taxonomy
- `allowed_classes`: reject sources containing any other class
- `allowed_stem_groups`: reject cross-stem or unwanted deployment groups
- `song_key_rules`: regex replacements used only to recover original song IDs
- `augmentation_pattern`: identify pitch/time variants for leakage-safe splits
- `max_augmentation_variants_per_group`: retain every original plus at most this
  many deterministically selected augmented variants per normalized source group
- `augmentation_selection_seed`: seed for the deterministic variant selection
- `require_midi_notes`: reject unreadable or empty MIDI before training
- `atomic_mode: song_variant`: keep all matching sources together
- `normalize_nfkc` and `alnum_only`: normalize atomic-pair keys

Private collection names appear only in the ignored local config and generated
artifacts. They do not appear in Python modules, tests, or public documentation.

## Prepare the manifest

```bash
python -m recipes.instrument_refinement.prepare_manifest
```

The default local config writes:

```text
instrument_agnostic_amt/instrument_refinement/artifacts/datasets/manifest.csv
instrument_agnostic_amt/instrument_refinement/artifacts/datasets/manifest.summary.json
```

Splits are assigned by the normalized original-song key. Augmented variants always
stay in the same split. The builder also guarantees that every represented class
has at least one train group.

## Atomic vocal units

Lead and backing sources can be configured with `atomic_mode: song_variant` plus a
private `song_key_rules` suffix rule. The dataset then treats both sources as one
sampling unit and never drops only one side. A MIDI containing both lead and backing
tracks keeps exact per-note `melody` and `vocal_harmony` targets.

## Same-song 1-to-N source mixtures

During training, extra sources can be selected only from the same song and the same
deployment stem group. Audio is mixed in memory and individual MIDI note lists are
merged. A full-song mix is never constructed.

Same-pitch notes from different source/classes whose onsets are within 30 ms are
merged into one partial-label query because the audio and query cannot distinguish
their original source.

## Class-balanced sampling

The default sampler first chooses an instrument class uniformly and then chooses an
atomic unit containing that class uniformly. Large classes therefore do not dominate
training. Use `--no-balanced-sampling` only for an unbalanced ablation.

## Train

```bash
python -m recipes.instrument_refinement.train \\
  --init-amt checkpoints/best_model.pth \\
  --freeze-backbone
```

Useful options:

- `--composite-probability`: probability of making a 1-to-N source mixture
- `--max-mixture-sources`: source cap that never breaks an atomic pair
- `--windows-per-source`: time windows generated from each selected source set
- `--no-balanced-sampling`: disable class-balanced sampling
- `--max-sources`: cap records for a smoke test without splitting an atomic unit
- `--dry-run`: run one train and one validation step

Validation uses one deterministic note-containing window per atomic unit instead of
scanning every hop of the full source collection.

## Infer

```bash
python -m instrument_agnostic_amt.instrument_refinement.cli.infer \\
  --audio separated/other.wav \\
  --midi amt/other.mid \\
  --stem-name other \\
  --checkpoint instrument_agnostic_amt/instrument_refinement/artifacts/checkpoints/best_model.pth \\
  --output-midi refined.mid \\
  --output-json refined.json
```

The MIDI program is not used as an instrument prior at inference. Classification
uses audio, pitch, onset, duration, and within-song timbre consistency.
