# Training recipes

`recipes/` contains training-only entrypoints, datasets, losses, augmentation,
and training utilities. Runtime model and inference code stays in
`instrument_agnostic_amt/`.

The dependency direction is one way: recipes may import
`instrument_agnostic_amt`, but the runtime package must not import recipes.

Each task has its own recipe directory:

```text
recipes/
|-- amt/
|-- beat_chord/
|-- instrument_refinement/
`-- velocity/
```

The existing top-level scripts remain stable entrypoints while the code is
moved incrementally.
