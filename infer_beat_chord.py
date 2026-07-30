"""Beat and chord inference entrypoint backed by the beat_chord CLI module."""

from __future__ import annotations

from instrument_agnostic_amt.beat_chord.cli.infer import (
    DEFAULT_BEAT_CHORD_CHECKPOINT_FILENAME,
    HF_CHECKPOINT_BASE_URL,
    ensure_beat_chord_checkpoint,
    load_beat_chord_model,
    main,
    predict_beat_chord_for_midi,
)

__all__ = [
    "main",
    "predict_beat_chord_for_midi",
    "load_beat_chord_model",
    "ensure_beat_chord_checkpoint",
    "DEFAULT_BEAT_CHORD_CHECKPOINT_FILENAME",
    "HF_CHECKPOINT_BASE_URL",
]

if __name__ == "__main__":
    main()
