"""V1-compatible velocity inference entrypoint backed by the velocity CLI module."""

from __future__ import annotations

from instrument_agnostic_amt.velocity.cli.infer_velocity import (
    ensure_velocity_checkpoint,
    load_velocity_model,
    main,
    predict_velocity_for_midi,
    predict_velocity_for_stem_midis,
)

__all__ = [
    "main",
    "predict_velocity_for_midi",
    "predict_velocity_for_stem_midis",
    "load_velocity_model",
    "ensure_velocity_checkpoint",
]

if __name__ == "__main__":
    main()
