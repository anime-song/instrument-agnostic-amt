"""Compatibility entry point for whole-stem instrument refinement."""

from instrument_agnostic_amt.instrument_refinement.cli.infer import (
    DEFAULT_REFINEMENT_CHECKPOINT_FILENAME,
    HF_CHECKPOINT_BASE_URL,
    ensure_refinement_checkpoint,
    main,
)
from instrument_agnostic_amt.instrument_refinement.inference.refine import (
    refine_midi_instruments,
)

__all__ = [
    "main",
    "refine_midi_instruments",
    "ensure_refinement_checkpoint",
    "DEFAULT_REFINEMENT_CHECKPOINT_FILENAME",
    "HF_CHECKPOINT_BASE_URL",
]


if __name__ == "__main__":
    main()
