"""V1 metadata postprocessing compatibility module.

V2 inference writes a single requested-instrument MIDI directly from
`instrument_agnostic_amt.cli.infer`. Beat/chord metadata embedding and
instrument-classifier based multi-track routing were removed from the V2 model.
"""

from __future__ import annotations


class V1PostprocessRemoved(RuntimeError):
    """Raised when removed V1 postprocessing is requested."""


def unavailable() -> None:
    raise V1PostprocessRemoved(
        "V1 beat/chord metadata postprocessing was removed in V2. "
        "Use `python -m instrument_agnostic_amt.cli.infer --checkpoint <v2.ckpt> "
        "--instrument <class> --audio <file>` instead."
    )
