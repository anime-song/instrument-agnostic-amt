"""V1-compatible inference entrypoint backed by the packaged implementation."""

from instrument_agnostic_amt.cli.infer import (
    DEFAULT_INSTRUMENT_VOLUMES,
    MODEL_CHECKPOINT_FILENAMES,
    _ensure_checkpoint,
    load_model,
    main,
    resolve_inference_settings,
)
from instrument_agnostic_amt.data.pitch_aliases import DEFAULT_DRUM_PITCH_ALIASES
from instrument_agnostic_amt.cli.compat import (
    _build_midi,
    _load_audio,
    _load_model_and_settings,
    run_inference,
)
from instrument_agnostic_amt.inference.instruments import (
    STEM_INSTRUMENT_CLASSES,
    filter_supported_instrument_class_ids,
    resolve_stem_instrument_class_ids,
)

__all__ = [
    "DEFAULT_DRUM_PITCH_ALIASES",
    "DEFAULT_INSTRUMENT_VOLUMES",
    "MODEL_CHECKPOINT_FILENAMES",
    "STEM_INSTRUMENT_CLASSES",
    "_build_midi",
    "_ensure_checkpoint",
    "_load_audio",
    "_load_model_and_settings",
    "filter_supported_instrument_class_ids",
    "load_model",
    "main",
    "resolve_inference_settings",
    "resolve_stem_instrument_class_ids",
    "run_inference",
]


if __name__ == "__main__":
    main()
