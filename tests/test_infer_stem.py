from __future__ import annotations

import gc
import weakref
from inspect import signature
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch

import infer_stem
from infer_stem import (
    get_stem_pipeline_models,
    merge_midis_logic,
    refine_stem_instrument_midis,
    resolve_stem_model_type,
    resolve_stem_paths,
    run_stem_separated_transcription,
)
from instrument_agnostic_amt.instrument_refinement.modeling.model import (
    InstrumentRefinementConfig,
)
from instrument_agnostic_amt.taxonomy.instrument_classes import (
    get_instrument_class_id_by_name,
)


def test_resolve_stem_model_type() -> None:
    assert resolve_stem_model_type("drums_stem") == "drums"
    assert resolve_stem_model_type("bass_stem") == "bass_v2"
    assert resolve_stem_model_type("vocal_stem") == "vocal_harmony"
    assert resolve_stem_model_type("guitar_stem") == "guitar_v1_5"
    assert resolve_stem_model_type("other_stem") == "other_v1_5"
    assert resolve_stem_model_type("piano_stem") == "default"


def test_stem_pipeline_auto_routes_models_to_mps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"")
    loaded_devices: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(
        infer_stem.infer,
        "_ensure_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )

    def fake_load_separator(
        _config: object,
        *,
        device: torch.device,
    ) -> object:
        loaded_devices.append(str(device))
        return object()

    def fake_load_amt(
        _checkpoint: Path,
        *,
        device: torch.device,
        **_kwargs: object,
    ) -> tuple[object, object, object]:
        loaded_devices.append(str(device))
        return object(), SimpleNamespace(), object()

    monkeypatch.setattr(infer_stem, "load_mss_model", fake_load_separator)
    monkeypatch.setattr(
        infer_stem.infer,
        "_load_model_and_settings",
        fake_load_amt,
    )
    infer_stem.STEM_PIPELINE_CACHE.clear()

    bundle = get_stem_pipeline_models(checkpoint_path=checkpoint)

    assert (str(bundle["device"]), loaded_devices) == ("mps", ["mps", "mps"])


def test_stem_pipeline_compiles_and_caches_only_the_core_amt_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"")
    eager_model = torch.nn.Linear(2, 2)
    compiled_forward = object()
    compile_calls: list[tuple[object, bool, str]] = []

    monkeypatch.setattr(
        infer_stem.infer,
        "_ensure_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        infer_stem.infer,
        "_load_model_and_settings",
        lambda *_args, **_kwargs: (eager_model, object(), object()),
    )

    def fake_compile(
        model: object,
        *,
        enabled: bool,
        mode: str,
    ) -> object:
        compile_calls.append((model, enabled, mode))
        return compiled_forward if enabled else model

    monkeypatch.setattr(
        infer_stem,
        "maybe_compile_forward",
        fake_compile,
        raising=False,
    )
    infer_stem.STEM_PIPELINE_CACHE.clear()
    infer_stem.STEM_PIPELINE_CACHE[("sep", "cpu")] = (
        SimpleNamespace(),
        object(),
        torch.float32,
    )

    first = get_stem_pipeline_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        compile_model=True,
        compile_mode="max-autotune",
    )
    second = get_stem_pipeline_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        compile_model=True,
        compile_mode="max-autotune",
    )
    eager = get_stem_pipeline_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
    )

    assert first["amt_model"] is eager_model
    assert first["amt_forward"] is compiled_forward
    assert second["amt_forward"] is compiled_forward
    assert eager["amt_forward"] is eager_model
    assert compile_calls == [
        (eager_model, True, "max-autotune"),
        (eager_model, False, "default"),
    ]


def test_velocity_pipeline_compile_is_independent_and_cached_by_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_velocity_models = getattr(infer_stem, "get_velocity_models", None)
    assert callable(get_velocity_models)
    checkpoint = tmp_path / "velocity.pth"
    checkpoint.write_bytes(b"")
    eager_model = torch.nn.Linear(2, 2)
    compiled_forward = object()
    load_calls: list[tuple[Path, torch.device]] = []
    compile_calls: list[tuple[object, bool, str]] = []

    monkeypatch.setattr(
        infer_stem,
        "ensure_velocity_checkpoint",
        lambda _path: checkpoint,
        raising=False,
    )

    def fake_load(
        path: Path,
        *,
        device: torch.device,
    ) -> tuple[object, object]:
        load_calls.append((path, device))
        return eager_model, object()

    monkeypatch.setattr(
        infer_stem,
        "load_velocity_model",
        fake_load,
        raising=False,
    )

    def fake_compile(
        model: object,
        *,
        enabled: bool,
        mode: str,
    ) -> object:
        compile_calls.append((model, enabled, mode))
        return compiled_forward

    monkeypatch.setattr(infer_stem, "maybe_compile_forward", fake_compile)
    infer_stem.STEM_PIPELINE_CACHE.clear()

    first = get_velocity_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        compile_velocity=True,
        compile_mode="reduce-overhead",
    )
    second = get_velocity_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        compile_velocity=True,
        compile_mode="reduce-overhead",
    )
    different_mode = get_velocity_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        compile_velocity=True,
        compile_mode="max-autotune",
    )

    assert first is second
    assert different_mode is not first
    assert first[0] is eager_model
    assert first[1] is compiled_forward
    assert load_calls == [
        (checkpoint, torch.device("cpu")),
        (checkpoint, torch.device("cpu")),
    ]
    assert compile_calls == [
        (eager_model, True, "reduce-overhead"),
        (eager_model, True, "max-autotune"),
    ]


def test_stem_workflow_exposes_device_and_amp_options() -> None:
    parameters = signature(run_stem_separated_transcription).parameters

    assert (
        parameters.get("stem_splitter_batch_size").default
        if parameters.get("stem_splitter_batch_size")
        else None,
        parameters.get("device").default if parameters.get("device") else None,
        parameters.get("amp").default if parameters.get("amp") else None,
        parameters.get("amp_dtype").default if parameters.get("amp_dtype") else "missing",
        parameters.get("compile_model").default
        if parameters.get("compile_model")
        else None,
        parameters.get("compile_velocity").default
        if parameters.get("compile_velocity")
        else None,
        parameters.get("compile_mode").default
        if parameters.get("compile_mode")
        else None,
    ) == (1, "auto", False, None, False, False, "default")


def test_stem_pipeline_changes_separator_batch_size_without_reloading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"")
    separator_load_batch_sizes: list[int] = []

    monkeypatch.setattr(
        infer_stem.infer,
        "_ensure_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        infer_stem.infer,
        "_load_model_and_settings",
        lambda *_args, **_kwargs: (object(), object(), object()),
    )

    def fake_load_separator(config: SimpleNamespace, **_kwargs: object) -> object:
        separator_load_batch_sizes.append(config.batch_size)
        return object()

    monkeypatch.setattr(
        infer_stem,
        "load_mss_model",
        fake_load_separator,
    )
    infer_stem.STEM_PIPELINE_CACHE.clear()

    batch_one = get_stem_pipeline_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        stem_splitter_batch_size=1,
    )
    batch_four = get_stem_pipeline_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        stem_splitter_batch_size=4,
    )

    assert batch_one["sep_config"].batch_size == 1
    assert batch_four["sep_config"].batch_size == 4
    assert batch_one["sep_model"] is batch_four["sep_model"]
    assert separator_load_batch_sizes == [1]


def test_stem_pipeline_rejects_invalid_separator_batch_size() -> None:
    with pytest.raises(ValueError, match="stem_splitter_batch_size"):
        get_stem_pipeline_models(stem_splitter_batch_size=0)


def test_merge_midis_logic(tmp_path: Path) -> None:
    midi1 = pretty_midi.PrettyMIDI()
    inst1 = pretty_midi.Instrument(program=0, name="Piano")
    inst1.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=1.0))
    midi1.instruments.append(inst1)
    path1 = tmp_path / "stem1.mid"
    midi1.write(str(path1))

    midi2 = pretty_midi.PrettyMIDI()
    inst2 = pretty_midi.Instrument(program=33, name="Bass")
    inst2.notes.append(pretty_midi.Note(velocity=90, pitch=36, start=0.0, end=1.0))
    midi2.instruments.append(inst2)
    path2 = tmp_path / "stem2.mid"
    midi2.write(str(path2))

    output_path = tmp_path / "merged.mid"
    merge_midis_logic([path1, path2], output_path, max_melodic_instruments=15)

    assert output_path.exists()
    merged_midi = pretty_midi.PrettyMIDI(str(output_path))
    assert len(merged_midi.instruments) == 2


class _FakeRefinementModel:
    """常に acoustic_guitar を最有力候補として返す refinement モデルの代役。"""

    def __init__(self, config: InstrumentRefinementConfig) -> None:
        self.config = config
        self.batch_sizes: list[int] = []

    def to(self, _device: torch.device) -> "_FakeRefinementModel":
        return self

    def eval(self) -> "_FakeRefinementModel":
        return self

    def __call__(
        self, _audio: torch.Tensor, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pitches = kwargs["note_pitch"]
        batch_size, note_count = pitches.shape
        self.batch_sizes.append(int(batch_size))
        device = pitches.device
        acoustic_guitar = get_instrument_class_id_by_name("acoustic_guitar")
        note_logits = torch.zeros(
            batch_size, note_count, self.config.num_instrument_classes, device=device
        )
        note_embedding = torch.zeros(
            batch_size, note_count, self.config.embedding_size, device=device
        )
        if note_count:
            note_logits[..., acoustic_guitar] = 5.0
            note_embedding[..., 0] = 1.0
        window_logits = torch.zeros(
            batch_size, self.config.num_instrument_classes, device=device
        )
        window_logits[..., acoustic_guitar] = 3.0
        window_embedding = torch.zeros(
            batch_size, self.config.embedding_size, device=device
        )
        window_embedding[..., 0] = 1.0
        return {
            "note_logits": note_logits,
            "note_embedding": note_embedding,
            "window_logits": window_logits,
            "window_embedding": window_embedding,
        }


def _write_stem_audio(path: Path, *, frequency: float, duration: float = 1.0) -> None:
    sample_rate = 8_000
    time = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    mono = 0.1 * np.sin(2.0 * np.pi * frequency * time)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.column_stack((mono, mono)), sample_rate, subtype="FLOAT")


def _write_stem_midi(path: Path, *, program: int, name: str, is_drum: bool = False) -> None:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    for index, pitch in enumerate((60, 67)):
        start = 0.1 + index * 0.4
        instrument.notes.append(
            pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.25)
        )
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def test_refine_stem_instrument_midis_skips_excluded_stems_and_rewrites_labels(
    tmp_path: Path,
) -> None:
    stem_audios = {
        "other": tmp_path / "song_other.wav",
        "drums": tmp_path / "song_drums.wav",
        "vocals": tmp_path / "song_vocals.wav",
    }
    stem_midis = {
        "other": tmp_path / "song_other.mid",
        "drums": tmp_path / "song_drums.mid",
        "vocals": tmp_path / "song_vocals.mid",
    }
    for stem_name, audio_path in stem_audios.items():
        _write_stem_audio(audio_path, frequency=220.0 if stem_name == "other" else 90.0)
    _write_stem_midi(stem_midis["other"], program=48, name="strings")
    _write_stem_midi(stem_midis["drums"], program=0, name="drums", is_drum=True)
    _write_stem_midi(stem_midis["vocals"], program=65, name="melody")

    config = InstrumentRefinementConfig(
        sample_rate=8_000,
        hop_length=800,
        hidden_size=24,
        base_ch=8,
        encoder_num_layers=0,
        encoder_num_heads=3,
        note_hidden_size=32,
        embedding_size=8,
        use_gradient_checkpoint=False,
    )
    model = _FakeRefinementModel(config)
    refined_paths = refine_stem_instrument_midis(
        stem_midis=stem_midis,
        stem_audios=stem_audios,
        output_dir=tmp_path / "refined",
        refinement_model=model,
        refinement_config=config,
        device="cpu",
        window_seconds=0.5,
        stride_seconds=0.25,
        window_batch_size=2,
    )

    assert set(refined_paths) == {"other"}
    assert model.batch_sizes == [2, 1]
    refined_midi = pretty_midi.PrettyMIDI(str(refined_paths["other"]))
    assert [instrument.name for instrument in refined_midi.instruments] == [
        "acoustic_guitar"
    ]


def test_refine_stem_instrument_midis_excludes_vocals_even_when_selected(
    tmp_path: Path,
) -> None:
    """vocals はドラムと同じく、明示指定しても常に refine 対象外にする。"""
    stem_audios = {"vocals": tmp_path / "song_vocals.wav"}
    stem_midis = {"vocals": tmp_path / "song_vocals.mid"}
    _write_stem_audio(stem_audios["vocals"], frequency=220.0)
    _write_stem_midi(stem_midis["vocals"], program=65, name="melody")

    config = InstrumentRefinementConfig(
        sample_rate=8_000,
        hop_length=800,
        hidden_size=24,
        base_ch=8,
        encoder_num_layers=0,
        encoder_num_heads=3,
        note_hidden_size=32,
        embedding_size=8,
        use_gradient_checkpoint=False,
    )
    refined_paths = refine_stem_instrument_midis(
        stem_midis=stem_midis,
        stem_audios=stem_audios,
        output_dir=tmp_path / "refined",
        refinement_model=_FakeRefinementModel(config),
        refinement_config=config,
        device="cpu",
        stem_names=("vocals",),
        window_seconds=0.5,
        stride_seconds=0.25,
    )

    assert refined_paths == {}


def test_refine_stem_instrument_midis_honors_stem_selection(tmp_path: Path) -> None:
    stem_audios = {"other": tmp_path / "song_other.wav", "bass": tmp_path / "song_bass.wav"}
    stem_midis = {"other": tmp_path / "song_other.mid", "bass": tmp_path / "song_bass.mid"}
    for audio_path in stem_audios.values():
        _write_stem_audio(audio_path, frequency=220.0)
    _write_stem_midi(stem_midis["other"], program=48, name="strings")
    _write_stem_midi(stem_midis["bass"], program=33, name="electric_bass")

    config = InstrumentRefinementConfig(
        sample_rate=8_000,
        hop_length=800,
        hidden_size=24,
        base_ch=8,
        encoder_num_layers=0,
        encoder_num_heads=3,
        note_hidden_size=32,
        embedding_size=8,
        use_gradient_checkpoint=False,
    )
    refined_paths = refine_stem_instrument_midis(
        stem_midis=stem_midis,
        stem_audios=stem_audios,
        output_dir=tmp_path / "refined",
        refinement_model=_FakeRefinementModel(config),
        refinement_config=config,
        device="cpu",
        stem_names=("other",),
        window_seconds=0.5,
        stride_seconds=0.25,
    )

    assert set(refined_paths) == {"other"}


def _install_fake_waveform_cache_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    amt_sample_rate: int = 8_000,
    refinement_sample_rate: int = 8_000,
    velocity_sample_rate: int = 8_000,
    retain_waveforms: bool = True,
    include_piano_stem: bool = False,
) -> SimpleNamespace:
    source_audio = tmp_path / "song.wav"
    stem_audio = tmp_path / "song_other.wav"
    _write_stem_audio(source_audio, frequency=440.0)
    _write_stem_audio(stem_audio, frequency=220.0)
    stems = {"other": stem_audio}
    if include_piano_stem:
        piano_audio = tmp_path / "song_piano.wav"
        _write_stem_audio(piano_audio, frequency=330.0)
        stems["piano"] = piano_audio

    model_config = SimpleNamespace(
        sample_rate=amt_sample_rate,
        num_instrument_classes=64,
    )
    refinement_config = SimpleNamespace(sample_rate=refinement_sample_rate)
    velocity_config = SimpleNamespace(sample_rate=velocity_sample_rate)
    amt_model = object()
    bundle = {
        "device": torch.device("cpu"),
        "amt_model": amt_model,
        "amt_forward": amt_model,
        "amt_config": model_config,
        "amt_settings": object(),
        "sep_config": SimpleNamespace(stem_names=tuple(stems)),
        "sep_model": object(),
        "sep_dtype": torch.float32,
    }
    monkeypatch.setattr(infer_stem, "get_stem_pipeline_models", lambda **_kwargs: bundle)
    monkeypatch.setattr(
        infer_stem,
        "prepare_audio_for_stem_separation",
        lambda *_args, **_kwargs: source_audio,
    )
    monkeypatch.setattr(
        infer_stem,
        "_separate_one_file",
        lambda *_args, **_kwargs: dict(stems),
    )

    loaded_waveforms: list[tuple[int, torch.Tensor]] = []
    loaded_waveform_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    def fake_load_audio(
        audio_path: Path,
        *,
        target_sample_rate: int,
    ) -> tuple[torch.Tensor, int, int]:
        if not Path(audio_path).is_file():
            raise FileNotFoundError(audio_path)
        waveform = torch.arange(16, dtype=torch.float32).reshape(2, 8).clone()
        loaded_waveform_refs.append(weakref.ref(waveform))
        if retain_waveforms:
            loaded_waveforms.append((int(target_sample_rate), waveform))
        return waveform, int(target_sample_rate), 2

    monkeypatch.setattr(infer_stem.infer, "_load_audio", fake_load_audio)
    amt_waveforms: list[torch.Tensor] = []

    def fake_run_inference(**kwargs: object) -> tuple[list[object], dict, dict]:
        if retain_waveforms:
            amt_waveforms.append(kwargs["waveform"])  # type: ignore[arg-type]
        return [], {}, {}

    monkeypatch.setattr(infer_stem.infer, "run_inference", fake_run_inference)

    def fake_build_midi(*_args: object, **_kwargs: object) -> pretty_midi.PrettyMIDI:
        midi = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0, name="piano")
        instrument.notes.append(
            pretty_midi.Note(velocity=100, pitch=60, start=0.1, end=0.4)
        )
        midi.instruments.append(instrument)
        return midi

    monkeypatch.setattr(infer_stem.infer, "_build_midi", fake_build_midi)
    monkeypatch.setattr(
        infer_stem,
        "get_refinement_models",
        lambda **_kwargs: {
            "refinement_model": object(),
            "refinement_config": refinement_config,
        },
    )

    refinement_waveforms: list[torch.Tensor] = []

    def fake_refine(
        audio_path: Path,
        midi_path: Path,
        *,
        output_midi_path: Path,
        preloaded_waveform: torch.Tensor | None = None,
        **_kwargs: object,
    ) -> dict[str, object]:
        waveform = preloaded_waveform
        if waveform is None:
            waveform, _, _ = fake_load_audio(
                audio_path,
                target_sample_rate=refinement_sample_rate,
            )
        if retain_waveforms:
            refinement_waveforms.append(waveform)
        output_midi_path.write_bytes(Path(midi_path).read_bytes())
        return {"note_count": 1, "cluster_count": 0, "clusters": []}

    monkeypatch.setattr(infer_stem, "refine_midi_instruments", fake_refine)
    monkeypatch.setattr(
        infer_stem,
        "get_velocity_models",
        lambda **_kwargs: (object(), object(), velocity_config),
        raising=False,
    )

    velocity_waveforms: list[torch.Tensor] = []

    def fake_velocity(
        *,
        stem_audios: dict[str, Path],
        output_midi_path: Path,
        template_midi_path: Path,
        preloaded_waveforms: dict[str, torch.Tensor] | None = None,
        **_kwargs: object,
    ) -> Path:
        if preloaded_waveforms is None:
            waveform, _, _ = fake_load_audio(
                stem_audios["other"],
                target_sample_rate=velocity_sample_rate,
            )
        else:
            waveform = preloaded_waveforms.get("other")
            if waveform is None:
                waveform = next(iter(preloaded_waveforms.values()))
        if retain_waveforms:
            velocity_waveforms.append(waveform)
        output_midi_path.write_bytes(template_midi_path.read_bytes())
        return output_midi_path

    monkeypatch.setattr(infer_stem, "predict_velocity_for_stem_midis", fake_velocity)

    beat_waveforms_alive: list[bool] = []

    def fake_beat_chord(
        *,
        input_midi_path: Path,
        output_midi_path: Path,
        **_kwargs: object,
    ) -> Path:
        gc.collect()
        beat_waveforms_alive.append(
            any(reference() is not None for reference in loaded_waveform_refs)
        )
        output_midi_path.write_bytes(input_midi_path.read_bytes())
        return output_midi_path

    monkeypatch.setattr(infer_stem, "predict_beat_chord_for_midi", fake_beat_chord)

    run_count = 0

    def run(
        *,
        refine_instruments: bool = True,
        predict_velocity: bool = True,
        predict_beat_chord: bool = False,
    ) -> dict[str, object]:
        nonlocal run_count
        run_count += 1
        return infer_stem.run_stem_separated_transcription(
            source_audio,
            output_root=tmp_path / f"output_{run_count}",
            refine_instruments=refine_instruments,
            predict_velocity=predict_velocity,
            predict_beat_chord=predict_beat_chord,
            device="cpu",
        )

    return SimpleNamespace(
        run=run,
        loaded_waveforms=loaded_waveforms,
        loaded_waveform_refs=loaded_waveform_refs,
        amt_waveforms=amt_waveforms,
        refinement_waveforms=refinement_waveforms,
        velocity_waveforms=velocity_waveforms,
        beat_waveforms_alive=beat_waveforms_alive,
        stems=stems,
    )


def test_stem_pipeline_reuses_waveform_across_audio_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_fake_waveform_cache_pipeline(tmp_path, monkeypatch)

    pipeline.run()

    assert [rate for rate, _ in pipeline.loaded_waveforms] == [8_000]
    waveform = pipeline.loaded_waveforms[0][1]
    assert pipeline.amt_waveforms[0] is waveform
    assert pipeline.refinement_waveforms[0] is waveform
    assert pipeline.velocity_waveforms[0] is waveform


def test_stem_pipeline_cache_distinguishes_sample_rates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_fake_waveform_cache_pipeline(
        tmp_path,
        monkeypatch,
        amt_sample_rate=8_000,
        refinement_sample_rate=16_000,
        velocity_sample_rate=16_000,
    )

    pipeline.run()

    assert [rate for rate, _ in pipeline.loaded_waveforms] == [8_000, 16_000]
    waveform_16k = pipeline.loaded_waveforms[1][1]
    assert pipeline.refinement_waveforms[0] is waveform_16k
    assert pipeline.velocity_waveforms[0] is waveform_16k


def test_stem_pipeline_does_not_share_waveforms_between_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_fake_waveform_cache_pipeline(tmp_path, monkeypatch)

    pipeline.run()
    pipeline.run()

    assert [rate for rate, _ in pipeline.loaded_waveforms] == [8_000, 8_000]
    first_waveform = pipeline.loaded_waveforms[0][1]
    second_waveform = pipeline.loaded_waveforms[1][1]
    assert first_waveform is not second_waveform
    assert pipeline.refinement_waveforms == [first_waveform, second_waveform]
    assert pipeline.velocity_waveforms == [first_waveform, second_waveform]


@pytest.mark.parametrize(
    ("refine_instruments", "predict_velocity"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_stem_pipeline_releases_waveforms_before_beat_chord(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    refine_instruments: bool,
    predict_velocity: bool,
) -> None:
    pipeline = _install_fake_waveform_cache_pipeline(
        tmp_path,
        monkeypatch,
        retain_waveforms=False,
    )

    pipeline.run(
        refine_instruments=refine_instruments,
        predict_velocity=predict_velocity,
        predict_beat_chord=True,
    )

    assert pipeline.beat_waveforms_alive == [False]


def test_stem_pipeline_releases_waveforms_when_velocity_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_fake_waveform_cache_pipeline(
        tmp_path,
        monkeypatch,
        retain_waveforms=False,
    )

    def fail_velocity(**_kwargs: object) -> None:
        raise RuntimeError("velocity failed")

    monkeypatch.setattr(
        infer_stem,
        "predict_velocity_for_stem_midis",
        fail_velocity,
    )

    pipeline.run(predict_beat_chord=True)

    assert pipeline.beat_waveforms_alive == [False]


def test_stem_pipeline_releases_waveforms_when_refinement_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_fake_waveform_cache_pipeline(
        tmp_path,
        monkeypatch,
        retain_waveforms=False,
    )

    def fail_refinement(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("refinement failed")

    monkeypatch.setattr(infer_stem, "refine_midi_instruments", fail_refinement)

    pipeline.run(predict_velocity=False, predict_beat_chord=True)

    assert pipeline.beat_waveforms_alive == [False]


def test_stem_pipeline_velocity_ignores_a_missing_stem_during_preload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = _install_fake_waveform_cache_pipeline(
        tmp_path,
        monkeypatch,
        velocity_sample_rate=16_000,
        include_piano_stem=True,
    )
    velocity_config = SimpleNamespace(sample_rate=16_000)

    def remove_other_before_velocity(
        **_kwargs: object,
    ) -> tuple[object, object, object]:
        pipeline.stems["other"].unlink()
        return object(), object(), velocity_config

    monkeypatch.setattr(
        infer_stem,
        "get_velocity_models",
        remove_other_before_velocity,
    )

    pipeline.run(refine_instruments=False)

    assert len(pipeline.velocity_waveforms) == 1
