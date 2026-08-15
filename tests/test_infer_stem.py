from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch

import infer_stem
from infer_stem import (
    get_refinement_models,
    get_stem_pipeline_models,
    get_velocity_models,
    merge_midis_logic,
    refine_stem_instrument_midis,
    resolve_stem_model_type,
    resolve_stem_paths,
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
    assert resolve_stem_model_type("other_stem") == "other"
    assert resolve_stem_model_type("piano_stem") == "default"


def test_stem_pipeline_compiles_only_the_amt_forward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.pth"
    checkpoint.write_bytes(b"")
    eager_model = torch.nn.Linear(2, 2)
    compiled_forward = object()
    compile_calls: list[tuple[object, bool, str, str]] = []

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
        scope: str,
    ) -> object:
        compile_calls.append((model, enabled, mode, scope))
        return compiled_forward

    monkeypatch.setattr(infer_stem, "maybe_compile_forward", fake_compile)
    infer_stem.STEM_PIPELINE_CACHE.clear()
    infer_stem.STEM_PIPELINE_CACHE[("sep", "cpu")] = (
        SimpleNamespace(),
        object(),
        torch.float32,
    )

    bundle = get_stem_pipeline_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        compile_model=True,
        compile_mode="max-autotune",
        compile_scope="regional",
    )

    assert bundle["amt_model"] is eager_model
    assert bundle["amt_forward"] is compiled_forward
    assert compile_calls == [(eager_model, True, "max-autotune", "regional")]


def test_velocity_pipeline_compile_is_independent_and_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "velocity.pth"
    checkpoint.write_bytes(b"")
    eager_model = torch.nn.Linear(2, 2)
    compiled_forward = object()
    load_calls: list[tuple[Path, torch.device]] = []
    compile_calls: list[tuple[object, bool, str, str]] = []

    monkeypatch.setattr(
        infer_stem,
        "ensure_velocity_checkpoint",
        lambda _path: checkpoint,
    )

    def fake_load(
        path: Path,
        *,
        device: torch.device,
    ) -> tuple[object, object]:
        load_calls.append((path, device))
        return eager_model, object()

    monkeypatch.setattr(infer_stem, "load_velocity_model", fake_load)

    def fake_compile(
        model: object,
        *,
        enabled: bool,
        mode: str,
        scope: str,
    ) -> object:
        compile_calls.append((model, enabled, mode, scope))
        return compiled_forward

    monkeypatch.setattr(infer_stem, "maybe_compile_forward", fake_compile)
    infer_stem.STEM_PIPELINE_CACHE.clear()

    first = get_velocity_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        compile_velocity=True,
        compile_mode="reduce-overhead",
        compile_scope="regional",
        window_seconds=8.0,
    )
    second = get_velocity_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        compile_velocity=True,
        compile_mode="reduce-overhead",
        compile_scope="regional",
        window_seconds=8.0,
    )
    different_scope = get_velocity_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        compile_velocity=True,
        compile_mode="reduce-overhead",
        compile_scope="whole",
        window_seconds=8.0,
    )
    different_window = get_velocity_models(
        checkpoint_path=checkpoint,
        device_preference="cpu",
        compile_velocity=True,
        compile_mode="reduce-overhead",
        compile_scope="regional",
        window_seconds=4.0,
    )

    assert first is second
    assert different_scope is not first
    assert different_window is not first
    assert first[0] is eager_model
    assert first[1] is compiled_forward
    assert load_calls == [
        (checkpoint, torch.device("cpu")),
        (checkpoint, torch.device("cpu")),
        (checkpoint, torch.device("cpu")),
    ]
    assert compile_calls == [
        (eager_model, True, "reduce-overhead", "regional"),
        (eager_model, True, "reduce-overhead", "whole"),
        (eager_model, True, "reduce-overhead", "regional"),
    ]


def test_refinement_cache_distinguishes_cuda_device_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "refinement.pth"
    checkpoint.write_bytes(b"")
    loaded_devices: list[str] = []

    monkeypatch.setattr(
        infer_stem,
        "resolve_device",
        lambda preference: torch.device(str(preference)),
    )
    monkeypatch.setattr(
        infer_stem,
        "ensure_refinement_checkpoint",
        lambda _path: checkpoint,
    )

    def fake_load(
        _path: Path,
        *,
        device: torch.device,
    ) -> tuple[object, object, object]:
        loaded_devices.append(str(device))
        return object(), object(), object()

    monkeypatch.setattr(infer_stem, "load_refinement_model", fake_load)
    infer_stem.STEM_PIPELINE_CACHE.clear()

    get_refinement_models(checkpoint, device_preference="cuda:0")
    get_refinement_models(checkpoint, device_preference="cuda:1")

    assert loaded_devices == ["cuda:0", "cuda:1"]


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

    def to(self, _device: torch.device) -> "_FakeRefinementModel":
        return self

    def eval(self) -> "_FakeRefinementModel":
        return self

    def __call__(
        self, _audio: torch.Tensor, **kwargs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        pitches = kwargs["note_pitch"]
        batch_size, note_count = pitches.shape
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
    refined_paths = refine_stem_instrument_midis(
        stem_midis=stem_midis,
        stem_audios=stem_audios,
        output_dir=tmp_path / "refined",
        refinement_model=_FakeRefinementModel(config),
        refinement_config=config,
        device="cpu",
        window_seconds=0.5,
        stride_seconds=0.25,
    )

    assert set(refined_paths) == {"other"}
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
