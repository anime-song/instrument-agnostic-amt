from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pretty_midi
import pytest
import torch

from instrument_agnostic_amt.beat_chord import key_only_candidates as candidates
from instrument_agnostic_amt.beat_chord.key_only_candidates import (
    StemTranscriptionRunner,
    build_midi_frame_infer_command,
    candidate_paths,
    discover_audio_files,
    merge_stem_midis,
    parse_arguments,
    resolve_beat_chord_checkpoint,
    resolve_stem_model_type,
    run_batch,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def _write_midi(
    path: Path,
    *,
    program: int,
    name: str,
    notes: list[tuple[float, float, int]],
) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=program, name=name)
    instrument.notes = [
        pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=start,
            end=end,
        )
        for start, end, pitch in notes
    ]
    midi.instruments.append(instrument)
    midi.write(str(path))


def _write_prediction_json(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"beats": [], "downbeats": [], "meters": [], "chords": [], "keys": []}',
        encoding="utf-8",
    )


def _make_runner(*, predict_velocity: bool = True) -> StemTranscriptionRunner:
    return StemTranscriptionRunner(
        device="cpu",
        amt_checkpoint_dir="checkpoints",
        separation_checkpoint="checkpoints/stem_splitter.pt",
        velocity_checkpoint="checkpoints/best_velocity_model.pth",
        window_batch_size=1,
        max_melodic_instruments=15,
        merge_onset_ms=50.0,
        transcribe_drums=True,
        predict_velocity=predict_velocity,
        strict_velocity=True,
        force=False,
        cleanup_stems=False,
    )


def test_batch_runner_auto_routes_to_mps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    runner = StemTranscriptionRunner(
        device="auto",
        amt_checkpoint_dir="checkpoints",
        separation_checkpoint="checkpoints/stem_splitter.pt",
        velocity_checkpoint="checkpoints/best_velocity_model.pth",
        window_batch_size=1,
        max_melodic_instruments=15,
        merge_onset_ms=50.0,
        transcribe_drums=True,
        predict_velocity=True,
        strict_velocity=True,
        force=False,
        cleanup_stems=False,
    )

    assert runner.device.type == "mps"


def test_discover_audio_files_filters_and_sorts(tmp_path: Path) -> None:
    _touch(tmp_path / "b.mp3")
    _touch(tmp_path / "A.wav")
    _touch(tmp_path / "ignored.txt")
    _touch(tmp_path / "nested" / "c.flac")

    assert [path.name for path in discover_audio_files(tmp_path)] == [
        "A.wav",
        "b.mp3",
    ]
    assert [path.name for path in discover_audio_files(tmp_path, recursive=True)] == [
        "A.wav",
        "b.mp3",
        "c.flac",
    ]


def test_discover_audio_files_rejects_duplicate_song_ids(tmp_path: Path) -> None:
    _touch(tmp_path / "same.mp3")
    _touch(tmp_path / "same.wav")

    with pytest.raises(ValueError, match="unique stems"):
        discover_audio_files(tmp_path)


def test_checkpoint_auto_resolution_prefers_current_mtime(tmp_path: Path) -> None:
    old_high_epoch = tmp_path / "checkpoint_epoch_999.pth"
    current_lower_epoch = tmp_path / "checkpoint_epoch_12.pth"
    _touch(old_high_epoch)
    _touch(current_lower_epoch)
    os.utime(old_high_epoch, (10, 10))
    os.utime(current_lower_epoch, (20, 20))

    assert resolve_beat_chord_checkpoint(None, checkpoint_dir=tmp_path) == (
        current_lower_epoch.resolve()
    )


@pytest.mark.parametrize(
    ("stem_name", "expected"),
    [
        ("drums", "drums"),
        ("bass", "bass_v2"),
        ("vocals", "vocal_harmony"),
        ("guitar", "guitar_v1_5"),
        ("other", "other_v1_5"),
        ("piano", "default"),
    ],
)
def test_stem_model_types_match_colab(stem_name: str, expected: str) -> None:
    assert resolve_stem_model_type(stem_name) == expected


def test_batch_cli_exposes_core_amt_amp_options() -> None:
    defaults = parse_arguments([])
    enabled = parse_arguments(["--amp", "--amp-dtype", "bf16"])

    assert (defaults.amp, defaults.amp_dtype) == (False, None)
    assert (enabled.amp, enabled.amp_dtype) == (True, "bf16")


def test_run_batch_routes_amp_options_to_stem_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner_options: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            runner_options.update(kwargs)

        @staticmethod
        def release() -> None:
            pass

    monkeypatch.setattr(candidates, "_validate_arguments", lambda _args: None)
    monkeypatch.setattr(candidates, "discover_audio_files", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        candidates,
        "resolve_beat_chord_checkpoint",
        lambda *_args, **_kwargs: tmp_path / "beat_chord.pth",
    )
    monkeypatch.setattr(candidates, "StemTranscriptionRunner", FakeRunner)
    args = parse_arguments(
        [
            "--output-dir",
            str(tmp_path / "output"),
            "--amp",
            "--amp-dtype",
            "bf16",
        ]
    )

    results = run_batch(args)

    assert results == []
    assert (runner_options.get("amp"), runner_options.get("amp_dtype")) == (
        True,
        "bf16",
    )


def test_batch_runner_routes_amp_to_core_amt_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instrument_agnostic_amt.amt.cli import infer as amt_infer

    inference_calls: list[dict[str, object]] = []

    class Waveform:
        def to(self, _device: torch.device) -> Waveform:
            return self

    class Midi:
        @staticmethod
        def write(path: str) -> None:
            Path(path).write_bytes(b"midi")

    monkeypatch.setattr(candidates, "resolve_device", lambda _value: torch.device("mps"))
    monkeypatch.setattr(
        amt_infer,
        "_ensure_checkpoint",
        lambda *_args, **_kwargs: tmp_path / "checkpoint.pth",
    )
    monkeypatch.setattr(
        amt_infer,
        "_load_model_and_settings",
        lambda *_args, **_kwargs: (
            object(),
            SimpleNamespace(num_instrument_classes=2, sample_rate=1_000),
            object(),
        ),
    )
    monkeypatch.setattr(
        amt_infer,
        "resolve_stem_instrument_class_ids",
        lambda _stem_name: (0,),
    )
    monkeypatch.setattr(
        amt_infer,
        "filter_supported_instrument_class_ids",
        lambda values, **_kwargs: values,
    )
    monkeypatch.setattr(
        amt_infer,
        "_load_audio",
        lambda *_args, **_kwargs: (Waveform(), 1_000, 2),
    )

    def fake_run_inference(**kwargs: object) -> tuple[list[object], dict, dict]:
        inference_calls.append(kwargs)
        return [], {}, {}

    monkeypatch.setattr(amt_infer, "run_inference", fake_run_inference)
    monkeypatch.setattr(amt_infer, "_build_midi", lambda *_args, **_kwargs: Midi())
    monkeypatch.setattr(
        candidates,
        "is_valid_midi_file",
        lambda path: Path(path).is_file(),
    )
    runner = StemTranscriptionRunner(
        device="mps",
        amt_checkpoint_dir=tmp_path,
        separation_checkpoint=tmp_path / "separation.pth",
        velocity_checkpoint=tmp_path / "velocity.pth",
        window_batch_size=1,
        max_melodic_instruments=15,
        merge_onset_ms=50.0,
        transcribe_drums=True,
        predict_velocity=False,
        strict_velocity=True,
        force=False,
        cleanup_stems=False,
        amp=True,
        amp_dtype="bf16",
    )

    runner._transcribe_stem(
        stem_name="piano",
        stem_path=tmp_path / "piano.wav",
        output_midi=tmp_path / "piano.mid",
    )

    assert len(inference_calls) == 1
    assert inference_calls[0]["amp_enabled"] is True
    assert inference_calls[0]["amp_dtype"] is torch.bfloat16


def test_batch_cli_exposes_core_amt_compile_options() -> None:
    defaults = parse_arguments([])
    enabled = parse_arguments(["--compile", "--compile-mode", "max-autotune"])

    assert (
        getattr(defaults, "compile", None),
        getattr(defaults, "compile_mode", None),
        getattr(enabled, "compile", None),
        getattr(enabled, "compile_mode", None),
    ) == (False, "default", True, "max-autotune")


def test_batch_cli_velocity_compile_is_independent_from_core_amt() -> None:
    defaults = parse_arguments([])
    velocity = parse_arguments(
        ["--compile-velocity", "--compile-mode", "reduce-overhead"]
    )

    assert (
        getattr(defaults, "compile_velocity", None),
        velocity.compile,
        getattr(velocity, "compile_velocity", None),
        velocity.compile_mode,
    ) == (False, False, True, "reduce-overhead")


def test_batch_runner_routes_compiled_forward_to_core_amt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instrument_agnostic_amt.amt.cli import infer as amt_infer

    eager_model = object()
    compiled_forward = object()
    compile_calls: list[tuple[object, bool, str]] = []
    inference_calls: list[dict[str, object]] = []

    class _Waveform:
        def to(self, _device: torch.device) -> _Waveform:
            return self

    class _Midi:
        @staticmethod
        def write(path: str) -> None:
            Path(path).write_bytes(b"midi")

    def fake_compile(
        model: object,
        *,
        enabled: bool,
        mode: str,
    ) -> object:
        compile_calls.append((model, enabled, mode))
        return compiled_forward

    monkeypatch.setattr(
        candidates,
        "maybe_compile_forward",
        fake_compile,
        raising=False,
    )
    monkeypatch.setattr(
        amt_infer,
        "_ensure_checkpoint",
        lambda *_args, **_kwargs: tmp_path / "checkpoint.pth",
    )
    monkeypatch.setattr(
        amt_infer,
        "_load_model_and_settings",
        lambda *_args, **_kwargs: (
            eager_model,
            SimpleNamespace(num_instrument_classes=2, sample_rate=1_000),
            object(),
        ),
    )
    monkeypatch.setattr(
        amt_infer,
        "resolve_stem_instrument_class_ids",
        lambda _stem_name: (0,),
    )
    monkeypatch.setattr(
        amt_infer,
        "filter_supported_instrument_class_ids",
        lambda values, **_kwargs: values,
    )
    monkeypatch.setattr(
        amt_infer,
        "_load_audio",
        lambda *_args, **_kwargs: (_Waveform(), 1_000, 2),
    )

    def fake_run_inference(**kwargs: object) -> tuple[list[object], dict, dict]:
        inference_calls.append(kwargs)
        return [], {}, {}

    monkeypatch.setattr(amt_infer, "run_inference", fake_run_inference)
    monkeypatch.setattr(amt_infer, "_build_midi", lambda *_args, **_kwargs: _Midi())
    monkeypatch.setattr(
        candidates,
        "is_valid_midi_file",
        lambda path: Path(path).is_file(),
    )
    runner = StemTranscriptionRunner(
        device="cpu",
        amt_checkpoint_dir=tmp_path,
        separation_checkpoint=tmp_path / "separation.pth",
        velocity_checkpoint=tmp_path / "velocity.pth",
        window_batch_size=1,
        max_melodic_instruments=15,
        merge_onset_ms=50.0,
        transcribe_drums=True,
        predict_velocity=False,
        strict_velocity=True,
        force=False,
        cleanup_stems=False,
        compile_model=True,
        compile_mode="max-autotune",
    )

    runner._transcribe_stem(
        stem_name="piano",
        stem_path=tmp_path / "piano.wav",
        output_midi=tmp_path / "piano.mid",
    )

    assert compile_calls == [(eager_model, True, "max-autotune")]
    assert inference_calls[0]["model"] is eager_model
    assert inference_calls[0]["forward_model"] is compiled_forward


def test_batch_runner_reuses_compiled_velocity_forward_across_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from instrument_agnostic_amt.velocity.cli import infer_velocity

    class FakeModel:
        def to(self, _device: torch.device) -> FakeModel:
            return self

        def eval(self) -> FakeModel:
            return self

    eager_model = FakeModel()
    compiled_forward = object()
    config = object()
    compile_calls: list[tuple[object, bool, str]] = []
    inference_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        infer_velocity,
        "load_velocity_model",
        lambda *_args, **_kwargs: (eager_model, config),
    )

    def fake_compile(
        model: object,
        *,
        enabled: bool,
        mode: str,
    ) -> object:
        compile_calls.append((model, enabled, mode))
        return compiled_forward

    def fake_predict(**kwargs: object) -> Path:
        inference_calls.append(kwargs)
        output_midi = Path(str(kwargs["output_midi_path"]))
        _write_midi(
            output_midi,
            program=0,
            name="piano",
            notes=[(0.0, 1.0, 60)],
        )
        return output_midi

    monkeypatch.setattr(candidates, "maybe_compile_forward", fake_compile)
    monkeypatch.setattr(
        infer_velocity,
        "predict_velocity_for_stem_midis",
        fake_predict,
    )
    runner = StemTranscriptionRunner(
        device="cpu",
        amt_checkpoint_dir=tmp_path,
        separation_checkpoint=tmp_path / "separation.pth",
        velocity_checkpoint=tmp_path / "velocity.pth",
        window_batch_size=1,
        max_melodic_instruments=15,
        merge_onset_ms=50.0,
        transcribe_drums=True,
        predict_velocity=True,
        strict_velocity=True,
        force=False,
        cleanup_stems=False,
        compile_velocity=True,
        compile_mode="max-autotune",
    )
    apply_kwargs = {
        "stems": {"piano": tmp_path / "piano.wav"},
        "stem_midis": {"piano": tmp_path / "piano.mid"},
        "template_midi": tmp_path / "template.mid",
    }

    runner._apply_velocity(
        **apply_kwargs,
        output_midi=tmp_path / "first.mid",
    )
    runner._apply_velocity(
        **apply_kwargs,
        output_midi=tmp_path / "second.mid",
    )

    assert compile_calls == [(eager_model, True, "max-autotune")]
    assert [
        (
            call["preloaded_model"],
            call["preloaded_forward"],
            call["preloaded_config"],
        )
        for call in inference_calls
    ] == [
        (eager_model, compiled_forward, config),
        (eager_model, compiled_forward, config),
    ]


def test_merge_stem_midis_limits_instruments_and_long_notes(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.mid"
    second = tmp_path / "second.mid"
    third = tmp_path / "third.mid"
    _write_midi(
        first,
        program=0,
        name="piano",
        notes=[(0.0, 1.0, 60), (1.0, 2.0, 64), (2.0, 20.0, 67)],
    )
    _write_midi(second, program=24, name="guitar", notes=[(0.0, 1.0, 55)])
    _write_midi(third, program=48, name="strings", notes=[(0.0, 1.0, 72)])

    output = merge_stem_midis(
        [first, second, third],
        tmp_path / "merged.mid",
        max_melodic_instruments=2,
    )
    merged = pretty_midi.PrettyMIDI(str(output))

    assert len(merged.instruments) == 2
    assert merged.instruments[0].name == "piano"
    assert merged.instruments[1].name == "Other / Merged"
    assert sum(len(instrument.notes) for instrument in merged.instruments) == 4
    assert all(
        note.end - note.start < 15.0
        for instrument in merged.instruments
        for note in instrument.notes
    )


def test_midi_frame_command_writes_explicit_prediction_and_midi_paths(
    tmp_path: Path,
) -> None:
    command = build_midi_frame_infer_command(
        checkpoint=tmp_path / "checkpoint.pth",
        midi_path=tmp_path / "曲_velocity.mid",
        prediction_json_path=tmp_path / "prediction.json",
        candidate_midi_path=tmp_path / "曲_velocity.beat_mapped.mid",
        quality_json=tmp_path / "quality.json",
        device="cuda",
    )

    assert command[1:3] == [
        "-m",
        "instrument_agnostic_amt.beat_chord.cli.infer",
    ]
    assert command[command.index("--midi_path") + 1].endswith("曲_velocity.mid")
    assert command[command.index("--beat_mapped_midi_path") + 1].endswith(
        "曲_velocity.beat_mapped.mid"
    )


def test_prediction_completion_requires_readable_midi_and_json(
    tmp_path: Path,
) -> None:
    midi_path = tmp_path / "candidate.mid"
    json_path = tmp_path / "prediction.json"
    _write_midi(
        midi_path,
        program=0,
        name="piano",
        notes=[(0.0, 1.0, 60)],
    )

    assert not candidates.prediction_outputs_complete(
        candidate_midi_path=midi_path,
        prediction_json_path=json_path,
    )

    _write_prediction_json(json_path)

    assert candidates.prediction_outputs_complete(
        candidate_midi_path=midi_path,
        prediction_json_path=json_path,
    )


def test_batch_dry_run_never_loads_models(tmp_path: Path) -> None:
    input_dir = tmp_path / "audio"
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint.pth"
    quality_json = tmp_path / "quality.json"
    _touch(input_dir / "song.mp3")
    _touch(checkpoint)
    quality_json.write_text("{}", encoding="utf-8")
    args = parse_arguments(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--beat-chord-checkpoint",
            str(checkpoint),
            "--quality-json",
            str(quality_json),
            "--dry-run",
        ]
    )

    results = run_batch(args)

    assert len(results) == 1
    assert results[0].transcription_status == "planned"
    assert results[0].prediction_status == "planned"
    assert results[0].candidate_midi_path is not None
    assert results[0].candidate_midi_path.endswith("song_velocity.beat_mapped.mid")
    assert not output_dir.exists()


def test_run_song_reuses_completed_velocity_midi_without_stems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "song.mp3"
    output_root = tmp_path / "output"
    paths = candidate_paths(output_root, "song")
    paths.merged_dir.mkdir(parents=True)
    velocity_midi = paths.merged_dir / "song_velocity.mid"
    _write_midi(
        velocity_midi,
        program=0,
        name="piano",
        notes=[(0.0, 1.0, 60)],
    )
    runner = _make_runner()
    monkeypatch.setattr(
        runner,
        "_separate",
        lambda *_args, **_kwargs: pytest.fail("separation should be skipped"),
    )

    assert runner.run_song(audio_path, output_root=output_root) == velocity_midi


def test_run_song_transcribes_only_missing_stem_midi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "song.mp3"
    output_root = tmp_path / "output"
    paths = candidate_paths(output_root, "song")
    paths.stem_midi_dir.mkdir(parents=True)
    stem_names = ("bass", "drums", "other", "vocals", "guitar", "piano")
    for stem_name in stem_names[1:]:
        _write_midi(
            paths.stem_midi_dir / f"song_{stem_name}.mid",
            program=0,
            name=stem_name,
            notes=[(0.0, 1.0, 60)],
        )

    runner = _make_runner(predict_velocity=False)
    monkeypatch.setattr(
        runner,
        "_separate",
        lambda *_args, **_kwargs: {
            name: tmp_path / f"song_{name}.wav" for name in stem_names
        },
    )
    transcribed: list[str] = []

    def fake_transcribe(**kwargs: object) -> Path:
        stem_name = str(kwargs["stem_name"])
        output_midi = Path(str(kwargs["output_midi"]))
        transcribed.append(stem_name)
        _write_midi(
            output_midi,
            program=32,
            name=stem_name,
            notes=[(0.0, 1.0, 48)],
        )
        return output_midi

    monkeypatch.setattr(runner, "_transcribe_stem", fake_transcribe)

    result = runner.run_song(audio_path, output_root=output_root)

    assert transcribed == ["bass"]
    assert result == paths.merged_dir / "song.mid"
    assert candidates.is_valid_midi_file(result)


def test_batch_completes_each_song_before_starting_the_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "audio"
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint.pth"
    quality_json = tmp_path / "quality.json"
    _touch(input_dir / "a.mp3")
    _touch(input_dir / "b.mp3")
    _touch(checkpoint)
    quality_json.write_text("{}", encoding="utf-8")
    events: list[tuple[str, str]] = []
    runner_options: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, **kwargs: object) -> None:
            runner_options.update(kwargs)
            self.current_song = ""

        def run_song(self, audio_path: str, *, output_root: Path) -> Path:
            self.current_song = Path(audio_path).stem
            events.append(("transcribe", self.current_song))
            paths = candidate_paths(output_root, self.current_song)
            paths.merged_dir.mkdir(parents=True, exist_ok=True)
            midi_path = paths.merged_dir / f"{self.current_song}_velocity.mid"
            _write_midi(
                midi_path,
                program=0,
                name="piano",
                notes=[(0.0, 1.0, 60)],
            )
            return midi_path

        def park(self) -> None:
            events.append(("park", self.current_song))

        def release(self) -> None:
            events.append(("release", ""))

    def fake_subprocess_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> None:
        del cwd, check
        midi_path = Path(command[command.index("--midi_path") + 1])
        song_name = midi_path.stem.removesuffix("_velocity")
        events.append(("predict", song_name))
        candidate_midi = Path(command[command.index("--beat_mapped_midi_path") + 1])
        prediction_json = Path(command[command.index("--output_path") + 1])
        candidate_midi.parent.mkdir(parents=True, exist_ok=True)
        _write_midi(
            candidate_midi,
            program=0,
            name="piano",
            notes=[(0.0, 1.0, 60)],
        )
        _write_prediction_json(prediction_json)

    monkeypatch.setattr(candidates, "StemTranscriptionRunner", FakeRunner)
    monkeypatch.setattr(candidates.subprocess, "run", fake_subprocess_run)
    args = parse_arguments(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--beat-chord-checkpoint",
            str(checkpoint),
            "--quality-json",
            str(quality_json),
            "--compile-velocity",
            "--compile-mode",
            "max-autotune",
        ]
    )

    results = run_batch(args)

    assert events == [
        ("transcribe", "a"),
        ("park", "a"),
        ("predict", "a"),
        ("transcribe", "b"),
        ("park", "b"),
        ("predict", "b"),
        ("release", ""),
    ]
    assert [result.prediction_status for result in results] == [
        "completed",
        "completed",
    ]
    assert (
        runner_options["compile_model"],
        runner_options["compile_velocity"],
        runner_options["compile_mode"],
    ) == (False, True, "max-autotune")


def test_batch_reruns_prediction_when_only_final_midi_exists(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_dir = tmp_path / "audio"
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint.pth"
    quality_json = tmp_path / "quality.json"
    _touch(input_dir / "song.mp3")
    _touch(checkpoint)
    quality_json.write_text("{}", encoding="utf-8")
    paths = candidate_paths(output_dir, "song")
    paths.merged_dir.mkdir(parents=True)
    source_midi = paths.merged_dir / "song_velocity.mid"
    _write_midi(
        source_midi,
        program=0,
        name="piano",
        notes=[(0.0, 1.0, 60)],
    )
    final_midi = output_dir / "midis" / "song_velocity.beat_mapped.mid"
    final_midi.parent.mkdir(parents=True)
    _write_midi(
        final_midi,
        program=0,
        name="piano",
        notes=[(0.0, 1.0, 60)],
    )
    called = False

    class FakeRunner:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run_song(self, *_args: object, **_kwargs: object) -> Path:
            return source_midi

        def park(self) -> None:
            pass

        def release(self) -> None:
            pass

    def fake_subprocess_run(
        command: list[str],
        *,
        cwd: Path,
        check: bool,
    ) -> None:
        nonlocal called
        del cwd, check
        called = True
        partial_midi = Path(command[command.index("--beat_mapped_midi_path") + 1])
        partial_json = Path(command[command.index("--output_path") + 1])
        _write_midi(
            partial_midi,
            program=0,
            name="piano",
            notes=[(0.0, 1.0, 60)],
        )
        _write_prediction_json(partial_json)

    monkeypatch.setattr(candidates, "StemTranscriptionRunner", FakeRunner)
    monkeypatch.setattr(candidates.subprocess, "run", fake_subprocess_run)
    args = parse_arguments(
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--beat-chord-checkpoint",
            str(checkpoint),
            "--quality-json",
            str(quality_json),
        ]
    )

    result = run_batch(args)[0]

    assert called
    assert result.prediction_status == "completed"
    assert candidates.is_valid_prediction_json(
        paths.prediction_dir / "song_velocity.prediction.json"
    )
