from __future__ import annotations

import io
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from threading import Event

import mido
import pretty_midi
import pytest
import soundfile as sf
import torch

import instrument_agnostic_amt
import instrument_agnostic_amt.transcription as transcription_module
from instrument_agnostic_amt import (
    DecodedAudio,
    MidiExportOptions,
    Transcriber,
    TranscriberBusyError,
    TranscriptionOptions,
    TranscriptionResult,
)
from instrument_agnostic_amt.cli import infer as cli_infer
from instrument_agnostic_amt.data.pitch_aliases import DEFAULT_DRUM_PITCH_ALIASES
from instrument_agnostic_amt.inference.types import InferenceSettings, PredictedNote
from instrument_agnostic_amt.modeling.model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)


def _write_tiny_checkpoint(
    path: Path,
    *,
    num_instrument_classes: int = 3,
    semi_crf_version: str = "v1",
) -> None:
    config = SemiCRFModelConfig(
        sample_rate=8_000,
        hop_length=128,
        n_fft=256,
        semi_crf_version=semi_crf_version,
        cqt_n_bins=48,
        cqt_bins_per_octave=12,
        harmonics=(1.0,),
        hidden_size=16,
        base_ch=4,
        encoder_num_layers=1,
        encoder_num_heads=1,
        dropout=0.0,
        semi_crf_head_dim=5,
        num_instrument_classes=num_instrument_classes,
        instrument_pair_gate_dim=8,
        use_gradient_checkpoint=False,
    )
    model = AudioSemiCRFTransformer(config)
    torch.save(
        {
            "model_config": asdict(config),
            "model_state_dict": model.state_dict(),
            "config": {
                "args": {
                    "window_ms": 1_250,
                    "semi_crf_track_batch_size": 17,
                }
            },
        },
        path,
    )


def _inference_stats(**overrides: int) -> dict[str, int]:
    values = {
        "window_count": 1,
        "decoded_window_count": 1,
        "skipped_silent_window_count": 0,
        "window_audio_frames": 10_000,
        "stride_audio_frames": 5_000,
        "selected_pair_count": 0,
        "decoded_interval_count": 0,
        "boundary_interval_count": 0,
        "boundary_no_onset_count": 0,
        "boundary_no_offset_count": 0,
    }
    values.update(overrides)
    return values


def test_package_exports_the_public_transcription_api() -> None:
    expected_names = {
        "DecodedAudio",
        "MidiExportOptions",
        "Transcriber",
        "TranscriberBusyError",
        "TranscriptionModelInfo",
        "TranscriptionOptions",
        "TranscriptionResult",
    }

    assert expected_names <= set(dir(instrument_agnostic_amt))


def test_decoded_audio_rejects_a_nonpositive_sample_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        DecodedAudio(samples=torch.zeros(1, 16), sample_rate=0)


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        (torch.zeros(16), "shape"),
        (torch.zeros(3, 16), "one or two channels"),
        (torch.zeros(1, 0), "must not be empty"),
        (torch.zeros(1, 16, dtype=torch.float64), "float32"),
    ],
)
def test_decoded_audio_rejects_noncanonical_pcm(
    samples: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DecodedAudio(samples=samples, sample_rate=44_100)


def test_transcription_options_reject_conflicting_instrument_filters() -> None:
    with pytest.raises(
        ValueError,
        match="instrument and allowed_instruments are mutually exclusive",
    ):
        TranscriptionOptions(
            instrument="drums",
            allowed_instruments=("drums", "wind_chimes"),
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("allowed_instruments", (), "must not be empty"),
        ("window_ms", 0, "window_ms must be positive"),
        ("stride_ms", 0, "stride_ms must be positive"),
        ("window_batch_size", 0, "window_batch_size must be positive"),
        (
            "semi_crf_track_batch_size",
            0,
            "semi_crf_track_batch_size must be positive",
        ),
        (
            "semi_crf_sparse_topk_per_start",
            0,
            "semi_crf_sparse_topk_per_start must be positive",
        ),
        ("instrument_pair_infer_topk", -1, "must be non-negative"),
        ("instrument_pair_max_pairs", 0, "must be positive"),
        ("merge_gap_ms", -1.0, "merge_gap_ms must be non-negative"),
        ("merge_onset_ms", -1.0, "merge_onset_ms must be non-negative"),
        ("silence_gate_rms_dbfs", 1.0, "must be <= 0"),
        ("midi_velocity", 128, "midi_velocity must be within MIDI 1..127"),
    ],
)
def test_transcription_options_reject_invalid_values(
    keyword: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        TranscriptionOptions(**{keyword: value})  # type: ignore[arg-type]


def test_sparse_decode_requires_a_torch_span_limit() -> None:
    with pytest.raises(ValueError, match="requires semi_crf_sparse_max_span_ms"):
        TranscriptionOptions(semi_crf_sparse_decode=True)
    with pytest.raises(ValueError, match="requires semi_crf_backend='torch'"):
        TranscriptionOptions(
            semi_crf_sparse_decode=True,
            semi_crf_sparse_max_span_ms=100.0,
            semi_crf_backend="triton",
        )


def test_midi_export_defaults_canonicalize_drums_without_adding_cc_volume() -> None:
    options = MidiExportOptions()

    assert options.instrument_volumes is None
    assert options.drum_pitch_aliases == DEFAULT_DRUM_PITCH_ALIASES
    assert options.drum_pitch_aliases is not DEFAULT_DRUM_PITCH_ALIASES


def test_midi_export_accepts_an_explicit_empty_volume_map() -> None:
    options = MidiExportOptions(instrument_volumes={})

    assert options.instrument_volumes == {}


def test_midi_export_options_snapshot_input_mappings_and_support_asdict() -> None:
    volumes = {"acoustic_bass": 80}
    aliases = {57: 49}
    options = MidiExportOptions(
        instrument_volumes=volumes,
        drum_pitch_aliases=aliases,
    )
    volumes["acoustic_bass"] = 120
    aliases[57] = 50

    serialized = asdict(options)

    assert serialized["instrument_volumes"] == {"acoustic_bass": 80}
    assert serialized["drum_pitch_aliases"] == {57: 49}


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"min_midi_note_ms": -1.0}, "must be non-negative"),
        ({"max_midi_melodic_instruments": -1}, "must be non-negative"),
        (
            {"instrument_volumes": {"acoustic_bass": 128}},
            "must be within MIDI 0..127",
        ),
        (
            {"instrument_volumes": {"unknown": 80}},
            "Unknown instrument class",
        ),
    ],
)
def test_midi_export_options_reject_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        MidiExportOptions(**kwargs)  # type: ignore[arg-type]


def test_cli_defaults_map_to_explicit_public_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["infer.py", "--audio", "input.wav"])
    args = cli_infer.parse_args()

    transcription_options = cli_infer._transcription_options_from_args(args)
    midi_options = cli_infer._midi_export_options_from_args(args)

    assert transcription_options == TranscriptionOptions(show_progress=True)
    assert midi_options.instrument_volumes == cli_infer.DEFAULT_INSTRUMENT_VOLUMES
    assert midi_options.drum_pitch_aliases == DEFAULT_DRUM_PITCH_ALIASES


def test_single_file_cli_writes_parseable_midi_with_a_real_transcriber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    audio_path = tmp_path / "silence.wav"
    output_path = tmp_path / "silence.mid"
    _write_tiny_checkpoint(checkpoint_path)
    sf.write(audio_path, torch.zeros(8_000).numpy(), 8_000, subtype="FLOAT")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer.py",
            "--audio",
            str(audio_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output-midi",
            str(output_path),
            "--device",
            "cpu",
            "--disable-tqdm",
        ],
    )

    cli_infer.main()

    midi = mido.MidiFile(output_path)
    assert len(midi.tracks) >= 1
    assert "wrote 0 notes:" in capsys.readouterr().out


def test_public_cli_adapter_matches_legacy_process_file_midi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    audio_path = tmp_path / "input.wav"
    legacy_output = tmp_path / "legacy.mid"
    _write_tiny_checkpoint(checkpoint_path)
    sf.write(audio_path, torch.zeros(8_000).numpy(), 8_000, subtype="FLOAT")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer.py",
            "--audio",
            str(audio_path),
            "--checkpoint",
            str(checkpoint_path),
            "--instrument",
            "acoustic_bass",
            "--device",
            "cpu",
            "--disable-tqdm",
        ],
    )
    args = cli_infer.parse_args()
    device = torch.device("cpu")
    model, config, checkpoint_args = cli_infer.load_model(
        checkpoint_path,
        device=device,
    )
    settings = cli_infer.resolve_inference_settings(config, checkpoint_args, args)

    def fake_decode_notes(
        *_args: object,
        velocity: int,
        **_kwargs: object,
    ) -> tuple[list[PredictedNote], dict[str, int]]:
        return [
            PredictedNote(
                instrument_id=1,
                pitch=45,
                start_sample=800,
                end_sample=1_600,
                velocity=velocity,
            )
        ], _inference_stats(selected_pair_count=1, decoded_interval_count=1)

    monkeypatch.setattr(cli_infer, "decode_notes", fake_decode_notes)
    cli_infer.process_file(
        audio_path,
        legacy_output,
        model=model,
        config=config,
        instrument_id=cli_infer.resolve_instrument_id(args.instrument),
        settings=settings,
        device=device,
        amp_enabled=False,
        amp_dtype=torch.float32,
        args=args,
    )
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device=device)
    monkeypatch.setattr(transcription_module, "decode_notes", fake_decode_notes)

    result = transcriber.transcribe(
        audio_path,
        options=cli_infer._transcription_options_from_args(args),
        midi_options=cli_infer._midi_export_options_from_args(args),
    )

    assert result.midi_bytes == legacy_output.read_bytes()


def test_transcriber_loads_one_explicit_checkpoint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)

    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")

    assert transcriber.checkpoint_path == checkpoint_path.resolve()
    assert transcriber.sample_rate == 8_000
    assert transcriber.device == torch.device("cpu")
    assert transcriber.supported_instruments == (
        "accordion_family",
        "acoustic_bass",
        "acoustic_guitar",
    )
    assert asdict(transcriber.model_info) == {
        "checkpoint_path": checkpoint_path.resolve(),
        "sample_rate": 8_000,
        "input_audio_channels": 2,
        "num_instrument_classes": 3,
        "supported_instruments": (
            "accordion_family",
            "acoustic_bass",
            "acoustic_guitar",
        ),
        "semi_crf_version": "v1",
        "device": "cpu",
        "amp_enabled": False,
        "amp_dtype": "torch.float32",
        "compile_enabled": False,
        "compile_mode": None,
    }
    assert capsys.readouterr() == ("", "")


def test_transcriber_rejects_a_checkpoint_without_inference_heads(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "backbone-only.pth"
    _write_tiny_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["model_state_dict"] = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("backbone.")
    }
    torch.save(checkpoint, checkpoint_path)

    with pytest.raises(ValueError, match="inference head"):
        Transcriber.from_checkpoint(checkpoint_path, device="cpu")


def test_transcriber_accepts_the_legacy_v1_instrument_classifier_fallback(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "legacy-v1.pth"
    _write_tiny_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["model_state_dict"] = {
        key: value
        for key, value in checkpoint["model_state_dict"].items()
        if not key.startswith("head.interval_instrument_predictor.")
        and not key.startswith("head.slot_embedding.")
    }
    torch.save(checkpoint, checkpoint_path)

    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")

    assert any(
        key.startswith("head.interval_instrument_predictor.")
        for key in transcriber.load_report.missing_keys
    )


def test_transcriber_accepts_a_complete_v2_inference_head(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "v2.pth"
    _write_tiny_checkpoint(checkpoint_path, semi_crf_version="v2")

    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")

    assert transcriber.model_info.semi_crf_version == "v2"


def test_transcriber_rejects_an_instrument_head_larger_than_the_taxonomy(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "unknown-instruments.pth"
    _write_tiny_checkpoint(checkpoint_path, num_instrument_classes=37)

    with pytest.raises(ValueError, match="taxonomy defines 36"):
        Transcriber.from_checkpoint(checkpoint_path, device="cpu")


def test_transcriber_returns_a_zero_note_result_for_silent_decoded_audio(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")
    audio = DecodedAudio(
        samples=torch.zeros(1, 4_000, dtype=torch.float32),
        sample_rate=4_000,
    )

    result = transcriber.transcribe(audio)

    assert isinstance(result, TranscriptionResult)
    assert result.notes == ()
    assert result.sample_rate == 8_000
    assert result.settings.window_ms == 1_250
    assert result.settings.stride_ms == 625
    assert result.settings.track_batch_size == 17
    assert result.settings.disable_tqdm is True
    assert result.model_info == transcriber.model_info
    assert result.inference_stats["skipped_silent_window_count"] == 1
    assert result.midi_stats["midi_instrument_count_after_remap"] == 0
    assert len(mido.MidiFile(file=io.BytesIO(result.midi_bytes)).tracks) >= 1
    assert capsys.readouterr() == ("", "")


def test_distilled_inference_config_overrides_training_defaults(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "distilled.pth"
    _write_tiny_checkpoint(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["inference_config"] = {"window_ms": 1_500}
    torch.save(checkpoint, checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")

    result = transcriber.transcribe(
        DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000)
    )

    assert result.settings.window_ms == 1_500
    assert result.settings.stride_ms == 750
    assert result.settings.track_batch_size == 17


def test_path_and_decoded_audio_inputs_have_the_same_semantics(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")
    samples = torch.zeros(1, 4_000, dtype=torch.float32)
    audio_path = tmp_path / "silence.wav"
    sf.write(audio_path, samples.squeeze(0).numpy(), 4_000, subtype="FLOAT")

    path_result = transcriber.transcribe(audio_path)
    decoded_result = transcriber.transcribe(DecodedAudio(samples, sample_rate=4_000))

    assert path_result.notes == decoded_result.notes
    assert path_result.inference_stats == decoded_result.inference_stats
    assert path_result.midi_stats == decoded_result.midi_stats
    assert path_result.midi_bytes == decoded_result.midi_bytes


def test_transcriber_resolves_instruments_and_returns_midi_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")
    predicted_note = PredictedNote(
        instrument_id=1,
        pitch=45,
        start_sample=800,
        end_sample=1_600,
        velocity=64,
    )

    def fake_decode_notes(
        _model: object,
        _config: object,
        waveform: torch.Tensor,
        *,
        instrument_filter_id: int | None,
        settings: InferenceSettings,
        velocity: int,
        **_kwargs: object,
    ) -> tuple[list[PredictedNote], dict[str, int]]:
        assert tuple(waveform.shape) == (2, 8_000)
        assert instrument_filter_id is None
        assert settings.allowed_instrument_ids == (1,)
        assert velocity == 64
        return [predicted_note], _inference_stats(
            selected_pair_count=1,
            decoded_interval_count=1,
        )

    monkeypatch.setattr(transcription_module, "decode_notes", fake_decode_notes)

    result = transcriber.transcribe(
        DecodedAudio(torch.zeros(1, 4_000), sample_rate=4_000),
        options=TranscriptionOptions(
            allowed_instruments=("acoustic_bass",),
            midi_velocity=64,
        ),
        midi_options=MidiExportOptions(
            instrument_volumes={"acoustic_bass": 87},
        ),
    )

    midi = pretty_midi.PrettyMIDI(
        mido_object=mido.MidiFile(file=io.BytesIO(result.midi_bytes))
    )
    assert result.notes == (predicted_note,)
    assert result.inference_stats["decoded_interval_count"] == 1
    assert result.midi_stats["midi_instrument_count_after_remap"] == 1
    assert len(midi.instruments) == 1
    assert midi.instruments[0].notes[0].velocity == 64
    assert [
        (control.number, control.value)
        for control in midi.instruments[0].control_changes
    ] == [(7, 87)]


def test_single_instrument_option_uses_the_hard_decoder_filter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")

    def fake_decode_notes(
        *_args: object,
        instrument_filter_id: int | None,
        **_kwargs: object,
    ) -> tuple[list[PredictedNote], dict[str, int]]:
        assert instrument_filter_id == 1
        return [], _inference_stats()

    monkeypatch.setattr(transcription_module, "decode_notes", fake_decode_notes)

    result = transcriber.transcribe(
        DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000),
        options=TranscriptionOptions(instrument="acoustic_bass"),
    )

    midi = pretty_midi.PrettyMIDI(
        mido_object=mido.MidiFile(file=io.BytesIO(result.midi_bytes))
    )
    assert midi.instruments == []


def test_transcriber_rejects_concurrent_calls_without_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")
    audio = DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000)
    first_call_entered = Event()
    release_first_call = Event()
    call_count = 0

    def blocking_decode_notes(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[list[PredictedNote], dict[str, int]]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            first_call_entered.set()
            assert release_first_call.wait(timeout=5.0)
        return [], _inference_stats()

    monkeypatch.setattr(
        transcription_module,
        "decode_notes",
        blocking_decode_notes,
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        first_result = executor.submit(transcriber.transcribe, audio)
        assert first_call_entered.wait(timeout=5.0)
        with pytest.raises(TranscriberBusyError):
            transcriber.transcribe(audio)
        release_first_call.set()
        first_result.result(timeout=5.0)

    transcriber.transcribe(audio)
    assert call_count == 2


def test_transcriber_rejects_calls_after_close(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")
    audio = DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000)
    metadata_before_close = (
        transcriber.model_info,
        transcriber.load_report,
        transcriber.checkpoint_path,
        transcriber.sample_rate,
        transcriber.device,
        transcriber.supported_instruments,
    )
    transcriber.close()

    assert (
        transcriber.model_info,
        transcriber.load_report,
        transcriber.checkpoint_path,
        transcriber.sample_rate,
        transcriber.device,
        transcriber.supported_instruments,
    ) == metadata_before_close
    with pytest.raises(RuntimeError, match="closed"):
        transcriber.transcribe(audio)


def test_transcriber_context_manager_closes_on_exit(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")

    with transcriber as entered:
        assert entered is transcriber

    assert transcriber.closed is True


def test_transcriber_close_empties_its_device_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")
    emptied_devices: list[torch.device] = []
    monkeypatch.setattr(
        transcription_module,
        "empty_device_cache",
        emptied_devices.append,
    )

    transcriber.close()

    assert emptied_devices == [torch.device("cpu")]


def test_transcriber_close_is_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")
    emptied_devices: list[torch.device] = []
    monkeypatch.setattr(
        transcription_module,
        "empty_device_cache",
        emptied_devices.append,
    )

    transcriber.close()
    transcriber.close()

    assert emptied_devices == [torch.device("cpu")]


def test_transcriber_close_waits_for_an_active_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")
    audio = DecodedAudio(torch.zeros(2, 8_000), sample_rate=8_000)
    call_entered = Event()
    release_call = Event()
    close_started = Event()
    close_finished = Event()

    def blocking_decode_notes(
        *_args: object,
        **_kwargs: object,
    ) -> tuple[list[PredictedNote], dict[str, int]]:
        call_entered.set()
        assert release_call.wait(timeout=5.0)
        return [], _inference_stats()

    def close_transcriber() -> None:
        close_started.set()
        transcriber.close()
        close_finished.set()

    monkeypatch.setattr(
        transcription_module,
        "decode_notes",
        blocking_decode_notes,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        call_result = executor.submit(transcriber.transcribe, audio)
        assert call_entered.wait(timeout=5.0)
        close_result = executor.submit(close_transcriber)
        assert close_started.wait(timeout=5.0)
        assert close_finished.wait(timeout=0.1) is False
        assert transcriber.closed is False
        release_call.set()
        assert call_result.result(timeout=5.0).notes == ()
        close_result.result(timeout=5.0)

    assert close_finished.is_set()
    assert transcriber.closed is True


def test_transcribe_leaves_decoded_audio_samples_unchanged(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "tiny.pth"
    _write_tiny_checkpoint(checkpoint_path)
    transcriber = Transcriber.from_checkpoint(checkpoint_path, device="cpu")
    samples = torch.linspace(-1.0, 1.0, steps=8_000).repeat(2, 1)
    samples_before = samples.clone()

    transcriber.transcribe(DecodedAudio(samples, sample_rate=8_000))

    assert torch.equal(samples, samples_before)
