from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch

from instrument_agnostic_amt.beat_chord import (
    MidiFrameBeatChordModel,
    MidiFrameModelConfig,
)
from instrument_agnostic_amt.beat_chord.cli.infer import load_beat_chord_model
from instrument_agnostic_amt.instrument_refinement.modeling.model import (
    InstrumentRefinementConfig,
    InstrumentRefinementModel,
)
from instrument_agnostic_amt.inference.types import InferenceSettings
from instrument_agnostic_amt.inference.windowed import decode_notes
from instrument_agnostic_amt.modeling.features.cqt import RecursiveCQT
from instrument_agnostic_amt.modeling.model import (
    AudioSemiCRFTransformer,
    SemiCRFModelConfig,
)
from instrument_agnostic_amt.runtime import maybe_compile_forward
from instrument_agnostic_amt.velocity.modeling.model import (
    VelocityModelConfig,
    VelocityPredictionModel,
)


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is not available",
)


def _small_cqt() -> RecursiveCQT:
    return RecursiveCQT(
        sr=16_000,
        hop_length=64,
        fmin=250.0,
        n_bins=12,
        bins_per_octave=12,
        filter_scale=0.5,
    )


def _small_amt_model(
    semi_crf_version: str = "v1",
) -> AudioSemiCRFTransformer:
    config = SemiCRFModelConfig(
        sample_rate=16_000,
        hop_length=64,
        n_fft=1024,
        semi_crf_version=semi_crf_version,
        cqt_fmin=250.0,
        cqt_n_bins=12,
        cqt_bins_per_octave=12,
        cqt_filter_scale=0.5,
        harmonics=(1.0,),
        hidden_size=8,
        base_ch=4,
        encoder_num_layers=1,
        encoder_num_heads=2,
        dropout=0.0,
        use_gradient_checkpoint=False,
        semi_crf_head_dim=8,
        num_instrument_classes=2,
        instrument_pair_gate_dim=8,
    )
    return AudioSemiCRFTransformer(config).eval()


def _small_v1_kwargs() -> dict[str, object]:
    return {
        "sample_rate": 16_000,
        "hop_length": 64,
        "cqt_fmin": 250.0,
        "cqt_n_bins": 12,
        "cqt_bins_per_octave": 12,
        "cqt_filter_scale": 0.5,
        "harmonics": (1.0,),
        "hidden_size": 8,
        "base_ch": 4,
        "encoder_num_layers": 0,
        "encoder_num_heads": 2,
        "dropout": 0.0,
        "use_gradient_checkpoint": False,
        "pitch_min": 21,
        "pitch_max": 32,
    }


def _small_velocity_model(
    device: torch.device | str = "cpu",
    *,
    predict_stem_gain: bool = True,
    encoder_num_layers: int = 0,
) -> VelocityPredictionModel:
    with torch.random.fork_rng():
        torch.manual_seed(42)
        model_kwargs = _small_v1_kwargs()
        model_kwargs["encoder_num_layers"] = encoder_num_layers
        model = (
            VelocityPredictionModel(
                VelocityModelConfig(
                    **model_kwargs,
                    note_hidden_size=8,
                    num_stem_classes=1,
                    local_frame_offsets=(0,),
                    predict_stem_gain=predict_stem_gain,
                )
            )
            .eval()
            .to(device)
        )
    return model


def _velocity_inputs(
    note_count: int,
    *,
    device: torch.device | str = "cpu",
    sample_count: int = 1024,
) -> dict[str, torch.Tensor]:
    # アクセラレータ間で同じ入力値を使えるよう、CPU上で決定的に生成する。
    waveform = torch.linspace(-0.25, 0.25, steps=2 * sample_count).reshape(
        1, 1, 2, sample_count
    )
    note_starts = torch.linspace(0.004, 0.030, steps=note_count).reshape(
        1, note_count
    )
    return {
        "audio": waveform.to(device),
        "note_start_seconds": note_starts.to(device),
        "note_end_seconds": (note_starts + 0.008).to(device),
        "note_pitch": (
            (torch.arange(note_count) % 12 + 21).reshape(1, note_count).to(device)
        ),
        "note_program": torch.zeros(
            1, note_count, dtype=torch.long, device=device
        ),
        "note_is_drum": torch.zeros(
            1, note_count, dtype=torch.bool, device=device
        ),
        "note_stem_index": torch.zeros(
            1, note_count, dtype=torch.long, device=device
        ),
        "stem_class_id": torch.zeros(1, 1, dtype=torch.long, device=device),
        "valid_audio_frames": torch.full(
            (1, 1), sample_count, dtype=torch.long, device=device
        ),
    }


def test_recursive_cqt_mps_matches_cpu() -> None:
    generator = torch.Generator().manual_seed(42)
    waveform = torch.randn(2, 4096, generator=generator)
    cpu_output = _small_cqt()(waveform)
    mps_output = _small_cqt().to("mps")(waveform.to("mps")).cpu()

    assert torch.isfinite(mps_output).all()
    torch.testing.assert_close(mps_output, cpu_output, rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("amp_dtype", [torch.float16, torch.bfloat16])
def test_recursive_cqt_supports_mps_autocast(amp_dtype: torch.dtype) -> None:
    waveform = torch.randn(2, 4096, device="mps")
    cqt = _small_cqt().to("mps")

    with torch.autocast(device_type="mps", dtype=amp_dtype):
        output = cqt(waveform)

    assert output.device.type == "mps"
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("amp_dtype", [None, torch.float16, torch.bfloat16])
def test_core_amt_forward_runs_on_mps(amp_dtype: torch.dtype | None) -> None:
    model = _small_amt_model().to("mps")
    waveform = torch.randn(1, 2, 4096, device="mps")
    valid_frames = torch.tensor([4096], device="mps")

    with torch.autocast(
        device_type="mps",
        dtype=amp_dtype or torch.bfloat16,
        enabled=amp_dtype is not None,
    ):
        outputs = model(
            waveform,
            valid_audio_frames=valid_frames,
            include_aux_outputs=False,
        )

    tensors = [value for value in outputs.values() if isinstance(value, torch.Tensor)]
    assert tensors
    assert all(value.device.type == "mps" for value in tensors)
    assert all(torch.isfinite(value).all() for value in tensors)


def test_core_amt_v2_forward_runs_on_mps() -> None:
    model = _small_amt_model("v2").to("mps")
    outputs = model(
        torch.randn(1, 2, 4096, device="mps"),
        valid_audio_frames=torch.tensor([4096], device="mps"),
        include_aux_outputs=False,
    )

    tensors = [value for value in outputs.values() if isinstance(value, torch.Tensor)]
    assert tensors
    assert all(value.device.type == "mps" for value in tensors)
    assert all(torch.isfinite(value).all() for value in tensors)


def test_core_amt_mps_matches_cpu() -> None:
    torch.manual_seed(7)
    cpu_model = _small_amt_model()
    mps_model = copy.deepcopy(cpu_model).to("mps")
    waveform = torch.randn(1, 2, 4096)
    valid_frames = torch.tensor([4096])

    cpu_outputs = cpu_model(
        waveform,
        valid_audio_frames=valid_frames,
        include_aux_outputs=False,
    )
    mps_outputs = mps_model(
        waveform.to("mps"),
        valid_audio_frames=valid_frames.to("mps"),
        include_aux_outputs=False,
    )

    for name, cpu_value in cpu_outputs.items():
        mps_value = mps_outputs[name]
        if not isinstance(cpu_value, torch.Tensor):
            assert mps_value is None
            continue
        assert isinstance(mps_value, torch.Tensor)
        torch.testing.assert_close(
            mps_value.cpu(),
            cpu_value,
            rtol=3e-3,
            atol=3e-4,
        )


def test_v2_semi_crf_decode_runs_on_mps_and_matches_cpu() -> None:
    cpu_model = _small_amt_model("v2")
    with torch.no_grad():
        for parameter in cpu_model.head.pair_gate.parameters():
            parameter.zero_()
        cpu_model.head.pair_gate.instrument_bias[0] = 1.0
        cpu_model.head.pair_gate.instrument_bias[1] = -1.0
        for parameter in cpu_model.head.interval_scorer.parameters():
            parameter.zero_()
    mps_model = copy.deepcopy(cpu_model).to("mps")
    settings = InferenceSettings(
        window_ms=64,
        stride_ms=64,
        track_batch_size=1,
        window_batch_size=1,
        merge_gap_ms=None,
        merge_onset_ms=0.0,
        silence_gate_rms_dbfs=None,
        note_bias=-100.0,
        disable_tqdm=True,
        use_boundary_head=False,
        instrument_pair_infer_topk=0,
        instrument_pair_gate_threshold=0.5,
        instrument_pair_max_pairs=1,
        allowed_instrument_ids=(0,),
    )
    waveform = torch.randn(2, 1024, generator=torch.Generator().manual_seed(19))

    cpu_notes, cpu_stats = decode_notes(
        cpu_model,
        cpu_model.config,
        waveform,
        instrument_filter_id=None,
        device=torch.device("cpu"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=settings,
        velocity=100,
    )
    mps_notes, mps_stats = decode_notes(
        mps_model,
        mps_model.config,
        waveform,
        instrument_filter_id=None,
        device=torch.device("mps"),
        amp_enabled=False,
        amp_dtype=torch.float32,
        settings=settings,
        velocity=100,
    )

    assert cpu_notes == mps_notes
    assert cpu_stats == mps_stats
    assert mps_stats["selected_pair_count"] == 1


def test_velocity_forward_runs_on_mps() -> None:
    device = torch.device("mps")
    model = _small_velocity_model(device)

    with torch.inference_mode():
        outputs = model(**_velocity_inputs(1, device=device))

    assert outputs["velocity_logits"].shape == (1, 1, 127)
    assert all(
        value.device.type == "mps" and torch.isfinite(value).all()
        for value in outputs.values()
    )


def test_refinement_forward_runs_on_mps() -> None:
    device = torch.device("mps")
    model = InstrumentRefinementModel(
        InstrumentRefinementConfig(
            **_small_v1_kwargs(),
            note_hidden_size=8,
            embedding_size=4,
            onset_frame_offsets=(0,),
            sustain_fractions=(0.5,),
            release_frame_offsets=(0,),
        )
    ).eval().to(device)

    with torch.inference_mode():
        outputs = model(
            torch.randn(1, 2, 1024, device=device),
            note_start_seconds=torch.tensor([[0.004]], device=device),
            note_end_seconds=torch.tensor([[0.012]], device=device),
            note_pitch=torch.tensor([[24]], device=device),
            note_prior_class=torch.tensor([[-1]], device=device),
            stem_context_id=torch.tensor([0], device=device),
            valid_audio_frames=torch.tensor([1024], device=device),
            window_seconds=1024 / 16_000,
        )

    assert outputs["note_logits"].shape == (1, 1, 36)
    assert outputs["window_embedding"].shape == (1, 4)
    assert all(
        value.device.type == "mps" and torch.isfinite(value).all()
        for value in outputs.values()
    )


def test_beat_chord_forward_runs_on_mps() -> None:
    device = torch.device("mps")
    config = MidiFrameModelConfig(
        sample_rate=22_050,
        hop_length=512,
        pitch_min=21,
        pitch_max=32,
        num_input_channels=4,
        base_ch=2,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        num_global_tokens=2,
        dropout=0.0,
        num_meter_classes=3,
        num_root_chord_classes=25,
    )
    model = MidiFrameBeatChordModel(config).eval().to(device)

    with torch.inference_mode():
        outputs = model(
            torch.zeros(1, 4, 32, 12, device=device),
            include_beat=True,
            include_chord=True,
        )

    assert outputs["beat_logits"].shape == (1, 32)
    assert outputs["root_chord_logits"].shape == (1, 32, 25)
    tensors = [value for value in outputs.values() if isinstance(value, torch.Tensor)]
    assert all(
        value.device.type == "mps" and torch.isfinite(value).all()
        for value in tensors
    )


def test_stem_splitter_forward_runs_on_mps() -> None:
    module = pytest.importorskip("stem_splitter.band_split_roformer")
    model = module.BSRoformer(
        dim=8,
        num_layers=1,
        sample_rate=16_000,
        num_channels=2,
        head_dim=4,
        num_heads=2,
        n_fft=64,
        hop_length=16,
        n_bands=4,
        num_stems=2,
        use_gradient_checkpoint=False,
    ).eval().to("mps")

    with torch.inference_mode():
        output = model(torch.randn(1, 2, 256, device="mps"))

    assert output.shape == (1, 2, 2, 256)
    assert output.device.type == "mps"
    assert torch.isfinite(output).all()


def test_beat_chord_checkpoint_loads_on_mps(tmp_path: Path) -> None:
    config = MidiFrameModelConfig(
        sample_rate=22_050,
        hop_length=512,
        pitch_min=21,
        pitch_max=32,
        num_input_channels=4,
        base_ch=2,
        hidden_size=8,
        num_layers=1,
        num_heads=2,
        num_global_tokens=2,
        dropout=0.0,
        num_meter_classes=3,
        num_root_chord_classes=25,
    )
    source_model = MidiFrameBeatChordModel(config)
    checkpoint_path = tmp_path / "beat_chord.pth"
    torch.save(
        {
            "model_config": config.__dict__,
            "model_state_dict": source_model.state_dict(),
            "beat_meter_classes": [[3, 4], [4, 4], [6, 8]],
            "chord_quality_map": {"0": "", "1": "m", "2": "N"},
        },
        checkpoint_path,
    )

    loaded_model, loaded_config, _ = load_beat_chord_model(
        checkpoint_path,
        device="mps",
    )

    assert loaded_config == config
    assert all(parameter.device.type == "mps" for parameter in loaded_model.parameters())


@pytest.mark.skipif(
    os.environ.get("RUN_ACCELERATOR_COMPILE_TEST") != "1",
    reason="accelerator compile regression is opt-in",
)
def test_core_amt_compiled_forward_runs_on_mps() -> None:
    torch.manual_seed(23)
    model = _small_amt_model().to("mps")
    eager_model = copy.deepcopy(model)
    compiled_forward = maybe_compile_forward(model, enabled=True)
    waveform = torch.randn(1, 2, 4096).to("mps")
    valid_frames = torch.tensor([4096], device="mps")

    for amp_dtype in (None, torch.float16, torch.bfloat16):
        with torch.autocast(
            device_type="mps",
            dtype=amp_dtype or torch.bfloat16,
            enabled=amp_dtype is not None,
        ):
            eager_outputs = eager_model(
                waveform,
                valid_audio_frames=valid_frames,
                include_aux_outputs=False,
            )
            outputs = compiled_forward(
                waveform,
                valid_audio_frames=valid_frames,
                include_aux_outputs=False,
            )

        tensors = [
            value for value in outputs.values() if isinstance(value, torch.Tensor)
        ]
        assert tensors
        assert all(value.device.type == "mps" for value in tensors)
        assert all(torch.isfinite(value).all() for value in tensors)
        for name, eager_value in eager_outputs.items():
            compiled_value = outputs[name]
            if not isinstance(eager_value, torch.Tensor):
                assert compiled_value is None
                continue
            assert isinstance(compiled_value, torch.Tensor)
            if eager_value.dtype == torch.bfloat16:
                rtol, atol = 5e-2, 2e-2
            elif eager_value.dtype == torch.float16:
                rtol, atol = 2e-2, 2e-3
            elif eager_value.is_floating_point() or eager_value.is_complex():
                rtol, atol = 1e-4, 1e-5
            else:
                rtol, atol = 0.0, 0.0
            torch.testing.assert_close(
                compiled_value,
                eager_value,
                rtol=rtol,
                atol=atol,
            )


@pytest.mark.skipif(
    os.environ.get("RUN_ACCELERATOR_COMPILE_TEST") != "1",
    reason="accelerator compile regression is opt-in",
)
def test_velocity_compiled_forward_handles_varying_note_counts_on_mps() -> None:
    device = torch.device("mps")
    model = _small_velocity_model(
        device,
        predict_stem_gain=False,
        encoder_num_layers=1,
    )
    eager_model = copy.deepcopy(model)
    compiled_forward = maybe_compile_forward(model, enabled=True)

    with torch.inference_mode():
        for note_count, sample_count in ((1, 1024), (2, 768), (3, 640)):
            inputs = _velocity_inputs(
                note_count,
                device=device,
                sample_count=sample_count,
            )
            eager_outputs = eager_model(**inputs)
            compiled_outputs = compiled_forward(**inputs)

            assert compiled_outputs.keys() == eager_outputs.keys()
            for name, eager_value in eager_outputs.items():
                compiled_value = compiled_outputs[name]
                assert compiled_value.device.type == "mps"
                assert torch.isfinite(compiled_value).all()
                torch.testing.assert_close(
                    compiled_value,
                    eager_value,
                    rtol=1e-4,
                    atol=1e-5,
                )
