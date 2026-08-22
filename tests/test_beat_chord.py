from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pretty_midi
import torch

from instrument_agnostic_amt.beat_chord import (
    MidiFrameBeatChordModel,
    MidiFrameModelConfig,
)
from instrument_agnostic_amt.beat_chord.cli.infer import (
    compute_key_js_divergence,
    decode_beats_with_meter_grid,
    decode_chord_segments,
    decode_key_class,
    decode_key_segments,
    enhance_key_boundary_probabilities,
    load_chord_quality_map,
    respell_chord_segments_with_romanizer,
    resolve_inference_window_settings,
    state_dict_has_major_grouping_head,
)
from instrument_agnostic_amt.beat_chord.midi_roll import (
    MidiFrameLoader,
    MidiFrameLoaderConfig,
)
from instrument_agnostic_amt.taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_instrument_class_id,
)


def test_inference_stride_defaults_to_half_of_checkpoint_window() -> None:
    window_ms, stride_ms = resolve_inference_window_settings(
        window_ms_override=None,
        stride_ms_override=None,
        checkpoint_args={"window_ms": 25_000},
    )

    assert window_ms == 25_000
    assert stride_ms == 12_500


def test_inference_stride_tracks_window_override_and_allows_explicit_override() -> None:
    assert resolve_inference_window_settings(
        window_ms_override=12_000,
        stride_ms_override=None,
        checkpoint_args={"window_ms": 25_000},
    ) == (12_000, 6_000)
    assert resolve_inference_window_settings(
        window_ms_override=12_000,
        stride_ms_override=4_000,
        checkpoint_args={"window_ms": 25_000},
    ) == (12_000, 4_000)


def test_inference_only_enables_grouping_for_new_checkpoint_state() -> None:
    assert not state_dict_has_major_grouping_head(
        {"beat_head.frame_proj.weight": torch.zeros(1)}
    )
    assert state_dict_has_major_grouping_head(
        {"module.beat_head.group_boundary_proj.weight": torch.zeros(1)}
    )


def test_inference_loads_embedded_chord_quality_map_without_dataset_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    model_config = MidiFrameModelConfig(
        sample_rate=100,
        hop_length=10,
        num_root_chord_classes=25,
    )

    quality_map = load_chord_quality_map(
        checkpoint={"chord_quality_map": {"0": "", "1": "m", "2": "N"}},
        model_config=model_config,
        quality_json=None,
    )

    assert quality_map == {"0": "", "1": "m", "2": "N"}


def test_grid_decoder_infers_a_missing_downbeat_as_multiple_bars() -> None:
    beat_probabilities = np.zeros(121, dtype=np.float32)
    beat_probabilities[::10] = 1.0
    meter_logits = np.zeros((121, 1), dtype=np.float32)

    beats, meters = decode_beats_with_meter_grid(
        beat_probabilities=beat_probabilities,
        downbeat_frames=[0, 40, 120],
        meter_logits=meter_logits,
        meter_classes=[(4, 4)],
        tolerance_frames=1,
        meter_score_weight=1.0,
        beat_grid_score_weight=1.0,
    )

    assert [segment.bar_count for segment in meters] == [1, 2]
    assert [segment.mapped_downbeat_frames for segment in meters] == [(0,), (40, 80)]
    assert beats == list(range(0, 120, 10))


def test_chord_decoder_uses_boundaries_and_adds_inversion_bass() -> None:
    quality_map = {"0": "", "1": "m", "2": "N"}
    chord_logits = np.zeros((12, 25), dtype=np.float32)
    chord_logits[:6, 0] = 4.0
    chord_logits[2, 14] = 8.0  # One noisy frame must not create a transition.
    chord_logits[6:, 14] = 4.0
    bass_logits = np.zeros((12, 13), dtype=np.float32)
    bass_logits[:6, 4] = 3.0
    bass_logits[6:, 7] = 3.0
    boundaries = np.zeros(12, dtype=np.float32)
    boundaries[3] = 0.49
    boundaries[6] = 0.9
    boundaries[7] = 0.9  # A flat-topped boundary still decodes once at its start.

    segments = decode_chord_segments(
        chord_logits=chord_logits,
        bass_logits=bass_logits,
        boundary_probabilities=boundaries,
        quality_map=quality_map,
        seconds_per_frame=0.1,
        duration_seconds=1.2,
        boundary_threshold=0.5,
        minimum_boundary_distance_frames=2,
    )

    assert segments == [
        {
            "start": 0.0,
            "end": 0.6,
            "chord": "C/E",
            "root_chord": "C",
            "bass": "E",
        },
        {
            "start": 0.6,
            "end": 1.2,
            "chord": "G",
            "root_chord": "G",
            "bass": "G",
        },
    ]


def test_key_decoder_uses_key_boundaries_and_merges_equal_regions() -> None:
    key_logits = np.zeros((10, 13), dtype=np.float32)
    key_logits[:7, 0] = 3.0
    key_logits[7:, 3] = 3.0
    boundaries = np.zeros(10, dtype=np.float32)
    boundaries[3] = 0.8
    boundaries[7] = 0.9

    segments = decode_key_segments(
        key_logits=key_logits,
        boundary_probabilities=boundaries,
        seconds_per_frame=0.2,
        duration_seconds=2.0,
        boundary_threshold=0.5,
        minimum_boundary_distance_frames=2,
    )

    assert segments == [
        {"start": 0.0, "end": 1.4, "key": "C"},
        {"start": 1.4, "end": 2.0, "key": "Eb"},
    ]


def test_key_js_divergence_recovers_a_missed_boundary() -> None:
    frame_count = 200
    key_logits = np.full((frame_count, 13), -10.0, dtype=np.float32)
    key_logits[:100, 0] = 10.0
    key_logits[100:, 7] = 10.0
    weak_boundaries = np.full(frame_count, 0.2, dtype=np.float32)

    segments = decode_key_segments(
        key_logits=key_logits,
        boundary_probabilities=weak_boundaries,
        seconds_per_frame=0.1,
        duration_seconds=20.0,
        boundary_threshold=0.5,
        minimum_boundary_distance_frames=5,
    )

    assert segments == [
        {"start": 0.0, "end": 10.0, "key": "C"},
        {"start": 10.0, "end": 20.0, "key": "G"},
    ]


def test_key_js_divergence_does_not_create_a_boundary_without_key_change() -> None:
    key_logits = np.full((200, 13), -10.0, dtype=np.float32)
    key_logits[:, 0] = 10.0
    weak_boundaries = np.full(200, 0.2, dtype=np.float32)

    divergence = compute_key_js_divergence(
        key_logits,
        seconds_per_frame=0.1,
    )
    enhanced = enhance_key_boundary_probabilities(
        key_logits=key_logits,
        boundary_probabilities=weak_boundaries,
        seconds_per_frame=0.1,
    )

    assert float(divergence.max()) < 1e-6
    assert np.allclose(enhanced, weak_boundaries)


def test_key_decoder_prefers_db_and_gb_major_names() -> None:
    assert decode_key_class(1) == "Db"
    assert decode_key_class(6) == "Gb"


def test_optional_chord_romanizer_respects_the_predicted_key(monkeypatch) -> None:
    class FakeResult:
        def __init__(
            self,
            event_index: int,
            symbol: str,
            combined_label: str,
        ) -> None:
            self.event_index = event_index
            self.symbol = symbol
            self.combined_label = combined_label

    class FakeRomanizer:
        options: list[tuple[str, bool]] = []

        @classmethod
        def strict(
            cls,
            *,
            default_tonic: str,
            simplify_accidentals: bool,
        ):
            cls.options.append((default_tonic, simplify_accidentals))
            return cls()

        def display_progression(self, symbols: list[str]) -> list[FakeResult]:
            replacements = {"D#:M7/F": "Eb:M7/F"}
            functions = {
                "D#:M7/F": "Eb:M7/F [IVM7|PD]",
                "F:7": "F:7 [V7|D]",
            }
            return [
                FakeResult(
                    event_index,
                    replacements.get(symbol, symbol),
                    functions.get(symbol, symbol),
                )
                for event_index, symbol in enumerate(symbols)
            ]

    fake_module = types.ModuleType("chord_romanizer")
    fake_module.Romanizer = FakeRomanizer
    monkeypatch.setitem(sys.modules, "chord_romanizer", fake_module)
    chords = [
        {
            "start": 0.0,
            "end": 1.0,
            "chord": "D#:M7/F",
            "root_chord": "D#:M7",
            "bass": "F",
        },
        {
            "start": 1.0,
            "end": 2.0,
            "chord": "F:7",
            "root_chord": "F:7",
            "bass": "F",
        },
    ]

    fixed, status = respell_chord_segments_with_romanizer(
        chords,
        [{"start": 0.0, "end": 2.0, "key": "Bb"}],
    )

    assert status == "applied"
    assert FakeRomanizer.options == [("Bb", True)]
    assert fixed[0]["chord"] == "Eb:M7/F"
    assert fixed[0]["combined_label"] == "Eb:M7/F"
    assert fixed[0]["root_chord"] == "Eb:M7"
    assert fixed[0]["bass"] == "F"
    assert fixed[1]["chord"] == "F:7"
    assert fixed[1]["combined_label"] == "F:7"


def test_missing_chord_romanizer_keeps_original_symbols(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "chord_romanizer", None)
    chords = [
        {
            "start": 0.0,
            "end": 1.0,
            "chord": "D#:M7",
            "root_chord": "D#:M7",
            "bass": "D#",
        }
    ]

    fixed, status = respell_chord_segments_with_romanizer(
        chords,
        [{"start": 0.0, "end": 1.0, "key": "Bb"}],
    )

    assert status == "unavailable"
    assert fixed == chords


def test_midi_frame_beat_chord_forward_shapes() -> None:
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
    model = MidiFrameBeatChordModel(config).eval()

    with torch.no_grad():
        outputs = model(
            torch.zeros(1, 4, 32, 12),
            include_beat=True,
            include_chord=True,
        )

    assert outputs["beat_logits"].shape == (1, 32)
    assert outputs["downbeat_logits"].shape == (1, 32)
    assert outputs["meter_logits"].shape == (1, 32, 3)
    assert outputs["chord_boundary_logits"].shape == (1, 32)
    assert outputs["root_chord_logits"].shape == (1, 32, 25)
    assert outputs["bass_logits"].shape == (1, 32, 13)
    assert outputs["key_logits"].shape == (1, 32, 13)
    assert outputs["chord_pitch_logits"].shape == (1, 32, 25)


def test_midi_frame_beat_chord_can_omit_auxiliary_outputs() -> None:
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
        inter_refine_layers=(0,),
    )
    model = MidiFrameBeatChordModel(config).eval()
    midi_frames = torch.zeros(1, 4, 32, 12)

    with torch.inference_mode():
        full = model(
            midi_frames,
            include_beat=True,
            include_chord=True,
        )
        minimal = model(
            midi_frames,
            include_beat=True,
            include_chord=True,
            include_aux_outputs=False,
        )

    auxiliary_keys = {
        "global_features",
        "lowres_global_features",
        "global_token_features",
        "lowres_global_tokens",
        "beat_features",
        "chord_features",
        "intermediate_outputs",
    }
    assert auxiliary_keys <= set(full)
    assert set(minimal) == set(full) - auxiliary_keys
    for key, value in minimal.items():
        assert isinstance(value, torch.Tensor)
        assert torch.equal(value, full[key])


def test_midi_frame_loader_builds_sustain_and_onset_channels(
    tmp_path: Path,
) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0, name="Acoustic Grand Piano")
    instrument.notes.append(
        pretty_midi.Note(velocity=100, pitch=60, start=0.1, end=0.4)
    )
    midi.instruments.append(instrument)
    midi_path = tmp_path / "song.mid"
    midi.write(str(midi_path))

    class_count = len(INSTRUMENT_CLASSES)
    class_id = get_instrument_class_id(0)
    loader = MidiFrameLoader(
        MidiFrameLoaderConfig(
            midi_dir=tmp_path,
            sample_rate=100,
            hop_length=10,
            pitch_min=60,
            pitch_max=60,
            num_channels=class_count * 2,
        )
    )

    roll = loader.load_window(
        song_name="song",
        window_start_sec=0.0,
        num_frames=10,
    )

    assert roll.shape == (class_count * 2, 10, 1)
    assert torch.equal(
        roll[class_id, :, 0],
        torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32),
    )
    assert roll[class_id + class_count, 1, 0] == 1.0
    assert roll[class_id + class_count, :, 0].sum() == 1.0
