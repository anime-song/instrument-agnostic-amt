from __future__ import annotations

import csv
import json
import random
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace

import mido
import numpy as np
import pretty_midi
import pytest
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from instrument_agnostic_amt.instrument_refinement.inference import refine as refine_module
from instrument_agnostic_amt.instrument_refinement.cli.infer import (
    DEFAULT_REFINEMENT_CHECKPOINT_FILENAME,
    ensure_refinement_checkpoint,
    parse_args as parse_refinement_args,
)
from instrument_agnostic_amt.instrument_refinement.data.labels import (
    inference_stem_group,
    refinement_stem_extra_class_ids,
)
from instrument_agnostic_amt.instrument_refinement.data.midi import (
    load_refinement_note_table,
)
from instrument_agnostic_amt.instrument_refinement.inference.aggregation import (
    cluster_note_embeddings,
    viterbi_smooth_classes,
)
from instrument_agnostic_amt.instrument_refinement.inference.refine import (
    refine_midi_instruments,
)
from instrument_agnostic_amt.instrument_refinement.modeling.model import (
    InstrumentRefinementConfig,
    InstrumentRefinementModel,
)
from recipes.instrument_refinement.collate import collate_refinement_batch
from recipes.instrument_refinement.dataset import (
    ClassBalancedSampler,
    ManifestInstrumentRefinementDataset,
    _manifest_path,
)
from recipes.instrument_refinement.forward import forward_refinement_batch
from recipes.instrument_refinement.losses import compute_refinement_losses
from recipes.instrument_refinement.manifest import build_refinement_manifest
from instrument_agnostic_amt.taxonomy.instrument_classes import (
    INSTRUMENT_CLASSES,
    get_instrument_class_id_by_name,
)


class _FakeBackbone(nn.Module):
    query_feature_dim = 12

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv1d(2, 88 * self.query_feature_dim, 5, stride=10, padding=2)

    def forward(self, audio: torch.Tensor) -> SimpleNamespace:
        values = self.projection(audio).transpose(1, 2)
        pitch = values.reshape(audio.shape[0], values.shape[1], 88, self.query_feature_dim)
        return SimpleNamespace(
            pitch_query_features=pitch,
            band_features=pitch[:, :, :8].permute(0, 2, 1, 3).contiguous(),
        )


def _write_audio(path: Path, *, frequency: float, duration: float = 1.0) -> None:
    sample_rate = 8_000
    time = np.arange(int(duration * sample_rate), dtype=np.float32) / sample_rate
    mono = 0.1 * np.sin(2.0 * np.pi * frequency * time)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.column_stack((mono, mono)), sample_rate, subtype="FLOAT")


def _write_midi(
    path: Path,
    pitches: tuple[int, ...] = (60, 67),
    *,
    program: int = 25,
    name: str = "AMT guitar",
) -> None:
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program, name=name)
    for index, pitch in enumerate(pitches):
        start = 0.1 + index * 0.4
        instrument.notes.append(pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=start + 0.25))
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def test_invalid_time_signature_keeps_midi_note_targets(tmp_path: Path) -> None:
    midi_path = tmp_path / "invalid_time_signature.mid"
    _write_midi(midi_path, pitches=(60, 67), program=25)
    midi_data = mido.MidiFile(str(midi_path))
    midi_data.tracks[0].insert(
        0,
        mido.MetaMessage("time_signature", numerator=0, denominator=4, time=0),
    )
    midi_data.save(str(midi_path))

    with pytest.warns(RuntimeWarning, match="invalid time signature"):
        notes = load_refinement_note_table(midi_path)

    assert notes.note_count == 2
    assert notes.pitch.tolist() == [60, 67]


def test_oversized_end_tick_keeps_midi_note_targets(tmp_path: Path) -> None:
    midi_path = tmp_path / "oversized_end_tick.mid"
    _write_midi(midi_path, pitches=(60, 67), program=25)
    midi_data = mido.MidiFile(str(midi_path))
    end_of_track = midi_data.tracks[-1][-1]
    assert end_of_track.type == "end_of_track"
    midi_data.tracks[-1][-1] = end_of_track.copy(time=20_014_081)
    midi_data.save(str(midi_path))

    with pytest.warns(RuntimeWarning, match="oversized end-of-track"):
        notes = load_refinement_note_table(midi_path)

    assert notes.note_count == 2
    assert notes.pitch.tolist() == [60, 67]


def test_unrepairable_midi_returns_empty_note_targets(tmp_path: Path) -> None:
    midi_path = tmp_path / "unrepairable_tick.mid"
    _write_midi(midi_path, pitches=(60,), program=25)
    midi_data = mido.MidiFile(str(midi_path))
    for message_index, message in enumerate(midi_data.tracks[-1]):
        if message.type == "note_on" and message.velocity > 0:
            midi_data.tracks[-1][message_index] = message.copy(time=20_014_081)
            break
    midi_data.save(str(midi_path))

    with pytest.warns(RuntimeWarning, match="Skipping unreadable MIDI"):
        notes = load_refinement_note_table(midi_path)

    assert notes.note_count == 0


def test_deployment_stem_groups_and_extra_candidates() -> None:
    assert inference_stem_group("other_keys") == "other"
    assert inference_stem_group("bowed_strings") == "other"
    assert inference_stem_group("wind") == "other"
    assert inference_stem_group("percussion") == "other"
    assert get_instrument_class_id_by_name("acoustic_guitar") in refinement_stem_extra_class_ids("other")
    assert refinement_stem_extra_class_ids("vocals") == ()


def test_windows_manifest_paths_are_portable_to_wsl() -> None:
    converted = _manifest_path(r"D:\datasets\stems\audio.wav", platform_name="posix")
    assert converted == Path("/mnt/d/datasets/stems/audio.wav")


def _add_single_stem(
    root: Path,
    *,
    class_name: str,
    song: str,
    instrument: str,
    program: int,
    frequency: float,
    pitches: tuple[int, ...] = (60, 67),
) -> None:
    base = root / "class_sources"
    stem = f"{song}__{instrument}"
    audio = base / f"{class_name}_stem" / f"{stem}.wav"
    midi = base / f"{class_name}_stem_midi" / f"{stem}.mid"
    _write_audio(audio, frequency=frequency)
    _write_midi(midi, pitches, program=program, name=instrument)
    manifest = base / "manifests" / f"{class_name}_stem_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not manifest.exists()
    with manifest.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "song_name",
                "stem_name",
                "wav_path",
                "npz_path",
                "duration_ms",
                "end_note_ms",
                "note_count",
                "sample_rate",
            ),
        )
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "song_name": song,
                "stem_name": stem,
                "wav_path": f"../{class_name}_stem/{stem}.wav",
                "npz_path": "",
                "duration_ms": 1000,
                "end_note_ms": 900,
                "note_count": len(pitches),
                "sample_rate": 8000,
            }
        )


def _write_dataset_config(path: Path, *, output: Path, datasets: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "output": str(output),
                "seed": 42,
                "train_fraction": 0.8,
                "validation_fraction": 0.1,
                "datasets": datasets,
            }
        ),
        encoding="utf-8",
    )


def _class_collection_spec(root: Path) -> dict[str, object]:
    return {
        "name": "collection_a",
        "type": "class_manifests",
        "root": str(root / "class_sources"),
        "manifest_glob": "manifests/*_stem_manifest.csv",
        "class_name_regex": r"(?P<class>.+)_stem_manifest",
        "midi_dir_template": "{class_name}_stem_midi",
        "song_key_mode": "prefix_before_last_double_underscore",
    }


def test_configured_manifest_and_class_balanced_dataset(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _add_single_stem(
        root,
        class_name="acoustic_guitar",
        song="song-a",
        instrument="acoustic_guitar",
        program=25,
        frequency=330.0,
    )
    _add_single_stem(
        root,
        class_name="strings",
        song="song-b",
        instrument="strings",
        program=48,
        frequency=220.0,
    )
    manifest = tmp_path / "manifest.csv"
    config = tmp_path / "datasets.local.json"
    _write_dataset_config(config, output=manifest, datasets=[_class_collection_spec(root)])
    summary = build_refinement_manifest(config)

    assert summary["source_count"] == 2
    with manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["target_class_names"] for row in rows} == {
        "acoustic_guitar",
        "strings",
    }

    dataset = ManifestInstrumentRefinementDataset(
        manifest,
        split="train",
        sample_rate=100,
        window_seconds=0.5,
        sampling="random",
        windows_per_source=2,
        random_gain_db=0.0,
    )
    item = dataset[0]
    batch = collate_refinement_batch([item])
    assert batch["audio"].shape == (2, 2, 50)
    assert batch["source_hash"][0] == batch["source_hash"][1]
    assert batch["view_group"].tolist() == [0, 0]
    assert batch["class_target_mask"].sum(dim=1).tolist() == [1, 1]
    sampler = ClassBalancedSampler(dataset.class_to_unit_indices, num_samples=200, seed=7)
    sampled_classes = [dataset.sources[dataset.units[index][0]].target_class_ids[0] for index in sampler]
    counts = {class_id: sampled_classes.count(class_id) for class_id in set(sampled_classes)}
    assert len(counts) == 2
    assert max(counts.values()) - min(counts.values()) < 40


def _augmentable_dataset(tmp_path: Path, **dataset_kwargs: object) -> ManifestInstrumentRefinementDataset:
    """AudioAugmentor が実際に通るサンプルレートで学習用 dataset を組む。

    他のテストは sample_rate=100 で足りるが、augmentor の EQ は数 kHz の
    フィルタを積むのでナイキストがそこまで届いていないと組み立てで落ちる。
    """
    root = tmp_path / "datasets"
    _add_single_stem(
        root,
        class_name="acoustic_guitar",
        song="song-a",
        instrument="acoustic_guitar",
        program=25,
        frequency=330.0,
    )
    manifest = tmp_path / "manifest.csv"
    config = tmp_path / "datasets.local.json"
    _write_dataset_config(config, output=manifest, datasets=[_class_collection_spec(root)])
    build_refinement_manifest(config)
    return ManifestInstrumentRefinementDataset(
        manifest,
        split="train",
        sample_rate=22_050,
        window_seconds=0.5,
        sampling="random",
        windows_per_source=1,
        random_gain_db=0.0,
        composite_probability=0.0,
        **dataset_kwargs,
    )


def test_augmentation_is_skipped_entirely_when_probability_is_zero(tmp_path: Path) -> None:
    """既定では augmentor を作らず、波形も素のままにする。"""
    dataset = _augmentable_dataset(tmp_path, augment_probability=0.0)

    assert dataset._get_augmentor() is None

    random.seed(0)
    first = dataset[0]["views"][0]["audio"]
    random.seed(0)
    second = dataset[0]["views"][0]["audio"]
    assert torch.equal(first, second)


def test_augmentation_changes_the_waveform_but_keeps_shape_and_labels(tmp_path: Path) -> None:
    """拡張後も窓長・チャンネル数・ノート正解は変わらない。

    窓長が変わるとバッチが組めなくなり、ノート正解がずれると
    refine の学習そのものが壊れるので、そこを固定する。
    """
    plain = _augmentable_dataset(tmp_path / "plain", augment_probability=0.0)
    augmented = _augmentable_dataset(tmp_path / "aug", augment_probability=1.0)

    random.seed(11)
    np.random.seed(11)
    plain_view = plain[0]["views"][0]
    random.seed(11)
    np.random.seed(11)
    augmented_view = augmented[0]["views"][0]

    assert augmented_view["audio"].shape == plain_view["audio"].shape
    assert augmented_view["audio"].dtype == plain_view["audio"].dtype
    assert not torch.allclose(augmented_view["audio"], plain_view["audio"])
    assert torch.equal(augmented_view["note_pitch"], plain_view["note_pitch"])
    assert torch.equal(augmented_view["note_start_seconds"], plain_view["note_start_seconds"])
    assert torch.equal(augmented_view["note_class_target_mask"], plain_view["note_class_target_mask"])
    assert augmented_view["valid_audio_frames"] == plain_view["valid_audio_frames"]


def test_augmentation_runs_per_source_before_mixing(tmp_path: Path) -> None:
    """混合サンプルでは音源ごとに別々に掛かる。

    混ぜた後に 1 回だけ掛けると、全音源が同じ EQ と同じ部屋を通ったことになり、
    別々に録られた音が同居している状況を学べない。
    """
    root = tmp_path / "datasets"
    _add_single_stem(
        root,
        class_name="acoustic_guitar",
        song="song-a",
        instrument="acoustic_guitar",
        program=25,
        frequency=330.0,
    )
    _add_single_stem(
        root,
        class_name="distorted_guitar",
        song="song-b",
        instrument="distorted_guitar",
        program=30,
        frequency=220.0,
    )
    manifest = tmp_path / "manifest.csv"
    config = tmp_path / "datasets.local.json"
    _write_dataset_config(config, output=manifest, datasets=[_class_collection_spec(root)])
    build_refinement_manifest(config)
    dataset = ManifestInstrumentRefinementDataset(
        manifest,
        split="train",
        sample_rate=100,
        window_seconds=0.5,
        sampling="random",
        windows_per_source=1,
        random_gain_db=0.0,
        composite_probability=1.0,
        cross_song_composite_ratio=1.0,
        augment_probability=1.0,
    )

    calls: list[tuple[int, ...]] = []

    def _record(audio: np.ndarray) -> np.ndarray:
        calls.append(tuple(audio.shape))
        return audio * 0.5

    dataset._augmentor = _record

    random.seed(5)
    for _ in range(20):
        view = dataset[0]["views"][0]
        if view["is_composite"]:
            break
    else:
        pytest.fail("composite サンプルが引けなかった")

    assert len(calls) >= 2
    assert all(shape == (2, 50) for shape in calls)


def test_augment_probability_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="augment_probability"):
        _augmentable_dataset(tmp_path, augment_probability=1.5)


def test_augmentor_is_not_sent_to_dataloader_workers(tmp_path: Path) -> None:
    """augmentor は pickle できないので、ワーカーへは渡さず各自で作らせる。"""
    dataset = _augmentable_dataset(tmp_path, augment_probability=1.0)
    dataset._augmentor = object()

    assert dataset.__getstate__()["_augmentor"] is None


def test_cross_song_composites_mix_instruments_that_never_share_a_song(tmp_path: Path) -> None:
    """曲をまたいだ混合で、同一曲には存在しない楽器の組み合わせを学習できる。

    実データのギターがこの状況で、アコギと歪みギターが同居する曲は 1 曲も無い。
    """
    root = tmp_path / "datasets"
    _add_single_stem(
        root,
        class_name="acoustic_guitar",
        song="song-a",
        instrument="acoustic_guitar",
        program=25,
        frequency=330.0,
    )
    _add_single_stem(
        root,
        class_name="distorted_guitar",
        song="song-b",
        instrument="distorted_guitar",
        program=30,
        frequency=220.0,
    )
    manifest = tmp_path / "manifest.csv"
    config = tmp_path / "datasets.local.json"
    _write_dataset_config(config, output=manifest, datasets=[_class_collection_spec(root)])
    build_refinement_manifest(config)

    def build_dataset(cross_song_ratio: float) -> ManifestInstrumentRefinementDataset:
        return ManifestInstrumentRefinementDataset(
            manifest,
            split="train",
            sample_rate=100,
            window_seconds=0.5,
            sampling="random",
            windows_per_source=1,
            random_gain_db=0.0,
            composite_probability=1.0,
            cross_song_composite_ratio=cross_song_ratio,
        )

    # 従来どおり同一曲だけで混ぜる場合、相手がいないので単独サンプルにしかならない。
    same_song_only = build_dataset(0.0)
    for index in range(len(same_song_only)):
        for view in same_song_only[index]["views"]:
            assert view["is_composite"] is False
            assert int(view["class_target_mask"].sum()) == 1

    # 曲をまたげば、2 種類のギターが同時に鳴るサンプルになる。
    random.seed(0)
    cross_song = build_dataset(1.0)
    for _ in range(5):
        for index in range(len(cross_song)):
            for view in cross_song[index]["views"]:
                assert view["is_composite"] is True
                assert int(view["class_target_mask"].sum()) == 2
                # 窓の楽器は一意に決まらないので、窓分類 loss からは外れる。
                assert view["window_sample_weight"] == 0.0


def test_refinement_model_loss_and_backward() -> None:
    config = InstrumentRefinementConfig(
        sample_rate=100,
        hop_length=10,
        hidden_size=24,
        base_ch=8,
        encoder_num_layers=0,
        encoder_num_heads=3,
        note_hidden_size=32,
        embedding_size=16,
        dropout=0.0,
        use_gradient_checkpoint=False,
        onset_frame_offsets=(-1, 0, 1),
        sustain_fractions=(0.5,),
        release_frame_offsets=(0,),
    )
    model = InstrumentRefinementModel(config, backbone=_FakeBackbone())
    acoustic_id = get_instrument_class_id_by_name("acoustic_guitar")
    class_target = torch.zeros(2, len(INSTRUMENT_CLASSES), dtype=torch.bool)
    class_target[:, acoustic_id] = True
    family_target = torch.zeros(2, config.num_family_classes, dtype=torch.bool)
    family_target[:, 3] = True
    batch = {
        "audio": torch.randn(2, 2, 100),
        "valid_audio_frames": torch.tensor([100, 80]),
        "note_start_seconds": torch.tensor([[0.1, 0.5], [0.2, 0.0]]),
        "note_end_seconds": torch.tensor([[0.3, 0.8], [0.5, 0.0]]),
        "note_pitch": torch.tensor([[60, 67], [64, 0]]),
        "note_prior_class": torch.tensor([[acoustic_id, acoustic_id], [acoustic_id, -1]]),
        "note_confidence": torch.ones(2, 2),
        "note_mask": torch.tensor([[True, True], [True, False]]),
        "stem_context_id": torch.tensor([2, 2]),
        "class_target_mask": class_target,
        "family_target_mask": family_target,
        "primary_class_id": torch.tensor([acoustic_id, acoustic_id]),
        "sample_weight": torch.ones(2),
        "source_hash": torch.tensor([7, 7]),
    }
    model.eval()
    first = forward_refinement_batch(model, batch, window_seconds=1.0)
    changed_prior = dict(batch)
    changed_prior["note_prior_class"] = torch.zeros_like(batch["note_prior_class"])
    second = forward_refinement_batch(model, changed_prior, window_seconds=1.0)
    assert torch.allclose(first["note_logits"], second["note_logits"])

    model.train()
    outputs = forward_refinement_batch(model, batch, window_seconds=1.0)
    assert outputs["note_logits"].shape == (2, 2, len(INSTRUMENT_CLASSES))
    assert outputs["window_embedding"].shape == (2, 16)
    loss, metrics = compute_refinement_losses(outputs, batch)
    assert torch.isfinite(loss)
    assert int(metrics["note_count"]) == 3
    loss.backward()
    assert model.instrument_head.weight.grad is not None
    assert model.timbre_projection[-1].weight.grad is not None


def test_composite_training_view_keeps_per_source_note_targets(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    _add_single_stem(
        root,
        class_name="acoustic_guitar",
        song="song-id",
        instrument="acoustic",
        program=25,
        frequency=330.0,
    )
    _add_single_stem(
        root,
        class_name="electric_guitar_clean",
        song="song-id",
        instrument="clean",
        program=27,
        frequency=440.0,
    )
    manifest = tmp_path / "composite_manifest.csv"
    config = tmp_path / "composite.local.json"
    _write_dataset_config(config, output=manifest, datasets=[_class_collection_spec(root)])
    build_refinement_manifest(config)
    dataset = ManifestInstrumentRefinementDataset(
        manifest,
        split="train",
        sample_rate=100,
        window_seconds=1.0,
        sampling="random",
        windows_per_source=2,
        random_gain_db=0.0,
        composite_probability=1.0,
    )
    batch = collate_refinement_batch([dataset[0]])

    acoustic = get_instrument_class_id_by_name("acoustic_guitar")
    clean = get_instrument_class_id_by_name("electric_guitar_clean")
    assert batch["is_composite"].tolist() == [True, True]
    assert batch["window_sample_weight"].tolist() == [0.0, 0.0]
    active_targets = batch["note_class_target_mask"][batch["note_mask"]]
    assert active_targets[:, acoustic].any()
    assert active_targets[:, clean].any()
    assert torch.all(active_targets[:, acoustic] & active_targets[:, clean])
    assert batch["note_mask"].sum(dim=1).tolist() == [2, 2]
    assert not batch["note_contrastive_mask"][batch["note_mask"]].any()
    assert torch.all(batch["note_prior_class"][batch["note_mask"]] == -1)


def test_other_group_can_mix_one_to_n_configured_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "datasets"
    definitions = (
        ("strings", "strings", 48, (60, 64), 220.0),
        ("flute_pipe", "flute", 73, (62, 65), 330.0),
        ("synth_pad", "pad", 88, (67, 71), 440.0),
    )
    for class_name, instrument, program, pitches, frequency in definitions:
        _add_single_stem(
            root,
            class_name=class_name,
            song="other-song",
            instrument=instrument,
            program=program,
            frequency=frequency,
            pitches=pitches,
        )
    manifest = tmp_path / "other_manifest.csv"
    config = tmp_path / "other.local.json"
    _write_dataset_config(config, output=manifest, datasets=[_class_collection_spec(root)])
    build_refinement_manifest(config)
    dataset = ManifestInstrumentRefinementDataset(
        manifest,
        split="train",
        sample_rate=100,
        window_seconds=1.0,
        sampling="random",
        windows_per_source=2,
        random_gain_db=0.0,
        composite_probability=1.0,
    )
    monkeypatch.setattr(
        "recipes.instrument_refinement.dataset.random.randint",
        lambda _minimum, maximum: maximum,
    )
    batch = collate_refinement_batch([dataset[0]])

    assert batch["is_composite"].tolist() == [True, True]
    assert batch["stem_name"] == ["other", "other"]
    assert all(len(source_id.split("+")) == 3 for source_id in batch["source_id"])
    assert batch["class_target_mask"].sum(dim=1).tolist() == [3, 3]
    assert torch.all(batch["note_prior_class"][batch["note_mask"]] == -1)


def test_flat_collection_limits_augmentation_variants_per_group(
    tmp_path: Path,
) -> None:
    root = tmp_path / "datasets"
    stems = (
        "take",
        "take_pitch_-2",
        "take_pitch_2",
        "take_stretch_0.9",
        "take_stretch_1.1",
    )
    for index, stem in enumerate(stems):
        _write_audio(root / "audio" / f"{stem}.wav", frequency=220.0 + 20.0 * index)
        _write_midi(root / "midi" / f"{stem}.mid", program=25)
    manifest = tmp_path / "limited_manifest.csv"
    config = tmp_path / "limited.local.json"
    _write_dataset_config(
        config,
        output=manifest,
        datasets=[
            {
                "name": "collection_c",
                "type": "flat",
                "root": str(root),
                "audio_glob": "audio/*.wav",
                "midi_dir": "midi",
                "class_source": "midi",
                "max_augmentation_variants_per_group": 2,
                "augmentation_selection_seed": 7,
            }
        ],
    )

    summary = build_refinement_manifest(config)
    with manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert summary["source_count"] == 3
    assert summary["skipped"]["collection_c"]["augmentation_variant_limit"] == 2
    assert len({row["split_group_id"] for row in rows}) == 1
    assert any(Path(row["audio_path"]).stem == "take" for row in rows)


def test_flat_collection_can_reject_empty_midi_targets(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    class_id = get_instrument_class_id_by_name("acoustic_guitar")
    for stem in ("has_notes", "empty"):
        _write_audio(root / "audio" / f"{stem}.wav", frequency=220.0)
        npz_path = root / "npz" / f"{stem}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz_path, note_instrument=np.asarray([class_id], dtype=np.int64))
    _write_midi(root / "midi" / "has_notes.mid", program=25)
    empty_midi = pretty_midi.PrettyMIDI()
    (root / "midi").mkdir(parents=True, exist_ok=True)
    empty_midi.write(str(root / "midi" / "empty.mid"))
    manifest = tmp_path / "nonempty_manifest.csv"
    config = tmp_path / "nonempty.local.json"
    _write_dataset_config(
        config,
        output=manifest,
        datasets=[
            {
                "name": "collection_d",
                "type": "flat",
                "root": str(root),
                "audio_glob": "audio/*.wav",
                "midi_dir": "midi",
                "npz_dir": "npz",
                "class_source": "npz",
                "require_midi_notes": True,
            }
        ],
    )

    summary = build_refinement_manifest(config)

    assert summary["source_count"] == 1
    assert summary["skipped"]["collection_d"]["no_midi_notes"] == 1


def test_vocal_and_backing_vocal_are_an_atomic_unit(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    for stem, name, pitches, frequency in (
        ("Test Song", "vocal", (60, 62), 330.0),
        ("Test Song__back", "vocal_harmony", (67, 69), 440.0),
    ):
        _write_audio(root / "voice_audio" / f"{stem}.wav", frequency=frequency)
        _write_midi(
            root / "voice_midi" / f"{stem}.mid",
            pitches,
            program=73,
            name=name,
        )
    manifest = tmp_path / "vocal_manifest.csv"
    config = tmp_path / "vocal.local.json"
    _write_dataset_config(
        config,
        output=manifest,
        datasets=[
            {
                "name": "collection_b",
                "type": "flat",
                "root": str(root),
                "audio_glob": "voice_audio/*.wav",
                "midi_dir": "voice_midi",
                "class_source": "midi",
                "allowed_classes": ["melody", "vocal_harmony"],
                "atomic_mode": "song_variant",
                "song_key_rules": [{"pattern": r"__back$", "replacement": ""}],
                "normalize_nfkc": True,
                "alnum_only": True,
            }
        ],
    )
    summary = build_refinement_manifest(config)
    assert summary["multi_source_atomic_unit_count"] == 1
    dataset = ManifestInstrumentRefinementDataset(
        manifest,
        split="train",
        sample_rate=100,
        window_seconds=1.0,
        sampling="random",
        windows_per_source=1,
        random_gain_db=0.0,
        composite_probability=0.0,
    )
    assert len(dataset.units) == 1
    assert len(dataset.units[0]) == 2
    batch = collate_refinement_batch([dataset[0]])
    melody = get_instrument_class_id_by_name("melody")
    harmony = get_instrument_class_id_by_name("vocal_harmony")
    assert batch["is_composite"].tolist() == [True]
    assert len(batch["source_id"][0].split("+")) == 2
    assert batch["class_target_mask"][0, melody]
    assert batch["class_target_mask"][0, harmony]
    active_targets = batch["note_class_target_mask"][batch["note_mask"]]
    assert active_targets[:, melody].any()
    assert active_targets[:, harmony].any()


def test_temporal_smoothing_and_embedding_clustering() -> None:
    logits = np.asarray([[4.0, 0.0], [0.0, 4.0], [4.0, 0.0], [4.0, 0.0]], dtype=np.float32)
    smoothed = viterbi_smooth_classes(logits, switch_penalty=5.0)
    assert smoothed.tolist() == [0, 0, 0, 0]

    embeddings = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]],
        dtype=np.float32,
    )
    clusters = cluster_note_embeddings(embeddings, distance_threshold=0.1, min_cluster_notes=2)
    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[0] != clusters[2]


class _FakeRefinementModel:
    def __init__(
        self,
        config: InstrumentRefinementConfig,
        *,
        record_inputs: bool = True,
    ) -> None:
        self.config = config
        self.record_inputs = record_inputs
        self.batch_sizes: list[int] = []
        self.note_counts: list[list[int]] = []

    def to(self, _device: torch.device) -> "_FakeRefinementModel":
        return self

    def eval(self) -> "_FakeRefinementModel":
        return self

    def __call__(self, _audio: torch.Tensor, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        pitches = kwargs["note_pitch"]
        batch_size, note_count = pitches.shape
        self.batch_sizes.append(int(batch_size))
        device = pitches.device
        if self.record_inputs:
            assert torch.all(kwargs["note_prior_class"] == -1)
            self.note_counts.append(
                kwargs["note_mask"].sum(dim=1).detach().cpu().tolist()
            )
        acoustic = get_instrument_class_id_by_name("acoustic_guitar")
        distorted = get_instrument_class_id_by_name("distorted_guitar")
        note_logits = torch.zeros(batch_size, note_count, self.config.num_instrument_classes, device=device)
        note_embedding = torch.zeros(batch_size, note_count, self.config.embedding_size, device=device)
        if note_count:
            first = pitches < 64
            note_logits[..., acoustic] = first.float() * 5.0
            note_logits[..., distorted] = (~first).float() * 5.0
            note_embedding[..., 0] = first.float()
            note_embedding[..., 1] = (~first).float()
        window_logits = torch.zeros(batch_size, self.config.num_instrument_classes, device=device)
        window_logits[..., acoustic] = 3.0
        window_embedding = torch.zeros(batch_size, self.config.embedding_size, device=device)
        window_embedding[..., 0] = 1.0
        return {
            "note_logits": note_logits,
            "note_embedding": note_embedding,
            "window_logits": window_logits,
            "window_embedding": window_embedding,
        }


class _FailingRefinementModel(_FakeRefinementModel):
    def __call__(
        self,
        _audio: torch.Tensor,
        **_kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        raise RuntimeError("refinement forward failed")


class _OutputLifetimeRefinementModel(_FakeRefinementModel):
    """次の窓を流す前に前回のdevice出力が解放されたか記録する。"""

    def __init__(self, config: InstrumentRefinementConfig) -> None:
        super().__init__(config, record_inputs=False)
        self.previous_output_refs: list[weakref.ReferenceType[torch.Tensor]] = []
        self.released_before_next_forward: list[bool] = []

    def __call__(
        self,
        audio: torch.Tensor,
        **kwargs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.previous_output_refs:
            self.released_before_next_forward.append(
                all(reference() is None for reference in self.previous_output_refs)
            )
        outputs = super().__call__(audio, **kwargs)
        self.previous_output_refs = [
            weakref.ref(value) for value in outputs.values()
        ]
        return outputs


class _NonCpuScalarReadCounter(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    def __torch_dispatch__(
        self,
        func: object,
        types: object,
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        if "_local_scalar_dense" in str(func):
            self.count += sum(
                int(isinstance(value, torch.Tensor) and value.device.type != "cpu")
                for value in args
            )
        return func(*args, **({} if kwargs is None else kwargs))  # type: ignore[operator]


def _small_refinement_config() -> InstrumentRefinementConfig:
    return InstrumentRefinementConfig(
        sample_rate=100,
        hop_length=10,
        hidden_size=24,
        base_ch=8,
        encoder_num_layers=0,
        encoder_num_heads=3,
        note_hidden_size=32,
        embedding_size=8,
        use_gradient_checkpoint=False,
    )


def _run_refinement_window_case(
    tmp_path: Path,
    *,
    batch_size: int,
    pitches: tuple[int, ...] = (60, 67),
    stride_seconds: float = 0.25,
    model_type: type[_FakeRefinementModel] = _FakeRefinementModel,
) -> tuple[_FakeRefinementModel, dict[str, object]]:
    audio_path = tmp_path / "window_case.wav"
    midi_path = tmp_path / "window_case.mid"
    _write_audio(audio_path, frequency=330.0)
    _write_midi(midi_path, pitches=pitches)
    config = _small_refinement_config()
    model = model_type(config)
    report = refine_midi_instruments(
        audio_path,
        midi_path,
        stem_name="guitar",
        window_seconds=0.5,
        stride_seconds=stride_seconds,
        window_batch_size=batch_size,
        cluster_distance=0.1,
        min_cluster_notes=1,
        disable_tqdm=True,
        preloaded_model=model,  # type: ignore[arg-type]
        preloaded_config=config,
    )
    return model, report


def test_ensure_refinement_checkpoint_prefers_local_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit = tmp_path / "custom_refinement.pth"
    explicit.write_bytes(b"")
    assert ensure_refinement_checkpoint(explicit) == explicit.resolve()

    monkeypatch.chdir(tmp_path)
    default_path = tmp_path / "checkpoints" / DEFAULT_REFINEMENT_CHECKPOINT_FILENAME
    default_path.parent.mkdir(parents=True, exist_ok=True)
    default_path.write_bytes(b"")
    assert ensure_refinement_checkpoint(None) == default_path.resolve()
    assert ensure_refinement_checkpoint("DEFAULT") == default_path.resolve()


def test_refinement_cli_accepts_window_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "instrument-refinement",
            "--audio",
            "audio.wav",
            "--midi",
            "input.mid",
            "--window-batch-size",
            "3",
        ],
    )

    assert parse_refinement_args().window_batch_size == 3


def test_refinement_auto_routes_model_to_mps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.wav"
    midi_path = tmp_path / "input.mid"
    audio_path.write_bytes(b"audio")
    midi_path.write_bytes(b"midi")
    loaded_devices: list[str] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    def stop_after_device_resolution(
        *_args: object,
        device: torch.device,
        **_kwargs: object,
    ) -> tuple[object, object]:
        loaded_devices.append(str(device))
        raise RuntimeError("device captured")

    monkeypatch.setattr(refine_module, "_resolve_model", stop_after_device_resolution)

    with pytest.raises(RuntimeError, match="device captured"):
        refine_midi_instruments(audio_path, midi_path, device=None)

    assert loaded_devices == ["mps"]


def test_whole_stem_refinement_rewrites_midi(tmp_path: Path) -> None:
    audio_path = tmp_path / "guitar.wav"
    midi_path = tmp_path / "guitar.mid"
    output_path = tmp_path / "refined.mid"
    _write_audio(audio_path, frequency=330.0)
    _write_midi(midi_path, pitches=(60, 67))
    config = _small_refinement_config()
    report = refine_midi_instruments(
        audio_path,
        midi_path,
        output_midi_path=output_path,
        stem_name="guitar",
        window_seconds=0.5,
        stride_seconds=0.25,
        cluster_distance=0.1,
        min_cluster_notes=1,
        disable_tqdm=True,
        preloaded_model=_FakeRefinementModel(config),  # type: ignore[arg-type]
        preloaded_config=config,
    )

    assert output_path.is_file()
    assert report["note_count"] == 2
    assert report["cluster_count"] == 2
    output = pretty_midi.PrettyMIDI(str(output_path))
    assert {instrument.name for instrument in output.instruments} == {
        "acoustic_guitar",
        "distorted_guitar",
    }


def test_refinement_uses_preloaded_waveform_without_reloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "guitar.wav"
    midi_path = tmp_path / "guitar.mid"
    _write_audio(audio_path, frequency=330.0)
    _write_midi(midi_path, pitches=(60,))
    config = _small_refinement_config()
    waveform = refine_module.load_audio(
        audio_path,
        target_sample_rate=config.sample_rate,
    )
    expected_report = refine_midi_instruments(
        audio_path,
        midi_path,
        stem_name="guitar",
        mode="single",
        disable_tqdm=True,
        preloaded_model=_FakeRefinementModel(config),  # type: ignore[arg-type]
        preloaded_config=config,
    )
    monkeypatch.setattr(
        refine_module,
        "load_audio",
        lambda *_args, **_kwargs: pytest.fail("audio should not be reloaded"),
    )

    report = refine_midi_instruments(
        audio_path,
        midi_path,
        stem_name="guitar",
        mode="single",
        disable_tqdm=True,
        preloaded_model=_FakeRefinementModel(config),  # type: ignore[arg-type]
        preloaded_config=config,
        preloaded_waveform=waveform,
    )

    assert report == expected_report


def test_refinement_window_batch_matches_single_window_output(tmp_path: Path) -> None:
    _, batched_report = _run_refinement_window_case(tmp_path, batch_size=2)
    _, single_report = _run_refinement_window_case(tmp_path, batch_size=1)

    assert batched_report == single_report


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS上のRefinement出力転送と寿命を検査するテストです",
)
def test_refinement_bulk_transfers_and_releases_device_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "transfer.wav"
    midi_path = tmp_path / "transfer.mid"
    _write_audio(audio_path, frequency=330.0, duration=1.0)
    _write_midi(midi_path, pitches=(60,))
    config = _small_refinement_config()
    model = _OutputLifetimeRefinementModel(config)
    scalar_reads = _NonCpuScalarReadCounter()
    device_to_host_calls: list[tuple[int, ...]] = []
    original_cpu = torch.Tensor.cpu

    def record_device_to_host(
        tensor: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        if tensor.device.type != "cpu":
            device_to_host_calls.append(tuple(tensor.shape))
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", record_device_to_host)
    with scalar_reads:
        refine_midi_instruments(
            audio_path,
            midi_path,
            stem_name="guitar",
            device="mps",
            window_seconds=0.5,
            stride_seconds=0.5,
            window_batch_size=1,
            disable_tqdm=True,
            preloaded_model=model,  # type: ignore[arg-type]
            preloaded_config=config,
        )

    assert (
        len(device_to_host_calls),
        scalar_reads.count,
        model.released_before_next_forward,
    ) == (2, 0, [True])


def test_refinement_bulk_transfer_preserves_report_and_midi_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "exact.wav"
    midi_path = tmp_path / "exact.mid"
    output_midi_path = tmp_path / "refined.mid"
    _write_audio(audio_path, frequency=330.0, duration=0.5)
    _write_midi(midi_path, pitches=(60, 67))
    config = _small_refinement_config()

    def transfer_outputs_individually(
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        return tuple(tensor.cpu() for tensor in tensors)

    bulk_transfer = refine_module.copy_tensors_to_cpu_once

    def run_refinement() -> dict[str, object]:
        return refine_midi_instruments(
            audio_path,
            midi_path,
            output_midi_path=output_midi_path,
            stem_name="guitar",
            window_seconds=0.5,
            stride_seconds=0.5,
            cluster_distance=0.1,
            min_cluster_notes=1,
            disable_tqdm=True,
            preloaded_model=_FakeRefinementModel(config),  # type: ignore[arg-type]
            preloaded_config=config,
        )

    monkeypatch.setattr(
        refine_module,
        "copy_tensors_to_cpu_once",
        transfer_outputs_individually,
    )
    legacy_report = run_refinement()
    legacy_midi = output_midi_path.read_bytes()

    monkeypatch.setattr(
        refine_module,
        "copy_tensors_to_cpu_once",
        bulk_transfer,
    )
    bulk_report = run_refinement()

    assert (bulk_report, output_midi_path.read_bytes()) == (
        legacy_report,
        legacy_midi,
    )


def test_refinement_window_batch_keeps_the_final_partial_batch(tmp_path: Path) -> None:
    model, _ = _run_refinement_window_case(tmp_path, batch_size=2)

    assert model.batch_sizes == [2, 1]


def test_refinement_window_batch_masks_windows_without_notes(tmp_path: Path) -> None:
    model, _ = _run_refinement_window_case(
        tmp_path,
        batch_size=2,
        pitches=(60,),
        stride_seconds=0.5,
    )

    assert model.note_counts == [[1, 0]]


def test_refinement_window_batch_preserves_forward_failures(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="refinement forward failed"):
        _run_refinement_window_case(
            tmp_path,
            batch_size=2,
            model_type=_FailingRefinementModel,
        )


def _class_dataset_units() -> dict[int, dict[str, tuple[int, ...]]]:
    """1 クラスの中に、件数が桁違いのデータセットが 2 つある状況を作る。

    実データの orchestral_woodwind がこの形で、合成音（cocochorales / single_stems）が
    ほぼ全部を占め、実録音は 3.4% しか引かれない。
    """
    return {
        0: {
            "synthetic": tuple(range(0, 970)),
            "real": tuple(range(970, 1000)),
        }
    }


def test_dataset_weights_lift_small_datasets_out_of_the_majority(seed: int = 3) -> None:
    class_dataset_units = _class_dataset_units()
    class_to_units = {0: tuple(range(1000))}

    without = ClassBalancedSampler(class_to_units, num_samples=4000, seed=seed)
    with_weights = ClassBalancedSampler(
        class_to_units,
        num_samples=4000,
        seed=seed,
        class_dataset_unit_indices=class_dataset_units,
        dataset_weights={"synthetic": 1.0, "real": 1.0},
    )

    real_without = sum(1 for index in without if index >= 970) / 4000
    real_with = sum(1 for index in with_weights if index >= 970) / 4000

    # 件数任せだと 3%、データセットを均等に引けば 50% 付近になる。
    assert real_without < 0.06
    assert 0.44 < real_with < 0.56


def test_dataset_weights_are_proportional() -> None:
    sampler = ClassBalancedSampler(
        {0: tuple(range(1000))},
        num_samples=6000,
        seed=11,
        class_dataset_unit_indices=_class_dataset_units(),
        dataset_weights={"synthetic": 3.0, "real": 1.0},
    )

    real_share = sum(1 for index in sampler if index >= 970) / 6000

    assert 0.21 < real_share < 0.29


def test_zero_weight_removes_a_dataset() -> None:
    sampler = ClassBalancedSampler(
        {0: tuple(range(1000))},
        num_samples=500,
        seed=5,
        class_dataset_unit_indices=_class_dataset_units(),
        dataset_weights={"synthetic": 0.0, "real": 1.0},
    )

    assert all(index >= 970 for index in sampler)


def test_sampler_without_weights_keeps_previous_behaviour() -> None:
    """重みを渡さないときは従来どおり unit を一様に引く（件数任せ）。"""
    plain = ClassBalancedSampler({0: tuple(range(1000))}, num_samples=300, seed=9)
    explicit = ClassBalancedSampler(
        {0: tuple(range(1000))},
        num_samples=300,
        seed=9,
        class_dataset_unit_indices=None,
        dataset_weights=None,
    )

    assert list(plain) == list(explicit)


def test_parse_dataset_weights_rejects_malformed_values() -> None:
    from recipes.instrument_refinement.train import parse_dataset_weights

    assert parse_dataset_weights(["real_recordings=3", "cocochorales=0.5"]) == {
        "real_recordings": 3.0,
        "cocochorales": 0.5,
    }
    assert parse_dataset_weights([]) == {}
    for bad in (["real_recordings"], ["=3"], ["real=abc"], ["real=-1"]):
        with pytest.raises(SystemExit):
            parse_dataset_weights(bad)


def _tiny_flat_spec(root: Path, song_count: int) -> dict[str, object]:
    """曲数の少ない flat データセットを作り、その spec を返す。"""
    for index in range(song_count):
        stem = f"song{index}__flute"
        _write_audio(root / "audio" / f"{stem}.wav", frequency=220.0 + 10.0 * index)
        _write_midi(root / "midi" / f"{stem}.mid", program=73)
    return {
        "name": "tiny",
        "type": "flat",
        "root": str(root),
        "audio_glob": "audio/*.wav",
        "midi_dir": "midi",
        "class_source": "midi",
        "song_key_mode": "prefix_before_last_double_underscore",
    }


def test_force_split_keeps_a_small_dataset_entirely_in_one_split(tmp_path: Path) -> None:
    """曲数の少ないデータセットを、分割ごと固定できる。

    実録音は 14 曲しかなく、ハッシュ分割だと train 7 / validation 5 / test 2 に偏った。
    評価用の音源を manifest の外に確保している場合は、全部を train へ寄せたい。
    """
    root = tmp_path / "datasets"
    spec = _tiny_flat_spec(root, song_count=12)
    manifest = tmp_path / "tiny_manifest.csv"
    config = tmp_path / "tiny.local.json"

    _write_dataset_config(config, output=manifest, datasets=[spec])
    build_refinement_manifest(config)
    with manifest.open("r", encoding="utf-8", newline="") as file:
        splits = {row["split"] for row in csv.DictReader(file)}
    assert len(splits) > 1

    _write_dataset_config(config, output=manifest, datasets=[{**spec, "force_split": "train"}])
    build_refinement_manifest(config)
    with manifest.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["split"] for row in rows} == {"train"}
    assert len(rows) == 12


def test_force_split_rejects_an_unknown_split(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    spec = _tiny_flat_spec(root, song_count=2)
    config = tmp_path / "bad.local.json"
    _write_dataset_config(
        config,
        output=tmp_path / "bad_manifest.csv",
        datasets=[{**spec, "force_split": "holdout"}],
    )

    with pytest.raises(ValueError, match="force_split"):
        build_refinement_manifest(config)


def test_dataset_weights_load_from_json_and_yaml(tmp_path: Path) -> None:
    from recipes.instrument_refinement.train import load_dataset_weights

    json_path = tmp_path / "weights.json"
    json_path.write_text(json.dumps({"real_recordings": 6, "cocochorales": 1}), encoding="utf-8")
    assert load_dataset_weights(json_path) == {"real_recordings": 6.0, "cocochorales": 1.0}

    # "weights:" の下にまとめた書き方も、重みだけを並べた書き方も読める。
    yaml_path = tmp_path / "weights.yaml"
    yaml_path.write_text(
        "weights:\n  real_recordings: 6\n  cocochorales: 1\n", encoding="utf-8"
    )
    assert load_dataset_weights(yaml_path) == {"real_recordings": 6.0, "cocochorales": 1.0}

    flat_path = tmp_path / "flat.yaml"
    flat_path.write_text("real_recordings: 2.5\n", encoding="utf-8")
    assert load_dataset_weights(flat_path) == {"real_recordings": 2.5}

    assert load_dataset_weights(None) == {}


def test_dataset_weights_file_rejects_bad_content(tmp_path: Path) -> None:
    from recipes.instrument_refinement.train import load_dataset_weights

    missing = tmp_path / "nope.json"
    with pytest.raises(SystemExit, match="not found"):
        load_dataset_weights(missing)

    negative = tmp_path / "negative.json"
    negative.write_text(json.dumps({"real_recordings": -1}), encoding="utf-8")
    with pytest.raises(SystemExit, match="non-negative"):
        load_dataset_weights(negative)

    not_a_number = tmp_path / "text.json"
    not_a_number.write_text(json.dumps({"real_recordings": "heavy"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="not a number"):
        load_dataset_weights(not_a_number)

    not_a_mapping = tmp_path / "list.json"
    not_a_mapping.write_text(json.dumps(["real_recordings"]), encoding="utf-8")
    with pytest.raises(SystemExit, match="mapping"):
        load_dataset_weights(not_a_mapping)


def test_command_line_overrides_the_weights_file_and_typos_are_rejected(tmp_path: Path) -> None:
    """名前を打ち間違えると重みが効かないまま学習が回ってしまうので、まとめて弾く。"""
    from recipes.instrument_refinement.train import resolve_dataset_weights

    weights_file = tmp_path / "weights.yaml"
    weights_file.write_text("real_recordings: 6\ncocochorales: 5\n", encoding="utf-8")
    known = {"real_recordings", "cocochorales"}

    args = SimpleNamespace(dataset_weights=weights_file, dataset_weight=["cocochorales=1"])
    assert resolve_dataset_weights(args, known) == {"real_recordings": 6.0, "cocochorales": 1.0}

    args = SimpleNamespace(dataset_weights=None, dataset_weight=[])
    assert resolve_dataset_weights(args, known) == {}

    typo = SimpleNamespace(dataset_weights=None, dataset_weight=["real_recording=3"])
    with pytest.raises(SystemExit, match="Unknown dataset name"):
        resolve_dataset_weights(typo, known)
