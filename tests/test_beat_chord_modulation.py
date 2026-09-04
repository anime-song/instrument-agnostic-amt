from __future__ import annotations

import random

import torch

from recipes.beat_chord.datasets.modulation import (
    ModulationAugmentConfig,
    ModulationCandidate,
    WindowHarmonyContext,
    apply_modulation_to_roll,
    choose_modulation_candidate,
    enumerate_modulation_candidates,
)
from recipes.beat_chord.chord import ChordConfig, ChordLoss


def test_modulation_candidates_do_not_require_shifted_song_files() -> None:
    item = {
        "chords": [
            {
                "start_time": 0.0,
                "end_time": 1.0,
                "root": "G",
                "quality": "7",
                "bass": "B",
            },
            {
                "start_time": 1.0,
                "end_time": 2.0,
                "root": "B",
                "quality": "",
                "bass": "D#",
            },
        ],
        "keys": [{"start_time": 0.0, "end_time": 2.0, "key": "C"}],
    }
    context = WindowHarmonyContext(
        source_item=item,
        window_start_sec=0.0,
        window_end_sec=2.0,
        sample_rate=100,
        hop_length=10,
    )

    candidates = enumerate_modulation_candidates(
        context,
        ModulationAugmentConfig(prob=1.0, allowed_shifts=(1,)),
    )
    selected = choose_modulation_candidate(candidates, random.Random(0))

    assert selected is not None
    assert selected.shift == 1
    assert selected.splice_frame == 10
    assert selected.source_key == "C"
    assert selected.target_key == "C#"


def test_modulation_transposes_roll_chord_bass_and_key_after_boundary() -> None:
    roll = torch.zeros(1, 6, 8)
    roll[0, :, 2] = 1.0
    chords = [
        {
            "start_time": 0.0,
            "end_time": 0.3,
            "root": "C",
            "quality": "",
            "bass": "E",
        },
        {
            "start_time": 0.3,
            "end_time": 0.6,
            "root": "D",
            "quality": "m",
            "bass": "F",
        },
    ]
    keys = [{"start_time": 0.0, "end_time": 0.6, "key": "C"}]
    candidate = ModulationCandidate(
        shift=2,
        splice_time_sec=0.3,
        splice_frame=3,
        family="direct_transposition",
        priority=5,
        support_duration_sec=0.6,
        source_key="C",
        target_key="D",
    )

    shifted_roll, shifted_chords, shifted_keys = apply_modulation_to_roll(
        roll,
        chords,
        keys,
        candidate,
    )

    assert torch.equal(shifted_roll[0, :3, 2], torch.ones(3))
    assert torch.equal(shifted_roll[0, 3:, 4], torch.ones(3))
    assert shifted_roll[0, 3:, 2].sum() == 0
    assert shifted_chords[0]["root"] == "C"
    assert shifted_chords[0]["bass"] == "E"
    assert shifted_chords[1]["root"] == "E"
    assert shifted_chords[1]["bass"] == "G"
    assert shifted_keys == [
        {"start_time": 0.0, "end_time": 0.3, "key": "C"},
        {"start_time": 0.3, "end_time": 0.6, "key": "D"},
    ]


def test_chord_and_key_boundary_loss_tolerances_are_separate() -> None:
    config = ChordConfig()
    loss = ChordLoss(config, torch.ones(2))

    assert config.chord_boundary_loss_tolerance == 1
    assert config.key_boundary_loss_tolerance == 8
    assert loss.chord_bce.tolerance == 1
    assert loss.key_bce.tolerance == 8
