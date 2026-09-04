from __future__ import annotations

import importlib.util
import sys

import pytest
import torch

from instrument_agnostic_amt.amt.cli import infer
from instrument_agnostic_amt.amt.modeling.heads import semi_crf
from instrument_agnostic_amt.amt.modeling.heads.semi_crf import viterbiBackward
from instrument_agnostic_amt.amt.modeling.model import SemiCRFModelConfig


TRITON_CUDA_AVAILABLE = bool(
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)


def test_viterbi_backend_rejects_unknown_name() -> None:
    score = torch.zeros(5, 5, 2)
    noise_score = torch.zeros(4, 2)

    with pytest.raises(ValueError, match="backend must be one of"):
        viterbiBackward(score, noise_score, backend="unknown")


def test_triton_backend_rejects_cpu_tensor_before_import() -> None:
    score = torch.zeros(5, 5, 2)
    noise_score = torch.zeros(4, 2)

    with pytest.raises(ValueError, match="requires a CUDA tensor"):
        viterbiBackward(score, noise_score, backend="triton")


def test_dense_decoders_propagate_backend_to_each_track_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_backends: list[str] = []

    def fake_decode(
        instance: semi_crf.NeuralSemiCRFInterval,
        forcedStartPos: list[int] | None = None,
        forward: bool = False,
        *,
        backend: str = "torch",
    ) -> list[list[tuple[int, int]]]:
        del forcedStartPos, forward
        recorded_backends.append(backend)
        return [[] for _ in range(int(instance.score.shape[2]))]

    monkeypatch.setattr(semi_crf.NeuralSemiCRFInterval, "decode", fake_decode)
    interval_query = torch.randn(1, 5, 3, 2)
    interval_key = torch.randn(1, 5, 3, 2)
    interval_diag = torch.randn(1, 5, 3)

    decoded = semi_crf.decode_pitch_intervals(
        interval_query,
        interval_key,
        interval_diag,
        [5],
        track_batch_size=2,
        backend="triton",
    )

    assert decoded == [[[], [], []]]
    assert recorded_backends == ["triton", "triton"]

    pitch_query = torch.randn(1, 5, 2, 3)
    pitch_key = torch.randn(1, 5, 2, 3)
    pitch_diag = torch.randn(1, 5, 2)
    instrument_query = torch.randn(2, 3)
    instrument_key = torch.randn(2, 3)
    instrument_diag = torch.randn(2)
    pair_batch_indices = torch.tensor([0, 0, 0], dtype=torch.long)
    pair_instrument_indices = torch.tensor([0, 1, 1], dtype=torch.long)
    pair_pitch_indices = torch.tensor([0, 0, 1], dtype=torch.long)

    factorized = semi_crf.decode_factorized_pair_intervals(
        pitch_query,
        pitch_key,
        pitch_diag,
        instrument_query,
        instrument_key,
        instrument_diag,
        pair_batch_indices,
        pair_instrument_indices,
        pair_pitch_indices,
        [5],
        track_batch_size=2,
        backend="triton",
    )

    assert factorized == [[], [], []]
    assert recorded_backends == ["triton", "triton", "triton", "triton"]


def test_cli_accepts_explicit_triton_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["infer.py", "--audio", "input.wav", "--semi-crf-backend", "triton"],
    )

    args = infer.parse_args()
    settings = infer.resolve_inference_settings(
        SemiCRFModelConfig(sample_rate=22_050, hop_length=512),
        {},
        args,
    )

    assert args.semi_crf_backend == "triton"
    assert settings.semi_crf_backend == "triton"


def test_cli_rejects_triton_with_sparse_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer.py",
            "--audio",
            "input.wav",
            "--semi-crf-backend",
            "triton",
            "--semi-crf-sparse-decode",
            "--semi-crf-sparse-max-span-ms",
            "100",
        ],
    )

    with pytest.raises(SystemExit):
        infer.parse_args()


@pytest.mark.skipif(
    not TRITON_CUDA_AVAILABLE,
    reason="Triton CUDA backend is not available",
)
def test_triton_viterbi_matches_torch_for_random_and_tied_scores() -> None:
    device = torch.device("cuda")

    # 1. 非ゼロnoiseとforced startを含む乱数ケースを比較する。
    for seed in range(20):
        generator = torch.Generator(device=device).manual_seed(seed)
        score = torch.randn(17, 17, 7, generator=generator, device=device)
        noise_score = torch.randn(16, 7, generator=generator, device=device)
        forced_start_pos = [(seed + track * 3) % 17 for track in range(7)]

        expected = viterbiBackward(score, noise_score, forced_start_pos)
        actual = viterbiBackward(
            score,
            noise_score,
            forced_start_pos,
            backend="triton",
        )

        assert actual == expected

    # 2. skipおよび複数区間の同点規則を明示的に比較する。
    skip_tie_score = torch.zeros(6, 6, 3, device=device)
    interval_tie_score = torch.zeros(6, 6, 3, device=device)
    interval_tie_score[2, 0, :] = 1.0
    interval_tie_score[3, 0, :] = 1.0
    noise_score = torch.zeros(5, 3, device=device)
    forced_start_pos = [0, 1, 2]
    for score in (skip_tie_score, interval_tie_score):
        assert viterbiBackward(
            score,
            noise_score,
            forced_start_pos,
            backend="triton",
        ) == viterbiBackward(score, noise_score, forced_start_pos)

    # 3. 実際の8秒窓と同じT=345でもwarp間依存を含めて比較する。
    generator = torch.Generator(device=device).manual_seed(1_000)
    score = torch.randn(345, 345, 5, generator=generator, device=device)
    noise_score = torch.randn(344, 5, generator=generator, device=device)
    assert viterbiBackward(score, noise_score, backend="triton") == viterbiBackward(
        score,
        noise_score,
    )


def _mostly_silent_scores() -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], list[int]]:
    """大半のtrackに区間が立たない、実音源の1窓に近いスコアを作る。"""
    generator = torch.Generator().manual_seed(7)
    time_steps, track_count = 12, 9
    # 既定は区間もsingletonも採用されない負のスコア。3 trackだけ乱数にする。
    score = torch.full((time_steps, time_steps, track_count), -5.0)
    noise_score = torch.zeros(time_steps - 1, track_count)
    active_tracks = (1, 4, 8)
    for track in active_tracks:
        score[:, :, track] = torch.randn(
            time_steps,
            time_steps,
            generator=generator,
        )
    forced_start_pos = [track % 3 for track in range(track_count)]
    return score, noise_score, active_tracks, forced_start_pos


def test_backward_decoding_keeps_track_order_when_most_tracks_are_silent() -> None:
    """空のtrackを転送から外しても、track indexと区間が単独デコードと一致する。"""
    score, noise_score, active_tracks, forced_start_pos = _mostly_silent_scores()

    decoded = viterbiBackward(score, noise_score, forced_start_pos)

    assert len(decoded) == int(score.shape[2])
    assert all(
        not decoded[track]
        for track in range(int(score.shape[2]))
        if track not in active_tracks
    )
    for track in active_tracks:
        assert decoded[track]
        single_track = viterbiBackward(
            score[:, :, track : track + 1],
            noise_score[:, track : track + 1],
            [forced_start_pos[track]],
        )
        assert decoded[track] == single_track[0]


@pytest.mark.skipif(
    not TRITON_CUDA_AVAILABLE,
    reason="Triton CUDA backend is not available",
)
def test_triton_backward_keeps_track_order_when_most_tracks_are_silent() -> None:
    score, noise_score, _, forced_start_pos = _mostly_silent_scores()
    device = torch.device("cuda")

    assert viterbiBackward(
        score.to(device),
        noise_score.to(device),
        forced_start_pos,
        backend="triton",
    ) == viterbiBackward(score, noise_score, forced_start_pos)
