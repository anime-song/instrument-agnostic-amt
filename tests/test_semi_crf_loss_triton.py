from __future__ import annotations

import importlib.util
import sys

import pytest
import torch

from instrument_agnostic_amt.cli import train
from instrument_agnostic_amt.modeling.heads import semi_crf
from instrument_agnostic_amt.modeling.heads.semi_crf import (
    NeuralSemiCRFInterval,
    compute_factorized_pair_interval_loss,
    compute_pitch_interval_loss,
)


TRITON_CUDA_AVAILABLE = bool(
    torch.cuda.is_available() and importlib.util.find_spec("triton") is not None
)


def test_loss_backend_rejects_unknown_name_and_cpu_triton() -> None:
    score = torch.zeros(5, 5, 2)
    noise_score = torch.zeros(4, 2)
    semi_crf_layer = NeuralSemiCRFInterval(score, noise_score)

    with pytest.raises(ValueError, match="backend must be one of"):
        semi_crf_layer.computeLogZ(backend="unknown")
    with pytest.raises(ValueError, match="requires a CUDA tensor"):
        semi_crf_layer.computeLogZ(backend="triton")


def test_training_cli_exposes_optional_triton_loss_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py"])
    defaults = train.parse_args()
    assert defaults.semi_crf_loss_backend == "torch"

    monkeypatch.setattr(
        sys,
        "argv",
        ["train.py", "--semi-crf-loss-backend", "triton"],
    )
    explicit = train.parse_args()
    assert explicit.semi_crf_loss_backend == "triton"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        train.validate_args(explicit)


def test_loss_functions_propagate_backend_with_active_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded_backends: list[str] = []

    # CUDAを必要とする実kernelは別テストで確認し、ここではchunk伝播だけを見る。
    monkeypatch.setattr(semi_crf, "_validate_loss_backend", lambda *args: None)

    def fake_compute_log_z(
        instance: NeuralSemiCRFInterval,
        noBackward: bool = False,
        *,
        backend: str = "torch",
    ) -> torch.Tensor:
        del noBackward
        recorded_backends.append(backend)
        return instance.score.sum(dim=(0, 1)) * 0.0

    monkeypatch.setattr(NeuralSemiCRFInterval, "computeLogZ", fake_compute_log_z)

    # 1. V1 loss-augmented分岐を複数track chunkに分けて確認する。
    interval_query = torch.randn(1, 5, 3, 2)
    interval_key = torch.randn(1, 5, 3, 2)
    interval_diag = torch.randn(1, 5, 3)
    v1_targets = [[[(0, 1)], [], [(2, 4)]]]
    _, track_count, _ = compute_pitch_interval_loss(
        interval_query,
        interval_key,
        interval_diag,
        v1_targets,
        [5],
        track_batch_size=2,
        false_negative_cost=0.3,
        false_positive_cost=0.2,
        backend="triton",
    )
    assert track_count == 3
    assert recorded_backends == ["triton", "triton"]

    # 2. V2 factorized lossでも同じbackendを各chunkへ渡す。
    pitch_query = torch.randn(1, 5, 2, 3)
    pitch_key = torch.randn(1, 5, 2, 3)
    pitch_diag = torch.randn(1, 5, 2)
    instrument_query = torch.randn(2, 3)
    instrument_key = torch.randn(2, 3)
    instrument_diag = torch.randn(2)
    pair_batch_indices = torch.tensor([0, 0, 0], dtype=torch.long)
    pair_instrument_indices = torch.tensor([0, 1, 1], dtype=torch.long)
    pair_pitch_indices = torch.tensor([0, 0, 1], dtype=torch.long)
    v2_targets = [[(0, 1)], [], [(2, 4)]]
    _, track_count, _ = compute_factorized_pair_interval_loss(
        pitch_query,
        pitch_key,
        pitch_diag,
        instrument_query,
        instrument_key,
        instrument_diag,
        pair_batch_indices,
        pair_instrument_indices,
        pair_pitch_indices,
        v2_targets,
        [5],
        track_batch_size=2,
        false_negative_cost=0.3,
        false_positive_cost=0.2,
        backend="triton",
    )
    assert track_count == 3
    assert recorded_backends == ["triton", "triton", "triton", "triton"]


def _log_z_with_gradients(
    score: torch.Tensor,
    noise_score: torch.Tensor,
    upstream: torch.Tensor,
    *,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """同一入力からlog Zとscore/noise勾配を計算する。"""
    score_input = score.detach().clone().requires_grad_()
    noise_input = noise_score.detach().clone().requires_grad_()
    log_z = NeuralSemiCRFInterval(score_input, noise_input).computeLogZ(
        backend=backend
    )
    (log_z * upstream).sum().backward()
    return log_z.detach(), score_input.grad, noise_input.grad


@pytest.mark.skipif(
    not TRITON_CUDA_AVAILABLE,
    reason="Triton CUDA backend is not available",
)
def test_triton_log_z_and_backward_match_torch() -> None:
    device = torch.device("cuda")

    # 小さい境界ケースと実際の8秒窓を同じ解析的DPで比較する。
    for time_steps, track_count in ((1, 3), (17, 5), (345, 5)):
        generator = torch.Generator(device=device).manual_seed(time_steps)
        score = (
            torch.randn(
                time_steps,
                time_steps,
                track_count,
                generator=generator,
                device=device,
            )
            * 0.2
        )
        noise_score = (
            torch.randn(
                max(0, time_steps - 1),
                track_count,
                generator=generator,
                device=device,
            )
            * 0.2
        )
        upstream = torch.randn(track_count, generator=generator, device=device)

        expected = _log_z_with_gradients(
            score,
            noise_score,
            upstream,
            backend="torch",
        )
        actual = _log_z_with_gradients(
            score,
            noise_score,
            upstream,
            backend="triton",
        )
        torch.testing.assert_close(
            actual[0],
            expected[0],
            atol=2e-4,
            rtol=1e-7,
        )
        for actual_tensor, expected_tensor in zip(actual[1:], expected[1:]):
            torch.testing.assert_close(
                actual_tensor,
                expected_tensor,
                atol=2e-4,
                rtol=2e-4,
            )


@pytest.mark.skipif(
    not TRITON_CUDA_AVAILABLE,
    reason="Triton CUDA backend is not available",
)
def test_triton_augmented_loss_backward_reaches_interval_projections() -> None:
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(101)
    batch_size, time_steps, pitch_count, head_dim = 1, 37, 6, 8
    base_query = torch.randn(
        batch_size,
        time_steps,
        pitch_count,
        head_dim,
        generator=generator,
        device=device,
    )
    base_key = torch.randn(
        batch_size,
        time_steps,
        pitch_count,
        head_dim,
        generator=generator,
        device=device,
    )
    base_diag = torch.randn(
        batch_size,
        time_steps,
        pitch_count,
        generator=generator,
        device=device,
    )
    targets = [
        [
            [(2, 8), (15, 24)] if pitch % 2 == 0 else []
            for pitch in range(pitch_count)
        ]
    ]

    def run(backend: str) -> tuple[torch.Tensor, ...]:
        query = base_query.detach().clone().requires_grad_()
        key = base_key.detach().clone().requires_grad_()
        diag = base_diag.detach().clone().requires_grad_()

        # 実学習と同じautocast下でもscoreはfloat32になり、上流まで勾配が戻る。
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss, _, _ = compute_pitch_interval_loss(
                query,
                key,
                diag,
                targets,
                [time_steps],
                length_scaling="none",
                false_negative_cost=0.3,
                false_positive_cost=0.2,
                backend=backend,
            )
        loss.backward()
        return loss.detach(), query.grad, key.grad, diag.grad

    expected = run("torch")
    actual = run("triton")
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert torch.all(torch.isfinite(actual_tensor))
        torch.testing.assert_close(
            actual_tensor,
            expected_tensor,
            atol=5e-4,
            rtol=5e-4,
        )
