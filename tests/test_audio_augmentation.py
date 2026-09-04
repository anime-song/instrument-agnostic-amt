"""AudioAugmentor の compressor 周りの回帰テスト。

compressor_augment のエンベロープフォロワは numba が入っていれば JIT 版へ差し替わる。
augmentation.py は本家 AMT と instrument_refinement の学習で共有しているので、
速くなった代わりに音が変わっていた、という事故を起こさないよう bit 一致を固定する。
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from recipes.common import augmentation as aug

SAMPLE_RATE = 22_050


def _reference_envelope(
    sidechain: np.ndarray,
    attack_coeff: float,
    release_coeff: float,
) -> np.ndarray:
    """高速化する前の実装をそのまま残したもの。これが正解の定義。

    累算は Python の float（= float64）で行い、配列へ書くときだけ float32 に丸める。
    """
    env = np.empty(sidechain.shape[0], dtype=np.float32)
    prev = 0.0
    for index in range(sidechain.shape[0]):
        current = float(sidechain[index])
        coeff = attack_coeff if current > prev else release_coeff
        prev = coeff * prev + (1.0 - coeff) * current
        env[index] = prev
    return env


def _signals() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(7)
    axis = np.arange(4096, dtype=np.float32) / SAMPLE_RATE
    tone = (0.4 * np.sin(2.0 * np.pi * 220.0 * axis)).astype(np.float32)
    impulse = np.zeros(4096, dtype=np.float32)
    impulse[1000] = 1.0
    return [
        ("正弦波", tone),
        ("雑音", (rng.standard_normal(4096) * 0.3).astype(np.float32)),
        ("無音", np.zeros(1024, dtype=np.float32)),
        # 立ち上がりで attack、そのあと release へ切り替わる境目を通す
        ("インパルス", impulse),
        ("フルスケール", np.full(2048, 0.999, dtype=np.float32)),
    ]


@pytest.mark.parametrize("label,signal", _signals(), ids=[name for name, _ in _signals()])
@pytest.mark.parametrize("attack_ms,release_ms", [(15.0, 120.0), (5.0, 50.0), (30.0, 250.0)])
def test_envelope_follower_matches_the_original_loop(
    label: str,
    signal: np.ndarray,
    attack_ms: float,
    release_ms: float,
) -> None:
    del label
    attack_coeff = float(np.exp(-1.0 / max(1.0, SAMPLE_RATE * attack_ms / 1000.0)))
    release_coeff = float(np.exp(-1.0 / max(1.0, SAMPLE_RATE * release_ms / 1000.0)))
    expected = _reference_envelope(signal, attack_coeff, release_coeff)

    follower = aug._get_envelope_follower()
    actual = follower(np.ascontiguousarray(signal, dtype=np.float32), attack_coeff, release_coeff)

    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)


def test_pure_python_fallback_is_used_when_numba_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """numba が無い環境でも落ちず、素の実装に戻るだけであること。

    numba は librosa 経由で入っていることが多いだけで直接の依存ではないので、
    欠けた環境で ImportError にならないことを保証する。
    """
    # sys.modules に None を置くと、その名前の import は ImportError になる。
    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setattr(aug, "_ENVELOPE_FOLLOWER", None)

    assert aug._get_envelope_follower() is aug._follow_envelope


@pytest.mark.parametrize("channels", [1, 2])
def test_compressor_keeps_shape_and_dtype(channels: int) -> None:
    axis = np.arange(2048, dtype=np.float32) / SAMPLE_RATE
    mono = (0.5 * np.sin(2.0 * np.pi * 220.0 * axis)).astype(np.float32)
    samples = mono if channels == 1 else np.stack([mono, mono * 0.6])

    result = aug.compressor_augment(samples, SAMPLE_RATE)

    assert result.shape == samples.shape
    assert result.dtype == np.float32
    assert np.isfinite(result).all()


def test_compressor_passes_empty_input_through() -> None:
    empty = np.zeros((2, 0), dtype=np.float32)
    assert aug.compressor_augment(empty, SAMPLE_RATE).shape == (2, 0)


def test_compressor_reduces_level_above_the_threshold() -> None:
    """しきい値を超える大きな信号は、比を上げるほど小さくなる。"""
    loud = np.full((2, 8192), 0.9, dtype=np.float32)
    gentle = aug.compressor_augment(loud, SAMPLE_RATE, threshold_db=-20.0, ratio=2.0)
    strong = aug.compressor_augment(loud, SAMPLE_RATE, threshold_db=-20.0, ratio=8.0)

    tail = slice(-1024, None)
    assert np.abs(strong[:, tail]).max() < np.abs(gentle[:, tail]).max() < 0.9
