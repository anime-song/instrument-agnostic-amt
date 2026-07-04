import copy
import random
from pathlib import Path
from typing import Optional, Tuple

from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt, sosfiltfilt
import numpy as np


_PEDALBOARD_CHORUS = None
_PEDALBOARD_PHASER = None
SUPPORTED_DISTORTION_AUGMENTATIONS = (
    "saturation",
    "soft_clipping",
    "hard_clipping",
    "asymmetric_saturation",
    "octaver",
    "fuzz",
)


def _load_pedalboard_modulation_plugins():
    global _PEDALBOARD_CHORUS, _PEDALBOARD_PHASER

    if _PEDALBOARD_CHORUS is not None and _PEDALBOARD_PHASER is not None:
        return _PEDALBOARD_CHORUS, _PEDALBOARD_PHASER

    try:
        from pedalboard import Chorus, Phaser
    except ImportError as exc:
        raise ImportError(
            "pedalboard is required for chorus/phaser augmentation. "
            "Please install it with `pip install pedalboard`."
        ) from exc

    _PEDALBOARD_CHORUS = Chorus
    _PEDALBOARD_PHASER = Phaser
    return _PEDALBOARD_CHORUS, _PEDALBOARD_PHASER


def transient_gain_augment(
    samples: np.ndarray,
    sample_rate: int,
    gain: float = 1.5,
    env_ms: float = 5.0,
    smooth_ms: float = 2.0,
    normalize: bool = False,
):
    """
    samples:
        mono:   shape (samples,)
        stereo: shape (channels, samples)

    gain:
        1.0 なら変化なし
        1.5 ならアタック強調
        0.6 ならアタック弱化
    """

    x = samples.astype(np.float32)

    # stereoなら平均してアタック検出用のmonoを作る
    if x.ndim == 1:
        mono = x
    else:
        mono = np.mean(x, axis=0)

    env_win = max(1, int(sample_rate * env_ms / 1000))
    smooth_win = max(1, int(sample_rate * smooth_ms / 1000))

    # 短時間の音量包絡
    env = uniform_filter1d(np.abs(mono), size=env_win)

    # 立ち上がり成分だけ取り出す
    diff = np.maximum(np.diff(env, prepend=env[0]), 0.0)

    if diff.max() < 1e-8:
        return samples

    # 0〜1 に正規化
    attack_mask = diff / (diff.max() + 1e-8)

    # 少し滑らかにする
    attack_mask = uniform_filter1d(attack_mask, size=smooth_win)

    # attack部分だけgainを変える
    gain_curve = 1.0 + (gain - 1.0) * attack_mask

    if x.ndim == 1:
        y = x * gain_curve
    else:
        y = x * gain_curve[None, :]

    if normalize:
        peak = np.max(np.abs(y))
        if peak > 1.0:
            y = y / peak

    return y.astype(np.float32)


def db_to_amp(db: float) -> float:
    return 10.0 ** (db / 20.0)


def amp_to_db(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, eps))


def compressor_augment(
    samples: np.ndarray,
    sample_rate: int,
    threshold_db: float = -18.0,
    ratio: float = 2.0,
    attack_ms: float = 15.0,
    release_ms: float = 120.0,
    makeup_gain_db: float = 0.0,
    normalize: bool = False,
) -> np.ndarray:
    """
    軽い dynamic range compressor。

    samples:
        mono:   shape (frames,)
        stereo: shape (channels, frames)

    threshold_db:
        このレベルを超えた部分を圧縮する

    ratio:
        2.0 なら threshold 超過分を 1/2 にする

    attack_ms:
        圧縮がかかり始める速さ
        小さいほどアタックを潰しやすい

    release_ms:
        圧縮が戻る速さ

    makeup_gain_db:
        圧縮後に全体を持ち上げる量
    """

    x = samples.astype(np.float32, copy=False)

    is_mono = x.ndim == 1
    if is_mono:
        x_ch = x[None, :]
        sidechain = np.abs(x)
    else:
        x_ch = x
        # stereo/マルチchでは全ch共通のゲインを作る
        sidechain = np.max(np.abs(x), axis=0)

    n = sidechain.shape[0]
    if n == 0:
        return samples

    # envelope follower
    attack_coeff = np.exp(-1.0 / max(1.0, sample_rate * attack_ms / 1000.0))
    release_coeff = np.exp(-1.0 / max(1.0, sample_rate * release_ms / 1000.0))

    env = np.empty_like(sidechain, dtype=np.float32)
    prev = 0.0

    for i in range(n):
        current = float(sidechain[i])

        if current > prev:
            coeff = attack_coeff
        else:
            coeff = release_coeff

        prev = coeff * prev + (1.0 - coeff) * current
        env[i] = prev

    env_db = amp_to_db(env)

    # thresholdを超えた分だけ圧縮
    over_db = env_db - threshold_db
    gain_reduction_db = np.zeros_like(env_db, dtype=np.float32)

    mask = over_db > 0.0
    # 入力がthresholdを over_db 超えたとき、
    # 出力では over_db / ratio だけ超えるようにする
    gain_reduction_db[mask] = -(over_db[mask] - over_db[mask] / ratio)

    gain_db = gain_reduction_db + makeup_gain_db
    gain = db_to_amp(gain_db).astype(np.float32)

    y = x_ch * gain[None, :]

    if is_mono:
        y = y[0]

    if normalize:
        peak = np.max(np.abs(y))
        if peak > 1.0:
            y = y / peak

    return y.astype(np.float32)


def random_transient_gain(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return transient_gain_augment(
        samples,
        sample_rate,
        gain=random.uniform(0.7, 1.4),
        env_ms=random.uniform(3.0, 8.0),
        smooth_ms=random.uniform(1.0, 4.0),
        normalize=False,
    )


def random_light_compression(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return compressor_augment(
        samples,
        sample_rate,
        threshold_db=random.uniform(-30.0, -12.0),
        ratio=random.uniform(2.0, 5.0),
        attack_ms=random.uniform(5.0, 30.0),
        release_ms=random.uniform(50.0, 250.0),
        makeup_gain_db=random.uniform(0.0, 6.0),
        normalize=False,
    )


def _normalize_peak_if_needed(samples: np.ndarray) -> np.ndarray:
    """必要なときだけピークを 1.0 以下へ正規化する。"""
    peak = float(np.max(np.abs(samples))) if samples.size > 0 else 0.0
    if peak > 1.0:
        samples = samples / peak
    return samples.astype(np.float32, copy=False)


def _blend_dry_wet(
    dry: np.ndarray,
    wet: np.ndarray,
    *,
    mix: float,
) -> np.ndarray:
    """dry/wet を線形補間し、必要ならピークを正規化する。"""
    mix = float(np.clip(mix, 0.0, 1.0))
    blended = (1.0 - mix) * dry + mix * wet
    return _normalize_peak_if_needed(blended)


def _apply_lowpass_filter(
    samples: np.ndarray,
    sample_rate: int,
    *,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    波形へ Butterworth low-pass filter を適用する。

    通常は zero-phase の filtfilt を使い、短すぎる波形だけ片方向フィルタへ
    フォールバックする。
    """
    x = samples.astype(np.float32, copy=False)
    nyquist = float(sample_rate) * 0.5
    cutoff_hz = float(np.clip(cutoff_hz, 20.0, nyquist * 0.95))
    sos = butter(int(order), cutoff_hz / nyquist, btype="low", output="sos")

    # sosfiltfilt は短すぎる配列で失敗するので、その場合だけ sosfilt を使う。
    axis_len = int(x.shape[-1])
    n_sections = int(sos.shape[0])
    padlen = 3 * max(1, 2 * n_sections)
    if axis_len <= padlen:
        filtered = sosfilt(sos, x, axis=-1)
    else:
        filtered = sosfiltfilt(sos, x, axis=-1)
    return filtered.astype(np.float32, copy=False)


def _build_suboctave_square(
    samples: np.ndarray,
    sample_rate: int,
    cutoff_hz: float = 240.0,
) -> np.ndarray:
    """
    簡易 sub-octave を生成する。

    rising zero crossing ごとに極性を反転させることで、
    元波形より 1 オクターブ下の矩形波に近い成分を作る。
    """
    x = samples.astype(np.float32, copy=False)
    mono = x if x.ndim == 1 else np.mean(x, axis=0)
    if mono.size == 0:
        return x.copy()

    # 1. rising zero crossing を見つける
    rising_crossing = (mono[1:] >= 0.0) & (mono[:-1] < 0.0)
    toggle = np.zeros_like(mono, dtype=np.int32)
    toggle[1:] = rising_crossing.astype(np.int32, copy=False)
    parity = np.cumsum(toggle, dtype=np.int32) % 2
    square = 1.0 - 2.0 * parity.astype(np.float32, copy=False)

    # 2. 元波形の包絡で音量を与える
    env_win = max(1, int(sample_rate * 0.0025))
    envelope = uniform_filter1d(np.abs(mono), size=env_win)
    sub = square * envelope

    # 3. 角を少し丸めてクリック感を弱める
    smooth_win = max(1, int(sample_rate * 0.0015))
    sub = uniform_filter1d(sub, size=smooth_win).astype(np.float32, copy=False)
    sub = _apply_lowpass_filter(
        sub,
        sample_rate,
        cutoff_hz=cutoff_hz,
        order=4,
    )

    if x.ndim == 1:
        return sub
    return np.repeat(sub[None, :], x.shape[0], axis=0)


def _build_octave_up_component(
    samples: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """
    簡易 octave-up 成分を生成する。

    全波整流で倍周波を強めたあと、DC 成分を落として少し平滑化する。
    """
    x = samples.astype(np.float32, copy=False)
    up = np.abs(x)
    up = up - up.mean(axis=-1, keepdims=True)
    smooth_win = max(1, int(sample_rate * 0.0008))
    up = uniform_filter1d(up, size=smooth_win, axis=-1)
    return up.astype(np.float32, copy=False)


def _validate_distortion_augmentations(
    distortion_augmentations: Optional[Tuple[str, ...] | list[str]],
) -> tuple[str, ...]:
    """歪み系 augment 名を検証し、重複を除いたタプルへ正規化する。"""
    if distortion_augmentations is None:
        return ()

    normalized: list[str] = []
    seen: set[str] = set()
    for effect_name in distortion_augmentations:
        effect_name = str(effect_name)
        if effect_name not in SUPPORTED_DISTORTION_AUGMENTATIONS:
            raise ValueError(
                "Unsupported distortion augmentation: "
                f"{effect_name}. Supported values are "
                f"{SUPPORTED_DISTORTION_AUGMENTATIONS}."
            )
        if effect_name in seen:
            continue
        seen.add(effect_name)
        normalized.append(effect_name)
    return tuple(normalized)


def saturation_augment(
    samples: np.ndarray,
    sample_rate: int,
    drive: float = 2.5,
    mix: float = 1.0,
) -> np.ndarray:
    """対称な tanh saturation を適用する。"""
    del sample_rate
    x = samples.astype(np.float32, copy=False)
    drive = max(1.0, float(drive))
    wet = np.tanh(drive * x) / np.tanh(drive)
    return _blend_dry_wet(x, wet, mix=mix)


def soft_clipping_augment(
    samples: np.ndarray,
    sample_rate: int,
    drive: float = 2.0,
    mix: float = 1.0,
) -> np.ndarray:
    """arctan ベースの滑らかな soft clipping を適用する。"""
    del sample_rate
    x = samples.astype(np.float32, copy=False)
    drive = max(1.0, float(drive))
    wet = (2.0 / np.pi) * np.arctan(drive * x)
    return _blend_dry_wet(x, wet, mix=mix)


def hard_clipping_augment(
    samples: np.ndarray,
    sample_rate: int,
    drive: float = 2.0,
    threshold: float = 0.6,
    mix: float = 1.0,
) -> np.ndarray:
    """増幅後に閾値で打ち切る hard clipping を適用する。"""
    del sample_rate
    x = samples.astype(np.float32, copy=False)
    drive = max(1.0, float(drive))
    threshold = float(np.clip(threshold, 1e-3, 1.0))
    wet = np.clip(drive * x, -threshold, threshold) / threshold
    return _blend_dry_wet(x, wet, mix=mix)


def asymmetric_saturation_augment(
    samples: np.ndarray,
    sample_rate: int,
    positive_drive: float = 2.5,
    negative_drive: float = 1.4,
    mix: float = 1.0,
) -> np.ndarray:
    """正負で異なる drive を使う非対称 saturation を適用する。"""
    del sample_rate
    x = samples.astype(np.float32, copy=False)
    positive_drive = max(1.0, float(positive_drive))
    negative_drive = max(1.0, float(negative_drive))

    positive = np.tanh(positive_drive * np.maximum(x, 0.0)) / np.tanh(positive_drive)
    negative = np.tanh(negative_drive * np.minimum(x, 0.0)) / np.tanh(negative_drive)
    wet = positive + negative
    return _blend_dry_wet(x, wet, mix=mix)


def octaver_augment(
    samples: np.ndarray,
    sample_rate: int,
    down_mix: float = 0.18,
    up_mix: float = 0.10,
    mix: float = 1.0,
) -> np.ndarray:
    """
    原音へ小さな sub-octave / octave-up 成分を加えた octaver 風の音を作る。
    """
    x = samples.astype(np.float32, copy=False)
    wet = x.copy()

    if down_mix > 0.0:
        wet = wet + float(down_mix) * _build_suboctave_square(x, sample_rate)
    if up_mix > 0.0:
        wet = wet + float(up_mix) * _build_octave_up_component(x, sample_rate)

    wet = _normalize_peak_if_needed(wet)
    return _blend_dry_wet(x, wet, mix=mix)


def fuzz_augment(
    samples: np.ndarray,
    sample_rate: int,
    drive: float = 10.0,
    threshold: float = 0.18,
    bias: float = 0.03,
    lowpass_cutoff_hz: float = 1800.0,
    mix: float = 1.0,
) -> np.ndarray:
    """
    波形を強く押し潰して、矩形波寄りの fuzz を作る。
    """
    x = samples.astype(np.float32, copy=False)
    drive = max(1.0, float(drive))
    threshold = float(np.clip(threshold, 1e-3, 1.0))
    bias = float(np.clip(bias, -0.25, 0.25))

    preamped = drive * (x + bias)
    saturated = np.tanh(preamped)
    wet = np.clip(saturated, -threshold, threshold) / threshold
    wet = np.tanh(1.5 * wet).astype(np.float32, copy=False)
    wet = _apply_lowpass_filter(
        wet,
        sample_rate,
        cutoff_hz=lowpass_cutoff_hz,
        order=4,
    )
    return _blend_dry_wet(x, wet, mix=mix)


def random_saturation(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return saturation_augment(
        samples,
        sample_rate,
        drive=random.uniform(1.4, 5.0),
        mix=random.uniform(0.2, 0.7),
    )


def random_soft_clipping(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return soft_clipping_augment(
        samples,
        sample_rate,
        drive=random.uniform(1.2, 4.0),
        mix=random.uniform(0.15, 0.6),
    )


def random_hard_clipping(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return hard_clipping_augment(
        samples,
        sample_rate,
        drive=random.uniform(1.4, 4.5),
        threshold=random.uniform(0.25, 0.8),
        mix=random.uniform(0.05, 0.3),
    )


def random_asymmetric_saturation(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return asymmetric_saturation_augment(
        samples,
        sample_rate,
        positive_drive=random.uniform(1.4, 5.0),
        negative_drive=random.uniform(1.1, 3.0),
        mix=random.uniform(0.1, 0.5),
    )


def random_octaver(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    include_down = random.random() < 0.85
    include_up = random.random() < 0.65
    if not include_down and not include_up:
        include_down = True

    return octaver_augment(
        samples,
        sample_rate,
        down_mix=(random.uniform(0.10, 0.28) if include_down else 0.0),
        up_mix=(random.uniform(0.04, 0.16) if include_up else 0.0),
        mix=random.uniform(0.25, 0.55),
    )


def random_fuzz(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    return fuzz_augment(
        samples,
        sample_rate,
        drive=random.uniform(6.0, 16.0),
        threshold=random.uniform(0.10, 0.28),
        bias=random.uniform(-0.05, 0.08),
        lowpass_cutoff_hz=random.uniform(900.0, 2600.0),
        mix=random.uniform(0.20, 0.55),
    )


def random_chorus(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    Chorus, _ = _load_pedalboard_modulation_plugins()

    chorus = Chorus(
        rate_hz=random.uniform(0.2, 1.1),
        depth=random.uniform(0.05, 0.25),
        centre_delay_ms=random.uniform(7.0, 18.0),
        feedback=random.uniform(0.0, 0.08),
        mix=random.uniform(0.12, 0.32),
    )
    y = chorus(samples.astype(np.float32, copy=False), sample_rate)
    return y.astype(np.float32, copy=False)


def random_phaser(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    _, Phaser = _load_pedalboard_modulation_plugins()

    phaser = Phaser(
        rate_hz=random.uniform(0.12, 0.8),
        depth=random.uniform(0.15, 0.45),
        centre_frequency_hz=random.uniform(350.0, 1600.0),
        feedback=random.uniform(0.0, 0.18),
        mix=random.uniform(0.08, 0.24),
    )
    y = phaser(samples.astype(np.float32, copy=False), sample_rate)
    return y.astype(np.float32, copy=False)


class AudioAugmentor:
    """
    オーディオ波形に対するデータ拡張
    （EQ、ピッチシフト(微小)、モジュレーション、歪み、リバーブ、ノイズ）
    を適用するクラス。
    """

    def __init__(
        self,
        sample_rate: int = 22050,
        pitch_shift_semitones: Tuple[float, float] = (
            -0.2,
            0.2,
        ),  # マイクロチューニング
        eq_db_range: Tuple[float, float] = (-6.0, 6.0),
        snr_range: Tuple[float, float] = (3.0, 40.0),
        ir_folder: Optional[str | Path] = None,
        noise_folder: Optional[str | Path] = None,
        distortion_augmentations: Optional[Tuple[str, ...] | list[str]] = None,
    ):
        try:
            from audiomentations import (
                AddGaussianSNR,
                Compose,
                SomeOf,
                OneOf,
                PitchShift,
                ApplyImpulseResponse,
                AddBackgroundNoise,
                SevenBandParametricEQ,
                LowShelfFilter,
                HighShelfFilter,
                PeakingFilter,
                HighPassFilter,
                PolarityInversion,
                Reverse,
                BandStopFilter,
                Aliasing,
                Lambda,
            )
        except ImportError:
            raise ImportError(
                "audiomentations is not installed. Please run `pip install audiomentations`"
            )

        # Lambda内でランダムに失敗しないよう、初期化時に依存を確認する。
        _load_pedalboard_modulation_plugins()

        self.sample_rate = sample_rate
        self.distortion_augmentations = _validate_distortion_augmentations(
            distortion_augmentations
        )

        nyq = sample_rate / 2
        high_shelf_max = min(9000.0, nyq * 0.85)

        # 1. 基本変換 (ピッチ微調整、EQ)
        self.basic_transform = Compose(
            [
                PitchShift(
                    min_semitones=pitch_shift_semitones[0],
                    max_semitones=pitch_shift_semitones[1],
                    p=0.5,
                ),
                SevenBandParametricEQ(
                    min_gain_db=eq_db_range[0], max_gain_db=eq_db_range[1], p=0.5
                ),
                SomeOf(
                    (1, 3),
                    [
                        # 低域の太さ・薄さ
                        LowShelfFilter(
                            min_center_freq=60,
                            max_center_freq=220,
                            min_gain_db=-6.0,
                            max_gain_db=6.0,
                            min_q=0.3,
                            max_q=0.8,
                            p=1.0,
                        ),
                        # 高域の明るさ・暗さ
                        HighShelfFilter(
                            min_center_freq=3500,
                            max_center_freq=high_shelf_max,
                            min_gain_db=-6.0,
                            max_gain_db=6.0,
                            min_q=0.3,
                            max_q=0.8,
                            p=1.0,
                        ),
                        # こもり / 箱鳴り
                        PeakingFilter(
                            min_center_freq=180,
                            max_center_freq=500,
                            min_gain_db=-6.0,
                            max_gain_db=6.0,
                            min_q=0.4,
                            max_q=1.2,
                            p=1.0,
                        ),
                        # 中域の張り出し
                        PeakingFilter(
                            min_center_freq=700,
                            max_center_freq=2500,
                            min_gain_db=-6.0,
                            max_gain_db=6.0,
                            min_q=0.4,
                            max_q=1.0,
                            p=1.0,
                        ),
                        # プレゼンス / 刺さり
                        PeakingFilter(
                            min_center_freq=2500,
                            max_center_freq=7000,
                            min_gain_db=-6.0,
                            max_gain_db=6.0,
                            min_q=0.5,
                            max_q=1.5,
                            p=1.0,
                        ),
                        # 特定帯域の自然な欠落
                        BandStopFilter(
                            min_center_freq=500.0,
                            max_center_freq=6000.0,
                            min_bandwidth_fraction=0.15,
                            max_bandwidth_fraction=0.7,
                            min_rolloff=6,
                            max_rolloff=18,
                            zero_phase=False,
                            p=1.0,
                        ),
                        # 低品質処理っぽい折り返し/ザラつき
                        Aliasing(
                            min_sample_rate=10000,
                            max_sample_rate=18000,
                            p=1.0,
                        ),
                    ],
                    p=0.75,
                ),
                # ミックス・マスタリング後の低域整理に近い軽いHPF
                OneOf(
                    [
                        HighPassFilter(
                            min_cutoff_freq=20,
                            max_cutoff_freq=50,
                            min_rolloff=12,
                            max_rolloff=24,
                            zero_phase=True,
                            p=1.0,
                        ),
                        HighPassFilter(
                            min_cutoff_freq=50,
                            max_cutoff_freq=120,
                            min_rolloff=12,
                            max_rolloff=24,
                            zero_phase=True,
                            p=1.0,
                        ),
                    ],
                    weights=[0.85, 0.15],
                    p=0.6,
                ),
                PolarityInversion(p=0.1),
            ]
        )

        self.dynamics_transform = OneOf(
            transforms=[
                # アタックだけ揺らす
                Lambda(
                    transform=random_transient_gain,
                    p=1.0,
                ),
                # 軽いコンプ
                Lambda(
                    transform=random_light_compression,
                    p=1.0,
                ),
            ],
            weights=[0.55, 0.45],
            p=0.35,
        )

        self.modulation_transform = OneOf(
            transforms=[
                Lambda(
                    transform=random_chorus,
                    p=1.0,
                ),
                Lambda(
                    transform=random_phaser,
                    p=1.0,
                ),
            ],
            weights=[0.5, 0.5],
            p=0.25,
        )

        # 1.9. dataset ごとに選択可能な歪み系 augmentation
        distortion_transform_builders = {
            "saturation": random_saturation,
            "soft_clipping": random_soft_clipping,
            "hard_clipping": random_hard_clipping,
            "asymmetric_saturation": random_asymmetric_saturation,
            "octaver": random_octaver,
            "fuzz": random_fuzz,
        }
        if self.distortion_augmentations:
            self.distortion_transform = OneOf(
                transforms=[
                    Lambda(
                        transform=distortion_transform_builders[effect_name],
                        p=1.0,
                    )
                    for effect_name in self.distortion_augmentations
                ],
                weights=[1.0] * len(self.distortion_augmentations),
                p=0.3,
            )
        else:
            self.distortion_transform = None

        # 2. リバーブ (IRコンボリューション)
        self.reverb = None
        if ir_folder is not None:
            ir_path = Path(ir_folder)
            if ir_path.exists():
                # ApplyImpulseResponseはフォルダパス(ir_path)を受け取る仕様
                self.reverb = ApplyImpulseResponse(
                    ir_path=str(ir_path),
                    p=0.5,
                    lru_cache_size=2000,
                    leave_length_unchanged=True,
                )

        # 3. 背景ノイズ・ホワイトノイズ
        noise_transforms = []
        if noise_folder is not None:
            noise_path = Path(noise_folder)
            if noise_path.exists():
                bg_noise_trans = Compose(
                    [
                        PolarityInversion(p=0.5),
                        Reverse(p=0.5),
                    ]
                )
                noise_transforms.append(
                    AddBackgroundNoise(
                        sounds_path=str(noise_path),
                        min_snr_db=snr_range[0],
                        max_snr_db=snr_range[1],
                        p=0.5,
                        lru_cache_size=256,
                        noise_transform=bg_noise_trans,
                    )
                )

        noise_transforms.append(
            AddGaussianSNR(min_snr_db=snr_range[0], max_snr_db=snr_range[1], p=0.5)
        )
        self.noise_transform = Compose(noise_transforms)

    def __call__(self, audio: np.ndarray) -> np.ndarray:
        """
        オーディオテンソルにデータ拡張を適用します。
        Args:
            audio: numpy array の波形。Shapeは [channels, frames] を想定。
        Returns:
            拡張された波形。Shapeは [2, frames] (ステレオ化して返す)。
        """
        audio = copy.deepcopy(audio)

        # audiomentations は [channels, frames] のステレオ入力をサポートしているため、
        # そのまま渡すことで左右チャンネルの位相・ステレオ感を維持したまま処理します。
        x = audio.astype(np.float32)

        # 1. 基本変換
        x = self.basic_transform(x, sample_rate=self.sample_rate)

        # 1.5. アタック/コンプなどのdynamic系
        x = self.dynamics_transform(x, sample_rate=self.sample_rate)

        # 1.75. chorus/phaserなどのmodulation系
        x = self.modulation_transform(x, sample_rate=self.sample_rate)

        # 1.9. dataset ごとに有効化された歪み系 augmentation
        if self.distortion_transform is not None:
            x = self.distortion_transform(x, sample_rate=self.sample_rate)

        # 2. リバーブ (Dry/Wetをランダムにブレンド)
        if self.reverb is not None:
            x_reverb = self.reverb(x, sample_rate=self.sample_rate)
            # DryとWet(リバーブ)の比率をランダムに決定
            alpha = random.random()
            x = alpha * x + (1 - alpha) * x_reverb

        # 3. ノイズ
        x = self.noise_transform(x, sample_rate=self.sample_rate)

        # 4. ステレオ・パンニング操作 (低確率でのみ適用)
        if len(x.shape) == 2 and x.shape[0] == 2:
            # L/Rの反転 (確率10%)
            if random.random() < 0.1:
                x = x[::-1, :].copy()

            # ランダムパンニング (左右の音量バランスの変更、確率20%)
            if random.random() < 0.2:
                # pan_factor が 0.5 なら左が半分、右が1.5倍になる（合計2.0を維持）
                pan_factor = random.uniform(0.5, 1.5)
                x[0, :] *= pan_factor
                x[1, :] *= 2.0 - pan_factor

                # パンニングによって1.0を超えた場合のクリッピング防止
                peak = np.abs(x).max()
                if peak > 1.0:
                    x /= peak

        return x


if __name__ == "__main__":
    import argparse
    import os
    import soundfile as sf

    parser = argparse.ArgumentParser(
        description="Save dry and chorus/phaser previews using librosa's trumpet example."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("augmentation_preview"),
        help="Directory to write preview wav files.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=22050,
        help="Preview sample rate.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional duration in seconds to load from the trumpet example.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible effect parameters.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir / "numba"))

    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "librosa is required for the preview. Please install it with `pip install librosa`."
        ) from exc

    try:
        audio_path = librosa.ex("trumpet")
        audio, _ = librosa.load(
            audio_path,
            sr=args.sample_rate,
            mono=False,
            duration=args.duration,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to load librosa's trumpet example. "
            "On first use, librosa downloads it from librosa.org, so network access may be required."
        ) from exc

    audio = audio.astype(np.float32, copy=False)
    if audio.ndim == 1:
        dry = np.stack([audio, audio], axis=0)
    elif audio.shape[0] == 1:
        dry = np.repeat(audio, 2, axis=0)
    else:
        dry = audio[:2]

    chorus = random_chorus(dry, args.sample_rate)
    phaser = random_phaser(dry, args.sample_rate)
    chorus_phaser = random_phaser(chorus, args.sample_rate)
    audio_augmentor = AudioAugmentor(sample_rate=args.sample_rate)(dry)

    outputs = {
        "trumpet_original.wav": dry,
        "trumpet_chorus.wav": chorus,
        "trumpet_phaser.wav": phaser,
        "trumpet_chorus_phaser.wav": chorus_phaser,
        "trumpet_audio_augmentor.wav": audio_augmentor,
    }

    for name, audio in outputs.items():
        path = args.out_dir / name
        y = audio.astype(np.float32, copy=False)
        if y.size > 0:
            peak = float(np.max(np.abs(y)))
            if peak > 1.0:
                y = y / peak
        if y.ndim == 2:
            y = y.T
        sf.write(str(path), y, args.sample_rate, subtype="PCM_16")
        print(path)
