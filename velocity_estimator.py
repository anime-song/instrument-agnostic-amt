#!/usr/bin/env python3
"""
MIDI velocity estimation from audio amplitude.
For each detected note, extracts a window of samples starting at onset,
applies a flat-top window, performs FFT, and maps the frequency amplitude
to MIDI velocity (0-127).

Features:
- Per-instrument-class fundamental-to-total energy ratio weighting
- Overlapping-note loudness sharing (evenly divides loudness among
  simultaneous notes)
"""

import numpy as np
import scipy.signal.windows as win
from typing import Sequence

from instrument_classes import INSTRUMENT_CLASSES

# ---------------------------------------------------------------------------
# Per-instrument-class fundamental-to-total energy ratio (Optimized Version)
# ---------------------------------------------------------------------------
# Higher value = more energy concentrated in the fundamental → less correction.
# Lower value = wide harmonic spread / noise → need higher multiplier for true loudness.
# ---------------------------------------------------------------------------

FUNDAMENTAL_ENERGY_RATIO: dict[str, float] = {
    # -----------------------------------------------------------------------
    # Percussion & Rhythm (非周期性噪声大，基音极弱或不存在)
    # -----------------------------------------------------------------------
    "drums": 0.04,                # 包含大量白噪/宽频打击，基音能量极低
    "timpani": 0.18,              # 定音鼓有明确音高，基音比一般鼓明显，但泛音衰减慢
    "chromatic_percussion": 0.55,  # 马林巴、铁琴等有共鸣管调谐，基音非常干净纯粹
    "percussive_fx": 0.06,

    # -----------------------------------------------------------------------
    # Keyboard Instruments (受弦长、琴身共鸣影响)
    # -----------------------------------------------------------------------
    "piano": 0.28,                # 钢琴中低音区谐波极大，全键盘平均基音能量并高
    "electric_piano": 0.50,       # 如 Rhodes 类似正弦波的柔和音色，基音较强
    "plucked_keyboard": 0.22,     # 羽管键琴（Harpsichord）泛音极度尖锐丰富
    "organ": 0.55,                # 管风琴（纯笛管音栓基音强，簧管音栓谐波强，取折中）
    "accordion_family": 0.35,     # 风琴类属于自由簧片，带有类似方波的丰富谐波
    "harmonica": 0.35,

    # -----------------------------------------------------------------------
    # Guitar Family (拨弦的脉冲特性与失真系数)
    # -----------------------------------------------------------------------
    "acoustic_guitar": 0.28,      # 原声吉他拨弦瞬间产生大量高频泛音
    "electric_guitar_clean": 0.35, # 电吉他拾音器会滤除部分极高频，基音相对显现
    "electric_guitar_muted": 0.20, # 切音/闷音（Palm Mute）会产生强烈的低频冲击和杂音
    "distorted_guitar": 0.06,      # 极端削波（Clipping）产生海量谐波，基音被严重稀释
    "guitar_harmonics": 0.85,      # 泛音点弹奏（Harmonics）几乎就是纯正弦波

    # -----------------------------------------------------------------------
    # Bass Instruments (声学特征：基音缺失严重)
    # -----------------------------------------------------------------------
    "acoustic_bass": 0.25,       # 大贝斯琴身容积不足以有效放大最低基音
    "electric_bass": 0.22,       # 贝斯原信号中 80Hz-200Hz 的 2、3 次谐波能量远超基音
    "slap_bass": 0.12,            # Slap 包含大量打弦的金属噪声和高频泛音
    "synth_bass": 0.40,           # 电子贝斯常包含 Sine 子低音，基音相对饱满

    # -----------------------------------------------------------------------
    # Strings & Orchestral (弓弦与管乐的物理特性)
    # -----------------------------------------------------------------------
    "strings": 0.30,              # 提琴拉弦呈锯齿波特性（1/n 衰减），谐波总能量超基音
    "pizzicato_strings": 0.35,     # 拨弦（Pizz）高频衰减快，后期基音相对明显
    "orchestral_harp": 0.40,      # 竖琴中高音区比较纯净
    "orchestra_hit": 0.10,        # 乐队齐奏大击乐，纯属全频段能量炸弹
    "choir": 0.35,                # 合唱由于多音色叠加和共振峰碰撞，基音稀释明显
    "brass": 0.18,                # 铜管（小号/长号）是非线性声波重灾区，谐波极度明亮
    "sax": 0.28,                  # 萨克斯（簧片乐器）带有强烈的丰富的木质谐波
    "orchestral_woodwind": 0.45,   # 木管（双簧管/单簧管）谐波较强，单簧管偶次谐波缺失
    "flute_pipe": 0.82,           # 长笛是声学乐器中最接近纯正弦波的，基音占绝对主导

    # -----------------------------------------------------------------------
    # Synths & Effects
    # -----------------------------------------------------------------------
    "synth_lead": 0.25,           # 传统 Lead 多用锯齿波或方波叠加
    "synth_pad": 0.35,            # 铺垫音色通常做过低通滤波，基音比 Lead 略高
    "synth_fx": 0.08,
    "ethnic": 0.32,
    "sound_fx": 0.05,

    # -----------------------------------------------------------------------
    # Solo Melody & Vocals
    # -----------------------------------------------------------------------
    "melody": 0.40,
    "vocal_harmony": 0.35,        # 人声由于 Vowel（元音）共振峰影响，谐波能量很大
}

# Default ratio for unknown / fallback instruments
_DEFAULT_FUNDAMENTAL_RATIO: float = 0.50


def _get_fundamental_ratio(instrument_id: int) -> float:
    """Look up the fundamental energy ratio for an instrument class ID."""
    if 0 <= instrument_id < len(INSTRUMENT_CLASSES):
        class_name = INSTRUMENT_CLASSES[instrument_id]
        return FUNDAMENTAL_ENERGY_RATIO.get(
            class_name.lower().replace("-", "_").replace(" ", "_"),
            _DEFAULT_FUNDAMENTAL_RATIO,
        )
    return _DEFAULT_FUNDAMENTAL_RATIO


# ---------------------------------------------------------------------------
# Flat-top window
# ---------------------------------------------------------------------------

def generate_flattop_window(window_size: int) -> np.ndarray:
    """Generate a flat-top window of the given size.

    A flat-top window provides very accurate amplitude measurements
    because its passband ripple is extremely small (~0.01 dB).
    """
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    return win.flattop(window_size).astype(np.float32)


# ---------------------------------------------------------------------------
# Frequency <-> MIDI pitch conversion
# ---------------------------------------------------------------------------

_MIDI_A4 = 69  # MIDI pitch number for A4
_A4_FREQ = 440.0


def midi_pitch_to_frequency(pitch: int, *, a4_freq: float = _A4_FREQ) -> float:
    """Convert MIDI pitch number to frequency in Hz."""
    return a4_freq * (2.0 ** ((float(pitch) - _MIDI_A4) / 12.0))


def frequency_to_midi_pitch(freq: float, *, a4_freq: float = _A4_FREQ) -> float:
    """Convert frequency in Hz to fractional MIDI pitch number."""
    if freq <= 0.0:
        return 0.0
    return _MIDI_A4 + 12.0 * np.log2(freq / a4_freq)


# ---------------------------------------------------------------------------
# Amplitude extraction for a single note
# ---------------------------------------------------------------------------

def _extract_note_amplitude(
    waveform: np.ndarray,
    start_sample: int,
    pitch: int,
    sample_rate: int,
    window_size: int = 1024,
    flattop_window: np.ndarray | None = None,
) -> float:
    """Extract the amplitude at the note's frequency from the audio.

    Args:
        waveform: 2D audio array [channels, samples]. Will be converted to mono if stereo.
        start_sample: Sample index where the note begins.
        pitch: MIDI pitch of the note.
        sample_rate: Audio sample rate in Hz.
        window_size: Number of samples to extract for FFT (should be power-of-2 for efficiency).
        flattop_window: Pre-computed flat-top window, or None to generate one.

    Returns:
        Linear amplitude (RMS magnitude) at the note's frequency.
        Returns 0.0 if the window extends beyond the audio length or other issues.
    """
    num_samples = int(waveform.shape[-1])
    if start_sample >= num_samples or start_sample < 0:
        return 0.0

    # Determine the actual number of available samples
    available_samples = num_samples - int(start_sample)
    actual_size = min(int(window_size), available_samples)

    if actual_size < 4:
        # Too few samples for meaningful FFT
        return 0.0

    # Extract the window of audio
    if waveform.ndim == 1:
        segment = waveform[start_sample : start_sample + actual_size].copy()
    elif waveform.ndim == 2:
        # Convert to mono by averaging channels
        segment = waveform[:, start_sample : start_sample + actual_size].mean(axis=0)
        if isinstance(segment, np.ndarray):
            segment = segment.copy()
        else:
            return 0.0
    else:
        return 0.0

    # If segment is shorter than window_size, zero-pad
    if actual_size < int(window_size):
        padded = np.zeros(int(window_size), dtype=np.float32)
        padded[:actual_size] = segment
        segment = padded
        actual_size = int(window_size)

    # Generate or use provided flat-top window
    if flattop_window is None:
        flattop_window = generate_flattop_window(actual_size)
    elif len(flattop_window) != actual_size:
        flattop_window = generate_flattop_window(actual_size)

    # Apply flat-top window
    segment = segment.astype(np.float32) * flattop_window[:actual_size]

    # Compute FFT (use rfft since audio is real)
    fft = np.fft.rfft(segment, n=actual_size)
    magnitude = np.abs(fft)

    # Get the frequency of each FFT bin
    freqs = np.fft.rfftfreq(actual_size, d=1.0 / float(sample_rate))

    # Find the bin closest to the note's frequency
    target_freq = midi_pitch_to_frequency(int(pitch))
    bin_index = np.argmin(np.abs(freqs - target_freq))

    # Use a 3-bin average (+1/-1 bin) for numerical stability.
    # Averaging the target bin and its immediate neighbors gives a
    # robust magnitude estimate that is insensitive to scalloping.
    bin_start = max(0, bin_index - 1)
    bin_end = min(len(magnitude), bin_index + 2)
    averaged_mag = float(np.mean(magnitude[bin_start:bin_end]))

    return averaged_mag


# ---------------------------------------------------------------------------
# Magnitude -> dB conversion
# ---------------------------------------------------------------------------

def magnitude_to_db(magnitude: float, *, epsilon: float = 1e-10) -> float:
    """Convert linear magnitude to dBFS."""
    if magnitude < epsilon:
        return -120.0
    return 20.0 * np.log10(float(magnitude) + epsilon)


# ---------------------------------------------------------------------------
# dB -> MIDI velocity mapping
# ---------------------------------------------------------------------------

def db_to_midi_velocity(
    db_value: float,
    *,
    min_db: float = -60.0,
    max_db: float = 0.0,
    min_velocity: int = 1,
    max_velocity: int = 127,
    curve_exponent: float = 0.5,
) -> int:
    """Map a dBFS value to MIDI velocity (0-127).

    Uses a power-law curve for more natural velocity response:
    - velocity ∝ (normalized_amplitude) ^ curve_exponent

    Args:
        db_value: Signal level in dBFS.
        min_db: dB level that maps to min_velocity. Default -60 dBFS.
        max_db: dB level that maps to max_velocity. Default 0 dBFS.
        min_velocity: Minimum MIDI velocity. Default 1.
        max_velocity: Maximum MIDI velocity. Default 127.
        curve_exponent: Power-law exponent. <1 gives more sensitivity to quiet sounds.

    Returns:
        Integer MIDI velocity in range [min_velocity, max_velocity].
    """
    clamped_db = np.clip(float(db_value), float(min_db), float(max_db))
    normalized = (clamped_db - float(min_db)) / (float(max_db) - float(min_db))
    curved = normalized ** float(curve_exponent)
    velocity = int(round(float(min_velocity) + curved * (float(max_velocity) - float(min_velocity))))
    return max(int(min_velocity), min(int(max_velocity), velocity))


# ---------------------------------------------------------------------------
# Main velocity estimation function
# ---------------------------------------------------------------------------

def estimate_velocity(
    amplitude: float,
    *,
    window_size: int = 1024,
    sample_rate: int = 22050,
    min_db: float = -60.0,
    max_db: float = 0.0,
    min_velocity: int = 1,
    max_velocity: int = 127,
    curve_exponent: float = 0.5,
) -> int:
    """Estimate MIDI velocity from an extracted FFT amplitude.

    Args:
        amplitude: Raw FFT magnitude from _extract_note_amplitude().
        window_size: FFT window size used for amplitude extraction.
        sample_rate: Audio sample rate.
        min_db: Minimum dBFS level for mapping.
        max_db: Maximum dBFS level for mapping.
        min_velocity: Minimum output velocity.
        max_velocity: Maximum output velocity.
        curve_exponent: Power-law curve exponent.

    Returns:
        Integer MIDI velocity.
    """
    if amplitude <= 0.0:
        return int(min_velocity)

    flattop = generate_flattop_window(int(window_size))
    coherent_gain = float(np.sum(flattop)) / float(window_size)

    # Convert from FFT magnitude to true linear amplitude
    true_amplitude = 2.0 * float(amplitude) / (float(window_size) * coherent_gain)
    true_amplitude = np.clip(true_amplitude, 0.0, 1.0)

    if true_amplitude < 1e-10:
        db_value = float(min_db)
    else:
        db_value = 20.0 * np.log10(true_amplitude)

    return db_to_midi_velocity(
        db_value,
        min_db=float(min_db),
        max_db=float(max_db),
        min_velocity=int(min_velocity),
        max_velocity=int(max_velocity),
        curve_exponent=float(curve_exponent),
    )


# ---------------------------------------------------------------------------
# Same-pitch overlap detection
# ---------------------------------------------------------------------------

def _build_same_pitch_overlap_groups(
    notes: Sequence,
) -> list[list[int]]:
    """Group notes that have the SAME pitch AND overlap in time.

    Only notes competing for the same FFT frequency bin need to
    share the measured amplitude. Notes at different pitches are
    measured independently — the FFT naturally separates them.

    Uses a union-find sweep-line within each pitch group.
    """
    n = len(notes)
    if n == 0:
        return []

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Group notes by pitch first, then sweep-line within each pitch
    by_pitch: dict[int, list[int]] = {}
    for i, note in enumerate(notes):
        pitch = int(note.pitch)
        by_pitch.setdefault(pitch, []).append(i)

    for pitch_indices in by_pitch.values():
        if len(pitch_indices) <= 1:
            continue
        # Sweep-line on time within same-pitch notes
        sorted_indices = sorted(pitch_indices, key=lambda i: int(notes[i].start_frame))
        active: list[int] = []
        for orig_i in sorted_indices:
            start_i = int(notes[orig_i].start_frame)
            # Expire notes that ended before this one starts
            active = [
                a for a in active
                if int(notes[a].end_frame) > start_i
            ]
            for a in active:
                union(orig_i, a)
            active.append(orig_i)

    # Collect groups
    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Batch velocity estimation for all notes in a stem
# ---------------------------------------------------------------------------

def estimate_velocities_for_notes(
    waveform: np.ndarray,
    notes: Sequence,
    sample_rate: int,
    *,
    window_size: int = 1024,
    min_db: float = -48.0,
    max_db: float = -3.0,
    min_velocity: int = 8,
    max_velocity: int = 127,
    curve_exponent: float = 0.6,
    fallback_velocity: int = 80,
) -> list[int]:
    """Estimate MIDI velocities for a list of PredictedNote objects.

    For each note:
    1. Extract FFT amplitude at the note's fundamental frequency
    2. Apply per-instrument-class fundamental energy ratio correction
    3. Detect overlapping notes and divide loudness equally among them
    4. Map to MIDI velocity

    Args:
        waveform: Audio waveform as numpy array [channels, samples] or [samples].
        notes: List of PredictedNote objects (with .start_frame, .pitch,
               .end_frame, .instrument_id attributes).
        sample_rate: Audio sample rate.
        window_size: Number of samples for FFT analysis window.
        min_db: Minimum dBFS for velocity mapping.
        max_db: Maximum dBFS for velocity mapping.
        min_velocity: Minimum output velocity.
        max_velocity: Maximum output velocity.
        curve_exponent: Power-law mapping exponent.
        fallback_velocity: Velocity to use when estimation fails.

    Returns:
        List of integer MIDI velocities, one per note.
    """
    if not notes:
        return []

    if waveform.ndim == 2:
        mono_waveform = waveform.mean(axis=0)
    else:
        mono_waveform = waveform

    num_samples = mono_waveform.shape[-1]
    flattop_window = generate_flattop_window(int(window_size))

    n = len(notes)

    # Step 1 & 2: Extract raw amplitude for each note, apply fundamental ratio
    raw_amplitudes: list[float] = []
    for i, note in enumerate(notes):
        try:
            start_sample = int(note.start_frame)
            pitch = int(note.pitch)
            instrument_id = int(note.instrument_id) if hasattr(note, 'instrument_id') else 0

            if start_sample < 0 or start_sample >= num_samples - 4:
                raw_amplitudes.append(0.0)
                continue

            amplitude = _extract_note_amplitude(
                waveform,
                start_sample=start_sample,
                pitch=pitch,
                sample_rate=int(sample_rate),
                window_size=int(window_size),
                flattop_window=flattop_window,
            )

            # Apply fundamental energy ratio correction
            # If the fundamental carries only `ratio` of the total energy,
            # we need to multiply the measured amplitude by 1/ratio to
            # estimate the true loudness.
            if amplitude > 0.0:
                fund_ratio = _get_fundamental_ratio(instrument_id)
                fund_ratio = max(fund_ratio, 0.01)  # prevent division by zero
                corrected_amplitude = float(amplitude) / fund_ratio
            else:
                corrected_amplitude = 0.0

            raw_amplitudes.append(corrected_amplitude)
        except Exception:
            raw_amplitudes.append(0.0)

    # Step 3: Detect same-pitch overlap groups
    overlap_groups = _build_same_pitch_overlap_groups(notes)

    # Step 4: Within each group, distribute loudness equally
    group_per_note_amplitude: dict[int, float] = {}

    for group_indices in overlap_groups:
        group_size = len(group_indices)
        if group_size == 0:
            continue

        # Sum all corrected amplitudes within the group
        total_amplitude = sum(raw_amplitudes[i] for i in group_indices)

        if total_amplitude <= 0.0:
            per_note = 0.0
        else:
            # Evenly divide the total loudness among overlapping notes
            per_note = total_amplitude / float(group_size)

        for i in group_indices:
            group_per_note_amplitude[i] = per_note

    # Step 5: Convert amplitude → velocity
    velocities: list[int] = []
    for i in range(n):
        amplitude = group_per_note_amplitude.get(i, raw_amplitudes[i])

        if amplitude <= 0.0:
            velocities.append(int(fallback_velocity))
            continue

        try:
            velocity = estimate_velocity(
                amplitude,
                window_size=int(window_size),
                sample_rate=int(sample_rate),
                min_db=float(min_db),
                max_db=float(max_db),
                min_velocity=int(min_velocity),
                max_velocity=int(max_velocity),
                curve_exponent=float(curve_exponent),
            )
            velocities.append(velocity)
        except Exception:
            velocities.append(int(fallback_velocity))

    return velocities


# ---------------------------------------------------------------------------
# Convenience: apply estimated velocities back onto PredictedNote objects
# ---------------------------------------------------------------------------

def apply_velocities_to_notes(
    notes: list,
    velocities: list[int],
) -> list:
    """Return a new list of notes with velocities replaced by estimated values.

    Modifies notes in-place by setting .velocity on each note and returns
    the same list. Also returns the list for chaining.

    Args:
        notes: List of PredictedNote objects.
        velocities: List of integer MIDI velocities, same length as notes.

    Returns:
        The same notes list (modified in-place).
    """
    for note, velocity in zip(notes, velocities):
        note.velocity = int(velocity)
    return notes