#!/usr/bin/env python3
"""
MIDI velocity estimation from audio amplitude — STFT-based approach.

Pipeline:
1. Pre-compute STFT magnitude spectrogram for the entire audio waveform
   (fixed chunk size, e.g. 2048-point FFT, 512-sample hop)
2. For each detected note, assign a non-overlapping set of STFT frames.
   - Notes at DIFFERENT pitches are measured independently (different FFT bins).
   - Notes at the SAME pitch that overlap in time share the overlapping frames
     equally.
3. For each note, extract the maximum magnitude at its fundamental frequency
   across all of its assigned STFT frames.
4. Apply per-instrument-class fundamental-to-total energy ratio correction.
5. Map to MIDI velocity via power-law curve.
"""

import math
import numpy as np
import scipy.signal.windows as win
from typing import Sequence

from instrument_classes import INSTRUMENT_CLASSES


# ===========================================================================
# Per-instrument-class fundamental-to-total energy ratio
# ===========================================================================
# Higher value = more energy concentrated in the fundamental → less correction.
# Lower value = wide harmonic spread / noise → need higher multiplier.
# ===========================================================================

FUNDAMENTAL_ENERGY_RATIO: dict[str, float] = {
    "drums": 0.04,
    "timpani": 0.18,
    "chromatic_percussion": 0.55,
    "percussive_fx": 0.06,
    "piano": 0.28,
    "electric_piano": 0.50,
    "plucked_keyboard": 0.22,
    "organ": 0.55,
    "accordion_family": 0.35,
    "harmonica": 0.35,
    "acoustic_guitar": 0.28,
    "electric_guitar_clean": 0.35,
    "electric_guitar_muted": 0.20,
    "distorted_guitar": 0.06,
    "guitar_harmonics": 0.85,
    "acoustic_bass": 0.25,
    "electric_bass": 0.22,
    "slap_bass": 0.12,
    "synth_bass": 0.40,
    "strings": 0.30,
    "pizzicato_strings": 0.35,
    "orchestral_harp": 0.40,
    "orchestra_hit": 0.10,
    "choir": 0.35,
    "brass": 0.18,
    "sax": 0.28,
    "orchestral_woodwind": 0.45,
    "flute_pipe": 0.82,
    "synth_lead": 0.25,
    "synth_pad": 0.35,
    "synth_fx": 0.08,
    "ethnic": 0.32,
    "sound_fx": 0.05,
    "melody": 0.40,
    "vocal_harmony": 0.35,
}

_DEFAULT_FUNDAMENTAL_RATIO: float = 0.50


def _get_fundamental_ratio(instrument_id: int) -> float:
    return 0.5
    # if 0 <= instrument_id < len(INSTRUMENT_CLASSES):
    #     class_name = INSTRUMENT_CLASSES[instrument_id]
    #     return FUNDAMENTAL_ENERGY_RATIO.get(
    #         class_name.lower().replace("-", "_").replace(" ", "_"),
    #         _DEFAULT_FUNDAMENTAL_RATIO,
    #     )
    # return _DEFAULT_FUNDAMENTAL_RATIO


# ===========================================================================
# Frequency <-> MIDI pitch conversion
# ===========================================================================

_MIDI_A4 = 69
_A4_FREQ = 440.0


def midi_pitch_to_frequency(pitch: int, *, a4_freq: float = _A4_FREQ) -> float:
    return a4_freq * (2.0 ** ((float(pitch) - _MIDI_A4) / 12.0))


# ===========================================================================
# Sample / frame conversion
# ===========================================================================

def _sample_to_stft_frame(sample_idx: int, hop_length: int) -> int:
    return max(0, int(round(float(sample_idx) / float(hop_length))))


def _stft_frame_to_sample(frame_idx: int, hop_length: int) -> int:
    return int(frame_idx) * int(hop_length)


def _pitch_to_stft_bin(pitch: int, sample_rate: int, n_fft: int) -> int:
    freq = midi_pitch_to_frequency(int(pitch))
    return max(0, min(n_fft // 2, int(round(freq * float(n_fft) / float(sample_rate)))))


# ===========================================================================
# STFT pre-computation
# ===========================================================================

def _precompute_stft_magnitude(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> tuple[np.ndarray, int]:
    """Compute STFT magnitude spectrogram of the entire waveform.

    Args:
        waveform: 1D or 2D [channels, samples] float array.
        sample_rate: Audio sample rate in Hz.
        n_fft: FFT window size (samples).
        hop_length: Hop length between successive STFT frames (samples).

    Returns:
        (stft_magnitude, num_total_frames)
        stft_magnitude shape: [num_frames, n_fft//2+1], dtype=float32
    """
    if waveform.ndim == 2:
        mono = waveform.mean(axis=0).astype(np.float64)
    else:
        mono = waveform.astype(np.float64)

    num_samples = mono.shape[-1]
    if num_samples < n_fft:
        # Pad to minimum FFT size
        mono = np.pad(mono, (0, n_fft - num_samples))
        num_samples = n_fft

    # Use librosa-style STFT: sliding window → rfft per frame
    num_frames = 1 + (num_samples - n_fft) // hop_length
    window = win.flattop(n_fft).astype(np.float64)
    stft_mag = np.zeros((num_frames, n_fft // 2 + 1), dtype=np.float32)

    for frame_idx in range(num_frames):
        start = frame_idx * hop_length
        segment = mono[start : start + n_fft] * window
        spec = np.abs(np.fft.rfft(segment))
        stft_mag[frame_idx, :] = spec.astype(np.float32)

    return stft_mag, num_frames


# ===========================================================================
# Non-overlapping interval assignment (per pitch)
# ===========================================================================

def _assign_per_pitch_frame_intervals(
    notes: Sequence,
    hop_length: int,
    stft_total_frames: int,
) -> tuple[list[list[tuple[int, int]]], list[list[tuple[int, int, int]]]]:
    """Assign each note exclusive and shared STFT frame ranges.

    For notes at the SAME pitch:
    - Each note receives an exclusive interval from its onset to either
      its end or the start of the next same-pitch note (whichever is first).
    - If a note's onset falls inside another same-pitch note, the overlapping
      portion is treated as "shared" — its amplitude is later averaged among
      the overlapping notes.

    Notes at DIFFERENT pitches do not interfere (they use different FFT bins).

    Args:
        notes: List of PredictedNote objects.
        hop_length: STFT hop length in samples.
        stft_total_frames: Total number of STFT frames.

    Returns:
        (exclusive_ranges, shared_ranges)
        exclusive_ranges[i]: list of (start_frame, end_frame) tuples owned by note i.
        shared_ranges[i]: list of (start_frame, end_frame, num_sharers) tuples
                          shared by note i.
    """
    n = len(notes)
    exclusive_ranges: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    shared_ranges: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]

    # Group notes by pitch
    by_pitch: dict[int, list[int]] = {}
    for i, note in enumerate(notes):
        pitch = int(note.pitch)
        by_pitch.setdefault(pitch, []).append(i)

    for pitch, indices in by_pitch.items():
        # Sort same-pitch notes by start_frame
        sorted_idx = sorted(indices, key=lambda i: int(notes[i].start_frame))

        # Build timeline events for this pitch
        events: list[tuple[int, str, int]] = []
        for idx in sorted_idx:
            note = notes[idx]
            start_f = _sample_to_stft_frame(int(note.start_frame), hop_length)
            end_f = _sample_to_stft_frame(int(note.end_frame), hop_length)
            end_f = max(end_f, start_f + 1)
            start_f = min(start_f, stft_total_frames - 1)
            end_f = min(end_f, stft_total_frames)
            events.append((start_f, "start", idx))
            events.append((end_f, "end", idx))

        events.sort(key=lambda e: (e[0], 0 if e[1] == "start" else 1))

        # Sweep-line to partition timeline into segments with constant note set
        active: set[int] = set()
        prev_frame = 0
        for frame, evt_type, idx in events:
            frame = max(frame, prev_frame)
            if frame > prev_frame and active:
                if len(active) == 1:
                    owner = next(iter(active))
                    exclusive_ranges[owner].append((prev_frame, frame))
                else:
                    num_sharers = len(active)
                    for idx_a in active:
                        shared_ranges[idx_a].append(
                            (prev_frame, frame, num_sharers)
                        )
            prev_frame = frame
            if evt_type == "start":
                active.add(idx)
            else:
                active.discard(idx)

    # Merge adjacent exclusive ranges for same note
    for i in range(n):
        if not exclusive_ranges[i]:
            continue
        merged: list[tuple[int, int]] = []
        curr_start, curr_end = exclusive_ranges[i][0]
        for s, e in exclusive_ranges[i][1:]:
            if s == curr_end:
                curr_end = e
            else:
                merged.append((curr_start, curr_end))
                curr_start, curr_end = s, e
        merged.append((curr_start, curr_end))
        exclusive_ranges[i] = merged

    return exclusive_ranges, shared_ranges


# ===========================================================================
# Magnitude extraction for a note across its assigned STFT frames
# ===========================================================================

def _extract_note_top_trimmed_magnitude(
    stft_magnitude: np.ndarray,
    pitch: int,
    sample_rate: int,
    n_fft: int,
    exclusive_ranges: list[tuple[int, int]],
    shared_ranges: list[tuple[int, int, int]],
    *,
    top_ratio: float = 0.1,
) -> float:
    """Extract representative magnitude via top-trimmed mean.

    Collects weighted magnitudes from all assigned STFT frames,
    sorts descending, and averages the top `top_ratio` fraction.

    top_ratio=0.1 means averaging the loudest 10% of frames.
    This preserves attack peaks for percussive instruments while
    avoiding a single-noisy-frame maximum.

    Args:
        stft_magnitude: [num_frames, n_fft//2+1] magnitude spectrogram.
        pitch: MIDI pitch of the note.
        sample_rate: Audio sample rate.
        n_fft: FFT window size.
        exclusive_ranges: Exclusive frame intervals (full weight).
        shared_ranges: Shared frame intervals (1/num_sharers weight).
        top_ratio: Fraction of loudest frames to average. Default 0.1.

    Returns:
        Representative weighted magnitude (float).
    """
    bin_idx = _pitch_to_stft_bin(int(pitch), sample_rate, n_fft)
    num_frames = stft_magnitude.shape[0]

    all_magnitudes: list[float] = []

    # Exclusive frames: full weight
    for start_f, end_f in exclusive_ranges:
        start_f = max(0, min(start_f, num_frames))
        end_f = max(0, min(end_f, num_frames))
        if end_f <= start_f:
            continue
        frame_mags = stft_magnitude[start_f:end_f, bin_idx]
        if frame_mags.size > 0:
            all_magnitudes.extend(frame_mags.tolist())

    # Shared frames: divide by number of sharers
    for start_f, end_f, num_sharers in shared_ranges:
        start_f = max(0, min(start_f, num_frames))
        end_f = max(0, min(end_f, num_frames))
        if end_f <= start_f or num_sharers <= 0:
            continue
        frame_mags = stft_magnitude[start_f:end_f, bin_idx]
        if frame_mags.size > 0:
            weighted = frame_mags / float(num_sharers)
            all_magnitudes.extend(weighted.tolist())

    if not all_magnitudes:
        return 0.0

    # Sort descending and average top fraction
    all_magnitudes.sort(reverse=True)
    top_count = max(1, int(round(len(all_magnitudes) * float(top_ratio))))
    top_values = all_magnitudes[:top_count]
    result = float(np.mean(top_values))
    # Debug: print top values for this note
    # print(
    #     f"  [vel-debug] pitch={pitch} frames={len(all_magnitudes)} "
    #     f"top{int(float(top_ratio)*100)}%_mean={result:.4f} "
    #     f"top_values={[f'{v:.4f}' for v in top_values[:5]]}"
    #     f"{'...' if len(top_values) > 5 else ''}"
    # )
    return result


# ===========================================================================
# Amplitude → dB → velocity mapping
# ===========================================================================

def _magnitude_to_linear_amplitude(
    magnitude: float,
    n_fft: int,
) -> float:
    """Convert STFT magnitude to linear amplitude (0..1).

    For a real FFT of a Hanning-windowed signal, the peak magnitude
    of a pure tone of amplitude A is approximately A * N * 0.5 / 2 = A * N / 4.
    So: A ≈ 4 * magnitude / N
    """
    if magnitude <= 0.0:
        return 0.0
    # Flat-top window coherent gain ≈ sum(window) / N ≈ 0.2156
    # FFT magnitude for real sinusoid: ~ A * N * coherent_gain / 2
    coherent_gain = 0.2156
    true_amplitude = 2.0 * float(magnitude) / (float(n_fft) * coherent_gain)
    return float(np.clip(true_amplitude, 0.0, 1.0))


def _amplitude_to_db(amplitude: float, epsilon: float = 1e-10) -> float:
    if amplitude < epsilon:
        return -120.0
    return 20.0 * np.log10(amplitude)


def _db_to_velocity(
    db_value: float,
    min_db: float = -60.0,
    max_db: float = -1.0,
    min_velocity: int = 8,
    max_velocity: int = 127,
    curve_exponent: float = 0.4,
) -> int:
    """Map dBFS to MIDI velocity via power-law curve."""
    clamped = np.clip(float(db_value), float(min_db), float(max_db))
    normalized = (clamped - float(min_db)) / (float(max_db) - float(min_db))
    curved = normalized ** float(curve_exponent)
    vel = int(round(float(min_velocity) + curved * (float(max_velocity) - float(min_velocity))))
    return max(int(min_velocity), min(int(max_velocity), vel))


# ===========================================================================
# Main batch velocity estimation
# ===========================================================================

def estimate_velocities_for_notes(
    waveform: np.ndarray,
    notes: Sequence,
    sample_rate: int,
    *,
    n_fft: int = 2048,
    hop_length: int = 512,
    min_db: float = -60.0,
    max_db: float = -1.0,
    min_velocity: int = 8,
    max_velocity: int = 127,
    curve_exponent: float = 0.4,
    fallback_velocity: int = 80,
) -> list[int]:
    """Estimate MIDI velocities for a list of PredictedNote objects.

    Algorithm:
    1. Pre-compute STFT magnitude spectrogram of the entire waveform.
    2. For each note, assign non-overlapping STFT frame intervals,
       handling same-pitch overlaps by averaging.
    3. For each note, extract the maximum magnitude at its fundamental
       frequency across its assigned frames.
    4. Apply per-instrument-class fundamental energy ratio correction.
    5. Convert to dBFS and map to MIDI velocity.

    Args:
        waveform: Audio waveform [channels, samples] or [samples].
        notes: List of PredictedNote objects.
        sample_rate: Audio sample rate in Hz.
        n_fft: STFT window size (samples). Default 2048.
        hop_length: STFT hop length (samples). Default 512.
        min_db: Minimum dBFS for velocity mapping. Default -48.
        max_db: Maximum dBFS for velocity mapping. Default -3.
        min_velocity: Minimum MIDI velocity. Default 8.
        max_velocity: Maximum MIDI velocity. Default 127.
        curve_exponent: Power-law mapping exponent. Default 0.6.
        fallback_velocity: Velocity when estimation fails. Default 80.

    Returns:
        List of integer MIDI velocities, one per note.
    """
    if not notes:
        return []

    # 1. Pre-compute STFT
    stft_magnitude, stft_total_frames = _precompute_stft_magnitude(
        waveform, sample_rate, n_fft=n_fft, hop_length=hop_length
    )

    # 2. Assign non-overlapping frame intervals per pitch
    exclusive_ranges, shared_ranges = _assign_per_pitch_frame_intervals(
        notes, hop_length, stft_total_frames
    )

    # 3. Extract max magnitude for each note
    velocities: list[int] = []
    for i, note in enumerate(notes):
        try:
            pitch = int(note.pitch)
            instrument_id = (
                int(note.instrument_id) if hasattr(note, "instrument_id") else 0
            )

            # Check if note has any valid frames
            has_exclusive = len(exclusive_ranges[i]) > 0
            has_shared = len(shared_ranges[i]) > 0
            if not has_exclusive and not has_shared:
                velocities.append(int(fallback_velocity))
                continue

            raw_mag = _extract_note_top_trimmed_magnitude(
                stft_magnitude,
                pitch=pitch,
                sample_rate=sample_rate,
                n_fft=n_fft,
                exclusive_ranges=exclusive_ranges[i],
                shared_ranges=shared_ranges[i],
            )

            if raw_mag <= 0.0:
                velocities.append(int(fallback_velocity))
                continue

            # 4. Apply fundamental energy ratio correction
            fund_ratio = max(_get_fundamental_ratio(instrument_id), 0.01)
            corrected_mag = raw_mag / fund_ratio

            # 5. Convert FFT magnitude directly to dB → velocity
            db_val = 20.0 * np.log10(max(corrected_mag, 1e-10))
            # print(
            #     f"  [vel-debug] note[{i}] pitch={pitch} inst={instrument_id} "
            #     f"ratio={fund_ratio:.2f} raw={raw_mag:.4f} corrected={corrected_mag:.4f} "
            #     f"db={db_val:.1f}"
            # )
            velocity = _db_to_velocity(
                db_val,
                min_db=0,
                max_db=48.0,
                min_velocity=int(min_velocity),
                max_velocity=int(max_velocity),
                curve_exponent=1.2,
            )
            # print(f"  [vel-debug] note[{i}] -> velocity={velocity}")
            velocities.append(velocity)
        except Exception:
            velocities.append(int(fallback_velocity))

    return velocities


# ===========================================================================
# Convenience
# ===========================================================================

def apply_velocities_to_notes(
    notes: list,
    velocities: list[int],
) -> list:
    """Write estimated velocities back onto PredictedNote objects in-place."""
    for note, velocity in zip(notes, velocities):
        note.velocity = int(velocity)
    return notes