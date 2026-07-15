#!/usr/bin/env python3
"""
Simple WAV to MIDI Converter (Enhanced with Spectral Filtering)
- 50% overlap STFT, local maxima extraction, merging of neighboring frequencies
- Dynamic noise floor threshold filtering to remove weak noise
- Minimum note duration filtering (default 0.05 seconds)
- Velocity automatically calculated based on average amplitude; notes below 40 are discarded
- Default instrument: Ocarina (GM 79)
- Outputs MIDI file (using pretty_midi)
"""

import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import stft
import pretty_midi


def _db_to_velocity(
    db_value: float,
    min_db: float = -48,
    max_db: float = -1.0,
    min_velocity: int = 8,
    max_velocity: int = 127,
    curve_exponent: float = 1.2,
) -> int:
    """Maps dBFS values to MIDI velocity (power law curve)"""
    clamped = np.clip(float(db_value), float(min_db), float(max_db))
    normalized = (clamped - float(min_db)) / (float(max_db) - float(min_db))
    curved = normalized ** float(curve_exponent)
    vel = int(round(float(min_velocity) + curved * (float(max_velocity) - float(min_velocity))))
    return max(int(min_velocity), min(int(max_velocity), vel))


def wav_to_midi(wav_path, midi_path, win_len=4096, n_peaks=8,
                min_velocity=40, noise_threshold_factor=3.0,
                min_note_duration=0.02):
    """
    Converts a WAV file to a MIDI file

    Parameters:
        wav_path              : Input WAV file path
        midi_path             : Output MIDI file path
        win_len               : STFT window length (default 2048)
        n_peaks               : Number of peaks to keep per frame (default 8)
        min_velocity          : Minimum velocity threshold, notes below this are discarded (default 40)
        noise_threshold_factor: Noise floor factor; amplitude must be > median × this factor to be kept (default 3.0)
        min_note_duration     : Minimum note duration (seconds); notes shorter than this are discarded (default 0.05)
    """
    # ---------- Read Audio ----------
    sr, audio = wavfile.read(wav_path)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # ---------- STFT ----------
    f, _, Zxx = stft(audio, fs=sr, window='hann',
                     nperseg=win_len, noverlap=win_len // 2,
                     boundary=None, padded=True)
    magnitude = np.abs(Zxx)
    hop_len = win_len // 2
    global_max = np.max(magnitude)
    if global_max == 0:
        global_max = 1e-12

    # Semitone frequency ratio threshold (for merging adjacent peaks)
    SEMITONE_RATIO = 2.0 ** (1.0 / 24.0)   # Half a semitone

    # Optional: Frequency axis median filtering (uncomment if enabled below)
    # from scipy.signal import medfilt
    # magnitude = medfilt(magnitude, kernel_size=(5, 1))  # Median filter along frequency axis, window size 5

    frames_data = []   # A dict {pitch: amplitude} for each frame

    for frame_idx in range(magnitude.shape[1]):
        mag = magnitude[:, frame_idx]

        # ---------- 1. Noise Floor Threshold Filtering ----------
        # Calculate median as noise floor (only consider frequencies f > 0)
        valid_idx = np.where(f > 0)[0]
        if len(valid_idx) == 0:
            frames_data.append({})
            continue
        mag_valid = mag[valid_idx]
        noise_floor = np.median(mag_valid) if mag_valid.size > 0 else 0.0
        threshold = noise_floor * noise_threshold_factor

        # Set amplitudes below the threshold to zero
        mag_filtered = mag.copy()
        mag_filtered[valid_idx] = np.where(mag_valid >= threshold, mag_valid, 0.0)

        # ---------- 2. Find Local Maxima ----------
        peaks = []
        # Only consider points where f > 0 and amplitude > 0
        for i in valid_idx:
            if i == 0 or i == len(mag_filtered)-1:
                continue
            # Check if both left and right neighbors are lower than the current value (local maximum)
            if mag_filtered[i] > mag_filtered[i-1] and mag_filtered[i] > mag_filtered[i+1]:
                if mag_filtered[i] > 0:
                    peaks.append((f[i], mag_filtered[i]))

        if not peaks:
            frames_data.append({})
            continue

        # ---------- 3. Sort by Frequency, Merge Close Frequencies ----------
        peaks.sort(key=lambda x: x[0])
        merged_peaks = []
        for freq, amp in peaks:
            if not merged_peaks:
                merged_peaks.append([freq, amp])
            else:
                last_freq, last_amp = merged_peaks[-1]
                if freq / last_freq < SEMITONE_RATIO:
                    if amp > last_amp:
                        merged_peaks[-1] = [freq, amp]
                else:
                    merged_peaks.append([freq, amp])

        # ---------- 4. Take the top n_peaks with largest amplitude ----------
        merged_peaks.sort(key=lambda x: x[1], reverse=True)
        selected = merged_peaks[:n_peaks]

        # ---------- 5. Map to MIDI Pitch ----------
        pitch_amp = {}
        for freq, amp in selected:
            note = int(round(69 + 12 * np.log2(freq / 440.0)))
            if 0 <= note <= 127:
                if note not in pitch_amp or amp > pitch_amp[note]:
                    pitch_amp[note] = amp

        frames_data.append(pitch_amp)

    # ---------- 6. Merge Consecutive Frames, Generate Notes (Velocity Filter + Duration Filter) ----------
    all_notes = set().union(*(d.keys() for d in frames_data)) if frames_data else set()
    midi_notes = []
    time_per_frame = hop_len / sr   # Duration per frame (seconds)

    for pitch in all_notes:
        in_note = False
        start_idx = None
        amp_list = []
        for i, pitch_amp in enumerate(frames_data):
            if pitch in pitch_amp:
                if not in_note:
                    in_note = True
                    start_idx = i
                    amp_list = []
                amp_list.append(pitch_amp[pitch])
            else:
                if in_note:
                    duration = (i - start_idx) * time_per_frame
                    if duration >= min_note_duration:
                        avg_amp = np.mean(amp_list) if amp_list else 0.0
                        if avg_amp > 0:
                            db = 20.0 * np.log10(avg_amp / global_max)
                        else:
                            db = -60.0
                        velocity = _db_to_velocity(db)
                        if velocity >= min_velocity:
                            start_time = start_idx * time_per_frame
                            end_time   = i * time_per_frame
                            midi_notes.append(
                                pretty_midi.Note(velocity=velocity, pitch=pitch,
                                                 start=start_time, end=end_time)
                            )
                    in_note = False
        # Handle notes that extend to the very end
        if in_note:
            duration = (len(frames_data) - start_idx) * time_per_frame
            if duration >= min_note_duration:
                avg_amp = np.mean(amp_list) if amp_list else 0.0
                if avg_amp > 0:
                    db = 20.0 * np.log10(avg_amp / global_max)
                else:
                    db = -60.0
                velocity = _db_to_velocity(db)
                if velocity >= min_velocity:
                    start_time = start_idx * time_per_frame
                    end_time   = len(frames_data) * time_per_frame
                    midi_notes.append(
                        pretty_midi.Note(velocity=velocity, pitch=pitch,
                                         start=start_time, end=end_time)
                    )

    # ---------- 7. Write MIDI ----------
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=79)   # Ocarina
    instrument.notes = midi_notes
    midi.instruments.append(instrument)
    midi.write(str(midi_path))

    print(f"Conversion complete, MIDI file saved to: {midi_path}")
    print(f"Generated {len(midi_notes)} notes (velocity >= {min_velocity}, duration >= {min_note_duration} seconds)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python wav2midi.py <input.wav> <output.mid>")
        print("Optional parameters: win_len, n_peaks, min_velocity, noise_threshold_factor, min_note_duration can be modified in the script")
        sys.exit(1)

    wav_to_midi(sys.argv[1], sys.argv[2])