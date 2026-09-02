"""Audio-side tempo evidence for the bar-lattice beat decoder.

The beat/chord network only sees a MIDI frame roll, so it has no access to the
timbral cues that separate a quarter-note pulse from the eighth notes played on
top of it: a transcription that renders an arpeggio faithfully looks much like
one that renders the beat.  The decoder inherits that blindness.  ``beat_grid``
only ever penalises the *ratio* between neighbouring segments, so a grid that is
uniformly double time costs exactly nothing, and its tempo is otherwise free to
drift inside an 8 % band.

This module rebuilds the missing evidence straight from the waveform.

``compute_tempo_prior``
    A per-frame distribution over quarter-note period, from a multi-band
    tempogram.  Low-frequency flux follows the kick and lands on the beat while
    high-frequency flux follows the hats and lands on subdivisions, so the bands
    are weighted rather than summed.  Autocorrelation and Fourier tempograms
    disagree about which metrical level to favour, and averaging them in the log
    domain suppresses the octave ghosts each of them produces.

``compute_pulse_curve``
    Predominant Local Pulse: a phase-continuous pulse train from the same onset
    envelope.  Its resolution is the onset hop rather than the beat model's
    23 ms frame, which is what makes it useful for pinning down beat placement.

The prior is returned on the beat model's own frame grid so the decoder consumes
it without knowing anything about the audio front end; the pulse curve keeps its
own finer grid, which is the only reason it is worth computing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPSILON = 1e-10


@dataclass(frozen=True)
class TempoPriorConfig:
    """Front-end settings for :func:`compute_tempo_prior`."""

    # 128 samples at 22.05 kHz is 5.8 ms, four times finer than the beat model's
    # frame, which is what lets the prior resolve tempo better than the grid it
    # is scoring.
    onset_hop_length: int = 128
    onset_n_fft: int = 1024
    n_mels: int = 128
    min_quarter_bpm: float = 30.0
    max_quarter_bpm: float = 300.0
    bins_per_octave: int = 48
    window_seconds: float = 8.0
    band_edges_hz: tuple[float, ...] = (250.0, 2000.0)
    band_weights: tuple[float, ...] = (1.0, 0.7, 0.4)
    fourier_weight: float = 0.5
    # Perceptual tempo is roughly log-normal around 120 BPM.  A weak pull towards
    # it breaks ties between octaves without overriding clear audio evidence.
    octave_prior_bpm: float = 120.0
    octave_prior_sigma_octaves: float = 0.9
    octave_prior_weight: float = 1.0
    # Sharpens (<1) or flattens (>1) the per-frame distribution before it is
    # log-normalised, which is how strongly the prior is allowed to argue.
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.onset_hop_length <= 0:
            raise ValueError("onset_hop_length must be positive")
        if self.onset_n_fft <= 0:
            raise ValueError("onset_n_fft must be positive")
        if self.n_mels <= 1:
            raise ValueError("n_mels must be greater than one")
        if self.min_quarter_bpm <= 0.0:
            raise ValueError("min_quarter_bpm must be positive")
        if self.max_quarter_bpm <= self.min_quarter_bpm:
            raise ValueError("max_quarter_bpm must exceed min_quarter_bpm")
        if self.bins_per_octave <= 0:
            raise ValueError("bins_per_octave must be positive")
        if self.window_seconds <= 0.0:
            raise ValueError("window_seconds must be positive")
        if len(self.band_weights) != len(self.band_edges_hz) + 1:
            raise ValueError("band_weights must hold one more entry than band_edges_hz")
        if any(weight < 0.0 for weight in self.band_weights):
            raise ValueError("band_weights must be non-negative")
        if not 0.0 <= self.fourier_weight <= 1.0:
            raise ValueError("fourier_weight must be between zero and one")
        if self.octave_prior_sigma_octaves <= 0.0:
            raise ValueError("octave_prior_sigma_octaves must be positive")
        if self.octave_prior_weight < 0.0:
            raise ValueError("octave_prior_weight must be non-negative")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")


@dataclass(frozen=True)
class TempoPrior:
    """Log-probability of each quarter-note period, per beat-model frame.

    Periods live on a grid that is uniform in ``log`` of *beat-model frames*, so
    a lookup is arithmetic rather than a search.  ``prefix`` is the running sum
    over frames, which turns "mean log-probability across this segment" into two
    array reads however long the segment is -- the same trick the meter evidence
    in :mod:`beat_grid` already uses.
    """

    log_period_frames: np.ndarray  # (K,) uniform grid of log(quarter period)
    prefix: np.ndarray  # (T + 1, K) cumulative log-probability over frames
    seconds_per_frame: float

    @property
    def frame_count(self) -> int:
        return int(self.prefix.shape[0] - 1)

    @property
    def bin_count(self) -> int:
        return int(self.prefix.shape[1])

    def mean_log_prob(
        self, start_frame: int, end_frame: int, quarter_period_frames: float
    ) -> float:
        """Mean log-probability of ``quarter_period_frames`` over a frame span."""

        if quarter_period_frames <= 0.0:
            return 0.0
        frame_count = self.frame_count
        start = max(0, min(frame_count, int(start_frame)))
        end = max(0, min(frame_count, int(end_frame)))
        if end <= start:
            return 0.0
        window = (self.prefix[end] - self.prefix[start]) / float(end - start)
        return self._interpolate(window, float(quarter_period_frames))

    def frame_log_prob(self, frame: int, quarter_period_frames: float) -> float:
        """Log-probability of a period at a single frame."""

        return self.mean_log_prob(int(frame), int(frame) + 1, quarter_period_frames)

    def best_period_frames(
        self, start_frame: int = 0, end_frame: int | None = None
    ) -> float:
        """Most likely quarter-note period, in frames, over a span."""

        frame_count = self.frame_count
        start = max(0, min(frame_count, int(start_frame)))
        end = frame_count if end_frame is None else max(0, min(frame_count, int(end_frame)))
        if end <= start or self.bin_count == 0:
            return 0.0
        window = self.prefix[end] - self.prefix[start]
        return float(math.exp(self.log_period_frames[int(np.argmax(window))]))

    def _interpolate(self, window: np.ndarray, quarter_period_frames: float) -> float:
        grid = self.log_period_frames
        if grid.size == 0:
            return 0.0
        if grid.size == 1:
            return float(window[0])
        step = float(grid[1] - grid[0])
        position = (math.log(quarter_period_frames) - float(grid[0])) / step
        if position <= 0.0:
            return float(window[0])
        if position >= grid.size - 1:
            return float(window[-1])
        lower = int(position)
        fraction = position - lower
        return float(window[lower] * (1.0 - fraction) + window[lower + 1] * fraction)


def focused_tempo_prior(
    prior: TempoPrior,
    quarter_period_frames: float,
    *,
    sigma_octaves: float = 0.06,
) -> TempoPrior:
    """A prior that asserts one period, on an existing prior's grid.

    Clamping ``min_quarter_bpm``/``max_quarter_bpm`` to steer a second decoding
    pass only deletes edges, which leaves the DP free to assemble something
    erratic from what remains.  Handing it a sharply peaked prior instead states
    the tempo as evidence, so the same scoring path that already balances beat,
    downbeat and meter terms does the steering.
    """

    if quarter_period_frames <= 0.0:
        return prior
    grid = prior.log_period_frames
    octaves = (grid - math.log(float(quarter_period_frames))) / math.log(2.0)
    log_prob = -0.5 * (octaves / float(sigma_octaves)) ** 2
    log_prob -= float(np.max(log_prob))
    log_prob -= math.log(float(np.exp(log_prob).sum()) + _EPSILON)
    frame_count = prior.frame_count
    prefix = np.vstack(
        [
            np.zeros((1, grid.size), dtype=np.float64),
            np.cumsum(np.repeat(log_prob[None, :], frame_count, axis=0), axis=0),
        ]
    )
    return TempoPrior(
        log_period_frames=grid,
        prefix=prefix,
        seconds_per_frame=prior.seconds_per_frame,
    )


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError("waveform must be one- or two-dimensional")
    # Accept both (channels, samples) and (samples, channels) layouts.
    if array.shape[0] <= array.shape[1]:
        return array.mean(axis=0)
    return array.mean(axis=1)


def _band_channel_bounds(
    *, sample_rate: int, n_mels: int, band_edges_hz: tuple[float, ...]
) -> list[int]:
    """Mel-channel indices that split the spectrum at the requested edges."""

    import librosa

    centers = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sample_rate / 2.0)
    bounds = [0]
    for edge_hz in band_edges_hz:
        index = int(np.searchsorted(centers, float(edge_hz)))
        bounds.append(max(bounds[-1] + 1, min(n_mels - 1, index)))
    bounds.append(n_mels)
    return bounds


def _log_period_grid(
    config: TempoPriorConfig, seconds_per_frame: float
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform log grid of quarter-note periods, in seconds and in frames."""

    octaves = math.log2(config.max_quarter_bpm / config.min_quarter_bpm)
    bin_count = max(2, int(round(octaves * config.bins_per_octave)) + 1)
    bpm = np.exp(
        np.linspace(
            math.log(config.min_quarter_bpm),
            math.log(config.max_quarter_bpm),
            bin_count,
        )
    )
    # Reverse so period ascends, which both np.interp and the log grid require.
    period_seconds = (60.0 / bpm)[::-1].copy()
    return period_seconds, np.log(period_seconds / float(seconds_per_frame))


def _normalize_surface(surface: np.ndarray) -> np.ndarray:
    """Per-frame L1 normalisation of a non-negative (K, T) tempo surface."""

    positive = np.maximum(surface, 0.0)
    totals = positive.sum(axis=0, keepdims=True)
    uniform = np.full_like(positive, 1.0 / max(1, positive.shape[0]))
    return np.where(totals > _EPSILON, positive / np.maximum(totals, _EPSILON), uniform)


def _resample_columns(
    surface: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    """Linearly interpolate a (K, T) surface onto a new time axis."""

    if surface.shape[1] == 0 or source_times.size == 0:
        return np.zeros((surface.shape[0], target_times.size), dtype=np.float64)
    if source_times.size == 1:
        return np.repeat(surface[:, :1], target_times.size, axis=1)
    clipped = np.clip(target_times, source_times[0], source_times[-1])
    upper = np.clip(
        np.searchsorted(source_times, clipped, side="left"), 1, source_times.size - 1
    )
    lower = upper - 1
    span = source_times[upper] - source_times[lower]
    fraction = np.where(span > 0.0, (clipped - source_times[lower]) / span, 0.0)
    return surface[:, lower] * (1.0 - fraction) + surface[:, upper] * fraction


def compute_onset_envelopes(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    config: TempoPriorConfig = TempoPriorConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Per-band spectral flux plus its band-weighted mixdown.

    Returns ``(bands, mixed)`` where ``bands`` has shape ``(n_bands, frames)``.
    """

    import librosa

    mono = _to_mono(waveform)
    bounds = _band_channel_bounds(
        sample_rate=sample_rate,
        n_mels=config.n_mels,
        band_edges_hz=config.band_edges_hz,
    )
    bands = np.atleast_2d(
        librosa.onset.onset_strength_multi(
            y=mono,
            sr=sample_rate,
            hop_length=config.onset_hop_length,
            n_fft=config.onset_n_fft,
            n_mels=config.n_mels,
            channels=bounds,
            aggregate=np.median,
        )
    ).astype(np.float64)
    weights = np.asarray(config.band_weights, dtype=np.float64)[: bands.shape[0]]
    total = float(weights.sum())
    if total <= 0.0:
        return bands, bands.mean(axis=0)
    return bands, (bands * weights[:, None]).sum(axis=0) / total


def _window_frames(config: TempoPriorConfig, sample_rate: int) -> int:
    return max(16, int(round(config.window_seconds * sample_rate / config.onset_hop_length)))


def _tempogram_surface(
    envelope: np.ndarray,
    *,
    sample_rate: int,
    config: TempoPriorConfig,
    period_seconds: np.ndarray,
) -> np.ndarray:
    """Blend autocorrelation and Fourier tempograms on the log-period grid."""

    import librosa

    hop = int(config.onset_hop_length)
    if envelope.size < 4:
        return np.full((period_seconds.size, max(1, envelope.size)), 1.0 / period_seconds.size)
    win_length = _window_frames(config, sample_rate)

    autocorrelation = np.asarray(
        librosa.feature.tempogram(
            onset_envelope=envelope,
            sr=sample_rate,
            hop_length=hop,
            win_length=win_length,
        ),
        dtype=np.float64,
    )
    lag_seconds = np.arange(autocorrelation.shape[0]) * hop / float(sample_rate)
    acf_surface = np.empty((period_seconds.size, autocorrelation.shape[1]))
    for index in range(autocorrelation.shape[1]):
        acf_surface[:, index] = np.interp(
            period_seconds, lag_seconds, autocorrelation[:, index], left=0.0, right=0.0
        )
    blended = _normalize_surface(acf_surface)
    if config.fourier_weight <= 0.0:
        return blended

    magnitude = np.abs(
        np.asarray(
            librosa.feature.fourier_tempogram(
                onset_envelope=envelope,
                sr=sample_rate,
                hop_length=hop,
                win_length=win_length,
            )
        )
    ).astype(np.float64)
    fourier_bpm = librosa.fourier_tempo_frequencies(
        sr=sample_rate, hop_length=hop, win_length=win_length
    )
    valid = fourier_bpm > 0.0  # drop DC, which carries no period
    if not np.any(valid):
        return blended
    fourier_periods = 60.0 / fourier_bpm[valid]
    order = np.argsort(fourier_periods)
    sorted_periods = fourier_periods[order]
    fourier_surface = np.empty((period_seconds.size, magnitude.shape[1]))
    for index in range(magnitude.shape[1]):
        fourier_surface[:, index] = np.interp(
            period_seconds,
            sorted_periods,
            magnitude[valid, index][order],
            left=0.0,
            right=0.0,
        )
    fourier_surface = _normalize_surface(fourier_surface)

    frames = min(blended.shape[1], fourier_surface.shape[1])
    # A geometric mean keeps only what both transforms agree on. Their octave
    # ghosts sit at different relative heights, so the product suppresses them
    # while the true pulse, which both rank highly, survives.
    return np.exp(
        (1.0 - config.fourier_weight) * np.log(blended[:, :frames] + _EPSILON)
        + config.fourier_weight * np.log(fourier_surface[:, :frames] + _EPSILON)
    )


def compute_tempo_prior(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    target_hop_length: int,
    target_frame_count: int,
    config: TempoPriorConfig = TempoPriorConfig(),
) -> TempoPrior:
    """Build a per-frame quarter-note period distribution from the waveform."""

    seconds_per_frame = float(target_hop_length) / float(sample_rate)
    period_seconds, log_period_frames = _log_period_grid(config, seconds_per_frame)
    bands, _mixed = compute_onset_envelopes(
        waveform, sample_rate=sample_rate, config=config
    )

    weights = np.asarray(config.band_weights, dtype=np.float64)[: bands.shape[0]]
    accumulated: np.ndarray | None = None
    applied_weight = 0.0
    for band_index in range(bands.shape[0]):
        weight = float(weights[band_index])
        if weight <= 0.0:
            continue
        contribution = weight * np.log(
            _tempogram_surface(
                bands[band_index],
                sample_rate=sample_rate,
                config=config,
                period_seconds=period_seconds,
            )
            + _EPSILON
        )
        applied_weight += weight
        if accumulated is None:
            accumulated = contribution
        else:
            frames = min(accumulated.shape[1], contribution.shape[1])
            accumulated = accumulated[:, :frames] + contribution[:, :frames]
    if accumulated is None:
        accumulated = np.zeros((period_seconds.size, 1), dtype=np.float64)
    else:
        accumulated = accumulated / max(applied_weight, _EPSILON)

    if config.octave_prior_weight > 0.0:
        reference = math.log(60.0 / float(config.octave_prior_bpm))
        octaves = (np.log(period_seconds) - reference) / math.log(2.0)
        accumulated = accumulated + config.octave_prior_weight * (
            -0.5 * (octaves / float(config.octave_prior_sigma_octaves)) ** 2
        )[:, None]

    accumulated = accumulated / float(config.temperature)
    accumulated -= accumulated.max(axis=0, keepdims=True)
    accumulated -= np.log(np.exp(accumulated).sum(axis=0, keepdims=True) + _EPSILON)

    source_times = (
        np.arange(accumulated.shape[1]) * config.onset_hop_length / float(sample_rate)
    )
    target_times = np.arange(int(target_frame_count)) * seconds_per_frame
    resampled = _resample_columns(accumulated, source_times, target_times).T

    prefix = np.vstack(
        [
            np.zeros((1, resampled.shape[1]), dtype=np.float64),
            np.cumsum(resampled, axis=0),
        ]
    )
    return TempoPrior(
        log_period_frames=log_period_frames,
        prefix=prefix,
        seconds_per_frame=seconds_per_frame,
    )


def compute_pulse_curve(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    config: TempoPriorConfig = TempoPriorConfig(),
) -> tuple[np.ndarray, float]:
    """Predominant Local Pulse at the onset hop, with its hop in seconds.

    PLP fits a sinusoid at the locally dominant tempo, so it is continuous in
    phase and carries beat placement far more finely than the 23 ms frames the
    beat network is quantised to.  It is deliberately *not* resampled onto that
    frame grid: the extra resolution is the whole reason to compute it.
    """

    import librosa

    hop_seconds = float(config.onset_hop_length) / float(sample_rate)
    _bands, mixed = compute_onset_envelopes(
        waveform, sample_rate=sample_rate, config=config
    )
    if mixed.size < 4:
        return np.zeros(0, dtype=np.float64), hop_seconds
    pulse = np.asarray(
        librosa.beat.plp(
            onset_envelope=mixed,
            sr=sample_rate,
            hop_length=int(config.onset_hop_length),
            win_length=_window_frames(config, sample_rate),
            tempo_min=float(config.min_quarter_bpm),
            tempo_max=float(config.max_quarter_bpm),
        ),
        dtype=np.float64,
    )
    return pulse, hop_seconds
