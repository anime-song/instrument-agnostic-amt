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
    # Onset-envelope frames between tempogram columns.  Zero means "match the
    # beat model's frame grid", which is what the surface is resampled onto
    # anyway; see :func:`_resolve_tempogram_stride`.
    tempogram_stride: int = 0
    # Columns per GPU batch.  The hop=1 spectra behind these surfaces are large
    # enough that a whole track at once would reserve several gigabytes.
    torch_chunk_frames: int = 2048

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
        if self.tempogram_stride < 0:
            raise ValueError("tempogram_stride must be non-negative")
        if self.torch_chunk_frames <= 0:
            raise ValueError("torch_chunk_frames must be positive")


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

    def mean_log_prob(self, start_frame: int, end_frame: int, quarter_period_frames: float) -> float:
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

    def best_period_frames(self, start_frame: int = 0, end_frame: int | None = None) -> float:
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


def _band_channel_bounds(*, sample_rate: int, n_mels: int, band_edges_hz: tuple[float, ...]) -> list[int]:
    """Mel-channel indices that split the spectrum at the requested edges."""

    import librosa

    centers = librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sample_rate / 2.0)
    bounds = [0]
    for edge_hz in band_edges_hz:
        index = int(np.searchsorted(centers, float(edge_hz)))
        bounds.append(max(bounds[-1] + 1, min(n_mels - 1, index)))
    bounds.append(n_mels)
    return bounds


def _log_period_grid(config: TempoPriorConfig, seconds_per_frame: float) -> tuple[np.ndarray, np.ndarray]:
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


def _resample_columns(surface: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Linearly interpolate a (K, T) surface onto a new time axis."""

    if surface.shape[1] == 0 or source_times.size == 0:
        return np.zeros((surface.shape[0], target_times.size), dtype=np.float64)
    if source_times.size == 1:
        return np.repeat(surface[:, :1], target_times.size, axis=1)
    clipped = np.clip(target_times, source_times[0], source_times[-1])
    upper = np.clip(np.searchsorted(source_times, clipped, side="left"), 1, source_times.size - 1)
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


def _resolve_tempogram_stride(config: TempoPriorConfig, *, target_hop_length: int) -> int:
    """Onset-envelope frames between tempogram columns.

    The surface is resampled onto the beat model's frame grid before anything
    reads it, and the analysis window is eight seconds wide, so columns spaced
    more finely than that target grid are computed and then thrown away.  With
    the shipped hops -- 512 for the model, 128 for the onset envelope -- this is
    one column in four, and the resampled result is unchanged.
    """

    if int(config.tempogram_stride) > 0:
        return int(config.tempogram_stride)
    return max(1, int(target_hop_length) // int(config.onset_hop_length))


def _resolve_torch_device(device: object) -> object | None:
    """The accelerator to run the spectra on, or ``None`` to stay in NumPy.

    Torch on the CPU buys nothing over the NumPy path -- both end up in the same
    FFT library -- so only a non-CPU device switches backends.  These transforms
    are float64 throughout: float32 moves the log-probabilities enough to change
    which period the prior favours, and MPS has no float64 at all.
    """

    if device is None:
        return None
    import torch

    resolved = torch.device(device)
    if resolved.type in ("cpu", "mps"):
        return None
    return resolved


def _interp_weights(query: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Indices and weights reproducing ``np.interp(query, grid, column)``.

    ``left`` and ``right`` are zero, matching the callers below.  Both period
    grids are fixed for a whole track, so the search each interpolation implies
    is done once here rather than once per tempogram column.
    """

    size = int(grid.size)
    upper = np.clip(np.searchsorted(grid, query, side="left"), 1, size - 1)
    lower = upper - 1
    span = grid[upper] - grid[lower]
    fraction = np.where(span > 0.0, (query - grid[lower]) / np.where(span > 0.0, span, 1.0), 0.0)
    inside = ((query >= grid[0]) & (query <= grid[-1])).astype(np.float64)
    return lower, upper, fraction, inside


def _apply_interp(
    matrix: np.ndarray,
    weights: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Interpolate every column of a ``(S, T)`` matrix onto the query grid."""

    lower, upper, fraction, inside = weights
    interpolated = matrix[lower] * (1.0 - fraction)[:, None] + matrix[upper] * fraction[:, None]
    return interpolated * inside[:, None]


def _strided_tempogram(envelope: np.ndarray, *, win_length: int, stride: int) -> np.ndarray:
    """``librosa.feature.tempogram`` evaluated every ``stride`` frames."""

    import librosa
    from librosa import util as librosa_util
    from librosa.filters import get_window

    ac_window = get_window("hann", win_length, fftbins=True)
    frame_count = int(envelope.shape[-1])
    padded = np.pad(envelope, int(win_length // 2), mode="linear_ramp", end_values=[0, 0])
    frames = librosa_util.frame(padded, frame_length=win_length, hop_length=stride)
    frames = frames[..., : int(math.ceil(frame_count / stride))]
    return librosa_util.normalize(
        librosa.autocorrelate(frames * ac_window[:, None], axis=-2),
        norm=np.inf,
        axis=-2,
    )


def _strided_fourier_tempogram(envelope: np.ndarray, *, win_length: int, stride: int) -> np.ndarray:
    """``librosa.feature.fourier_tempogram`` evaluated every ``stride`` frames."""

    import librosa

    return librosa.stft(envelope, n_fft=win_length, hop_length=stride, window="hann", center=True)


def _fourier_period_order(
    *, sample_rate: int, config: TempoPriorConfig, win_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of the Fourier tempogram reordered by ascending period."""

    import librosa

    fourier_bpm = librosa.fourier_tempo_frequencies(
        sr=sample_rate, hop_length=int(config.onset_hop_length), win_length=win_length
    )
    valid = fourier_bpm > 0.0  # drop DC, which carries no period
    periods = 60.0 / fourier_bpm[valid]
    order = np.argsort(periods)
    return np.flatnonzero(valid)[order], periods[order]


def _tempogram_surface(
    envelope: np.ndarray,
    *,
    sample_rate: int,
    config: TempoPriorConfig,
    period_seconds: np.ndarray,
    stride: int = 1,
    weights_cache: dict | None = None,
) -> np.ndarray:
    """Blend autocorrelation and Fourier tempograms on the log-period grid."""

    hop = int(config.onset_hop_length)
    if envelope.size < 4:
        return np.full((period_seconds.size, max(1, envelope.size)), 1.0 / period_seconds.size)
    win_length = _window_frames(config, sample_rate)
    cache = {} if weights_cache is None else weights_cache

    autocorrelation = np.asarray(
        _strided_tempogram(envelope, win_length=win_length, stride=stride),
        dtype=np.float64,
    )
    if "acf" not in cache:
        lag_seconds = np.arange(autocorrelation.shape[0]) * hop / float(sample_rate)
        cache["acf"] = _interp_weights(period_seconds, lag_seconds)
    blended = _normalize_surface(_apply_interp(autocorrelation, cache["acf"]))
    if config.fourier_weight <= 0.0:
        return blended

    magnitude = np.abs(np.asarray(_strided_fourier_tempogram(envelope, win_length=win_length, stride=stride))).astype(
        np.float64
    )
    if "fourier" not in cache:
        rows, sorted_periods = _fourier_period_order(sample_rate=sample_rate, config=config, win_length=win_length)
        cache["fourier"] = None if rows.size == 0 else (rows, _interp_weights(period_seconds, sorted_periods))
    if cache["fourier"] is None:
        return blended
    rows, fourier_weights = cache["fourier"]
    fourier_surface = _normalize_surface(_apply_interp(magnitude[rows], fourier_weights))

    frames = min(blended.shape[1], fourier_surface.shape[1])
    # A geometric mean keeps only what both transforms agree on. Their octave
    # ghosts sit at different relative heights, so the product suppresses them
    # while the true pulse, which both rank highly, survives.
    return np.exp(
        (1.0 - config.fourier_weight) * np.log(blended[:, :frames] + _EPSILON)
        + config.fourier_weight * np.log(fourier_surface[:, :frames] + _EPSILON)
    )


def _normalize_surface_torch(surface):
    """:func:`_normalize_surface` for a ``(..., K, T)`` torch tensor."""

    import torch

    positive = surface.clamp_min(0.0)
    totals = positive.sum(dim=-2, keepdim=True)
    uniform = torch.full_like(positive, 1.0 / max(1, positive.shape[-2]))
    return torch.where(totals > _EPSILON, positive / totals.clamp_min(_EPSILON), uniform)


def _interp_weights_torch(weights, *, device, dtype):
    import torch

    lower, upper, fraction, inside = weights
    return (
        torch.as_tensor(lower, device=device, dtype=torch.long),
        torch.as_tensor(upper, device=device, dtype=torch.long),
        torch.as_tensor(fraction, device=device, dtype=dtype),
        torch.as_tensor(inside, device=device, dtype=dtype),
    )


def _apply_interp_torch(matrix, weights):
    """Interpolate along the last axis of a ``(..., S)`` tensor."""

    lower, upper, fraction, inside = weights
    interpolated = matrix.index_select(-1, lower) * (1.0 - fraction) + matrix.index_select(-1, upper) * fraction
    return interpolated * inside


def _tempogram_surfaces_torch(
    bands: np.ndarray,
    *,
    sample_rate: int,
    config: TempoPriorConfig,
    period_seconds: np.ndarray,
    stride: int,
    device,
):
    """:func:`_tempogram_surface` for every band, on an accelerator.

    Chunked along time: the spectra behind these surfaces run to gigabytes for a
    full track, and the columns are independent, so a bounded slice at a time
    gives the same numbers in a fraction of the memory.
    """

    import scipy.fft
    import torch
    from librosa.filters import get_window

    dtype = torch.float64
    win_length = _window_frames(config, sample_rate)
    hop = int(config.onset_hop_length)
    frame_count = int(bands.shape[-1])
    column_count = int(math.ceil(frame_count / stride))
    transform_length = int(scipy.fft.next_fast_len(2 * win_length - 1, real=True))
    smallest_positive = float(np.finfo(np.float64).tiny)

    window = torch.as_tensor(get_window("hann", win_length, fftbins=True), dtype=dtype, device=device)
    lag_seconds = np.arange(win_length) * hop / float(sample_rate)
    acf_weights = _interp_weights_torch(_interp_weights(period_seconds, lag_seconds), device=device, dtype=dtype)
    rows, sorted_periods = _fourier_period_order(sample_rate=sample_rate, config=config, win_length=win_length)
    use_fourier = config.fourier_weight > 0.0 and rows.size > 0
    if use_fourier:
        row_index = torch.as_tensor(rows, device=device, dtype=torch.long)
        fourier_weights = _interp_weights_torch(
            _interp_weights(period_seconds, sorted_periods), device=device, dtype=dtype
        )

    surfaces = []
    for band in bands:
        padded = torch.as_tensor(
            np.pad(band, int(win_length // 2), mode="linear_ramp", end_values=[0, 0]),
            dtype=dtype,
            device=device,
        )
        columns = []
        for start in range(0, column_count, int(config.torch_chunk_frames)):
            stop = min(start + int(config.torch_chunk_frames), column_count)
            span = padded[start * stride : (stop - 1) * stride + win_length]
            frames = span.unfold(0, win_length, stride) * window
            spectrum = torch.fft.rfft(frames, n=transform_length, dim=-1)
            power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
            del spectrum, frames
            autocorrelation = torch.fft.irfft(power, n=transform_length, dim=-1)[..., :win_length]
            del power
            # librosa.util.normalize leaves columns below float tiny alone.
            length = autocorrelation.abs().amax(dim=-1, keepdim=True)
            autocorrelation = autocorrelation / torch.where(length < smallest_positive, torch.ones_like(length), length)
            columns.append(_apply_interp_torch(autocorrelation, acf_weights))
            del autocorrelation
        blended = _normalize_surface_torch(torch.cat(columns, dim=0).transpose(0, 1))
        del columns
        if not use_fourier:
            surfaces.append(blended)
            continue

        envelope = torch.as_tensor(np.ascontiguousarray(band), dtype=dtype, device=device)
        spectrum = torch.stft(
            envelope,
            n_fft=win_length,
            hop_length=stride,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
            normalized=False,
        )
        magnitude = spectrum.abs().index_select(0, row_index).transpose(0, 1)
        del spectrum
        fourier_surface = _normalize_surface_torch(_apply_interp_torch(magnitude, fourier_weights).transpose(0, 1))
        del magnitude
        frames_kept = min(blended.shape[-1], fourier_surface.shape[-1])
        surfaces.append(
            torch.exp(
                (1.0 - config.fourier_weight) * torch.log(blended[..., :frames_kept] + _EPSILON)
                + config.fourier_weight * torch.log(fourier_surface[..., :frames_kept] + _EPSILON)
            )
        )

    frames_kept = min(surface.shape[-1] for surface in surfaces)
    return torch.stack([surface[..., :frames_kept] for surface in surfaces])


def compute_tempo_prior(
    waveform: np.ndarray | None = None,
    *,
    sample_rate: int,
    target_hop_length: int,
    target_frame_count: int,
    config: TempoPriorConfig = TempoPriorConfig(),
    onset_bands: np.ndarray | None = None,
    device: object | None = None,
) -> TempoPrior:
    """Build a per-frame quarter-note period distribution from the waveform.

    ``onset_bands`` skips the onset front end when the caller already ran it --
    :func:`compute_pulse_curve` needs the same envelopes, and computing them
    twice is pure duplication.  ``device`` runs the spectra on an accelerator.
    """

    seconds_per_frame = float(target_hop_length) / float(sample_rate)
    period_seconds, log_period_frames = _log_period_grid(config, seconds_per_frame)
    if onset_bands is None:
        if waveform is None:
            raise ValueError("compute_tempo_prior needs either waveform or onset_bands")
        onset_bands, _mixed = compute_onset_envelopes(waveform, sample_rate=sample_rate, config=config)
    bands = np.asarray(onset_bands, dtype=np.float64)
    stride = _resolve_tempogram_stride(config, target_hop_length=target_hop_length)
    weights = np.asarray(config.band_weights, dtype=np.float64)[: bands.shape[0]]

    torch_device = _resolve_torch_device(device)
    accumulated = None
    if torch_device is not None and bands.shape[-1] >= 4:
        import torch

        try:
            surfaces = _tempogram_surfaces_torch(
                bands,
                sample_rate=sample_rate,
                config=config,
                period_seconds=period_seconds,
                stride=stride,
                device=torch_device,
            )
            applied_weight = float(weights[weights > 0.0].sum())
            band_weights = torch.as_tensor(weights, device=surfaces.device, dtype=surfaces.dtype)
            accumulated = (band_weights[:, None, None] * torch.log(surfaces + _EPSILON)).sum(dim=0) / max(
                applied_weight, _EPSILON
            )
            accumulated = accumulated.double().cpu().numpy()
            del surfaces
        except RuntimeError as error:  # out of memory is a RuntimeError
            # The accelerator shares its memory with whatever else the pipeline
            # left resident.  Losing it costs speed, not the prior itself.
            print(f"Warning: tempo prior fell back to CPU: {error}")
            accumulated = None

    if accumulated is None:
        applied_weight = 0.0
        weights_cache: dict = {}
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
                    stride=stride,
                    weights_cache=weights_cache,
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
        accumulated = (
            accumulated
            + config.octave_prior_weight * (-0.5 * (octaves / float(config.octave_prior_sigma_octaves)) ** 2)[:, None]
        )

    accumulated = accumulated / float(config.temperature)
    accumulated -= accumulated.max(axis=0, keepdims=True)
    accumulated -= np.log(np.exp(accumulated).sum(axis=0, keepdims=True) + _EPSILON)

    source_times = np.arange(accumulated.shape[1]) * stride * config.onset_hop_length / float(sample_rate)
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


def _plp_torch(
    onset_envelope: np.ndarray,
    *,
    sample_rate: int,
    config: TempoPriorConfig,
    device,
) -> np.ndarray:
    """``librosa.beat.plp`` on an accelerator, chunked along time.

    PLP needs its hop=1 resolution, so there is no stride to take here; the
    saving is the transform itself.  Each chunk carries a ``win_length`` margin
    on both sides, which is the whole reach of the hop=1 overlap-add, so the
    interior it contributes is exact.
    """

    import librosa
    import torch
    from librosa.filters import get_window

    dtype = torch.float64
    win_length = _window_frames(config, sample_rate)
    frame_count = int(onset_envelope.shape[-1])
    smallest_positive = float(np.finfo(np.float64).tiny)

    window = torch.as_tensor(get_window("hann", win_length, fftbins=True), dtype=dtype, device=device)
    envelope = torch.as_tensor(np.ascontiguousarray(onset_envelope), dtype=dtype, device=device)
    tempo_bpm = torch.as_tensor(
        librosa.fourier_tempo_frequencies(
            sr=sample_rate,
            hop_length=int(config.onset_hop_length),
            win_length=win_length,
        ),
        dtype=dtype,
        device=device,
    )
    out_of_range = (tempo_bpm < float(config.min_quarter_bpm)) | (tempo_bpm > float(config.max_quarter_bpm))

    margin = int(win_length)
    chunk = int(config.torch_chunk_frames) * 4
    pieces = []
    for start in range(0, frame_count, chunk):
        stop = min(start + chunk, frame_count)
        low = max(0, start - margin)
        high = min(frame_count, stop + margin)
        spectrum = torch.stft(
            envelope[low:high],
            n_fft=win_length,
            hop_length=1,
            win_length=win_length,
            window=window,
            center=True,
            pad_mode="constant",
            return_complex=True,
            normalized=False,
        )
        spectrum[out_of_range, :] = 0
        magnitude = torch.log1p(1e6 * spectrum.abs())
        spectrum[magnitude < magnitude.amax(dim=-2, keepdim=True)] = 0
        del magnitude
        # NumPy orders complex numbers lexicographically, so librosa's
        # ``np.abs(ftgram.max(axis=-2))`` is the magnitude of the entry with the
        # largest real part rather than the one with the largest magnitude.
        lexicographic = spectrum.real.argmax(dim=-2, keepdim=True)
        spectrum = spectrum / (smallest_positive**0.5 + spectrum.gather(-2, lexicographic).abs())
        pulse = torch.istft(
            spectrum,
            n_fft=win_length,
            hop_length=1,
            win_length=win_length,
            window=window,
            center=True,
            length=high - low,
            normalized=False,
            return_complex=False,
        )
        del spectrum
        pieces.append(pulse[start - low : stop - low].clone())
        del pulse

    pulse = torch.cat(pieces).clamp_min(0.0)
    peak = pulse.abs().amax()
    if float(peak) > smallest_positive:
        pulse = pulse / peak
    return pulse.double().cpu().numpy()


def compute_pulse_curve(
    waveform: np.ndarray | None = None,
    *,
    sample_rate: int,
    config: TempoPriorConfig = TempoPriorConfig(),
    onset_envelope: np.ndarray | None = None,
    device: object | None = None,
) -> tuple[np.ndarray, float]:
    """Predominant Local Pulse at the onset hop, with its hop in seconds.

    PLP fits a sinusoid at the locally dominant tempo, so it is continuous in
    phase and carries beat placement far more finely than the 23 ms frames the
    beat network is quantised to.  It is deliberately *not* resampled onto that
    frame grid: the extra resolution is the whole reason to compute it.

    ``onset_envelope`` reuses the mixdown :func:`compute_tempo_prior` already
    built; ``device`` runs the hop=1 transforms on an accelerator.
    """

    import librosa

    hop_seconds = float(config.onset_hop_length) / float(sample_rate)
    if onset_envelope is None:
        if waveform is None:
            raise ValueError("compute_pulse_curve needs either waveform or onset_envelope")
        _bands, onset_envelope = compute_onset_envelopes(waveform, sample_rate=sample_rate, config=config)
    mixed = np.asarray(onset_envelope, dtype=np.float64)
    if mixed.size < 4:
        return np.zeros(0, dtype=np.float64), hop_seconds

    torch_device = _resolve_torch_device(device)
    if torch_device is not None:
        try:
            return (
                _plp_torch(mixed, sample_rate=sample_rate, config=config, device=torch_device),
                hop_seconds,
            )
        except RuntimeError as error:  # out of memory is a RuntimeError
            print(f"Warning: pulse curve fell back to CPU: {error}")

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
