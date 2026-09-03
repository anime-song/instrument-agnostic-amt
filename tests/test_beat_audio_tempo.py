"""Tests for the audio tempo prior and the beat refinement passes."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from instrument_agnostic_amt.beat_chord.decoding.audio_refinement import (
    rescale_decode_to_level,
)
from instrument_agnostic_amt.beat_chord.decoding.audio_tempo import (
    TempoPrior,
    TempoPriorConfig,
    _resolve_tempogram_stride,
    compute_onset_envelopes,
    compute_pulse_curve,
    compute_tempo_prior,
)
from instrument_agnostic_amt.beat_chord.decoding.beat_grid import (
    BeatGridDPConfig,
    BeatGridDecodeResult,
    MeterGridSegment,
    _tempo_transition_penalty,
)
from instrument_agnostic_amt.beat_chord.decoding.beat_refine import (
    fit_piecewise_constant_tempo,
    resample_metrical_level,
    rescale_meter,
    score_beat_grid,
    select_metrical_level,
    snap_to_curve,
)

SAMPLE_RATE = 22050
HOP_LENGTH = 512
SECONDS_PER_FRAME = HOP_LENGTH / SAMPLE_RATE


def _click_track(bpm: float, duration_seconds: float = 20.0) -> np.ndarray:
    """A percussive click train at ``bpm``, with a decaying noise burst per beat."""

    rng = np.random.default_rng(0)
    samples = int(duration_seconds * SAMPLE_RATE)
    audio = np.zeros(samples, dtype=np.float32)
    period = 60.0 / bpm
    burst = int(0.02 * SAMPLE_RATE)
    envelope = np.exp(-np.linspace(0.0, 9.0, burst)).astype(np.float32)
    index = 0
    while True:
        start = int(index * period * SAMPLE_RATE)
        if start + burst >= samples:
            break
        audio[start : start + burst] += (
            rng.standard_normal(burst).astype(np.float32) * envelope
        )
        index += 1
    return audio


class TestTempoPrior:
    @pytest.mark.parametrize("bpm", [90.0, 120.0, 150.0])
    def test_recovers_click_tempo_within_two_percent(self, bpm: float) -> None:
        audio = _click_track(bpm)
        frames = len(audio) // HOP_LENGTH
        prior = compute_tempo_prior(
            audio,
            sample_rate=SAMPLE_RATE,
            target_hop_length=HOP_LENGTH,
            target_frame_count=frames,
            config=TempoPriorConfig(),
        )
        estimated_bpm = 60.0 / (prior.best_period_frames() * SECONDS_PER_FRAME)
        # Octave confusion is the decoder's job to resolve, so accept any octave
        # but require the period itself to be accurate.
        folded = estimated_bpm
        while folded > bpm * 1.4:
            folded /= 2.0
        while folded < bpm / 1.4:
            folded *= 2.0
        assert folded == pytest.approx(bpm, rel=0.02)

    def test_is_a_normalised_distribution_per_frame(self) -> None:
        prior = compute_tempo_prior(
            _click_track(120.0),
            sample_rate=SAMPLE_RATE,
            target_hop_length=HOP_LENGTH,
            target_frame_count=len(_click_track(120.0)) // HOP_LENGTH,
            config=TempoPriorConfig(),
        )
        per_frame = np.diff(prior.prefix, axis=0)
        totals = np.exp(per_frame).sum(axis=1)
        assert np.allclose(totals, 1.0, atol=1e-3)

    def test_prefers_the_true_period_over_a_wrong_one(self) -> None:
        audio = _click_track(120.0)
        frames = len(audio) // HOP_LENGTH
        prior = compute_tempo_prior(
            audio,
            sample_rate=SAMPLE_RATE,
            target_hop_length=HOP_LENGTH,
            target_frame_count=frames,
            config=TempoPriorConfig(),
        )
        true_period = 0.5 / SECONDS_PER_FRAME
        detuned_period = (60.0 / 137.0) / SECONDS_PER_FRAME
        assert prior.mean_log_prob(0, frames, true_period) > prior.mean_log_prob(
            0, frames, detuned_period
        )

    def test_lookup_clamps_outside_the_grid(self) -> None:
        prior = TempoPrior(
            log_period_frames=np.log(np.array([10.0, 20.0, 40.0])),
            prefix=np.cumsum(np.zeros((4, 3)) - 1.0, axis=0),
            seconds_per_frame=SECONDS_PER_FRAME,
        )
        assert prior.mean_log_prob(0, 3, 1.0) == pytest.approx(-1.0)
        assert prior.mean_log_prob(0, 3, 1000.0) == pytest.approx(-1.0)
        assert prior.mean_log_prob(2, 2, 20.0) == 0.0
        assert prior.mean_log_prob(0, 3, -1.0) == 0.0

    def test_pulse_curve_keeps_the_onset_hop(self) -> None:
        config = TempoPriorConfig()
        audio = _click_track(120.0, duration_seconds=10.0)
        pulse, hop_seconds = compute_pulse_curve(
            audio, sample_rate=SAMPLE_RATE, config=config
        )
        assert hop_seconds == pytest.approx(config.onset_hop_length / SAMPLE_RATE)
        assert hop_seconds < SECONDS_PER_FRAME
        assert pulse.size > int(10.0 / hop_seconds) * 0.9

    def test_default_stride_matches_the_target_frame_grid(self) -> None:
        config = TempoPriorConfig()
        assert _resolve_tempogram_stride(config, target_hop_length=HOP_LENGTH) == (
            HOP_LENGTH // config.onset_hop_length
        )
        assert _resolve_tempogram_stride(
            TempoPriorConfig(tempogram_stride=3), target_hop_length=HOP_LENGTH
        ) == 3
        # A target grid finer than the onset hop still asks for every column.
        assert _resolve_tempogram_stride(config, target_hop_length=64) == 1

    def test_strided_tempogram_matches_the_dense_one(self) -> None:
        """The columns the stride skips are recovered by the resampling.

        The analysis window is eight seconds wide, so columns 5.8 ms apart are
        near-duplicates; the surface is resampled onto the model's own frame
        grid regardless, which is exactly what the default stride computes.
        """

        audio = _click_track(120.0, duration_seconds=25.0)
        frames = len(audio) // HOP_LENGTH
        arguments = dict(
            sample_rate=SAMPLE_RATE,
            target_hop_length=HOP_LENGTH,
            target_frame_count=frames,
        )
        dense = compute_tempo_prior(
            audio, config=TempoPriorConfig(tempogram_stride=1), **arguments
        )
        strided = compute_tempo_prior(audio, config=TempoPriorConfig(), **arguments)
        assert np.allclose(dense.prefix, strided.prefix, atol=1e-6)

    def test_reuses_precomputed_onset_bands(self) -> None:
        audio = _click_track(120.0, duration_seconds=12.0)
        frames = len(audio) // HOP_LENGTH
        bands, _mixed = compute_onset_envelopes(
            audio, sample_rate=SAMPLE_RATE, config=TempoPriorConfig()
        )
        arguments = dict(
            sample_rate=SAMPLE_RATE,
            target_hop_length=HOP_LENGTH,
            target_frame_count=frames,
            config=TempoPriorConfig(),
        )
        assert np.array_equal(
            compute_tempo_prior(audio, **arguments).prefix,
            compute_tempo_prior(onset_bands=bands, **arguments).prefix,
        )

    def test_requires_a_waveform_or_precomputed_envelope(self) -> None:
        with pytest.raises(ValueError):
            compute_tempo_prior(
                sample_rate=SAMPLE_RATE, target_hop_length=HOP_LENGTH,
                target_frame_count=4,
            )
        with pytest.raises(ValueError):
            compute_pulse_curve(sample_rate=SAMPLE_RATE)

    def test_analyze_audio_builds_the_onset_envelope_once(self, monkeypatch) -> None:
        """Both passes read the same envelopes; building them twice is waste."""

        from instrument_agnostic_amt.beat_chord.decoding import audio_refinement

        calls = 0
        original = audio_refinement.compute_onset_envelopes

        def counted(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(audio_refinement, "compute_onset_envelopes", counted)
        audio = _click_track(120.0, duration_seconds=12.0)
        audio_refinement.analyze_audio(
            audio,
            sample_rate=SAMPLE_RATE,
            hop_length=HOP_LENGTH,
            frame_count=len(audio) // HOP_LENGTH,
        )
        assert calls == 1


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the accelerated path needs CUDA"
)
class TestAcceleratedTempoFrontEnd:
    """The torch path must agree with the NumPy one it replaces."""

    def test_tempo_prior_matches_the_numpy_path(self) -> None:
        audio = _click_track(132.0, duration_seconds=30.0)
        frames = len(audio) // HOP_LENGTH
        arguments = dict(
            sample_rate=SAMPLE_RATE,
            target_hop_length=HOP_LENGTH,
            target_frame_count=frames,
            config=TempoPriorConfig(),
        )
        reference = compute_tempo_prior(audio, **arguments)
        accelerated = compute_tempo_prior(audio, device="cuda", **arguments)
        assert np.allclose(reference.prefix, accelerated.prefix, atol=1e-6)

    def test_pulse_curve_matches_the_librosa_path(self) -> None:
        audio = _click_track(132.0, duration_seconds=30.0)
        config = TempoPriorConfig()
        reference, hop_seconds = compute_pulse_curve(
            audio, sample_rate=SAMPLE_RATE, config=config
        )
        accelerated, accelerated_hop = compute_pulse_curve(
            audio, sample_rate=SAMPLE_RATE, config=config, device="cuda"
        )
        assert accelerated_hop == pytest.approx(hop_seconds)
        assert np.allclose(reference, accelerated, atol=1e-9)

    def test_chunking_does_not_change_the_result(self) -> None:
        audio = _click_track(132.0, duration_seconds=30.0)
        frames = len(audio) // HOP_LENGTH
        arguments = dict(
            sample_rate=SAMPLE_RATE,
            target_hop_length=HOP_LENGTH,
            target_frame_count=frames,
            device="cuda",
        )
        whole = compute_tempo_prior(
            audio, config=TempoPriorConfig(torch_chunk_frames=1 << 20), **arguments
        )
        chunked = compute_tempo_prior(
            audio, config=TempoPriorConfig(torch_chunk_frames=64), **arguments
        )
        assert np.allclose(whole.prefix, chunked.prefix, atol=1e-9)

    def test_cpu_and_mps_devices_stay_on_the_numpy_path(self) -> None:
        audio = _click_track(120.0, duration_seconds=12.0)
        frames = len(audio) // HOP_LENGTH
        arguments = dict(
            sample_rate=SAMPLE_RATE,
            target_hop_length=HOP_LENGTH,
            target_frame_count=frames,
            config=TempoPriorConfig(),
        )
        reference = compute_tempo_prior(audio, **arguments)
        assert np.array_equal(
            reference.prefix, compute_tempo_prior(audio, device="cpu", **arguments).prefix
        )


class TestSnapToCurve:
    def test_recovers_sub_frame_placement(self) -> None:
        hop = 0.0058
        offset = 0.013
        curve = np.zeros(int(20.0 / hop))
        for beat in range(40):
            index = int(round((beat * 0.5 + offset) / hop))
            if 0 < index < curve.size - 1:
                curve[index - 1 : index + 2] += np.array([0.5, 1.0, 0.5])
        truth = np.arange(40) * 0.5 + offset
        quantised = np.round(truth / SECONDS_PER_FRAME) * SECONDS_PER_FRAME

        snapped = snap_to_curve(quantised, curve=curve, curve_hop_seconds=hop)

        assert np.mean(np.abs(snapped - truth)) < np.mean(
            np.abs(quantised - truth)
        ) / 2.0

    def test_output_stays_monotonic(self) -> None:
        rng = np.random.default_rng(1)
        curve = rng.random(4000)
        beats = np.arange(30) * 0.5
        snapped = snap_to_curve(beats, curve=curve, curve_hop_seconds=0.0058)
        assert np.all(np.diff(snapped) >= 0.0)

    def test_flat_curve_leaves_beats_alone(self) -> None:
        beats = np.arange(10) * 0.5
        snapped = snap_to_curve(
            beats, curve=np.zeros(4000), curve_hop_seconds=0.0058
        )
        assert np.allclose(snapped, beats)


class TestPiecewiseTempo:
    def test_constant_tempo_collapses_to_one_segment(self) -> None:
        truth = np.arange(60) * 0.5
        quantised = np.round(truth / SECONDS_PER_FRAME) * SECONDS_PER_FRAME
        fitted, segments = fit_piecewise_constant_tempo(quantised)
        assert len(segments) == 1
        assert segments[0].bpm == pytest.approx(120.0, rel=0.01)
        assert np.max(np.abs(fitted - truth)) < SECONDS_PER_FRAME

    def test_a_real_tempo_change_survives(self) -> None:
        first = np.arange(30) * 0.5
        second = first[-1] + (np.arange(1, 31) * (60.0 / 90.0))
        quantised = (
            np.round(np.concatenate([first, second]) / SECONDS_PER_FRAME)
            * SECONDS_PER_FRAME
        )
        _fitted, segments = fit_piecewise_constant_tempo(quantised)
        assert len(segments) == 2
        assert segments[0].bpm == pytest.approx(120.0, rel=0.02)
        assert segments[1].bpm == pytest.approx(90.0, rel=0.02)

    def test_a_large_penalty_forces_a_single_tempo(self) -> None:
        first = np.arange(30) * 0.5
        second = first[-1] + (np.arange(1, 31) * (60.0 / 90.0))
        _fitted, segments = fit_piecewise_constant_tempo(
            np.concatenate([first, second]), change_penalty_seconds=100.0
        )
        assert len(segments) == 1

    def test_short_input_passes_through(self) -> None:
        beats = np.arange(4) * 0.5
        fitted, segments = fit_piecewise_constant_tempo(beats, min_segment_beats=8)
        assert segments == ()
        assert np.allclose(fitted, beats)


class TestMetricalLevel:
    def test_resampling_scales_the_pulse(self) -> None:
        beats = np.arange(20) * 0.5
        doubled = resample_metrical_level(beats, 2.0)
        halved = resample_metrical_level(beats, 0.5)
        assert np.median(np.diff(doubled)) == pytest.approx(0.25)
        assert np.median(np.diff(halved)) == pytest.approx(1.0)
        assert resample_metrical_level(beats, 1.0) is beats

    def test_audio_evidence_pulls_a_double_time_grid_back(self) -> None:
        audio = _click_track(120.0, duration_seconds=20.0)
        frames = len(audio) // HOP_LENGTH
        prior = compute_tempo_prior(
            audio,
            sample_rate=SAMPLE_RATE,
            target_hop_length=HOP_LENGTH,
            target_frame_count=frames,
            config=TempoPriorConfig(),
        )
        double_time = np.arange(0.0, 19.0, 0.25)
        corrected, ratio = select_metrical_level(
            double_time,
            tempo_prior=prior,
            beat_log_odds=None,
            seconds_per_frame=SECONDS_PER_FRAME,
        )
        assert ratio == pytest.approx(0.5)
        assert np.median(np.diff(corrected)) == pytest.approx(0.5, rel=0.01)

    def test_a_large_margin_keeps_the_decoder_choice(self) -> None:
        audio = _click_track(120.0, duration_seconds=20.0)
        frames = len(audio) // HOP_LENGTH
        prior = compute_tempo_prior(
            audio,
            sample_rate=SAMPLE_RATE,
            target_hop_length=HOP_LENGTH,
            target_frame_count=frames,
            config=TempoPriorConfig(),
        )
        double_time = np.arange(0.0, 19.0, 0.25)
        _corrected, ratio = select_metrical_level(
            double_time,
            tempo_prior=prior,
            beat_log_odds=None,
            seconds_per_frame=SECONDS_PER_FRAME,
            margin=1e6,
        )
        assert ratio == pytest.approx(1.0)

    def test_scoring_needs_at_least_three_beats(self) -> None:
        assert score_beat_grid(
            np.zeros(1),
            tempo_prior=None,
            beat_log_odds=None,
            seconds_per_frame=SECONDS_PER_FRAME,
        ) == -float("inf")


class TestTempoTransitionPenalty:
    def _config(self, **overrides: float) -> BeatGridDPConfig:
        return BeatGridDPConfig(
            sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH, **overrides
        )

    def test_default_free_band_still_costs_nothing(self) -> None:
        config = self._config()
        assert _tempo_transition_penalty(100.0, 105.0, config) == 0.0

    def test_change_penalty_applies_once_past_the_free_band(self) -> None:
        config = self._config(tempo_free_ratio=1.015, tempo_change_penalty=3.0)
        small = _tempo_transition_penalty(100.0, 102.0, config)
        assert small >= 3.0
        # A drift of 2 % and a real change to 3/4 speed differ by the Huber term
        # only; the fixed cost makes many small steps expensive relative to one.
        large = _tempo_transition_penalty(100.0, 133.0, config)
        assert large > small

    def test_octave_jumps_stay_penalised(self) -> None:
        config = self._config()
        octave = _tempo_transition_penalty(100.0, 200.0, config)
        near = _tempo_transition_penalty(100.0, 170.0, config)
        assert octave > near

    def test_penalty_is_symmetric_in_log_ratio(self) -> None:
        config = self._config(tempo_free_ratio=1.015, tempo_change_penalty=1.0)
        up = _tempo_transition_penalty(100.0, 100.0 * math.e**0.1, config)
        down = _tempo_transition_penalty(100.0, 100.0 / math.e**0.1, config)
        assert up == pytest.approx(down)


class TestDecoderIntegration:
    def test_tempo_prior_defaults_leave_the_decoder_unchanged(self) -> None:
        config = BeatGridDPConfig(sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH)
        assert config.tempo_prior_weight == 0.0
        assert config.tempo_change_penalty == 0.0

    def test_negative_weights_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            BeatGridDPConfig(
                sample_rate=SAMPLE_RATE, hop_length=HOP_LENGTH, tempo_prior_weight=-1.0
            )
        with pytest.raises(ValueError):
            BeatGridDPConfig(
                sample_rate=SAMPLE_RATE,
                hop_length=HOP_LENGTH,
                tempo_change_penalty=-1.0,
            )


class TestMeterRescaling:
    """Restating a decode at another metrical level must keep bars intact."""

    def test_halving_the_pulse_turns_eight_eight_into_four_four(self) -> None:
        assert rescale_meter(8, 8, 0.5) == (4, 4)
        assert rescale_meter(4, 4, 2.0) == (8, 8)

    def test_non_integral_or_odd_denominators_are_refused(self) -> None:
        assert rescale_meter(3, 4, 0.5) is None  # 1.5/2 is not a meter
        assert rescale_meter(4, 4, 1.5) is None  # 6/6 has no MIDI denominator


class TestDecodeRescaling:
    """A restated decode must keep its bars, downbeats and quarter tempo."""

    def test_rescaled_decode_preserves_bars_and_quarter_tempo(self) -> None:
        # Two bars of 8/8, one beat every 10 frames.
        segments = tuple(
            MeterGridSegment(
                start_frame=bar * 80,
                end_frame=(bar + 1) * 80,
                meter_index=0,
                meter_num=8,
                meter_den=8,
                bar_count=1,
                mapped_downbeat_frames=(bar * 80,),
                score=1.0,
                quarter_note_bpm=120.0,
            )
            for bar in range(2)
        )
        decode = BeatGridDecodeResult(
            beat_frames=tuple(range(0, 160, 10)),
            downbeat_frames=(0, 80),
            meter_segments=segments,
            raw_downbeat_candidates=(),
            all_boundary_candidates=(),
            rejected_downbeat_candidates=(),
            inferred_downbeat_frames=(),
            total_score=0.0,
            confidence_margin=None,
        )

        rescaled = rescale_decode_to_level(decode, 0.5)

        assert rescaled is not None
        assert [(s.meter_num, s.meter_den) for s in rescaled.meter_segments] == [
            (4, 4),
            (4, 4),
        ]
        # A bar still spans the same frames and still holds a whole number of
        # beats -- half as many, each twice as long.
        assert [s.start_frame for s in rescaled.meter_segments] == [0, 80]
        assert [s.quarter_note_bpm for s in rescaled.meter_segments] == [120.0, 120.0]
        assert list(rescaled.beat_frames) == list(range(0, 160, 20))

    def test_a_meter_that_cannot_scale_is_refused(self) -> None:
        decode = BeatGridDecodeResult(
            beat_frames=tuple(range(0, 120, 10)),
            downbeat_frames=(0,),
            meter_segments=(
                MeterGridSegment(
                    start_frame=0,
                    end_frame=30,
                    meter_index=0,
                    meter_num=3,
                    meter_den=4,
                    bar_count=1,
                    mapped_downbeat_frames=(0,),
                    score=1.0,
                ),
            ),
            raw_downbeat_candidates=(),
            all_boundary_candidates=(),
            rejected_downbeat_candidates=(),
            inferred_downbeat_frames=(),
            total_score=0.0,
            confidence_margin=None,
        )

        assert rescale_decode_to_level(decode, 0.5) is None
