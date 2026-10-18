"""Heart-rate readout and window gating, on synthetic signals with known answers."""
import numpy as np
import pytest

from app.live_vitals import config
from app.live_vitals.estimators.base import aggregate_windows
from app.live_vitals.signal.hr import hr_from_bvp

FPS = 30.0


def pulse(bpm, n=config.CLIP_LEN, noise=0.0, seed=0):
    """A sinusoidal pulse at bpm, sampled at FPS, with optional white noise."""
    x = np.sin(2 * np.pi * bpm / 60.0 * np.arange(n) / FPS)
    if noise:
        x = x + np.random.default_rng(seed).normal(0.0, noise, n)
    return x


def aggregate(rates, confidences=None, n_attempted=None, starts=None):
    """Gates per-window readings the way every estimator does."""
    confidences = [0.9] * len(rates) if confidences is None else confidences
    waves = [np.zeros(config.CLIP_LEN) for _ in rates]
    return aggregate_windows("heart_rate", "bpm", list(rates), list(confidences), waves, FPS,
                             len(rates) if n_attempted is None else n_attempted,
                             window_starts=starts)


# ---------------------------------------------------------------- readout --

@pytest.mark.parametrize("bpm", [48, 60, 72, 90, 120, 150, 170])
def test_readout_recovers_a_clean_pulse(bpm):
    reading = hr_from_bvp(pulse(bpm), FPS)
    assert reading["status"] == "fundamental"
    assert reading["hr_bpm"] == pytest.approx(bpm, abs=1.0)
    assert reading["confidence"] >= config.MIN_CONFIDENCE


def test_a_peak_below_the_band_is_not_a_reading():
    # 30 bpm lies below the band, so the in-band maximum is the band's own edge.
    reading = hr_from_bvp(pulse(30), FPS)
    assert reading["status"] == "band_edge"
    assert np.isnan(reading["hr_bpm"]) and reading["confidence"] == 0.0


@pytest.mark.parametrize("signal", [np.zeros(config.CLIP_LEN), np.full(config.CLIP_LEN, np.nan),
                                    np.ones(8)])
def test_degenerate_input_is_not_a_reading(signal):
    reading = hr_from_bvp(signal, FPS)
    assert reading["status"] == "degenerate" and np.isnan(reading["hr_bpm"])


def test_noise_lowers_confidence():
    clean = hr_from_bvp(pulse(72), FPS)["confidence"]
    noisy = hr_from_bvp(pulse(72, noise=2.0), FPS)["confidence"]
    assert noisy < clean


# ---------------------------------------------------------------- gating --

def test_steady_windows_give_an_ok_reading():
    result = aggregate([72.0, 72.5, 71.5, 72.2, 71.8, 72.1])
    assert result.status == "ok"
    assert result.value == pytest.approx(72.05, abs=0.01)
    assert result.detail["n_windows"] == 6 and all(result.detail["window_kept"])


def test_no_window_with_a_peak_is_no_estimate():
    result = aggregate([], n_attempted=6)
    assert result.status == "no_estimate" and np.isnan(result.value)
    assert result.detail["n_total"] == 6 and result.detail["n_no_peak"] == 6


def test_too_few_windows_are_refused():
    result = aggregate([72.0] * (config.MIN_USABLE_WINDOWS - 1))
    assert result.status == "insufficient_quality" and np.isnan(result.value)


def test_low_confidence_windows_are_dropped():
    low = config.MIN_CONFIDENCE - 0.1
    result = aggregate([72.0] * 6, confidences=[0.9, 0.9, 0.9, 0.9, low, low])
    assert result.detail["window_kept"] == [True, True, True, True, False, False]
    assert result.status == "ok" and result.value == 72.0


def test_an_octave_error_is_dropped_by_the_mad_filter():
    result = aggregate([72.0, 72.4, 71.6, 144.0, 72.2, 71.9])
    assert result.detail["window_kept"][3] is False
    assert result.value == pytest.approx(72.0, abs=0.5)


def test_a_low_surviving_share_degrades_the_reading():
    # Four of ten attempted windows: above the reportable floor, below the usable share.
    result = aggregate([72.0] * 4, n_attempted=10)
    assert config.MIN_REPORTABLE_FRACTION <= 0.4 < config.MIN_USABLE_FRACTION
    assert result.status == "degraded_capture" and result.value == 72.0


def test_below_the_reportable_share_there_is_no_reading():
    result = aggregate([72.0] * 4, n_attempted=20)
    assert 0.2 < config.MIN_REPORTABLE_FRACTION
    assert result.status == "insufficient_quality" and np.isnan(result.value)


def test_a_wide_spread_reads_unstable():
    # Every window survives the MAD filter, but their spread exceeds the backstop.
    result = aggregate([45.0, 60.0, 75.0, 90.0, 105.0, 120.0])
    assert all(result.detail["window_kept"])
    assert result.detail["spread_bpm"] > config.MAX_HR_SPREAD_BPM
    assert result.status == "unstable"


def test_window_starts_must_match_the_readings():
    with pytest.raises(ValueError):
        aggregate([72.0, 72.0, 72.0], starts=[0, 80])