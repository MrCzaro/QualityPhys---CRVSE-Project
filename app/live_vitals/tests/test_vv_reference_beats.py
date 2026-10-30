"""Beat references of the VitalVideos variability run, on synthetic signals with known beats.

The ECG is a sum of P, Q, R, S and T waves at known R times, with baseline wander and noise;
the pulse wave rises and falls at known times after each R peak. Beat times follow a
breathing-modulated rhythm, so the variability the detectors must recover is known.
"""
import numpy as np
import pytest

from app.live_vitals.scripts.check_vv_hrv import (outcome, ppg_upstrokes, r_peaks,
                                                  relative_error, transit_delays, variability,
                                                  wilson)

FS = 500.0
TRANSIT_S = 0.25


def beat_times(n=70, mean=0.82, breathing=0.04, jitter=0.012, seed=0):
    """Beat times starting at 1 s: a mean period swung by breathing at 0.25 Hz, plus jitter."""
    rng = np.random.default_rng(seed)
    t, times = 1.0, []
    for _ in range(n):
        times.append(t)
        t += mean + breathing * np.sin(2 * np.pi * 0.25 * t) + jitter * rng.standard_normal()
    return np.array(times)


def synthetic_ecg(r_times, polarity=1.0, t_wave=0.3, noise=0.02, seed=1):
    """A lead with P, Q, R, S and T waves around each R time; the R wave's peak sits at R."""
    t = np.arange(int((r_times[-1] + 1.0) * FS)) / FS
    x = np.zeros_like(t)
    waves = ((-0.20, 0.12, 0.025), (-0.025, -0.15, 0.008), (0.0, 1.0, 0.009),
             (0.025, -0.25, 0.008), (0.25, t_wave, 0.045))
    for r in r_times:
        for offset, amplitude, width in waves:
            x += amplitude * np.exp(-0.5 * ((t - r - offset) / width) ** 2)
    rng = np.random.default_rng(seed)
    x += 0.2 * np.sin(2 * np.pi * 0.3 * t) + noise * rng.standard_normal(t.size)
    return polarity * x


def synthetic_ppg(r_times, noise=0.005, seed=2):
    """A pulse wave whose systolic rise begins TRANSIT_S after each R time."""
    t = np.arange(int((r_times[-1] + 1.5) * FS)) / FS
    x = np.zeros_like(t)
    for r in r_times:
        u = t - r - TRANSIT_S
        x += np.exp(-0.5 * ((u - 0.12) / 0.05) ** 2) + 0.4 * np.exp(-0.5 * ((u - 0.38) / 0.08) ** 2)
    rng = np.random.default_rng(seed)
    return x + noise * rng.standard_normal(t.size)


def timing_errors(found, truth):
    """Signed error of each true beat's nearest found beat, in ms; asserts nothing is missed."""
    index = np.clip(np.searchsorted(found, truth), 1, len(found) - 1)
    nearest = np.where(np.abs(found[index] - truth) < np.abs(found[index - 1] - truth),
                       found[index], found[index - 1])
    return (nearest - truth) * 1000.0


@pytest.mark.parametrize("polarity", [1.0, -1.0])
def test_r_peaks_time_every_beat_to_under_a_millisecond(polarity):
    truth = beat_times()
    found, info = r_peaks(synthetic_ecg(truth, polarity=polarity), FS)
    assert info["polarity"] == int(polarity)
    assert len(found) == len(truth)
    errors = timing_errors(found, truth)
    assert np.max(np.abs(errors - np.median(errors))) < 1.0


def test_tall_t_waves_are_not_counted_as_beats():
    truth = beat_times(seed=3)
    found, _ = r_peaks(synthetic_ecg(truth, t_wave=0.8), FS)
    assert len(found) == len(truth)


def test_r_peak_variability_matches_the_truth():
    truth = beat_times(seed=4)
    found, _ = r_peaks(synthetic_ecg(truth), FS)
    end = truth[-1] + 1.0
    got, want = variability(found, end), variability(truth, end)
    assert abs(relative_error(got["sdnn"], want["sdnn"])) < 0.01
    assert abs(relative_error(got["rmssd"], want["rmssd"])) < 0.02


def test_upstrokes_follow_the_pulse_with_a_constant_delay():
    r_times = beat_times(seed=5)
    found = ppg_upstrokes(synthetic_ppg(r_times), FS)
    errors = timing_errors(found, r_times + TRANSIT_S)
    inner = errors[2:-2]  # the band-pass settles within a beat or two of either end
    assert np.std(inner) < 3.0
    end = r_times[-1] + 1.0
    got, want = variability(found, end), variability(r_times, end)
    assert abs(relative_error(got["sdnn"], want["sdnn"])) < 0.05


def test_transit_delays_pair_each_beat_and_mark_a_missing_one():
    r_times = beat_times(n=20, seed=6)
    upstrokes = r_times + TRANSIT_S
    delays = transit_delays(r_times, upstrokes)
    assert np.allclose(delays, TRANSIT_S)
    delays = transit_delays(r_times, np.delete(upstrokes, 7))
    assert np.isnan(delays[7]) and np.isfinite(np.delete(delays, 7)).all()


def test_a_reference_against_itself_scores_exactly_zero():
    truth = beat_times(seed=7)
    end = truth[-1] + 1.0
    a, b = variability(truth, end), variability(truth.copy(), end)
    assert relative_error(a["sdnn"], b["sdnn"]) == 0.0
    assert relative_error(a["rmssd"], b["rmssd"]) == 0.0


def test_wilson_interval_and_the_rules_three_outcomes():
    low, high = wilson(8, 12)
    assert low == pytest.approx(0.391, abs=0.001) and high == pytest.approx(0.862, abs=0.001)
    assert wilson(135, 238)[0] >= 0.5 > wilson(134, 238)[0]
    assert outcome(0.55, 0.70) == "confirmed"
    assert outcome(0.30, 0.45) == "not confirmed"
    assert outcome(0.45, 0.60) == "inconclusive"
    assert outcome(float("nan"), float("nan")) == "not estimable"