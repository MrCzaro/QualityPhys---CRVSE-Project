"""Stage 2 of the VitalVideos variability run, on synthetic beats with known timing.

The reference's beats follow a breathing-modulated rhythm; a camera's are the same beats moved
by a constant offset and given known jitter, so the clock it belongs on, the variability it
should recover, and the shares and intervals the rule reads all have known answers.
"""
import numpy as np
import pytest

from app.live_vitals.scripts.check_vv_hrv import (CAMERA_COLUMNS, ECG_SDNN_CUTS_MS, HRV_MARGIN,
                                                  camera_columns, hits, hrv_earned, hypotheses,
                                                  not_run, on_clock, paired_bootstrap,
                                                  secondaries, share, tercile)


def rhythm(n=75, mean=0.8, swing=0.04, breath_hz=0.25, jitter=0.01, seed=0):
    """Beat times from 0.5 s: a period swung by breathing, plus beat-to-beat variation."""
    rng = np.random.default_rng(seed)
    t, times = 0.5, []
    for _ in range(n):
        times.append(t)
        t += mean + swing * np.sin(2 * np.pi * breath_hz * t) + jitter * rng.standard_normal()
    return np.array(times)


def references(beats, breath_hz=0.25):
    """A recording whose ECG, contact PPG and oximeter all beat at `beats`, on one clock."""
    end = float(beats[-1] + 0.5)
    return dict(upstrokes=beats, r_times=beats, cms=beats, t_end=end, cms_end=end,
                breath_hz=breath_hz)


def test_the_reference_scored_as_a_camera_reads_zero():
    beats = rhythm()
    columns = camera_columns("self", beats, references(beats))
    assert set(columns) == {f"self_{name}" for name in CAMERA_COLUMNS}
    assert abs(columns["self_offset_ms"]) < 1e-9
    assert columns["self_sdnn_error_ppg"] == pytest.approx(0.0, abs=1e-12)
    assert columns["self_rmssd_error_ppg"] == pytest.approx(0.0, abs=1e-12)
    assert columns["self_matched"] == 1.0
    assert columns["self_timing_ms"] == pytest.approx(0.0, abs=1e-9)


def test_the_clock_finds_a_constant_offset_and_timing_reads_the_jitter():
    beats = rhythm(seed=1)
    camera = beats - 0.137 + 0.004 * np.random.default_rng(2).standard_normal(beats.size)
    offset, margin = on_clock(camera, beats)
    assert offset == pytest.approx(0.137, abs=0.005)
    assert margin > 20.0  # a whole-beat slip fits far worse
    columns = camera_columns("v2", camera, references(beats))
    assert 1.0 < columns["v2_timing_ms"] < 6.0
    assert abs(columns["v2_sdnn_error_ppg"]) < 0.05
    assert columns["v2_scored_ppg"] and columns["v2_scored_ecg"]


def test_jitter_inflates_rmssd_more_than_sdnn_and_shows_as_implied_jitter():
    beats = rhythm(seed=3)
    camera = beats + 0.015 * np.random.default_rng(4).standard_normal(beats.size)
    columns = camera_columns("v2", camera, references(beats))
    assert columns["v2_rmssd_error_ppg"] > columns["v2_sdnn_error_ppg"] > 0.0
    assert 8.0 < columns["v2_jitter_sdnn_ms"] < 25.0


def test_a_camera_with_too_few_intervals_is_a_miss_that_stays_in_the_denominator():
    beats = rhythm(seed=5)
    columns = camera_columns("v2", beats[:25], references(beats))
    assert not columns["v2_scored_ppg"] and np.isnan(columns["v2_sdnn_error_ppg"])
    rows = [{"e": 0.05}, {"e": columns["v2_sdnn_error_ppg"]}, {"e": -0.08}, {"e": 0.30}]
    m = share(rows, "e")
    assert (m["n"], m["k"], m["n_scored"], m["within_10"]) == (4, 2, 3, 0.5)
    assert list(hits([0.1, -0.1, 0.1000001, float("nan")])) == [True, True, False, False]


def test_an_estimator_not_run_is_never_scored():
    columns = not_run("v3")
    assert set(columns) == {f"v3_{name}" for name in CAMERA_COLUMNS}
    assert columns["v3_scored_ppg"] is False and np.isnan(columns["v3_sdnn_error_ppg"])


def test_the_paired_bootstrap_is_paired_and_fixed_by_its_seed():
    a = np.random.default_rng(6).random(200) < 0.6
    same = paired_bootstrap(a, a.copy())
    assert same["difference"] == same["low"] == same["high"] == 0.0
    b = a.copy()
    b[:20] = ~b[:20]
    first, again = paired_bootstrap(a, b), paired_bootstrap(a, b)
    assert first == again
    assert first["low"] <= first["difference"] <= first["high"]


def test_the_word_hrv_needs_the_prv_bar_and_all_three_conditions():
    assert hrv_earned("confirmed", "confirmed", "confirmed", -0.05)
    assert hrv_earned("confirmed", "confirmed", "confirmed", HRV_MARGIN)
    assert not hrv_earned("confirmed", "confirmed", "confirmed", -0.11)
    assert not hrv_earned("inconclusive", "confirmed", "confirmed", 0.0)
    assert not hrv_earned("confirmed", "inconclusive", "confirmed", 0.0)
    assert not hrv_earned("confirmed", "confirmed", "not confirmed", 0.0)
    assert not hrv_earned("confirmed", "confirmed", "confirmed", None)


def test_the_secondaries_follow_a_known_breathing_swing():
    ends = 0.8 + 0.8 * np.arange(70)
    kept = np.ones(ends.size, dtype=bool)

    def swung(amplitude_s, at=ends):
        return 0.8 + amplitude_s * np.sin(2 * np.pi * 0.25 * at)

    one = secondaries(ends, swung(0.03), kept, 0.0, 60.0, 0.25)
    two = secondaries(ends, swung(0.06), kept, 0.0, 60.0, 0.25)
    assert one["breath"] == pytest.approx(30.0, rel=0.02)
    assert two["hf"] / one["hf"] == pytest.approx(4.0, rel=0.02)
    fewer = kept.copy()
    fewer[::3] = False  # the camera drops a third of the intervals over the same seconds
    assert secondaries(ends, swung(0.03), fewer, 0.0, 60.0, 0.25)["breath"] == pytest.approx(
        30.0, rel=0.02)
    dense = 0.4 + 0.4 * np.arange(140)  # twice the beats over the same seconds
    denser = secondaries(dense, swung(0.03, dense), np.ones(dense.size, dtype=bool), 0.0, 60.0,
                         0.25)
    assert denser["hf"] / one["hf"] == pytest.approx(1.0, abs=0.05)
    steady = secondaries(ends, np.full(ends.size, 0.8), kept, 0.0, 60.0, float("nan"))
    assert steady["pnn50"] == 0.0 and np.isnan(steady["breath"])
    alternating = secondaries(ends, 0.8 + 0.06 * (np.arange(ends.size) % 2), kept, 0.0, 60.0, 0.25)
    assert alternating["pnn50"] == 100.0


def test_terciles_put_a_value_on_a_cut_in_the_lower_one():
    low, high = ECG_SDNN_CUTS_MS
    assert tercile(low, ECG_SDNN_CUTS_MS) == "lowest"
    assert tercile(np.nextafter(low, np.inf), ECG_SDNN_CUTS_MS) == "middle"
    assert tercile(high, ECG_SDNN_CUTS_MS) == "middle"
    assert tercile(high + 1.0, ECG_SDNN_CUTS_MS) == "highest"
    assert tercile(float("nan"), ECG_SDNN_CUTS_MS) == "n/a"


def test_the_hypotheses_are_judged_on_point_estimates():
    def m(x):
        return dict(within_10=x)

    summary = dict(
        columns=dict(v2=dict(ppg=dict(sdnn=m(0.55))), pos=dict(ppg=dict(sdnn=m(0.45)))),
        strata=dict(ppg=dict(
            ecg_sdnn_tercile=dict(lowest=m(0.40), middle=m(0.55), highest=m(0.70)),
            site_group={"CH*": m(0.40), "CRC/SCH/BET": m(0.60), "MOL/RING": m(0.55)})),
        rmssd_trigger=dict(share=0.20), hrv=dict(earned=False))
    assert {name: h["verdict"] for name, h in hypotheses(summary).items()} == dict(
        H1="consistent", H2b="consistent", H3="consistent", H4="consistent",
        H5="consistent", H6="consistent")
    summary["strata"]["ppg"]["site_group"]["MOL/RING"] = m(0.35)  # CH* is no longer the weakest
    summary["hrv"]["earned"] = True
    verdicts = hypotheses(summary)
    assert verdicts["H4"]["verdict"] == verdicts["H2b"]["verdict"] == "contradicted"