"""Check 1 of check_hr_robustness on synthetic rows: classes, flags, interval, outcomes, outputs.

No store is read: the benchmark scripts' evaluation functions are replaced by rows built
here, so these tests say nothing about any dataset.
"""
import argparse
import json

import pytest

from app.live_vitals.scripts import check_hr_robustness as robust


def window(recording, ref, pred, kept=True, primary=True):
    return dict(recording=recording, ref_hr=ref, ref_confidence=0.7, ref_kept=True,
                pred_hr=pred, pred_confidence=0.8, pred_kept=kept, primary=primary)


def recordings_with(prefix, n_recordings, rate, good, windows=10):
    """n recordings at one reference rate, each with `good` of its windows right."""
    return [window(f"{prefix}{r}", rate, rate if w < good else rate * 1.5)
            for r in range(n_recordings) for w in range(windows)]


@pytest.mark.parametrize("rate, expected", [
    (45.0, "low"), (59.99, "low"), (60.0, "gap"), (64.99, "gap"), (65.0, "typical"),
    (89.99, "typical"), (90.0, "high"), (150.0, "high"), (float("nan"), None)])
def test_rate_classes_follow_part0(rate, expected):
    assert robust.rate_class(rate) == expected


def test_window_flags_and_primary_windows_only():
    frame = robust.scored_windows([
        window("a", 50.0, 54.9),                       # kept, 9.8% off: good
        window("a", 50.0, 56.0),                       # kept, 12% off: off
        window("a", 50.0, 100.0),                      # kept at twice the rate: off, doubled
        window("a", 50.0, 92.0, kept=False),           # 8% below twice, discarded: doubled
        window("a", 50.0, float("nan"), kept=False),   # no peak
        window("b", 70.0, 70.0, primary=False),        # the reference refused it: left out
    ])
    assert len(frame) == 5
    assert frame["kept"].tolist() == [True, True, True, False, False]
    assert frame["good"].tolist() == [True, False, False, False, False]
    assert frame["kept_off"].tolist() == [False, True, True, False, False]
    assert frame["doubled"].tolist() == [False, False, True, True, False]


def test_class_figures():
    rows = recordings_with("low", 2, 50.0, 4) + recordings_with("typ", 3, 75.0, 9)
    figures = robust.class_figures(robust.scored_windows(rows))
    assert figures["low"]["windows"] == 20 and figures["low"]["recordings"] == 2
    assert figures["low"]["share_good"] == pytest.approx(0.4)
    assert figures["typical"]["share_good"] == pytest.approx(0.9)
    assert figures["gap"]["windows"] == 0 and figures["gap"]["share_good"] is None


def test_a_large_loss_is_confirmed():
    rows = recordings_with("low", 30, 50.0, 3) + recordings_with("typ", 30, 75.0, 9)
    diff = robust.good_share_difference(robust.scored_windows(rows))
    assert diff["point"] == pytest.approx(-60.0)
    assert diff["upper"] <= robust.BAR_POINTS
    assert robust.outcome(diff["lower"], diff["upper"]) == "confirmed"


def test_no_loss_is_not_confirmed():
    rows = recordings_with("low", 30, 50.0, 9) + recordings_with("typ", 30, 75.0, 9)
    diff = robust.good_share_difference(robust.scored_windows(rows))
    assert diff["point"] == pytest.approx(0.0)
    assert diff["lower"] > robust.BAR_POINTS
    assert robust.outcome(diff["lower"], diff["upper"]) == "not confirmed"


def test_few_mixed_low_recordings_are_inconclusive():
    # Three low recordings that disagree: the interval has to carry that disagreement,
    # which only resampling whole recordings does.
    rows = (recordings_with("lowA", 1, 50.0, 2) + recordings_with("lowB", 1, 50.0, 9)
            + recordings_with("lowC", 1, 50.0, 10) + recordings_with("typ", 40, 75.0, 9))
    diff = robust.good_share_difference(robust.scored_windows(rows))
    assert diff["point"] == pytest.approx(-20.0)
    assert diff["lower"] <= robust.BAR_POINTS < diff["upper"]
    assert diff["upper"] > 0  # resampling single windows would stop near -4 here
    assert robust.outcome(diff["lower"], diff["upper"]) == "inconclusive"


def test_the_interval_is_reproducible_and_counts_empty_draws():
    rows = recordings_with("low", 1, 50.0, 5) + recordings_with("typ", 30, 75.0, 9)
    first = robust.good_share_difference(robust.scored_windows(rows))
    again = robust.good_share_difference(robust.scored_windows(rows))
    assert first == again
    # Draws that miss the one low recording have no difference and are left out.
    assert 0 < first["draws_used"] < first["draws"] == robust.BOOT_DRAWS


@pytest.mark.parametrize("lower, upper, expected", [
    (-30.0, -10.0, "confirmed"), (-30.0, -10.01, "confirmed"), (-20.0, -9.99, "inconclusive"),
    (-10.0, 5.0, "inconclusive"), (-9.99, 5.0, "not confirmed"), (None, None, "not estimable")])
def test_outcomes_follow_part0(lower, upper, expected):
    assert robust.outcome(lower, upper) == expected


def test_recording_bands():
    rows = [dict(ref_hr=50.0, status="ok", error_vs_reference=1.0),
            dict(ref_hr=55.0, status="insufficient_quality", error_vs_reference=float("nan")),
            dict(ref_hr=58.0, status="degraded_capture", error_vs_reference=-2.0),
            dict(ref_hr=95.0, status="unstable", error_vs_reference=4.0),
            dict(ref_hr=float("nan"), status="ok", error_vs_reference=0.0)]
    bands = {b["band"]: b for b in robust.recording_bands(rows)}
    assert list(bands) == ["<=55", "55-60", ">90"]
    assert bands["<=55"] == dict(band="<=55", recordings=2, reported=1, ok=1, median_abs_error=1.0)
    assert bands["55-60"]["reported"] == 1 and bands["55-60"]["ok"] == 0
    assert bands[">90"]["median_abs_error"] == 4.0


class FakeEstimator:
    def __init__(self, name):
        self.name, self.checkpoint, self.trained_on = name, None, ()


@pytest.mark.parametrize("limit, summary_name", [(None, "summary_low_hr.json"),
                                                 (3, "summary_low_hr_limit3.json")])
def test_the_run_writes_its_summary_and_tables(tmp_path, monkeypatch, limit, summary_name):
    rows = recordings_with("low", 20, 50.0, 4) + recordings_with("typ", 20, 75.0, 9)
    recordings = [dict(recording=f"low{i}", ref_hr=50.0, status="ok", error_vs_reference=0.5)
                  for i in range(20)]
    seen = []

    def evaluate(spec, estimators, limit_given):
        seen.append((spec.key, limit_given))
        return ({e.name: recordings for e in estimators}, {e.name: rows for e in estimators})

    monkeypatch.setattr(robust, "evaluate_set", evaluate)
    monkeypatch.setattr(robust, "make_estimators",
                        lambda names, device: [FakeEstimator(name) for name in names])
    args = argparse.Namespace(device="cpu", limit=limit, vv_store=tmp_path / "vv.h5",
                              ubfc_phys_store=tmp_path / "ubfc.h5", rice_store=tmp_path / "rice.h5",
                              out_dir=tmp_path / "out")
    robust.run_low_hr(args)

    assert seen == [("vitalvideos", limit), ("ubfc_phys_t1", limit), ("rice_clamped", limit)]
    summary = json.loads((tmp_path / "out" / summary_name).read_text(encoding="utf-8"))
    assert summary["verdict"]["set"] == "VitalVideos-WW"
    assert summary["verdict"]["estimator"] == "hr_physnet_v2"
    assert summary["verdict"]["outcome"] == "confirmed"
    assert summary["settings"]["bar_points"] == -10.0 and summary["settings"]["limit"] == limit
    assert summary["settings"]["interval"] == dict(unit="recording", draws=10_000, seed=42,
                                                   percentiles=[2.5, 97.5])
    assert [s["role"] for s in summary["settings"]["sets"]] == ["decisive", "descriptive",
                                                                "descriptive"]
    assert summary["readout"]["MIN_CONFIDENCE"] == 0.65
    assert set(summary["sets"]["Rice iPPG, clamped crop"]["estimators"]) == {"hr_physnet_v2",
                                                                            "hr_spectral"}
    assert "provenance" in summary
    assert len(list((tmp_path / "out").glob("windows_low_hr_*.csv"))) == 6
    assert len(list((tmp_path / "out").glob("recordings_low_hr_*.csv"))) == 6