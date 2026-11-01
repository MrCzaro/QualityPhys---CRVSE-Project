"""check_hr_robustness on synthetic data only.

Check 1 on rows built here (classes, flags, interval, outcomes, outputs): the benchmark
scripts' evaluation functions are replaced, so no store is read. Check 2 on a synthetic face
recorded losslessly at a quarter of the raw recordings' size (the ladder, its self-test, the
outcome rule, the app's own reading path, the run). Check 3 on a small store of the VitalVideos
schema and a synthetic face at 60 fps (the thinning, its self-test, the readout told the rate,
the outcome rule, the verdict, the run). None of it says anything about a dataset.
"""
import argparse
import json
from fractions import Fraction

import av
import cv2
import h5py
import numpy as np
import pytest

from app.live_vitals.estimators import registry
from app.live_vitals.estimators.base import EstimatorResult
from app.live_vitals.scripts import check_hr_robustness as robust
from app.live_vitals.tests import synth


# ------------------------------------------------------------------ check 1 --

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


# ------------------------------------------------------------------ check 2 --
# The recipe at a quarter of the real size: a 480x528 lossless 4:4:4 source at 60 fps,
# laddered to 240x264 at 30 fps. Only the sizes and the frame count are changed.

def small_recipe(monkeypatch, frames):
    monkeypatch.setattr(robust, "SOURCE", dict(codec="h264", pix_fmt="yuv444p", height=528,
                                               width=480, fps=60.0, frames=frames))
    monkeypatch.setattr(robust, "LADDER_HEIGHT", 264)
    monkeypatch.setattr(robust, "LADDER_WIDTH", 240)


def write_source(path, frames, pix_fmt="yuv444p", bpm=72.0):
    """A lossless H.264 recording of the synthetic face at 60 fps, laid out like the raw subset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = synth.Scene(width=480, height=528, fps=60.0, bpm=bpm)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=60, options={"qp": "0", "preset": "veryfast"})
        stream.width, stream.height, stream.pix_fmt = 480, 528, pix_fmt
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(scene.frame(i), format="rgb24").reformat(format=pix_fmt)
            frame.pts, frame.time_base = i, Fraction(1, 60)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    return path


def lossy_copy(r0, path):
    """The frames of an R0 encoded again with loss, as a broken baseline would be."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30, options={"qp": "20", "preset": "veryfast"})
        stream.width, stream.height, stream.pix_fmt = 240, 264, "yuv444p"
        for i, planes in enumerate(robust.decoded_planes(r0)):
            frame = av.VideoFrame.from_ndarray(planes, format="yuv444p")
            frame.pts, frame.time_base = i, Fraction(1, 30)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))


def test_the_encoder_settings_are_part0s():
    by_rung = {rung["rung"]: robust.encoder_options(rung) for rung in robust.RUNGS}
    assert by_rung["R0"] == by_rung["R1"] == {"qp": "0", "preset": "veryfast", "g": "60"}
    assert by_rung["R2"] == {"preset": "veryfast", "tune": "zerolatency", "maxrate": "4950000",
                             "bufsize": "4950000", "g": "60"}
    assert by_rung["R3"]["maxrate"] == "2475000" and by_rung["R4"]["maxrate"] == "1237500"
    assert by_rung["R5"] == {"deadline": "realtime", "cpu-used": "4", "maxrate": "4950000",
                             "bufsize": "4950000", "g": "60"}
    assert [rung["pix_fmt"] for rung in robust.RUNGS] == ["yuv444p"] + ["yuv420p"] * 5
    assert robust.BROWSER_BITS == pytest.approx(0.651, abs=5e-4)


def test_source_planes_are_every_second_frame_scaled_by_area(tmp_path, monkeypatch):
    small_recipe(monkeypatch, frames=8)
    source = write_source(tmp_path / "raw" / "rec" / "rec_1.mp4", frames=8)
    with av.open(str(source)) as container:
        decoded = [frame.to_ndarray(format="yuv444p") for frame in container.decode(video=0)]
    planes = list(robust.source_planes(source))
    assert len(planes) == 4 and planes[0].shape == (3, 264, 240)
    for got, index in zip(planes, (0, 2, 4, 6)):
        expected = np.stack([cv2.resize(p, (240, 264), interpolation=cv2.INTER_AREA)
                             for p in decoded[index]])
        assert np.array_equal(got, expected)


def test_a_source_not_as_described_is_refused(tmp_path, monkeypatch):
    small_recipe(monkeypatch, frames=8)
    source = write_source(tmp_path / "raw" / "rec" / "rec_1.mp4", frames=8, pix_fmt="yuv420p")
    with pytest.raises(ValueError, match="Part 0 describes"):
        list(robust.source_planes(source))


def test_the_ladder_and_its_self_test(tmp_path, monkeypatch):
    small_recipe(monkeypatch, frames=40)
    source = write_source(tmp_path / "raw" / "rec" / "rec_1.mp4", frames=40)
    robust.write_ladder(source, tmp_path / "ladder", "rec")
    for rung in robust.RUNGS:
        facts = robust.file_facts(robust.rung_file(tmp_path / "ladder", "rec", rung))
        assert robust.facts_match(facts, rung), (rung["rung"], facts)
    r0 = robust.rung_file(tmp_path / "ladder", "rec", robust.RUNGS[0])
    assert robust.r0_self_test(source, r0) == dict(equal=True, frames=20, max_abs_diff=0)
    lossy_copy(r0, tmp_path / "lossy.mp4")
    broken = robust.r0_self_test(source, tmp_path / "lossy.mp4")
    assert not broken["equal"] and broken["max_abs_diff"] > 0
    # A 4:2:0 baseline cannot hold 4:4:4 planes either.
    r1 = robust.rung_file(tmp_path / "ladder", "rec", robust.RUNGS[1])
    assert not robust.r0_self_test(source, r1)["equal"]


def reading(value, reported=True):
    return dict(value=value, reported=reported)


@pytest.mark.parametrize("values, expected", [
    ([72.0, 60.0, 80.0], "no effect"),
    ([73.0, 59.0, 81.0], "no effect"),       # each exactly 1.0 away
    ([73.5, 60.0, 80.0], "inconclusive"),    # one moved 1.5
    ([75.0, 60.0, 80.0], "inconclusive"),    # one moved 3.0: one is not enough
    ([75.0, 63.0, 80.0], "effect"),          # two moved more than 2.0
    ([None, 60.0, 80.0], "inconclusive"),    # one lost
    ([None, None, 80.0], "effect"),          # two lost
    ([None, 63.0, 80.0], "effect"),          # one lost, one moved more than 2.0
    ([74.0, 62.0, 80.0], "inconclusive"),    # two moved exactly 2.0: not more than it
])
def test_rung_outcomes_follow_part0(values, expected):
    base = {"a": reading(72.0), "b": reading(60.0), "c": reading(80.0),
            "d": reading(float("nan"), reported=False)}
    rung = {name: reading(float("nan"), reported=False) if value is None else reading(value)
            for name, value in zip("abc", values)}
    rung["d"] = reading(66.0)  # R0 did not report it: gained, and outside the rule
    result = robust.rung_outcome(base, rung)
    assert result["outcome"] == expected
    assert result["base_reported"] == 3 and result["gained"] == 1


def test_nothing_reported_at_r0_is_not_estimable():
    base = {"a": reading(float("nan"), reported=False)}
    assert robust.rung_outcome(base, {"a": reading(70.0)})["outcome"] == "not estimable"


def test_a_rung_is_read_as_the_web_app_reads_an_upload(tmp_path, monkeypatch, landmarker):
    small_recipe(monkeypatch, frames=640)  # 320 frames at 30 fps: three analysis windows
    source = write_source(tmp_path / "raw" / "rec" / "rec_1.mp4", frames=640, bpm=72.0)
    robust.write_ladder(source, tmp_path / "ladder", "rec")
    pos = registry.get_estimator("hr_spectral")
    for rung in (robust.RUNGS[0], robust.RUNGS[2], robust.RUNGS[5]):
        path = robust.rung_file(tmp_path / "ladder", "rec", rung)
        result = robust.read_like_the_app(path, landmarker, [pos])["hr_spectral"]
        assert result["reported"], (rung["rung"], result)
        assert result["value"] == pytest.approx(72.0, abs=3.0)
        assert result["capture_verdict"] != "REJECT" and result["frames"] == 320


def test_the_reencode_run(tmp_path, monkeypatch):
    small_recipe(monkeypatch, frames=40)
    sources = []
    for name in ("rec_a", "rec_b"):
        sources.append(write_source(tmp_path / "raw" / name / f"{name}_1.mp4", frames=40))
        robust.write_ladder(sources[-1], tmp_path / "ladder", name)
    robust.write_manifest(tmp_path / "ladder", sources)

    def read(path, landmarker, estimators):
        rung, name = path.stem, path.parent.name
        moved = 1.5 if (rung == "R4" and name == "rec_a") else 0.0
        lost = rung == "R5"
        return {e.name: dict(value=float("nan") if lost else 70.0 + moved,
                             status="capture_rejected" if lost else "ok", reported=not lost,
                             capture_verdict="WARN", capture_reasons="", effective_fps=30.0,
                             frames=20, confidence=0.9, usable_fraction=1.0) for e in estimators}

    class Closable:
        def close(self):
            pass

    monkeypatch.setattr(robust, "read_like_the_app", read)
    monkeypatch.setattr(robust, "make_landmarker", Closable)
    monkeypatch.setattr(robust, "reference_rates", lambda names: {n: 70.5 for n in names})
    monkeypatch.setattr(robust, "make_estimators",
                        lambda names, device: [FakeEstimator(name) for name in names])
    args = argparse.Namespace(device="cpu", limit=None, source=tmp_path / "raw",
                              ladder_dir=tmp_path / "ladder", out_dir=tmp_path / "out")
    robust.run_reencode(args)

    summary = json.loads((tmp_path / "out" / "summary_reencode.json").read_text(encoding="utf-8"))
    assert summary["self_test"] == dict(recordings=2, equal=2, frames_each=[20], max_abs_diff=0)
    assert summary["verdict"] == dict(rung="R2", estimator="hr_physnet_v2", outcome="no effect")
    outcomes = {rung: block["hr_physnet_v2"]["outcome"] for rung, block in summary["rungs"].items()}
    assert outcomes == {"R1": "no effect", "R2": "no effect", "R3": "no effect",
                        "R4": "inconclusive", "R5": "effect"}
    assert summary["settings"]["decisive_rung"] == "R2" and summary["settings"]["recordings"] == 2
    assert [r["rung"] for r in summary["settings"]["rungs"]] == ["R0", "R1", "R2", "R3", "R4", "R5"]
    assert summary["files"]["R5"]["codec"] == "vp8" and summary["files"]["R0"]["pix_fmt"] == "yuv444p"
    rows = (tmp_path / "out" / "recordings_reencode.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 2 * 6 * 2
    assert len(summary["ladder"]["manifest_sha256"]) == 64

    def run_again(out):
        robust.run_reencode(argparse.Namespace(**{**vars(args), "out_dir": tmp_path / out}))

    # A file changed after the manifest was written is refused.
    r3_a = robust.rung_file(tmp_path / "ladder", "rec_a", robust.RUNGS[3])
    kept = r3_a.read_bytes()
    r3_a.write_bytes(kept + b"\0")
    with pytest.raises(SystemExit, match="rec_a/R3.mp4 does not match the manifest"):
        run_again("out2")
    r3_a.write_bytes(kept)
    # A baseline that is not lossless stops the run before anything is scored, even when
    # the manifest records it.
    r0_b = robust.rung_file(tmp_path / "ladder", "rec_b", robust.RUNGS[0])
    lossy_copy(r0_b, tmp_path / "lossy.mp4")
    r0_b.write_bytes((tmp_path / "lossy.mp4").read_bytes())
    robust.write_manifest(tmp_path / "ladder", sources)
    with pytest.raises(SystemExit, match="does not decode to the frames it was made from"):
        run_again("out3")
    # A ladder without its manifest is refused, and so is a missing rung.
    (tmp_path / "ladder" / robust.MANIFEST).unlink()
    with pytest.raises(SystemExit, match="no manifest.json"):
        run_again("out4")
    robust.rung_file(tmp_path / "ladder", "rec_a", robust.RUNGS[3]).unlink()
    with pytest.raises(SystemExit, match="1 rung files missing"):
        run_again("out5")


# ------------------------------------------------------------------ check 3 --

def test_the_thinning_follows_part0():
    for rate, count in ((30, 1800), (27, 1620), (25, 1500), (24, 1440), (22, 1320), (20, 1200)):
        index = robust.thinned_indices(3600, rate)
        assert len(index) == count and index[0] == 0 and index[-1] < 3600
        assert np.all(np.diff(index) > 0)
        # The frame nearest each tick: never more than half a 60 fps frame (8.3 ms) off it.
        assert np.abs(index - np.arange(count) * 60 / rate).max() <= 0.5
    assert np.array_equal(robust.thinned_indices(3600, 30), np.arange(0, 3600, 2))
    assert np.array_equal(robust.thinned_indices(3600, 20), np.arange(0, 3600, 3))
    # At 24 fps every second tick lies halfway between two frames and goes to the even one.
    assert robust.thinned_indices(3600, 24)[:8].tolist() == [0, 2, 5, 8, 10, 12, 15, 18]


def test_the_readout_is_told_the_thinned_rate():
    scene = synth.Scene(width=72, height=72, fps=60.0, bpm=72.0)
    frames_60 = np.stack(list(scene.frames(1200)))
    bvp_60 = np.sin(2 * np.pi * 1.2 * np.arange(1200) / 60).astype(np.float32)
    index = robust.thinned_indices(1200, 24)
    pos = registry.get_estimator("hr_spectral")
    row = robust.read_at_rate(frames_60[index], bvp_60[index], 24, [pos])["hr_spectral"]
    assert row["frames"] == 480 and row["windows"] == 5
    assert row["reported"] and row["value"] == pytest.approx(72.0, abs=2.0)
    assert row["ref_value"] == pytest.approx(72.0, abs=1.0) and row["abs_error"] < 2.0
    # A reference that gives no value leaves no error, however the estimator reads.
    flat = robust.read_at_rate(frames_60[index], np.zeros(len(index), np.float32), 24, [pos])
    assert flat["hr_spectral"]["reported"] and np.isnan(flat["hr_spectral"]["abs_error"])


def errors(values, unreported=0, prefix="r"):
    """Rows of one rate: a reported recording per error given, then unreported ones."""
    rows = {f"{prefix}{i:02d}": dict(abs_error=value, reported=True) for i, value in enumerate(values)}
    rows.update({f"u{i:02d}": dict(abs_error=float("nan"), reported=False) for i in range(unreported)})
    return rows


def lose(rows, n):
    """The same rows with the first n reported recordings no longer reported."""
    rows = {name: dict(row) for name, row in rows.items()}
    for name in sorted(rows)[:n]:
        rows[name] = dict(abs_error=float("nan"), reported=False)
    return rows


BASE = [0.25] * 40


@pytest.mark.parametrize("at_rate, expected", [
    (errors([0.25] * 40), "holds"),
    (errors([0.75] * 40), "holds"),                       # +0.5 exactly: at the bound, not above it
    (errors([0.875] * 40), "fails"),                      # the whole interval above +0.5
    (errors([0.25, 1.25] * 20), "inconclusive"),          # mean +0.5, the interval across it
    (lose(errors([0.25] * 40), 3), "holds"),              # 3 fewer is allowed
    (lose(errors([0.25] * 40), 4), "inconclusive"),
    (lose(errors([0.25] * 40), 15), "inconclusive"),      # 15 fewer does not fail ...
    (lose(errors([0.25] * 40), 16), "fails"),             # ... 16 does
])
def test_rate_outcomes_follow_part0(at_rate, expected):
    result = robust.rate_outcome(errors(BASE), at_rate)
    assert result["outcome"] == expected
    assert result["reported_at_30"] == 40


def test_fewer_counts_every_recording_and_pairs_need_an_error_at_both_rates():
    base = errors([0.25] * 10, unreported=2)
    at_rate = errors([0.25] * 10, unreported=2)
    at_rate["r00"] = dict(abs_error=float("nan"), reported=True)   # reported, but no reference
    at_rate["u00"] = dict(abs_error=0.3, reported=True)            # reported at the rate only
    result = robust.rate_outcome(base, at_rate)
    assert result["fewer"] == -1 and result["reported"] == 11
    assert result["pairs"] == 9 and result["pairs_without_reference"] == 1
    assert result["point"] == pytest.approx(0.0)


def test_no_pairs_is_not_estimable_unless_too_few_are_reported():
    assert robust.rate_outcome(errors(BASE), lose(errors(BASE), 40))["outcome"] == "fails"
    nothing = errors([float("nan")] * 5)
    assert robust.rate_outcome(nothing, nothing)["outcome"] == "not estimable"


def test_the_interval_is_reproducible():
    rows = errors([0.1 * (i % 7) for i in range(40)])
    first = robust.rate_outcome(errors(BASE), rows)
    assert first == robust.rate_outcome(errors(BASE), rows)
    assert first["lower"] < first["point"] < first["upper"] and first["draws"] == robust.BOOT_DRAWS


@pytest.mark.parametrize("outcomes, expected", [
    (["holds"] * 5, 20),
    (["holds", "holds", "inconclusive", "holds", "holds"], 25),
    (["fails", "holds", "holds", "holds", "holds"], 30),
    (["holds", "holds", "holds", "fails", "holds"], 24),
    (["not estimable", "holds", "holds", "holds", "holds"], 30),
])
def test_the_verdict_is_the_lowest_rate_holding_with_every_rate_above(outcomes, expected):
    by_rate = {rate: dict(outcome=outcome) for rate, outcome in zip(robust.RATES[1:], outcomes)}
    assert robust.lowest_holding_rate(by_rate) == expected


def write_store(path, n_recordings, frames=1200, broken=None):
    """A VitalVideos-schema store: 60 fps crops whose pixel (0, 0) names the recording, a
    72 bpm reference, and the 30 fps streams as every second frame - except in `broken`."""
    t = np.arange(frames) / 60
    with h5py.File(path, "w") as store:
        for i in range(n_recordings):
            group = store.create_group(f"rec{i}")
            crops = np.zeros((frames, 8, 8, 3), np.uint8)
            crops[:, 0, 0, 0] = i
            crops[:, 0, 0, 1] = np.arange(frames) % 251
            group["frames_60"] = crops
            group["frames"] = crops[1::2] if f"rec{i}" == broken else crops[::2]
            group["bvp_60"] = np.sin(2 * np.pi * 1.2 * t).astype(np.float32)
            group["bvp"] = group["bvp_60"][()][::2]
            group.attrs.update(fps=30.0, fps_native=60.0, hr_ecg=72.0)
    return path


class RateFake(FakeEstimator):
    """Reads 72 bpm down to 24 fps and 75 bpm below; recording 0 stops reporting below 25 fps."""

    def estimate(self, frames, fps):
        recording = int(frames[0, 0, 0, 0])
        if recording == 0 and fps < 25:
            return EstimatorResult("heart_rate", float("nan"), "bpm", 0.0, "insufficient_quality",
                                   detail=dict(usable_fraction=0.1))
        value = 72.0 if fps >= 24 or self.name == "hr_spectral" else 75.0
        return EstimatorResult("heart_rate", value, "bpm", 0.9, "ok", detail=dict(usable_fraction=1.0))


@pytest.mark.parametrize("limit, summary_name", [(None, "summary_frame_rate.json"),
                                                 (3, "summary_frame_rate_limit3.json")])
def test_the_frame_rate_run(tmp_path, monkeypatch, limit, summary_name):
    store = write_store(tmp_path / "vv.h5", 6)
    monkeypatch.setattr(robust, "make_estimators",
                        lambda names, device: [RateFake(name) for name in names])
    args = argparse.Namespace(device="cpu", limit=limit, store=store, out_dir=tmp_path / "out")
    robust.run_frame_rate(args)

    summary = json.loads((tmp_path / "out" / summary_name).read_text(encoding="utf-8"))
    n = limit or 6
    assert summary["self_test"] == dict(recordings=n, frames_equal=n, bvp_equal=n)
    v2 = {row["rate"]: row for row in summary["estimators"]["hr_physnet_v2"]["rates"]}
    assert [v2[rate]["frames_each"] for rate in robust.RATES] == [[600], [540], [500], [480], [440], [400]]
    assert v2[30]["against_30"] is None and v2[30]["reported"] == n
    outcomes = {rate: v2[rate]["against_30"]["outcome"] for rate in robust.RATES[1:]}
    assert outcomes == {27: "holds", 25: "holds", 24: "holds", 22: "fails", 20: "fails"}
    assert v2[24]["against_30"]["fewer"] == 1 and v2[24]["against_30"]["pairs"] == n - 1
    assert summary["verdict"] == dict(estimator="hr_physnet_v2", lowest_holding_rate=24)
    assert summary["estimators"]["hr_spectral"]["lowest_holding_rate"] == 20
    assert summary["settings"]["rates"] == [30, 27, 25, 24, 22, 20]
    assert summary["settings"]["hold_bpm"] == 0.5 and summary["settings"]["limit"] == limit
    assert summary["readout"]["MIN_CONFIDENCE"] == 0.65 and "provenance" in summary
    table = (tmp_path / "out" / summary_name.replace("summary", "recordings").replace(".json", ".csv"))
    assert len(table.read_text(encoding="utf-8").splitlines()) == 1 + n * 6 * 2


def test_a_thinning_that_misses_the_stored_frames_scores_nothing(tmp_path, monkeypatch):
    store = write_store(tmp_path / "vv.h5", 3, broken="rec1")
    monkeypatch.setattr(robust, "make_estimators",
                        lambda names, device: [RateFake(name) for name in names])
    args = argparse.Namespace(device="cpu", limit=None, store=store, out_dir=tmp_path / "out")
    with pytest.raises(SystemExit, match="does not return the stored frames"):
        robust.run_frame_rate(args)
    assert not (tmp_path / "out").exists()