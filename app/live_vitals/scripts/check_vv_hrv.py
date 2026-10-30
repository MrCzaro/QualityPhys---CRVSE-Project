"""Beat references on VitalVideos-WW: stage 1 of the variability run, camera-free.

Stage 1 establishes, from the recordings' own references, what the variability rule needs
before any camera beat is taken on this set. Nothing here computes a beat from an estimator:
the camera enters only through the app's capture gates and through PhysNet v2's heart-rate
status, neither of which carries a beat time. What it measures:

- **ECG R-peaks** on the 500 Hz lead. QRS-band energy finds each beat whatever the lead's
  polarity; the apex is then located on the band-passed lead and refined between samples.
  The audit detector that set `hr_ecg` was built to check heart rate and double-counts tall
  T waves; this one is built for interval timing.
- **Contact-PPG upstrokes** on the 500 Hz BIO PPG, with the camera harness's own fiducial
  (`steepest_rises`), so that a later camera-against-contact comparison measures timing,
  not a difference between two definitions of a beat.
- **The reference screen.** An ECG keeps its variability only when the physiological
  interval rules keep at least 90% of its intervals and at least 30 of them.
- **The floor.** SDNN and RMSSD of the contact PPG against the ECG over each 60 s
  recording: how close a pulse sensor gets to the heart's own intervals here, and so
  whether the word HRV can be earned by any pulse sensor at all.
- **The accepted set.** The app's capture gates re-run on the stored fields - framing on
  the stored box, rate, motion on the stored crops, the face count - and the reading status
  of PhysNet v2, the estimator the claim names. The primary set is the accepted recordings
  whose ECG is usable and carries no ECG or cardiac note.
- **The strata** the rule fixes, and their sizes on the primary set.

Every interval rule is the camera harness's (`kept_intervals`, `span_hrv`), so stage 2 can
score camera beats against these references under the same definitions. The reference is
scored against itself as a self-test: any row other than zero error means the harness is
broken. The finger oximeter (`cms_ppg`) is not read here.

The second-face and face-outside-the-box checks need per-sample boxes the store does not
keep, so they cannot be re-run; every other capture gate is the app's current code.

Usage:
    python -m app.live_vitals.scripts.check_vv_hrv [--store PATH] [--device cpu|cuda]
        [--limit N] [--out-dir DIR]

No frames are rendered or written. The store is read-only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, detrend, find_peaks, sosfiltfilt

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals.capture.motion import motion_verdict
from app.live_vitals.capture.session import (face_found_notes, framing_verdict, rate_verdict,
                                             worst_verdict)
from app.live_vitals.estimators import registry
from app.live_vitals.scripts.bench_outputs import provenance, subset_suffix
from app.live_vitals.scripts.check_hrv_sampling_ceiling import dominant_period, refine
from app.live_vitals.scripts.check_rice_hrv import (MIN_SPAN_INTERVALS, kept_intervals, span_hrv,
                                                    steepest_rises)
from app.live_vitals.scripts.check_ubfc_phys import REPORTED, _plain, _write_csv
from app.live_vitals.scripts.check_vitalvideos import DATASET, age_band

DEFAULT_STORE = Path(r"D:\QualityPhys\phase 3 datasets\vitalvideos_phase3_rc_none.h5")
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "vv_hrv_eval"
STATUS_ESTIMATOR = "hr_physnet_v2"   # the estimator the claim names sets the accepted set
QRS_BAND_HZ = (5.0, 20.0)
ENERGY_WINDOW_S = 0.15     # the moving integral spans about one QRS complex
R_SPACING = 0.55           # the least an R peak may follow the last, as a share of the period
R_ENERGY_SHARE = 0.3       # a candidate below this share of the median candidate is not a beat
R_SEARCH_S = 0.08          # the apex lies within this of the energy peak
PTT_WINDOW_S = (0.08, 0.45)  # an upstroke this long after an R peak belongs to that beat
REF_MIN_KEPT_SHARE = 0.90  # a reference keeps its variability above this share of kept intervals
TOLERANCE = 0.10           # the rule's tolerance, fixed before the camera run
TOLERANCE_REPORTED = 0.20
BAR = 0.50                 # the share's 95% lower bound must reach this
Z95 = 1.959964
FITZPATRICK_GROUPS = {1: "1-2", 2: "1-2", 3: "3-4", 4: "3-4", 5: "5-6", 6: "5-6"}


# --------------------------------------------------------------- the references --

def r_peaks(ecg, fs):
    """R-peak times in seconds from the record's start, and how they were found.

    Squared QRS-band energy, integrated over about one complex, finds the beats without
    caring about the lead's polarity; the apex is then taken on the band-passed lead itself,
    turned so that the R wave points up, and refined with a parabola through three samples.
    Candidates closer than R_SPACING of the dominant period keep the stronger, which is what
    stops a tall T wave from passing as a beat. Where premature beats crowd a record, the
    dominant period falls between the short and the long intervals, the spacing shrinks and
    a premature beat's T wave can pass as a beat. The interval screen then drops the
    reference; with that many premature beats, its normal-to-normal intervals could not
    support an SDNN in any case.
    """
    sos = butter(3, [QRS_BAND_HZ[0] / (fs / 2), QRS_BAND_HZ[1] / (fs / 2)], btype="band",
                 output="sos")
    x = sosfiltfilt(sos, detrend(np.asarray(ecg, dtype=float)))
    polarity = 1.0 if np.percentile(x, 99.5) >= -np.percentile(x, 0.5) else -1.0
    x = polarity * x
    width = max(1, int(round(ENERGY_WINDOW_S * fs)))
    energy = np.convolve(x ** 2, np.ones(width) / width, mode="same")
    period = dominant_period(energy, fs)
    candidates, _ = find_peaks(energy, distance=max(1, int(R_SPACING * period * fs)))
    found = dict(polarity=int(polarity), period_s=float(period))
    if candidates.size < 3:
        return np.array([]), found
    candidates = candidates[energy[candidates] >= R_ENERGY_SHARE * np.median(energy[candidates])]
    half = int(round(R_SEARCH_S * fs))
    apexes = []
    for c in candidates:
        lo, hi = max(1, c - half), min(len(x) - 1, c + half + 1)
        apexes.append(refine(x, lo + int(np.argmax(x[lo:hi]))))
    return np.unique(np.array(apexes)) / fs, found


def ppg_upstrokes(ppg, fs):
    """Contact-PPG beat times in seconds: the camera harness's fiducial, at the PPG's rate."""
    return steepest_rises(np.asarray(ppg, dtype=float), fs)


def transit_delays(r_times, upstrokes):
    """For each R peak, the delay to the first upstroke within PTT_WINDOW_S; NaN where none."""
    delays = np.full(len(r_times), np.nan)
    if not len(r_times) or not len(upstrokes):
        return delays
    index = np.searchsorted(upstrokes, r_times + PTT_WINDOW_S[0])
    inside = index < len(upstrokes)
    delays[inside] = upstrokes[index[inside]] - r_times[inside]
    delays[delays > PTT_WINDOW_S[1]] = np.nan
    return delays


def variability(beats, t_end):
    """SDNN and RMSSD (ms) over [0, t_end], the kept count and share, and the kept heart rate."""
    ends, ibi, kept = kept_intervals(beats)
    sdnn, rmssd, n = span_hrv(ends, ibi, kept, 0.0, t_end) if ibi.size else (np.nan, np.nan, 0)
    return dict(sdnn=sdnn, rmssd=rmssd, n_kept=int(n),
                kept_share=float(kept.mean()) if kept.size else 0.0,
                hr=float(60.0 / np.median(ibi[kept])) if kept.any() else float("nan"))


def relative_error(value, reference):
    """value / reference - 1, NaN where either is missing."""
    if not (np.isfinite(value) and np.isfinite(reference)) or reference <= 0:
        return float("nan")
    return float(value / reference - 1.0)


# ------------------------------------------------------------------ the capture --

def capture_gates(attrs, frames, fps):
    """The app's capture gates on the stored fields: verdicts, the overall verdict and notes."""
    box = tuple(float(attrs[k]) for k in ("box_rx0", "box_ry0", "box_rx1", "box_ry1"))
    framing, report, framing_notes = framing_verdict(
        box, int(attrs["frame_width"]), int(attrs["frame_height"]))
    rate, rate_note = rate_verdict(fps)
    motion, motion_note, _ = motion_verdict(frames, fps)
    face_notes = face_found_notes(int(attrs["face_detections"]),
                                  int(attrs["face_detections_attempted"]))
    face = worst_verdict(*(level for level, _ in face_notes))
    return dict(framing_verdict=framing, frac_side_lost=float(report["frac_side_lost"]),
                rate_verdict=rate, motion_verdict=motion, face_verdict=face,
                capture_verdict=worst_verdict(framing, rate, motion, face),
                capture_notes="; ".join(n for n in (*framing_notes, rate_note, motion_note,
                                                    *(note for _, note in face_notes)) if n))


# -------------------------------------------------------------------- the run --

def evaluate(store_path, estimator, limit):
    """One row per recording: references, floor, capture gates, reading status, strata."""
    rows = []
    with h5py.File(store_path, "r") as store:
        names = sorted(store.keys())[:limit] if limit else sorted(store.keys())
        t0 = time.perf_counter()
        for i, name in enumerate(names, 1):
            group = store[name]
            attrs = {k: _plain(v) for k, v in group.attrs.items()}
            fs, fps = float(attrs["bio_fs"]), float(attrs["fps"])
            ecg, ppg = group["bio_ecg"][()], group["bio_ppg"][()]
            t_end = len(ecg) / fs
            r_times, found = r_peaks(ecg, fs)
            upstrokes = ppg_upstrokes(ppg, fs)
            ecg_hrv, ppg_hrv = variability(r_times, t_end), variability(upstrokes, t_end)
            delays = transit_delays(r_times, upstrokes)
            usable = (ecg_hrv["kept_share"] >= REF_MIN_KEPT_SHARE
                      and ecg_hrv["n_kept"] >= MIN_SPAN_INTERVALS)

            frames = group["frames"][()]
            gates = capture_gates(attrs, frames, fps)
            result = estimator.estimate(frames, fps)
            accepted = gates["capture_verdict"] != "REJECT" and result.status in REPORTED
            unflagged = (bool(attrs["ecg_ok"]) and not bool(attrs["note_ecg_bad"])
                         and not bool(attrs["note_cardiac"]))
            rows.append(dict(
                recording=name, split=attrs["split"], fitzpatrick=int(attrs["fitzpatrick"]),
                fitzpatrick_group=FITZPATRICK_GROUPS[int(attrs["fitzpatrick"])],
                site_group=attrs["site_group"], age=int(attrs["age"]),
                age_band=age_band(int(attrs["age"])),
                ecg_ok=bool(attrs["ecg_ok"]), note_ecg_bad=bool(attrs["note_ecg_bad"]),
                note_cardiac=bool(attrs["note_cardiac"]), qc_pos_bvp_r=float(attrs["qc_pos_bvp_r"]),
                **gates, status=result.status, value=float(result.value),
                accepted=bool(accepted), unflagged=unflagged, ecg_usable=bool(usable),
                primary=bool(accepted and unflagged and usable),
                duration_s=t_end, ecg_polarity=found["polarity"],
                n_r_peaks=int(len(r_times)), n_upstrokes=int(len(upstrokes)),
                hr_ecg_audit=float(attrs["hr_ecg"]), hr_ecg=ecg_hrv["hr"], hr_ppg=ppg_hrv["hr"],
                ecg_kept_share=ecg_hrv["kept_share"], ecg_n_kept=ecg_hrv["n_kept"],
                ppg_kept_share=ppg_hrv["kept_share"], ppg_n_kept=ppg_hrv["n_kept"],
                paired_share=float(np.isfinite(delays).mean()) if len(delays) else 0.0,
                ptt_median_ms=float(np.nanmedian(delays) * 1000.0)
                if np.isfinite(delays).any() else float("nan"),
                ptt_iqr_ms=float(np.subtract(*np.nanpercentile(delays, [75, 25])) * 1000.0)
                if np.isfinite(delays).sum() >= 4 else float("nan"),
                sdnn_ecg=ecg_hrv["sdnn"], sdnn_ppg=ppg_hrv["sdnn"],
                rmssd_ecg=ecg_hrv["rmssd"], rmssd_ppg=ppg_hrv["rmssd"],
                sdnn_error_ppg=relative_error(ppg_hrv["sdnn"], ecg_hrv["sdnn"]),
                rmssd_error_ppg=relative_error(ppg_hrv["rmssd"], ecg_hrv["rmssd"]),
                sdnn_error_self=relative_error(ecg_hrv["sdnn"], ecg_hrv["sdnn"]),
                rmssd_error_self=relative_error(ecg_hrv["rmssd"], ecg_hrv["rmssd"])))
            if i % 25 == 0 or i == len(names):
                print(f"  {i}/{len(names)} recordings | {(time.perf_counter() - t0) / 60:.1f} min",
                      flush=True)
    return rows


def wilson(k, n):
    """95% Wilson interval of a share k/n; (nan, nan) for n = 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = p + Z95 ** 2 / (2 * n)
    half = Z95 * np.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n))
    denominator = 1 + Z95 ** 2 / n
    return float((centre - half) / denominator), float((centre + half) / denominator)


def outcome(low, high):
    """The rule's three outcomes for a share's 95% interval against BAR."""
    if not np.isfinite(low):
        return "not estimable"
    return "confirmed" if low >= BAR else ("not confirmed" if high < BAR else "inconclusive")


def error_summary(errors):
    """Median, median absolute and the shares within the tolerances, with the 10% interval."""
    v = np.array([e for e in errors if np.isfinite(e)], dtype=float)
    if not v.size:
        return dict(n=0)
    k = int(np.sum(np.abs(v) <= TOLERANCE))
    low, high = wilson(k, v.size)
    return dict(n=int(v.size), median=float(np.median(v)), median_abs=float(np.median(np.abs(v))),
                within_10=k / v.size, within_10_low=low, within_10_high=high,
                within_10_outcome=outcome(low, high),
                within_20=float(np.mean(np.abs(v) <= TOLERANCE_REPORTED)))


def _quantiles(values, qs):
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    return [float(np.quantile(v, q)) for q in qs] if v.size else []


def tercile_counts(rows, key):
    """Cut points of a value's terciles over the rows, and how many rows fall in each."""
    cuts = _quantiles([r[key] for r in rows], (1 / 3, 2 / 3))
    if len(cuts) < 2:
        return dict(cuts=cuts, counts=[])
    v = np.array([r[key] for r in rows], dtype=float)
    return dict(cuts=cuts, counts=[int(np.sum(v <= cuts[0])),
                                   int(np.sum((v > cuts[0]) & (v <= cuts[1]))),
                                   int(np.sum(v > cuts[1]))])


def summarise(rows):
    """Aggregates only: no recording appears beside its skin type, site or age."""
    primary = [r for r in rows if r["primary"]]
    usable_any = [r for r in rows if r["ecg_usable"]]
    hr_gap = np.array([abs(r["hr_ecg"] - r["hr_ecg_audit"]) for r in rows
                       if np.isfinite(r["hr_ecg"]) and np.isfinite(r["hr_ecg_audit"])])
    count = lambda key, value=True: sum(1 for r in rows if r[key] == value)  # noqa: E731
    return dict(
        n_recordings=len(rows),
        funnel=dict(capture_not_rejected=sum(1 for r in rows if r["capture_verdict"] != "REJECT"),
                    reported_by_v2=sum(1 for r in rows if r["status"] in REPORTED),
                    accepted=count("accepted"),
                    accepted_unflagged=sum(1 for r in rows if r["accepted"] and r["unflagged"]),
                    primary=len(primary)),
        capture_rejected={gate: count(f"{gate}_verdict", "REJECT")
                          for gate in ("framing", "rate", "motion", "face")},
        statuses={s: count("status", s) for s in sorted({r["status"] for r in rows})},
        reference=dict(
            ecg_usable=len(usable_any),
            ecg_usable_among_accepted_unflagged=sum(
                1 for r in rows if r["accepted"] and r["unflagged"] and r["ecg_usable"]),
            negative_leads=count("ecg_polarity", -1),
            hr_against_audit=dict(n=int(hr_gap.size),
                                  median_abs_bpm=float(np.median(hr_gap)) if hr_gap.size else None,
                                  within_1_bpm=float(np.mean(hr_gap <= 1.0)) if hr_gap.size else None,
                                  over_2_bpm=int(np.sum(hr_gap > 2.0))),
            paired_share=dict(zip(("q10", "median"), _quantiles([r["paired_share"] for r in rows],
                                                               (0.1, 0.5)))),
            recordings_paired_95=sum(1 for r in rows if r["paired_share"] >= 0.95),
            ptt_median_ms=dict(zip(("q25", "median", "q75"),
                                   _quantiles([r["ptt_median_ms"] for r in rows],
                                              (0.25, 0.5, 0.75)))),
            ptt_iqr_ms_median=(_quantiles([r["ptt_iqr_ms"] for r in rows], (0.5,)) or [None])[0]),
        floor=dict(sdnn=error_summary([r["sdnn_error_ppg"] for r in primary]),
                   rmssd=error_summary([r["rmssd_error_ppg"] for r in primary]),
                   reference_sdnn_ms=dict(zip(("q25", "median", "q75"),
                                              _quantiles([r["sdnn_ecg"] for r in primary],
                                                         (0.25, 0.5, 0.75))))),
        self_test=dict(sdnn=error_summary([r["sdnn_error_self"] for r in primary]),
                       rmssd=error_summary([r["rmssd_error_self"] for r in primary])),
        strata=dict(
            fitzpatrick_group={g: sum(1 for r in primary if r["fitzpatrick_group"] == g)
                               for g in ("1-2", "3-4", "5-6")},
            site_group={s: sum(1 for r in primary if r["site_group"] == s)
                        for s in sorted({r["site_group"] for r in rows})},
            age_band={b: sum(1 for r in primary if r["age_band"] == b)
                      for b in sorted({r["age_band"] for r in rows})},
            ecg_sdnn_terciles=tercile_counts(primary, "sdnn_ecg"),
            qc_pos_bvp_r_terciles=tercile_counts(primary, "qc_pos_bvp_r")),
    )


def build_summary(rows, args, estimator):
    """The JSON summary: settings, aggregates and provenance."""
    return dict(
        dataset=DATASET, stage=1, store=str(args.store), limit=args.limit,
        status_estimator=estimator.name,
        settings=dict(qrs_band_hz=list(QRS_BAND_HZ), energy_window_s=ENERGY_WINDOW_S,
                      r_spacing=R_SPACING, r_energy_share=R_ENERGY_SHARE,
                      r_search_s=R_SEARCH_S, ptt_window_s=list(PTT_WINDOW_S),
                      ref_min_kept_share=REF_MIN_KEPT_SHARE,
                      min_span_intervals=MIN_SPAN_INTERVALS, tolerance=TOLERANCE,
                      tolerance_reported=TOLERANCE_REPORTED, bar_lower_bound=BAR),
        not_rerun=["second face", "face outside the held box"],
        **summarise(rows),
        provenance=provenance(estimator),
    )


def _share(m):
    if not m.get("n"):
        return "n/a"
    return (f"{m['within_10']:.0%} within 10% (95% {m['within_10_low']:.0%}-"
            f"{m['within_10_high']:.0%}, {m['within_10_outcome']}), median {m['median']:+.1%}, "
            f"within 20% {m['within_20']:.0%}, n={m['n']}")


def print_summary(s):
    """The stage-1 facts, in the order the rule needs them."""
    f, ref = s["funnel"], s["reference"]
    print(f"\n=== VitalVideos stage 1 | {s['n_recordings']} recordings ===")
    print(f"capture gates pass {f['capture_not_rejected']} (refused: {s['capture_rejected']}) | "
          f"v2 reports {f['reported_by_v2']} {s['statuses']}")
    print(f"accepted {f['accepted']} -> no ECG or cardiac note {f['accepted_unflagged']} -> "
          f"ECG screen passed: primary {f['primary']}")
    hr = ref["hr_against_audit"]
    print(f"ECG usable in {ref['ecg_usable']} of {s['n_recordings']} | negative leads "
          f"{ref['negative_leads']} | R-peak HR against the audit: median |gap| "
          f"{hr['median_abs_bpm']:.2f} bpm, within 1 bpm {hr['within_1_bpm']:.0%}, "
          f"over 2 bpm {hr['over_2_bpm']}")
    print(f"R peaks paired with an upstroke: median {ref['paired_share'].get('median', float('nan')):.1%}"
          f" (10th percentile {ref['paired_share'].get('q10', float('nan')):.1%}), "
          f"{ref['recordings_paired_95']} recordings at 95% or more | transit "
          f"{ref['ptt_median_ms'].get('median', float('nan')):.0f} ms in the median recording")
    print(f"floor, contact PPG against ECG, primary set (reference SDNN median "
          f"{s['floor']['reference_sdnn_ms'].get('median', float('nan')):.1f} ms):")
    print(f"  SDNN  {_share(s['floor']['sdnn'])}")
    print(f"  RMSSD {_share(s['floor']['rmssd'])}")
    print(f"self-test, ECG against itself: SDNN {_share(s['self_test']['sdnn'])}")
    st = s["strata"]
    print(f"strata on the primary set: Fitzpatrick {st['fitzpatrick_group']} | sites "
          f"{st['site_group']} | age {st['age_band']}")
    print(f"  ECG SDNN terciles {st['ecg_sdnn_terciles']} | POS-label r terciles "
          f"{st['qc_pos_bvp_r_terciles']}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--device", default="cpu", help="torch device for v2's reading status")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N recordings (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    estimator = registry.get_estimator(STATUS_ESTIMATOR, device=args.device)
    if not estimator.is_available():
        raise SystemExit(f"ERROR: checkpoint missing for {STATUS_ESTIMATOR}")
    if any(dataset == DATASET for dataset, _ in estimator.trained_on):
        raise SystemExit(f"ERROR: {STATUS_ESTIMATOR} trained on {DATASET}")
    print(f"store: {args.store}\nstage 1, camera-free | status estimator {STATUS_ESTIMATOR} | "
          f"device {args.device}" + (f" | limit {args.limit}" if args.limit else ""))

    rows = evaluate(args.store, estimator, args.limit)
    suffix = subset_suffix(("limit", args.limit or None, None))
    summary = build_summary(rows, args, estimator)
    _write_csv(args.out_dir / f"stage1_recordings{suffix}.csv", rows)
    (args.out_dir / f"summary_stage1{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary)
    print(f"\noutputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())