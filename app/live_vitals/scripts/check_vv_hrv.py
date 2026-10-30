"""Pulse-rate variability on VitalVideos-WW: beat references (stage 1) and the camera run (stage 2).

Stage 1, camera-free, establishes from the recordings' own references what the variability rule
needs before any camera beat is taken on this set. Nothing in it computes a beat from an
estimator: the camera enters only through the app's capture gates and through PhysNet v2's
heart-rate status, neither of which carries a beat time. What it measures:

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

Every interval rule is the camera harness's (`kept_intervals`, `span_hrv`), so stage 2 scores
camera beats against these references under the same definitions. The reference is scored
against itself as a self-test: any row other than zero error means the harness is broken.
Stage 1 does not read the finger oximeter (`cms_ppg`). The second-face and face-outside-the-box
checks need per-sample boxes the store does not keep, so they cannot be re-run; every other
capture gate is the app's current code.

Stage 2 is the run NB_P3_37's Part 0 fixed (commit 6985df6) before any camera beat touched this
set, and it runs once. It rebuilds stage 1's rows recording by recording through the same
function, so its reference side - the funnel, the floor, the primary set - must reproduce stage
1's summary exactly, and adds the camera:

- **Beats.** `check_rice_hrv.camera_beats`, unchanged: PhysNet v2 and POS on all 300
  recordings, v3 on the 60 of the validation split (it trained on the other 240); POS
  inverted; the store's `frames` as stored.
- **Clock.** One offset per recording against the contact PPG's upstrokes - the harness's
  median-distance search, +-0.4 s and then +-0.25 s around it - moves the camera onto the BIO
  clock, which the ECG shares. The finger oximeter keeps its own clock and gets its own offset
  by the same search.
- **Scoring.** SDNN and RMSSD of the kept intervals over the recording's span, against the
  contact PPG and against ECG. A comparison is scored only where both sides keep 30
  intervals; in the primary set the references always do, so an unscored camera is a miss
  and stays in the denominator.
- **The rule's figures.** Shares within 10% with Wilson intervals and the three outcomes, the
  paired bootstrap that is condition 3 of the word "HRV", the strata at stage 1's cuts, the
  RMSSD trigger and the hypotheses' point estimates. NB_P3_37 checks them and states the
  outcome.
- **Description.** Timing against the contact PPG, the jitter each excess implies, the
  oximeter, pNN50, HF-band power and the breathing-band amplitude at the belt's rate, and the
  forced and flagged recordings - each on its own, never pooled.
- **The self-test first:** the contact PPG's upstrokes scored as a camera against themselves,
  which must read zero error, every recording within 10% and 0.0 ms.

Every estimator's beat times are saved beside the outputs, which stay local, for the
after-items.

Usage:
    python -m app.live_vitals.scripts.check_vv_hrv [--stage 1|2] [--store PATH]
        [--device cpu|cuda] [--limit N] [--out-dir DIR]

No frames are rendered or written. The store is read-only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import butter, detrend, find_peaks, lombscargle, sosfiltfilt

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals.capture.motion import motion_verdict
from app.live_vitals.capture.session import (face_found_notes, framing_verdict, rate_verdict,
                                             worst_verdict)
from app.live_vitals.config import CLIP_LEN, WINDOW_STRIDE
from app.live_vitals.estimators import registry
from app.live_vitals.scripts.bench_outputs import file_sha256, provenance, subset_suffix
from app.live_vitals.scripts.check_hrv_from_video import (ANCHOR_HALF_WIDTH_S, MATCH_TOLERANCE_S,
                                                          MAX_LAG_S, align_offset, nearest_gaps,
                                                          pooled_offset)
from app.live_vitals.scripts.check_hrv_sampling_ceiling import dominant_period, refine
from app.live_vitals.scripts.check_rice_hrv import (BEAT_BAND_HZ, BEAT_SPACING, HR_MAX_BPM,
                                                    HR_MIN_BPM, IBI_LOCAL_TOL, INVERTED,
                                                    MIN_SPAN_INTERVALS, SPAN_S, camera_beats,
                                                    implied_jitter, kept_intervals, load_physnet,
                                                    span_hrv, steepest_rises)
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

# Stage 2, as NB_P3_37's Part 0 fixed it.
CAMERA_ESTIMATORS = ("hr_physnet_v2", "hr_spectral")   # every recording
CANDIDATE = "hr_physnet_v3"   # trained on VitalVideos' train split, so scored on val alone
CANDIDATE_SPLIT = "val"
PREFIX = {"hr_physnet_v2": "v2", "hr_spectral": "pos", "hr_physnet_v3": "v3"}
HRV_MARGIN = -0.10         # condition 3: the paired lower bound of v2 minus the contact PPG
RMSSD_TRIGGER = 0.25       # v2's RMSSD share against the contact PPG that reopens RMSSD and fusion
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 42
NN50_S = 0.05
HF_BAND_HZ = (0.15, 0.40)
HF_STEP_HZ = 0.005
# Stage 1's tercile cuts on its primary set, at full precision (NB_P3_37's freeze record).
ECG_SDNN_CUTS_MS = (17.828441526103333, 29.73209391696718)
POS_R_CUTS = (-0.8475013083159053, -0.7588309270069545)
TERCILES = ("lowest", "middle", "highest")
STRATA = {"fitzpatrick_group": ("1-2", "3-4", "5-6"),
          "site_group": ("CH*", "CRC/SCH/BET", "MOL/RING"),
          "age_band": ("<=40", "41-60", ">60"),
          "ecg_sdnn_tercile": TERCILES,
          "qc_pos_bvp_r_tercile": TERCILES}
CAMERA_COLUMNS = ("n_beats", "offset_ms", "offset_margin_ms", "matched", "timing_ms", "n_kept",
                  "sdnn", "rmssd", "scored_ppg", "sdnn_error_ppg", "rmssd_error_ppg",
                  "scored_ecg", "sdnn_error_ecg", "rmssd_error_ecg", "jitter_sdnn_ms",
                  "jitter_rmssd_ms", "pnn50_diff_ppg", "pnn50_diff_ecg", "hf_error_ppg",
                  "hf_error_ecg", "breath_error_ppg", "breath_error_ecg", "cms_offset_ms",
                  "cms_offset_margin_ms", "sdnn_error_cms")

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


# --------------------------------------------------------------- stage 1: the run --

def reference_row(name, group, estimator):
    """One recording's stage-1 row, and what stage 2 needs from the same pass.

    Both stages build their rows here, so stage 2's reference side is stage 1's by
    construction: the same beats, the same accepted set, the same primary set.
    """
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
    row = dict(
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
        rmssd_error_self=relative_error(ecg_hrv["rmssd"], ecg_hrv["rmssd"]))
    return row, dict(attrs=attrs, fps=fps, frames=frames, t_end=t_end, r_times=r_times,
                     upstrokes=upstrokes)


def _progress(i, n, t0):
    if i % 25 == 0 or i == n:
        print(f"  {i}/{n} recordings | {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def evaluate(store_path, estimator, limit):
    """Stage 1: one row per recording - references, floor, capture gates, reading status, strata."""
    rows = []
    with h5py.File(store_path, "r") as store:
        names = sorted(store.keys())[:limit] if limit else sorted(store.keys())
        t0 = time.perf_counter()
        for i, name in enumerate(names, 1):
            rows.append(reference_row(name, store[name], estimator)[0])
            _progress(i, len(names), t0)
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


def stage1_settings():
    """The reference side's settings and the rule's, as stage 1's summary records them."""
    return dict(qrs_band_hz=list(QRS_BAND_HZ), energy_window_s=ENERGY_WINDOW_S,
                r_spacing=R_SPACING, r_energy_share=R_ENERGY_SHARE,
                r_search_s=R_SEARCH_S, ptt_window_s=list(PTT_WINDOW_S),
                ref_min_kept_share=REF_MIN_KEPT_SHARE,
                min_span_intervals=MIN_SPAN_INTERVALS, tolerance=TOLERANCE,
                tolerance_reported=TOLERANCE_REPORTED, bar_lower_bound=BAR)


def build_summary(rows, args, estimator):
    """The JSON summary: settings, aggregates and provenance."""
    return dict(
        dataset=DATASET, stage=1, store=str(args.store), limit=args.limit,
        status_estimator=estimator.name,
        settings=stage1_settings(),
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


# ------------------------------------------------------------ stage 2: the camera --

def tercile(value, cuts):
    """The tercile a value falls in at fixed cuts; a value on a cut joins the lower one."""
    if not np.isfinite(value):
        return "n/a"
    return TERCILES[0] if value <= cuts[0] else (TERCILES[1] if value <= cuts[1] else TERCILES[2])


def on_clock(beats, reference):
    """The offset (s) that moves beats onto a reference's clock, and its margin (ms).

    The harness's search: the smallest median distance between paired beats over +-0.4 s
    (`pooled_offset`; a participant has one recording, so the pool is that recording), then
    within +-0.25 s of it (`align_offset`). The margin is how much worse the best alignment
    more than 0.25 s away fits; a few milliseconds means a whole-beat slip would fit as well.
    """
    anchor = pooled_offset([(beats, reference)])
    offset, _, _, margin = align_offset(beats, reference, limit_s=ANCHOR_HALF_WIDTH_S,
                                        centre_s=anchor)
    return offset, margin


def _inside(ends, ibi, kept, t0, t1):
    """The kept intervals ending in [t0, t1] - their end times (s), lengths (ms) and successive
    differences (ms) - taking a difference only where both intervals are kept, as `span_hrv`."""
    inside = kept & (ends >= t0) & (ends <= t1)
    pair = inside[1:] & inside[:-1]
    return ends[inside], ibi[inside] * 1000.0, np.diff(ibi)[pair] * 1000.0


def secondaries(ends, ibi, kept, t0, t1, breath_hz):
    """pNN50 (%), HF-band power (ms²) and the breathing-band amplitude (ms): description only.

    HF power is the Lomb-Scargle power of the kept intervals at their end times, mean removed,
    taken as each fitted sinusoid's variance (amplitude² / 2) so that it does not grow with
    the number of intervals, and summed over 0.15-0.40 Hz in 0.005 Hz steps. Intervals the
    camera loses still change the spectral window - a third of them lost in a regular pattern
    adds about a quarter to a pure swing's band sum - which is one reason it decides nothing.
    The breathing-band amplitude is a sinusoid at the belt's rate fitted with a constant by
    least squares, NaN without a belt rate; a single frequency's fit is unmoved by lost
    intervals. All three are NaN below MIN_SPAN_INTERVALS kept intervals.
    """
    nan = float("nan")
    t, values, diffs = _inside(ends, ibi, kept, t0, t1)
    if values.size < MIN_SPAN_INTERVALS:
        return dict(pnn50=nan, hf=nan, breath=nan)
    pnn50 = float(100.0 * np.mean(np.abs(diffs) > NN50_S * 1000.0)) if diffs.size else nan
    hz = np.arange(HF_BAND_HZ[0], HF_BAND_HZ[1] + HF_STEP_HZ / 2, HF_STEP_HZ)
    amplitude = np.abs(lombscargle(t, values - values.mean(), 2 * np.pi * hz,
                                   normalize="amplitude"))
    breath = nan
    if np.isfinite(breath_hz) and breath_hz > 0:
        phase = 2 * np.pi * breath_hz * t
        design = np.column_stack([np.ones_like(t), np.cos(phase), np.sin(phase)])
        _, a, b = np.linalg.lstsq(design, values, rcond=None)[0]
        breath = float(np.hypot(a, b))
    return dict(pnn50=pnn50, hf=float(np.sum(amplitude ** 2 / 2.0)), breath=breath)


def compare(camera, reference, t_end, breath_hz):
    """Camera beats against reference beats on one clock, over [0, t_end].

    The errors are NaN unless both sides keep MIN_SPAN_INTERVALS intervals. In the primary
    set the references always do, so a NaN there is the camera's miss.
    """
    nan = float("nan")
    cam, ref = kept_intervals(camera), kept_intervals(reference)
    c_sdnn, c_rmssd, c_n = span_hrv(*cam, 0.0, t_end)
    r_sdnn, r_rmssd, r_n = span_hrv(*ref, 0.0, t_end)
    scored = bool(min(c_n, r_n) >= MIN_SPAN_INTERVALS and np.isfinite(c_sdnn)
                  and np.isfinite(r_sdnn))
    found = dict(n_kept=int(c_n), sdnn=c_sdnn, rmssd=c_rmssd, scored=scored)
    if not scored:
        return dict(found, sdnn_error=nan, rmssd_error=nan, jitter_sdnn_ms=nan,
                    jitter_rmssd_ms=nan, pnn50_diff=nan, hf_error=nan, breath_error=nan)
    c2 = secondaries(*cam, 0.0, t_end, breath_hz)
    r2 = secondaries(*ref, 0.0, t_end, breath_hz)
    return dict(found, sdnn_error=relative_error(c_sdnn, r_sdnn),
                rmssd_error=relative_error(c_rmssd, r_rmssd),
                jitter_sdnn_ms=implied_jitter(c_sdnn, r_sdnn, 2.0),
                jitter_rmssd_ms=implied_jitter(c_rmssd, r_rmssd, 6.0),
                pnn50_diff=c2["pnn50"] - r2["pnn50"], hf_error=relative_error(c2["hf"], r2["hf"]),
                breath_error=relative_error(c2["breath"], r2["breath"]))


def camera_columns(prefix, beats, refs):
    """One estimator's columns for one recording, each named `<prefix>_<column>`.

    `refs` holds the recording's reference beats - `upstrokes` and `r_times` on the BIO clock,
    `cms` on the oximeter's - with `t_end`, `cms_end` and `breath_hz`. The camera moves onto the
    BIO clock by its offset against the contact PPG, and onto the oximeter's by its own.
    """
    nan = float("nan")
    beats = np.asarray(beats, dtype=float)
    upstrokes = refs["upstrokes"]
    offset, margin = on_clock(beats, upstrokes)
    camera = beats + offset
    gaps = (nearest_gaps(camera, upstrokes) if camera.size >= 3 and upstrokes.size
            else np.array([np.inf]))
    near = gaps <= MATCH_TOLERANCE_S
    ppg = compare(camera, upstrokes, refs["t_end"], refs["breath_hz"])
    ecg = compare(camera, refs["r_times"], refs["t_end"], refs["breath_hz"])
    cms_offset, cms_margin = on_clock(beats, refs["cms"])
    cms = compare(beats + cms_offset, refs["cms"], refs["cms_end"], refs["breath_hz"])
    values = dict(
        n_beats=int(beats.size), offset_ms=offset * 1000.0, offset_margin_ms=margin,
        matched=float(np.mean(near)),
        timing_ms=float(np.median(gaps[near]) * 1000.0) if near.any() else nan,
        n_kept=ppg["n_kept"], sdnn=ppg["sdnn"], rmssd=ppg["rmssd"],
        scored_ppg=ppg["scored"], sdnn_error_ppg=ppg["sdnn_error"],
        rmssd_error_ppg=ppg["rmssd_error"],
        scored_ecg=ecg["scored"], sdnn_error_ecg=ecg["sdnn_error"],
        rmssd_error_ecg=ecg["rmssd_error"],
        jitter_sdnn_ms=ppg["jitter_sdnn_ms"], jitter_rmssd_ms=ppg["jitter_rmssd_ms"],
        pnn50_diff_ppg=ppg["pnn50_diff"], pnn50_diff_ecg=ecg["pnn50_diff"],
        hf_error_ppg=ppg["hf_error"], hf_error_ecg=ecg["hf_error"],
        breath_error_ppg=ppg["breath_error"], breath_error_ecg=ecg["breath_error"],
        cms_offset_ms=cms_offset * 1000.0, cms_offset_margin_ms=cms_margin,
        sdnn_error_cms=cms["sdnn_error"])
    return {f"{prefix}_{name}": values[name] for name in CAMERA_COLUMNS}


def not_run(prefix):
    """The columns of an estimator not run on a recording: never scored, every value NaN."""
    return {f"{prefix}_{name}": (False if name.startswith("scored") else float("nan"))
            for name in CAMERA_COLUMNS}


def evaluate_stage2(store_path, status, cameras, candidate, device, limit):
    """Stage 2: stage 1's rows rebuilt, every estimator's camera columns, and the beats."""
    models = {e.name: (None if e.name == "hr_spectral" else load_physnet(e, device))
              for e in (*cameras, candidate)}
    rows, beats = [], {}
    with h5py.File(store_path, "r") as store:
        names = sorted(store.keys())[:limit] if limit else sorted(store.keys())
        t0 = time.perf_counter()
        for i, name in enumerate(names, 1):
            group = store[name]
            row, refs = reference_row(name, group, status)
            attrs = refs["attrs"]
            cms_fs = float(attrs["cms_fs"])
            refs.update(cms=ppg_upstrokes(group["cms_ppg"][()], cms_fs),
                        cms_end=len(group["cms_ppg"]) / cms_fs,
                        breath_hz=float(attrs["rr_belt"]) / 60.0 if bool(attrs["resp_ok"])
                        else float("nan"))
            row.update(ecg_sdnn_tercile=tercile(row["sdnn_ecg"], ECG_SDNN_CUTS_MS),
                       qc_pos_bvp_r_tercile=tercile(row["qc_pos_bvp_r"], POS_R_CUTS),
                       resp_ok=bool(attrs["resp_ok"]), rr_belt=float(attrs["rr_belt"]),
                       note_cms_bad=bool(attrs["note_cms_bad"]),
                       n_cms_upstrokes=int(refs["cms"].size))
            row.update(camera_columns("self", refs["upstrokes"], refs))
            beats.update({f"{name}/ecg": refs["r_times"], f"{name}/ppg": refs["upstrokes"],
                          f"{name}/cms": refs["cms"]})
            for estimator in (*cameras, candidate):
                prefix = PREFIX[estimator.name]
                if estimator is candidate and attrs["split"] != CANDIDATE_SPLIT:
                    row.update(not_run(prefix))
                    continue
                found, _ = camera_beats(estimator, models[estimator.name], refs["frames"],
                                        refs["fps"], device)
                beats[f"{name}/{prefix}"] = found
                row.update(camera_columns(prefix, found, refs))
            rows.append(row)
            _progress(i, len(names), t0)
    return rows, beats


def hits(values, tolerance=TOLERANCE):
    """Which errors lie within the tolerance; a missing error is a miss."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    return finite & (np.abs(np.where(finite, v, 0.0)) <= tolerance)


def share(rows, key):
    """The rule's share within 10% over a set, a missing error counting as a miss.

    Unlike stage 1's `error_summary`, which describes the errors there are, this keeps every
    recording of the set in the denominator: a camera that yields no SDNN has failed there.
    """
    v = np.array([r[key] for r in rows], dtype=float)
    if not v.size:
        return dict(n=0)
    k = int(hits(v).sum())
    low, high = wilson(k, v.size)
    finite = v[np.isfinite(v)]
    middle = ([float(np.percentile(finite, q)) for q in (25, 50, 75)] if finite.size
              else [None, None, None])
    return dict(n=int(v.size), k=k, within_10=k / v.size, within_10_low=low,
                within_10_high=high, within_10_outcome=outcome(low, high),
                within_20=float(hits(v, TOLERANCE_REPORTED).mean()), n_scored=int(finite.size),
                median=middle[1], q1=middle[0], q3=middle[2])


def paired_bootstrap(a, b, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    """mean(a) - mean(b) with its 95% percentile interval, resampling recordings together."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if not a.size:
        return dict(n=0)
    index = np.random.default_rng(seed).integers(0, a.size, size=(draws, a.size))
    diff = a[index].mean(axis=1) - b[index].mean(axis=1)
    return dict(n=int(a.size), difference=float(a.mean() - b.mean()),
                low=float(np.percentile(diff, 2.5)), high=float(np.percentile(diff, 97.5)))


def hrv_earned(prv, floor, camera_ecg, margin_low):
    """Part 0's word "HRV": the PRV bar confirmed, then all three conditions met."""
    return bool(prv == "confirmed" and floor == "confirmed" and camera_ecg == "confirmed"
                and margin_low is not None and margin_low >= HRV_MARGIN - 1e-12)


def _spread(rows, key):
    """Median and interquartile range of a column's finite values, with their count."""
    v = np.array([r[key] for r in rows], dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)), q1=float(np.percentile(v, 25)),
                q3=float(np.percentile(v, 75)))


def _column(prefix, subset):
    """One estimator's figures over a set of recordings."""
    return dict(
        n=len(subset),
        ppg=dict(sdnn=share(subset, f"{prefix}_sdnn_error_ppg"),
                 rmssd=share(subset, f"{prefix}_rmssd_error_ppg")),
        ecg=dict(sdnn=share(subset, f"{prefix}_sdnn_error_ecg"),
                 rmssd=share(subset, f"{prefix}_rmssd_error_ecg")),
        oximeter=share([r for r in subset if not r["note_cms_bad"]], f"{prefix}_sdnn_error_cms"),
        unscored=sum(1 for r in subset if not r[f"{prefix}_scored_ppg"]),
        timing={key: _spread(subset, f"{prefix}_{key}")
                for key in ("matched", "timing_ms", "offset_ms", "jitter_sdnn_ms",
                            "jitter_rmssd_ms")},
        secondaries={key: _spread(subset, f"{prefix}_{key}")
                     for key in ("pnn50_diff_ppg", "pnn50_diff_ecg", "hf_error_ppg",
                                 "hf_error_ecg", "breath_error_ppg", "breath_error_ecg")})


def camera_summary(rows):
    """Stage 2's figures: the rule's shares, condition 3, strata, trigger and description.

    Aggregates only; no recording appears beside its skin type, site or age.
    """
    primary = [r for r in rows if r["primary"]]
    candidate = [r for r in primary if r["split"] == CANDIDATE_SPLIT]
    columns = {"v2": _column("v2", primary), "pos": _column("pos", primary),
               "v3": _column("v3", candidate)}
    self_errors = np.array([r["self_sdnn_error_ppg"] for r in primary], dtype=float)
    self_test = dict(_column("self", primary),
                     largest_error=float(np.max(np.abs(self_errors)))
                     if self_errors.size and np.isfinite(self_errors).all() else None)
    floor = share(primary, "sdnn_error_ppg")
    margin = paired_bootstrap(hits([r["v2_sdnn_error_ecg"] for r in primary]),
                              hits([r["sdnn_error_ppg"] for r in primary]))
    prv = columns["v2"]["ppg"]["sdnn"].get("within_10_outcome")
    camera_ecg = columns["v2"]["ecg"]["sdnn"].get("within_10_outcome")
    hrv = dict(prv=prv, floor=floor.get("within_10_outcome"), camera_ecg=camera_ecg,
               margin=margin, margin_bound=HRV_MARGIN,
               margin_met=bool(margin.get("low") is not None
                               and margin["low"] >= HRV_MARGIN - 1e-12),
               earned=hrv_earned(prv, floor.get("within_10_outcome"), camera_ecg,
                                 margin.get("low")))
    paired = dict(
        v2_minus_pos=paired_bootstrap(hits([r["v2_sdnn_error_ppg"] for r in primary]),
                                      hits([r["pos_sdnn_error_ppg"] for r in primary])),
        v3_minus_v2=paired_bootstrap(hits([r["v3_sdnn_error_ppg"] for r in candidate]),
                                     hits([r["v2_sdnn_error_ppg"] for r in candidate])))
    strata = {ref: {axis: {level: share([r for r in primary if r[axis] == level],
                                        f"v2_sdnn_error_{ref}") for level in levels}
                    for axis, levels in STRATA.items()}
              for ref in ("ppg", "ecg")}
    outcomes = [(f"{axis} {level}", m.get("within_10_outcome"))
                for axis, levels in strata["ppg"].items() for level, m in levels.items()]
    rmssd = columns["v2"]["ppg"]["rmssd"].get("within_10")
    apart = {}
    for label, subset in (("forced", [r for r in rows if not r["accepted"]]),
                          ("flagged", [r for r in rows if r["accepted"] and not r["primary"]])):
        apart[label] = {}
        for prefix in PREFIX.values():
            ran = [r for r in subset if prefix != "v3" or r["split"] == CANDIDATE_SPLIT]
            apart[label][prefix] = dict(
                n=len(ran), ppg=share(ran, f"{prefix}_sdnn_error_ppg"),
                ecg=share([r for r in ran if r["ecg_usable"]], f"{prefix}_sdnn_error_ecg"))
    return dict(
        n_primary=len(primary), n_candidate=len(candidate), self_test=self_test,
        columns=columns, floor=floor, hrv=hrv, paired=paired, strata=strata,
        strata_not_confirmed=[name for name, verdict in outcomes if verdict == "not confirmed"],
        strata_inconclusive=sum(1 for _, verdict in outcomes if verdict == "inconclusive"),
        rmssd_trigger=dict(share=rmssd, threshold=RMSSD_TRIGGER,
                           met=bool(rmssd is not None and rmssd >= RMSSD_TRIGGER)),
        apart=apart)


def hypotheses(c):
    """Part 0's hypotheses, each judged on its point estimate; H2 was answered by stage 1."""
    eps = 1e-9
    within = lambda m: m.get("within_10")  # noqa: E731
    v2 = within(c["columns"]["v2"]["ppg"]["sdnn"])
    pos = within(c["columns"]["pos"]["ppg"]["sdnn"])
    terciles = {k: within(m) for k, m in c["strata"]["ppg"]["ecg_sdnn_tercile"].items()}
    sites = {k: within(m) for k, m in c["strata"]["ppg"]["site_group"].items()}
    rmssd = c["rmssd_trigger"]["share"]
    if None in (v2, pos, rmssd, *terciles.values(), *sites.values()):
        return {name: dict(verdict="not estimable")
                for name in ("H1", "H2b", "H3", "H4", "H5", "H6")}
    level_gap = terciles["highest"] - terciles["lowest"]
    site_gap = max(sites.values()) - sites["CH*"]
    judged = (
        ("H1", "v2 within 10% of the contact PPG in 45-65%", v2, 0.45 - eps <= v2 <= 0.65 + eps),
        ("H2b", 'the word "HRV" stays withheld', c["hrv"]["earned"], not c["hrv"]["earned"]),
        ("H3", "lowest ECG-SDNN tercile at least 15 points below the highest", level_gap,
         level_gap >= 0.15 - eps),
        ("H4", "CH* the weakest site group, at least 10 points below the best", site_gap,
         sites["CH*"] <= min(sites.values()) + eps and site_gap >= 0.10 - eps),
        ("H5", "v2's RMSSD within 10% of the contact PPG's in 15-25%", rmssd,
         0.15 - eps <= rmssd <= 0.25 + eps),
        ("H6", "v2 above POS by 5-15 points", v2 - pos, 0.05 - eps <= v2 - pos <= 0.15 + eps))
    return {name: dict(prediction=text, value=value,
                       verdict="consistent" if ok else "contradicted")
            for name, text, value, ok in judged}


def stage2_settings():
    """The camera harness's settings and the rule's, as Part 0 states them."""
    return dict(beat_band_hz=list(BEAT_BAND_HZ), beat_spacing=BEAT_SPACING,
                hr_range_bpm=[HR_MIN_BPM, HR_MAX_BPM], local_interval_tolerance=IBI_LOCAL_TOL,
                span_s=SPAN_S, inverted=list(INVERTED),
                reconstruction_window_stride=[CLIP_LEN, WINDOW_STRIDE],
                offset_search_s=MAX_LAG_S, offset_around_anchor_s=ANCHOR_HALF_WIDTH_S,
                beat_match_s=MATCH_TOLERANCE_S, wilson_z=Z95, hrv_margin=HRV_MARGIN,
                rmssd_trigger=RMSSD_TRIGGER, bootstrap_draws=BOOTSTRAP_DRAWS,
                bootstrap_seed=BOOTSTRAP_SEED, nn50_s=NN50_S, hf_band_hz=list(HF_BAND_HZ),
                hf_step_hz=HF_STEP_HZ, ecg_sdnn_cuts_ms=list(ECG_SDNN_CUTS_MS),
                qc_pos_bvp_r_cuts=list(POS_R_CUTS))


def build_summary_stage2(rows, args, status, candidate):
    """Stage 2's JSON summary: stage 1's sections rebuilt, the camera's, and provenance."""
    camera = camera_summary(rows)
    camera["hypotheses"] = hypotheses(camera)
    weights = Path(candidate.checkpoint)
    return dict(
        dataset=DATASET, stage=2, store=str(args.store), limit=args.limit,
        status_estimator=status.name, cameras=list(CAMERA_ESTIMATORS),
        candidate=dict(name=candidate.name, split=CANDIDATE_SPLIT,
                       weights=dict(file=weights.name, sha256=file_sha256(weights),
                                    bytes=weights.stat().st_size)),
        settings=dict(stage1_settings(), **stage2_settings()),
        not_rerun=["second face", "face outside the held box"],
        **summarise(rows),
        camera=camera,
        provenance=provenance(status),
    )


def _rule(m):
    """A share as the rule reads it."""
    if not m.get("n"):
        return "n/a"
    median = f", median {m['median']:+.1%}" if m.get("median") is not None else ""
    return (f"{m['within_10']:.1%} within 10% (95% {m['within_10_low']:.1%}-"
            f"{m['within_10_high']:.1%}, {m['within_10_outcome']}){median}, within 20% "
            f"{m['within_20']:.0%}, n={m['n']}")


def _diff(m):
    if not m.get("n"):
        return "n/a"
    return f"{m['difference']:+.1%} (95% {m['low']:+.1%} to {m['high']:+.1%}, n={m['n']})"


def _middle(m, unit="", scale=1.0, fmt=".1f"):
    if not m.get("n"):
        return "n/a"
    return f"{m['median'] * scale:{fmt}}{unit} ({m['q1'] * scale:{fmt}}-{m['q3'] * scale:{fmt}})"


def print_summary_stage2(s):
    """The self-test first, then the rule's figures in Part 0's order, then description."""
    c = s["camera"]
    st, f = c["self_test"], s["funnel"]
    print(f"\n=== VitalVideos stage 2 | {s['n_recordings']} recordings | "
          f"primary {c['n_primary']} ===")
    largest = st["largest_error"]
    print(f"self-test, the contact PPG scored as a camera against itself: "
          f"{_rule(st['ppg']['sdnn'])} | timing {_middle(st['timing']['timing_ms'], ' ms')} | "
          f"largest |error| {largest if largest is None else f'{largest:.1e}'}")
    print(f"reference side, as stage 1: capture gates pass {f['capture_not_rejected']} | accepted "
          f"{f['accepted']} | no ECG or cardiac note {f['accepted_unflagged']} | primary "
          f"{f['primary']}")
    print(f"floor, contact PPG against ECG: {_rule(c['floor'])}")
    v2, pos, v3 = (c["columns"][k] for k in ("v2", "pos", "v3"))
    print(f"PRV, v2 against the contact PPG: {_rule(v2['ppg']['sdnn'])} | no SDNN in "
          f"{v2['unscored']}")
    print(f"v2 against ECG: {_rule(v2['ecg']['sdnn'])}")
    h = c["hrv"]
    print(f'condition 3, v2 minus the contact PPG against ECG: {_diff(h["margin"])} - '
          f'{"met" if h["margin_met"] else "not met"}; "HRV" {"is" if h["earned"] else "is not"} '
          f"earned")
    print(f"POS: {_rule(pos['ppg']['sdnn'])}")
    print(f"v3 on the validation split: {_rule(v3['ppg']['sdnn'])}")
    print(f"  v2 minus POS {_diff(c['paired']['v2_minus_pos'])} | v3 minus v2 "
          f"{_diff(c['paired']['v3_minus_v2'])}")
    t = c["rmssd_trigger"]
    print(f"RMSSD, v2 against the contact PPG: {_rule(v2['ppg']['rmssd'])} -> trigger at "
          f"{RMSSD_TRIGGER:.0%} {'met' if t['met'] else 'not met'}")
    print(f"strata, v2 against the contact PPG: not confirmed {c['strata_not_confirmed'] or 'none'}"
          f" | inconclusive {c['strata_inconclusive']} of "
          f"{sum(len(levels) for levels in STRATA.values())}")
    print("hypotheses: " + " | ".join(f"{name} {v['verdict']}"
                                      for name, v in c["hypotheses"].items()))
    print("timing against the contact PPG (median recording, IQR): " + "; ".join(
        f"{name} matched {_middle(col['timing']['matched'], '%', 100.0, '.0f')}, "
        f"{_middle(col['timing']['timing_ms'], ' ms')}" for name, col in (("v2", v2), ("POS", pos),
                                                                          ("v3", v3))))
    print(f"oximeter: v2 {_rule(v2['oximeter'])} | the contact PPG itself {_rule(st['oximeter'])}")
    print("secondaries, v2 against the contact PPG (median, IQR): pNN50 "
          f"{_middle(v2['secondaries']['pnn50_diff_ppg'], ' points')} | HF power "
          f"{_middle(v2['secondaries']['hf_error_ppg'], '%', 100.0, '+.0f')} | breathing band "
          f"{_middle(v2['secondaries']['breath_error_ppg'], '%', 100.0, '+.0f')}")
    for label in ("forced", "flagged"):
        g = c["apart"][label]
        print(f"apart, {label}: " + "; ".join(f"{p} {_rule(g[p]['ppg'])}"
                                              for p in ("v2", "pos", "v3")))


def run_stage2(args, status):
    """Stage 2: the camera run NB_P3_37's Part 0 fixed, once."""
    if args.limit and Path(args.store).resolve() == DEFAULT_STORE.resolve():
        raise SystemExit("ERROR: stage 2 runs once, on every recording; a --limit run would put "
                         "camera beats on VitalVideos before that run (NB_P3_37 Part 0)")
    pos = registry.get_estimator("hr_spectral")
    candidate = registry.get_estimator(CANDIDATE, device=args.device)
    if not candidate.is_available():
        raise SystemExit(f"ERROR: checkpoint missing for {CANDIDATE}")
    if any(dataset == DATASET for dataset, _ in pos.trained_on):
        raise SystemExit(f"ERROR: {pos.name} trained on {DATASET}")
    if {split for dataset, split in candidate.trained_on if dataset == DATASET} != {"train"}:
        raise SystemExit(f"ERROR: {CANDIDATE} must have trained on {DATASET}'s train split alone")
    print(f"store: {args.store}\nstage 2, the camera run | v2 and POS on every recording, v3 on "
          f"the {CANDIDATE_SPLIT} split | device {args.device}"
          + (f" | limit {args.limit}" if args.limit else ""))

    rows, beats = evaluate_stage2(args.store, status, (status, pos), candidate, args.device,
                                  args.limit)
    suffix = subset_suffix(("limit", args.limit or None, None))
    summary = build_summary_stage2(rows, args, status, candidate)
    _write_csv(args.out_dir / f"stage2_recordings{suffix}.csv", rows)
    np.savez_compressed(args.out_dir / f"stage2_beats{suffix}.npz", **beats)
    (args.out_dir / f"summary_stage2{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print_summary_stage2(summary)
    print(f"\noutputs: {args.out_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, choices=(1, 2), default=1,
                        help="1: the references, camera-free (default); 2: the camera run, once")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--device", default="cpu", help="torch device for the PhysNet models")
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
    if args.stage == 2:
        return run_stage2(args, estimator)
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