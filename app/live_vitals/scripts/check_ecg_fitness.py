"""Refusal benchmark on ECG-Fitness, implementing NB_P3_29 Part 0 as written.

Part 0 was fixed before any estimator saw an ECG-Fitness frame. Its question: when the app
cannot see a pulse - violent motion, high heart rate, poor light - does it decline, or does
it report a confident wrong number? Each estimator runs through `estimate(frames, fps)`, the
app's own windows, gates and aggregation, at the capture clock's measured rate, and is
scored against the per-window ECG consensus frozen in the store. No threshold is changed
for this corpus.

Primary set: recordings with a reference in at least 80% of their windows, less those the
protocol reports separately. Primary metrics, per activity and never pooled:
1. refusal - recordings with no reported value, and windows the gates discard;
2. confident-wrong - windows the app keeps, in recordings it reports, more than 10 bpm off
   the reference;
3. accuracy on those windows (MAE, bias), beside the coverage that produced it.

Reported separately: recordings below the reference gate; crops clamped beyond 2% of their
side (whether the app's framing gate refuses them is itself a result); recordings whose
face was found in fewer than half the sampled frames; windows whose real span is more than
5% off nominal. Hypotheses H1-H3 are read out at the end as numbers for review, with a
mechanical first reading from subject-bootstrap intervals; the verdict is made in review.

The store is evaluation only and its licence forbids redistribution. Nothing written here
contains a frame.

Usage:
    python -m app.live_vitals.scripts.check_ecg_fitness [--estimator NAME ...]
        [--store PATH] [--device cpu|cuda] [--limit N] [--out-dir DIR] [--boot N] [--seed N]
"""
import argparse
import csv
import inspect
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals import config
from app.live_vitals.capture.session import framing_verdict, rate_verdict, worst_verdict
from app.live_vitals.estimators import registry

DEFAULT_STORE = Path(r"D:\QualityPhys\phase 3 benchmarks\ecg_fitness_phase3_rc_none.h5")
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "ecg_fitness_eval"
ACTIVITIES = ("01", "02", "03", "04", "05", "06")
REPORTED = ("ok", "degraded_capture", "unstable")  # statuses that carry a value
WRONG_BPM = 10.0           # Part 0: a kept window further off than this is confident-wrong
REF_MIN_AGREE = 2          # a window has a reference when two readouts agree (NB_P3_29)
REF_RECORDING_MIN = 0.80   # a recording enters the primary set at this referenced fraction
SPAN_TOLERANCE = 0.05      # a window whose real span is further off nominal is reported apart
LOW_DETECTION = config.N_DETECT_FRAMES / 2  # fewer detections than this: reported apart
NAN = float("nan")


# ------------------------------------------------------------------- reading --

def window_grid(n_frames):
    """The app's analysis-window starts, which the frozen reference must match."""
    return list(range(0, n_frames - config.CLIP_LEN + 1, config.WINDOW_STRIDE))


def read_recording(name, group):
    """Reads one group: frames, the frozen reference, the capture gates and its set.

    The capture gates are re-run with the app's current code on the stored box, so the
    benchmark refuses exactly what the app would refuse today. The rate is the capture
    clock's: an uncompressed AVI's container stamps only repeat the declared 30 fps.
    """
    attrs = {k: _plain(v) for k, v in group.attrs.items()}
    if attrs.get("role") != "benchmark":
        raise SystemExit(f"ERROR: {name} has role {attrs.get('role')!r}; "
                         f"this is not the ECG-Fitness benchmark store")
    fps = float(attrs["fps"])
    frames = group["frames"][()]
    starts = [int(s) for s in group["ref_window_start"][()]]
    if starts != window_grid(len(frames)):
        raise SystemExit(f"ERROR: {name}: the frozen reference windows do not match the app's "
                         f"window grid (CLIP_LEN {config.CLIP_LEN}, stride {config.WINDOW_STRIDE})")
    ref_hr = group["ref_hr"][()].astype(float)
    agree = group["ref_agree"][()].astype(int)
    referenced = (agree >= REF_MIN_AGREE) & np.isfinite(ref_hr)
    span_s = group["ref_span_s"][()].astype(float)
    off_span = np.abs(span_s / ((config.CLIP_LEN - 1) / fps) - 1.0) > SPAN_TOLERANCE

    box = tuple(float(attrs[k]) for k in ("box_rx0", "box_ry0", "box_rx1", "box_ry1"))
    framing, report, framing_notes = framing_verdict(
        box, int(attrs["frame_width"]), int(attrs["frame_height"]))
    rate, rate_note = rate_verdict(fps)

    ref_frac = float(referenced.mean())
    flags = dict(below_gate=ref_frac < REF_RECORDING_MIN,
                 clamped=report["frac_side_lost"] > config.MAX_BOX_SIDE_LOST,
                 low_detection=int(attrs["face_detections"]) < LOW_DETECTION)
    # The store recorded the same three facts when it was built; a disagreement means the
    # store and this script no longer describe the same protocol.
    stored = dict(below_gate=not bool(attrs["ref_ok"]),
                  off_span=int(attrs["ref_windows_off_span"]))
    mismatch = []
    if flags["below_gate"] != stored["below_gate"]:
        mismatch.append("reference gate")
    if int(off_span.sum()) != stored["off_span"]:
        mismatch.append("off-span windows")
    if flags["clamped"] != (float(attrs["box_side_lost"]) > config.MAX_BOX_SIDE_LOST):
        mismatch.append("clamp (app definition against the per-axis box_side_lost)")

    return dict(
        name=name, attrs=attrs, fps=fps, frames=frames, starts=starts, ref_hr=ref_hr,
        ref_agree=agree, referenced=referenced,
        span_s=span_s, off_span=off_span, ref_frac=ref_frac,
        ref_hr_median=float(np.median(ref_hr[referenced])) if referenced.any() else NAN,
        framing=framing, frac_side_lost=float(report["frac_side_lost"]),
        rate=rate, capture=worst_verdict(framing, rate),
        capture_notes=[n for n in (*framing_notes, rate_note) if n],
        primary=not any(flags.values()), mismatch=mismatch, **flags)


def predicted_windows(result):
    """Returns the estimator's per-window readings keyed by window start frame."""
    detail = result.detail or {}
    if "window_start" not in detail:
        return {}
    if detail["n_total"] - detail["n_no_peak"] != len(detail["window_hr"]):
        raise RuntimeError("estimator per-window report is inconsistent")
    return {int(s): dict(pred_hr=float(h), pred_confidence=float(c), pred_kept=bool(k))
            for s, h, c, k in zip(detail["window_start"], detail["window_hr"],
                                  detail["window_confidence"], detail["window_kept"])}


# ----------------------------------------------------------------------- run --

def make_estimators(names, device):
    """Instantiates the requested estimators, passing a device where one is accepted."""
    estimators = []
    for name in names:
        cls = registry.ESTIMATORS[name]
        kwargs = {"device": device} if "device" in inspect.signature(cls).parameters else {}
        estimator = registry.get_estimator(name, **kwargs)
        if not estimator.is_available():
            raise SystemExit(f"ERROR: checkpoint missing for {name}")
        estimators.append(estimator)
    return estimators


def evaluate(store_path, estimators, limit):
    """Runs every estimator over the store; returns per-estimator recording and window rows."""
    recordings = {e.name: [] for e in estimators}
    windows = {e.name: [] for e in estimators}
    with h5py.File(store_path, "r") as store:
        names = sorted(store.keys())[:limit] if limit else sorted(store.keys())
        t0 = time.perf_counter()
        for i, name in enumerate(names, 1):
            rec = read_recording(name, store[name])
            a = rec["attrs"]
            for estimator in estimators:
                # The app never runs an estimator on a capture its gates refuse. The forced
                # result is kept only to show what would have been reported without them.
                forced = estimator.estimate(rec["frames"], rec["fps"])
                gated = rec["capture"] != "REJECT"
                status = forced.status if gated else "capture_rejected"
                value = float(forced.value) if gated else NAN
                reported = status in REPORTED
                pred = predicted_windows(forced)
                detail = forced.detail or {}
                rows = []
                for j, start in enumerate(rec["starts"]):
                    p = pred.get(start, dict(pred_hr=NAN, pred_confidence=NAN, pred_kept=False))
                    referenced = bool(rec["referenced"][j])
                    error = (p["pred_hr"] - rec["ref_hr"][j]
                             if referenced and np.isfinite(p["pred_hr"]) else NAN)
                    is_scored = bool(reported and p["pred_kept"] and referenced
                                     and not rec["off_span"][j])
                    rows.append(dict(
                        recording=name, subject=a["subject_id"], activity=a["activity"],
                        lighting=a["lighting"], motion_class=a["motion_class"],
                        primary=rec["primary"], start=start, t_start_s=round(start / rec["fps"], 3),
                        ref_hr=float(rec["ref_hr"][j]), ref_agree=int(rec["ref_agree"][j]),
                        referenced=referenced, off_span=bool(rec["off_span"][j]),
                        span_s=float(rec["span_s"][j]), no_peak=start not in pred, **p,
                        recording_status=status, error=float(error), scored=is_scored,
                        confident_wrong=bool(is_scored and abs(error) > WRONG_BPM)))
                windows[estimator.name].extend(rows)
                evaluable = [w for w in rows if not w["off_span"]]
                scored = [w for w in evaluable if w["scored"]]
                recordings[estimator.name].append(dict(
                    recording=name, subject=a["subject_id"], activity=a["activity"],
                    lighting=a["lighting"], motion_class=a["motion_class"], fps=rec["fps"],
                    primary=rec["primary"], below_gate=rec["below_gate"],
                    clamped=rec["clamped"], low_detection=rec["low_detection"],
                    ref_frac=rec["ref_frac"], ref_hr_median=rec["ref_hr_median"],
                    n_windows=len(rows), n_off_span=int(rec["off_span"].sum()),
                    face_detections=int(a["face_detections"]),
                    frac_side_lost=rec["frac_side_lost"],
                    box_side_lost_stored=float(a["box_side_lost"]),
                    framing_verdict=rec["framing"], rate_verdict=rec["rate"],
                    capture_verdict=rec["capture"], capture_notes="; ".join(rec["capture_notes"]),
                    status=status, value=value, reported=reported, refused=int(not reported),
                    confidence=float(forced.confidence) if gated else NAN,
                    usable_fraction=float(detail.get("usable_fraction", 0.0)),
                    forced_status=forced.status, forced_value=float(forced.value),
                    n_evaluable=len(evaluable),
                    n_discarded=sum(not w["pred_kept"] for w in evaluable),
                    n_scored=len(scored),
                    n_confident_wrong=sum(w["confident_wrong"] for w in scored),
                    abs_error_sum=float(sum(abs(w["error"]) for w in scored)),
                    error_sum=float(sum(w["error"] for w in scored)),
                    n_kept_unreferenced=sum(1 for w in evaluable if reported and w["pred_kept"]
                                            and not w["referenced"]),
                    error_vs_reference=float(value - rec["ref_hr_median"]),
                    store_mismatch="; ".join(rec["mismatch"])))
            if i % 10 == 0 or i == len(names):
                print(f"  {i}/{len(names)} recordings | {(time.perf_counter() - t0) / 60:.1f} min",
                      flush=True)
    return recordings, windows


# ------------------------------------------------------------------- metrics --

def subject_weights(subjects, n_boot, seed):
    """How often each subject is drawn in each bootstrap resample, as (n_boot, n_subjects)."""
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subjects), size=(n_boot, len(subjects)))
    return np.stack([np.bincount(d, minlength=len(subjects)) for d in draws]).astype(float)


class Bootstrap:
    """Percentile intervals for ratios of sums, resampling subjects rather than windows.

    Windows within a recording, and recordings of one subject, are not independent, so the
    subject is the unit drawn. Every group is evaluated on the same resamples, which makes
    a difference between two groups a paired comparison.
    """

    def __init__(self, recordings, n_boot, seed):
        subjects = sorted({r["subject"] for r in recordings})
        self.index = {s: i for i, s in enumerate(subjects)}
        self.weights = subject_weights(subjects, n_boot, seed)

    def samples(self, rows, num, den):
        """Bootstrap distribution of sum(num) / sum(den) over the given recording rows."""
        if not rows:
            return np.full(len(self.weights), NAN)
        w = self.weights[:, [self.index[r["subject"]] for r in rows]]
        n = w @ np.array([r[num] for r in rows], dtype=float)
        d = w @ np.array([r[den] if den else 1.0 for r in rows], dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(d > 0, n / d, NAN)

    @staticmethod
    def interval(samples):
        """95% percentile interval, or NaNs when too few resamples are defined."""
        finite = samples[np.isfinite(samples)]
        if finite.size < 0.5 * samples.size:
            return [NAN, NAN]
        return [float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))]


# Each metric is a ratio of sums of two recording-row fields; None counts recordings.
REFUSAL = ("refused", None)
DISCARD = ("n_discarded", "n_evaluable")
CONFIDENT_WRONG = ("n_confident_wrong", "n_scored")
COVERAGE = ("n_scored", "n_evaluable")


def ratio(rows, num, den):
    """sum(num) / sum(den) over recording rows, NaN when the denominator is zero."""
    d = sum(r[den] for r in rows) if den else len(rows)
    return sum(r[num] for r in rows) / d if d else NAN


def group_metrics(rows, boot):
    """The three primary metrics for one group of recordings, with subject-bootstrap CIs."""
    n_scored = sum(r["n_scored"] for r in rows)
    return dict(
        n_recordings=len(rows),
        n_reported=sum(r["reported"] for r in rows),
        refusal=ratio(rows, *REFUSAL),
        refusal_ci=boot.interval(boot.samples(rows, *REFUSAL)),
        n_windows=sum(r["n_evaluable"] for r in rows),
        n_discarded=sum(r["n_discarded"] for r in rows),
        window_discard=ratio(rows, *DISCARD),
        window_discard_ci=boot.interval(boot.samples(rows, *DISCARD)),
        n_scored=n_scored,
        n_confident_wrong=sum(r["n_confident_wrong"] for r in rows),
        confident_wrong=ratio(rows, *CONFIDENT_WRONG),
        confident_wrong_ci=boot.interval(boot.samples(rows, *CONFIDENT_WRONG)),
        mae=sum(r["abs_error_sum"] for r in rows) / n_scored if n_scored else NAN,
        bias=sum(r["error_sum"] for r in rows) / n_scored if n_scored else NAN,
        coverage=ratio(rows, *COVERAGE),
        n_kept_unreferenced=sum(r["n_kept_unreferenced"] for r in rows))


def per_activity(primary, boot):
    """The primary metrics for each activity separately - never pooled."""
    out = {}
    for act in ACTIVITIES:
        rows = [r for r in primary if r["activity"] == act]
        out[act] = dict(lighting=rows[0]["lighting"] if rows else None,
                        motion_class=rows[0]["motion_class"] if rows else None,
                        **group_metrics(rows, boot))
    return out


def difference(boot, rows_a, rows_b, metric):
    """Point difference a - b of a ratio metric, with its paired bootstrap interval."""
    point = ratio(rows_a, *metric) - ratio(rows_b, *metric)
    samples = boot.samples(rows_a, *metric) - boot.samples(rows_b, *metric)
    return dict(difference=point, ci=Bootstrap.interval(samples))


def reading(ci, predicted_sign):
    """Mechanical first reading of a difference against the sign a hypothesis predicts."""
    low, high = ci
    if not np.isfinite(low):
        return "not estimable"
    if (low > 0 and predicted_sign > 0) or (high < 0 and predicted_sign < 0):
        return "consistent"
    if (high < 0 and predicted_sign > 0) or (low > 0 and predicted_sign < 0):
        return "contradicted"
    return "unresolved"


def hypotheses(primary, boot):
    """H1-H3 as stated in Part 0, with paired subject-bootstrap differences."""
    by = {act: [r for r in primary if r["activity"] == act] for act in ACTIVITIES}

    def pair(*acts):
        return [r for act in acts for r in by[act]]

    high, rest = pair("05", "06"), pair("01", "02", "03", "04")
    cw_total = sum(r["n_confident_wrong"] for r in primary)
    scored_total = sum(r["n_scored"] for r in primary)
    h1 = difference(boot, high, rest, CONFIDENT_WRONG)
    h1.update(
        share_of_confident_wrong=(sum(r["n_confident_wrong"] for r in high) / cw_total
                                  if cw_total else NAN),
        share_of_scored_windows=(sum(r["n_scored"] for r in high) / scored_total
                                 if scored_total else NAN),
        reading=reading(h1["ci"], +1))
    low_motion, moderate, high_motion = pair("01", "03"), pair("02", "04"), pair("05", "06")
    step1 = difference(boot, moderate, low_motion, REFUSAL)
    step2 = difference(boot, high_motion, moderate, REFUSAL)
    readings = {reading(step1["ci"], +1), reading(step2["ci"], +1)}
    h2 = dict(refusal={"01/03": ratio(low_motion, *REFUSAL), "02/04": ratio(moderate, *REFUSAL),
                       "05/06": ratio(high_motion, *REFUSAL)},
              moderate_minus_low=step1, high_minus_moderate=step2,
              reading=("contradicted" if "contradicted" in readings
                       else "consistent" if readings == {"consistent"} else "unresolved"))
    h3 = {}
    for halogen, ambient in (("03", "01"), ("04", "02")):
        entry = {}
        for label, metric in (("refusal", REFUSAL), ("confident_wrong", CONFIDENT_WRONG)):
            d = difference(boot, by[halogen], by[ambient], metric)
            # "No worse" can only be contradicted: halogen reliably higher on either metric.
            d["reading"] = ("not estimable" if not np.isfinite(d["ci"][0])
                            else "contradicted" if d["ci"][0] > 0 else "not contradicted")
            entry[label] = d
        entry["n_recordings"] = {halogen: len(by[halogen]), ambient: len(by[ambient])}
        h3[f"{halogen}_vs_{ambient}"] = entry
    return dict(H1=h1, H2=h2, H3=h3)


def supplementary(primary, windows):
    """Views that were not pre-registered, labelled as such wherever they are shown."""
    names = {r["recording"] for r in primary}
    scored = [w for w in windows if w["recording"] in names and w["scored"]]
    by_status = {}
    for w in scored:
        key = "ok" if w["recording_status"] == "ok" else "flagged (degraded/unstable)"
        entry = by_status.setdefault(key, dict(n_scored=0, n_confident_wrong=0))
        entry["n_scored"] += 1
        entry["n_confident_wrong"] += int(w["confident_wrong"])
    recording_wrong = {}
    for act in ACTIVITIES:
        reported = [r for r in primary if r["activity"] == act and r["reported"]]
        recording_wrong[act] = dict(
            n_reported=len(reported),
            n_off_by_more_than_10=sum(abs(r["error_vs_reference"]) > WRONG_BPM for r in reported),
            n_ok_status_off_by_more_than_10=sum(abs(r["error_vs_reference"]) > WRONG_BPM
                                                and r["status"] == "ok" for r in reported))
    return dict(confident_wrong_by_recording_status=by_status,
                recordings_reported_off_by_more_than_10_bpm=recording_wrong)


def separate_sets(recordings, windows):
    """The four sets Part 0 reports apart from the primary set."""
    def row(r):
        return dict(recording=r["recording"], activity=r["activity"], ref_frac=round(r["ref_frac"], 3),
                    ref_hr_median=r["ref_hr_median"], face_detections=r["face_detections"],
                    frac_side_lost=round(r["frac_side_lost"], 4),
                    box_side_lost_stored=round(r["box_side_lost_stored"], 4),
                    capture_verdict=r["capture_verdict"], status=r["status"], value=r["value"],
                    forced_status=r["forced_status"], forced_value=r["forced_value"],
                    error_vs_reference=r["error_vs_reference"])
    off = [w for w in windows if w["off_span"]]
    return dict(
        below_reference_gate=[row(r) for r in recordings if r["below_gate"]],
        clamped=[row(r) for r in recordings if r["clamped"]],
        low_detection=[row(r) for r in recordings if r["low_detection"]],
        off_span_windows=dict(
            n_windows=len(off), recordings=sorted({w["recording"] for w in off}),
            n_kept=sum(w["pred_kept"] for w in off),
            n_kept_referenced_reported=sum(w["pred_kept"] and w["referenced"]
                                           and w["recording_status"] in REPORTED for w in off),
            n_off_by_more_than_10=sum(w["pred_kept"] and w["referenced"]
                                      and abs(w["error"]) > WRONG_BPM for w in off)))


def build_summary(estimator, recordings, windows, args):
    """Assembles the JSON summary for one estimator."""
    primary = [r for r in recordings if r["primary"]]
    boot = Bootstrap(recordings, args.boot, args.seed)
    fps_values = [r["fps"] for r in recordings]
    return dict(
        estimator=estimator.name, store=str(args.store), device=args.device, limit=args.limit,
        protocol="NB_P3_29 Part 0 (pre-registered 2026-10-03; low-detection set added 2026-10-04)",
        settings=dict(clip_len=config.CLIP_LEN, window_stride=config.WINDOW_STRIDE,
                      min_confidence=config.MIN_CONFIDENCE, wrong_bpm=WRONG_BPM,
                      ref_min_agree=REF_MIN_AGREE, ref_recording_min=REF_RECORDING_MIN,
                      span_tolerance=SPAN_TOLERANCE, low_detection_below=LOW_DETECTION,
                      max_box_side_lost=config.MAX_BOX_SIDE_LOST,
                      fps_range=[min(fps_values), max(fps_values)] if fps_values else [],
                      bootstrap=dict(unit="subject", resamples=args.boot, seed=args.seed)),
        sets=dict(n_recordings=len(recordings), n_primary=len(primary),
                  primary_by_activity={a: sum(r["activity"] == a for r in primary)
                                       for a in ACTIVITIES},
                  store_mismatches={r["recording"]: r["store_mismatch"]
                                    for r in recordings if r["store_mismatch"]}),
        primary_by_activity=per_activity(primary, boot),
        hypotheses=hypotheses(primary, boot),
        supplementary_not_preregistered=supplementary(primary, windows),
        reported_separately=separate_sets(recordings, windows))


# ------------------------------------------------------------------ printing --

def _pct(x):
    return f"{100 * x:5.1f}" if np.isfinite(x) else f"{'-':>5}"


def _ci(ci):
    return (f"[{100 * ci[0]:4.0f},{100 * ci[1]:4.0f}]" if np.isfinite(ci[0])
            else f"{'[-]':>11}")


def _num(x, fmt="6.2f"):
    return format(x, fmt) if isinstance(x, (int, float)) and np.isfinite(x) else "-"


def print_summary(s):
    """Prints the tables to paste back for review."""
    print(f"\n=== {s['estimator']} === primary set {s['sets']['n_primary']} of "
          f"{s['sets']['n_recordings']} recordings | protocol: {s['protocol']}")
    if s["sets"]["store_mismatches"]:
        print(f"!! store and script disagree on: {s['sets']['store_mismatches']}")
    print(f"\n{'act':3} {'light':7} {'motion':8} {'rec':>3} {'rep':>3} {'refused%':>8} {'95% CI':>11} "
          f"{'discard%':>8} {'CW k/n':>9} {'CW%':>5} {'95% CI':>11} {'MAE':>6} {'bias':>6} {'cover%':>6}")
    for act, m in s["primary_by_activity"].items():
        if not m["n_recordings"]:
            print(f"{act:3} (no primary recordings)")
            continue
        print(f"{act:3} {m['lighting']:7} {m['motion_class']:8} {m['n_recordings']:3d} "
              f"{m['n_reported']:3d} {_pct(m['refusal']):>8} {_ci(m['refusal_ci'])} "
              f"{_pct(m['window_discard']):>8} {m['n_confident_wrong']:4d}/{m['n_scored']:<4d} "
              f"{_pct(m['confident_wrong'])} {_ci(m['confident_wrong_ci'])} "
              f"{_num(m['mae'])} {_num(m['bias'], '+6.2f')} {_pct(m['coverage']):>6}")
    h = s["hypotheses"]
    h1 = h["H1"]
    print(f"\nH1 confident-wrong concentrates in 05/06: rate 05/06 minus 01-04 "
          f"{_num(100 * h1['difference'], '+.1f')} pts {_ci(h1['ci'])} | 05/06 hold "
          f"{_pct(h1['share_of_confident_wrong']).strip()}% of confident-wrong windows and "
          f"{_pct(h1['share_of_scored_windows']).strip()}% of scored windows -> {h1['reading']}")
    h2 = h["H2"]
    print(f"H2 refusal rises with motion: 01/03 {_pct(h2['refusal']['01/03']).strip()}%, "
          f"02/04 {_pct(h2['refusal']['02/04']).strip()}%, 05/06 {_pct(h2['refusal']['05/06']).strip()}% "
          f"| steps {_num(100 * h2['moderate_minus_low']['difference'], '+.1f')} "
          f"{_ci(h2['moderate_minus_low']['ci'])} and "
          f"{_num(100 * h2['high_minus_moderate']['difference'], '+.1f')} "
          f"{_ci(h2['high_minus_moderate']['ci'])} -> {h2['reading']}")
    for key, entry in h["H3"].items():
        halogen, ambient = key.split("_vs_")
        parts = [f"{label} {_num(100 * d['difference'], '+.1f')} pts {_ci(d['ci'])} {d['reading']}"
                 for label, d in entry.items() if label != "n_recordings"]
        n = entry["n_recordings"]
        print(f"H3 halogen {halogen} vs ambient {ambient} (n {n[halogen]} vs {n[ambient]}): "
              + " | ".join(parts))
    sup = s["supplementary_not_preregistered"]
    print("\nsupplementary, not pre-registered:")
    print("  confident-wrong by recording status: " + "; ".join(
        f"{k} {v['n_confident_wrong']}/{v['n_scored']}"
        for k, v in sup["confident_wrong_by_recording_status"].items()))
    print("  reported recordings off the reference median by >10 bpm (all / status ok): " + "  ".join(
        f"{a}: {v['n_off_by_more_than_10']}/{v['n_ok_status_off_by_more_than_10']} of {v['n_reported']}"
        for a, v in sup["recordings_reported_off_by_more_than_10_bpm"].items()))
    sep = s["reported_separately"]
    print("\nreported separately:")
    below = sep["below_reference_gate"]
    print(f"  below the reference gate: {len(below)} recordings, app reported "
          f"{sum(r['status'] in REPORTED for r in below)}")
    for label in ("clamped", "low_detection"):
        print(f"  {label}:")
        for r in sep[label]:
            print(f"    {r['recording']}  capture {r['capture_verdict']:6}  frac_side_lost "
                  f"{r['frac_side_lost']:.3f} (per-axis {r['box_side_lost_stored']:.3f})  "
                  f"detections {r['face_detections']:2d}  app {r['status']} {_num(r['value'], '.1f')}  "
                  f"forced {r['forced_status']} {_num(r['forced_value'], '.1f')}  "
                  f"ref {_num(r['ref_hr_median'], '.1f')}  ref_frac {r['ref_frac']:.2f}")
    off = sep["off_span_windows"]
    print(f"  off-span windows: {off['n_windows']} in {off['recordings']} | kept {off['n_kept']}, "
          f"kept+referenced in a reported recording {off['n_kept_referenced_reported']}, "
          f"of those >10 bpm off {off['n_off_by_more_than_10']}")


# --------------------------------------------------------------------- utils --

def _plain(value):
    """Converts h5py attribute values to plain Python types."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_csv(path, rows):
    """Writes a list of dicts to CSV."""
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", action="append", choices=registry.available(),
                        help="registered estimator name; repeat for several (default: all)")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--device", default="cpu", help="torch device for learned estimators")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N recordings (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--boot", type=int, default=2000, help="subject-bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    args = parser.parse_args()

    if not args.store.exists():
        raise SystemExit(f"ERROR: no store at {args.store}")
    names = args.estimator or registry.available()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"store: {args.store}\nestimators: {', '.join(names)} | device: {args.device}"
          + (f" | limit {args.limit}" if args.limit else ""))
    estimators = make_estimators(names, args.device)
    recordings, windows = evaluate(args.store, estimators, args.limit)

    for estimator in estimators:
        name = estimator.name
        summary = build_summary(estimator, recordings[name], windows[name], args)
        _write_csv(args.out_dir / f"windows_{name}.csv", windows[name])
        _write_csv(args.out_dir / f"recordings_{name}.csv", recordings[name])
        (args.out_dir / f"summary_{name}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print_summary(summary)

    print(f"\noutputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())