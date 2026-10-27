"""Held-out evaluation of heart-rate estimators on the Rice iPPG (HRVCam) dataset.

Runs registered estimators over the Rice Phase-3 store and scores their per-window heart
rate against the finger reference read out over the same windows. No model in this project
has trained on these recordings, so every figure is cross-dataset.

The store holds **two crops of every recording**, and this script scores whichever is asked
for, naming the crop in every output:

  frames_clamped   the app's box clamped to the frame, which is what the app does today
  frames           the frame padded out so the whole 0.6-padded box fits, which is the
                   geometry the models trained on

Which to prefer is the open question the store was built to answer (NB_P3_34), and it is
answered by running both, not by the geometry alone: on the store's own POS readings the
clamped crop reads 69 of 70 recordings and the padded one 65, the five it loses all talking.

The analysis sets are the ones check_ubfc_phys.py fixed, decided by the reference alone:

- primary: windows kept by the reference's own gating, within recordings whose reference
  yields a reading;
- sensitivity: every window with a finite reference reading.

Four caveats belong beside every figure here, and are written into every summary:

- **the app's framing gate refuses all 70 recordings.** A face filling 60-95% of the frame's
  height cannot fit a padded box, so these are recordings the app would decline to read. The
  figures say what the readout does when forced, which is what a crop rule has to be chosen on.
- the reference is a 60 Hz, 7-bit finger pulse oximeter, aligned per recording by the offset
  the store carries. A finger pulse arrives later than a facial one and each device adds its
  own latency, so the offset mixes them and is not a transit time.
- talking is the hard activity and is reported on its own, never pooled: it holds the weakest
  POS-label correlations, the widest beat distances and the only recording POS cannot read.
- three recordings lose their last seconds to an oximeter that stopped early; windows whose
  frames are not covered by the reference are excluded, and counted.

Heart-rate variability is not scored here. It needs a continuous reconstruction and the
store's own beat times rather than a per-window rate, which is the HRV harness's business.

Usage:
    python -m app.live_vitals.scripts.check_rice_ippg [--estimator NAME ...] [--crop NAME]
        [--activities stationary talking ...] [--store PATH] [--device cpu|cuda]
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals import config
from app.live_vitals.estimators import registry
from app.live_vitals.scripts.bench_outputs import provenance, subset_suffix, unique
from app.live_vitals.scripts.check_ubfc_regression import window_starts
from app.live_vitals.scripts.check_ubfc_phys import (REPORTED, _plain, _write_csv, agreement,
                                                     common_windows, make_estimators,
                                                     predicted_windows, reference_windows,
                                                     summarise_task, window_set)

DEFAULT_STORE = Path(r"D:\QualityPhys\phase 3 benchmarks\rice_ippg_phase3_rc_none.h5")
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "rice_ippg_eval"
DATASET = "Rice-iPPG"
ACTIVITIES = ("stationary", "reading", "watching", "talking", "deep")
CROPS = ("frames_clamped", "frames")
DEFAULT_CROP = "frames_clamped"
WEAK_TRACE_R = -0.6   # above this, the store's POS barely correlates with the label
CONFOUNDS = [
    "the app's framing gate refuses all 70 recordings: a face filling 60-95% of the frame's "
    "height cannot fit a box padded by 0.6, so these readings are forced",
    "the reference is a 60 Hz 7-bit finger pulse oximeter, aligned per recording by the "
    "store's offset, which mixes both devices' latencies with the pulse's travel",
    "talking is reported on its own and never pooled: the weakest correlations, the widest "
    "beat distances and the only recording POS cannot read",
    "windows whose frames the reference does not cover are excluded and counted",
]


# ------------------------------------------------------------------- metrics --

def strata(recordings, windows):
    """Breaks the primary window agreement down by activity, and by what the store already
    knows about each recording: whether its reference carries HRV, and how well POS
    correlated with that reference when the store was written."""
    by_recording = {r["recording"]: r for r in recordings}
    _, scored = window_set(windows, "primary")
    groups = {
        "activity": lambda r: r["activity"],
        "has_hrv": lambda r: "hrv" if r["has_hrv"] else "hr only",
        "trace": lambda r: (f"POS r > {WEAK_TRACE_R:g}" if r["qc_pos_bvp_r"] > WEAK_TRACE_R
                            else f"POS r <= {WEAK_TRACE_R:g}"),
    }
    out = {}
    for field, label_of in groups.items():
        buckets = {}
        for w in scored:
            buckets.setdefault(label_of(by_recording[w["recording"]]), []).append(w)
        out[field] = {label: dict(agreement([w["pred_hr"] for w in rows],
                                            [w["ref_hr"] for w in rows]),
                                  n_recordings=len({w["recording"] for w in rows}))
                      for label, rows in sorted(buckets.items())}
    return out


def beat_reference_check(recordings):
    """Recording-level agreement against the reference's own beat-based heart rate.

    The windows are scored against a spectral readout of the same finger waveform. The
    store also carries `ref_hr_bpm`, taken from the beats NB_P3_33's rules kept, which is a
    second readout of that reference rather than a second sensor - it catches a window-level
    readout gone wrong, not a reference gone wrong.
    """
    rows = [r for r in recordings if r["status"] in REPORTED and np.isfinite(r["ref_hr_beats"])]
    return dict(n_recordings=len(rows),
                agreement=agreement([r["value"] for r in rows],
                                    [r["ref_hr_beats"] for r in rows]))


def hr_spread(values):
    """The heart-rate range a run actually covered, which is what limits its claims."""
    hrs = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if not len(hrs):
        return {}
    return dict(n=len(hrs), min=float(hrs.min()), median=float(np.median(hrs)),
                max=float(hrs.max()), above_100=int((hrs > 100).sum()))


# ----------------------------------------------------------------------- run --

def covered_starts(starts, covered):
    """The window starts the reference covers end to end, and how many it does not.

    Three recordings lose their last seconds to an oximeter that stopped before the camera.
    A window reaching into that stretch would be scored against a reference held at its last
    value, which reads as a flat pulse rather than as missing data.
    """
    kept = [s for s in starts if bool(covered[s:s + config.CLIP_LEN].all())]
    return kept, len(starts) - len(kept)


def evaluate(store_path, estimators, crop, activities, limit):
    """Runs every estimator over one crop of the store; returns recording and window rows."""
    recordings = {e.name: [] for e in estimators}
    windows = {e.name: [] for e in estimators}
    with h5py.File(store_path, "r") as store:
        names = [n for n in sorted(store.keys())
                 if str(store[n].attrs["activity"]) in activities]
        if limit:
            names = names[:limit]
        t0 = time.perf_counter()
        for i, name in enumerate(names, 1):
            group = store[name]
            attrs = {k: _plain(v) for k, v in group.attrs.items()}
            if crop not in group:
                raise SystemExit(f"ERROR: {name} has no dataset {crop}")
            fps = float(attrs["fps"])
            frames = group[crop][()]
            bvp = group["bvp"][()]
            starts, uncovered = covered_starts(list(window_starts(len(frames))),
                                               group["bvp_covered"][()])
            if not starts:
                raise SystemExit(f"ERROR: {name} has no window the reference covers")

            ref_result, ref_rows = reference_windows(bvp, fps, starts)
            ref_accepted = ref_result.status in REPORTED
            finite_ref = [row["ref_hr"] for row in ref_rows.values() if np.isfinite(row["ref_hr"])]
            ref_hr_all = float(np.median(finite_ref)) if finite_ref else float("nan")
            cohort = dict(subject=attrs["subject_id"], activity=attrs["activity"],
                          has_hrv=bool(attrs["has_hrv"]),
                          qc_pos_bvp_r=float(attrs["qc_pos_bvp_r"]))

            for estimator in estimators:
                result = estimator.estimate(frames, fps)
                pred_rows = predicted_windows(result)
                detail = result.detail or {}
                for s in starts:
                    ref = ref_rows[s]
                    pred = pred_rows.get(s, dict(pred_hr=float("nan"),
                                                 pred_confidence=float("nan"), pred_kept=False))
                    finite = np.isfinite(ref["ref_hr"])
                    windows[estimator.name].append(dict(
                        recording=name, subject=cohort["subject"], activity=cohort["activity"],
                        crop=crop, start=s, t_start_s=s / fps, **ref, **pred,
                        primary=bool(ref_accepted and ref["ref_kept"] and finite),
                        sensitivity=bool(finite)))
                recordings[estimator.name].append(dict(
                    recording=name, **cohort, crop=crop,
                    n_frames=int(attrs["n_frames"]), fps=fps, n_windows=len(starts),
                    n_windows_uncovered=uncovered,
                    ref_hr_beats=float(attrs["ref_hr_bpm"]),
                    ref_rmssd_ms=float(attrs["ref_rmssd_ms"]),
                    ref_kept_share=float(attrs["ref_kept_share"]),
                    offset_ms=float(attrs["offset_ms"]),
                    framing_verdict=attrs["framing_verdict"],
                    motion_verdict=attrs["motion_verdict"],
                    store_pos_hr=float(attrs["qc_pos_hr"]),
                    ref_status=ref_result.status, ref_accepted=ref_accepted,
                    ref_hr=float(ref_result.value), ref_hr_all=ref_hr_all,
                    ref_usable_fraction=float(ref_result.detail.get("usable_fraction", 0.0)),
                    n_ref_refused=len(starts) - sum(r["ref_kept"] for r in ref_rows.values()),
                    value=float(result.value), status=result.status,
                    confidence=float(result.confidence),
                    usable_fraction=float(detail.get("usable_fraction", 0.0)),
                    n_no_peak=int(detail.get("n_no_peak", len(starts))),
                    error_vs_reference=float(result.value - ref_result.value),
                    error_vs_beats=float(result.value - attrs["ref_hr_bpm"])))
            if i % 10 == 0 or i == len(names):
                print(f"  {i}/{len(names)} recordings | {(time.perf_counter() - t0) / 60:.1f} min",
                      flush=True)
    return recordings, windows


def build_summary(estimator, recordings, windows, args, crop):
    """Assembles the JSON summary for one estimator on one crop."""
    reported = [r for r in recordings if r["status"] in REPORTED]
    return dict(
        estimator=estimator.name, dataset=DATASET, crop=crop,
        store=str(args.store), device=args.device, activities=list(args.activities),
        limit=args.limit,
        settings=dict(clip_len=config.CLIP_LEN, window_stride=config.WINDOW_STRIDE,
                      min_confidence=config.MIN_CONFIDENCE,
                      fps=sorted({round(r["fps"], 3) for r in recordings}),
                      window_seconds=round(config.CLIP_LEN / config.TARGET_FPS, 2),
                      training_window_seconds=round(config.CLIP_LEN / config.TARGET_FPS, 2)),
        confounds=CONFOUNDS,
        analysis_sets=dict(
            primary="windows kept by the reference's own gating, in recordings whose "
                    "reference yields a reading",
            sensitivity="every window with a finite reference reading"),
        n_recordings=len(recordings),
        activity_counts={a: sum(1 for r in recordings if r["activity"] == a)
                         for a in args.activities},
        windows_uncovered=int(sum(r["n_windows_uncovered"] for r in recordings)),
        reference_accepted=dict(accepted=sum(r["ref_accepted"] for r in recordings),
                                total=len(recordings)),
        reference_hr=hr_spread([r["ref_hr_beats"] for r in recordings]),
        reported_hr=hr_spread([r["value"] for r in reported]),
        sets=summarise_task(recordings, windows),
        by_activity={a: summarise_task([r for r in recordings if r["activity"] == a],
                                       [w for w in windows if w["activity"] == a])
                     for a in args.activities},
        strata=strata(recordings, windows),
        beat_reference_check=beat_reference_check(recordings),
    )


def print_summary(summary):
    """Prints the headline tables for one estimator on one crop."""
    s = summary["settings"]
    print(f"\n=== {summary['estimator']} | crop {summary['crop']} ===")
    print(f"{summary['n_recordings']} recordings {summary['activity_counts']} | windows "
          f"{s['window_seconds']} s at {s['fps']} fps | {summary['windows_uncovered']} windows "
          f"excluded for want of a reference")
    ref, rep = summary["reference_hr"], summary["reported_hr"]
    if ref:
        print(f"reference HR {ref['min']:.0f}-{ref['max']:.0f} (median {ref['median']:.0f}), "
              f"{ref['above_100']} above 100 | reported HR "
              f"{rep.get('min', float('nan')):.0f}-{rep.get('max', float('nan')):.0f} "
              f"(median {rep.get('median', float('nan')):.0f}) in {rep.get('n', 0)} recordings")
    print(f"reference accepted: {summary['reference_accepted']['accepted']}"
          f"/{summary['reference_accepted']['total']}")
    header = (f"\n{'set':12} {'rec':>4} {'win':>6} {'nopk':>5} {'MAE':>6} {'RMSE':>6} "
              f"{'MAPE':>6} {'bias':>6} {'r':>6} | {'cover':>5} {'recMAE':>7}")
    print(header)
    for name, m in summary["sets"].items():
        print(_set_line(name, m))
    print("\nby activity, the primary set:")
    print(header)
    for activity, sets in summary["by_activity"].items():
        if sets["primary"]["n_recordings"]:   # a subset run leaves the other activities empty
            print(_set_line(activity, sets["primary"]))
    for field, labels in summary["strata"].items():
        print(f"\nprimary by {field}:")
        for label, m in labels.items():
            if m.get("n", 0):
                print(f"  {label:18} MAE {m['mae']:6.2f} bias {m['bias']:+6.2f} r {m['r']:5.2f}"
                      f" | {m['n']:5d} windows, {m['n_recordings']:3d} recordings")
    beats = summary["beat_reference_check"]
    a = beats["agreement"]
    if a.get("n", 0):
        print(f"\nagainst the reference's own beat rate ({beats['n_recordings']} recordings "
              f"reported): MAE {a['mae']:.2f} bias {a['bias']:+.2f} r {a['r']:.2f}")


def _set_line(label, m):
    """One row of the agreement tables."""
    w, rec = m["window"], m["recording"]
    if w.get("n", 0):
        body = (f"{w['mae']:6.2f} {w['rmse']:6.2f} {w['mape']:6.1f} {w['bias']:+6.2f} "
                f"{w['r']:6.2f}")
    else:
        body = f"{'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>6}"
    rec_mae = f"{rec['mae']:7.2f}" if rec.get("n", 0) else f"{'-':>7}"
    return (f"{label:12} {m['n_recordings']:4d} {w.get('n', 0):6d} "
            f"{m['n_windows_model_no_peak']:5d} {body} | {m['coverage']:5.2f} {rec_mae}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", action="append", choices=registry.available(),
                        help="registered estimator name; repeat for several (default: all)")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--crop", action="append", choices=CROPS,
                        help=f"crop to score; repeat for both (default: {DEFAULT_CROP})")
    parser.add_argument("--activities", nargs="+", choices=ACTIVITIES, default=list(ACTIVITIES))
    parser.add_argument("--device", default="cpu", help="torch device for learned estimators")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N recordings (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    names = unique(args.estimator or registry.available())
    crops = unique(args.crop or [DEFAULT_CROP])
    args.activities = [a for a in ACTIVITIES if a in args.activities]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"store: {args.store}\nestimators: {', '.join(names)} | crops: {', '.join(crops)} | "
          f"activities: {' '.join(args.activities)} | device: {args.device}"
          + (f" | limit {args.limit}" if args.limit else ""))
    estimators = make_estimators(names, args.device)
    trained = [e.name for e in estimators
               if any(dataset == DATASET for dataset, _ in e.trained_on)]
    if trained:
        raise SystemExit(f"ERROR: {', '.join(trained)} trained on {DATASET}; this store is an "
                         f"evaluation set only")

    for crop in crops:
        recordings, windows = evaluate(args.store, estimators, crop, args.activities, args.limit)
        # The crop is named in every file, including the default one: which crop to prefer is
        # the question this dataset is here to answer, so a file that does not say is a trap.
        suffix = f"_crop-{crop}" + subset_suffix(
            ("activities-", "-".join(args.activities), "-".join(ACTIVITIES)),
            ("limit", args.limit or None, None))
        for estimator in estimators:
            name = estimator.name
            summary = build_summary(estimator, recordings[name], windows[name], args, crop)
            if len(estimators) > 1:
                summary["common_windows"] = common_windows(
                    windows[name], {o.name: windows[o.name] for o in estimators if o.name != name})
            summary["provenance"] = provenance(estimator)
            _write_csv(args.out_dir / f"windows_{name}{suffix}.csv", windows[name])
            _write_csv(args.out_dir / f"recordings_{name}{suffix}.csv", recordings[name])
            (args.out_dir / f"summary_{name}{suffix}.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8")
            print_summary(summary)

    print(f"\noutputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())