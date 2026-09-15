"""Held-out evaluation of heart-rate estimators on UBFC-Phys.

Runs registered estimators over the UBFC-Phys Phase-3 store and scores their
per-window heart rate against the reference BVP read out over the same windows.
UBFC-Phys was never used in training, so every figure here is cross-dataset.

Two analysis sets are fixed before any estimator is run, and both are decided by
the reference alone, never by the estimator under test:

- primary: windows kept by the reference's own gating (the gates the app applies
  to an estimate), within recordings whose reference yields a reading;
- sensitivity: every window with a finite reference reading, in every recording.

Two confounds apply to every figure and are written into every output. The corpus
runs at 35.138 fps and is not decimated, so a 160-frame window spans 4.55 s against
the 5.33 s the model was trained on. And the reference is a wrist Empatica E4 signal,
noisier than the finger and contact references of the training corpora.

Usage:
    python -m app.live_vitals.scripts.check_ubfc_phys [--estimator NAME ...]
        [--store PATH] [--tasks T1 T2 T3] [--device cpu|cuda] [--limit N]
        [--out-dir DIR] [--parity-dir DIR]

No frames are rendered or written. The store and the video subset are read-only.
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
from app.live_vitals.estimators import registry
from app.live_vitals.estimators.base import aggregate_windows
from app.live_vitals.signal.hr import hr_from_bvp
from app.live_vitals.scripts.check_ubfc_regression import window_starts

DEFAULT_STORE = Path(r"D:\QualityPhys\phase 3 datasets\ubfc_phys_phase3_rc_none.h5")
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "ubfc_phys_eval"
TASKS = ("T1", "T2", "T3")
REPORTED = ("ok", "degraded_capture", "unstable")  # statuses that carry a value
SQI_SPLIT = 0.40  # floored SQI below which the dataset paper's own exclusions fall


# ----------------------------------------------------------------- reference --

def reference_windows(bvp, fps, starts):
    """Reads the reference BVP over the analysis windows and gates it like an estimate.

    Returns the gated reference result and, per window start, the reference heart
    rate, its confidence and whether the gating kept it.
    """
    readings = [hr_from_bvp(bvp[s:s + config.CLIP_LEN], fps) for s in starts]
    finite = [(s, r) for s, r in zip(starts, readings) if np.isfinite(r["hr_bpm"])]
    result = aggregate_windows("heart_rate", "bpm",
                               [r["hr_bpm"] for _, r in finite],
                               [r["confidence"] for _, r in finite],
                               [], fps, len(starts),
                               window_starts=[s for s, _ in finite])
    kept = dict(zip(result.detail["window_start"], result.detail["window_kept"]))
    rows = {s: dict(ref_hr=float(r["hr_bpm"]), ref_confidence=float(r["confidence"]),
                    ref_kept=bool(kept.get(s, False)))
            for s, r in zip(starts, readings)}
    return result, rows


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


# ------------------------------------------------------------------- metrics --

def agreement(predicted, reference):
    """Returns error statistics of predicted against reference heart rates."""
    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    ok = np.isfinite(predicted) & np.isfinite(reference)
    predicted, reference = predicted[ok], reference[ok]
    n = int(ok.sum())
    if n == 0:
        return dict(n=0)
    error = predicted - reference
    bias = float(error.mean())
    sd = float(error.std(ddof=1)) if n > 1 else float("nan")
    r = (float(np.corrcoef(predicted, reference)[0, 1])
         if n > 2 and predicted.std() > 0 and reference.std() > 0 else float("nan"))
    return dict(n=n, mae=float(np.abs(error).mean()),
                rmse=float(np.sqrt((error ** 2).mean())),
                mape=float((np.abs(error) / reference).mean() * 100.0),
                bias=bias, loa_low=bias - 1.96 * sd, loa_high=bias + 1.96 * sd, r=r)


def window_set(windows, name):
    """Returns the window rows scored in an analysis set, and those eligible for it."""
    eligible = [w for w in windows if w[name]]
    scored = [w for w in eligible if np.isfinite(w["pred_hr"])]
    return eligible, scored


def summarise_task(recordings, windows):
    """Computes the primary and sensitivity figures for one task."""
    out = {}
    for name, ref_key in (("primary", "ref_hr"), ("sensitivity", "ref_hr_all")):
        eligible, scored = window_set(windows, name)
        recs = ([r for r in recordings if r["ref_accepted"]] if name == "primary"
                else [r for r in recordings if np.isfinite(r["ref_hr_all"])])
        reported = [r for r in recs if r["status"] in REPORTED]
        statuses = {}
        for r in recs:
            statuses[r["status"]] = statuses.get(r["status"], 0) + 1
        out[name] = dict(
            n_recordings=len(recs),
            n_windows_eligible=len(eligible),
            n_windows_model_no_peak=len(eligible) - len(scored),
            window=agreement([w["pred_hr"] for w in scored], [w["ref_hr"] for w in scored]),
            coverage=(len(reported) / len(recs)) if recs else float("nan"),
            recording=agreement([r["value"] for r in reported], [r[ref_key] for r in reported]),
            status_counts=statuses,
            mean_usable_fraction=(float(np.mean([r["usable_fraction"] for r in recs]))
                                  if recs else float("nan")))
    return out


def strata(recordings, windows):
    """Breaks the primary window agreement down by reference SQI, sex and scenario."""
    by_recording = {r["recording"]: r for r in recordings}
    _, scored = window_set(windows, "primary")
    groups = {
        "cardiac_sqi_30s": lambda r: f"< {SQI_SPLIT}" if r["cardiac_sqi_30s"] < SQI_SPLIT
                                     else f">= {SQI_SPLIT}",
        "gender": lambda r: r["gender"],
        "scenario": lambda r: r["scenario"],
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


def paired_task_effect(recordings):
    """Compares within-subject HR change across tasks, model against reference.

    Uses only subjects whose reference is accepted in all three tasks and whose
    estimate is reported in all three, so the comparison is not a selection artefact.
    """
    table = {}
    for r in recordings:
        table.setdefault(r["subject"], {})[r["task"]] = r
    subjects = [s for s, tasks in table.items()
                if all(t in tasks and tasks[t]["ref_accepted"]
                       and tasks[t]["status"] in REPORTED for t in TASKS)]
    out = dict(n_subjects=len(subjects))
    for later in ("T2", "T3"):
        model = np.array([table[s][later]["value"] - table[s]["T1"]["value"] for s in subjects])
        ref = np.array([table[s][later]["ref_hr"] - table[s]["T1"]["ref_hr"] for s in subjects])
        out[f"{later}-T1"] = (dict(model_mean=float(model.mean()), model_median=float(np.median(model)),
                                   reference_mean=float(ref.mean()),
                                   reference_median=float(np.median(ref)),
                                   mean_abs_difference=float(np.abs(model - ref).mean()))
                              if subjects else {})
    return out


# ---------------------------------------------------------------------- run --

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


def evaluate(store_path, estimators, tasks, limit):
    """Runs every estimator over the store; returns per-estimator recording and window rows."""
    recordings = {e.name: [] for e in estimators}
    windows = {e.name: [] for e in estimators}
    with h5py.File(store_path, "r") as store:
        names = [n for n in sorted(store.keys(), key=_natural_key)
                 if str(store[n].attrs["task"]) in tasks]
        if limit:
            names = names[:limit]
        t0 = time.perf_counter()
        for i, name in enumerate(names, 1):
            group = store[name]
            attrs = {k: _plain(v) for k, v in group.attrs.items()}
            fps = float(attrs["fps"])
            frames = group["frames"][()]
            bvp = group["bvp"][()]
            starts = list(window_starts(len(frames)))

            ref_result, ref_rows = reference_windows(bvp, fps, starts)
            ref_accepted = ref_result.status in REPORTED
            finite_ref = [row["ref_hr"] for row in ref_rows.values() if np.isfinite(row["ref_hr"])]
            ref_hr_all = float(np.median(finite_ref)) if finite_ref else float("nan")

            for estimator in estimators:
                result = estimator.estimate(frames, fps)
                pred_rows = predicted_windows(result)
                detail = result.detail or {}
                for s in starts:
                    ref = ref_rows[s]
                    pred = pred_rows.get(s, dict(pred_hr=float("nan"), pred_confidence=float("nan"),
                                                 pred_kept=False))
                    finite = np.isfinite(ref["ref_hr"])
                    windows[estimator.name].append(dict(
                        recording=name, subject=attrs["subject"], task=attrs["task"],
                        start=s, t_start_s=s / fps, **ref, **pred,
                        primary=bool(ref_accepted and ref["ref_kept"] and finite),
                        sensitivity=bool(finite)))
                recordings[estimator.name].append(dict(
                    recording=name, subject=attrs["subject"], task=attrs["task"],
                    state=attrs.get("state"), gender=attrs.get("gender"),
                    scenario=attrs.get("scenario"),
                    cardiac_sqi_30s=float(attrs.get("cardiac_sqi_30s", float("nan"))),
                    bvp_windowed_hr=float(attrs.get("bvp_windowed_hr", float("nan"))),
                    n_frames=len(frames), fps=fps, n_windows=len(starts),
                    ref_status=ref_result.status, ref_accepted=ref_accepted,
                    ref_hr=float(ref_result.value), ref_hr_all=ref_hr_all,
                    ref_usable_fraction=float(ref_result.detail.get("usable_fraction", 0.0)),
                    n_ref_refused=len(starts) - sum(r["ref_kept"] for r in ref_rows.values()),
                    value=float(result.value), status=result.status,
                    confidence=float(result.confidence),
                    usable_fraction=float(detail.get("usable_fraction", 0.0)),
                    n_no_peak=int(detail.get("n_no_peak", len(starts))),
                    error_vs_reference=float(result.value - ref_result.value),
                    error_vs_bvp_windowed_hr=float(result.value - attrs.get("bvp_windowed_hr",
                                                                          float("nan")))))
            if i % 10 == 0 or i == len(names):
                print(f"  {i}/{len(names)} recordings | {(time.perf_counter() - t0) / 60:.1f} min",
                      flush=True)
    return recordings, windows


def build_summary(estimator, recordings, windows, args, tasks):
    """Assembles the JSON summary for one estimator."""
    fps_values = sorted({round(r["fps"], 3) for r in recordings})
    summary = dict(
        estimator=estimator.name,
        store=str(args.store), device=args.device, tasks_run=list(tasks), limit=args.limit,
        settings=dict(clip_len=config.CLIP_LEN, window_stride=config.WINDOW_STRIDE,
                      min_confidence=config.MIN_CONFIDENCE, fps=fps_values,
                      window_seconds=[round(config.CLIP_LEN / f, 2) for f in fps_values],
                      training_window_seconds=round(config.CLIP_LEN / config.TARGET_FPS, 2)),
        confounds=["35.138 fps is not decimated: windows span 4.55 s against 5.33 s in training",
                   "reference is a wrist Empatica E4 BVP"],
        analysis_sets=dict(
            primary="windows kept by the reference's own gating, in recordings whose "
                    "reference yields a reading",
            sensitivity="every window with a finite reference reading"),
        reference_accepted={t: dict(accepted=sum(r["ref_accepted"] for r in recordings if r["task"] == t),
                                    total=sum(1 for r in recordings if r["task"] == t))
                            for t in tasks},
        tasks={t: summarise_task([r for r in recordings if r["task"] == t],
                                 [w for w in windows if w["task"] == t]) for t in tasks},
        strata={t: strata([r for r in recordings if r["task"] == t],
                          [w for w in windows if w["task"] == t]) for t in tasks},
    )
    if set(TASKS) <= set(tasks):
        summary["paired_task_effect"] = paired_task_effect(recordings)
    return summary


def print_summary(summary):
    """Prints the headline tables for one estimator."""
    s = summary["settings"]
    print(f"\n=== {summary['estimator']} ===")
    print(f"fps {s['fps']} undecimated -> windows {s['window_seconds']} s "
          f"(training {s['training_window_seconds']} s) | reference: wrist E4 BVP")
    print("reference accepted: " + ", ".join(
        f"{t} {v['accepted']}/{v['total']}" for t, v in summary["reference_accepted"].items()))
    print(f"\n{'task':4} {'set':11} {'rec':>4} {'win':>5} {'nopk':>4} {'MAE':>6} {'RMSE':>6} "
          f"{'MAPE':>6} {'bias':>6} {'LoA':>15} {'r':>5} | {'cover':>5} {'recMAE':>6}")
    for task, sets in summary["tasks"].items():
        for name, m in sets.items():
            w, rec = m["window"], m["recording"]
            if w.get("n", 0):
                loa = f"{w['loa_low']:+.1f}..{w['loa_high']:+.1f}"
                line = (f"{w['mae']:6.2f} {w['rmse']:6.2f} {w['mape']:6.1f} {w['bias']:+6.2f} "
                        f"{loa:>15} {w['r']:5.2f}")
            else:
                line = f"{'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>15} {'-':>5}"
            rec_mae = f"{rec['mae']:6.2f}" if rec.get("n", 0) else f"{'-':>6}"
            print(f"{task:4} {name:11} {m['n_recordings']:4d} {w.get('n', 0):5d} "
                  f"{m['n_windows_model_no_peak']:4d} {line} | {m['coverage']:5.2f} {rec_mae}")
    for task, fields in summary["strata"].items():
        parts = []
        for field, labels in fields.items():
            for label, m in labels.items():
                if m.get("n", 0):
                    parts.append(f"{field} {label}: MAE {m['mae']:.2f} n={m['n']} "
                                 f"({m['n_recordings']} rec)")
        print(f"\nprimary strata {task}: " + "; ".join(parts))
    effect = summary.get("paired_task_effect")
    if effect:
        print(f"\npaired task effect, {effect['n_subjects']} subjects "
              f"(reference accepted and estimate reported in T1-T3):")
        for key in ("T2-T1", "T3-T1"):
            e = effect.get(key)
            if e:
                print(f"  {key}: model mean {e['model_mean']:+.2f} (median {e['model_median']:+.2f}) | "
                      f"reference mean {e['reference_mean']:+.2f} (median {e['reference_median']:+.2f}) | "
                      f"mean |diff| {e['mean_abs_difference']:.2f}")


# -------------------------------------------------------------------- parity --

def parity(parity_dir, store_path):
    """Compares app-path crops from raw videos with the frames stored for them."""
    from app.live_vitals.capture.session import crops_from_video
    from app.live_vitals.preprocess.face_box import make_landmarker

    videos = sorted(Path(parity_dir).glob("s*/vid_s*_T*.avi"), key=lambda p: _natural_key(p.stem))
    landmarker = make_landmarker()
    rows = []
    print(f"\nparity: {len(videos)} videos under {parity_dir}")
    with h5py.File(store_path, "r") as store:
        for video in videos:
            name = video.stem.replace("vid_", "")
            if name not in store:
                print(f"  {name}: not in store - skipped")
                continue
            clip, effective_fps, quality = crops_from_video(video, landmarker)
            stored = store[name]["frames"]
            n = min(len(clip), len(stored))
            max_diff, same = 0, 0
            for i in range(0, n, 512):
                a = clip[i:i + 512].astype(np.int16)
                b = stored[i:min(i + 512, n)].astype(np.int16)
                diff = np.abs(a[:len(b)] - b)
                max_diff = max(max_diff, int(diff.max()))
                same += int((diff.reshape(len(b), -1).max(axis=1) == 0).sum())
            row = dict(recording=name, app_frames=len(clip), stored_frames=len(stored),
                       max_abs_diff=max_diff, identical_fraction=same / n if n else float("nan"),
                       effective_fps=float(effective_fps), verdict=quality.verdict,
                       notes=" | ".join(quality.notes))
            rows.append(row)
            print(f"  {name:8} frames {len(clip)}/{len(stored)}  max|diff| {max_diff:3d}  "
                  f"identical {row['identical_fraction']:.3f}  fps {effective_fps:.3f}  "
                  f"{quality.verdict}", flush=True)
    return rows


# --------------------------------------------------------------------- utils --

def _plain(value):
    """Converts h5py attribute values to plain Python types."""
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _natural_key(name):
    """Sorts s2_T1 before s10_T1."""
    head, _, task = name.partition("_T")
    digits = "".join(ch for ch in head if ch.isdigit())
    return (int(digits) if digits else 0, task)


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
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--device", default="cpu", help="torch device for learned estimators")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N recordings (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--parity-dir", type=Path, default=None,
                        help="also compare app-path crops of these raw videos with the store")
    args = parser.parse_args()

    names = args.estimator or registry.available()
    tasks = [t for t in TASKS if t in args.tasks]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"store: {args.store}\nestimators: {', '.join(names)} | tasks: {' '.join(tasks)} | "
          f"device: {args.device}" + (f" | limit {args.limit}" if args.limit else ""))
    estimators = make_estimators(names, args.device)
    recordings, windows = evaluate(args.store, estimators, tasks, args.limit)

    for estimator in estimators:
        name = estimator.name
        summary = build_summary(estimator, recordings[name], windows[name], args, tasks)
        _write_csv(args.out_dir / f"windows_{name}.csv", windows[name])
        _write_csv(args.out_dir / f"recordings_{name}.csv", recordings[name])
        (args.out_dir / f"summary_{name}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print_summary(summary)

    if args.parity_dir:
        rows = parity(args.parity_dir, args.store)
        _write_csv(args.out_dir / "parity.csv", rows)

    print(f"\noutputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())