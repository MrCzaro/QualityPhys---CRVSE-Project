"""Zero-shot evaluation of heart-rate estimators on VitalVideos-WorldWide.

Runs registered estimators over the VitalVideos Phase-3 store and scores their
per-window heart rate against the reference BVP read out over the same windows. No
model in this project has been trained on VitalVideos, so every figure here is
cross-dataset. Each row carries its `split`, so once a model is trained on the 240
training subjects the held-out 60 can be re-cut from the same outputs.

The two analysis sets are the ones check_ubfc_phys.py fixed, and both are decided by
the reference alone, never by the estimator under test:

- primary: windows kept by the reference's own gating (the gates the app applies to
  an estimate), within recordings whose reference yields a reading;
- sensitivity: every window with a finite reference reading, in every recording.

Unlike UBFC-Phys there is no window confound here: the store is 30 fps, so a 160-frame
window spans the 5.33 s the model was trained on. Three caveats belong beside every
figure and are written into every summary:

- site, lighting and skin tone are confounded. The outdoor CH* group is almost all
  Fitzpatrick 5-6 and the indoor MOL/RING group almost all 1-3, so no figure here
  separates skin tone from the site that recorded it.
- the video runs about 0.1 s behind the reference (median lag -0.10 s across the 300
  recordings) and nothing here corrects it.
- the reference is the study's 500 Hz contact PPG interpolated to frame times. A
  separate 3-lead ECG gives an independent recording-level cross-check.

--speed applies a synthetic time warp: every k-th frame of the 60 fps streams is kept
and the result is declared 30 fps, so the pulse is compressed by exactly that factor
(x1.5 moves the cohort's 48-120 bpm to 72-180). Frames and reference are decimated by
the same index vector, so the reference warps with the video. This measures how the
estimator responds to a faster periodicity; it is not evidence about real tachycardia,
where diastole shortens far more than systole and motion, perfusion and respiration
all change with it.

Usage:
    python -m app.live_vitals.scripts.check_vitalvideos [--estimator NAME ...]
        [--store PATH] [--split all|train|val] [--speed 1.0] [--device cpu|cuda]
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
from app.live_vitals.scripts.check_ubfc_regression import window_starts
from app.live_vitals.scripts.check_ubfc_phys import (REPORTED, _plain, _write_csv, agreement,
                                                     make_estimators, predicted_windows,
                                                     reference_windows, summarise_task,
                                                     window_set)

DEFAULT_STORE = Path(r"D:\QualityPhys\phase 3 datasets\vitalvideos_phase3_rc_none.h5")
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "vitalvideos_eval"
SPLITS = ("all", "train", "val")
AGE_BANDS = ((40, "<=40"), (60, "41-60"))  # anything above the last bound is ">60"
CONFOUNDS = [
    "site, lighting and Fitzpatrick type are confounded: CH* is outdoor and almost all "
    "Fitzpatrick 5-6, MOL/RING indoor and almost all 1-3",
    "the video runs about 0.1 s behind the reference PPG and this is not corrected",
    "the reference is the study's 500 Hz contact PPG interpolated to frame times",
]
WARP_CONFOUND = ("synthetic time warp: frames and reference are decimated together from the "
                 "60 fps streams and declared 30 fps, so waveform morphology, motion and "
                 "respiration are scaled with the pulse; not evidence about real tachycardia")


# ------------------------------------------------------------------- streams --

def decimation_for(speed):
    """Returns the 60 fps decimation step for a playback speed (k = 2 x speed)."""
    k = speed * 2.0
    if abs(k - round(k)) > 1e-9 or round(k) < 2:
        raise SystemExit("ERROR: --speed must be at least 1.0 and a multiple of 0.5")
    return int(round(k))


def load_streams(group, speed):
    """Returns (frames, bvp) at the requested playback speed, always declared 30 fps.

    Speed 1.0 reads the stored 30 fps streams. A faster speed keeps every k-th frame of
    the 60 fps streams, which are index-aligned with each other, so the reference warps
    with the video instead of being rescaled after the fact.
    """
    if speed == 1.0:
        return group["frames"][()], group["bvp"][()]
    k = decimation_for(speed)
    return group["frames_60"][::k], group["bvp_60"][::k]


def age_band(age):
    """Returns the age band label of a recording."""
    for bound, label in AGE_BANDS:
        if age <= bound:
            return label
    return f">{AGE_BANDS[-1][0]}"


# ------------------------------------------------------------------- metrics --

def strata(recordings, windows):
    """Breaks the primary window agreement down by the cohort fields that matter here."""
    by_recording = {r["recording"]: r for r in recordings}
    _, scored = window_set(windows, "primary")
    groups = {
        "fitzpatrick": lambda r: str(r["fitzpatrick"]),
        "site_group": lambda r: r["site_group"],
        "gender": lambda r: r["gender"],
        "age_band": lambda r: r["age_band"],
        "split": lambda r: r["split"],
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


def ecg_cross_check(recordings, speed):
    """Recording-level agreement against the independent ECG beat rate.

    VitalVideos carries a 3-lead ECG, so `hr_ecg` shares no sensor with the BVP the
    windows are scored against. Only recordings whose ECG the audit accepted are used,
    and under a time warp the ECG rate is scaled by the same factor as the video.
    """
    rows = [r for r in recordings if r["ecg_ok"] and r["status"] in REPORTED]
    return dict(n_recordings=len(rows),
                agreement=agreement([r["value"] for r in rows],
                                    [r["hr_ecg"] * speed for r in rows]))


def hr_spread(values):
    """Heart-rate range actually covered, which is what limits the claims a run can make."""
    hrs = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if not len(hrs):
        return {}
    return dict(n=len(hrs), min=float(hrs.min()), median=float(np.median(hrs)),
                max=float(hrs.max()), above_100=int((hrs > 100).sum()),
                above_120=int((hrs > 120).sum()))


# ----------------------------------------------------------------------- run --

def evaluate(store_path, estimators, split, speed, limit):
    """Runs every estimator over the store; returns per-estimator recording and window rows."""
    recordings = {e.name: [] for e in estimators}
    windows = {e.name: [] for e in estimators}
    with h5py.File(store_path, "r") as store:
        names = [n for n in sorted(store.keys())
                 if split == "all" or str(store[n].attrs["split"]) == split]
        if limit:
            names = names[:limit]
        t0 = time.perf_counter()
        for i, name in enumerate(names, 1):
            group = store[name]
            attrs = {k: _plain(v) for k, v in group.attrs.items()}
            # Declaring the rate rather than reading it is only safe while the store holds
            # what this script assumes: 30 fps frames decimated from a 60 fps stream.
            if speed == 1.0 and abs(float(attrs["fps"]) - config.TARGET_FPS) > 0.01:
                raise SystemExit(f"ERROR: {name} is stored at {attrs['fps']} fps, "
                                 f"not {config.TARGET_FPS}")
            if speed != 1.0 and abs(float(attrs["fps_native"]) - 2 * config.TARGET_FPS) > 0.01:
                raise SystemExit(f"ERROR: {name} has no {2 * config.TARGET_FPS:g} fps stream "
                                 f"to warp from")
            frames, bvp = load_streams(group, speed)
            fps = config.TARGET_FPS  # both streams are declared at the training rate
            starts = list(window_starts(len(frames)))

            ref_result, ref_rows = reference_windows(bvp, fps, starts)
            ref_accepted = ref_result.status in REPORTED
            finite_ref = [row["ref_hr"] for row in ref_rows.values() if np.isfinite(row["ref_hr"])]
            ref_hr_all = float(np.median(finite_ref)) if finite_ref else float("nan")
            cohort = dict(split=attrs["split"], fitzpatrick=int(attrs["fitzpatrick"]),
                          site_group=attrs["site_group"], environment=attrs["environment"],
                          gender=attrs["gender"], age=int(attrs["age"]),
                          age_band=age_band(int(attrs["age"])))

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
                        recording=name, split=cohort["split"], fitzpatrick=cohort["fitzpatrick"],
                        site_group=cohort["site_group"], start=s, t_start_s=s / fps,
                        **ref, **pred,
                        primary=bool(ref_accepted and ref["ref_kept"] and finite),
                        sensitivity=bool(finite)))
                recordings[estimator.name].append(dict(
                    recording=name, subject=attrs["subject_id"], **cohort, speed=speed,
                    n_frames=len(frames), fps=fps, n_windows=len(starts),
                    hr_ppg=float(attrs["hr_ppg"]), hr_ecg=float(attrs["hr_ecg"]),
                    hr_cms=float(attrs["hr_cms"]), ecg_ok=bool(attrs["ecg_ok"]),
                    resp_ok=bool(attrs["resp_ok"]),
                    cardiac_sqi_30s=float(attrs["cardiac_sqi_30s"]),
                    brightness=float(attrs["brightness"]),
                    box_side_lost=float(attrs["box_side_lost"]),
                    qc_pos_bvp_r=float(attrs["qc_pos_bvp_r"]),
                    ref_status=ref_result.status, ref_accepted=ref_accepted,
                    ref_hr=float(ref_result.value), ref_hr_all=ref_hr_all,
                    ref_usable_fraction=float(ref_result.detail.get("usable_fraction", 0.0)),
                    n_ref_refused=len(starts) - sum(r["ref_kept"] for r in ref_rows.values()),
                    value=float(result.value), status=result.status,
                    confidence=float(result.confidence),
                    usable_fraction=float(detail.get("usable_fraction", 0.0)),
                    n_no_peak=int(detail.get("n_no_peak", len(starts))),
                    error_vs_reference=float(result.value - ref_result.value),
                    error_vs_hr_ecg=float(result.value - attrs["hr_ecg"] * speed)))
            if i % 10 == 0 or i == len(names):
                print(f"  {i}/{len(names)} recordings | {(time.perf_counter() - t0) / 60:.1f} min",
                      flush=True)
    return recordings, windows


def build_summary(estimator, recordings, windows, args):
    """Assembles the JSON summary for one estimator."""
    reported = [r for r in recordings if r["status"] in REPORTED]
    summary = dict(
        estimator=estimator.name, dataset="VitalVideos-WW",
        store=str(args.store), device=args.device, split=args.split, speed=args.speed,
        limit=args.limit,
        settings=dict(clip_len=config.CLIP_LEN, window_stride=config.WINDOW_STRIDE,
                      min_confidence=config.MIN_CONFIDENCE, fps=config.TARGET_FPS,
                      window_seconds=round(config.CLIP_LEN / config.TARGET_FPS, 2),
                      training_window_seconds=round(config.CLIP_LEN / config.TARGET_FPS, 2)),
        confounds=CONFOUNDS + ([WARP_CONFOUND] if args.speed != 1.0 else []),
        analysis_sets=dict(
            primary="windows kept by the reference's own gating, in recordings whose "
                    "reference yields a reading",
            sensitivity="every window with a finite reference reading"),
        n_recordings=len(recordings),
        split_counts={s: sum(1 for r in recordings if r["split"] == s)
                      for s in sorted({r["split"] for r in recordings})},
        reference_accepted=dict(accepted=sum(r["ref_accepted"] for r in recordings),
                                total=len(recordings)),
        reference_hr=hr_spread([r["ref_hr"] for r in recordings]),
        reported_hr=hr_spread([r["value"] for r in reported]),
        sets=summarise_task(recordings, windows),
        strata=strata(recordings, windows),
        ecg_cross_check=ecg_cross_check(recordings, args.speed),
    )
    return summary


def print_summary(summary):
    """Prints the headline tables for one estimator."""
    s = summary["settings"]
    print(f"\n=== {summary['estimator']} | split {summary['split']} | speed x{summary['speed']:g} ===")
    print(f"{summary['n_recordings']} recordings {summary['split_counts']} | "
          f"windows {s['window_seconds']} s at {s['fps']:g} fps (training "
          f"{s['training_window_seconds']} s)")
    ref, rep = summary["reference_hr"], summary["reported_hr"]
    if ref:
        print(f"reference HR {ref['min']:.0f}-{ref['max']:.0f} (median {ref['median']:.0f}), "
              f"{ref['above_100']} above 100, {ref['above_120']} above 120 | reported HR "
              f"{rep.get('min', float('nan')):.0f}-{rep.get('max', float('nan')):.0f} "
              f"(median {rep.get('median', float('nan')):.0f})")
    print(f"reference accepted: {summary['reference_accepted']['accepted']}"
          f"/{summary['reference_accepted']['total']}")
    print(f"\n{'set':11} {'rec':>4} {'win':>6} {'nopk':>5} {'MAE':>6} {'RMSE':>6} {'MAPE':>6} "
          f"{'bias':>6} {'LoA':>15} {'r':>5} | {'cover':>5} {'recMAE':>6}")
    for name, m in summary["sets"].items():
        w, rec = m["window"], m["recording"]
        if w.get("n", 0):
            loa = f"{w['loa_low']:+.1f}..{w['loa_high']:+.1f}"
            line = (f"{w['mae']:6.2f} {w['rmse']:6.2f} {w['mape']:6.1f} {w['bias']:+6.2f} "
                    f"{loa:>15} {w['r']:5.2f}")
        else:
            line = f"{'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>15} {'-':>5}"
        rec_mae = f"{rec['mae']:6.2f}" if rec.get("n", 0) else f"{'-':>6}"
        print(f"{name:11} {m['n_recordings']:4d} {w.get('n', 0):6d} "
              f"{m['n_windows_model_no_peak']:5d} {line} | {m['coverage']:5.2f} {rec_mae}")
    for field, labels in summary["strata"].items():
        print(f"\nprimary by {field}:")
        for label, m in labels.items():
            if m.get("n", 0):
                print(f"  {label:12} MAE {m['mae']:6.2f} bias {m['bias']:+6.2f} "
                      f"r {m['r']:5.2f} | {m['n']:5d} windows, {m['n_recordings']:3d} recordings")
    ecg = summary["ecg_cross_check"]
    a = ecg["agreement"]
    if a.get("n", 0):
        print(f"\nECG cross-check ({ecg['n_recordings']} recordings the audit accepted): "
              f"MAE {a['mae']:.2f} bias {a['bias']:+.2f} r {a['r']:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", action="append", choices=registry.available(),
                        help="registered estimator name; repeat for several (default: all)")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--split", choices=SPLITS, default="all",
                        help="all (every subject is zero-shot today), or one split")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="synthetic time warp, a multiple of 0.5 (1.0 = the stored video)")
    parser.add_argument("--device", default="cpu", help="torch device for learned estimators")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N recordings (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    decimation_for(args.speed)  # fail before loading anything if the speed is unusable
    names = args.estimator or registry.available()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.speed == 1.0 else f"_speed{args.speed:g}"

    print(f"store: {args.store}\nestimators: {', '.join(names)} | split: {args.split} | "
          f"speed: x{args.speed:g} | device: {args.device}"
          + (f" | limit {args.limit}" if args.limit else ""))
    estimators = make_estimators(names, args.device)
    recordings, windows = evaluate(args.store, estimators, args.split, args.speed, args.limit)

    for estimator in estimators:
        name = estimator.name
        summary = build_summary(estimator, recordings[name], windows[name], args)
        _write_csv(args.out_dir / f"windows_{name}{suffix}.csv", windows[name])
        _write_csv(args.out_dir / f"recordings_{name}{suffix}.csv", recordings[name])
        (args.out_dir / f"summary_{name}{suffix}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print_summary(summary)

    print(f"\noutputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
