"""App-level evaluation of heart-rate estimators on the Phase-3 stores' held-out recordings.

Runs registered estimators, through the app's estimator code and gates, over recordings
of the Phase-3 training stores that the models did not train on, and scores their
per-window heart rate against the reference BVP read out over the same windows.

- `--split val` (the default): v2's validation split, the recordings NB_P3_18 and NB_P3_31
  validated on (indexed: MCD 232, DLCN 168, UBFC-rPPG 7, PhysDrive 68), so a model-level
  score and this app-level one describe the same recordings.
- `--split heldout`: every indexed recording that none of the estimators run trained on;
  for v2 and v3, v2's validation split of the three training corpora and all of
  PhysDrive, which no model trained on.

Which recordings an estimator trained on is its own declaration (`trained_on`, pairs of
dataset and split), and no estimator is scored on them: with `val` a run that would do so
stops, with `heldout` they are left out for every estimator in the run.

The two analysis sets are the ones check_ubfc_phys.py fixed, and both are decided by the
reference alone, never by the estimator under test:

- primary: windows kept by the reference's own gating (the gates the app applies to an
  estimate), within recordings whose reference yields a reading;
- sensitivity: every window with a finite reference reading, in every recording.

`shared` pools MCD, DLCN and UBFC-rPPG, as NB_P3_18's validation metric did, and is
reported when a run covers all three. Two caveats are written into every summary:

- the frames are the stores' crops at the stores' frame rates, so the capture path - box,
  decimation, rate verdict - is not exercised here; the regression check covers it;
- labels are used as stored. The lags NB_P3_31 aligned for training are fractions of a
  second, which move a window's heart rate very little, and an inverted reference reads
  the same heart rate.

Usage:
    python -m app.live_vitals.scripts.check_phase3_stores [--estimator NAME ...]
        [--corpus MCD DLCN UBFC-rPPG PhysDrive] [--split val|heldout]
        [--store-dir DIR] [--split-file FILE] [--device cpu|cuda] [--limit N]
        [--out-dir DIR]

No frames are rendered or written. The stores are read-only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals import config
from app.live_vitals.estimators import registry
from app.live_vitals.scripts.check_ubfc_regression import window_starts
from app.live_vitals.scripts.check_ubfc_phys import (REPORTED, _plain, _write_csv,
                                                     common_windows, make_estimators,
                                                     predicted_windows, print_common,
                                                     reference_windows, summarise_task)
from app.live_vitals.scripts.bench_outputs import provenance, subset_suffix, unique

DEFAULT_STORE_DIR = Path(r"D:\QualityPhys\phase 3 datasets")
DEFAULT_SPLIT_FILE = _REPO_ROOT / "Data" / "phase3_split.csv"
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "phase3_eval"
STORES = {"MCD": "mcd_phase3_rc_none.h5", "DLCN": "dlcn_phase3_rc_none.h5",
          "UBFC-rPPG": "ubfc_rppg_phase3_rc_none.h5",
          "PhysDrive": "physdrive_phase3_rc_none.h5"}
CORPORA = tuple(STORES)
SHARED = ("MCD", "DLCN", "UBFC-rPPG")  # pooled as NB_P3_18's validation metric
SPLITS = ("val", "heldout")
CAVEATS = [
    "frames are the stores' crops at the stores' frame rates: the capture path (box, "
    "decimation, rate verdict) is not exercised",
    "labels as stored: the training lags of a fraction of a second are not shifted out",
]


# ---------------------------------------------------------------- selection --

def select(split_file, corpora, split, estimators):
    """The recordings to score, in the split file's order, and what was left out and why.

    Only indexed recordings count - those NB_P3_18's filters admitted. With `val`, v2's
    validation split; with `heldout`, every recording no estimator in the run trained on.
    Either way a recording an estimator trained on is never scored: `val` stops if one
    is, `heldout` leaves it out for every estimator, so all of them score the same set.
    """
    table = pd.read_csv(split_file, dtype=str, keep_default_na=False)
    table = table[table["dataset"].isin(corpora)]
    trained = set().union(*(e.trained_on for e in estimators))
    indexed = table["indexed"] == "True"
    on_training = pd.Series([(d, s) in trained for d, s in zip(table["dataset"], table["split"])],
                            index=table.index, dtype=bool)
    if split == "val":
        chosen = indexed & (table["split"] == "val")
        clash = {(d, s) for d, s in zip(table.loc[chosen & on_training, "dataset"],
                                        table.loc[chosen & on_training, "split"])}
        if clash:
            names = [e.name for e in estimators if clash & set(e.trained_on)]
            raise SystemExit(f"ERROR: {', '.join(names)} trained on {sorted(clash)}; "
                             f"leave {'it' if len(names) == 1 else 'them'} out with --estimator")
    else:
        chosen = indexed & table["split"].isin(["train", "val"]) & ~on_training
    left_out = dict(not_indexed=int((~indexed).sum()),
                    trained_on=int((indexed & on_training).sum()),
                    other_split=int((indexed & ~on_training & ~chosen).sum()))
    return table[chosen], left_out


# ----------------------------------------------------------------------- run --

def evaluate(store_dir, rows, estimators, limit):
    """Runs every estimator over the chosen recordings; returns recording and window rows."""
    recordings = {e.name: [] for e in estimators}
    windows = {e.name: [] for e in estimators}
    todo = []
    for corpus in CORPORA:
        part = rows[rows["dataset"] == corpus]
        part = part.head(limit) if limit else part
        if len(part):
            todo.append((corpus, part))
    total, done, t0 = sum(len(p) for _, p in todo), 0, time.perf_counter()
    for corpus, part in todo:
        with h5py.File(store_dir / STORES[corpus], "r") as store:
            for row in part.itertuples():
                group = store[row.group]
                attrs = {k: _plain(v) for k, v in group.attrs.items()}
                frames = group["frames"][()]
                bvp = group["bvp"][()]
                fps = float(attrs.get("fps", config.TARGET_FPS))
                starts = list(window_starts(len(frames)))

                ref_result, ref_rows = reference_windows(bvp, fps, starts)
                ref_accepted = ref_result.status in REPORTED
                finite_ref = [r["ref_hr"] for r in ref_rows.values() if np.isfinite(r["ref_hr"])]
                ref_hr_all = float(np.median(finite_ref)) if finite_ref else float("nan")
                key = f"{corpus}/{row.group}"

                for estimator in estimators:
                    result = estimator.estimate(frames, fps)
                    pred_rows = predicted_windows(result)
                    detail = result.detail or {}
                    for s in starts:
                        ref = ref_rows[s]
                        pred = pred_rows.get(s, dict(pred_hr=float("nan"),
                                                     pred_confidence=float("nan"),
                                                     pred_kept=False))
                        finite = np.isfinite(ref["ref_hr"])
                        windows[estimator.name].append(dict(
                            recording=key, dataset=corpus, subject=row.subject, start=s,
                            t_start_s=s / fps, **ref, **pred,
                            primary=bool(ref_accepted and ref["ref_kept"] and finite),
                            sensitivity=bool(finite)))
                    recordings[estimator.name].append(dict(
                        recording=key, dataset=corpus, subject=row.subject, split=row.split,
                        n_frames=len(frames), fps=fps, n_windows=len(starts),
                        cardiac_sqi=float(attrs.get("cardiac_sqi", float("nan"))),
                        ref_status=ref_result.status, ref_accepted=ref_accepted,
                        ref_hr=float(ref_result.value), ref_hr_all=ref_hr_all,
                        ref_usable_fraction=float(ref_result.detail.get("usable_fraction", 0.0)),
                        n_ref_refused=len(starts) - sum(r["ref_kept"] for r in ref_rows.values()),
                        value=float(result.value), status=result.status,
                        confidence=float(result.confidence),
                        usable_fraction=float(detail.get("usable_fraction", 0.0)),
                        n_no_peak=int(detail.get("n_no_peak", len(starts))),
                        error_vs_reference=float(result.value - ref_result.value)))
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"  {done}/{total} recordings | "
                          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return recordings, windows


def groups_of(corpora):
    """The groups a summary reports: each corpus, and `shared` when all three are present."""
    groups = {c: (c,) for c in CORPORA if c in corpora}
    if set(SHARED) <= set(corpora):
        groups["shared"] = SHARED
    return groups


def build_summary(estimator, recordings, windows, args, corpora, left_out):
    """Assembles the JSON summary for one estimator."""
    fps_values = sorted({round(r["fps"], 2) for r in recordings})
    groups = groups_of(corpora)

    def pick(rows, members):
        return [r for r in rows if r["dataset"] in members]

    return dict(
        estimator=estimator.name,
        store_dir=str(args.store_dir), split_file=str(args.split_file), split=args.split,
        corpora=list(corpora), device=args.device, limit=args.limit,
        trained_on=sorted(list(p) for p in estimator.trained_on),
        settings=dict(clip_len=config.CLIP_LEN, window_stride=config.WINDOW_STRIDE,
                      min_confidence=config.MIN_CONFIDENCE, fps=fps_values),
        caveats=CAVEATS,
        analysis_sets=dict(
            primary="windows kept by the reference's own gating, in recordings whose "
                    "reference yields a reading",
            sensitivity="every window with a finite reference reading"),
        recordings_left_out=left_out,
        n_recordings={g: len(pick(recordings, m)) for g, m in groups.items()},
        reference_accepted={g: dict(accepted=sum(r["ref_accepted"] for r in pick(recordings, m)),
                                    total=len(pick(recordings, m)))
                            for g, m in groups.items()},
        groups={g: summarise_task(pick(recordings, m), pick(windows, m))
                for g, m in groups.items()},
    )


def print_summary(summary):
    """Prints the headline table for one estimator."""
    s = summary["settings"]
    print(f"\n=== {summary['estimator']} | split {summary['split']} ===")
    print(f"recordings {summary['n_recordings']} | fps {s['fps']} | "
          f"left out {summary['recordings_left_out']}")
    print("reference accepted: " + ", ".join(
        f"{g} {v['accepted']}/{v['total']}" for g, v in summary["reference_accepted"].items()))
    print(f"\n{'group':10} {'set':11} {'rec':>4} {'win':>6} {'nopk':>5} {'MAE':>6} {'RMSE':>6} "
          f"{'MAPE':>6} {'bias':>6} {'LoA':>15} {'r':>5} | {'cover':>5} {'recMAE':>6}")
    for group, sets in summary["groups"].items():
        for name, m in sets.items():
            w, rec = m["window"], m["recording"]
            if w.get("n", 0):
                loa = f"{w['loa_low']:+.1f}..{w['loa_high']:+.1f}"
                line = (f"{w['mae']:6.2f} {w['rmse']:6.2f} {w['mape']:6.1f} {w['bias']:+6.2f} "
                        f"{loa:>15} {w['r']:5.2f}")
            else:
                line = f"{'-':>6} {'-':>6} {'-':>6} {'-':>6} {'-':>15} {'-':>5}"
            rec_mae = f"{rec['mae']:6.2f}" if rec.get("n", 0) else f"{'-':>6}"
            print(f"{group:10} {name:11} {m['n_recordings']:4d} {w.get('n', 0):6d} "
                  f"{m['n_windows_model_no_peak']:5d} {line} | {m['coverage']:5.2f} {rec_mae}")
    if summary.get("common_windows"):
        print("\nlike for like, primary set, on the windows both estimators read:")
        for group, common in summary["common_windows"].items():
            print_common(common, label=f"{group:<10}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", action="append", choices=registry.available(),
                        help="registered estimator name; repeat for several (default: all)")
    parser.add_argument("--corpus", nargs="+", choices=CORPORA, default=list(CORPORA))
    parser.add_argument("--split", choices=SPLITS, default="val",
                        help="val: v2's validation split; heldout: every recording no "
                             "estimator in the run trained on")
    parser.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    parser.add_argument("--split-file", type=Path, default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--device", default="cpu", help="torch device for learned estimators")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N recordings of each corpus (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    names = unique(args.estimator or registry.available())
    corpora = [c for c in CORPORA if c in args.corpus]
    suffix = subset_suffix(("corpus-", "-".join(corpora), "-".join(CORPORA)),
                           ("split-", args.split, "val"),
                           ("limit", args.limit or None, None))
    estimators = make_estimators(names, args.device)
    rows, left_out = select(args.split_file, corpora, args.split, estimators)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"stores: {args.store_dir}\nestimators: {', '.join(names)} | split: {args.split} | "
          f"device: {args.device}" + (f" | limit {args.limit} per corpus" if args.limit else ""))
    print("recordings: " + ", ".join(f"{c} {int((rows['dataset'] == c).sum())}" for c in corpora)
          + (f" (the first {args.limit} of each scored)" if args.limit else "")
          + f" | left out: {left_out}")
    recordings, windows = evaluate(args.store_dir, rows, estimators, args.limit)

    groups = groups_of(corpora)
    for estimator in estimators:
        name = estimator.name
        summary = build_summary(estimator, recordings[name], windows[name], args, corpora,
                                left_out)
        if len(estimators) > 1:
            summary["common_windows"] = {
                g: common_windows([w for w in windows[name] if w["dataset"] in m],
                                  {o.name: [w for w in windows[o.name] if w["dataset"] in m]
                                   for o in estimators if o.name != name})
                for g, m in groups.items()}
        summary["provenance"] = provenance(estimator)
        _write_csv(args.out_dir / f"windows_{name}{suffix}.csv", windows[name])
        _write_csv(args.out_dir / f"recordings_{name}{suffix}.csv", recordings[name])
        (args.out_dir / f"summary_{name}{suffix}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print_summary(summary)

    print(f"\noutputs: {args.out_dir}" + (f" (subset run: names end in {suffix})" if suffix else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())