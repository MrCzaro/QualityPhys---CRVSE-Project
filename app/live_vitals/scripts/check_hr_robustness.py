"""Pre-registered checks of the heart-rate readout (NB_P3_38, Part 0).

The checks run once each, at a commit that follows Part 0's, through the app's own
estimator code. Each writes its summary (committed) and its tables (local) to
Data/hr_robustness/. A summary carries the rule's settings and the readout's, so that the
notebook can hold them against Part 0, and the provenance block every benchmark summary
carries.

low-hr: does the readout keep fewer usable, right windows below 60 bpm? The window rows
come from the benchmark scripts' own evaluation functions: VitalVideos-WW (all 300
recordings) decides; UBFC-Phys T1 and Rice's clamped crop are described. Only primary
windows count - those the reference's own readout keeps, in recordings whose reference is
reported - and each is classed by the reference's rate in it: low below 60 bpm, typical
from 65 to below 90, high from 90; 60 to 65 is a gap that keeps windows near the line out
of both. A window is good when the readout keeps it and its rate lies within 10% of the
reference's. The decisive figure is v2's share of good windows among low windows minus that
among typical ones, in points, with a 95% interval from recordings resampled whole; a loss
of 10 points is the bar.

Usage:
    python -m app.live_vitals.scripts.check_hr_robustness low-hr [--device cpu|cuda]
        [--limit N] [--vv-store PATH] [--ubfc-phys-store PATH] [--rice-store PATH]
        [--out-dir DIR]

No frame is rendered or written; every store is read-only.
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals import config
from app.live_vitals.scripts import check_rice_ippg, check_ubfc_phys, check_vitalvideos
from app.live_vitals.scripts.bench_outputs import provenance, subset_suffix
from app.live_vitals.scripts.check_ubfc_phys import REPORTED, _write_csv, make_estimators

DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "hr_robustness"
PART0 = "NB_P3_38 Part 0, pre-registered 2026-11-01"
ESTIMATORS = ("hr_physnet_v2", "hr_spectral")
DECIDING = "hr_physnet_v2"  # every verdict is about the shipped model; POS is described

# Check 1 (low-hr), Part 0's numbers.
LOW_BELOW = 60.0
TYPICAL = (65.0, 90.0)      # from 65 up to, not including, 90 bpm
TOLERANCE = 0.10            # a window is right within 10% of the reference's rate ...
DOUBLE_TOLERANCE = 0.10     # ... and reads double within 10% of twice that rate
BAR_POINTS = -10.0          # a loss of 10 points of good windows is material
BOOT_DRAWS = 10_000
BOOT_SEED = 42
CLASSES = ("low", "gap", "typical", "high")
RATE_BANDS = ((55.0, "<=55"), (60.0, "55-60"), (65.0, "60-65"), (75.0, "65-75"),
              (90.0, "75-90"))  # anything above the last bound is ">90"
READOUT = ("CLIP_LEN", "WINDOW_STRIDE", "TARGET_FPS", "HR_LOW_HZ", "HR_HIGH_HZ",
           "CONFIDENCE_FLOOR_HZ", "MIN_CONFIDENCE", "MAD_OUTLIER_K", "MAD_FLOOR_BPM",
           "MIN_USABLE_WINDOWS", "MIN_USABLE_FRACTION", "MIN_REPORTABLE_FRACTION",
           "HARMONIC_GUARD_ENABLED", "MIN_FPS_ACCEPT", "MIN_FPS_WARN", "MAX_FPS_ACCEPT")


# --------------------------------------------------------------------- sets --

@dataclass(frozen=True)
class LowHrSet:
    """One of check 1's sets: its name, its role, the key its tables are named by, its store."""
    name: str
    role: str
    key: str
    store: Path


def low_hr_sets(vv_store, ubfc_phys_store, rice_store):
    """Check 1's three sets in Part 0's order, the decisive one first."""
    return (LowHrSet("VitalVideos-WW", "decisive", "vitalvideos", Path(vv_store)),
            LowHrSet("UBFC-Phys T1", "descriptive", "ubfc_phys_t1", Path(ubfc_phys_store)),
            LowHrSet("Rice iPPG, clamped crop", "descriptive", "rice_clamped", Path(rice_store)))


def evaluate_set(spec, estimators, limit):
    """Recording and window rows for one set, from its benchmark script's own evaluation."""
    if spec.key == "vitalvideos":
        return check_vitalvideos.evaluate(spec.store, estimators, "all", 1.0, limit)
    if spec.key == "ubfc_phys_t1":
        return check_ubfc_phys.evaluate(spec.store, estimators, ["T1"], limit)
    if spec.key == "rice_clamped":
        return check_rice_ippg.evaluate(spec.store, estimators, "frames_clamped",
                                        list(check_rice_ippg.ACTIVITIES), limit)
    raise ValueError(f"unknown set: {spec.key}")


# ------------------------------------------------------------------ windows --

def rate_class(rate):
    """Part 0's class of a window's reference rate (bpm); None when the rate is not finite."""
    if not math.isfinite(rate):
        return None
    if rate < LOW_BELOW:
        return "low"
    if rate < TYPICAL[0]:
        return "gap"
    if rate < TYPICAL[1]:
        return "typical"
    return "high"


def scored_windows(rows):
    """The primary windows, each with its class and its kept, good, off and doubled flags.

    Kept means the readout kept the window with a finite rate; good, kept and within the
    tolerance of the reference; off, kept and outside it. Doubled - within the tolerance of
    twice the reference's rate - is flagged whether the readout kept the window or not.
    """
    columns = ["recording", "ref_hr", "ref_confidence", "pred_hr", "pred_confidence",
               "pred_kept", "primary"]
    frame = pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)
    frame = frame[frame["primary"].astype(bool)].copy()
    ref = frame["ref_hr"].astype(float)
    pred = frame["pred_hr"].astype(float)
    kept = frame["pred_kept"].astype(bool) & np.isfinite(pred)
    right = (pred - ref).abs() <= TOLERANCE * ref
    frame["rate_class"] = [rate_class(rate) for rate in ref]
    frame["kept"] = kept
    frame["good"] = kept & right
    frame["kept_off"] = kept & ~right
    frame["doubled"] = np.isfinite(pred) & ((pred - 2.0 * ref).abs() <= DOUBLE_TOLERANCE * 2.0 * ref)
    return frame


def _share(flags):
    return float(np.mean(flags)) if len(flags) else None


def _median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def class_figures(frame):
    """Per class: windows, recordings, the shares Part 0 describes and the median confidences."""
    figures = {}
    for name in CLASSES:
        part = frame[frame["rate_class"] == name]
        figures[name] = dict(
            windows=int(len(part)), recordings=int(part["recording"].nunique()),
            share_kept=_share(part["kept"]), share_good=_share(part["good"]),
            share_kept_off=_share(part["kept_off"]), share_doubled=_share(part["doubled"]),
            median_confidence=_median(part["pred_confidence"]),
            median_reference_confidence=_median(part["ref_confidence"]))
    return figures


def good_share_difference(frame, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """The low minus typical share of good windows, in points, with its 95% interval.

    Recordings are resampled with replacement and keep all their windows together, because
    the windows of one recording are not independent. A draw that holds no low or no
    typical window has no difference; such draws are counted and left out, never imputed.
    """
    part = frame[frame["rate_class"].isin(("low", "typical"))]
    names = sorted(part["recording"].unique())
    if not names:
        return dict(point=None, lower=None, upper=None, recordings=0, draws=draws, draws_used=0)
    index = part["recording"].map({name: i for i, name in enumerate(names)}).to_numpy()
    low = (part["rate_class"] == "low").to_numpy()
    good = part["good"].to_numpy(dtype=float)
    n = len(names)
    n_low = np.bincount(index[low], minlength=n).astype(float)
    g_low = np.bincount(index[low], weights=good[low], minlength=n)
    n_typical = np.bincount(index[~low], minlength=n).astype(float)
    g_typical = np.bincount(index[~low], weights=good[~low], minlength=n)

    def difference(nl, gl, nt, gt):
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100.0 * (gl / nl - gt / nt)

    point = float(difference(n_low.sum(), g_low.sum(), n_typical.sum(), g_typical.sum()))
    pick = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    values = difference(n_low[pick].sum(1), g_low[pick].sum(1), n_typical[pick].sum(1),
                        g_typical[pick].sum(1))
    values = values[np.isfinite(values)]
    if not math.isfinite(point) or not values.size:
        return dict(point=point if math.isfinite(point) else None, lower=None, upper=None,
                    recordings=n, draws=draws, draws_used=int(values.size))
    lower, upper = np.percentile(values, [2.5, 97.5])
    return dict(point=point, lower=float(lower), upper=float(upper), recordings=n, draws=draws,
                draws_used=int(values.size))


def outcome(lower, upper, bar=BAR_POINTS):
    """Part 0's outcome of a loss against the bar: confirmed, not confirmed or inconclusive."""
    if lower is None or upper is None:
        return "not estimable"
    if upper <= bar:
        return "confirmed"
    if lower > bar:
        return "not confirmed"
    return "inconclusive"


# --------------------------------------------------------------- recordings --

def rate_band(rate):
    """The band of a recording's reference rate (bpm), as in the table seen before the freeze."""
    for bound, label in RATE_BANDS:
        if rate <= bound:
            return label
    return f">{RATE_BANDS[-1][0]:g}"


def recording_bands(rows):
    """Per band of the recording's reference rate: recordings, reported, ok, median |error|."""
    bands = {}
    for row in rows:
        rate = float(row["ref_hr"])
        if not math.isfinite(rate):
            continue
        band = bands.setdefault(rate_band(rate), dict(recordings=0, reported=0, ok=0, errors=[]))
        band["recordings"] += 1
        band["ok"] += row["status"] == "ok"
        if row["status"] in REPORTED:
            band["reported"] += 1
            error = float(row["error_vs_reference"])
            if math.isfinite(error):
                band["errors"].append(abs(error))
    order = [label for _, label in RATE_BANDS] + [f">{RATE_BANDS[-1][0]:g}"]
    return [dict(band=label, recordings=b["recordings"], reported=b["reported"], ok=b["ok"],
                 median_abs_error=_median(b["errors"]))
            for label in order if (b := bands.get(label))]


# ------------------------------------------------------------------ summary --

def readout_settings():
    """The readout's settings at this commit, the values Part 0's freeze record lists."""
    return {name: getattr(config, name) for name in READOUT}


def low_hr_settings(sets, limit):
    """Check 1's rule as this run applies it, for the notebook to hold against Part 0."""
    return dict(
        classes=dict(low_below=LOW_BELOW, typical=list(TYPICAL), high_from=TYPICAL[1]),
        windows="primary: kept by the reference's own readout, in recordings whose reference "
                "is reported",
        tolerance=TOLERANCE, double_tolerance=DOUBLE_TOLERANCE, bar_points=BAR_POINTS,
        interval=dict(unit="recording", draws=BOOT_DRAWS, seed=BOOT_SEED, percentiles=[2.5, 97.5]),
        estimators=list(ESTIMATORS), deciding=DECIDING,
        sets=[dict(name=s.name, role=s.role, store=str(s.store)) for s in sets],
        reported_statuses=list(REPORTED), limit=limit)


def _fmt(value, pct=False):
    if value is None:
        return "     -"
    return f"{100 * value:5.1f}%" if pct else f"{value:6.3f}"


def print_low_hr(summary):
    """Prints each set's table by class and the decisive figure."""
    for set_name, block in summary["sets"].items():
        for name, figures in block["estimators"].items():
            print(f"\n=== {set_name} ({block['role']}) | {name} | "
                  f"{figures['primary_windows']} primary windows in {figures['recordings']} recordings")
            print(f"  {'class':<8} {'windows':>8} {'recs':>5}   {'kept':>6} {'good':>6} "
                  f"{'off':>6} {'double':>6}   {'conf':>6} {'ref':>6}")
            for cls in CLASSES:
                c = figures["classes"][cls]
                shares = " ".join(_fmt(c[key], pct=True) for key in
                                  ("share_kept", "share_good", "share_kept_off", "share_doubled"))
                print(f"  {cls:<8} {c['windows']:>8} {c['recordings']:>5}   {shares}   "
                      f"{_fmt(c['median_confidence'])} {_fmt(c['median_reference_confidence'])}")
            d = figures["low_minus_typical_points"]
            if d["lower"] is None:
                print(f"  low - typical, good windows: not estimable -> {figures['outcome']}")
            else:
                print(f"  low - typical, good windows: {d['point']:+.1f} points (95% {d['lower']:+.1f} "
                      f"to {d['upper']:+.1f}; {d['recordings']} recordings, {d['draws_used']} of "
                      f"{d['draws']} draws) -> {figures['outcome']}")
    verdict = summary["verdict"]
    print(f"\nVERDICT, check 1 ({verdict['set']}, {verdict['estimator']}): {verdict['outcome']}")


# ---------------------------------------------------------------------- run --

def run_low_hr(args):
    """Check 1: evaluates the three sets, scores their windows and writes the outputs."""
    sets = low_hr_sets(args.vv_store, args.ubfc_phys_store, args.rice_store)
    estimators = make_estimators(list(ESTIMATORS), args.device)
    deciding = next(e for e in estimators if e.name == DECIDING)
    # Read before anything is written, so the run's own outputs never list as changed files.
    produced_by = provenance(deciding)
    suffix = subset_suffix(("limit", args.limit or None, None))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = dict(check="low-hr", part0=PART0, settings=low_hr_settings(sets, args.limit),
                   readout=readout_settings(), sets={})
    for spec in sets:
        print(f"\n--- {spec.name} ({spec.role}): {spec.store}", flush=True)
        recordings, windows = evaluate_set(spec, estimators, args.limit)
        scored = {}
        for name in ESTIMATORS:
            frame = scored_windows(windows[name])
            difference = good_share_difference(frame)
            scored[name] = dict(
                recordings=len(recordings[name]),
                reported=sum(r["status"] in REPORTED for r in recordings[name]),
                primary_windows=int(len(frame)),
                classes=class_figures(frame),
                low_minus_typical_points=difference,
                outcome=outcome(difference["lower"], difference["upper"]),
                recording_bands=recording_bands(recordings[name]))
            _write_csv(args.out_dir / f"windows_low_hr_{spec.key}_{name}{suffix}.csv", windows[name])
            _write_csv(args.out_dir / f"recordings_low_hr_{spec.key}_{name}{suffix}.csv",
                       recordings[name])
        summary["sets"][spec.name] = dict(role=spec.role, estimators=scored)

    decisive = summary["sets"][sets[0].name]["estimators"][DECIDING]
    summary["verdict"] = dict(set=sets[0].name, estimator=DECIDING, outcome=decisive["outcome"],
                              low_minus_typical_points=decisive["low_minus_typical_points"])
    summary["provenance"] = produced_by
    (args.out_dir / f"summary_low_hr{suffix}.json").write_text(json.dumps(summary, indent=2),
                                                               encoding="utf-8")
    print_low_hr(summary)
    print(f"\noutputs: {args.out_dir}" + (f" (subset run: names end in {suffix})" if suffix else ""))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    checks = parser.add_subparsers(dest="check", required=True)
    low = checks.add_parser("low-hr", help="check 1: low heart rates and the confidence gate")
    low.add_argument("--device", default="cpu", help="torch device for learned estimators")
    low.add_argument("--limit", type=int, default=None,
                     help="the first N recordings of each set only (smoke test)")
    low.add_argument("--vv-store", type=Path, default=check_vitalvideos.DEFAULT_STORE)
    low.add_argument("--ubfc-phys-store", type=Path, default=check_ubfc_phys.DEFAULT_STORE)
    low.add_argument("--rice-store", type=Path, default=check_rice_ippg.DEFAULT_STORE)
    low.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.check == "low-hr":
        run_low_hr(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())