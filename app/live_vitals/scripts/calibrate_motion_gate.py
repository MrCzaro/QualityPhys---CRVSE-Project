"""Calibrates a periodic-motion capture gate on recordings it will not be judged on.

The ECG-Fitness motion diagnostic (2026-10-08, exploratory) found that during exercise
both estimators lock onto the head's movement rhythm, that the app's own crop sees that
rhythm, and that one number - the median in-band displacement of the crop - separated
the exercise recordings from the resting ones there. A gate on that number is honest
only if its threshold is set somewhere else. This script measures it, with the
diagnostic's own functions, on the still corpora and on your own captures, and proposes
a threshold by a rule fixed before the run. It never reads ECG-Fitness.

Per recording, over the app's analysis windows (CLIP_LEN frames, WINDOW_STRIDE apart):
  amp_median   median in-band (cardiac band) rms displacement of the crop, crop pixels
  amp_p90      90th percentile of the same over windows
  clear_frac   share of windows whose crop rhythm reaches MIN_CONFIDENCE periodicity;
               a window with no readable rhythm counts as not clear
Two candidate rules: A refuses a capture whose amp_median reaches T; B also needs
clear_frac of at least 0.5 - periodic motion in at least half the windows. C is A or B,
each at its own proposed threshold.

The rule for the proposed threshold, fixed before any still recording was measured: the
smallest T on a 0.1 px grid at which no still stratum loses more than --budget (1%) of
its usable recordings. The strata are the stores, with UBFC-Phys split by task, since T2
is speech and the likeliest false refusal. PhysDrive (a moving vehicle) and your own
captures are reported beside the rule, never inside it.

Added 2026-10-09, after the --limit 20 run and before the full results: rule B never goes
below the resting range - rule A's threshold over the strata where people sit still
(UBFC-Phys T2 and T3 left out, since the subjects talk). That run showed rule B falling to
the bottom of the grid, and still recordings whose crop rhythm is clear but tiny (at most
0.1 px, at heart-rate frequencies - most likely the head's own heartbeat motion): a
periodic motion no larger than a resting head's should never refuse a capture.

Every Phase-3 store crops with the app's frozen box (the median of N_DETECT_FRAMES
landmark boxes, never tracked), so its crops carry head motion as the app's do. Your
captures go through the app's own crops_from_video.

Writes calibration_recordings.csv and calibration_windows.csv (git-ignored; appended as
it goes, so an interrupted run resumes where it stopped) and summary_calibration.json to
--out-dir. Nothing written here contains a frame.

Usage:
    python -m app.live_vitals.scripts.calibrate_motion_gate
        [--stores H5 ...] [--videos PATH ...] [--no-videos] [--out-dir DIR]
        [--limit N] [--budget F] [--fresh]
"""
import argparse
import csv
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
from app.live_vitals.scripts.diagnose_ecg_fitness_motion import crop_shifts, window_rhythms

DEFAULT_STORE_DIR = Path(r"D:\QualityPhys\phase 3 datasets")
DEFAULT_VIDEO_DIR = Path(r"D:\QualityPhys\demo_data")
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "motion_gate"
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MOTION_STORES = {"physdrive"}  # reported beside the rule, never in it: a moving vehicle
OWN = "own captures"
RULE_B_CLEAR = 0.5
TALKING = {"ubfc_phys T2", "ubfc_phys T3"}  # still strata in which the subjects speak
GRID = [round(0.1 * k, 1) for k in range(1, 51)]  # candidate thresholds, px
SHOWN = (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0)
REC_FIELDS = ["source", "stratum", "recording", "usable", "fps", "n_frames", "n_windows",
              "amp_median", "amp_p90", "clear_frac", "rhythm_when_clear", "verdict"]
WIN_FIELDS = ["source", "recording", "start", "crop_rhythm", "crop_periodicity",
              "crop_channel", "crop_amp"]
NAN = float("nan")


# ----------------------------------------------------------------- measure --

def crop_motion(frames, fps):
    """Per-window crop rhythm, periodicity and in-band displacement, as the diagnostic
    computes them for its crop source, plus the per-recording summary. None if the
    recording is shorter than one analysis window."""
    n = len(frames)
    if n < config.CLIP_LEN:
        return None, None
    starts = list(range(0, n - config.CLIP_LEN + 1, config.WINDOW_STRIDE))
    shift = crop_shifts(frames)
    position = np.cumsum(shift, axis=0)
    rows = window_rhythms(dict(crop_x=shift[:, 0], crop_y=shift[:, 1]),
                          [position[:, 0], position[:, 1]], 1.0, fps, starts, "crop")
    amp = np.array([r["crop_amp"] for r in rows], dtype=float)
    periodicity = np.array([r["crop_periodicity"] for r in rows], dtype=float)
    clear = np.nan_to_num(periodicity, nan=0.0) >= config.MIN_CONFIDENCE
    rhythm = np.array([r["crop_rhythm"] for r in rows], dtype=float)
    summary = dict(n_frames=n, n_windows=len(rows),
                   amp_median=float(np.median(amp)), amp_p90=float(np.percentile(amp, 90)),
                   clear_frac=float(clear.mean()),
                   rhythm_when_clear=float(np.median(rhythm[clear])) if clear.any() else NAN)
    return rows, summary


def _text(value):
    """An HDF5 string attribute as str, whichever way h5py hands it back."""
    return value.decode() if isinstance(value, bytes) else str(value)


def _flag(value):
    """An HDF5 boolean attribute, whether stored as a bool, a number or a string."""
    if isinstance(value, (bytes, str)):
        return _text(value).strip().lower() in ("true", "1", "yes")
    return bool(value)


def store_label(path):
    """'ubfc_phys_phase3_rc_none.h5' -> 'ubfc_phys'."""
    return path.stem.replace("_phase3_rc_none", "").replace("_rc_none", "")


def stratum_of(source, attrs):
    """UBFC-Phys is split by task; every other store is one stratum."""
    if source == "ubfc_phys" and "task" in attrs:
        return f"ubfc_phys {_text(attrs['task'])}"
    return source


# ------------------------------------------------------------------ output --

class Writer:
    """Appends recording and window rows as they are measured, so a run can resume."""

    def __init__(self, out_dir, fresh):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.rec_path = out_dir / "calibration_recordings.csv"
        self.win_path = out_dir / "calibration_windows.csv"
        if fresh:
            for path in (self.rec_path, self.win_path):
                path.unlink(missing_ok=True)
        self.done = set()
        if self.rec_path.exists():
            with open(self.rec_path, newline="", encoding="utf-8") as f:
                self.done = {(r["source"], r["recording"]) for r in csv.DictReader(f)}
            self._drop_orphan_windows()

    def _drop_orphan_windows(self):
        """Window rows of a recording interrupted before its summary row are re-measured."""
        if not self.win_path.exists():
            return
        with open(self.win_path, newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if (r["source"], r["recording"]) in self.done]
        self._write(self.win_path, WIN_FIELDS, rows, mode="w")

    @staticmethod
    def _write(path, fields, rows, mode="a"):
        new = mode == "w" or not path.exists()
        with open(path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new:
                writer.writeheader()
            writer.writerows(rows)

    def add(self, source, recording, windows, record):
        self._write(self.win_path, WIN_FIELDS,
                    [{"source": source, "recording": recording,
                      **{k: w[k] for k in WIN_FIELDS[2:]}} for w in windows])
        self._write(self.rec_path, REC_FIELDS, [record])
        self.done.add((source, recording))


# ------------------------------------------------------------------ sources --

def refuse_benchmarks(stores):
    """Stops before measuring anything if any store holds a benchmark group."""
    for path in stores:
        with h5py.File(path, "r") as store:
            bench = [n for n in store if _text(store[n].attrs.get("role", "")) == "benchmark"]
        if bench:
            raise SystemExit(f"ERROR: {path.name} holds {len(bench)} benchmark group(s), e.g. "
                             f"{bench[0]}; this script never reads the benchmark")


def measure_group(source, name, group, writer, problems):
    """Measures one store recording and records it, or notes why it was skipped."""
    attrs = group.attrs
    if "frames" not in group or "fps" not in attrs:
        problems.append(f"{source}/{name}: no frames or no fps; skipped")
        return
    fps = float(attrs["fps"])
    windows, summary = crop_motion(group["frames"][()], fps)
    if summary is None:
        problems.append(f"{source}/{name}: shorter than one window; skipped")
        return
    usable = _flag(attrs["usable"]) if "usable" in attrs else True
    writer.add(source, name, windows,
               dict(source=source, stratum=stratum_of(source, attrs), recording=name,
                    usable=usable, fps=fps, verdict="", **summary))


def measure_store(path, writer, limit, problems):
    source = store_label(path)
    t0 = time.perf_counter()
    with h5py.File(path, "r") as store:
        names = sorted(store.keys())[:limit]
        todo = [n for n in names if (source, n) not in writer.done]
        print(f"{source}: {len(names)} recordings, {len(names) - len(todo)} already measured",
              flush=True)
        for i, name in enumerate(todo, 1):
            measure_group(source, name, store[name], writer, problems)
            if i % 25 == 0 or i == len(todo):
                left = (time.perf_counter() - t0) / i * (len(todo) - i)
                print(f"  {source}: {i}/{len(todo)}, {left / 60:.1f} min left", flush=True)


def video_files(paths):
    files = []
    for p in paths:
        if p.is_dir():
            files += sorted(f for f in p.iterdir()
                            if f.is_file() and f.suffix.lower() in VIDEO_SUFFIXES)
        elif p.is_file():
            files.append(p)
    return files


def measure_videos(paths, writer, limit, problems):
    files = [f for f in video_files(paths)[:limit] if (OWN, f.name) not in writer.done]
    if not files:
        return
    # Imported here so a stores-only run does not need MediaPipe.
    from app.live_vitals.capture.session import crops_from_video
    from app.live_vitals.preprocess.face_box import make_landmarker
    landmarker = make_landmarker()
    print(f"{OWN}: {len(files)} videos through the app's crops_from_video", flush=True)
    for f in files:
        try:
            clip, fps, quality = crops_from_video(f, landmarker)
        except Exception as exc:  # one unreadable file must not cost the whole run
            problems.append(f"{OWN}/{f.name}: {type(exc).__name__}: {exc}")
            continue
        windows, summary = crop_motion(clip, fps)
        if summary is None:
            problems.append(f"{OWN}/{f.name}: shorter than one window; skipped")
            continue
        writer.add(OWN, f.name, windows,
                   dict(source=OWN, stratum=OWN, recording=f.name, usable=True,
                        fps=float(fps), verdict=quality.verdict, **summary))
        print(f"  {f.name}: {quality.verdict}, {fps:.1f} fps, amp {summary['amp_median']:.2f} px",
              flush=True)


# ----------------------------------------------------------------- summary --

def refused(table, threshold, rule):
    hit = table["amp_median"] >= threshold
    if rule == "B":
        hit &= table["clear_frac"] >= RULE_B_CLEAR
    return hit


def refused_c(table, proposed):
    """Rule C: A or B, each at its proposed threshold."""
    hit = pd.Series(False, index=table.index)
    for rule in ("A", "B"):
        if proposed[rule] is not None:
            hit |= refused(table, proposed[rule], rule)
    return hit


def smallest_threshold(groups, rule, budget):
    """The smallest grid T at which no group loses more than budget, and the group that
    sets it (the one needing the highest T; None when no group needs more than the grid's
    lowest step). (None, None) if the grid is not enough."""
    need = {}
    for name, g in groups:
        ok = [t for t in GRID if refused(g, t, rule).mean() <= budget]
        need[name] = ok[0] if ok else None
    if not need or None in need.values():
        return None, None
    binding = max(need, key=need.get)
    return need[binding], (binding if need[binding] > GRID[0] else None)


def summarise(rec, budget):
    rec = rec.copy()
    rec["usable"] = rec["usable"].astype(str).str.lower().isin(["true", "1"])
    usable = rec[rec["usable"]]
    still = usable[~usable["source"].isin(MOTION_STORES | {OWN})]
    resting = still[~still["stratum"].isin(TALKING)]

    t_a, set_a = smallest_threshold(still.groupby("stratum"), "A", budget)
    t_b, set_b = smallest_threshold(still.groupby("stratum"), "B", budget)
    floor, set_floor = smallest_threshold(resting.groupby("stratum"), "A", budget)
    proposed = dict(A=t_a, A_set_by=set_a, B_budget=t_b, B_budget_set_by=set_b,
                    resting_floor=floor, resting_floor_set_by=set_floor,
                    B=(max(t_b, floor) if t_b is not None and floor is not None else None))

    strata = {}
    for name, g in usable.groupby("stratum"):
        q = g["amp_median"].quantile([0.5, 0.9, 0.99]).to_numpy()
        strata[name] = dict(
            in_rule=name in set(still["stratum"]), recordings=int(len(g)),
            amp_p50=float(q[0]), amp_p90=float(q[1]), amp_p99=float(q[2]),
            amp_max=float(g["amp_median"].max()),
            clear_frac_p50=float(g["clear_frac"].median()),
            periodic_share=float((g["clear_frac"] >= RULE_B_CLEAR).mean()),
            refused_A={str(t): float(refused(g, t, "A").mean()) for t in GRID},
            refused_B={str(t): float(refused(g, t, "B").mean()) for t in GRID},
            refused_C=float(refused_c(g, proposed).mean()))
    return dict(strata=strata, proposed=proposed, unusable_excluded=int((~rec["usable"]).sum()))


def fmt(x, pct=False, digits=2):
    if x is None or not np.isfinite(x):
        return "-"
    return f"{100 * x:.0f}%" if pct else f"{x:.{digits}f}"


def print_summary(s, rec, budget, out_dir):
    pr = s["proposed"]
    print("\nMotion-gate calibration - ECG-Fitness never read")
    print("statistic: median in-band crop displacement per recording (px). Rule A refuses at "
          f">= T; rule B also needs >= {RULE_B_CLEAR:.0%} of windows with a clear rhythm "
          f"('periodic'); rule C is A or B.")
    for problem in s["problems"]:
        print(f"  ! {problem}")
    if s["unusable_excluded"]:
        print(f"  {s['unusable_excluded']} recordings marked unusable in their store are left out")
    for rule in ("A", "B"):
        print(f"\nRule {rule}: share of recordings refused at threshold T (px)")
        extra = f" {'periodic':>8}" if rule == "B" else ""
        print(f"{'stratum':26} {'rec':>5} {'p50':>5} {'p90':>5} {'p99':>5} {'max':>5}{extra} |"
              + "".join(f"{t:>6}" for t in SHOWN))
        for name, r in s["strata"].items():
            label = name if r["in_rule"] else f"{name} (not in rule)"
            extra = f" {fmt(r['periodic_share'], pct=True):>8}" if rule == "B" else ""
            print(f"{label:26.26} {r['recordings']:>5} {fmt(r['amp_p50']):>5} "
                  f"{fmt(r['amp_p90']):>5} {fmt(r['amp_p99']):>5} {fmt(r['amp_max']):>5}{extra} |"
                  + "".join(f"{fmt(r['refused_' + rule][str(t)], pct=True):>6}" for t in SHOWN))

    print(f"\nproposed thresholds (smallest 0.1 px step at which no still stratum loses more "
          f"than {budget:.0%}):")

    def by(name):
        return f"set by {name}" if name else "no stratum needs more than the lowest step"

    if pr["A"] is not None:
        print(f"  rule A: {pr['A']:.1f} px, {by(pr['A_set_by'])}")
    else:
        print("  rule A: none on the grid")
    if pr["B"] is not None:
        print(f"  rule B: {pr['B']:.1f} px - the larger of {pr['B_budget']:.1f} px "
              f"({by(pr['B_budget_set_by'])}) and the resting range {pr['resting_floor']:.1f} px "
              f"({by(pr['resting_floor_set_by'])})")
    else:
        print("  rule B: none on the grid")
    print("  rule C (A or B at those thresholds), share refused: "
          + ", ".join(f"{name} {fmt(r['refused_C'], pct=True)}"
                      for name, r in s["strata"].items()))

    still = rec[~rec["source"].isin(MOTION_STORES | {OWN})]
    print("\nmost-moving still recordings (look at these before trusting the threshold):")
    for _, r in still.sort_values("amp_median", ascending=False).head(12).iterrows():
        print(f"  {r['stratum']:14} {r['recording']:40.40} amp {r['amp_median']:.2f} px, "
              f"p90 {r['amp_p90']:.2f}, clear {r['clear_frac']:.2f}, usable {r['usable']}")
    own = rec[rec["source"] == OWN]
    if len(own):
        print("\nyour own captures:")
        for _, r in own.sort_values("recording").iterrows():
            row = pd.DataFrame([r])
            flags = [f"refused by {rule}" for rule in ("A", "B")
                     if pr[rule] is not None and refused(row, pr[rule], rule).iloc[0]]
            print(f"  {r['recording']:32.32} {r['verdict']:6} {r['fps']:5.1f} fps, "
                  f"amp {r['amp_median']:.2f} px, clear {r['clear_frac']:.2f}"
                  f"{'  <- ' + ', '.join(flags) if flags else ''}")
    print(f"\nwrote calibration_recordings.csv, calibration_windows.csv, "
          f"summary_calibration.json to {out_dir}")


# --------------------------------------------------------------------- main --

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stores", type=Path, nargs="+", default=None,
                   help="HDF5 stores (default: every *_rc_none.h5 in the Phase-3 folder)")
    p.add_argument("--videos", type=Path, nargs="+", default=[DEFAULT_VIDEO_DIR],
                   help="video files or folders (a folder's own files, not its subfolders)")
    p.add_argument("--no-videos", action="store_true", help="stores only")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--limit", type=int, default=None, help="first N recordings per source")
    p.add_argument("--budget", type=float, default=0.01,
                   help="largest share of any still stratum the proposed threshold may refuse")
    p.add_argument("--fresh", action="store_true", help="discard earlier results and start over")
    args = p.parse_args()

    stores = args.stores or sorted(DEFAULT_STORE_DIR.glob("*_rc_none.h5"))
    if not stores:
        raise SystemExit(f"ERROR: no stores found in {DEFAULT_STORE_DIR}")
    for path in stores:
        if not path.exists():
            raise SystemExit(f"ERROR: store not found: {path}")
    refuse_benchmarks(stores)
    writer = Writer(args.out_dir, args.fresh)
    problems = []
    t0 = time.perf_counter()
    for path in stores:
        measure_store(path, writer, args.limit, problems)
    if not args.no_videos:
        measure_videos(args.videos, writer, args.limit, problems)
    print(f"measured in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    if not writer.rec_path.exists():
        for problem in problems:
            print(f"  ! {problem}")
        raise SystemExit("ERROR: nothing was measured")

    rec = pd.read_csv(writer.rec_path, dtype={"source": str, "stratum": str,
                                              "recording": str, "verdict": str})
    rec["verdict"] = rec["verdict"].fillna("")
    summary = summarise(rec, args.budget)
    summary.update(script="calibrate_motion_gate", budget=args.budget,
                   rule_b_clear=RULE_B_CLEAR, stores=[str(p) for p in stores],
                   videos=[] if args.no_videos else [str(p) for p in args.videos],
                   limit=args.limit, problems=problems)
    with open(args.out_dir / "summary_calibration.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary, rec, args.budget, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())