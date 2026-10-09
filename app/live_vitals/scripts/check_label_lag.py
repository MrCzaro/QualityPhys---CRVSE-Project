"""Measures the time offset between the video pulse and the label in each training corpus.

PhysNet v2 trains on a loss with a time-domain term (negative Pearson against the stored
`bvp`), so a label that leads or trails the video by a different amount in each corpus
gives the model conflicting targets. NB_P3_27's QC found VitalVideos' video trailing its
PPG label by a median 0.10 s (IQR 0.07-0.13); the other corpora were never measured.
This script measures the four training corpora the same way, on a seeded sample of each
one's training recordings from Data/phase3_split.csv.

Per recording: POS from the stored 72x72 crops (the app's own rgb_trace and pos) and the
stored bvp, both band-passed to the app's HR band, then Pearson r at every lag within
--max-lag. A lag > 0 means the video trails the label: the video at time t matches the
label at t - lag.

A pulse correlates again one beat later, and with its sign flipped half a beat later,
so one recording's largest |r| is not its offset. Per corpus, the
recordings' correlograms are averaged on a common lag grid: heart rates differ between
recordings, so those echoes fall at different lags and wash out, while an offset the
recordings share adds up. The average's extremum gives the corpus's sign and anchors
its offset; each recording's own offset is the extremum of that sign within half a beat
of the anchor, and the corpus's offset is the median of those. A recording whose best
alignment of that sign lies elsewhere by a clear margin (r higher by ELSEWHERE_MARGIN)
is counted, since it may carry an offset of its own.

Decision rule, fixed before the run: labels are shifted only if
the training corpora's offsets differ by more than 2 frames at 30 fps (0.067 s); then
every corpus is shifted by its own offset, in every training run.

Writes label_lag_recordings.csv and summary_label_lag.json to --out-dir. Nothing written
contains a frame or a demographic.

Usage:
    python -m app.live_vitals.scripts.check_label_lag
        [--per-corpus N] [--max-lag S] [--seed N] [--datasets NAME ...]
        [--split FILE] [--store-dir DIR] [--out-dir DIR]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals.config import HR_HIGH_HZ, HR_LOW_HZ
from app.live_vitals.signal.spectral import bandpass, pos, rgb_trace

DEFAULT_SPLIT = _REPO_ROOT / "Data" / "phase3_split.csv"
DEFAULT_STORE_DIR = Path(r"D:\QualityPhys\phase 3 datasets")
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "label_lag"
TRAINING = ["MCD", "DLCN", "UBFC-rPPG", "VitalVideos-WW"]
GRID_STEP = 0.005                 # s, the common lag grid the correlograms are averaged on
DECISION_FRAMES, DECISION_FPS = 2, 30.0
MIN_SECONDS = 10.0                # shorter recordings are skipped
WEAK = 0.05                       # a corpus whose averaged peak |r| is below this is unreadable
ELSEWHERE_MARGIN = 0.1            # best r away from the corpus offset exceeds r near it by this
VV_QC = "NB_P3_27's QC: video trails the label by a median 0.10 s (IQR 0.07-0.13)"


# ------------------------------------------------------------------ signal --

def lag_correlogram(video, label, max_lag_frames):
    """Pearson r between video[t] and label[t - k] for k = -K..K (k > 0: video trails)."""
    n = len(video)
    ks = np.arange(-max_lag_frames, max_lag_frames + 1)
    r = np.full(len(ks), np.nan)
    for i, k in enumerate(ks):
        a, b = (video[k:], label[:n - k]) if k >= 0 else (video[:n + k], label[-k:])
        a = a - a.mean()
        b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        if den > 0:
            r[i] = (a * b).sum() / den
    return ks, r


def spectral_hr(x, fps):
    """The label's dominant rate over the whole recording (bpm), for the half-beat window."""
    p = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / fps)
    band = (f >= HR_LOW_HZ) & (f <= HR_HIGH_HZ)
    return float(f[band][np.argmax(p[band])] * 60.0)


def measure(group, max_lag):
    """One recording's correlogram on the common grid, or the reason it was skipped."""
    fps = float(group.attrs["fps"])
    label = np.asarray(group["bvp"][()], dtype=np.float64)
    frames = group["frames"]
    n = min(len(frames), len(label))
    label = label[:n]
    if n < MIN_SECONDS * fps:
        return None, "shorter than 10 s"
    if not np.all(np.isfinite(label)) or label.std() < 1e-9:
        return None, "label not finite or flat"
    video = pos(rgb_trace(frames[:n]), fps)
    label = bandpass(label - label.mean(), fps)
    if video.std() < 1e-12:
        return None, "no POS signal"
    ks, r = lag_correlogram(video, label, int(np.ceil(max_lag * fps)) + 2)
    if not np.all(np.isfinite(r)):
        return None, "correlation undefined at some lag"
    grid = np.arange(-max_lag, max_lag + GRID_STEP / 2, GRID_STEP)
    curve = CubicSpline(ks / fps, r)(grid)
    return dict(fps=fps, n_frames=n, hr_label=spectral_hr(label, fps), curve=curve), None


# ------------------------------------------------------------------ corpus --

def corpus_offset(curves, grid):
    """Sign and offset from the averaged correlogram's extremum."""
    mean = np.nanmean(np.vstack(curves), axis=0)
    i = int(np.nanargmax(np.abs(mean)))
    return float(np.sign(mean[i])), float(grid[i]), float(abs(mean[i]))


def own_offset(curve, grid, sign, centre, hr_bpm):
    """The extremum of the corpus's sign within half a beat of the corpus offset, and
    the extremum of that sign anywhere in the searched range."""
    half_beat = 30.0 / hr_bpm
    near = np.abs(grid - centre) <= half_beat
    signed = sign * curve
    i = np.flatnonzero(near)[np.argmax(signed[near])]
    j = int(np.argmax(signed))
    return float(grid[i]), float(signed[i]), float(grid[j]), float(signed[j])


def sample(split_path, datasets, per_corpus, seed):
    table = pd.read_csv(split_path, dtype=str, keep_default_na=False)
    rows = table[table["dataset"].isin(datasets) & (table["indexed"] == "True")
                 & (table["split"] == "train")]
    rng = np.random.default_rng(seed)
    picked = []
    for name in datasets:
        pool = rows[rows["dataset"] == name]
        if pool.empty:
            raise SystemExit(f"ERROR: no indexed training recordings of {name} in {split_path}")
        take = pool.iloc[np.sort(rng.choice(len(pool), min(per_corpus, len(pool)),
                                            replace=False))]
        picked.append(take)
    return pd.concat(picked, ignore_index=True)


def run(picked, store_dir, max_lag):
    grid = np.arange(-max_lag, max_lag + GRID_STEP / 2, GRID_STEP)
    measured, skipped = [], []
    for name, recs in picked.groupby("dataset", sort=False):
        t0 = time.perf_counter()
        for store, group_names in recs.groupby("store")["group"]:
            path = store_dir / store
            if not path.exists():
                raise SystemExit(f"ERROR: store not found: {path}")
            with h5py.File(path, "r") as f:
                for g in group_names:
                    result, why = measure(f[g], max_lag)
                    if result is None:
                        skipped.append(f"{name}/{g}: {why}")
                    else:
                        measured.append(dict(dataset=name, store=store, group=g, **result))
        print(f"  {name}: {len(recs)} recordings read in {time.perf_counter() - t0:.0f} s",
              flush=True)
    return grid, measured, skipped


def summarise(grid, measured, datasets):
    rows, corpora = [], {}
    for name in datasets:
        recs = [m for m in measured if m["dataset"] == name]
        if not recs:
            corpora[name] = dict(n=0)
            continue
        sign, anchor, peak = corpus_offset([m["curve"] for m in recs], grid)
        own, elsewhere = [], 0
        for m in recs:
            lag_rec, r_rec, best_lag, best_r = own_offset(m["curve"], grid, sign, anchor,
                                                          m["hr_label"])
            own.append(lag_rec)
            elsewhere += int(best_r - r_rec > ELSEWHERE_MARGIN)
            rows.append(dict(dataset=name, store=m["store"], group=m["group"], fps=m["fps"],
                             n_frames=m["n_frames"], hr_label=round(m["hr_label"], 1),
                             lag_s=round(lag_rec, 3), r=round(r_rec, 3),
                             best_lag_s=round(best_lag, 3), best_r=round(best_r, 3)))
        own = np.array(own)
        lag = float(np.median(own))
        rs = np.array([r["r"] for r in rows if r["dataset"] == name])
        corpora[name] = dict(
            n=len(recs), sign="+" if sign > 0 else "-", lag_s=round(lag, 3),
            lag_frames_at_30fps=round(lag * DECISION_FPS, 2),
            lag_iqr_s=[round(float(np.percentile(own, 25)), 3),
                       round(float(np.percentile(own, 75)), 3)],
            within_one_frame=round(float(np.mean(np.abs(own - lag) <= 1.0 / DECISION_FPS)), 3),
            recording_r_median=round(float(np.median(rs)), 3),
            offset_elsewhere=elsewhere,
            averaged_lag_s=round(anchor, 3), averaged_peak_r=round(peak, 3),
            readable=bool(peak >= WEAK))
    return pd.DataFrame(rows), corpora


def decide(corpora, datasets):
    readable = {k: v for k, v in corpora.items() if k in datasets and v.get("readable")}
    unreadable = [k for k in datasets if k not in readable]
    if len(readable) < 2:
        return dict(verdict="undecided: fewer than two readable corpora",
                    unreadable=unreadable)
    lags = {k: v["lag_s"] for k, v in readable.items()}
    spread = max(lags.values()) - min(lags.values())
    limit = DECISION_FRAMES / DECISION_FPS
    verdict = ("shift each corpus's labels by its own offset" if spread > limit
               else "no label shift: the corpora agree within 2 frames")
    if unreadable:
        verdict += f" (unreadable, so not judged: {', '.join(unreadable)})"
    return dict(spread_s=round(spread, 3), spread_frames_at_30fps=round(spread * DECISION_FPS, 2),
                limit_s=round(limit, 3), verdict=verdict, unreadable=unreadable)


def print_summary(corpora, decision, skipped, out_dir):
    print("\nLabel lag per corpus (lag > 0: the video trails the label; median over recordings)")
    print(f"{'corpus':16} {'n':>3} {'sign':>4} {'lag s':>7} {'frames':>6} {'IQR s':>15} "
          f"{'within 1 fr':>11} {'median r':>8} {'elsewhere':>9} | {'averaged: lag':>13} {'peak r':>6}")
    for name, c in corpora.items():
        if not c.get("n"):
            print(f"{name:16} {0:>3}  no recordings measured")
            continue
        flag = "" if c["readable"] else "  <- too weak to read"
        iqr = f"{c['lag_iqr_s'][0]:+.3f}..{c['lag_iqr_s'][1]:+.3f}"
        print(f"{name:16} {c['n']:>3} {c['sign']:>4} {c['lag_s']:>+7.3f} "
              f"{c['lag_frames_at_30fps']:>+6.2f} {iqr:>15} {c['within_one_frame']:>11.0%} "
              f"{c['recording_r_median']:>8.3f} {c['offset_elsewhere']:>9} | "
              f"{c['averaged_lag_s']:>+13.3f} {c['averaged_peak_r']:>6.3f}{flag}")
    vv = corpora.get("VitalVideos-WW", {})
    if vv.get("n"):
        print(f"\ncontrol: {VV_QC}; this run {vv['lag_s']:+.3f} s "
              f"(IQR {vv['lag_iqr_s'][0]:+.3f}..{vv['lag_iqr_s'][1]:+.3f})")
    for s in skipped:
        print(f"  ! skipped {s}")
    if "spread_s" in decision:
        print(f"\nspread between corpora: {decision['spread_s']:.3f} s "
              f"({decision['spread_frames_at_30fps']:.2f} frames at 30 fps); "
              f"rule: shift only above {decision['limit_s']:.3f} s")
    print(f"decision: {decision['verdict']}")
    print(f"\nwrote label_lag_recordings.csv and summary_label_lag.json to {out_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--per-corpus", type=int, default=60,
                   help="training recordings sampled per corpus (all, if fewer)")
    p.add_argument("--max-lag", type=float, default=1.0, help="largest offset searched, s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--datasets", nargs="+", default=TRAINING)
    p.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--store-dir", type=Path, default=DEFAULT_STORE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    if not args.split.exists():
        raise SystemExit(f"ERROR: {args.split} not found - run freeze_phase3_split.py first")

    picked = sample(args.split, args.datasets, args.per_corpus, args.seed)
    print(f"Measuring {len(picked)} training recordings "
          f"({', '.join(f'{k} {v}' for k, v in picked['dataset'].value_counts(sort=False).items())}),"
          f" offsets up to +/-{args.max_lag:g} s", flush=True)
    t0 = time.perf_counter()
    grid, measured, skipped = run(picked, args.store_dir, args.max_lag)
    table, corpora = summarise(grid, measured, args.datasets)
    decision = decide(corpora, [d for d in args.datasets if d in TRAINING])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "label_lag_recordings.csv", index=False)
    with open(args.out_dir / "summary_label_lag.json", "w", encoding="utf-8") as f:
        json.dump(dict(script="check_label_lag", per_corpus=args.per_corpus,
                       max_lag_s=args.max_lag, seed=args.seed, grid_step_s=GRID_STEP,
                       convention="lag > 0: the video trails the label",
                       corpora=corpora, decision=decision, skipped=skipped,
                       minutes=round((time.perf_counter() - t0) / 60, 1)), f, indent=2)
    print_summary(corpora, decision, skipped, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())