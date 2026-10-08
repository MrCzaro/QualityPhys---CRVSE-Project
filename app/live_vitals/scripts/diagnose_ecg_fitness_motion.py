"""Exploratory: do the ECG-Fitness readings that miss the ECG follow the head's rhythm?

Decided after the refusal benchmark (NB_P3_29 Part 0, run 2026-10-05), so it is not part
of the protocol and nothing here is a pre-registered result. The benchmark found wrong
readings during exercise that the app's confidence, status and POS cross-check do not
flag. The hypothesis: the estimators lock onto a mechanical rhythm - pedalling, striding,
rowing - instead of the pulse. This tests it without running a model, from the readings
the benchmark already wrote.

Two motion sources per recording, each band-passed to the cardiac band and read window by
window with the app's own readout (`hr_from_bvp`), so a motion rhythm comes out in the
same band, resolution and units as a heart rate:
- box: the dataset authors' per-frame face box - centre x, centre y and size - at full
  resolution, read from the copy that the tools script ecg_fitness_subset.py keeps. It is
  independent of the colour signal.
- crop: the frame-to-frame shift of the stored 72x72 crop, by cross-correlation. The app
  has the crop too, so this is what a gate could use.
In each window the rhythm is that of the channel the readout finds most periodic, and its
periodicity is the readout's spectral concentration - the quantity the app calls
confidence when the signal is a pulse.

A reading sits on the rhythm when it is within --tolerance bpm of it, or of twice or half
of it. The ECG reference is the control: if the true heart rate sits on the rhythm as
often as the reading does, the coincidence explains nothing.

Writes motion_windows.csv and motion_recordings.csv (git-ignored) and summary_motion.json
beside the benchmark's outputs. Nothing written here contains a frame.

Usage:
    python -m app.live_vitals.scripts.diagnose_ecg_fitness_motion
        [--store PATH] [--bbox-root DIR] [--eval-dir DIR] [--tolerance BPM]
        [--clear-rhythm C] [--limit N]
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
from app.live_vitals.signal.hr import hr_from_bvp
from app.live_vitals.signal.spectral import bandpass

DEFAULT_STORE = Path(r"D:\QualityPhys\phase 3 benchmarks\ecg_fitness_phase3_rc_none.h5")
DEFAULT_BBOX_ROOT = Path(r"D:\QualityPhys\demo_data\ecg_fitness_subset\bbox")
DEFAULT_EVAL_DIR = _REPO_ROOT / "Data" / "ecg_fitness_eval"
ESTIMATORS = {"physnet": "hr_physnet_v2", "pos": "hr_spectral"}
SOURCES = ("box", "crop")
PAIRS = {"01": "01/03", "03": "01/03", "02": "02/04", "04": "02/04",
         "05": "05/06", "06": "05/06"}
WRONG_BPM = 10.0  # Part 0's confident-wrong margin, applied here to every window
SHARED = ["activity", "ref_hr", "referenced", "off_span", "primary"]
PER_ESTIMATOR = ["pred_hr", "pred_kept", "scored", "confident_wrong", "recording_status"]
SHORT_STATUS = {"ok": "ok", "degraded_capture": "degraded", "unstable": "unstable"}
NAN = float("nan")


# ------------------------------------------------------------------- inputs --

def read_windows(eval_dir):
    """Both estimators' per-window outputs on one row per window, checked to agree."""
    tables = []
    for short, name in ESTIMATORS.items():
        path = eval_dir / f"windows_{name}.csv"
        if not path.exists():
            raise SystemExit(f"ERROR: {path} not found; run check_ecg_fitness for {name} first")
        table = pd.read_csv(path, dtype={"recording": str, "activity": str})
        tables.append(table[["recording", "start"] + SHARED + PER_ESTIMATOR].rename(
            columns={c: f"{short}_{c}" for c in PER_ESTIMATOR}))
    both = tables[0].merge(tables[1], on=["recording", "start"], how="outer",
                           suffixes=("", "_other"), validate="one_to_one", indicator=True)
    if (both["_merge"] != "both").any():
        raise SystemExit("ERROR: the two window files cover different windows")
    for col in SHARED:
        a, b = both[col], both[f"{col}_other"]
        same = (np.isclose(a, b, equal_nan=True) if col == "ref_hr"
                else (a == b).to_numpy())
        if not np.all(same):
            raise SystemExit(f"ERROR: the two window files disagree on {col}")
    return both.drop(columns=[f"{c}_other" for c in SHARED] + ["_merge"])


def read_recordings(eval_dir):
    """Each estimator's reported status and value per recording."""
    out = None
    for short, name in ESTIMATORS.items():
        path = eval_dir / f"recordings_{name}.csv"
        if not path.exists():
            raise SystemExit(f"ERROR: {path} not found; run check_ecg_fitness for {name} first")
        table = pd.read_csv(path, dtype={"recording": str})
        table = table[["recording", "primary", "ref_hr_median", "status", "value"]].rename(
            columns={"status": f"{short}_status", "value": f"{short}_value"})
        out = table if out is None else out.merge(
            table.drop(columns=["primary", "ref_hr_median"]), on="recording",
            validate="one_to_one")
    return out.set_index("recording")


# ------------------------------------------------------------ motion tracks --

def box_tracks(path, n_frames):
    """Centre x, centre y and size of the authors' face box per frame, full resolution.

    Columns 1-4 are x, y, w, h - the reading NB_P3_28 used. Rows without a face (w or h
    of 0) are interpolated over. Returns the tracks and the median face size in pixels.
    """
    a = np.loadtxt(path, dtype=np.float64, ndmin=2)
    if a.shape[0] != n_frames or a.shape[1] < 5:
        raise ValueError(f"{a.shape[0]} rows x {a.shape[1]} columns, expected {n_frames} x 5")
    x, y, w, h = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    found = (w > 0) & (h > 0)
    if found.sum() < n_frames // 2:
        raise ValueError(f"a face in only {int(found.sum())} of {n_frames} rows")
    t = np.arange(n_frames)

    def fill(v):
        return np.interp(t, t[found], v[found])

    size = np.sqrt(np.clip(w, 0, None) * np.clip(h, 0, None))
    tracks = dict(box_x=fill(x + w / 2), box_y=fill(y + h / 2), box_size=fill(size))
    return tracks, float(np.median(size[found]))


def crop_shifts(frames):
    """Frame-to-frame shift of the crop's content (crop px), by cross-correlation.

    Plain cross-correlation of the mean-removed, Hann-weighted grey crop, peak refined to
    sub-pixel by a parabola on each axis. The Hann weight favours the centre, where the
    face is, over the padded border. OpenCV's phaseCorrelate was tried first: its
    whitening hands the decision to sensor noise wherever the crop has little texture,
    and on synthetic crops it returned shifts of thousands of pixels.
    """
    gray = np.asarray(frames, dtype=np.float64).mean(axis=3)
    n, h, w = gray.shape
    weight = np.outer(np.hanning(h), np.hanning(w))
    spectra = np.fft.rfft2((gray - gray.mean(axis=(1, 2), keepdims=True)) * weight)
    shift = np.zeros((n, 2))
    for lo in range(1, n, 256):  # in blocks, to bound memory
        hi = min(n, lo + 256)
        corr = np.fft.irfft2(spectra[lo:hi] * np.conj(spectra[lo - 1:hi - 1]), s=(h, w))
        corr = np.fft.fftshift(corr, axes=(1, 2))
        peak = corr.reshape(len(corr), -1).argmax(axis=1)
        iy, ix = np.unravel_index(peak, (h, w))
        iy, ix = np.clip(iy, 1, h - 2), np.clip(ix, 1, w - 2)
        k = np.arange(len(corr))
        shift[lo:hi, 0] = ix - w // 2 + _vertex(corr[k, iy, ix - 1], corr[k, iy, ix],
                                                 corr[k, iy, ix + 1])
        shift[lo:hi, 1] = iy - h // 2 + _vertex(corr[k, iy - 1, ix], corr[k, iy, ix],
                                                 corr[k, iy + 1, ix])
    return shift


def _vertex(left, centre, right):
    """Sub-sample offset of a parabola's vertex through three points; 0 if not a peak."""
    curvature = left - 2.0 * centre + right
    safe = np.where(np.abs(curvature) > 1e-12, curvature, 1.0)
    offset = np.where(np.abs(curvature) > 1e-12, 0.5 * (left - right) / safe, 0.0)
    return np.where(np.abs(offset) <= 0.5, offset, 0.0)


# ----------------------------------------------------------------- readout --

def window_rhythms(read_tracks, amp_tracks, amp_scale, fps, starts, source):
    """Per window: the rhythm (bpm), periodicity and channel of the most periodic channel
    in `read_tracks`, and the in-band rms displacement of `amp_tracks` times `amp_scale`.

    Every channel is band-passed over the whole recording with the app's cardiac band-pass,
    as POS is, then read on each analysis window with the app's own readout.
    """
    readable = {k: bandpass(v - v.mean(), fps) for k, v in read_tracks.items()}
    moving = [bandpass(v - v.mean(), fps) for v in amp_tracks]
    rows = []
    for s in starts:
        best_hr, best_conf, best_channel = NAN, -1.0, ""
        for channel, signal in readable.items():
            reading = hr_from_bvp(signal[s:s + config.CLIP_LEN], fps)
            if np.isfinite(reading["hr_bpm"]) and reading["confidence"] > best_conf:
                best_hr, best_conf, best_channel = (reading["hr_bpm"], reading["confidence"],
                                                    channel)
        amp = np.sqrt(sum(np.mean(v[s:s + config.CLIP_LEN] ** 2) for v in moving)) * amp_scale
        rows.append({"start": s, f"{source}_rhythm": best_hr,
                     f"{source}_periodicity": best_conf if best_channel else NAN,
                     f"{source}_channel": best_channel, f"{source}_amp": amp})
    return rows


def _text(value):
    """An HDF5 string attribute as str, whichever way h5py hands it back."""
    return value.decode() if isinstance(value, bytes) else str(value)


def recording_motion(group, bbox_root, problems):
    """Per-window rhythm, periodicity and amplitude from both sources for one recording."""
    attrs = group.attrs
    name = group.name.strip("/")
    if _text(attrs.get("role", "")) != "benchmark":
        raise SystemExit(f"ERROR: {name} is not a benchmark group")
    fps = float(attrs["fps"])
    frames = group["frames"][()]
    starts = [int(s) for s in group["ref_window_start"][()]]

    box_path = (bbox_root / _text(attrs["subject_id"]) / _text(attrs["activity"])
                / f"{_text(attrs['camera'])}.face")
    try:
        tracks, face_px = box_tracks(box_path, len(frames))
        box = window_rhythms(tracks, [tracks["box_x"], tracks["box_y"]], 100.0 / face_px,
                             fps, starts, "box")
    except (OSError, ValueError) as exc:
        problems.append(f"{name}: no box track ({exc})")
        box = [{"start": s, "box_rhythm": NAN, "box_periodicity": NAN, "box_channel": "",
                "box_amp": NAN} for s in starts]

    # The crop is read on its frame-to-frame shifts rather than their running sum: the
    # registration error is independent from frame to frame, so the sum would turn it
    # into a random walk whose power piles up at the bottom of the band.
    shift = crop_shifts(frames)
    position = np.cumsum(shift, axis=0)
    crop = window_rhythms(dict(crop_x=shift[:, 0], crop_y=shift[:, 1]),
                          [position[:, 0], position[:, 1]], 1.0, fps, starts, "crop")
    return [{"recording": name, **b, **{k: v for k, v in c.items() if k != "start"}}
            for b, c in zip(box, crop)]


def on_rhythm(value, rhythm, tol):
    """1 on the rhythm, 2 on twice or half of it, 0 on neither; NaN where either is missing."""
    v = np.asarray(value, dtype=float)
    r = np.asarray(rhythm, dtype=float)
    with np.errstate(invalid="ignore"):
        code = np.where(np.abs(v - r) <= tol, 1.0,
                        np.where((np.abs(v - 2 * r) <= tol) | (np.abs(v - r / 2) <= tol),
                                 2.0, 0.0))
    return np.where(np.isfinite(v) & np.isfinite(r), code, NAN)


# --------------------------------------------------------------- summaries --

def rate(code, which):
    """Fraction of the finite codes that are `which` (1, 2, or "any" for either)."""
    code = pd.Series(code, dtype=float).dropna()
    if not len(code):
        return NAN
    return float((code >= 1).mean() if which == "any" else (code == which).mean())


def clear_fraction(periodicity, clear):
    """Fraction of windows with a motion reading whose periodicity reaches `clear`."""
    p = pd.Series(periodicity, dtype=float).dropna()
    return float((p >= clear).mean()) if len(p) else NAN


def motion_by_activity(w, clear):
    rows = {}
    for act, g in w.groupby("activity"):
        clear_box = g["box_periodicity"] >= clear
        rows[act] = dict(
            windows=int(len(g)),
            box_amp_pct_face=float(g["box_amp"].median()),
            box_clear=clear_fraction(g["box_periodicity"], clear),
            box_rhythm_when_clear=(float(g.loc[clear_box, "box_rhythm"].median())
                                   if clear_box.any() else NAN),
            crop_amp_px=float(g["crop_amp"].median()),
            crop_clear=clear_fraction(g["crop_periodicity"], clear))
    return rows


def lock_table(w, short, source, population, wrong):
    """Wrong against right windows: how often the reading, and the ECG, sit on the rhythm."""
    rows = {}
    for pair in ("01/03", "02/04", "05/06"):
        in_pair = population & (w["pair"] == pair)
        bad, good = w[in_pair & wrong], w[in_pair & ~wrong]
        bad_code = bad[f"{short}_{source}_code"]
        kept = bad[f"{short}_pred_kept"].astype(bool)
        rows[pair] = dict(
            wrong=int(len(bad)),
            wrong_kept_when_on=(float(kept[bad_code >= 1].mean()) if (bad_code >= 1).any()
                                else NAN),
            wrong_kept_when_off=(float(kept[bad_code == 0].mean()) if (bad_code == 0).any()
                                 else NAN),
            wrong_reading_on=rate(bad_code, 1),
            wrong_reading_harmonic=rate(bad_code, 2),
            wrong_ecg_on=rate(bad[f"ecg_{source}_code"], 1),
            wrong_ecg_harmonic=rate(bad[f"ecg_{source}_code"], 2),
            right=int(len(good)),
            right_reading_on_any=rate(good[f"{short}_{source}_code"], "any"))
    return rows


def crop_sees_box(w, clear, tol):
    rows = {}
    for act, g in w.groupby("activity"):
        g = g[(g["box_periodicity"] >= clear) & g["crop_rhythm"].notna()]
        code = on_rhythm(g["crop_rhythm"], g["box_rhythm"], tol)
        rows[act] = dict(windows=int(len(g)), same=rate(code, 1), harmonic=rate(code, 2),
                         median_abs_diff=(float((g["crop_rhythm"] - g["box_rhythm"]).abs()
                                                .median()) if len(g) else NAN))
    return rows


def recording_table(w, recordings, clear):
    rows = []
    for name, g in w.groupby("recording"):
        rec = recordings.loc[name]
        row = dict(recording=name, activity=g["activity"].iloc[0], primary=bool(rec["primary"]),
                   ecg=float(rec["ref_hr_median"]),
                   physnet_status=rec["physnet_status"], physnet=float(rec["physnet_value"]),
                   pos_status=rec["pos_status"], pos=float(rec["pos_value"]))
        for source in SOURCES:
            clear_rows = g[g[f"{source}_periodicity"] >= clear]
            row[f"{source}_clear_windows"] = int(len(clear_rows))
            row[f"{source}_rhythm"] = (float(clear_rows[f"{source}_rhythm"].median())
                                       if len(clear_rows) else NAN)
            row[f"{source}_periodicity"] = float(g[f"{source}_periodicity"].median())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- main --

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--store", type=Path, default=DEFAULT_STORE)
    p.add_argument("--bbox-root", type=Path, default=DEFAULT_BBOX_ROOT,
                   help="the authors' bbox folder (SS/AA/<camera>.face inside)")
    p.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_DIR,
                   help="where check_ecg_fitness wrote its CSVs; outputs go here too")
    p.add_argument("--tolerance", type=float, default=5.0,
                   help="bpm within which a reading counts as sitting on the rhythm")
    p.add_argument("--clear-rhythm", type=float, default=config.MIN_CONFIDENCE,
                   help="periodicity from which a motion rhythm counts as clear; default "
                        "the app's MIN_CONFIDENCE, a rhythm it would accept as a pulse "
                        "(descriptive sections only)")
    p.add_argument("--limit", type=int, default=None, help="first N recordings only")
    args = p.parse_args()

    for path, what in ((args.store, "store"), (args.bbox_root, "bbox folder"),
                       (args.eval_dir, "benchmark output folder")):
        if not path.exists():
            raise SystemExit(f"ERROR: {what} not found: {path}")
    windows = read_windows(args.eval_dir)
    recordings = read_recordings(args.eval_dir)

    t0 = time.perf_counter()
    rows, problems = [], []
    with h5py.File(args.store, "r") as store:
        names = sorted(store.keys())[:args.limit]
        for i, name in enumerate(names, 1):
            rows.extend(recording_motion(store[name], args.bbox_root, problems))
            if i % 10 == 0 or i == len(names):
                print(f"  {i}/{len(names)} recordings, {time.perf_counter() - t0:.0f} s",
                      flush=True)

    motion = pd.DataFrame(rows)
    w = windows.merge(motion, on=["recording", "start"], how="inner", validate="one_to_one")
    expected = windows["recording"].isin(motion["recording"].unique()).sum()
    if len(w) != expected or len(w) != len(motion):
        raise SystemExit(f"ERROR: benchmark windows ({expected}) and motion windows "
                         f"({len(motion)}) do not line up ({len(w)} matched)")
    w["pair"] = w["activity"].map(PAIRS)
    tol = args.tolerance
    for source in SOURCES:
        w[f"ecg_{source}_code"] = on_rhythm(w["ref_hr"], w[f"{source}_rhythm"], tol)
        for short in ESTIMATORS:
            w[f"{short}_{source}_code"] = on_rhythm(w[f"{short}_pred_hr"],
                                                    w[f"{source}_rhythm"], tol)

    measurable = w["referenced"].astype(bool) & ~w["off_span"].astype(bool)
    summary = dict(
        script="diagnose_ecg_fitness_motion", exploratory=True,
        store=str(args.store), bbox_root=str(args.bbox_root), eval_dir=str(args.eval_dir),
        recordings=int(w["recording"].nunique()), windows=int(len(w)),
        tolerance_bpm=tol, clear_rhythm=args.clear_rhythm, problems=problems,
        motion_by_activity=motion_by_activity(w, args.clear_rhythm),
        crop_sees_box=crop_sees_box(w, args.clear_rhythm, tol),
        scored={}, all_referenced={})
    for short in ESTIMATORS:
        scored = w[f"{short}_scored"].astype(bool) & w["primary"].astype(bool)
        confident_wrong = w[f"{short}_confident_wrong"].astype(bool)
        off = (w[f"{short}_pred_hr"] - w["ref_hr"]).abs() > WRONG_BPM
        with_reading = measurable & w[f"{short}_pred_hr"].notna()
        summary["scored"][short] = {
            src: lock_table(w, short, src, scored, confident_wrong) for src in SOURCES}
        summary["all_referenced"][short] = {
            src: lock_table(w, short, src, with_reading, off) for src in SOURCES}

    per_recording = recording_table(w, recordings, args.clear_rhythm)
    args.eval_dir.mkdir(parents=True, exist_ok=True)
    w.to_csv(args.eval_dir / "motion_windows.csv", index=False)
    per_recording.to_csv(args.eval_dir / "motion_recordings.csv", index=False)
    with open(args.eval_dir / "summary_motion.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print_summary(summary, per_recording, args.eval_dir)
    return 0


def fmt(x, pct=False, digits=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{100 * x:.0f}%" if pct else f"{x:.{digits}f}"


def print_summary(s, per_recording, out_dir):
    print("\nECG-Fitness motion diagnostic - EXPLORATORY, decided after the benchmark")
    print(f"{s['recordings']} recordings, {s['windows']} windows | on the rhythm = within "
          f"{s['tolerance_bpm']:g} bpm | clear rhythm = periodicity >= {s['clear_rhythm']:g}")
    for problem in s["problems"]:
        print(f"  ! {problem}")

    print("\n1. Head motion by activity (window medians; amp = in-band displacement)")
    print(f"{'act':>4} {'windows':>8} {'box amp %face':>14} {'box clear':>10} "
          f"{'box rhythm':>11} {'crop amp px':>12} {'crop clear':>11}")
    for act, r in s["motion_by_activity"].items():
        print(f"{act:>4} {r['windows']:>8} {fmt(r['box_amp_pct_face'], digits=2):>14} "
              f"{fmt(r['box_clear'], pct=True):>10} {fmt(r['box_rhythm_when_clear']):>11} "
              f"{fmt(r['crop_amp_px'], digits=2):>12} {fmt(r['crop_clear'], pct=True):>11}")

    def lock_rows(block, title, kept_columns=False):
        print(title)
        extra = f" {'kept if on':>10} {'kept if off':>11}" if kept_columns else ""
        print(f"{'':>15} {'pair':>6} {'wrong':>6} {'reading on':>11} {'on 2x/.5x':>10} "
              f"{'ECG on':>7} {'on 2x/.5x':>10} {'right':>6} {'right on':>9}{extra}")
        for short in ESTIMATORS:
            for source in SOURCES:
                for pair, r in block[short][source].items():
                    extra = (f" {fmt(r['wrong_kept_when_on'], pct=True):>10} "
                             f"{fmt(r['wrong_kept_when_off'], pct=True):>11}"
                             if kept_columns else "")
                    print(f"{short + ' / ' + source:>15} {pair:>6} {r['wrong']:>6} "
                          f"{fmt(r['wrong_reading_on'], pct=True):>11} "
                          f"{fmt(r['wrong_reading_harmonic'], pct=True):>10} "
                          f"{fmt(r['wrong_ecg_on'], pct=True):>7} "
                          f"{fmt(r['wrong_ecg_harmonic'], pct=True):>10} {r['right']:>6} "
                          f"{fmt(r['right_reading_on_any'], pct=True):>9}{extra}")

    lock_rows(s["scored"], "\n2. Part 0's scored windows (kept, in reported primary "
                           "recordings): wrong = confident-wrong")
    lock_rows(s["all_referenced"], f"\n3. Every referenced window, kept or not, all "
                                   f"recordings: wrong = more than {WRONG_BPM:g} bpm off; "
                                   f"kept = share of wrong windows the gates let through",
              kept_columns=True)

    print("\n4. Can the app see the rhythm in its own crop? (windows with a clear box rhythm)")
    print(f"{'act':>4} {'windows':>8} {'crop = box':>11} {'2x/.5x':>7} {'median |diff|':>14}")
    for act, r in s["crop_sees_box"].items():
        print(f"{act:>4} {r['windows']:>8} {fmt(r['same'], pct=True):>11} "
              f"{fmt(r['harmonic'], pct=True):>7} {fmt(r['median_abs_diff']):>14}")

    answered = per_recording[per_recording["activity"].isin(["02", "04", "05", "06"])
                             & (per_recording["physnet"].notna()
                                | per_recording["pos"].notna())]
    print("\n5. Answered recordings on 02/04/05/06 (rhythm = median over windows with a clear "
          "rhythm; their count of 21 in brackets)")
    print(f"{'rec':>6} {'set':>8} {'ECG':>5} {'PhysNet':>14} {'POS':>14} "
          f"{'box rhythm':>12} {'crop rhythm':>12}")
    for _, r in answered.sort_values("recording").iterrows():
        cells = [f"{fmt(r[k], digits=0)} {SHORT_STATUS.get(r[k + '_status'], '?')}"
                 if np.isfinite(r[k]) else "-" for k in ("physnet", "pos")]
        print(f"{r['recording']:>6} {'primary' if r['primary'] else 'separate':>8} "
              f"{fmt(r['ecg'], digits=0):>5} {cells[0]:>14} {cells[1]:>14} "
              f"{fmt(r['box_rhythm'], digits=0):>7} ({r['box_clear_windows']:>2}) "
              f"{fmt(r['crop_rhythm'], digits=0):>7} ({r['crop_clear_windows']:>2})")
    print(f"\nwrote motion_windows.csv, motion_recordings.csv, summary_motion.json to {out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())