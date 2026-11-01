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

reencode: does compressing a recording the way a browser recorder does move the reading?
The twelve raw VitalVideos-WW recordings are re-encoded on a ladder of six rungs - R0
lossless 4:4:4, R1 lossless 4:2:0, R2 H.264 at the app's recorder setting in bits per pixel
per frame (decisive), R3 and R4 at a half and a quarter of it, R5 VP8 at it - from the same
frames: every second one, scaled to half size plane by plane in YUV 4:4:4. The recipe is
here; the files are written by a standalone tool outside the repository, which calls
write_ladder and write_manifest. This command first holds the files to their manifest and
checks that every R0 decodes to exactly the planes it was made from, then reads every rung
as the web app reads an upload and judges each against R0:
no effect when every recording R0 reports stays reported within 1.0 bpm; an effect when at
least two lose their reading or move by more than 2.0 bpm; inconclusive between.

frame-rate: down to which frame rate does v2's reading hold? Every VitalVideos-WW recording is
thinned from its 60 fps crops to 30, 27, 25, 24, 22 and 20 fps - the frame nearest each tick
of a uniform grid, index round(k x 60 / r) - with the reference read at the same indices, and
the readout is told the rate. The 30 fps thinning has to return the store's own frames in
every recording before anything is scored. Per rate, over the recordings reported at both
rates: the mean change in absolute error against 30 fps, with a 95% interval from recordings
resampled whole, and how many fewer recordings are reported. A rate holds within +0.5 bpm and
3 fewer, and fails beyond +0.5 bpm or 15 fewer; the verdict is the lowest rate that holds with
every rate above it holding too.

Usage:
    python -m app.live_vitals.scripts.check_hr_robustness low-hr [--device cpu|cuda]
        [--limit N] [--vv-store PATH] [--ubfc-phys-store PATH] [--rice-store PATH]
        [--out-dir DIR]
    python -m app.live_vitals.scripts.check_hr_robustness reencode [--device cpu|cuda]
        [--limit N] [--source DIR] [--ladder-dir DIR] [--out-dir DIR]
    python -m app.live_vitals.scripts.check_hr_robustness frame-rate [--device cpu|cuda]
        [--limit N] [--store PATH] [--out-dir DIR]

No frame is rendered; every store and source video is read-only.
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
from itertools import zip_longest
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals import config
from app.live_vitals.capture import session
from app.live_vitals.preprocess.face_box import make_landmarker
from app.live_vitals.scripts import check_rice_ippg, check_ubfc_phys, check_vitalvideos
from app.live_vitals.scripts.bench_outputs import file_sha256, git_state, provenance, subset_suffix
from app.live_vitals.scripts.check_ubfc_phys import REPORTED, _write_csv, make_estimators
from app.live_vitals.scripts.check_ubfc_regression import window_starts

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

# Check 2 (reencode), Part 0's numbers.
RAW_SUBSET = Path(r"D:\QualityPhys\demo_data\vitalvideos_subset")
DEFAULT_LADDER_DIR = Path(r"D:\QualityPhys\demo_data\vitalvideos_reencoded")
SOURCE = dict(codec="h264", pix_fmt="yuv444p", height=1056, width=960, fps=60.0, frames=3600)
FRAME_STEP = 2                          # frames 0, 2, 4 ...: the 30 fps frames the store keeps
LADDER_HEIGHT, LADDER_WIDTH = 528, 480  # half size, so the face spans a webcam's pixels
LADDER_FPS = 30
BROWSER_BITS = 6e6 / (640 * 480 * 30)   # the app's recorder: 640x480 at 6 Mbit/s and 30 fps
KEYFRAME_INTERVAL = 60
RUNGS = (  # bits: per pixel per frame; None is lossless
    dict(rung="R0", codec="libx264", container="mp4", pix_fmt="yuv444p", bits=None),
    dict(rung="R1", codec="libx264", container="mp4", pix_fmt="yuv420p", bits=None),
    dict(rung="R2", codec="libx264", container="mp4", pix_fmt="yuv420p", bits=BROWSER_BITS),
    dict(rung="R3", codec="libx264", container="mp4", pix_fmt="yuv420p", bits=BROWSER_BITS / 2),
    dict(rung="R4", codec="libx264", container="mp4", pix_fmt="yuv420p", bits=BROWSER_BITS / 4),
    dict(rung="R5", codec="libvpx", container="webm", pix_fmt="yuv420p", bits=BROWSER_BITS),
)
DECODED_AS = {"libx264": "h264", "libvpx": "vp8"}
MANIFEST = "manifest.json"  # written with the ladder by the tool outside the repository
DECISIVE_RUNG = "R2"
NO_EFFECT_BPM = 1.0        # every recording R0 reports stays reported within this: no effect
EFFECT_BPM = 2.0           # lost, or moved by more than this, ...
EFFECT_RECORDINGS = 2      # ... in at least this many recordings: an effect

# Check 3 (frame-rate), Part 0's numbers.
RATES = (30, 27, 25, 24, 22, 20)  # fps; 30 is the baseline the lower rates are judged against
NATIVE_FPS = 60                   # the store's frames_60 and bvp_60
HOLD_BPM = 0.5                    # holds: the interval's upper bound at or below this ...
HOLD_FEWER = 3                    # ... and at most this many fewer recordings reported
FAIL_FEWER = 15                   # fails: the lower bound above HOLD_BPM, or more fewer than this

# ================================================================== check 1 ==

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


# ================================================================== check 2 ==

# ------------------------------------------------------------------- ladder --

def rung_file(ladder_dir, recording, rung):
    """Where one recording's rung lives: <ladder_dir>/<recording>/<rung>.<container>."""
    return Path(ladder_dir) / recording / f"{rung['rung']}.{rung['container']}"


def target_bit_rate(rung):
    """A lossy rung's target in bits per second, from its bits per pixel per frame."""
    return None if rung["bits"] is None else rung["bits"] * LADDER_WIDTH * LADDER_HEIGHT * LADDER_FPS


def encoder_options(rung):
    """Part 0's encoder settings for a rung.

    Lossy rungs imitate a real-time encoder: no B-frames and no look-ahead for H.264, the
    real-time deadline for VP8; the target is both the average and the maximum rate, with a
    one-second buffer; a key frame every KEYFRAME_INTERVAL frames.
    """
    keyframes = {"g": str(KEYFRAME_INTERVAL)}
    if rung["bits"] is None:
        return {"qp": "0", "preset": "veryfast", **keyframes}
    rate = str(int(round(target_bit_rate(rung))))
    capped = {"maxrate": rate, "bufsize": rate, **keyframes}
    if rung["codec"] == "libx264":
        return {"preset": "veryfast", "tune": "zerolatency", **capped}
    return {"deadline": "realtime", "cpu-used": "4", **capped}


def source_planes(path):
    """Every second frame of a raw recording as half-size YUV 4:4:4 planes, uint8 (3, H, W).

    Each plane is scaled on its own with INTER_AREA, in the recording's own colour space, so
    that the lossless baseline can hold the scaled frames to the bit. Stops unless the file is
    the recording Part 0 describes.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        # Frame threads decode these files about ten times faster, to the same frames.
        stream.thread_type = "AUTO"
        found = dict(codec=stream.codec_context.name, pix_fmt=stream.codec_context.pix_fmt,
                     height=stream.codec_context.height, width=stream.codec_context.width,
                     fps=float(stream.average_rate), frames=stream.frames)
        if found != SOURCE:
            raise ValueError(f"{Path(path).name}: {found}, but Part 0 describes {SOURCE}")
        count = 0
        for index, frame in enumerate(container.decode(stream)):
            count += 1
            if index % FRAME_STEP == 0:
                yield np.stack([cv2.resize(plane, (LADDER_WIDTH, LADDER_HEIGHT),
                                           interpolation=cv2.INTER_AREA)
                                for plane in frame.to_ndarray(format="yuv444p")])
        if count != SOURCE["frames"]:
            raise ValueError(f"{Path(path).name}: decoded {count} frames, not {SOURCE['frames']}")


def write_ladder(source_path, ladder_dir, recording):
    """Writes R0-R5 of one raw recording in a single pass over its frames."""
    (Path(ladder_dir) / recording).mkdir(parents=True, exist_ok=True)
    opened = []
    try:
        for rung in RUNGS:
            container = av.open(str(rung_file(ladder_dir, recording, rung)), "w")
            opened.append((container, None))  # listed before its stream, so a failure still closes it
            stream = container.add_stream(rung["codec"], rate=LADDER_FPS,
                                          options=encoder_options(rung))
            stream.width, stream.height, stream.pix_fmt = LADDER_WIDTH, LADDER_HEIGHT, rung["pix_fmt"]
            if rung["bits"] is not None:
                stream.bit_rate = int(round(target_bit_rate(rung)))
            opened[-1] = (container, stream)
        for index, planes in enumerate(source_planes(source_path)):
            frame = av.VideoFrame.from_ndarray(planes, format="yuv444p")
            frame.pts, frame.time_base = index, Fraction(1, LADDER_FPS)
            for container, stream in opened:
                container.mux(stream.encode(frame))
        for container, stream in opened:
            container.mux(stream.encode(None))
    finally:
        for container, _ in opened:
            container.close()


# -------------------------------------------------------------------- files --

def decoded_planes(path):
    """A video's frames as YUV 4:4:4 planes, uint8 (3, H, W)."""
    with av.open(str(path)) as container:
        container.streams.video[0].thread_type = "AUTO"
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="yuv444p")


def r0_self_test(source_path, r0_path):
    """Whether R0 decodes to exactly the scaled planes of the recording it was made from."""
    frames, worst = 0, 0
    for expected, decoded in zip_longest(source_planes(source_path), decoded_planes(r0_path)):
        if expected is None or decoded is None:
            return dict(equal=False, frames=frames, max_abs_diff=None)
        worst = max(worst, int(np.abs(expected.astype(np.int16) - decoded.astype(np.int16)).max()))
        frames += 1
    return dict(equal=worst == 0, frames=frames, max_abs_diff=worst)


def file_facts(path):
    """What a rung file holds: codec, pixel format, size, frames, rate, bits per pixel per frame.

    Frames are counted, and the rate measured from their timestamps, by decoding: a WebM
    file states neither.
    """
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        codec, pix_fmt = stream.codec_context.name, stream.codec_context.pix_fmt
        width, height = stream.codec_context.width, stream.codec_context.height
        times = [frame.time for frame in container.decode(stream)]
    frames = len(times)
    rate = (frames - 1) / (times[-1] - times[0]) if frames > 1 and times[-1] > times[0] else float("nan")
    size = Path(path).stat().st_size
    bits = size * 8 / (width * height * frames) if frames else float("nan")
    return dict(codec=codec, pix_fmt=pix_fmt, width=width, height=height, frames=frames,
                fps=rate, bytes=size, bits_per_pixel_frame=bits)


def facts_match(facts, rung):
    """Whether a rung file is what the recipe says it is.

    The rate is allowed 0.1 fps: WebM stamps frames to the millisecond, which moves a rate
    measured over a short file by a few hundredths.
    """
    return (facts["codec"] == DECODED_AS[rung["codec"]] and facts["pix_fmt"] == rung["pix_fmt"]
            and (facts["width"], facts["height"]) == (LADDER_WIDTH, LADDER_HEIGHT)
            and facts["frames"] == SOURCE["frames"] // FRAME_STEP
            and abs(facts["fps"] - LADDER_FPS) < 0.1)


# ----------------------------------------------------------------- readings --

def read_like_the_app(path, landmarker, estimators):
    """A rung file read as the web app reads an upload: one reading per estimator.

    The capture path decides first: a refused capture is `capture_rejected` and too few
    frames `insufficient_frames`, with no value; otherwise the estimator's own status stands.
    """
    clip, fps, quality = session.crops_from_video(path, landmarker)
    readings = {}
    for estimator in estimators:
        result = None
        if quality.verdict == "REJECT":
            status = "capture_rejected"
        elif len(clip) < config.CLIP_LEN:
            status = "insufficient_frames"
        else:
            result = estimator.estimate(clip, fps)
            status = result.status
        detail = (result.detail or {}) if result is not None else {}
        readings[estimator.name] = dict(
            value=float(result.value) if result is not None else float("nan"),
            status=status, reported=status in REPORTED,
            capture_verdict=quality.verdict, capture_reasons="; ".join(quality.reasons),
            effective_fps=float(quality.effective_fps), frames=int(quality.n_frames),
            confidence=float(result.confidence) if result is not None else float("nan"),
            usable_fraction=float(detail.get("usable_fraction", float("nan"))))
    return readings


def rung_outcome(base, rung):
    """Part 0's outcome of one rung against R0, over the recordings R0 reports.

    `base` and `rung` map each recording to its reading. No effect: every recording R0
    reports is reported at the rung too, within NO_EFFECT_BPM of its R0 value. An effect: at
    least EFFECT_RECORDINGS of them lose their reading or move by more than EFFECT_BPM.
    Anything between is inconclusive.
    """
    names = sorted(name for name, reading in base.items() if reading["reported"])
    lost = [name for name in names if not rung[name]["reported"]]
    moves = [abs(rung[name]["value"] - base[name]["value"])
             for name in names if rung[name]["reported"]]
    hits = len(lost) + sum(move > EFFECT_BPM for move in moves)
    if not names:
        verdict = "not estimable"
    elif not lost and all(move <= NO_EFFECT_BPM for move in moves):
        verdict = "no effect"
    elif hits >= EFFECT_RECORDINGS:
        verdict = "effect"
    else:
        verdict = "inconclusive"
    return dict(base_reported=len(names), lost=len(lost),
                gained=sum(1 for name, reading in base.items()
                           if not reading["reported"] and rung[name]["reported"]),
                moved_over_no_effect=sum(move > NO_EFFECT_BPM for move in moves),
                moved_over_effect=sum(move > EFFECT_BPM for move in moves),
                max_move=max(moves) if moves else None,
                median_move=_median(moves), outcome=verdict)


def reference_rates(recordings, store=check_vitalvideos.DEFAULT_STORE):
    """Each recording's ECG heart rate from the VitalVideos-WW store, for description only."""
    with h5py.File(store, "r") as handle:
        return {name: float(handle[name].attrs["hr_ecg"]) if name in handle else float("nan")
                for name in recordings}


# ----------------------------------------------------------------- manifest --

def ladder_settings():
    """The ladder as this code builds it, for the tool's manifest and the check's summary."""
    return dict(source=dict(SOURCE), frame_step=FRAME_STEP,
                ladder=dict(height=LADDER_HEIGHT, width=LADDER_WIDTH, fps=LADDER_FPS,
                            scaling="INTER_AREA, plane by plane in YUV 4:4:4"),
                browser_bits_per_pixel_frame=BROWSER_BITS, keyframe_interval=KEYFRAME_INTERVAL,
                rungs=[dict(rung, target_bit_rate=target_bit_rate(rung),
                            options=encoder_options(rung)) for rung in RUNGS],
                pyav=av.__version__)


def write_manifest(ladder_dir, sources, minutes=None):
    """Records the recipe, the repository's state and every rung file written, with its hash."""
    files = [rung_file(ladder_dir, source.parent.name, rung) for source in sources for rung in RUNGS]
    manifest = dict(
        written_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), minutes=minutes,
        recipe=ladder_settings(), repository=git_state(),
        recordings=[source.parent.name for source in sources],
        files={path.relative_to(ladder_dir).as_posix(): dict(bytes=path.stat().st_size,
                                                              sha256=file_sha256(path))
               for path in files})
    (Path(ladder_dir) / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def manifest_problems(ladder_dir, recordings):
    """What keeps the ladder on disk from being the one its manifest records under this recipe.

    The encoder's settings cannot be read back from the files, so the manifest written with
    them has to name this code's recipe, and every file has to be the one it hashed.
    """
    path = Path(ladder_dir) / MANIFEST
    if not path.is_file():
        return [f"no {MANIFEST} in {ladder_dir}: the ladder was not written by the tool"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    problems = []
    if manifest.get("recipe") != json.loads(json.dumps(ladder_settings())):
        problems.append("the manifest's recipe is not this code's")
    for name in recordings:
        for rung in RUNGS:
            file = rung_file(ladder_dir, name, rung)
            entry = manifest.get("files", {}).get(file.relative_to(ladder_dir).as_posix())
            if entry is None or entry["sha256"] != file_sha256(file):
                problems.append(f"{file.relative_to(ladder_dir).as_posix()} does not match the manifest")
    return problems


# ------------------------------------------------------------------ summary --

def reencode_settings(args, recordings):
    """Check 2's rule as this run applies it, for the notebook to hold against Part 0."""
    return dict(ladder_settings(), decisive_rung=DECISIVE_RUNG, no_effect_bpm=NO_EFFECT_BPM,
                effect_bpm=EFFECT_BPM, effect_recordings=EFFECT_RECORDINGS,
                estimators=list(ESTIMATORS), deciding=DECIDING, reported_statuses=list(REPORTED),
                recordings=len(recordings), raw_subset=str(args.source),
                ladder_dir=str(args.ladder_dir), limit=args.limit)


def print_reencode(summary):
    """Prints the self-test, the files and each rung's outcome against R0."""
    test = summary["self_test"]
    print(f"\nself-test: R0 decodes to its scaled source in {test['equal']} of {test['recordings']} "
          f"recordings ({test['frames_each']} frames each; largest difference {test['max_abs_diff']})")
    for rung, files in summary["files"].items():
        target = files["target_bits_per_pixel_frame"]
        print(f"  {rung}: {files['codec']} {files['pix_fmt']}, median "
              f"{files['median_bits_per_pixel_frame']:.3f} bits per pixel per frame "
              f"(target {'lossless' if target is None else f'{target:.3f}'})")
    for rung, block in summary["rungs"].items():
        for name, result in block.items():
            move = "-" if result["max_move"] is None else f"{result['max_move']:.2f}"
            print(f"  {rung} | {name:<14}: R0 reports {result['base_reported']}, lost "
                  f"{result['lost']}, gained {result['gained']}, moved >{NO_EFFECT_BPM:g} bpm "
                  f"{result['moved_over_no_effect']}, >{EFFECT_BPM:g} bpm "
                  f"{result['moved_over_effect']}, largest {move} -> {result['outcome']}")
    verdict = summary["verdict"]
    print(f"\nVERDICT, check 2 ({verdict['rung']}, {verdict['estimator']}): {verdict['outcome']}")


# ---------------------------------------------------------------------- run --

def run_reencode(args):
    """Check 2: the R0 self-test first, then every rung read like an upload; writes the outputs."""
    sources = sorted(Path(args.source).glob("*/*_1.mp4"))
    if args.limit:
        sources = sources[:args.limit]
    recordings = [source.parent.name for source in sources]
    missing = [rung_file(args.ladder_dir, name, rung) for name in recordings for rung in RUNGS
               if not rung_file(args.ladder_dir, name, rung).is_file()]
    if not recordings or missing:
        raise SystemExit(f"ERROR: {len(recordings)} recordings under {args.source}, "
                         f"{len(missing)} rung files missing under {args.ladder_dir}")
    problems = manifest_problems(args.ladder_dir, recordings)
    if problems:
        raise SystemExit("ERROR: the ladder is not the recorded one:\n  " + "\n  ".join(problems))
    manifest = json.loads((Path(args.ladder_dir) / MANIFEST).read_text(encoding="utf-8"))
    estimators = make_estimators(list(ESTIMATORS), args.device)
    deciding = next(e for e in estimators if e.name == DECIDING)
    produced_by = provenance(deciding)
    suffix = subset_suffix(("limit", args.limit or None, None))

    # Self-tests come first: every R0 must hold the scaled source frames to the bit.
    self_test = {}
    for source, name in zip(sources, recordings):
        self_test[name] = r0_self_test(source, rung_file(args.ladder_dir, name, RUNGS[0]))
        print(f"  self-test {name}: {self_test[name]}", flush=True)
    if not all(result["equal"] for result in self_test.values()):
        raise SystemExit("ERROR: an R0 does not decode to the frames it was made from; nothing scored")

    facts = {rung["rung"]: [] for rung in RUNGS}
    for name in recordings:
        for rung in RUNGS:
            found = file_facts(rung_file(args.ladder_dir, name, rung))
            if not facts_match(found, rung):
                raise SystemExit(f"ERROR: {name} {rung['rung']} is not the recipe's file: {found}")
            facts[rung["rung"]].append(found)

    reference = reference_rates(recordings)
    readings = {e.name: {rung["rung"]: {} for rung in RUNGS} for e in estimators}
    landmarker = make_landmarker()
    try:
        for i, name in enumerate(recordings, 1):
            for rung in RUNGS:
                path = rung_file(args.ladder_dir, name, rung)
                for estimator_name, reading in read_like_the_app(path, landmarker, estimators).items():
                    readings[estimator_name][rung["rung"]][name] = reading
            print(f"  {i}/{len(recordings)} recordings read on every rung", flush=True)
    finally:
        landmarker.close()

    rows = []
    for estimator_name, by_rung in readings.items():
        base = by_rung[RUNGS[0]["rung"]]
        for rung, by_name in by_rung.items():
            for name, reading in by_name.items():
                rows.append(dict(recording=name, rung=rung, estimator=estimator_name, **reading,
                                 change_vs_r0=(reading["value"] - base[name]["value"]
                                               if reading["reported"] and base[name]["reported"]
                                               else float("nan")),
                                 error_vs_hr_ecg=reading["value"] - reference[name]))
    summary = dict(
        check="reencode", part0=PART0, settings=reencode_settings(args, recordings),
        readout=readout_settings(),
        ladder=dict(manifest_sha256=file_sha256(Path(args.ladder_dir) / MANIFEST),
                    written_utc=manifest["written_utc"], repository=manifest["repository"]),
        self_test=dict(recordings=len(self_test),
                       equal=sum(result["equal"] for result in self_test.values()),
                       frames_each=sorted({result["frames"] for result in self_test.values()}),
                       max_abs_diff=max(result["max_abs_diff"] for result in self_test.values())),
        files={rung["rung"]: dict(
            codec=facts[rung["rung"]][0]["codec"], pix_fmt=facts[rung["rung"]][0]["pix_fmt"],
            target_bits_per_pixel_frame=rung["bits"],
            median_bits_per_pixel_frame=_median([f["bits_per_pixel_frame"] for f in facts[rung["rung"]]]),
            bytes=sum(f["bytes"] for f in facts[rung["rung"]])) for rung in RUNGS},
        rungs={rung["rung"]: {name: rung_outcome(by_rung[RUNGS[0]["rung"]], by_rung[rung["rung"]])
                              for name, by_rung in readings.items()}
               for rung in RUNGS[1:]},
        reference=dict(source="hr_ecg of the VitalVideos-WW store, description only",
                       median_abs_error={name: {rung["rung"]: _median(
                           [abs(r["value"] - reference[n]) for n, r in readings[name][rung["rung"]].items()
                            if r["reported"]]) for rung in RUNGS} for name in readings}))
    summary["verdict"] = dict(rung=DECISIVE_RUNG, estimator=DECIDING,
                              outcome=summary["rungs"][DECISIVE_RUNG][DECIDING]["outcome"])
    summary["provenance"] = produced_by
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / f"recordings_reencode{suffix}.csv", rows)
    (args.out_dir / f"summary_reencode{suffix}.json").write_text(json.dumps(summary, indent=2),
                                                                 encoding="utf-8")
    print_reencode(summary)
    print(f"\noutputs: {args.out_dir}" + (f" (subset run: names end in {suffix})" if suffix else ""))
    return summary


# ================================================================== check 3 ==

# -------------------------------------------------------------------- rates --

def thinned_indices(n_frames, rate):
    """Part 0's thinning: the 60 fps frame nearest each tick of a uniform grid at `rate`.

    Index round(k x 60 / rate) for k = 0, 1, 2 ... while it stays inside the recording, in
    exact fractions: a tick halfway between two frames - every second one at 24 fps - goes
    to the even index, as Python's round does.
    """
    indices, k = [], 0
    while (index := round(Fraction(k * NATIVE_FPS, rate))) < n_frames:
        indices.append(index)
        k += 1
    return np.array(indices, dtype=int)


def thinning_self_test(store, names):
    """Whether each recording's 30 fps thinning returns the store's own 30 fps streams.

    Part 0 asks it of the frames, byte for byte; the reference is compared as well and
    reported beside it.
    """
    results = {}
    for name in names:
        group = store[name]
        index = thinned_indices(len(group["frames_60"]), RATES[0])
        results[name] = dict(
            frames_equal=bool(np.array_equal(group["frames_60"][()][index], group["frames"][()])),
            bvp_equal=bool(np.array_equal(group["bvp_60"][()][index], group["bvp"][()])))
    return results


def read_at_rate(frames, bvp, rate, estimators):
    """One recording at one rate: the reference read through the windows, then each estimator.

    The readout is told `rate`, as the app tells it a capture's measured rate, and its
    windows stay CLIP_LEN frames, so they last longer at lower rates. The frame-rate gate is
    not applied: it is what this check measures. An absolute error exists only where both
    the estimator and the reference report a value.
    """
    starts = list(window_starts(len(frames)))
    reference, _ = check_ubfc_phys.reference_windows(bvp, rate, starts)
    ref_reported = reference.status in REPORTED
    rows = {}
    for estimator in estimators:
        result = estimator.estimate(frames, rate)
        reported = result.status in REPORTED
        detail = result.detail or {}
        rows[estimator.name] = dict(
            frames=len(frames), windows=len(starts), value=float(result.value),
            status=result.status, reported=reported, confidence=float(result.confidence),
            usable_fraction=float(detail.get("usable_fraction", 0.0)),
            ref_value=float(reference.value), ref_status=reference.status,
            abs_error=(abs(float(result.value) - float(reference.value))
                       if reported and ref_reported else float("nan")))
    return rows


def evaluate_rates(store_path, estimators, limit):
    """The self-test over every recording first, then one row per recording, rate and estimator."""
    rows = []
    with h5py.File(store_path, "r") as store:
        names = sorted(store.keys())
        if limit:
            names = names[:limit]
        for name in names:
            if abs(float(store[name].attrs["fps_native"]) - NATIVE_FPS) > 0.01:
                raise SystemExit(f"ERROR: {name} has no {NATIVE_FPS} fps stream to thin")
        self_test = thinning_self_test(store, names)
        if not all(result["frames_equal"] for result in self_test.values()):
            raise SystemExit("ERROR: the 30 fps thinning does not return the stored frames; "
                             "nothing scored")
        for i, name in enumerate(names, 1):
            group = store[name]
            frames_60, bvp_60 = group["frames_60"][()], group["bvp_60"][()]
            hr_ecg = float(group.attrs["hr_ecg"])
            for rate in RATES:
                index = thinned_indices(len(frames_60), rate)
                for estimator_name, row in read_at_rate(frames_60[index], bvp_60[index], rate,
                                                        estimators).items():
                    rows.append(dict(recording=name, rate=rate, estimator=estimator_name, **row,
                                     error_vs_hr_ecg=row["value"] - hr_ecg))
            if i % 10 == 0 or i == len(names):
                print(f"  {i}/{len(names)} recordings read at every rate", flush=True)
    return self_test, rows


# ------------------------------------------------------------------ outcome --

def rate_outcome(base, at_rate, draws=BOOT_DRAWS, seed=BOOT_SEED):
    """Part 0's outcome of one rate against 30 fps, for one estimator.

    `base` and `at_rate` map each recording to its row. Over the recordings reported at both
    rates with an error at both: the mean of (absolute error at the rate) minus (absolute
    error at 30 fps), with a 95% interval from recordings resampled with replacement. Fewer
    is the recordings reported at 30 fps minus those reported at the rate, over all of them.
    Holds: upper bound at most HOLD_BPM and at most HOLD_FEWER fewer. Fails: lower bound
    above HOLD_BPM, or more than FAIL_FEWER fewer. Inconclusive between.
    """
    both = [name for name in sorted(base) if base[name]["reported"] and at_rate[name]["reported"]]
    names = [name for name in both
             if math.isfinite(base[name]["abs_error"]) and math.isfinite(at_rate[name]["abs_error"])]
    reported_at_30 = sum(row["reported"] for row in base.values())
    reported = sum(row["reported"] for row in at_rate.values())
    fewer = reported_at_30 - reported
    point = lower = upper = None
    if names:
        changes = np.array([at_rate[name]["abs_error"] - base[name]["abs_error"] for name in names])
        point = float(changes.mean())
        pick = np.random.default_rng(seed).integers(0, len(changes), size=(draws, len(changes)))
        lower, upper = (float(v) for v in np.percentile(changes[pick].mean(axis=1), [2.5, 97.5]))
    if upper is not None and upper <= HOLD_BPM and fewer <= HOLD_FEWER:
        verdict = "holds"
    elif (lower is not None and lower > HOLD_BPM) or fewer > FAIL_FEWER:
        verdict = "fails"
    elif lower is None:
        verdict = "not estimable"
    else:
        verdict = "inconclusive"
    return dict(reported_at_30=reported_at_30, reported=reported, fewer=fewer, pairs=len(names),
                pairs_without_reference=len(both) - len(names), point=point, lower=lower,
                upper=upper, draws=draws, outcome=verdict)


def lowest_holding_rate(outcomes):
    """Part 0's verdict: the lowest rate that holds with every rate above it holding too.

    `outcomes` maps each rate below 30 fps to its outcome; 30 means no lower rate holds.
    """
    lowest = RATES[0]
    for rate in RATES[1:]:
        if outcomes[rate]["outcome"] != "holds":
            break
        lowest = rate
    return lowest


# ------------------------------------------------------------------ summary --

def frame_rate_settings(args, recordings):
    """Check 3's rule as this run applies it, for the notebook to hold against Part 0."""
    return dict(rates=list(RATES), native_fps=NATIVE_FPS,
                thinning="index round(k x 60 / r) while inside the recording; exact fractions, "
                         "half to even",
                window_frames=config.CLIP_LEN, frame_rate_gate="not applied",
                hold_bpm=HOLD_BPM, hold_fewer=HOLD_FEWER, fail_fewer=FAIL_FEWER,
                interval=dict(unit="recording", draws=BOOT_DRAWS, seed=BOOT_SEED,
                              percentiles=[2.5, 97.5]),
                estimators=list(ESTIMATORS), deciding=DECIDING, reported_statuses=list(REPORTED),
                store=str(args.store), recordings=len(recordings), limit=args.limit)


def print_frame_rate(summary):
    """Prints the self-test, then each estimator's rates against 30 fps and its lowest holding rate."""
    test = summary["self_test"]
    print(f"\nself-test: the 30 fps thinning returns the stored frames in {test['frames_equal']} of "
          f"{test['recordings']} recordings (the stored reference in {test['bvp_equal']})")
    for name, block in summary["estimators"].items():
        print(f"\n=== {name}")
        print(f"  {'fps':>4} {'frames':>6} {'reported':>8} {'fewer':>5} {'pairs':>5}   "
              f"{'change in |error|, bpm (95%)':<44} {'|error|':>7}")
        for row in block["rates"]:
            error = "-" if row["median_abs_error"] is None else f"{row['median_abs_error']:.2f}"
            judged = row["against_30"]
            if judged is None:
                fewer, pairs, change = "-", "-", "baseline"
            else:
                fewer, pairs = judged["fewer"], judged["pairs"]
                change = ("not estimable" if judged["point"] is None else
                          f"{judged['point']:+.2f} ({judged['lower']:+.2f} to {judged['upper']:+.2f})")
                change += f" -> {judged['outcome']}"
            print(f"  {row['rate']:>4} {row['frames_each'][0]:>6} {row['reported']:>8} {fewer:>5} "
                  f"{pairs:>5}   {change:<44} {error:>7}")
        lowest = block["lowest_holding_rate"]
        print(f"  lowest rate that holds, every rate above it holding too: {lowest} fps"
              + (" (no rate below 30 holds)" if lowest == RATES[0] else ""))
    verdict = summary["verdict"]
    print(f"\nVERDICT, check 3 ({verdict['estimator']}): holds down to {verdict['lowest_holding_rate']} fps")


# ---------------------------------------------------------------------- run --

def run_frame_rate(args):
    """Check 3: the self-test first, then every recording read at every rate; writes the outputs."""
    estimators = make_estimators(list(ESTIMATORS), args.device)
    deciding = next(e for e in estimators if e.name == DECIDING)
    # Read before anything is written, so the run's own outputs never list as changed files.
    produced_by = provenance(deciding)
    suffix = subset_suffix(("limit", args.limit or None, None))
    self_test, rows = evaluate_rates(args.store, estimators, args.limit)
    recordings = sorted(self_test)

    blocks = {}
    for name in ESTIMATORS:
        by_rate = {rate: {row["recording"]: row for row in rows
                          if row["estimator"] == name and row["rate"] == rate} for rate in RATES}
        outcomes = {rate: rate_outcome(by_rate[RATES[0]], by_rate[rate]) for rate in RATES[1:]}
        blocks[name] = dict(
            rates=[dict(rate=rate,
                        frames_each=sorted({row["frames"] for row in by_rate[rate].values()}),
                        windows_each=sorted({row["windows"] for row in by_rate[rate].values()}),
                        reported=sum(row["reported"] for row in by_rate[rate].values()),
                        median_abs_error=_median([row["abs_error"] for row in by_rate[rate].values()]),
                        median_abs_error_vs_hr_ecg=_median([abs(row["error_vs_hr_ecg"])
                                                            for row in by_rate[rate].values()
                                                            if row["reported"]]),
                        against_30=outcomes.get(rate))
                   for rate in RATES],
            lowest_holding_rate=lowest_holding_rate(outcomes))
    summary = dict(
        check="frame-rate", part0=PART0, settings=frame_rate_settings(args, recordings),
        readout=readout_settings(),
        self_test=dict(recordings=len(self_test),
                       frames_equal=sum(result["frames_equal"] for result in self_test.values()),
                       bvp_equal=sum(result["bvp_equal"] for result in self_test.values())),
        estimators=blocks,
        verdict=dict(estimator=DECIDING, lowest_holding_rate=blocks[DECIDING]["lowest_holding_rate"]))
    summary["provenance"] = produced_by
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out_dir / f"recordings_frame_rate{suffix}.csv", rows)
    (args.out_dir / f"summary_frame_rate{suffix}.json").write_text(json.dumps(summary, indent=2),
                                                                   encoding="utf-8")
    print_frame_rate(summary)
    print(f"\noutputs: {args.out_dir}" + (f" (subset run: names end in {suffix})" if suffix else ""))
    return summary

# ===================================================================== main ==


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
    encode = checks.add_parser("reencode", help="check 2: a browser-grade encode")
    encode.add_argument("--device", default="cpu", help="torch device for learned estimators")
    encode.add_argument("--limit", type=int, default=None,
                        help="the first N recordings only (smoke test)")
    encode.add_argument("--source", type=Path, default=RAW_SUBSET)
    encode.add_argument("--ladder-dir", type=Path, default=DEFAULT_LADDER_DIR)
    encode.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    frame_rate = checks.add_parser("frame-rate", help="check 3: lower frame rates")
    frame_rate.add_argument("--device", default="cpu", help="torch device for learned estimators")
    frame_rate.add_argument("--limit", type=int, default=None,
                            help="the first N recordings only (smoke test)")
    frame_rate.add_argument("--store", type=Path, default=check_vitalvideos.DEFAULT_STORE)
    frame_rate.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.check == "low-hr":
        run_low_hr(args)
    elif args.check == "reencode":
        run_reencode(args)
    elif args.check == "frame-rate":
        run_frame_rate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())