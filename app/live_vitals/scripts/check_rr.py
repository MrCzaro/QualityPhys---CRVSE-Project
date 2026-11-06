"""Respiratory rate from the torso's motion on VitalVideos-WW: the development stage and the
held-out stage.

**--stage dev** runs readout A (`signal.respiration.torso_rate`, the primary readout NB_P3_39
chose at its development gate) on every recording of VitalVideos' training split (VV-240), with
readout B, the torso's vertical shift, described beside it and deciding nothing, and checks that
both reproduce NB_P3_39 exactly:

- **The readings.** The digest of A's and B's whole-minute readings on all 240 training
  recordings (torso rows; A's rate, concentration and component; B's rate and
  concentration; rounded to six decimals, sorted by GUID) must equal NB_P3_39 section 11's.
- **The gate's counts.** Over the 208 recordings whose belt an independent surrogate
  confirms (`resp_ok`), A's counts must equal section 3's: read, within 2 and within 1
  breaths/min of the belt, and the two discordant counts against the baseline.

**--stage confirm** is the one run NB_P3_40's Part 0 fixed (2026-11-06) before any frame of
VitalVideos' held-out split (VV-60) was read for breathing. It runs once - it refuses to start
when its summary exists, and takes no --limit - and in this order:

1. The development stage again, at this commit. Unless it reproduces NB_P3_39, the run stops
   before reading a held-out frame.
2. The ladder held to its manifest - the commit that wrote it, every file's byte count and
   SHA-256 - and the fingerprint of the ladder's and the iPhone recordings' names and sizes.
3. VV-60's stored frames: A and B on all 60 recordings; on the 56 with `resp_ok`, A's share
   within 2 breaths/min of the belt, its coverage and the paired test against the baseline,
   and the outcome by Part 0's bar; descriptive figures beside them.
4. The ladder: the self-test on R0 for all 12 raw recordings first, then every rung, each
   decoded as NB_P3_27 decoded the raw video, reduced to `frames_full`'s geometry and read by
   A; R2 against R0 on the 11 `resp_ok` recordings decides, the other rungs describe.
5. The iPhone recordings, described: the face box from the app's detector, the frame reduced
   8-fold and averaged over 6 frames, read by A.

The reference is the waist belt as NB_P3_27 read it (`rr_belt`). The baseline gives every
recording the median belt rate of VV-240's `resp_ok` recordings. Agreement is the share within
2 breaths/min among the recordings read, beside coverage; the exact McNemar test against the
baseline counts an unread recording as a miss. B lives here rather than in `signal.respiration`
because it only describes.

Usage:
    python -m app.live_vitals.scripts.check_rr [--stage dev|confirm] [--store PATH] [--limit N]
        [--out-dir DIR]

The development stage takes a few minutes on the CPU and exits non-zero unless it reproduces
NB_P3_39; the held-out stage about 20-25 minutes. Summaries are committed; the per-recording
tables stay local. No frames are rendered or written. Stores and videos are read-only. Not a
medical device.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.stats import binomtest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals.scripts.bench_outputs import file_sha256, provenance, subset_suffix
from app.live_vitals.signal import respiration
from app.live_vitals.signal.respiration import (breathing_band, peak_share, resp_rate, torso_rate,
                                                torso_region)

DEFAULT_STORE = Path(r"D:\QualityPhys\phase 3 datasets\vitalvideos_phase3_rc_none.h5")
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "rr_eval"
SPLIT_CSV = _REPO_ROOT / "Data" / "vitalvideos_split.csv"  # GUID and split, frozen by NB_P3_27
WITHIN_BPM = 2.0
WILSON_Z = 1.959964
GATE_SHARE, GATE_COVERAGE, GATE_P = 0.82, 0.80, 0.01  # NB_P3_39's development gate
# What NB_P3_39 printed: section 11's digest of the readings, section 3's counts for A.
NB39_DIGEST = "f75095c4a475f6d8"
NB39_A_COUNTS = dict(scored=208, read=208, within_2=197, within_1=185, readout_only=128,
                     baseline_only=2)
READING_COLUMNS = ["guid", "torso_rows", "a_rate", "a_share", "a_component", "b_rate", "b_share"]

# The held-out stage: NB_P3_40's Part 0 (2026-11-06), fixed before any held-out frame was read.
BAR_LOWER, BAR_COVERAGE, BAR_P = 0.70, 0.80, 0.05
LADDER_DIR = Path(r"D:\QualityPhys\demo_data\vitalvideos_reencoded")
LADDER_COMMIT = "41736314e7dafd7799b31ec97c9dd0452825f2cc"  # the tool's commit, in the manifest
RAW_SUBSET = Path(r"D:\QualityPhys\demo_data\vitalvideos_subset")
RUNGS = ("R0", "R1", "R2", "R3", "R4", "R5")
RUNG_FILES = dict(R0="R0.mp4", R1="R1.mp4", R2="R2.mp4", R3="R3.mp4", R4="R4.mp4", R5="R5.webm")
DECIDING_RUNG = "R2"
NO_EFFECT_BPM = 1.0  # no effect: every recording R0 reads is read at the rung, within this ...
EFFECT_BPM = 2.0  # ... effect: at least EFFECT_RECORDINGS lose their reading or move by more
EFFECT_RECORDINGS = 2
LADDER_FS = 30.0  # the ladder holds every second frame of the 60 fps original
LADDER_DOWNSCALE, LADDER_BLOCK = 4, 3  # 480x528 to frames_full's 120x132; 30 fps to 10
SELF_TEST_LEVELS = 1.0  # R0 against frames_full over the torso, each pixel's mean removed
IPHONE_DOWNSCALE, IPHONE_BLOCK = 8, 6  # 1080x1920 to 135x240; 59.96 fps to 9.99
SETTINGS = {  # the freeze record's RULE, as this code carries it
    "bar: Wilson 95% lower bound at least": BAR_LOWER,
    "bar: coverage at least": BAR_COVERAGE,
    "bar: McNemar p below": BAR_P,
    "Wilson z": WILSON_Z,
    "ladder: deciding rung": DECIDING_RUNG,
    "ladder: no effect, every move within (breaths/min)": NO_EFFECT_BPM,
    "ladder: effect, a move beyond (breaths/min)": EFFECT_BPM,
    "ladder: effect, least recordings lost or moved": EFFECT_RECORDINGS,
    "ladder: downscale from half size, frames averaged": (LADDER_DOWNSCALE, LADDER_BLOCK),
    "self-test: mean absolute difference, pixel means removed (grey levels)": SELF_TEST_LEVELS,
    "iPhone: downscale, frames averaged": (IPHONE_DOWNSCALE, IPHONE_BLOCK),
}
FITZPATRICK_PAIRS = {1: "1-2", 2: "1-2", 3: "3-4", 4: "3-4", 5: "5-6", 6: "5-6"}
AGE_BANDS = ([0, 40, 60, 120], ["<=40", "41-60", ">60"])
RATE_BANDS = ([0, 8, 11, 20, 24, 99], ["<=8", "9-11", "12-20", "21-24", ">=25"])


# ------------------------------------------------------------------- the store --

def store_splits(store, split_csv=SPLIT_CSV):
    """Each GUID's split, once the store's split attributes match the frozen split file."""
    frozen = pd.read_csv(split_csv).set_index("guid").split
    in_store = pd.Series({guid: str(store[guid].attrs["split"]) for guid in store})
    if not in_store.sort_index().equals(frozen.sort_index()):
        raise SystemExit(f"ERROR: the store's split attributes differ from {Path(split_csv).name}")
    return in_store


def train_guids(store, split_csv=SPLIT_CSV):
    """VV-240's GUIDs, sorted, once the store's split attributes match the frozen split file."""
    in_store = store_splits(store, split_csv)
    return sorted(in_store.index[in_store == "train"])


def held_out_guids(store, split_csv=SPLIT_CSV):
    """VV-60's GUIDs, sorted, once the store's split attributes match the frozen split file."""
    in_store = store_splits(store, split_csv)
    return sorted(in_store.index[in_store == "val"])


def read_frames(group, name):
    """The development stage's frame reader: refuses any recording outside VV-240."""
    if str(group.attrs["split"]) != "train":
        raise RuntimeError(f"{group.name}: held-out frames are read only by NB_P3_40's stage")
    return group[name][:]


def read_held_out(group, name):
    """The held-out stage's frame reader: refuses any recording outside VV-60."""
    if str(group.attrs["split"]) != "val":
        raise RuntimeError(f"{group.name}: the held-out stage reads VV-60 only")
    return group[name][:]


# ---------------------------------------------------------------- the readouts --

def vertical_shift(region):
    """Each frame's vertical offset from the recording's mean region, in pixels: Hann-weighted
    cross-correlation, its peak refined by a parabola through the neighbouring rows."""
    n, h, w = region.shape
    weight = np.outer(np.hanning(h), np.hanning(w))
    mean_region = region.mean(axis=0)
    reference = np.conj(np.fft.rfft2((mean_region - mean_region.mean()) * weight))
    dy = np.zeros(n)
    for i in range(n):
        frame = region[i]
        corr = np.fft.fftshift(np.fft.irfft2(np.fft.rfft2((frame - frame.mean()) * weight) * reference,
                                             s=(h, w)))
        iy, ix = np.unravel_index(int(corr.argmax()), corr.shape)
        iy, ix = min(max(iy, 1), h - 2), min(max(ix, 1), w - 2)
        left, centre, right = corr[iy - 1, ix], corr[iy, ix], corr[iy + 1, ix]
        curvature = left - 2.0 * centre + right
        offset = 0.5 * (left - right) / curvature if abs(curvature) > 1e-12 else 0.0
        dy[i] = iy - h // 2 + (offset if abs(offset) <= 0.5 else 0.0)
    return dy


def shift_rate(dy, fs):
    """Readout B: (rate, share) of the vertical shift's breathing-band series."""
    series = breathing_band(dy, fs)
    return resp_rate(series, fs), peak_share(series, fs)


def read_recording(store, guid, reader=read_frames):
    """One recording's row: what scoring needs from its attributes, and A's and B's readings."""
    group = store[guid]
    a = group.attrs
    frames = reader(group, "frames_full")
    fs, box_ry1 = float(a["full_fps"]), float(a["box_ry1"])
    reading = torso_rate(frames, box_ry1, fs)
    region = torso_region(frames, box_ry1)
    b_rate, b_share = shift_rate(vertical_shift(region), fs) if region is not None else (np.nan, np.nan)
    return dict(guid=guid, resp_ok=bool(a["resp_ok"]), note_resp_bad=bool(a["note_resp_bad"]),
                site_group=str(a["site_group"]), fitzpatrick=int(a["fitzpatrick"]), age=int(a["age"]),
                belt=float(a["rr_belt"]), rr_surrogate=float(a["rr_surrogate"]),
                torso_rows=reading.rows, a_rate=reading.rate_bpm, a_share=reading.share,
                a_component=reading.component, b_rate=b_rate, b_share=b_share)


# ------------------------------------------------------------------ the scoring --

def text_sha256(path):
    """SHA-256 of a text file with its line endings read as LF, so that a CRLF checkout of the
    same commit hashes the same."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def readings_digest(frame):
    """SHA-256 of the readings NB_P3_39 section 11 fixed, as that cell computes it."""
    target = frame[READING_COLUMNS].sort_values("guid").reset_index(drop=True)
    return hashlib.sha256(target.round(6).to_csv(index=False).encode("utf-8")).hexdigest()


def wilson(k, n, z=WILSON_Z):
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(centre - half), float(centre + half)


def gate(scored, col, baseline):
    """One readout's figures over the scored recordings, as NB_P3_39 computed them."""
    base_ok = (scored.belt - baseline).abs() <= WITHIN_BPM
    read = scored[col].notna() & scored.belt.notna()
    ok = read & ((scored[col] - scored.belt).abs() <= WITHIN_BPM)
    n_read, k = int(read.sum()), int(ok.sum())
    share, coverage = (k / n_read if n_read else float("nan")), n_read / len(scored)
    b, c = int((ok & ~base_ok).sum()), int((~ok & base_ok).sum())
    p = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    beats = b > c and p < GATE_P
    outcome = ("goes forward" if beats and share >= GATE_SHARE and coverage >= GATE_COVERAGE
               else "partial signal" if beats and share < GATE_SHARE else "development negative")
    err = scored.loc[read, col] - scored.loc[read, "belt"]
    return dict(scored=len(scored), read=n_read, coverage=coverage, within_2=k, share=share,
                wilson_95=list(wilson(k, n_read)), within_1=int((err.abs() <= 1.0).sum()),
                mean_abs_error=float(err.abs().mean()), bias=float(err.mean()),
                limits_of_agreement=[float(err.mean() - 1.96 * err.std()),
                                     float(err.mean() + 1.96 * err.std())],
                readout_only=b, baseline_only=c, mcnemar_p=p, outcome=outcome)


def held_out_outcome(g):
    """Part 0's three outcomes, from `gate`'s figures over the held-out scored set."""
    ahead = g["readout_only"] > g["baseline_only"] and g["mcnemar_p"] < BAR_P
    if g["read"] and g["wilson_95"][0] >= BAR_LOWER and g["coverage"] >= BAR_COVERAGE and ahead:
        return "confirmed"
    if not g["read"] or g["share"] < BAR_LOWER or not ahead:
        return "not confirmed"
    return "inconclusive"


def print_gate(name, g):
    """One readout's line, as NB_P3_39 printed it."""
    lo, hi = g["wilson_95"]
    loa = g["limits_of_agreement"]
    within1 = g["within_1"] / g["read"] if g["read"] else float("nan")
    print(f"  {name}: read {g['read']} of {g['scored']} ({g['coverage']:.1%}) | within 2 {g['within_2']} "
          f"({g['share']:.1%}, 95% {lo:.1%}-{hi:.1%}) | within 1 {g['within_1']} ({within1:.1%}) | "
          f"mean abs error {g['mean_abs_error']:.2f}, bias {g['bias']:+.2f}, "
          f"limits of agreement {loa[0]:+.2f} to {loa[1]:+.2f}")
    print(f"      McNemar against the baseline: readout only {g['readout_only']}, baseline only "
          f"{g['baseline_only']}, p = {g['mcnemar_p']:.2g} -> {g['outcome']}")


def strata(scored, col):
    """Right over read by Part 0's strata: site group, Fitzpatrick pair, age band, belt rate band."""
    read = scored[col].notna()
    ok = read & ((scored[col] - scored.belt).abs() <= WITHIN_BPM)
    groups = {"site group": (scored.site_group, sorted(scored.site_group.unique())),
              "Fitzpatrick": (scored.fitzpatrick.map(FITZPATRICK_PAIRS), ["1-2", "3-4", "5-6"]),
              "age": (pd.cut(scored.age, AGE_BANDS[0], labels=AGE_BANDS[1]).astype(str), AGE_BANDS[1]),
              "belt rate": (pd.cut(scored.belt.round(), RATE_BANDS[0], labels=RATE_BANDS[1]).astype(str),
                            RATE_BANDS[1])}
    return {label: {key: dict(n=int((values == key).sum()), read=int(read[values == key].sum()),
                              right=int(ok[values == key].sum())) for key in keys}
            for label, (values, keys) in groups.items()}


def concentration_bands(scored, cuts):
    """A's right over read in the five bands cut at VV-240's quintiles of its concentration."""
    read = scored.a_rate.notna() & scored.a_share.notna()
    ok = read & ((scored.a_rate - scored.belt).abs() <= WITHIN_BPM)
    band = pd.Series(np.searchsorted(np.asarray(cuts), scored.a_share.fillna(0.0), side="right"),
                     index=scored.index)
    return [dict(read=int((read & (band == k)).sum()), right=int((ok & (band == k)).sum()))
            for k in range(len(cuts) + 1)]


# ------------------------------------------------------------------ the videos --

def reduced_video(path, downscale, block, keep=()):
    """Decodes a video as NB_P3_27 decoded the raw video - OpenCV, BGR to RGB - and returns
    (frames, samples, n_decoded, fps): every frame reduced `downscale`-fold with INTER_AREA and
    averaged over each `block` consecutive frames, rounded to 8 bits; the full-size RGB frames
    at the indices in `keep`; the number of frames decoded; the container's frame rate."""
    keep = set(keep)
    frames, samples, total, n, fps = [], {}, None, 0, float("nan")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(path)
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if n in keep:
                samples[n] = rgb
            height, width = rgb.shape[:2]
            small = cv2.resize(rgb, (width // downscale, height // downscale),
                               interpolation=cv2.INTER_AREA).astype(np.float32)
            total = small if total is None else total + small
            n += 1
            if n % block == 0:
                frames.append(np.clip(np.rint(total / block), 0, 255).astype(np.uint8))
                total = None
    finally:
        capture.release()
    return (np.asarray(frames) if frames else np.zeros((0, 0, 0, 3), np.uint8)), samples, n, fps


def torso_difference(reduced, stored, box_ry1):
    """The self-test's figure: the mean absolute difference, in grey levels, between two
    [T, H, W, 3] streams over A's torso region, each pixel's mean over the minute removed;
    NaN when their shapes differ or no torso region remains."""
    if reduced.shape != stored.shape:
        return float("nan")
    ours, theirs = torso_region(reduced, box_ry1), torso_region(stored, box_ry1)
    if ours is None:
        return float("nan")
    return float(np.abs((ours - ours.mean(axis=0)) - (theirs - theirs.mean(axis=0))).mean())


def rung_outcome(base, rung):
    """The ladder's rule for one rung: R0's readings (base) against the rung's, both
    {guid: rate}. With no recording read at R0 there is nothing to compare: inconclusive."""
    read = [guid for guid, rate in base.items() if np.isfinite(rate)]
    lost = [guid for guid in read if not np.isfinite(rung.get(guid, np.nan))]
    moves = [abs(rung[guid] - base[guid]) for guid in read if np.isfinite(rung.get(guid, np.nan))]
    hits = len(lost) + sum(move > EFFECT_BPM for move in moves)
    if hits >= EFFECT_RECORDINGS:
        result = "effect"
    elif read and not lost and all(move <= NO_EFFECT_BPM for move in moves):
        result = "no effect"
    else:
        result = "inconclusive"
    return dict(recordings=len(base), read_at_r0=len(read), lost=len(lost),
                moved_over_no_effect=sum(move > NO_EFFECT_BPM for move in moves),
                moved_over_effect=sum(move > EFFECT_BPM for move in moves),
                max_move=float(max(moves)) if moves else None,
                median_move=float(np.median(moves)) if moves else None, outcome=result)


def verify_ladder(ladder_dir=LADDER_DIR, commit=LADDER_COMMIT):
    """Holds the ladder to its manifest before any of it is read: the commit that wrote it,
    and every file's byte count and SHA-256. Returns (manifest, problems)."""
    manifest = json.loads((Path(ladder_dir) / "manifest.json").read_text(encoding="utf-8"))
    problems = []
    if manifest["repository"]["commit"] != commit or manifest["repository"]["dirty"]:
        problems.append(f"the ladder was written from {manifest['repository']}")
    for name, entry in sorted(manifest["files"].items()):
        path = Path(ladder_dir) / name
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            problems.append(f"{name}: not the manifest's {entry['bytes']} bytes")
        elif file_sha256(path) != entry["sha256"]:
            problems.append(f"{name}: SHA-256 differs from the manifest's")
    return manifest, problems


def files_fingerprint(manifest, ladder_dir, iphone_paths):
    """SHA-256 of the ladder's and the iPhone recordings' sorted names and sizes, as NB_P3_40's
    freeze record computes it."""
    files = [(name, (Path(ladder_dir) / name).stat().st_size) for name in sorted(manifest["files"])]
    files += [(f"{p.parent.name}/{p.name}", p.stat().st_size) for p in iphone_paths]
    return hashlib.sha256("".join(f"{n} {s}\n" for n, s in files).encode("utf-8")).hexdigest()


def iphone_reading(path, landmarker):
    """One iPhone recording: the app's face box on its detection samples, its lower edge in
    place of box_ry1, and A on the frame reduced IPHONE_DOWNSCALE-fold and averaged over
    IPHONE_BLOCK frames."""
    from app.live_vitals.capture.session import detection_positions, stable_box
    capture = cv2.VideoCapture(str(path))
    declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    positions = detection_positions(declared)
    frames, samples, n, fps = reduced_video(path, IPHONE_DOWNSCALE, IPHONE_BLOCK, keep=positions)
    box, detections, _ = stable_box([samples[i] for i in positions if i in samples], landmarker)
    row = dict(frames_decoded=n, fps=fps, detections=detections, box_found=box is not None,
               box_ry1=np.nan, rows=0, rate=np.nan, share=np.nan, component=-1)
    if box is not None and len(frames):
        row["box_ry1"] = float(min(max(box[3], 0.0), 1.0))
        reading = torso_rate(frames, row["box_ry1"], fps / IPHONE_BLOCK)
        row.update(rows=reading.rows, rate=reading.rate_bpm, share=reading.share,
                   component=reading.component)
    return row


# ----------------------------------------------------------------- the stages --

def run_dev(args):
    """The development stage: VV-240 read, scored, and checked against NB_P3_39."""
    started = time.perf_counter()
    with h5py.File(args.store, "r") as store:
        guids = train_guids(store)
        baseline = float(np.median([float(store[g].attrs["rr_belt"]) for g in guids
                                    if bool(store[g].attrs["resp_ok"])]))
        chosen = guids[:args.limit] if args.limit else guids
        print(f"store: {args.store}\nstage dev | {len(chosen)} of {len(guids)} VV-240 recordings")
        rows = []
        for i, guid in enumerate(chosen, 1):
            rows.append(read_recording(store, guid))
            if i % 40 == 0 or i == len(chosen):
                print(f"  {i}/{len(chosen)} read, {time.perf_counter() - started:.0f} s", flush=True)
    frame = pd.DataFrame(rows)
    scored = frame[frame.resp_ok]
    figures = {"A pixel components": gate(scored, "a_rate", baseline),
               "B vertical shift, described": gate(scored, "b_rate", baseline)}
    digest = readings_digest(frame)[:16]
    a_counts = {key: figures["A pixel components"][key] for key in NB39_A_COUNTS}
    full = len(chosen) == len(guids)
    reproduces = full and digest == NB39_DIGEST and a_counts == NB39_A_COUNTS

    base_hits = int(((scored.belt - baseline).abs() <= WITHIN_BPM).sum())
    print(f"\nBaseline: the VV-240 resp_ok median, {baseline:.2f} breaths/min, within 2 in "
          f"{base_hits} of {len(scored)}")
    print(f"The gate, over the {len(scored)} resp_ok recordings read here"
          + ("" if full else " (a partial run: the figures decide nothing)"))
    for name, g in figures.items():
        print_gate(name, g)
    chosen_rank = frame.a_component.value_counts().sort_index()
    print(f"\nAll {len(frame)} recordings: A read {int(frame.a_rate.notna().sum())}, B read "
          f"{int(frame.b_rate.notna().sum())} | A's component chosen (0 = largest): "
          + ", ".join(f"{int(k)}: {int(v)}" for k, v in chosen_rank.items()) + f" | digest {digest}")
    if full:
        print(f"NB_P3_39: digest {NB39_DIGEST} {'equal' if digest == NB39_DIGEST else 'DIFFERENT'}; "
              f"A's counts {'equal' if a_counts == NB39_A_COUNTS else 'DIFFERENT'} "
              f"-> {'reproduces NB_P3_39 exactly' if reproduces else 'DOES NOT REPRODUCE NB_P3_39'}")
    else:
        print("A partial run (--limit): the reproduction is not checked.")
    print(f"{time.perf_counter() - started:.0f} s")

    summary = dict(stage="dev", dataset="VitalVideos-WW", split="train", store=str(args.store),
                   recordings=len(frame), limit=args.limit, baseline_bpm=baseline,
                   baseline_within_2=base_hits, readouts=figures,
                   reproduction=dict(checked=full, digest=digest, digest_nb_p3_39=NB39_DIGEST,
                                     a_counts=a_counts, a_counts_nb_p3_39=NB39_A_COUNTS,
                                     reproduces=reproduces),
                   module=dict(file="app/live_vitals/signal/respiration.py",
                               sha256_lf=text_sha256(respiration.__file__)))
    return frame, summary, reproduces or not full


def run_confirm(args):
    """The held-out stage, NB_P3_40's one run: see the module's docstring for its order."""
    started = time.perf_counter()
    print("1. The development readings, at this commit")
    dev_frame, dev_summary, _ = run_dev(argparse.Namespace(store=args.store, limit=None))
    if not dev_summary["reproduction"]["reproduces"]:
        raise SystemExit("ERROR: the development readings do not reproduce NB_P3_39; "
                         "no held-out frame was read")
    baseline = dev_summary["baseline_bpm"]
    cuts = np.quantile(dev_frame.loc[dev_frame.resp_ok, "a_share"], [0.2, 0.4, 0.6, 0.8]).tolist()

    print("\n2. The ladder and the iPhone recordings, before any of them is read")
    manifest, problems = verify_ladder(LADDER_DIR, LADDER_COMMIT)
    raw = sorted(manifest["recordings"])
    iphone_paths = sorted(RAW_SUBSET.glob("*/*.MOV"))
    if sorted(p.parent.name for p in iphone_paths) != raw:
        problems.append(f"{len(iphone_paths)} iPhone recordings for {len(raw)} participants")
    if problems:
        raise SystemExit("ERROR: the ladder does not match its manifest; no held-out frame was "
                         "read:\n  " + "\n  ".join(problems))
    fingerprint = files_fingerprint(manifest, LADDER_DIR, iphone_paths)
    print(f"  {len(manifest['files'])} ladder files match the manifest (bytes and SHA-256) | "
          f"{len(iphone_paths)} iPhone recordings | files fingerprint {fingerprint}")

    print("\n3. VV-60's stored frames")
    with h5py.File(args.store, "r") as store:
        guids = held_out_guids(store)
        if not set(raw) <= set(guids):
            raise SystemExit("ERROR: a raw recording is not in VV-60")
        frame = pd.DataFrame([read_recording(store, guid, reader=read_held_out) for guid in guids])
        stored = {guid: read_held_out(store[guid], "frames_full") for guid in raw}
        boxes = {guid: float(store[guid].attrs["box_ry1"]) for guid in raw}
    scored = frame[frame.resp_ok]
    figures = {"A pixel components": gate(scored, "a_rate", baseline),
               "B vertical shift, described": gate(scored, "b_rate", baseline)}
    a = figures["A pixel components"]
    verdict = held_out_outcome(a)
    base_hits = int(((scored.belt - baseline).abs() <= WITHIN_BPM).sum())
    print(f"  {len(frame)} recordings, {len(scored)} resp_ok | baseline {baseline:.2f} breaths/min, "
          f"within 2 in {base_hits} of {len(scored)}")
    for name, g in figures.items():
        print_gate(name, g)
    print(f"  -> A: {verdict.upper()} (bar: Wilson lower bound >= {BAR_LOWER:.0%}, coverage >= "
          f"{BAR_COVERAGE:.0%}, McNemar p < {BAR_P} with A ahead)")
    both = scored.a_rate.notna() & scored.b_rate.notna()
    a_ok = scored.a_rate.notna() & ((scored.a_rate - scored.belt).abs() <= WITHIN_BPM)
    b_ok = scored.b_rate.notna() & ((scored.b_rate - scored.belt).abs() <= WITHIN_BPM)
    unconfirmed = frame[~frame.resp_ok]
    with_surrogate = unconfirmed[unconfirmed.rr_surrogate.notna()]
    held_out = dict(
        recordings=len(frame), scored=len(scored), baseline_bpm=baseline, baseline_within_2=base_hits,
        readouts=figures, outcome=verdict,
        a_against_b=dict(both_read=int(both.sum()),
                         within_2_of_each_other=int(((scored.a_rate - scored.b_rate).abs()[both]
                                                     <= WITHIN_BPM).sum()),
                         both_right=int((a_ok & b_ok).sum()), a_only=int((a_ok & ~b_ok).sum()),
                         b_only=int((~a_ok & b_ok).sum()), neither=int((both & ~a_ok & ~b_ok).sum())),
        strata=dict(a=strata(scored, "a_rate"), b=strata(scored, "b_rate")),
        concentration=dict(cuts=cuts, bands=concentration_bands(scored, cuts)),
        without_confirmed_belt=dict(
            recordings=len(unconfirmed), with_surrogate=len(with_surrogate),
            a_read=int(with_surrogate.a_rate.notna().sum()),
            a_within_2_of_surrogate=int(((with_surrogate.a_rate - with_surrogate.rr_surrogate).abs()
                                         <= WITHIN_BPM).sum())))

    print("\n4. The ladder: the self-test on R0 first")
    ladder_ok = [guid for guid in raw if bool(frame.set_index("guid").loc[guid, "resp_ok"])]
    stored_rate = frame.set_index("guid").a_rate
    readings = {rung: {} for rung in RUNGS}
    ladder_rows, self_test = [], {}
    for guid in raw:
        frames, _, n, _ = reduced_video(LADDER_DIR / guid / RUNG_FILES["R0"], LADDER_DOWNSCALE, LADDER_BLOCK)
        self_test[guid] = torso_difference(frames, stored[guid], boxes[guid])
        reading = torso_rate(frames, boxes[guid], LADDER_FS / LADDER_BLOCK)
        readings["R0"][guid] = reading.rate_bpm
        ladder_rows.append(dict(guid=guid, rung="R0", resp_ok=guid in ladder_ok, frames_decoded=n,
                                rate=reading.rate_bpm, share=reading.share, component=reading.component,
                                self_test=self_test[guid], stored_rate=stored_rate[guid]))
    differences = np.array(list(self_test.values()))
    passed = bool(np.all(np.isfinite(differences) & (differences <= SELF_TEST_LEVELS)))
    r0_moves = [abs(readings["R0"][g] - stored_rate[g]) for g in raw
                if np.isfinite(readings["R0"][g]) and np.isfinite(stored_rate[g])]
    print(f"  self-test: {int(np.sum(np.isfinite(differences) & (differences <= SELF_TEST_LEVELS)))} of "
          f"{len(raw)} within {SELF_TEST_LEVELS} grey level (median {np.nanmedian(differences):.3f}, "
          f"max {np.nanmax(differences):.3f}) -> {'passes' if passed else 'FAILS: the ladder is not assessed'}")
    if r0_moves:
        print(f"  R0 against the stored frames: median |difference| {np.median(r0_moves):.2f}, max "
              f"{max(r0_moves):.2f} breaths/min, {len(r0_moves)} recordings read at both")
    rungs = {}
    if passed:
        for rung in RUNGS[1:]:
            for guid in raw:
                frames, _, n, _ = reduced_video(LADDER_DIR / guid / RUNG_FILES[rung], LADDER_DOWNSCALE,
                                                LADDER_BLOCK)
                reading = torso_rate(frames, boxes[guid], LADDER_FS / LADDER_BLOCK)
                readings[rung][guid] = reading.rate_bpm
                ladder_rows.append(dict(guid=guid, rung=rung, resp_ok=guid in ladder_ok, frames_decoded=n,
                                        rate=reading.rate_bpm, share=reading.share,
                                        component=reading.component, self_test=np.nan,
                                        stored_rate=stored_rate[guid]))
            rungs[rung] = rung_outcome({g: readings["R0"][g] for g in ladder_ok},
                                       {g: readings[rung][g] for g in ladder_ok})
            others = [g for g in raw if g not in ladder_ok]
            rungs[rung]["described_without_belt"] = rung_outcome({g: readings["R0"][g] for g in others},
                                                                  {g: readings[rung][g] for g in others})
            largest = rungs[rung]["max_move"]
            print(f"  {rung} against R0, {len(ladder_ok)} resp_ok: lost {rungs[rung]['lost']}, moved over "
                  f"{NO_EFFECT_BPM:g} {rungs[rung]['moved_over_no_effect']}, over {EFFECT_BPM:g} "
                  f"{rungs[rung]['moved_over_effect']}, largest move "
                  f"{'-' if largest is None else f'{largest:.2f}'} -> "
                  f"{rungs[rung]['outcome']}{' (DECIDES)' if rung == DECIDING_RUNG else ''}")
    ladder_outcome = rungs[DECIDING_RUNG]["outcome"] if passed else "not assessed"
    print(f"  -> the ladder: {ladder_outcome.upper()}")
    ladder = dict(recordings=len(raw), resp_ok=len(ladder_ok),
                  self_test=dict(passed=passed, within=int(np.sum(np.isfinite(differences)
                                                                  & (differences <= SELF_TEST_LEVELS))),
                                 median=float(np.nanmedian(differences)), max=float(np.nanmax(differences))),
                  r0_against_stored=dict(read_at_both=len(r0_moves), median=float(np.median(r0_moves)),
                                         max=float(max(r0_moves))) if r0_moves else None,
                  rungs=rungs, outcome=ladder_outcome)

    print("\n5. The iPhone recordings, described")
    from app.live_vitals.preprocess.face_box import make_landmarker
    landmarker = make_landmarker()
    iphone_rows = []
    for path in iphone_paths:
        guid = path.parent.name
        row = iphone_reading(path, landmarker)
        iphone_rows.append(dict(guid=guid, resp_ok=guid in ladder_ok,
                                belt=float(frame.set_index("guid").loc[guid, "belt"]),
                                stored_rate=stored_rate[guid], **row))
    phone = pd.DataFrame(iphone_rows)
    read = phone.rate.notna()
    confirmed = phone.resp_ok
    iphone = dict(recordings=len(phone), box_found=int(phone.box_found.sum()), read=int(read.sum()),
                  rows=dict(median=float(phone.rows.median()), min=int(phone.rows.min()),
                            max=int(phone.rows.max())),
                  resp_ok=int(confirmed.sum()), resp_ok_read=int((read & confirmed).sum()),
                  resp_ok_within_2_of_belt=int((read & confirmed & ((phone.rate - phone.belt).abs()
                                                                    <= WITHIN_BPM)).sum()),
                  within_2_of_stored=int((read & ((phone.rate - phone.stored_rate).abs()
                                                  <= WITHIN_BPM)).sum()))
    print(f"  {iphone['recordings']} recordings: box found {iphone['box_found']}, read {iphone['read']}, "
          f"rows below the box median {iphone['rows']['median']:.0f} ({iphone['rows']['min']}-"
          f"{iphone['rows']['max']}) | resp_ok {iphone['resp_ok']}: read {iphone['resp_ok_read']}, within 2 "
          f"of the belt {iphone['resp_ok_within_2_of_belt']} | within 2 of the stored frames' reading "
          f"{iphone['within_2_of_stored']}")
    print(f"\n{time.perf_counter() - started:.0f} s")

    summary = dict(stage="confirm", dataset="VitalVideos-WW", split="val", store=str(args.store),
                   settings=SETTINGS, limit=None,
                   development=dict(reproduces=True, digest=dev_summary["reproduction"]["digest"],
                                    a_counts=dev_summary["reproduction"]["a_counts"]),
                   module=dev_summary["module"],
                   files=dict(ladder_commit=manifest["repository"]["commit"],
                              ladder_files=len(manifest["files"]), iphone_files=len(iphone_paths),
                              fingerprint=fingerprint),
                   held_out=held_out, ladder=ladder, iphone=iphone)
    tables = dict(recordings=frame, ladder=pd.DataFrame(ladder_rows), iphone=phone)
    return tables, summary


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("dev", "confirm"), default="dev",
                        help="dev: VV-240 against NB_P3_39; confirm: NB_P3_40's one held-out run")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--limit", type=int, default=None,
                        help="dev only: read only the first N recordings (smoke test; nothing is checked)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if args.stage == "confirm":
        if args.limit:
            raise SystemExit("ERROR: the held-out stage runs whole, once; --limit is for the dev stage")
        if (args.out_dir / "summary_confirm.json").exists():
            raise SystemExit(f"ERROR: {args.out_dir / 'summary_confirm.json'} exists: the held-out "
                             "stage runs once")
    run_provenance = provenance(None)  # before any output exists, so git sees the tree as run
    run_provenance.pop("weights")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "confirm":
        tables, summary = run_confirm(args)
        summary["provenance"] = run_provenance
        for name, table in tables.items():
            table.to_csv(args.out_dir / f"{name}_confirm.csv", index=False)
        (args.out_dir / "summary_confirm.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"outputs: {args.out_dir}")
        return
    frame, summary, passed = run_dev(args)
    summary["provenance"] = run_provenance
    suffix = subset_suffix(("limit", args.limit or None, None))
    frame.to_csv(args.out_dir / f"recordings_{args.stage}{suffix}.csv", index=False)
    (args.out_dir / f"summary_{args.stage}{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"outputs: {args.out_dir}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()