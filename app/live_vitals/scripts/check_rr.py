"""Respiratory rate from the torso's motion on VitalVideos-WW: the development stage.

Runs readout A (`signal.respiration.torso_rate`, the primary readout NB_P3_39 chose at its
development gate) on every recording of VitalVideos' training split (VV-240), with readout B,
the torso's vertical shift, described beside it and deciding nothing, and checks that both
reproduce NB_P3_39 exactly:

- **The readings.** The digest of A's and B's whole-minute readings on all 240 training
  recordings (torso rows; A's rate, concentration and component; B's rate and
  concentration; rounded to six decimals, sorted by GUID) must equal NB_P3_39 section 11's.
- **The gate's counts.** Over the 208 recordings whose belt an independent surrogate
  confirms (`resp_ok`), A's counts must equal section 3's: read, within 2 and within 1
  breaths/min of the belt, and the two discordant counts against the baseline.

The reference is the waist belt as NB_P3_27 read it (`rr_belt`). The baseline gives every
recording the median belt rate of VV-240's `resp_ok` recordings. Agreement is the share within
2 breaths/min among the recordings read, beside coverage; the exact McNemar test against the
baseline counts an unread recording as a miss.

Only the training split is read. The held-out split's frames are read once, by the stage
NB_P3_40's Part 0 fixes, which this script does not have yet: a held-out recording reaching
the frame reader stops the run. B lives here rather than in `signal.respiration` because it
only describes.

Usage:
    python -m app.live_vitals.scripts.check_rr [--stage dev] [--store PATH] [--limit N]
        [--out-dir DIR]

A full run takes a few minutes on the CPU and exits non-zero unless it reproduces NB_P3_39.
No frames are rendered or written. The store is read-only. Not a medical device.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.stats import binomtest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals.scripts.bench_outputs import provenance, subset_suffix
from app.live_vitals.signal import respiration
from app.live_vitals.signal.respiration import (breathing_band, peak_share, resp_rate, torso_rate,
                                                torso_region)

DEFAULT_STORE = Path(r"D:\QualityPhys\phase 3 datasets\vitalvideos_phase3_rc_none.h5")
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "rr_eval"
SPLIT_CSV = _REPO_ROOT / "Data" / "vitalvideos_split.csv"  # GUID and split, frozen by NB_P3_27
WITHIN_BPM = 2.0
GATE_SHARE, GATE_COVERAGE, GATE_P = 0.82, 0.80, 0.01  # NB_P3_39's development gate
# What NB_P3_39 printed: section 11's digest of the readings, section 3's counts for A.
NB39_DIGEST = "f75095c4a475f6d8"
NB39_A_COUNTS = dict(scored=208, read=208, within_2=197, within_1=185, readout_only=128,
                     baseline_only=2)
READING_COLUMNS = ["guid", "torso_rows", "a_rate", "a_share", "a_component", "b_rate", "b_share"]


def train_guids(store, split_csv=SPLIT_CSV):
    """VV-240's GUIDs, sorted, once the store's split attributes match the frozen split file."""
    frozen = pd.read_csv(split_csv).set_index("guid").split
    in_store = pd.Series({guid: str(store[guid].attrs["split"]) for guid in store})
    if not in_store.sort_index().equals(frozen.sort_index()):
        raise SystemExit(f"ERROR: the store's split attributes differ from {Path(split_csv).name}")
    return sorted(in_store.index[in_store == "train"])


def read_frames(group, name):
    """The only route by which this script reads frames: refuses any recording outside VV-240."""
    if str(group.attrs["split"]) != "train":
        raise RuntimeError(f"{group.name}: held-out frames are read only by NB_P3_40's stage")
    return group[name][:]


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


def read_recording(store, guid):
    """One recording's row: what scoring needs from its attributes, and A's and B's readings."""
    group = store[guid]
    a = group.attrs
    frames = read_frames(group, "frames_full")
    fs, box_ry1 = float(a["full_fps"]), float(a["box_ry1"])
    reading = torso_rate(frames, box_ry1, fs)
    region = torso_region(frames, box_ry1)
    b_rate, b_share = shift_rate(vertical_shift(region), fs) if region is not None else (np.nan, np.nan)
    return dict(guid=guid, resp_ok=bool(a["resp_ok"]), note_resp_bad=bool(a["note_resp_bad"]),
                site_group=str(a["site_group"]), fitzpatrick=int(a["fitzpatrick"]), age=int(a["age"]),
                belt=float(a["rr_belt"]), rr_surrogate=float(a["rr_surrogate"]),
                torso_rows=reading.rows, a_rate=reading.rate_bpm, a_share=reading.share,
                a_component=reading.component, b_rate=b_rate, b_share=b_share)


def text_sha256(path):
    """SHA-256 of a text file with its line endings read as LF, so that a CRLF checkout of the
    same commit hashes the same."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def readings_digest(frame):
    """SHA-256 of the readings NB_P3_39 section 11 fixed, as that cell computes it."""
    target = frame[READING_COLUMNS].sort_values("guid").reset_index(drop=True)
    return hashlib.sha256(target.round(6).to_csv(index=False).encode("utf-8")).hexdigest()


def wilson(k, n, z=1.959964):
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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("dev",), default="dev",
                        help="dev: VV-240 against NB_P3_39 (the held-out stage comes with NB_P3_40)")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--limit", type=int, default=None,
                        help="read only the first N recordings (smoke test; nothing is checked)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    run_provenance = provenance(None)  # before any output exists, so git sees the tree as run
    run_provenance.pop("weights")
    frame, summary, passed = run_dev(args)
    summary["provenance"] = run_provenance
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = subset_suffix(("limit", args.limit or None, None))
    frame.to_csv(args.out_dir / f"recordings_{args.stage}{suffix}.csv", index=False)
    (args.out_dir / f"summary_{args.stage}{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(f"outputs: {args.out_dir}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()