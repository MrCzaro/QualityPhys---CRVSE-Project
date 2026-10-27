"""Beat timing and heart-rate variability from video, on the Rice iPPG (HRVCam) store.

The earlier HRV measurement ran on MCD-rPPG, whose reference was never recorded for HRV: a
100 Hz contact PPG with a resting RMSSD of 20-27 ms, where 46 ms of camera jitter leaves
nothing to measure. Rice was acquired for this question. Its reference is a finger pulse
oximeter worn throughout, its beats are timed to well under a millisecond after sub-sample
refinement, resting RMSSD is 40 ms in the median subject, and the protocol includes paced
deep breathing, which drives a large slow swing that any beat-timing method should recover.

What this scores, per estimator, against the store's own reference beats:

- **beat agreement**: how many reference beats have a camera beat within 150 ms, how far off
  they are, and how many beats the camera finds per reference beat;
- **HRV**: SDNN and RMSSD from the camera's own kept intervals against the reference's over
  the same seconds, over the whole recording and over the 60 s an app records;
- **the jitter each excess implies**: independent jitter of SD s inflates SDNN by a factor
  sqrt(1 + 2s²/SDNN²) and RMSSD by sqrt(1 + 6s²/RMSSD²), so the excess over the reference
  gives s. Under independent jitter the two estimates agree, which is the check that the
  excess is jitter and not something else.

Three things are taken from the store rather than recomputed, so that the camera and the
reference are compared on the same terms NB_P3_33 settled:

- the reference's beats (`ref_beat_ms`) and which intervals its rules kept (`ref_beat_kept`),
  from the steepest point of the upstroke refined between samples - not the systolic peak,
  which moves between the systolic and diastolic waves in seven recordings;
- each recording's camera-to-finger offset (`offset_ms`);
- which references carry HRV at all (`has_hrv`): 58 of 70, the rest set aside for a doubted
  clock or too few intervals, and reported separately here.

The camera's beats use the same fiducial as the reference. POS is inverted first, because it
correlates negatively with the stored label in all 70 recordings; PhysNet is not, because it
was trained to match that label. A reconstruction is built by taking the central stride of
each window, never by overlap-adding: separate windows carry no shared phase, and the
estimator's own `waveform` concatenates overlapping windows, so its time axis runs long.

Three caveats hold for every figure, and are written into every summary:

- **the app's framing gate refuses all 70 recordings**, so these are forced readings;
- talking is reported on its own, never pooled;
- the camera-to-finger offset mixes both devices' latencies with the pulse's travel, which
  does not matter for variability: a constant offset cannot change an interval.

Region fusion - timing three skin patches separately and fusing them, which NB_P3_33 part 3
found halves the jitter - is not here yet. The store carries `region_rgb` for it.

Usage:
    python -m app.live_vitals.scripts.check_rice_hrv [--estimator NAME ...] [--crop NAME]
        [--activities stationary talking ...] [--store PATH] [--device cpu|cuda]
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
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals.config import CLIP_LEN
from app.live_vitals.estimators import registry
from app.live_vitals.models.architectures.physnet import PhysNet
from app.live_vitals.scripts.bench_outputs import provenance, subset_suffix, unique
from app.live_vitals.scripts.check_hrv_from_video import (ANCHOR_HALF_WIDTH_S, MATCH_TOLERANCE_S,
                                                          align_offset, continuous_bvp,
                                                          nearest_gaps, pooled_offset)
from app.live_vitals.scripts.check_hrv_sampling_ceiling import (bandpass, dominant_period,
                                                                refine)
from app.live_vitals.scripts.check_rice_ippg import (ACTIVITIES, CROPS, DATASET, DEFAULT_CROP,
                                                     DEFAULT_STORE)
from app.live_vitals.scripts.check_ubfc_phys import _plain, _write_csv
from app.live_vitals.signal.spectral import pos, rgb_trace
from scipy.signal import find_peaks

DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "rice_hrv_eval"
BEAT_BAND_HZ = (0.7, 3.5)
BEAT_SPACING = 0.55        # the least a beat may follow the last, as a share of the period
HR_MIN_BPM, HR_MAX_BPM = 35.0, 180.0
IBI_LOCAL_TOL = 0.25       # an interval this far from the median around it is an artefact
SPAN_S = 60.0              # the app's capture, scored at the start and the end of a recording
MIN_SPAN_INTERVALS = 30    # a span is scored only where both sides keep this many intervals
INVERTED = ("hr_spectral",)   # estimators whose trace runs opposite to the stored label
CONFOUNDS = [
    "the app's framing gate refuses all 70 recordings: these are forced readings",
    "talking is reported on its own and never pooled",
    "the camera-to-finger offset mixes both devices' latencies with the pulse's travel; a "
    "constant offset cannot change an interval, so it does not affect variability",
    "the reference is a 60 Hz 7-bit finger waveform; after sub-sample refinement its own "
    "timing adds under 0.05 ms to an 18 ms RMSSD",
]


# ------------------------------------------------------------------- the beats --

def steepest_rises(trace, fps):
    """Sub-sample times, in seconds from the first sample, of the steepest upstrokes.

    The same fiducial, spacing and slope rule the reference's beats were found with: the
    spacing follows the trace's own dominant period, and a candidate must climb at least 40%
    as steeply as the median of the candidates around it, which drops ripples on the
    downstroke without a threshold that a breathing-driven amplitude swing would defeat.
    """
    filtered = bandpass(trace, fps, *BEAT_BAND_HZ)
    slope = np.gradient(filtered)
    period = dominant_period(filtered, fps, *BEAT_BAND_HZ)
    idx, _ = find_peaks(slope, distance=max(1, int(BEAT_SPACING * period * fps)))
    if idx.size < 3:
        return np.array([])
    local = np.array([np.median(slope[idx][max(0, k - 4):k + 5]) for k in range(idx.size)])
    idx = idx[slope[idx] >= 0.4 * local]
    return np.array([refine(slope, int(i)) for i in idx]) / fps


def kept_intervals(beats):
    """Intervals between beats, and which of them the artefact rules keep.

    The two physiological rules the reference uses: inside 35-180 bpm, and within 25% of the
    median of the seven intervals around it. Nothing here needs the reference, so an app can
    apply them to its own beats, which is the point.

    The reference drops a third class this cannot see - an interval bordering a lost or
    clipped oximeter sample, which only the device knows about. Measured over all 70
    references, that is the difference between keeping 96.7% of intervals and keeping 100%.
    The reference side therefore keeps the store's own flags and the camera side uses these
    rules: each is judged by what it can actually apply, and where the oximeter lost samples
    the two sides cover slightly different beats inside a span.
    """
    beats = np.asarray(beats, dtype=float)
    if beats.size < 3:
        return np.array([]), np.array([]), np.array([], dtype=bool)
    ibi = np.diff(beats)
    in_range = (ibi > 60.0 / HR_MAX_BPM) & (ibi < 60.0 / HR_MIN_BPM)
    local = np.array([np.median(ibi[max(0, k - 3):k + 4][in_range[max(0, k - 3):k + 4]])
                      if in_range[max(0, k - 3):k + 4].any() else np.nan
                      for k in range(ibi.size)])
    consistent = np.abs(ibi - local) <= IBI_LOCAL_TOL * local
    return beats[1:], ibi, in_range & consistent


def span_hrv(ends, ibi, kept, t0, t1):
    """SDNN, RMSSD (ms) and the interval count over the kept intervals ending in [t0, t1].

    A successive difference is taken only where both of its intervals are kept, so one
    rejected interval does not manufacture a large difference out of its neighbours.
    """
    inside = kept & (ends >= t0) & (ends <= t1)
    pair = inside[1:] & inside[:-1]
    values, diffs = ibi[inside], np.diff(ibi)[pair]
    if values.size < 3 or diffs.size < 3:
        return float("nan"), float("nan"), int(inside.sum())
    return (float(np.std(values, ddof=1) * 1000.0),
            float(np.sqrt(np.mean(diffs ** 2)) * 1000.0), int(inside.sum()))


def implied_jitter(camera_ms, reference_ms, factor):
    """The independent jitter (ms) an excess implies: sqrt((camera^2 - reference^2) / factor).

    factor is 2 for SDNN and 6 for RMSSD. Returns NaN where the camera is not the larger,
    which is not an error - it means that span's excess is not jitter.
    """
    if not (np.isfinite(camera_ms) and np.isfinite(reference_ms)) or camera_ms <= reference_ms:
        return float("nan")
    return float(np.sqrt((camera_ms ** 2 - reference_ms ** 2) / factor))


# ----------------------------------------------------------------- the camera --

def load_physnet(estimator, device):
    """The estimator's own weights in a bare PhysNet, for a continuous reconstruction.

    The estimator reads a rate per window and its `waveform` concatenates those windows, so
    it cannot be used for beat timing; the reconstruction has to be built from the model
    itself. Loading it here the same way the estimator does keeps the weights identical.
    """
    state = torch.load(estimator.checkpoint, map_location=device, weights_only=True)
    model = PhysNet(frames=CLIP_LEN)
    model.load_state_dict(state)
    return model.eval().to(device)


def camera_beats(estimator, model, frames, fps, device):
    """Camera beat times in seconds from the first frame, for one estimator and recording."""
    if model is None:
        trace = pos(rgb_trace(frames), fps)
        first = 0
    else:
        trace, first = continuous_bvp(model, frames, device)
    if estimator.name in INVERTED:
        trace = -np.asarray(trace, dtype=float)
    beats = steepest_rises(np.asarray(trace, dtype=float), fps)
    return beats + first / fps, len(trace)


def evaluate(store_path, estimators, crop, activities, limit, device):
    """Beats, spans and recording rows for every estimator over one crop of the store."""
    models = {e.name: (None if e.name == "hr_spectral" else load_physnet(e, device))
              for e in estimators}
    found = {e.name: {} for e in estimators}
    reference = {}
    with h5py.File(store_path, "r") as store:
        names = [n for n in sorted(store.keys())
                 if str(store[n].attrs["activity"]) in activities]
        if limit:
            names = names[:limit]
        t0 = time.perf_counter()
        for i, name in enumerate(names, 1):
            group = store[name]
            attrs = {k: _plain(v) for k, v in group.attrs.items()}
            fps = float(attrs["fps"])
            frames = group[crop][()]
            # The reference's beats are on the frames' clock; a camera beat at frame time t
            # belongs with a reference beat at t plus the recording's offset.
            ref_beats = group["ref_beat_ms"][()] / 1000.0 - attrs["offset_ms"] / 1000.0
            reference[name] = dict(
                attrs=attrs, fps=fps, beats=ref_beats,
                kept=group["ref_beat_kept"][()].astype(bool),
                span_s=float(attrs["n_frames"]) / fps)
            for estimator in estimators:
                beats, n = camera_beats(estimator, models[estimator.name], frames, fps, device)
                found[estimator.name][name] = beats
            if i % 10 == 0 or i == len(names):
                print(f"  {i}/{len(names)} recordings | {(time.perf_counter() - t0) / 60:.1f} min",
                      flush=True)

    rows = {e.name: {"recordings": [], "spans": []} for e in estimators}
    for estimator in estimators:
        name = estimator.name
        # Each subject's recordings share a camera-to-sensor geometry, so their offsets are
        # found together and each recording is then aligned within a quarter second of that.
        subjects = {}
        for recording in reference:
            subjects.setdefault(recording.split("_")[0], []).append(recording)
        anchors = {s: pooled_offset([(found[name][r], reference[r]["beats"]) for r in recs])
                   for s, recs in subjects.items()}
        for recording, ref in reference.items():
            rows[name]["recordings"].append(
                _recording_row(estimator, recording, ref, found[name][recording],
                               anchors[recording.split("_")[0]], rows[name]["spans"]))
    return rows, anchors


def _recording_row(estimator, recording, ref, beats, anchor, span_rows):
    """One recording's beat agreement, and its spans appended to span_rows."""
    attrs = ref["attrs"]
    offset, gap, matched, margin = align_offset(beats, ref["beats"],
                                                limit_s=ANCHOR_HALF_WIDTH_S, centre_s=anchor)
    camera = np.asarray(beats, dtype=float) + offset
    gaps = nearest_gaps(camera, ref["beats"]) if camera.size >= 3 else np.array([np.nan])
    cam_ends, cam_ibi, cam_kept = kept_intervals(camera)
    ref_ends, ref_ibi, _ = kept_intervals(ref["beats"])
    ref_kept = ref["kept"] if ref["kept"].size == ref_ibi.size else np.ones(ref_ibi.size, bool)

    spans = {"whole": (0.0, ref["span_s"]),
             "first 60 s": (0.0, SPAN_S),
             "last 60 s": (ref["span_s"] - SPAN_S, ref["span_s"])}
    for label, (t0, t1) in spans.items():
        cam_sdnn, cam_rmssd, cam_n = span_hrv(cam_ends, cam_ibi, cam_kept, t0, t1)
        ref_sdnn, ref_rmssd, ref_n = span_hrv(ref_ends, ref_ibi, ref_kept, t0, t1)
        scored = (min(cam_n, ref_n) >= MIN_SPAN_INTERVALS and np.isfinite(cam_rmssd)
                  and np.isfinite(ref_rmssd))
        span_rows.append(dict(
            estimator=estimator.name, recording=recording, subject=attrs["subject_id"],
            activity=attrs["activity"], has_hrv=bool(attrs["has_hrv"]), span=label,
            scored=scored, n_camera=cam_n, n_reference=ref_n,
            sdnn_camera=cam_sdnn, sdnn_reference=ref_sdnn,
            rmssd_camera=cam_rmssd, rmssd_reference=ref_rmssd,
            sdnn_error=cam_sdnn / ref_sdnn - 1.0 if ref_sdnn else float("nan"),
            rmssd_error=cam_rmssd / ref_rmssd - 1.0 if ref_rmssd else float("nan"),
            jitter_sdnn_ms=implied_jitter(cam_sdnn, ref_sdnn, 2.0),
            jitter_rmssd_ms=implied_jitter(cam_rmssd, ref_rmssd, 6.0)))

    return dict(
        estimator=estimator.name, recording=recording, subject=attrs["subject_id"],
        activity=attrs["activity"], has_hrv=bool(attrs["has_hrv"]),
        fps=ref["fps"], n_camera_beats=int(camera.size), n_reference_beats=int(ref["beats"].size),
        beats_per_reference=float(camera.size / max(ref["beats"].size, 1)),
        offset_ms=float(offset * 1000.0), offset_anchor_ms=float(anchor * 1000.0),
        offset_gap_ms=float(gap * 1000.0), offset_margin_ms=float(margin),
        matched=float(matched), timing_ms=float(np.median(gaps[gaps <= MATCH_TOLERANCE_S]) * 1000.0)
        if np.any(gaps <= MATCH_TOLERANCE_S) else float("nan"),
        kept_share=float(cam_kept.mean()) if cam_kept.size else float("nan"),
        store_pos_bvp_r=float(attrs["qc_pos_bvp_r"]),
        ref_rmssd_store_ms=float(attrs["ref_rmssd_ms"]))


# ------------------------------------------------------------------- summary --

def _stats(values):
    """Median, interquartile range and count of the finite values."""
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)),
                q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)))


def _error_stats(values):
    """How a relative error sits: median, median absolute, and the shares within 10% and 20%."""
    v = np.array([x for x in values if np.isfinite(x)], dtype=float)
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)),
                median_abs=float(np.median(np.abs(v))),
                within_10=float(np.mean(np.abs(v) <= 0.10)),
                within_20=float(np.mean(np.abs(v) <= 0.20)))


def summarise(recordings, spans, hrv_only=True):
    """Beat agreement over the recordings, and HRV over the scored spans."""
    recs = [r for r in recordings if r["has_hrv"]] if hrv_only else list(recordings)
    scored = [s for s in spans if s["scored"] and (s["has_hrv"] or not hrv_only)]
    out = dict(n_recordings=len(recs),
               beats=dict(matched=_stats([r["matched"] for r in recs]),
                          timing_ms=_stats([r["timing_ms"] for r in recs]),
                          beats_per_reference=_stats([r["beats_per_reference"] for r in recs]),
                          offset_ms=_stats([r["offset_ms"] for r in recs]),
                          offset_gap_ms=_stats([r["offset_gap_ms"] for r in recs])))
    for label in ("whole", "first 60 s", "last 60 s"):
        rows = [s for s in scored if s["span"] == label]
        out[label] = dict(
            n_spans=len(rows),
            sdnn=_error_stats([s["sdnn_error"] for s in rows]),
            rmssd=_error_stats([s["rmssd_error"] for s in rows]),
            jitter_sdnn_ms=_stats([s["jitter_sdnn_ms"] for s in rows]),
            jitter_rmssd_ms=_stats([s["jitter_rmssd_ms"] for s in rows]),
            reference_rmssd_ms=_stats([s["rmssd_reference"] for s in rows]))
    return out


def build_summary(estimator, recordings, spans, args, crop):
    """The JSON summary for one estimator on one crop."""
    return dict(
        estimator=estimator.name, dataset=DATASET, crop=crop, store=str(args.store),
        device=args.device, activities=list(args.activities), limit=args.limit,
        confounds=CONFOUNDS,
        settings=dict(clip_len=CLIP_LEN, beat_band_hz=list(BEAT_BAND_HZ),
                      beat_spacing=BEAT_SPACING, ibi_local_tol=IBI_LOCAL_TOL,
                      span_s=SPAN_S, min_span_intervals=MIN_SPAN_INTERVALS,
                      match_tolerance_s=MATCH_TOLERANCE_S,
                      inverted=estimator.name in INVERTED),
        hrv_recordings=sum(1 for r in recordings if r["has_hrv"]),
        set_aside=[r["recording"] for r in recordings if not r["has_hrv"]],
        all_recordings=summarise(recordings, spans, hrv_only=False),
        hrv_set=summarise(recordings, spans),
        by_activity={a: summarise([r for r in recordings if r["activity"] == a],
                                  [s for s in spans if s["activity"] == a])
                     for a in args.activities},
    )


def print_summary(summary):
    """The headline tables for one estimator on one crop."""
    print(f"\n=== {summary['estimator']} | crop {summary['crop']} ===")
    hrv = summary["hrv_set"]
    beats = hrv["beats"]
    print(f"{hrv['n_recordings']} recordings carry HRV, {len(summary['set_aside'])} set aside | "
          f"beats found per reference beat {beats['beats_per_reference'].get('median', float('nan')):.2f}")
    print(f"matched within {summary['settings']['match_tolerance_s'] * 1000:.0f} ms: "
          f"{beats['matched'].get('median', float('nan')):.0%} in the median recording | timing "
          f"{beats['timing_ms'].get('median', float('nan')):.1f} ms "
          f"({beats['timing_ms'].get('q1', float('nan')):.1f}-"
          f"{beats['timing_ms'].get('q3', float('nan')):.1f}) | offset "
          f"{beats['offset_ms'].get('median', float('nan')):+.0f} ms")
    header = (f"\n{'span':11} {'spans':>6} {'SDNN err':>9} {'|SDNN|':>7} {'w10':>5} "
              f"{'RMSSD err':>10} {'|RMSSD|':>8} {'w10':>5} {'jitter ms':>10} {'ref RMSSD':>10}")
    print(header)
    for label in ("whole", "first 60 s", "last 60 s"):
        print(_span_line(label, hrv[label]))
    print("\nby activity, whole recordings:")
    print(header)
    for activity, sets in summary["by_activity"].items():
        if sets["whole"]["n_spans"]:
            print(_span_line(activity, sets["whole"]))


def _span_line(label, m):
    """One row of the HRV tables."""
    sdnn, rmssd = m["sdnn"], m["rmssd"]
    if not sdnn.get("n"):
        return f"{label:11} {m['n_spans']:6d} " + " ".join(["-".rjust(w) for w in (9, 7, 5, 10, 8, 5, 10, 10)])
    return (f"{label:11} {m['n_spans']:6d} {sdnn['median']:+9.1%} {sdnn['median_abs']:7.1%} "
            f"{sdnn['within_10']:5.0%} {rmssd['median']:+10.1%} {rmssd['median_abs']:8.1%} "
            f"{rmssd['within_10']:5.0%} {m['jitter_rmssd_ms'].get('median', float('nan')):10.1f} "
            f"{m['reference_rmssd_ms'].get('median', float('nan')):10.1f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", action="append", choices=registry.available(),
                        help="registered estimator name; repeat for several (default: all)")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--crop", choices=CROPS, default=DEFAULT_CROP)
    parser.add_argument("--activities", nargs="+", choices=ACTIVITIES, default=list(ACTIVITIES))
    parser.add_argument("--device", default="cpu", help="torch device for learned estimators")
    parser.add_argument("--limit", type=int, default=None,
                        help="evaluate only the first N recordings (smoke test)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    names = unique(args.estimator or registry.available())
    args.activities = [a for a in ACTIVITIES if a in args.activities]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"store: {args.store}\nestimators: {', '.join(names)} | crop: {args.crop} | "
          f"activities: {' '.join(args.activities)} | device: {args.device}"
          + (f" | limit {args.limit}" if args.limit else ""))
    estimators = []
    for name in names:
        estimator = registry.get_estimator(name, device=args.device) \
            if name != "hr_spectral" else registry.get_estimator(name)
        if not estimator.is_available():
            raise SystemExit(f"ERROR: checkpoint missing for {name}")
        estimators.append(estimator)
    trained = [e.name for e in estimators
               if any(dataset == DATASET for dataset, _ in e.trained_on)]
    if trained:
        raise SystemExit(f"ERROR: {', '.join(trained)} trained on {DATASET}; this store is an "
                         f"evaluation set only")

    rows, _ = evaluate(args.store, estimators, args.crop, args.activities, args.limit,
                       args.device)
    suffix = f"_crop-{args.crop}" + subset_suffix(
        ("activities-", "-".join(args.activities), "-".join(ACTIVITIES)),
        ("limit", args.limit or None, None))
    for estimator in estimators:
        name = estimator.name
        recordings, spans = rows[name]["recordings"], rows[name]["spans"]
        summary = build_summary(estimator, recordings, spans, args, args.crop)
        summary["provenance"] = provenance(estimator)
        _write_csv(args.out_dir / f"recordings_{name}{suffix}.csv", recordings)
        _write_csv(args.out_dir / f"spans_{name}{suffix}.csv", spans)
        (args.out_dir / f"summary_{name}{suffix}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print_summary(summary)

    print(f"\noutputs: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())