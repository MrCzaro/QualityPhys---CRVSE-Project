"""Can beat-to-beat timing be recovered from video, not just an average heart rate?

Test 1 established that 30 fps is not the obstacle: resampling a contact PPG to
30 Hz costs 2% of RMSSD as long as beat positions are interpolated sub-sample.
This asks the next question, which is whether the *reconstructed* pulse carries
beat timing at all, or only the right average rate.

Two reconstructions are compared against the frame-aligned contact PPG that
MCD-rPPG ships with each recording:

  physnet   the neural model, inferred in 160-frame windows
  pos       the classical projection, computed over the whole capture at once

POS is included because it has a structural advantage here that it does not have
for heart rate. It produces one continuous signal, whereas the model's windows are
independent and carry no shared phase, so the model's trace has a seam every
WINDOW_STRIDE samples. Average rate survives seams; beat timing may not.

Heart rate needs only the dominant frequency to be right. Beat timing needs each
individual peak in the right place, which is a far stronger requirement, so a
model that reads heart rate to 1 bpm can still be useless here. The metrics are
therefore about beats, not rates: how many reference beats have a predicted beat
near them, how far off it is, and whether the resulting RMSSD survives.

Usage:
    python -m app.live_vitals.scripts.check_hrv_from_video
        [--cache-dir DIR] [--recordings 4087_before ...] [--no-cache]

Runs the model over every cached recording, which takes a few minutes the first
time; reconstructions are then cached beside the crops. Nothing else is written.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals.config import (CLIP_LEN, WINDOW_STRIDE, PHYSNET_CKPT,
                                    SPECTRAL_ROI_FRACTION)
from app.live_vitals.models.architectures.physnet import PhysNet
from app.live_vitals.preprocess.frames import clip_to_tensor
from app.live_vitals.signal.spectral import METHODS, rgb_trace
from app.live_vitals.scripts.check_hrv_sampling_ceiling import (
    bandpass, beat_times, hrv)

DEFAULT_CACHE = Path(r"D:\QualityPhys\demo_data\mcd_rppg_subset\.crop_cache")
MATCH_TOLERANCE_S = 0.15
# Still below half a beat interval, and for the same reason: a wider window lets the alignment
# slip by a whole beat. The median distance between paired beats discriminates better than
# counting matches, but it does not make a wide search safe - searched over +/-1 s these
# recordings produce subject anchors of up to 972 ms, which MCD's frame-aligned PPG (documented
# within about twenty frames) cannot have. Widen this only for a dataset whose own offset is
# known to be larger, and then only with several recordings of differing heart rate pooled.
MAX_LAG_S = 0.4
MIN_SLIP_MARGIN_S = 0.25    # how far a rival alignment must sit before it counts as a rival
ANCHOR_HALF_WIDTH_S = 0.25  # each recording is aligned this far either side of its subject's


def continuous_bvp(model, clip, device="cpu"):
    """Returns (trace, first_index) tiling the capture from independent windows.

    Windows carry no shared phase or scale, so overlap-adding them can cancel a
    genuine pulse. Taking the central WINDOW_STRIDE samples of each window instead
    covers the capture exactly once, with every sample drawn from a window where it
    sits far from the edges, and z-scoring each piece before joining stops the
    amplitude step at a seam from registering as a beat.
    """
    margin = (CLIP_LEN - WINDOW_STRIDE) // 2
    starts = list(range(0, len(clip) - CLIP_LEN + 1, WINDOW_STRIDE))
    pieces = []
    for start in starts:
        tensor = clip_to_tensor(clip[start:start + CLIP_LEN]).to(device)
        with torch.no_grad():
            bvp = model(tensor)[0].cpu().numpy()
        piece = bvp[margin:margin + WINDOW_STRIDE]
        spread = piece.std()
        pieces.append((piece - piece.mean()) / (spread if spread > 1e-9 else 1.0))
    return np.concatenate(pieces), starts[0] + margin


def nearest_gaps(predicted, reference):
    """Distance from each reference beat to the nearest predicted beat, in seconds."""
    index = np.clip(np.searchsorted(predicted, reference), 1, len(predicted) - 1)
    left = np.abs(predicted[index - 1] - reference)
    right = np.abs(predicted[index] - reference)
    return np.minimum(left, right)


def offset_scores(pairs, limit_s=MAX_LAG_S, step_s=0.0025, centre_s=0.0):
    """Returns (offsets, median distance at each) over one or more (predicted, reference) pairs.

    Pooling pairs that share an offset - the recordings of one subject, whose camera and
    sensor sat at the same remove from each other - is what makes a whole-beat slip
    impossible to mistake for the truth: the recordings' heart rates differ, so a slip
    fits each of them at a different shift and only the true offset fits them together.
    """
    offsets = np.arange(centre_s - limit_s, centre_s + limit_s + step_s / 2, step_s)
    usable = [(np.asarray(p), np.asarray(r)) for p, r in pairs
              if len(p) >= 3 and len(r) >= 3]
    if not usable:
        return offsets, np.full(offsets.size, np.nan)
    medians = []
    for offset in offsets:
        gaps = np.concatenate([nearest_gaps(p + offset, r) for p, r in usable])
        medians.append(float(np.median(gaps)))
    return offsets, np.array(medians)


def align_offset(predicted, reference, limit_s=ANCHOR_HALF_WIDTH_S, step_s=0.0025,
                 centre_s=0.0):
    """Returns (offset, median distance, matched share, margin) for one recording.

    The offset is the one whose reference beats sit closest to a predicted beat, measured
    by the **median distance** between paired beats rather than by how many pairs fall
    within a tolerance. Counting matches cannot see a whole-beat slip: most beat-to-beat
    changes are far below the tolerance, so a shift of one beat keeps nearly every pair
    inside it - 88% against 100% on one Rice recording - while the median distance
    separates the two several-fold (55 against 13 ms there).

    `margin` is how much worse, in milliseconds, the best alignment more than
    MIN_SLIP_MARGIN_S away fits. A few milliseconds means a second alignment fits about as
    well and the offset should not be trusted; the reference, which aligns with itself
    exactly, shows a margin the size of its own timing spread.

    The search is narrow by default, and is meant to be centred on an offset found for
    several recordings at once. Neither criterion is safe over a wide window on one short
    span: measured on 30 s spans of the Rice recordings, counting matches and the median
    distance choose alignments a whole beat apart in 3 of 15, and on one of those the
    median distance prefers the slipped one, because 30 s of a talking recording genuinely
    fits a shifted alignment better. What rules a slip out is that the recordings of one
    subject share an offset while their heart rates differ, which is what `pooled_offset`
    uses.

    A constant offset is not itself an error for HRV: interval variability is unaffected
    by when the recording is considered to start. Only jitter matters, which is what the
    timing spread reports.
    """
    offsets, medians = offset_scores([(predicted, reference)], limit_s, step_s, centre_s)
    if np.all(np.isnan(medians)):
        return 0.0, float("nan"), 0.0, float("nan")
    best = int(np.nanargmin(medians))
    away = np.abs(offsets - offsets[best]) > MIN_SLIP_MARGIN_S
    alternative = np.nanmin(medians[away]) if away.any() else np.nan
    shifted = np.asarray(predicted) + offsets[best]
    matched = float(np.mean(nearest_gaps(shifted, np.asarray(reference)) <= MATCH_TOLERANCE_S))
    return (float(offsets[best]), float(medians[best]), matched,
            float((alternative - medians[best]) * 1000.0))


def pooled_offset(pairs, limit_s=MAX_LAG_S, step_s=0.0025):  # the wide search lives here
    """The offset that fits every pair at once, for recordings that share one."""
    offsets, medians = offset_scores(pairs, limit_s, step_s)
    if np.all(np.isnan(medians)):
        return 0.0
    return float(offsets[int(np.nanargmin(medians))])


def match_beats(predicted, reference, tolerance=MATCH_TOLERANCE_S):
    """Pairs each reference beat with the nearest predicted beat within tolerance.

    Returns (n_matched, n_missed, n_extra, timing_errors). A prediction may only
    be spent once, so a burst of spurious peaks cannot inflate the match rate.
    """
    used = np.zeros(len(predicted), dtype=bool)
    errors = []
    for t in reference:
        if not len(predicted):
            break
        gaps = np.abs(predicted - t)
        gaps[used] = np.inf
        index = int(np.argmin(gaps))
        if gaps[index] <= tolerance:
            used[index] = True
            errors.append(predicted[index] - t)
    return len(errors), len(reference) - len(errors), int((~used).sum()), np.array(errors)


def is_recording_cache(path):
    """True for a crop cache, which is the only kind this script reads.

    The folder also holds this script's own reconstructions and the window caches another
    script writes, and more kinds may appear. Asking what a file contains, rather than
    excluding the suffixes known today, keeps the selection correct as that grows.
    """
    try:
        with np.load(path) as archive:
            return {"clip", "ppg", "fps"} <= set(archive.files)
    except (OSError, ValueError):
        return False

    
def reconstructions(path, cache=True):
    """Returns (traces, reference, fps) for one cached recording."""
    store = path.with_name(path.stem + "_bvp.npz")
    with np.load(path) as z:
        clip, ppg, fps = z["clip"], z["ppg"], float(z["fps"])

    if cache and store.exists():
        with np.load(store) as z:
            return ({"physnet": z["physnet"], "pos": z["pos"]},
                    z["reference"], float(z["fps"]))

    state = torch.load(PHYSNET_CKPT, map_location="cpu", weights_only=True)
    model = PhysNet(frames=CLIP_LEN)
    model.load_state_dict(state)
    model.eval()

    net, first = continuous_bvp(model, clip)
    # POS is continuous over the whole capture; it is cut to the same span as the
    # model's trace so both are scored on identical seconds of the recording.
    pos = METHODS["pos"](rgb_trace(clip, SPECTRAL_ROI_FRACTION), fps)[first:first + len(net)]
    reference = ppg[first:first + len(net)]
    traces = {"physnet": net, "pos": np.asarray(pos, dtype=float)}
    if cache:
        np.savez_compressed(store, physnet=traces["physnet"], pos=traces["pos"],
                            reference=reference, fps=fps)
    return traces, reference, fps


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--recordings", nargs="+", default=None)
    parser.add_argument("--no-cache", action="store_true",
                        help="recompute reconstructions instead of reusing them")
    args = parser.parse_args()

    paths = sorted(p for p in args.cache_dir.glob("*.npz") if is_recording_cache(p))

    if args.recordings:
        wanted = set(args.recordings)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        print(f"No cached recordings in {args.cache_dir}.")
        return 2

    print("Heart rate needs one frequency to be right. Beat timing needs every")
    print("peak in the right place. This measures the second thing.\n")

    # Every recording is read once, and its beats found once, before any alignment: the
    # offset of a recording is searched around one found for its subject's recordings
    # together, which is what a whole-beat slip cannot survive.
    loaded = {}
    for path in paths:
        traces, reference, fps = reconstructions(path, cache=not args.no_cache)
        beats = {name: beat_times(trace, fps, sub_sample=True)
                 for name, trace in list(traces.items()) + [("reference", reference)]}
        loaded[path.stem] = {"fps": fps, "beats": beats,
                             "reference": beats["reference"]}

    methods = [name for name in loaded[paths[0].stem]["beats"]]
    subjects = {}
    for stem in loaded:
        subjects.setdefault(stem.split("_")[0], []).append(stem)
    anchors = {(subject, method):
               pooled_offset([(loaded[s]["beats"][method], loaded[s]["reference"])
                              for s in stems])
               for subject, stems in subjects.items() for method in methods}
    print("anchor offsets, each subject's recordings searched together "
          f"(+/-{MAX_LAG_S:g} s), then each recording within +/-{ANCHOR_HALF_WIDTH_S:g} s of it:")
    for (subject, method), anchor in sorted(anchors.items()):
        print(f"  {subject:>6} {method:>9} {anchor * 1000:+8.0f} ms")
    print()
    print(f"{'recording':>14} {'method':>9} {'lag s':>7} {'matched':>8} "
          f"{'missed':>7} {'extra':>6} {'|dt| ms':>8} {'gap ms':>7} {'margin':>7} "
          f"{'RMSSD pred':>11} {'ref':>7} {'err':>7}")

    summary = {}
    for path in paths:
        entry = loaded[path.stem]
        fps, ref_beats = entry["fps"], entry["reference"]
        ref_rmssd, ref_sdnn, _, _ = hrv(ref_beats)

        # The reference is scored against itself as a control. Anything other than
        # a perfect row there means the harness is broken, not the model.
        for name in methods:
            anchor = anchors[(path.stem.split("_")[0], name)]
            offset, gap, _, margin = align_offset(entry["beats"][name], ref_beats,
                                                  limit_s=ANCHOR_HALF_WIDTH_S,
                                                  centre_s=anchor)
            pred_beats = entry["beats"][name] + offset
            matched, missed, extra, errors = match_beats(pred_beats, ref_beats)
            rmssd, sdnn, _, _ = hrv(pred_beats)
            rate = matched / max(len(ref_beats), 1)
            timing = np.median(np.abs(errors)) * 1000.0 if len(errors) else float("nan")
            summary.setdefault(name, []).append(
                (rate, timing, rmssd - ref_rmssd, sdnn - ref_sdnn, gap * 1000.0, margin))
            print(f"{path.stem:>14} {name:>9} {offset:7.2f} "
                  f"{rate:7.0%} {missed:7d} {extra:6d} {timing:8.1f} {gap * 1000:7.1f} "
                  f"{margin:7.0f} {rmssd:11.1f} {ref_rmssd:7.1f} {rmssd - ref_rmssd:+7.1f}")
        print()

    print(f"{'method':>9} {'match rate':>11} {'|dt| ms':>9} "
          f"{'RMSSD err':>10} {'SDNN err':>9} {'gap ms':>8} {'margin':>7}")
    for name, rows in summary.items():
        rate = np.nanmedian([r[0] for r in rows])
        timing = np.nanmedian([r[1] for r in rows])
        rmssd = np.nanmedian(np.abs([r[2] for r in rows]))
        sdnn = np.nanmedian(np.abs([r[3] for r in rows]))
        gap = np.nanmedian([r[4] for r in rows])
        margin = np.nanmin([r[5] for r in rows])
        print(f"{name:>9} {rate:10.0%} {timing:9.1f} {rmssd:10.1f} {sdnn:9.1f} "
              f"{gap:8.1f} {margin:7.0f}")
    print("\n`gap` is the median distance between paired beats at the chosen offset, and")
    print("`margin` how much worse, in ms, the best alignment 0.25 s away fits. A few ms means")
    print("a second alignment fits as well, and that offset is not to be trusted.")

    print("\nA high match rate with small |dt| means the waveform carries real beat")
    print("timing and HRV is worth pursuing. A high match rate with large RMSSD")
    print("error means the beats are found but jittered, which inflates RMSSD")
    print("specifically -- the quantity is built from successive differences, so it")
    print("amplifies exactly the noise a smooth average would hide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())