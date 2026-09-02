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
MAX_LAG_S = 0.4   # below half a beat interval; see align_offset


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


def align_offset(predicted, reference, limit_s=MAX_LAG_S, step_s=0.005):
    """Returns the constant time offset that best pairs predicted beats to reference.

    The offset is searched on the beat times themselves rather than by
    cross-correlating the waveforms. A pulse is quasi-periodic, so waveform
    correlation peaks again at every multiple of the beat interval and the search
    settles on an arbitrary one; counting matched beats has a single maximum.

    The search is capped below half a beat interval for the same reason. A larger
    window would let the alignment slip by a whole beat, which scores just as well
    and is silently wrong. MCD-rPPG documents its own PPG-to-video offset as within
    about twenty frames, comfortably inside this.

    A constant offset is not itself an error for HRV: interval variability is
    unaffected by when the recording is considered to start. Only jitter matters,
    which is what the timing spread reports.
    """
    if not len(predicted) or not len(reference):
        return 0.0, 0
    # Every offset inside the match tolerance pairs the same beats, so counting
    # matches alone leaves a plateau the width of that tolerance and the search
    # settles at its edge. Ties are broken on total timing error, which has one
    # minimum, so the reported jitter is the signal's and not the search's.
    best = (0.0, -1, np.inf)
    for offset in np.arange(-limit_s, limit_s + step_s, step_s):
        shifted = predicted + offset
        matched, error = 0, 0.0
        for t in reference:
            gap = float(np.min(np.abs(shifted - t)))
            if gap <= MATCH_TOLERANCE_S:
                matched += 1
                error += gap
        if matched > best[1] or (matched == best[1] and error < best[2]):
            best = (float(offset), matched, error)
    return best[0], best[1]


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

    paths = sorted(p for p in args.cache_dir.glob("*.npz")
                   if not p.stem.endswith("_bvp"))
    if args.recordings:
        wanted = set(args.recordings)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        print(f"No cached recordings in {args.cache_dir}.")
        return 2

    print("Heart rate needs one frequency to be right. Beat timing needs every")
    print("peak in the right place. This measures the second thing.\n")
    print(f"{'recording':>14} {'method':>8} {'lag s':>7} {'matched':>8} "
          f"{'missed':>7} {'extra':>6} {'|dt| ms':>8} "
          f"{'RMSSD pred':>11} {'ref':>7} {'err':>7}")

    summary = {}
    for path in paths:
        traces, reference, fps = reconstructions(path, cache=not args.no_cache)
        ref_beats = beat_times(reference, fps, sub_sample=True)
        ref_rmssd, ref_sdnn, _, _ = hrv(ref_beats)

        # The reference is scored against itself as a control. Anything other than
        # a perfect row there means the harness is broken, not the model.
        for name, trace in list(traces.items()) + [("reference", reference)]:
            pred_beats = beat_times(trace, fps, sub_sample=True)
            offset, _ = align_offset(pred_beats, ref_beats)
            pred_beats = pred_beats + offset
            lag = offset * fps
            matched, missed, extra, errors = match_beats(pred_beats, ref_beats)
            rmssd, sdnn, _, _ = hrv(pred_beats)
            rate = matched / max(len(ref_beats), 1)
            timing = np.median(np.abs(errors)) * 1000.0 if len(errors) else float("nan")
            summary.setdefault(name, []).append(
                (rate, timing, rmssd - ref_rmssd, sdnn - ref_sdnn))
            print(f"{path.stem:>14} {name:>8} {lag / fps:7.2f} "
                  f"{rate:7.0%} {missed:7d} {extra:6d} {timing:8.1f} "
                  f"{rmssd:11.1f} {ref_rmssd:7.1f} {rmssd - ref_rmssd:+7.1f}")
        print()

    print(f"{'method':>8} {'match rate':>11} {'|dt| ms':>9} "
          f"{'RMSSD err':>10} {'SDNN err':>9}")
    for name, rows in summary.items():
        rate = np.nanmedian([r[0] for r in rows])
        timing = np.nanmedian([r[1] for r in rows])
        rmssd = np.nanmedian(np.abs([r[2] for r in rows]))
        sdnn = np.nanmedian(np.abs([r[3] for r in rows]))
        print(f"{name:>8} {rate:10.0%} {timing:9.1f} {rmssd:10.1f} {sdnn:9.1f}")

    print("\nA high match rate with small |dt| means the waveform carries real beat")
    print("timing and HRV is worth pursuing. A high match rate with large RMSSD")
    print("error means the beats are found but jittered, which inflates RMSSD")
    print("specifically -- the quantity is built from successive differences, so it")
    print("amplifies exactly the noise a smooth average would hide.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())