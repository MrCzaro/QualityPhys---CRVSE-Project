"""Does anatomical ROI selection beat the centre square once resolution is not the limit?

`SPECTRAL_ROI_FRACTION = 0.35` averages the central 35% of the 72x72 crop, which lands
on the eye line and excludes the forehead and cheeks -- the two regions the rPPG
literature normally picks. Tested inside the crop, that centre square wins anyway, and
the ordering of every skin-only region tracks its pixel count: 625 px for the centre,
189 for the forehead. Averaging N pixels suppresses sensor and quantisation noise by
sqrt(N) while the pulse is a fraction of a percent of mean intensity, so at 72x72 the
region is chosen by area rather than by anatomy.

That is a statement about resolution, not about anatomy, and it is testable. At
640x480 a forehead ROI is thousands of pixels rather than 189. This extracts the same
regions from the **full-resolution frame, before the 72x72 downsample**, and asks
whether the ordering reverses.

The arms separate the two effects deliberately:

  crop centre 35%       current behaviour, measured from the 72x72 crop  (control)
  fullres centre 35%    the same region at full resolution               (resolution only)
  fullres forehead      landmark-placed, above the brows                 (anatomy)
  fullres cheeks        landmark-placed, below the eyes, lateral of nose (anatomy)
  fullres forehead+cheeks   both, POS per region then averaged           (anatomy, more area)

Only the spectral estimator could ever use this: PhysNet is bound by contract to the
72x72 crop. Regions are placed from the median landmark set over the same frames the
face box is built from, so they are frozen for the capture exactly as the box is.

Usage:
    python -m app.live_vitals.scripts.check_roi_resolution [--recordings NAME ...]
        [--ubfc-only] [--mcd-only] [--no-cache]

Decoding is the cost; ROI traces are cached as <name>_roi.npz so re-runs are seconds.
No frames are written and no image is rendered.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mediapipe as mp

from app.live_vitals import config
from app.live_vitals.preprocess.face_box import make_landmarker, square_padded_box
from app.live_vitals.preprocess.frames import crop_resize
from app.live_vitals.signal.hr import hr_from_bvp
from app.live_vitals.signal.spectral import bandpass
from app.live_vitals.scripts.check_contract_parity import load_reference

UBFC_DIR = Path(r"D:\QualityPhys\demo_data")
MCD_DIR = Path(r"D:\QualityPhys\demo_data\mcd_rppg_subset")
UBFC_SUBJECTS = [11, 13, 24, 25, 34, 35, 42, 47]
CACHE = Path(r"D:\QualityPhys\demo_data\.roi_cache")

# MediaPipe FaceMesh anchor points. Regions are built from distances between these
# rather than from fixed indices bounding each region, so they scale with the face.
NOSE_TIP, FOREHEAD_TOP, CHIN = 1, 10, 152
EYE_OUTER_R, EYE_OUTER_L = 33, 263
MOUTH_R, MOUTH_L = 61, 291
FACE_EDGE_R, FACE_EDGE_L = 234, 454


def median_landmarks(frames_rgb, landmarker):
    """Returns the per-point median landmark array over several frames, or None.

    Taking a median over the same frames the face box is built from freezes the
    regions for the capture, matching how the box itself is frozen. A per-frame
    placement would track the face better but would no longer be the same contract.
    """
    stacks = []
    for frame in frames_rgb:
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(frame))
        result = landmarker.detect(image)
        if result.face_landmarks:
            pts = result.face_landmarks[0]
            stacks.append(np.array([[p.x, p.y] for p in pts], dtype=np.float64))
    if not stacks:
        return None
    return np.median(np.stack(stacks), axis=0)


def anatomical_boxes(lm, width, height):
    """Returns named pixel rectangles for the forehead and both cheeks.

    Everything is expressed as a fraction of a measured facial distance, so the
    regions follow face size and position instead of assuming a framing.
    """
    px = lm[:, 0] * width
    py = lm[:, 1] * height
    eye_y = (py[EYE_OUTER_R] + py[EYE_OUTER_L]) / 2.0
    top_y = py[FOREHEAD_TOP]
    mouth_y = (py[MOUTH_R] + py[MOUTH_L]) / 2.0
    nose_x = px[NOSE_TIP]
    eye_span = abs(px[EYE_OUTER_L] - px[EYE_OUTER_R])

    # Forehead: the middle half of the gap between hairline and eyes, inset
    # horizontally so temples and hair stay out.
    brow_gap = eye_y - top_y
    forehead = (nose_x - 0.45 * eye_span, top_y + 0.15 * brow_gap,
                nose_x + 0.45 * eye_span, eye_y - 0.15 * brow_gap)

    # Cheeks: below the eyes, above the mouth, between the face edge and the nose.
    cheek_top = eye_y + 0.18 * (mouth_y - eye_y)
    cheek_bottom = eye_y + 0.80 * (mouth_y - eye_y)
    right = (px[FACE_EDGE_R] + 0.18 * (nose_x - px[FACE_EDGE_R]), cheek_top,
             nose_x - 0.22 * (nose_x - px[FACE_EDGE_R]), cheek_bottom)
    left = (nose_x + 0.22 * (px[FACE_EDGE_L] - nose_x), cheek_top,
            px[FACE_EDGE_L] - 0.18 * (px[FACE_EDGE_L] - nose_x), cheek_bottom)
    return dict(forehead=forehead, cheek_r=right, cheek_l=left)


def shrink(box, fraction):
    """Returns the central `fraction` of a rectangle, about its centre."""
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) * fraction / 2.0, (y1 - y0) * fraction / 2.0
    return (cx - hw, cy - hh, cx + hw, cy + hh)


def patch_mean(frame, box):
    """Mean RGB inside a pixel rectangle, clamped to the frame."""
    h, w = frame.shape[:2]
    x0 = max(0, int(box[0])); y0 = max(0, int(box[1]))
    x1 = min(w, max(int(box[2]), x0 + 1)); y1 = min(h, max(int(box[3]), y0 + 1))
    return frame[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


def extract(video_path, cache_path, use_cache=True):
    """Returns (traces dict of [T,3] arrays, fps, region pixel areas)."""
    if use_cache and cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as z:
            return ({k: z[k] for k in z.files if k not in ("fps", "areas")},
                    float(z["fps"]), z["areas"].item())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total <= 0:
        probe = cv2.VideoCapture(str(video_path))
        n_total = sum(1 for _ in iter(probe.grab, False))
        probe.release()

    sample_at = sorted(set(np.linspace(0, max(n_total - 1, 0),
                                       config.N_DETECT_FRAMES).astype(int)))
    samples = []
    for index in sample_at:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if ok:
            samples.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    landmarker = make_landmarker()
    lm = median_landmarks(samples, landmarker)
    if lm is None:
        raise RuntimeError(f"no face detected in {video_path}")
    box = square_padded_box([(lm[:, 0].min(), lm[:, 1].min(),
                              lm[:, 0].max(), lm[:, 1].max())])

    height, width = samples[0].shape[:2]
    crop_px = (box[0] * width, box[1] * height, box[2] * width, box[3] * height)
    regions = anatomical_boxes(lm, width, height)
    regions["fullres_centre"] = shrink(crop_px, config.SPECTRAL_ROI_FRACTION)
    areas = {k: int(max(1, (v[2] - v[0])) * max(1, (v[3] - v[1])))
             for k, v in regions.items()}

    names = list(regions)
    rows = {k: [] for k in names}
    rows["crop_centre"] = []
    inset = int(config.TARGET_SIZE * (1 - config.SPECTRAL_ROI_FRACTION) / 2)
    # Both centre arms cover the same source pixels. INTER_AREA is itself an area
    # average, so a 72x72 crop pixel already carries the mean of several source
    # pixels and the noise suppression largely survives the downsample. Counting
    # the crop arm as 676 px would have overstated the resize's cost by ~6x.
    areas["crop_centre"] = areas["fullres_centre"]

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for name in names:
            rows[name].append(patch_mean(rgb, regions[name]))
        crop = crop_resize(rgb, box)
        if crop is None:
            rows["crop_centre"].append(np.full(3, np.nan))
        else:
            centre = crop[inset:config.TARGET_SIZE - inset,
                          inset:config.TARGET_SIZE - inset]
            rows["crop_centre"].append(centre.reshape(-1, 3).mean(axis=0))
    cap.release()

    traces = {k: np.asarray(v, dtype=np.float64) for k, v in rows.items()}
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, fps=fps, areas=np.array(areas, dtype=object),
                            **traces)
    return traces, fps, areas


def pos_signal(rgb_stack, fps, window_seconds=config.SPECTRAL_WINDOW_SECONDS):
    """POS over one or more regions: computed per region, then averaged.

    `rgb_stack` is [T, n_region, 3]. Combining after projection rather than before
    is how both the Phase-2 code and the MCD-rPPG paper handle multiple ROIs.
    """
    n_frames, n_region = rgb_stack.shape[0], rgb_stack.shape[1]
    length = int(round(window_seconds * float(fps)))
    if length < 8 or n_frames < length:
        return np.zeros(n_frames)
    signals = []
    for r in range(n_region):
        rgb, pulse = rgb_stack[:, r, :], np.zeros(n_frames)
        for start in range(0, n_frames - length + 1):
            block = rgb[start:start + length]
            mean = block.mean(axis=0)
            if np.any(mean <= 1e-9) or not np.all(np.isfinite(mean)):
                continue
            cn = block / mean
            s1 = cn[:, 1] - cn[:, 2]
            s2 = cn[:, 1] + cn[:, 2] - 2.0 * cn[:, 0]
            sd2 = s2.std()
            h = s1 + (s1.std() / sd2) * s2 if sd2 > 1e-12 else s1
            sd = h.std()
            if sd > 1e-12:
                pulse[start:start + length] += (h - h.mean()) / sd
        sd = pulse.std()
        signals.append(pulse / sd if sd > 1e-12 else pulse)
    return bandpass(np.mean(signals, axis=0), fps)


def score(signal, reference, fps):
    """Window MAE against the reference readout, and the kept fraction."""
    starts = list(range(0, len(signal) - config.CLIP_LEN + 1, config.WINDOW_STRIDE))
    got, conf, ref = [], [], []
    for i in starts:
        r = hr_from_bvp(signal[i:i + config.CLIP_LEN], fps)
        got.append(r["hr_bpm"]); conf.append(r["confidence"])
        ref.append(hr_from_bvp(reference[i:i + config.CLIP_LEN], fps)["hr_bpm"])
    got = np.asarray(got); conf = np.asarray(conf); ref = np.asarray(ref)
    ok = np.isfinite(got) & np.isfinite(ref)
    mae = float(np.mean(np.abs(got[ok] - ref[ok]))) if ok.any() else float("nan")
    return mae, float(np.mean(conf >= config.MIN_CONFIDENCE))


ARMS = {
    "crop centre 35%": ["crop_centre"],
    "fullres centre 35%": ["fullres_centre"],
    "fullres forehead": ["forehead"],
    "fullres cheeks": ["cheek_r", "cheek_l"],
    "fullres forehead+cheeks": ["forehead", "cheek_r", "cheek_l"],
}


def mcd_fps(name):
    """Frame rate from the PPG clock; the container lies on three recordings."""
    s, step = name.split("_")
    rows = sum(1 for l in open(MCD_DIR / s / f"{s}_FullHDwebcam_{step}.txt") if l.strip())
    stamps = [l.strip().partition("  ")[2].strip()
              for l in open(MCD_DIR / s / f"{s}_{step}.PW") if l.strip()]
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    span = (datetime.strptime(stamps[-1], fmt)
            - datetime.strptime(stamps[0], fmt)).total_seconds()
    return rows / span


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recordings", nargs="+", default=None)
    parser.add_argument("--ubfc-only", action="store_true")
    parser.add_argument("--mcd-only", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    jobs = []
    if not args.mcd_only:
        for s in UBFC_SUBJECTS:
            jobs.append(("UBFC", f"subject{s}", UBFC_DIR / f"subject{s}" / "vid.avi"))
    if not args.ubfc_only:
        for s in ["4087", "4874", "5130", "6137", "8584", "9092"]:
            for step in ("before", "after"):
                jobs.append(("MCD", f"{s}_{step}",
                             MCD_DIR / s / f"{s}_FullHDwebcam_{step}.avi"))
    if args.recordings:
        want = set(args.recordings)
        jobs = [j for j in jobs if j[1] in want]

    print("Region choice at 72x72 is decided by pixel count. This asks whether")
    print("anatomy wins once the regions are taken at full resolution.\n")

    results, areas_seen = {}, {}
    for corpus, name, video in jobs:
        if not video.exists():
            print(f"  missing {video}, skipped"); continue
        traces, fps, areas = extract(video, CACHE / f"{name}_roi.npz",
                                     use_cache=not args.no_cache)
        if corpus == "MCD":
            fps = mcd_fps(name)
            ppg = np.array([float(l.split()[0]) for l in
                            open(MCD_DIR / name.split("_")[0] /
                                 f"{name.split('_')[0]}_FullHDwebcam_{name.split('_')[1]}.txt")
                            if l.strip()])
            n = min(len(ppg), len(traces["crop_centre"]))
            reference = ppg[:n]
        else:
            n = len(traces["crop_centre"])
            reference, _ = load_reference(UBFC_DIR / name / "ground_truth.txt",
                                          np.arange(n) / fps)
        for key, value in areas.items():
            areas_seen.setdefault(key, []).append(value)
            
        for arm, keys in ARMS.items():
            stack = np.stack([traces[k][:n] for k in keys], axis=1)
            mae, kept = score(pos_signal(stack, fps), reference[:n], fps)
            results.setdefault((corpus, arm), []).append((mae, kept))
        print(f"  {name} done ({fps:.2f} fps)", flush=True)

    print(f"\n{'arm':>26} {'mean px':>12} | {'UBFC MAE':>9} {'kept':>6} "
          f"| {'MCD MAE':>8} {'kept':>6}")
    for arm, keys in ARMS.items():
        px = int(sum(np.mean(areas_seen.get(k, [0])) for k in keys))
        cells = []
        for corpus in ("UBFC", "MCD"):
            rows = results.get((corpus, arm))
            if not rows:
                cells.append("      --     --"); continue
            mae = np.nanmean([r[0] for r in rows])
            kept = np.nanmean([r[1] for r in rows])
            cells.append(f"{mae:9.2f} {kept:6.0%}")
        print(f"{arm:>26} {px:12,d} | {cells[0]} | {cells[1]}")

    print("\nThe two centre rows cover the SAME source pixels and differ only in whether")
    print("the average passes through the 72x72 resize first, so the gap between them is")
    print("what the resize costs. The anatomical rows then test region choice at their")
    print("true areas, which is the question the in-crop sweep could not answer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())