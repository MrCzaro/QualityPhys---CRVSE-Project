"""Contract parity and reference-accuracy check for the Phase-3 live_vitals pipeline.

Verifies that the app preprocessing reproduces the NB_P3_05/06 reference implementation
bit-exactly, reports capture-quality diagnostics, and optionally runs the HR estimator
against a UBFC-rPPG ground-truth trace.

Accepts either a video file or a UBFC-rPPG subject directory containing `vid.avi` and
`ground_truth.txt`.

Usage:
    python -m app.live_vitals.scripts.check_contract_parity <video|subject_dir>
        [--infer] [--reference] [--frames N]

Examples:
    python -m app.live_vitals.scripts.check_contract_parity G:/UBFC-rPPG/subject11 \
        --infer --reference
    python -m app.live_vitals.scripts.check_contract_parity clip.mov --frames 300

No frames are rendered or written; numeric summaries only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp

# Allow direct execution as a path as well as `python -m`.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals import config
from app.live_vitals.preprocess.face_box import (
    make_landmarker, landmark_box, square_padded_box, box_clamp_report)
from app.live_vitals.preprocess.frames import crop_resize
from app.live_vitals.signal.hr import hr_from_bvp



# Reference implementations, kept verbatim from NB_P3_05 / NB_P3_07 so that any
# drift in the app modules shows up as a mismatch rather than being absorbed.
def ref_stable_face_box(frames_rgb, landmarker, padding=config.CROP_PADDING):
    """Returns the median square face box over sample frames (NB_P3_05)."""
    boxes = []
    for frame in frames_rgb:
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame))
        res = landmarker.detect(mp_img)
        if not res.face_landmarks:
            continue
        lm = res.face_landmarks[0]
        xs = np.array([p.x for p in lm]); ys = np.array([p.y for p in lm])
        boxes.append([xs.min(), ys.min(), xs.max(), ys.max()])
    if not boxes:
        return None
    x0, y0, x1, y1 = np.median(np.array(boxes), axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) * (1.0 + padding) / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def ref_crop_resize(frame_rgb, box, size):
    """Crops to a clamped relative box and resizes with INTER_AREA (NB_P3_05)."""
    h, w = frame_rgb.shape[:2]
    x0 = max(0, int(box[0] * w)); y0 = max(0, int(box[1] * h))
    x1 = min(w, int(box[2] * w)); y1 = min(h, int(box[3] * h))
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.resize(frame_rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def nb_compute_hr(bvp, fps, low=config.HR_LOW_HZ, high=config.HR_HIGH_HZ):
    """Bin-argmax spectral HR, as used for the reported notebook metrics (NB_P3_07)."""
    x = np.asarray(bvp, dtype=np.float64)
    x = x - x.mean()
    if x.size < 16 or not np.all(np.isfinite(x)) or np.std(x) < 1e-9:
        return np.nan
    p = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    fr = np.fft.rfftfreq(x.size, 1.0 / fps)
    b = (fr >= low) & (fr <= high)
    if not b.any() or p[b].sum() <= 0:
        return np.nan
    return float(fr[b][int(np.argmax(p[b]))] * 60.0)


def resolve_target(target):
    """Resolves a video path and optional ground-truth path from a CLI target."""
    path = Path(target)
    if path.is_dir():
        video = path / "vid.avi"
        truth = path / "ground_truth.txt"
        if not video.exists():
            raise FileNotFoundError(f"no vid.avi in {path}")
        return video, (truth if truth.exists() else None)
    return path, None


def load_reference(truth_path, frame_times):
    """Interpolates the UBFC-rPPG ground truth onto a frame timeline.

    `ground_truth.txt` holds three rows: PPG waveform, HR trace, and timestamps.
    """
    gt = np.loadtxt(truth_path)
    ppg, hr_trace, gt_t = gt[0], gt[1], gt[2]
    bvp = np.interp(frame_times, gt_t, ppg).astype(np.float32)
    hr = np.interp(frame_times, gt_t, hr_trace).astype(np.float32)
    return bvp, hr


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", help="video file, or a UBFC-rPPG subject directory")
    parser.add_argument("--frames", type=int, default=0,
                        help="limit frames compared (0 = whole clip)")
    parser.add_argument("--infer", action="store_true", help="run the HR estimator")
    parser.add_argument("--reference", action="store_true",
                        help="compare against ground_truth.txt and report MAE")
    args = parser.parse_args()

    video, truth = resolve_target(args.target)
    if args.reference and truth is None:
        print("ERROR: --reference needs a subject directory containing ground_truth.txt")
        return 2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print("ERROR: could not open", video)
        return 1
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 66)
    print(f"target   : {video}")
    print(f"format   : {width}x{height} @ {fps:.3f} fps, {n_total} frames, "
          f"{n_total / max(fps, 1e-9):.1f} s")

    stride = max(1, int(round(fps / config.TARGET_FPS)))
    eff_fps = fps / stride
    print(f"fps gate : target {config.TARGET_FPS} -> stride {stride} "
          f"(effective {eff_fps:.2f} fps)")
    if abs(eff_fps - config.TARGET_FPS) > 3.0:
        print("  WARNING: effective fps is far from the training rate")

    # --- stable face box, app vs reference ---
    detector = make_landmarker()
    sample_at = sorted(set(np.linspace(0, max(n_total - 1, 0),
                                       config.N_DETECT_FRAMES).astype(int)))
    samples = []
    for index in sample_at:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if ok:
            samples.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    boxes = [b for b in (landmark_box(f, detector) for f in samples) if b is not None]
    print(f"\ndetections: {len(boxes)}/{len(sample_at)} sampled frames")
    if not boxes:
        print("ERROR: no face detected")
        return 1

    app_box = square_padded_box(boxes)
    reference_box = ref_stable_face_box(samples, detector)
    box_identical = np.allclose(app_box, reference_box, atol=0, rtol=0)
    print(f"box parity vs notebook: {'IDENTICAL' if box_identical else 'DIFFERENT'}")

    report = box_clamp_report(app_box, width, height)
    print("\n-- capture quality --")
    print(f"clamped : {report['clamped']}")
    print(f"overflow px : {np.round(report['overflow_px'], 1)}")
    print(f"crop aspect : {report['kept_aspect']:.3f} "
          f"(expected {report['expected_aspect']:.3f} for this frame shape)")
    print(f"distortion : {report['distortion']:.3f}  (1.000 = undistorted)")
    print(f"frac side lost : {report['frac_side_lost']:.3f}  "
          f"(gate {config.MAX_BOX_SIDE_LOST})")
    accepted = report["frac_side_lost"] <= config.MAX_BOX_SIDE_LOST
    print(f"framing : {'ACCEPT' if accepted else 'REJECT'}")

    # --- crop every kept frame, comparing app against reference ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    clip, kept_indices, compared, mismatches = [], [], 0, 0
    source_index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if source_index % stride:
            source_index += 1
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        expected = ref_crop_resize(rgb, reference_box, config.TARGET_SIZE)
        actual = crop_resize(rgb, app_box, config.TARGET_SIZE)
        if expected is not None and actual is not None:
            if not np.array_equal(expected, actual):
                mismatches += 1
            clip.append(actual)
            kept_indices.append(source_index)
            compared += 1
        source_index += 1
        if args.frames and compared >= args.frames:
            break
    cap.release()

    print("\n-- preprocessing parity --")
    print(f"frames compared: {compared}")
    print(f"bit mismatches : {mismatches}")
    parity_ok = box_identical and mismatches == 0 and compared > 0
    print(f"PARITY : {'PASS' if parity_ok else 'FAIL'}")

    if not args.infer:
        return 0 if parity_ok else 1

    # --- inference ---
    if len(clip) < config.CLIP_LEN:
        print(f"\nskipping inference: {len(clip)}/{config.CLIP_LEN} frames available")
        return 0 if parity_ok else 1

    import torch
    from app.live_vitals.models.architectures.physnet import PhysNet

    state = torch.load(config.PHYSNET_CKPT, map_location="cpu", weights_only=True)
    model = PhysNet(frames=config.CLIP_LEN)
    model.load_state_dict(state)
    model.eval()

    from app.live_vitals.preprocess.frames import clip_to_tensor

    clip = np.asarray(clip, dtype=np.uint8)
    frame_times = np.asarray(kept_indices, dtype=np.float64) / float(fps)

    reference_bvp = reference_hr_trace = None
    if args.reference:
        reference_bvp, reference_hr_trace = load_reference(truth, frame_times)

    starts = list(range(0, len(clip) - config.CLIP_LEN + 1, config.WINDOW_STRIDE))
    rows = []
    for start in starts:
        window = slice(start, start + config.CLIP_LEN)
        with torch.no_grad():
            bvp = model(clip_to_tensor(clip[window]))[0].numpy()
        row = dict(app=hr_from_bvp(bvp, eff_fps)["hr_bpm"],
                   nb=nb_compute_hr(bvp, eff_fps))
        if reference_bvp is not None:
            row["ref_nb"] = nb_compute_hr(reference_bvp[window], eff_fps)
            row["ref_app"] = hr_from_bvp(reference_bvp[window], eff_fps)["hr_bpm"]
            trace = reference_hr_trace[window]
            valid = trace[(trace >= config.HR_LOW_HZ * 60.0)
                          & (trace <= config.HR_HIGH_HZ * 60.0)]
            row["ref_trace"] = float(np.median(valid)) if valid.size else float("nan")
        rows.append(row)

    print("\n-- heart rate --")
    print(f"windows        : {len(rows)} (clip {config.CLIP_LEN}, "
          f"stride {config.WINDOW_STRIDE})")
    app_hr = np.array([r["app"] for r in rows], dtype=float)
    nb_hr = np.array([r["nb"] for r in rows], dtype=float)
    spread = float(np.nanpercentile(app_hr, 75) - np.nanpercentile(app_hr, 25))
    print(f"app HR (median): {np.nanmedian(app_hr):7.2f} bpm  (IQR spread {spread:.2f})")
    print(f"nb  HR (median): {np.nanmedian(nb_hr):7.2f} bpm")

    if reference_bvp is None:
        if not accepted:
            print("\nNOTE: framing was REJECTED - this HR is not meaningful.")
        return 0 if parity_ok else 1

    ref_nb = np.array([r["ref_nb"] for r in rows], dtype=float)
    ref_app = np.array([r["ref_app"] for r in rows], dtype=float)
    ref_trace = np.array([r["ref_trace"] for r in rows], dtype=float)

    def mae(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        return float(np.mean(np.abs(a[mask] - b[mask]))) if mask.any() else float("nan")

    print("\n-- reference agreement --")
    print(f"reference HR (spectral median): {np.nanmedian(ref_nb):7.2f} bpm")
    print(f"reference HR (trace median) : {np.nanmedian(ref_trace):7.2f} bpm")
    print(f"MAE notebook-style (argmax both sides) : {mae(nb_hr, ref_nb):6.2f} bpm"
          " <- compare to the 2.80-3.12 screen number")
    print(f"MAE app-style (padded + guard) : {mae(app_hr, ref_app):6.2f} bpm")
    print(f"MAE app vs reference HR trace : {mae(app_hr, ref_trace):6.2f} bpm")
    bias = np.nanmean(app_hr - ref_app)
    print(f"bias (app - reference) : {bias:+6.2f} bpm")

    return 0 if parity_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
