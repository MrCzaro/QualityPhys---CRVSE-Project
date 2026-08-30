"""Regression check for Phase-3 estimators against the held-out UBFC-rPPG subjects.

Runs a registered estimator over the eight held-out UBFC-rPPG subjects, compares each
analysis window against the reference BVP, and fails if accuracy has drifted from a
stored baseline. The pipeline is deterministic, so any change beyond floating-point
noise indicates a real behavioural change rather than run-to-run variation.

Usage:
    python -m app.live_vitals.scripts.check_ubfc_regression [--estimator NAME]
        [--data-dir DIR] [--cache-dir DIR] [--subjects 11 13] [--update-baseline]

Exits non-zero when a regression is detected. No frames are rendered or written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals import config
from app.live_vitals.estimators import registry
from app.live_vitals.preprocess.face_box import (
    make_landmarker, landmark_box, square_padded_box)
from app.live_vitals.preprocess.frames import crop_resize
from app.live_vitals.signal.hr import hr_from_bvp
from app.live_vitals.scripts.check_contract_parity import load_reference

HELD_OUT_SUBJECTS = [11, 13, 24, 25, 34, 35, 42, 47]
DEFAULT_DATA_DIR = Path(r"D:\QualityPhys\demo_data")
BASELINE_PATH = Path(__file__).with_name("ubfc_baseline.json")
TOLERANCE_BPM = 0.05
TOLERANCE_FRACTION = 0.01


def face_crops(video_path, cache_path=None):
    """Returns the 72x72 face-crop sequence and source fps for one recording."""
    if cache_path and cache_path.exists():
        with np.load(cache_path) as store:
            return store["clip"], float(store["fps"])

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total <= 0:
        probe = cv2.VideoCapture(str(video_path))
        n_total = 0
        while probe.grab():
            n_total += 1
        probe.release()

    detector = make_landmarker()
    samples = []
    for index in sorted(set(np.linspace(0, n_total - 1,
                                        config.N_DETECT_FRAMES).astype(int))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if ok:
            samples.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    boxes = [b for b in (landmark_box(f, detector) for f in samples) if b is not None]
    if not boxes:
        raise RuntimeError(f"no face detected in {video_path}")
    box = square_padded_box(boxes)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    clip = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        crop = crop_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), box)
        if crop is not None:
            clip.append(crop)
    cap.release()
    clip = np.asarray(clip, dtype=np.uint8)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, clip=clip, fps=fps)
    return clip, fps


def window_starts(n_frames):
    """Returns the analysis-window start indices the estimator uses."""
    return range(0, n_frames - config.CLIP_LEN + 1, config.WINDOW_STRIDE)


def evaluate_subject(subject_dir, estimator, cache_path=None):
    """Returns one subject's estimator reading and window-level agreement."""
    clip, fps = face_crops(subject_dir / "vid.avi", cache_path)
    reference_bvp, _ = load_reference(subject_dir / "ground_truth.txt",
                                      np.arange(len(clip)) / fps)

    result = estimator.estimate(clip, fps)

    # Per-window values come from the estimator's own report, so this stays on the
    # public interface, works for any registered model, and does not re-run the
    # model once per window.
    predicted = result.detail.get("window_hr")
    if predicted is None:
        raise RuntimeError(f"{estimator.name} does not report per-window values")
    reference = [hr_from_bvp(reference_bvp[start:start + config.CLIP_LEN],
                             fps)["hr_bpm"]
                 for start in window_starts(len(clip))]

    predicted = np.asarray(predicted, dtype=float)
    reference = np.asarray(reference, dtype=float)
    finite = np.isfinite(predicted) & np.isfinite(reference)
    mae = float(np.mean(np.abs(predicted[finite] - reference[finite])))
    reference_hr = float(np.median(reference[np.isfinite(reference)]))

    return dict(value=float(result.value),
                reference_hr=reference_hr,
                error=float(result.value - reference_hr),
                mae=mae,
                confidence=float(result.confidence),
                status=result.status,
                usable_fraction=float(result.detail.get("usable_fraction",
                                                        float("nan"))))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", default=registry.DEFAULT_ESTIMATOR,
                        choices=registry.available(),
                        help="registered estimator name")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="cache face crops here to speed up repeat runs")
    parser.add_argument("--subjects", type=int, nargs="+", default=HELD_OUT_SUBJECTS)
    parser.add_argument("--update-baseline", action="store_true",
                        help="overwrite the stored baseline with this run")
    args = parser.parse_args()

    estimator = registry.get_estimator(args.estimator)
    if not estimator.is_available():
        print(f"ERROR: checkpoint missing for {args.estimator}")
        return 2

    print(f"estimator: {args.estimator}   subjects: {len(args.subjects)}")
    print(f"{'subj':>5} {'value':>7} {'ref':>7} {'err':>6} {'MAE':>6} "
          f"{'conf':>6} {'frac':>5}  status")

    results = {}
    for subject in args.subjects:
        subject_dir = args.data_dir / f"subject{subject}"
        cache = (args.cache_dir / f"subject{subject}.npz") if args.cache_dir else None
        row = evaluate_subject(subject_dir, estimator, cache)
        results[str(subject)] = row
        print(f"{subject:5d} {row['value']:7.2f} {row['reference_hr']:7.2f} "
              f"{row['error']:+6.2f} {row['mae']:6.2f} {row['confidence']:6.3f} "
              f"{row['usable_fraction']:5.2f}  {row['status']}")

    mean_mae = float(np.mean([r["mae"] for r in results.values()]))
    mean_error = float(np.mean([r["error"] for r in results.values()]))
    print(f"\nmean MAE {mean_mae:.3f} bpm | mean signed error {mean_error:+.3f} bpm")

    summary = dict(estimator=args.estimator, mean_mae=mean_mae,
                   mean_error=mean_error, subjects=results)

    if args.update_baseline:
        BASELINE_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"baseline written: {BASELINE_PATH.name}")
        return 0

    if not BASELINE_PATH.exists():
        print(f"\nno baseline at {BASELINE_PATH.name}; "
              f"run with --update-baseline first")
        return 0

    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    if baseline.get("estimator") != args.estimator:
        print(f"\nbaseline is for {baseline.get('estimator')!r}; not comparable")
        return 0

    print("\n-- regression check --")
    if set(results) == set(baseline["subjects"]):
        print(f"  baseline mean MAE {baseline['mean_mae']:.3f} -> {mean_mae:.3f} "
              f"({mean_mae - baseline['mean_mae']:+.3f} bpm)")
    else:
        # A subset mean is not comparable to the full-set baseline mean; the
        # per-subject comparisons below are what decide pass or fail.
        print(f"  subset of {len(results)}/{len(baseline['subjects'])} subjects; "
              f"comparing per-subject only")

    # Per-window MAE exercises the model and readout but not the aggregation layer:
    # a single window always falls back past the gates, so MAE alone cannot see a
    # gating change. The reported value, status and usable fraction are compared too.
    drifted = []
    for subject, row in results.items():
        want = baseline["subjects"].get(subject)
        if want is None:
            print(f"  subject{subject}: not in baseline")
            continue
        for field, tolerance in (("mae", TOLERANCE_BPM),
                                 ("value", TOLERANCE_BPM),
                                 ("usable_fraction", TOLERANCE_FRACTION)):
            delta = row[field] - want[field]
            # NaN compares false against any tolerance, so a value that became
            # NaN would otherwise pass silently. Treat it as drift explicitly.
            if delta != delta or abs(delta) > tolerance:
                drifted.append((subject, field, want[field], row[field], delta))
        if row["status"] != want["status"]:
            drifted.append((subject, "status", want["status"], row["status"], None))

    if drifted:
        print(f"  REGRESSION: {len(drifted)} change(s) from baseline")
        for subject, field, was, now, delta in drifted:
            shift = "" if delta is None else f" ({delta:+.3f})"
            was_s = was if isinstance(was, str) else f"{was:.3f}"
            now_s = now if isinstance(now, str) else f"{now:.3f}"
            print(f"    subject{subject} {field}: {was_s} -> {now_s}{shift}")
        return 1

    print(f"  PASS: value, MAE, usable fraction and status all match baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
