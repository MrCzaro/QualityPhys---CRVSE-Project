"""Regression check for Phase-3 estimators on eight UBFC-rPPG subjects.

Runs a registered estimator over the subjects through the app's own capture path
(`crops_from_video`: face box, decimation, measured frame rate, capture verdict), compares
each analysis window against the reference BVP, and fails if anything has drifted from the
stored baseline. The pipeline is deterministic, so any change beyond floating-point noise
indicates a real behavioural change rather than run-to-run variation. The subjects were
drawn from the screen's validation split, not v2's, and six of the eight were in v2's
training data: this check catches drift, it does not measure held-out accuracy.

Compared per subject: the reported value, window MAE, usable fraction and status, and the
capture path's frame count, measured rate, head motion and verdict. The baseline also
records the conditions it was taken under. The config constants, the face model and the
weights are part of what it certifies, so a run under different ones fails until a new
baseline is recorded, even if its figures happen to match. The capture code's hash and the
library versions are recorded for diagnosis only: an edit or upgrade that changes no figure
passes, with a note.

Usage:
    python -m app.live_vitals.scripts.check_ubfc_regression [--estimator NAME]
        [--data-dir DIR] [--subjects 11 13] [--update-baseline]

Exit codes: 0 pass; 1 drift, or changed conditions; 2 nothing comparable to check against
(no baseline, another estimator's, one written before these fields existed, a subject it
does not cover) or a missing checkpoint. No frames are rendered or written.
"""
import argparse
import hashlib
import inspect
import json
import platform
import sys
from pathlib import Path

import cv2
import mediapipe
import numpy as np
import scipy
import torch


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.live_vitals import config
from app.live_vitals.capture import motion, session
from app.live_vitals.estimators import registry
from app.live_vitals.preprocess import face_box, frames
from app.live_vitals.signal.hr import hr_from_bvp
from app.live_vitals.scripts.bench_outputs import file_sha256, provenance, unique
from app.live_vitals.scripts.check_contract_parity import load_reference

SUBJECTS = [11, 13, 24, 25, 34, 35, 42, 47]  # from the screen's validation split
DEFAULT_DATA_DIR = Path(r"D:\QualityPhys\demo_data\ubfc_rppg_subset")
BASELINE_FORMAT = 2  # 2 (2026-10-14): capture-path fields, recorded conditions, provenance
TOLERANCE_BPM = 0.05
TOLERANCE_FRACTION = 0.01
TOLERANCE_FPS = 0.001
TOLERANCE_MOTION_PX = 0.01
# Compared per subject, with the tolerance of each; None compares exactly. A field the
# baseline does not hold is not compared (motion_px came with the motion gate, 2026-10-15).
COMPARED = (("mae", TOLERANCE_BPM), ("value", TOLERANCE_BPM),
            ("usable_fraction", TOLERANCE_FRACTION), ("status", None),
            ("n_frames", None), ("effective_fps", TOLERANCE_FPS),
            ("motion_px", TOLERANCE_MOTION_PX), ("verdict", None))
# Config settings that never reach a recorded video's reading, so they are not recorded.
RUNTIME_ONLY = ("CAMERA_STALL_SECONDS", "DEBUG_REPORT")
# Recorded conditions whose change fails the check; the rest only add a note.
CERTIFIED = ("config.", "face_model_sha256", "weights_sha256")


def baseline_path(estimator_name):
    """Returns the baseline file for one estimator.

    Baselines are per-estimator because they are absolute accuracy records, not
    relative ones: comparing a model against another model's baseline would
    report a difference between two models as a regression in one of them.
    """
    return Path(__file__).with_name(f"ubfc_baseline_{estimator_name}.json")


def conditions(estimator):
    """What the readings depend on besides the data, as recorded with a baseline.

    Paths are left out of the config: they differ between machines and say nothing about
    behaviour; the files that matter are recorded by hash. The capture modules are hashed
    whole, comments included, which is why that hash only ever adds a note.
    """
    source = hashlib.sha256()
    for module in (session, motion, face_box, frames):
        source.update(inspect.getsource(module).encode("utf-8"))
    checkpoint = getattr(estimator, "checkpoint", None)
    recorded = dict(
        config={name: value for name, value in sorted(vars(config).items())
                if name.isupper() and name not in RUNTIME_ONLY
                and not isinstance(value, Path)},
        face_model_sha256=file_sha256(config.MP_MODEL),
        weights_sha256=file_sha256(checkpoint) if checkpoint is not None else None,
        capture_code_sha256=source.hexdigest(),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      scipy=scipy.__version__, torch=torch.__version__,
                      opencv=cv2.__version__, mediapipe=mediapipe.__version__))
    return json.loads(json.dumps(recorded))  # tuples become lists, as in a stored baseline


def condition_changes(was, now, prefix=""):
    """Lists the entries of two nested condition records that differ, as 'key: was -> now'."""
    changes = []
    for key in sorted(set(was) | set(now)):
        a, b = was.get(key), now.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            changes += condition_changes(a, b, f"{prefix}{key}.")
        elif a != b:
            changes.append(f"{prefix}{key}: {_short(a)} -> {_short(b)}")
    return changes


def window_starts(n_frames):
    """Returns the analysis-window start indices the estimator uses."""
    return range(0, n_frames - config.CLIP_LEN + 1, config.WINDOW_STRIDE)


def evaluate_subject(subject_dir, estimator, landmarker):
    """Returns one subject's reading, window-level agreement and capture report.

    The crops come from the app's capture path, so a change there - box, decimation, rate
    measurement, verdict - reaches this check as it would reach a user. They are not
    cached: a cached crop would hide exactly such a change, and the decode costs about a
    second per subject.
    """
    clip, fps, quality = session.crops_from_video(subject_dir / "vid.avi", landmarker)
    reference_bvp, _ = load_reference(subject_dir / "ground_truth.txt",
                                      np.arange(len(clip)) / fps)

    result = estimator.estimate(clip, fps)

    # Per-window values come from the estimator's own report, so this stays on the
    # public interface, works for any registered model, and does not re-run the
    # model once per window.
    predicted = result.detail.get("window_hr")
    predicted_starts = result.detail.get("window_start")
    if predicted is None or predicted_starts is None:
        raise RuntimeError(f"{estimator.name} does not report per-window values "
                           f"with their start frames")
    if (len(predicted_starts) != len(predicted)
            or result.detail["n_total"] - result.detail["n_no_peak"] != len(predicted)):
        raise RuntimeError(f"{estimator.name} per-window report is inconsistent")

    starts = list(window_starts(len(clip)))
    reference = np.asarray([hr_from_bvp(reference_bvp[start:start + config.CLIP_LEN],
                                        fps)["hr_bpm"]
                            for start in starts], dtype=float)

    # Windows without a readable peak are absent from the estimator's report, so the
    # two lists are paired by start frame. Pairing by position would compare every
    # later window against the wrong stretch of reference once a single window drops.
    reference_at = dict(zip(starts, reference))
    predicted = np.asarray(predicted, dtype=float)
    paired_reference = np.asarray([reference_at[s] for s in predicted_starts], dtype=float)
    finite = np.isfinite(predicted) & np.isfinite(paired_reference)
    mae = float(np.mean(np.abs(predicted[finite] - paired_reference[finite])))
    reference_hr = float(np.median(reference[np.isfinite(reference)]))

    return dict(value=float(result.value),
                reference_hr=reference_hr,
                error=float(result.value - reference_hr),
                mae=mae,
                confidence=float(result.confidence),
                status=result.status,
                usable_fraction=float(result.detail.get("usable_fraction",
                                                        float("nan"))),
                n_frames=int(len(clip)),
                effective_fps=float(fps),
                motion_px=float(quality.motion_px),
                verdict=quality.verdict)


def same(now, was, tolerance):
    """True when a value matches its baseline; NaN matches only NaN."""
    if tolerance is None:
        return now == was
    if np.isnan(now) or np.isnan(was):
        return bool(np.isnan(now) and np.isnan(was))
    return abs(now - was) <= tolerance


def compare(run, baseline_file):
    """Checks a run against the stored baseline and returns the exit code."""
    if not baseline_file.exists():
        print(f"\nNO BASELINE: {baseline_file.name} does not exist; "
              f"run with --update-baseline to create it")
        return 2
    baseline = json.loads(baseline_file.read_text(encoding="utf-8"))
    if baseline.get("estimator") != run["estimator"]:
        print(f"\nNOT COMPARABLE: {baseline_file.name} is for {baseline.get('estimator')!r}")
        return 2
    older = baseline.get("format", 1) < BASELINE_FORMAT

    print("\n-- regression check --")
    if set(run["subjects"]) == set(baseline["subjects"]):
        print(f"  baseline mean MAE {baseline['mean_mae']:.3f} -> {run['mean_mae']:.3f} "
              f"({run['mean_mae'] - baseline['mean_mae']:+.3f} bpm)")
    else:
        # A subset mean is not comparable to the full-set baseline mean; the
        # per-subject comparisons below are what decide pass or fail.
        print(f"  {len(run['subjects'])} subject(s) against a baseline of "
              f"{len(baseline['subjects'])}; comparing per subject only")

    # Per-window MAE exercises the model and readout but not the aggregation layer:
    # a single window always falls back past the gates, so MAE alone cannot see a
    # gating change. The reported value, status and usable fraction are compared too,
    # and the capture report catches a change in how the frames reached the model.
    drifted, uncovered = [], []
    for subject, row in run["subjects"].items():
        want = baseline["subjects"].get(subject)
        if want is None:
            uncovered.append(subject)
            continue
        for field, tolerance in COMPARED:
            if field in want and not same(row[field], want[field], tolerance):
                drifted.append((subject, field, want[field], row[field]))

    changes = [] if older else condition_changes(baseline["conditions"], run["conditions"])
    certified = [c for c in changes if c.startswith(CERTIFIED)]
    noted = [c for c in changes if not c.startswith(CERTIFIED)]

    if drifted or certified:
        if drifted:
            print(f"  REGRESSION: {len(drifted)} change(s) from baseline")
            for subject, field, was, now in drifted:
                print(f"    subject{subject} {field}: {_value(was)} -> {_value(now)}"
                      f"{_shift(was, now)}")
        if certified:
            print("  CHANGED SINCE THE BASELINE (record a new one if this is intended):")
            for change in certified:
                print(f"    {change}")
        if noted:
            print("  also differs from the baseline's run:")
            for change in noted:
                print(f"    {change}")
        elif not certified and not older:
            print("  nothing recorded with the baseline moved, so look at the estimator, "
                  "readout\n  and gating code, or the data")
        return 1
    if uncovered:
        print(f"  NOT COMPARABLE: the baseline has no "
              f"{', '.join(f'subject{s}' for s in uncovered)}")
        return 2
    if older:
        print("  value, MAE, usable fraction and status match the baseline")
        print("  NOT COMPARABLE: the baseline predates the capture-path fields (frame count, "
              "measured rate,\n  verdict); record a new one with --update-baseline")
        return 2
    print("  PASS: value, MAE, usable fraction, status, frame count, measured rate, motion "
          "and verdict all match baseline")
    if noted:
        print("  note - differs from the baseline's run, with no effect on the figures:")
        for change in noted:
            print(f"    {change}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--estimator", default=registry.DEFAULT_ESTIMATOR,
                        choices=registry.available(),
                        help="registered estimator name")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    parser.add_argument("--update-baseline", action="store_true",
                        help="overwrite the stored baseline with this run (all subjects)")
    args = parser.parse_args()

    subjects = unique(args.subjects)
    if args.update_baseline and sorted(subjects) != sorted(SUBJECTS):
        print("ERROR: a baseline is written from all eight subjects; drop --subjects")
        return 2

    baseline_file = baseline_path(args.estimator)
    estimator = registry.get_estimator(args.estimator)
    if not estimator.is_available():
        print(f"ERROR: checkpoint missing for {args.estimator}")
        return 2

    print(f"estimator: {args.estimator}   subjects: {len(subjects)}")
    landmarker = face_box.make_landmarker()
    print(f"{'subj':>5} {'value':>7} {'ref':>7} {'err':>6} {'MAE':>6} {'conf':>6} "
            f"{'frac':>5} {'frames':>6} {'fps':>7} {'motion':>6}  {'verdict':7} status")

    results = {}
    for subject in subjects:
        row = evaluate_subject(args.data_dir / f"subject{subject}", estimator, landmarker)
        results[str(subject)] = row
        print(f"{subject:5d} {row['value']:7.2f} {row['reference_hr']:7.2f} "
              f"{row['error']:+6.2f} {row['mae']:6.2f} {row['confidence']:6.3f} "
              f"{row['usable_fraction']:5.2f} {row['n_frames']:6d} {row['effective_fps']:7.3f} "
              f"{row['motion_px']:6.2f}  {row['verdict']:7} {row['status']}")

    mean_mae = float(np.mean([r["mae"] for r in results.values()]))
    mean_error = float(np.mean([r["error"] for r in results.values()]))
    print(f"\nmean MAE {mean_mae:.3f} bpm | mean signed error {mean_error:+.3f} bpm")

    run = dict(format=BASELINE_FORMAT, estimator=args.estimator, mean_mae=mean_mae,
               mean_error=mean_error, subjects=results, conditions=conditions(estimator),
               provenance=provenance(estimator))

    if args.update_baseline:
        baseline_file.write_text(json.dumps(run, indent=2), encoding="utf-8")
        print(f"baseline written: {baseline_file.name}")
        return 0
    return compare(run, baseline_file)


def _short(value):
    """Shortens a hash for display; other values print as they are."""
    return f"{value[:12]}..." if isinstance(value, str) and len(value) == 64 else value


def _value(value):
    """Formats a compared value for the drift report."""
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _shift(was, now):
    """The signed change of a numeric value, when both ends are numbers."""
    numeric = all(isinstance(v, (int, float)) and np.isfinite(v) for v in (was, now))
    return f" ({now - was:+.3f})" if numeric else ""


if __name__ == "__main__":
    raise SystemExit(main())