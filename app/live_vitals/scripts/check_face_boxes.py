"""Measures, on raw videos, what three candidate face checks would see in the app.

The app holds one face box for a whole capture: the median of the landmark boxes
found in N_DETECT_FRAMES frames spread evenly over it (capture/session.py). Today
nothing checks that box beyond its fit in the frame, so this script measures, before
any gate is set, what three checks would see:

  faces     how many of the samples find a face. The app refuses a capture only when
            none does.
  second    samples that find a second face, and that face's width beside the
            subject's. The app asks its detector for one face, so a second person in
            view is never noticed.
  outside   how much of the face's landmark box lies outside the held box, at each
            sample. A face that drifts out of the crop part-way through a capture goes
            unseen by the motion gate, which reads movement in the cardiac band only.

Detection follows the app: the same sample positions, the same landmarker model and
the same box rule (square_padded_box). The landmarker is asked for two faces, and the
larger is taken as the subject. The app's own one-face landmarker runs beside it, to
show whether it ever takes the other face.

Frames are fetched by seeking, not by the app's front-to-back pass: the files here
are indexed (AVI, MP4, MOV), and a sample a frame or two off changes no box. A file
whose container declares no frame count is read front to back instead, as the app
reads it.

Every video file under each --root is measured; folders whose names start with a dot
are skipped. Results are grouped by the first folder under the root, and files
directly in a root form their own group.

Writes face_boxes.csv (one row per video; git-ignored) and summary_face_boxes.json
(aggregates per group) to --out-dir. Nothing written contains a frame.

Usage:
    python -m app.live_vitals.scripts.check_face_boxes [--root DIR ...] [--out-dir DIR]
        [--limit N]
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import mediapipe as mp  # noqa: E402
from mediapipe.tasks import python as mp_python  # noqa: E402
from mediapipe.tasks.python import vision as mp_vision  # noqa: E402

from app.live_vitals import config  # noqa: E402
from app.live_vitals.capture.session import detection_positions, read_through  # noqa: E402
from app.live_vitals.preprocess.face_box import (  # noqa: E402
    landmark_box, make_landmarker, square_padded_box)
from app.live_vitals.scripts.bench_outputs import (  # noqa: E402
    file_sha256, provenance, subset_suffix)

DEFAULT_ROOTS = [Path(r"D:\QualityPhys\demo_data")]
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "face_gate"
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}
# A sample counts as partly outside the held box when more than this share of the
# face's landmark box lies outside it.
OUTSIDE_SHARE = 0.05
# Second faces at least this wide beside the subject's are counted apart: a bystander
# close to the camera, rather than a face in a picture on the wall.
NEAR_SECOND = 0.5
ROOT_GROUP = "(files in the root)"


def make_two_face_landmarker(model_path=config.MP_MODEL):
    """The app's landmarker model, asked for up to two faces."""
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=2)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def face_boxes(frame_rgb, landmarker):
    """Every face's relative (x0, y0, x1, y1) landmark box, largest first."""
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
    boxes = []
    for pts in landmarker.detect(img).face_landmarks:
        xs = np.fromiter((p.x for p in pts), dtype=np.float64)
        ys = np.fromiter((p.y for p in pts), dtype=np.float64)
        boxes.append((float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())))
    return sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def overlap(a, b):
    """The area two boxes share."""
    return area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def iou(a, b):
    shared = overlap(a, b)
    return shared / max(area(a) + area(b) - shared, 1e-12)


def sample_frames(path):
    """Returns (frames_rgb, n_frames, info) at the app's sample positions.

    A frame that cannot be read is None in its place. info holds the container's
    width, height, rate, codec and rotation; a rotated file's frames arrive upright.
    """
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise OSError("OpenCV cannot open the file")
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        info = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    fps=cap.get(cv2.CAP_PROP_FPS),
                    codec="".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\0 "),
                    rotation=int(cap.get(cv2.CAP_PROP_ORIENTATION_META)))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n_frames > 0:
            frames = []
            for position in detection_positions(n_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, bgr = cap.read()
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None)
            return frames, n_frames, info
    finally:
        cap.release()
    # No declared count (a streamed WebM): count front to back, then sample, as the app does.
    stamps, _ = read_through(path, [])
    _, frames = read_through(path, detection_positions(len(stamps)))
    return frames, len(stamps), info


def measure(path, two_faces, one_face):
    """Measures one video; returns its row."""
    frames, n_frames, info = sample_frames(path)
    # Frames arrive upright, so a quarter-turned file's sides swap.
    upright = next((f for f in frames if f is not None), None)
    height, width = (upright.shape[:2] if upright is not None
                     else (info["height"], info["width"]))
    subjects, seconds, app_faces, app_other = [], [], 0, 0
    for rgb in frames:
        if rgb is None:
            continue
        app_box = landmark_box(rgb, one_face)
        app_faces += app_box is not None
        boxes = face_boxes(rgb, two_faces)
        if not boxes:
            continue
        subjects.append(boxes[0])
        if len(boxes) > 1:
            seconds.append((boxes[1][2] - boxes[1][0]) / max(boxes[0][2] - boxes[0][0], 1e-12))
        if app_box is not None and iou(app_box, boxes[0]) < 0.5:
            app_other += 1
    fps = info["fps"]
    row = dict(n_frames=n_frames, width=width, height=height, fps=round(fps, 3),
               codec=info["codec"], rotation=info["rotation"],
               duration_s=round(n_frames / fps, 1) if fps > 0 else float("nan"),
               samples=len(frames), read=sum(f is not None for f in frames),
               faces=len(subjects), faces_app=app_faces, second_samples=len(seconds),
               second_width_max=round(max(seconds), 3) if seconds else 0.0,
               near_second_samples=int(sum(s >= NEAR_SECOND for s in seconds)),
               app_other_samples=app_other)
    if not subjects:
        return dict(row, inside_min=float("nan"), outside_samples=0, shift_max=float("nan"),
                    shift_median=float("nan"), width_ratio=float("nan"))
    held = square_padded_box(subjects)
    inside = [float(overlap(b, held) / max(area(b), 1e-12)) for b in subjects]
    # Distances in pixels, in units of the subject's median face width: the relative
    # coordinates are stretched differently along each axis of a non-square frame.
    centres = np.array([((b[0] + b[2]) / 2 * width, (b[1] + b[3]) / 2 * height)
                        for b in subjects])
    widths = np.array([(b[2] - b[0]) * width for b in subjects])
    shifts = np.hypot(*(centres - np.median(centres, axis=0)).T) / np.median(widths)
    return dict(row, inside_min=round(min(inside), 4),
                outside_samples=int(sum(share < 1.0 - OUTSIDE_SHARE for share in inside)),
                shift_max=round(float(shifts.max()), 4),
                shift_median=round(float(np.median(shifts)), 4),
                width_ratio=round(float(widths.max() / widths.min()), 4))


def find_videos(roots):
    """(group, path) for every video file under the roots, in a stable order."""
    found = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if (path.suffix.lower() not in VIDEO_SUFFIXES or not path.is_file()
                    or any(part.startswith(".") for part in relative.parts[:-1])):
                continue
            group = relative.parts[0] if len(relative.parts) > 1 else ROOT_GROUP
            found.append((group, path))
    return found


def quantiles(values, qs):
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if not len(values):
        return {}
    return {f"p{int(q * 100):02d}": round(float(np.quantile(values, q)), 4) for q in qs}


def summarise(rows):
    """Aggregates per group: counts and the spread of each measure."""
    out = {}
    for group in dict.fromkeys(r["group"] for r in rows):
        sel = [r for r in rows if r["group"] == group]
        ok = [r for r in sel if not r["error"]]
        faces = [r["faces"] for r in ok]
        out[group] = dict(
            n_videos=len(sel), n_unreadable=len(sel) - len(ok),
            faces_by_count={str(k): faces.count(k) for k in sorted(set(faces), reverse=True)},
            all_samples_found=sum(r["faces"] == r["samples"] for r in ok),
            faces_differ_from_app=sum(r["faces"] != r["faces_app"] for r in ok),
            with_second_face=sum(r["second_samples"] > 0 for r in ok),
            with_near_second_face=sum(r["near_second_samples"] > 0 for r in ok),
            app_took_other_face=sum(r["app_other_samples"] > 0 for r in ok),
            with_face_partly_outside=sum(r["outside_samples"] > 0 for r in ok),
            inside_min=quantiles([r["inside_min"] for r in ok], (0.0, 0.05, 0.5)),
            shift_max=quantiles([r["shift_max"] for r in ok], (0.5, 0.95, 1.0)),
            width_ratio=quantiles([r["width_ratio"] for r in ok], (0.5, 0.95, 1.0)))
    return out


def print_summary(summary):
    print(f"\n{'group':26} {'videos':>6} {'12/12':>6} {'fewest':>6} {'2nd':>4} {'near':>4} "
          f"{'app':>4} {'out':>4} | {'inside min':>10} {'shift p95':>9} {'shift max':>9} "
          f"{'width max':>9}")
    for group, s in summary.items():
        fewest = min((int(k) for k in s["faces_by_count"]), default=0)
        print(f"{group[:26]:26} {s['n_videos']:6d} {s['all_samples_found']:6d} {fewest:6d} "
              f"{s['with_second_face']:4d} {s['with_near_second_face']:4d} "
              f"{s['app_took_other_face']:4d} {s['with_face_partly_outside']:4d} | "
              f"{s['inside_min'].get('p00', float('nan')):10.3f} "
              f"{s['shift_max'].get('p95', float('nan')):9.3f} "
              f"{s['shift_max'].get('p100', float('nan')):9.3f} "
              f"{s['width_ratio'].get('p100', float('nan')):9.3f}")
        if s["n_unreadable"]:
            print(f"{'':26} {s['n_unreadable']} unreadable")
    print("\n12/12: videos with a face in every sample; fewest: the fewest samples with a face;"
          "\n2nd / near: videos with a second face (near: at least half the subject's width);"
          "\napp: videos where the app's one-face detector took the other face;"
          f"\nout: videos with a sample whose face is more than {OUTSIDE_SHARE:.0%} outside the"
          "\nheld box; inside min: the lowest share of a face inside it, over all videos;"
          "\nshift: a face's distance from its median position, in face widths;"
          "\nwidth max: the largest ratio of a video's widest to narrowest face.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append",
                        help=f"folder to search for videos; repeat for several "
                             f"(default: {DEFAULT_ROOTS[0]})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="measure only the first N videos (smoke test)")
    args = parser.parse_args()

    roots = args.root or DEFAULT_ROOTS
    missing = [str(r) for r in roots if not r.is_dir()]
    if missing:
        raise SystemExit(f"ERROR: not a folder: {', '.join(missing)}")
    videos = find_videos(roots)
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit("ERROR: no video files found")
    print(f"{len(videos)} videos under {', '.join(str(r) for r in roots)}")

    run_provenance = provenance(None)  # read git's state before this run writes anything
    two_faces, one_face = make_two_face_landmarker(), make_landmarker()
    rows, t0 = [], time.perf_counter()
    try:
        for i, (group, path) in enumerate(videos, 1):
            root = next(r for r in roots if path.is_relative_to(r))
            row = dict(group=group, file=str(path.relative_to(root)), error="")
            try:
                row.update(measure(path, two_faces, one_face))
                print(f"[{i}/{len(videos)}] {row['file']}: faces {row['faces']}/"
                      f"{row['samples']}, second {row['second_samples']}, inside min "
                      f"{row['inside_min']:.3f}, shift max {row['shift_max']:.3f}", flush=True)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{i}/{len(videos)}] {row['file']}: {row['error']}", flush=True)
            rows.append(row)
    finally:
        two_faces.close()
        one_face.close()
    print(f"measured in {(time.perf_counter() - t0) / 60:.1f} min")

    summary = dict(
        settings=dict(n_detect_frames=config.N_DETECT_FRAMES, crop_padding=config.CROP_PADDING,
                      outside_share=OUTSIDE_SHARE, near_second=NEAR_SECOND,
                      face_model_sha256=file_sha256(config.MP_MODEL)),
        roots=[str(r) for r in roots], n_videos=len(rows), groups=summarise(rows),
        provenance=run_provenance)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = subset_suffix(("limit", args.limit or None, None))
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(args.out_dir / f"face_boxes{suffix}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / f"summary_face_boxes{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(summary["groups"])
    print(f"\noutputs: {args.out_dir}" + (f" (subset run: names end in {suffix})" if suffix else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())