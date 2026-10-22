"""Measures, on raw videos, what the app's face checks see.

The app holds one face box for a whole capture: the median of the subject's landmark
boxes in N_DETECT_FRAMES frames spread evenly over it. Three checks judge the face over
those samples (capture/session.py, face_verdict; thresholds in config.py):

  faces     how many of the samples find the face;
  second    samples with a second face at least SECOND_FACE_MIN_WIDTH of the subject's
            width;
  outside   how much of the subject's face lies outside the held box, at each sample. A
            face drifting out of the crop passes the motion gate, which reads movement
            in the cardiac band only.

This script measured them before their thresholds were set (2026-10-21); run again, it
shows what they decide on the same videos. Detection is the app's own: the same sample
positions, landmarker (two faces, the larger the subject), box rule and verdict. A
one-face landmarker, as the app used before, runs beside it to show whether it would
have taken the other face. Each video's framing verdict (how the held box fits the
frame) is recorded too, with the box itself, so a change to the framing rule can be
judged on these videos without detecting again.

Frames are fetched by seeking, not by the app's front-to-back pass: the files here are
indexed (AVI, MP4, MOV), and a sample a frame or two off changes no box. A file whose
container declares no frame count is read front to back instead, as the app reads it.

Every video file under each --root is measured; folders whose names start with a dot
are skipped. Results are grouped by the first folder under the root, and files directly
in a root form their own group.

Writes face_boxes.csv (one row per video; git-ignored) and summary_face_boxes.json
(aggregates per group) to --out-dir. Nothing written contains a frame.

--detail VIDEO prints one video's samples instead - each sample's time, faces and share
of the face inside the held box, then the verdict - reading the file front to back as the
app does, so a frame damaged by seeking cannot pass for a missed face. It writes nothing.

Usage:
    python -m app.live_vitals.scripts.check_face_boxes [--root DIR ...] [--out-dir DIR]
        [--limit N]
    python -m app.live_vitals.scripts.check_face_boxes --detail VIDEO [--detail VIDEO ...]
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

from app.live_vitals import config  # noqa: E402
from app.live_vitals.capture.session import (  # noqa: E402
    detection_positions, face_verdict, framing_verdict, opened, read_through, second_faces,
    stable_box)
from app.live_vitals.preprocess.face_box import (  # noqa: E402
    box_area, face_boxes, landmark_box, make_landmarker, share_inside, square_padded_box)
from app.live_vitals.scripts.bench_outputs import (  # noqa: E402
    file_sha256, provenance, subset_suffix)

DEFAULT_ROOTS = [Path(r"D:\QualityPhys\demo_data")]
DEFAULT_OUT_DIR = _REPO_ROOT / "Data" / "face_gate"
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}
ROOT_GROUP = "(files in the root)"
NAN = float("nan")


def width_of(box):
    return box[2] - box[0]


def iou(a, b):
    """Intersection over union of two relative boxes."""
    shared = share_inside(a, b) * box_area(a)
    return shared / max(box_area(a) + box_area(b) - shared, 1e-12)


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


def measure(path, landmarker, one_face):
    """Measures one video; returns its row."""
    frames, n_frames, info = sample_frames(path)
    read = [frame for frame in frames if frame is not None]
    # Frames arrive upright, so a quarter-turned file's sides swap.
    height, width = read[0].shape[:2] if read else (info["height"], info["width"])
    box, detections, faces = stable_box(read, landmarker)
    one_face_boxes = [landmark_box(frame, one_face) for frame in read]
    seconds = [width_of(boxes[1]) / width_of(boxes[0]) for boxes in faces if len(boxes) > 1]
    fps = info["fps"]
    row = dict(n_frames=n_frames, width=width, height=height, fps=round(fps, 3),
               codec=info["codec"], rotation=info["rotation"],
               duration_s=round(n_frames / fps, 1) if fps > 0 else NAN,
               samples=len(frames), read=len(read), faces=detections,
               faces_one_face=sum(b is not None for b in one_face_boxes),
               second_samples=len(seconds),
               second_width_max=round(float(max(seconds)), 3) if seconds else 0.0,
               near_second_samples=second_faces(faces),
               one_face_other_samples=sum(
                   1 for boxes, one in zip(faces, one_face_boxes)
                   if boxes and one is not None and iou(one, boxes[0]) < 0.5))
    if box is None:
        return dict(row, verdict="", reasons="no face in any sample", inside_min=NAN,
                    outside_samples=0, shift_max=NAN, shift_median=NAN, width_ratio=NAN,
                    framing="", frac_side_lost=NAN, box_x0=NAN, box_y0=NAN, box_x1=NAN,
                    box_y1=NAN)
    verdict, gate_notes, _ = face_verdict(faces, box)
    framing, report, _ = framing_verdict(box, width, height)
    subjects = [boxes[0] for boxes in faces if boxes]
    inside = [share_inside(face, box) for face in subjects]
    # Distances in pixels, in units of the subject's median face width: the relative
    # coordinates are stretched differently along each axis of a non-square frame.
    centres = np.array([((b[0] + b[2]) / 2 * width, (b[1] + b[3]) / 2 * height)
                        for b in subjects])
    widths = np.array([width_of(b) * width for b in subjects])
    shifts = np.hypot(*(centres - np.median(centres, axis=0)).T) / np.median(widths)
    return dict(row, verdict=verdict, reasons="; ".join(note for _, note in gate_notes),
                inside_min=round(min(inside), 4),
                outside_samples=int(sum(s < 1.0 - config.FACE_OUTSIDE_WARN for s in inside)),
                shift_max=round(float(shifts.max()), 4),
                shift_median=round(float(np.median(shifts)), 4),
                width_ratio=round(float(widths.max() / widths.min()), 4),
                framing=framing, frac_side_lost=round(report["frac_side_lost"], 4),
                **{f"box_{name}": round(float(value), 4)
                   for name, value in zip(("x0", "y0", "x1", "y1"), box)})


def detail(path, landmarker):
    """Prints each sample of one video, read front to back as the app reads it."""
    with opened(path) as cap:
        if not cap.isOpened():
            raise SystemExit(f"ERROR: OpenCV cannot open {path}")
        n_frames = 0
        while cap.grab():
            n_frames += 1
    positions = detection_positions(n_frames)
    wanted, frames, stamps = set(positions), {}, {}
    with opened(path) as cap:
        for index in range(n_frames):
            if not cap.grab():
                break
            if index in wanted:
                ok, bgr = cap.retrieve()
                frames[index] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None
                stamps[index] = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    found = {p: face_boxes(frames[p], landmarker) for p in positions
             if frames.get(p) is not None}
    subjects = [boxes[0] for boxes in found.values() if boxes]
    box = square_padded_box(subjects) if subjects else None
    print(f"\n{path}\n{n_frames} frames read front to back; the face in {len(subjects)} of "
          f"{len(found)} samples read")
    print(f"{'sample':>6} {'frame':>6} {'time s':>7} {'faces':>5} {'second/subject':>14} "
          f"{'inside':>7}")
    for i, p in enumerate(positions, 1):
        time_s = stamps.get(p, NAN)
        if p not in found:
            print(f"{i:6d} {p:6d} {time_s:7.1f}   frame not read")
            continue
        boxes = found[p]
        second = (f"{width_of(boxes[1]) / width_of(boxes[0]):14.2f}" if len(boxes) > 1
                  else f"{'-':>14}")
        inside = f"{share_inside(boxes[0], box):7.3f}" if boxes else f"{'-':>7}"
        print(f"{i:6d} {p:6d} {time_s:7.1f} {len(boxes):5d} {second} {inside}")
    if box is not None:
        verdict, gate_notes, _ = face_verdict(list(found.values()), box)
        print(f"verdict: {verdict}" + "".join(f"\n  {level}: {note}" for level, note in gate_notes))


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
    """Aggregates per group: counts, verdicts and the spread of each measure."""
    out = {}
    for group in dict.fromkeys(r["group"] for r in rows):
        sel = [r for r in rows if r["group"] == group]
        ok = [r for r in sel if not r["error"]]
        faces = [r["faces"] for r in ok]
        out[group] = dict(
            n_videos=len(sel), n_unreadable=len(sel) - len(ok),
            faces_by_count={str(k): faces.count(k) for k in sorted(set(faces), reverse=True)},
            all_read_samples_found=sum(r["faces"] == r["read"] for r in ok),
            faces_differ_from_one_face=sum(r["faces"] != r["faces_one_face"] for r in ok),
            with_second_face=sum(r["second_samples"] > 0 for r in ok),
            with_near_second_face=sum(r["near_second_samples"] > 0 for r in ok),
            one_face_took_other_face=sum(r["one_face_other_samples"] > 0 for r in ok),
            with_face_partly_outside=sum(r["outside_samples"] > 0 for r in ok),
            verdicts={v: sum(r["verdict"] == v for r in ok) for v in ("ACCEPT", "WARN", "REJECT")},
            framing={v: sum(r["framing"] == v for r in ok) for v in ("ACCEPT", "WARN", "REJECT")},
            clamped=sum(r["frac_side_lost"] > config.MAX_BOX_SIDE_LOST for r in ok),
            inside_min=quantiles([r["inside_min"] for r in ok], (0.0, 0.05, 0.5)),
            shift_max=quantiles([r["shift_max"] for r in ok], (0.5, 0.95, 1.0)),
            width_ratio=quantiles([r["width_ratio"] for r in ok], (0.5, 0.95, 1.0)))
    return out


def print_summary(summary):
    print(f"\n{'group':26} {'videos':>6} {'all':>4} {'fewest':>6} {'2nd':>4} {'near':>4} "
          f"{'1-face':>6} {'out':>4} {'warn':>4} {'refuse':>6} {'clamp':>5} | {'inside min':>10} "
          f"{'shift p95':>9} {'shift max':>9} {'width max':>9}")
    for group, s in summary.items():
        fewest = min((int(k) for k in s["faces_by_count"]), default=0)
        print(f"{group[:26]:26} {s['n_videos']:6d} {s['all_read_samples_found']:4d} {fewest:6d} "
              f"{s['with_second_face']:4d} {s['with_near_second_face']:4d} "
              f"{s['one_face_took_other_face']:6d} {s['with_face_partly_outside']:4d} "
              f"{s['verdicts']['WARN']:4d} {s['verdicts']['REJECT']:6d} {s['clamped']:5d} | "
              f"{s['inside_min'].get('p00', NAN):10.3f} "
              f"{s['shift_max'].get('p95', NAN):9.3f} "
              f"{s['shift_max'].get('p100', NAN):9.3f} "
              f"{s['width_ratio'].get('p100', NAN):9.3f}")
        if s["n_unreadable"]:
            print(f"{'':26} {s['n_unreadable']} unreadable")
    print("\nall: videos with the face in every sample read; fewest: the fewest samples with"
          "\na face; 2nd / near: videos with a second face (near: at least"
          f" {config.SECOND_FACE_MIN_WIDTH:g} of the subject's"
          "\nwidth); 1-face: videos where a one-face detector took the other face; out:"
          f"\nvideos with a sample whose face is more than {config.FACE_OUTSIDE_WARN:.0%} outside"
          " the held box;"
          "\nwarn / refuse: the face checks' verdict; clamp: videos whose held box has more"
          f"\nthan {config.MAX_BOX_SIDE_LOST:.0%} of its width or height outside the frame, which"
          " the framing gate refuses;"
          "\ninside min: the lowest share of a face inside the held box, over all videos;"
          "\nshift: a face's distance from its median position, in face widths; width max: the"
          "\nlargest ratio of a video's widest to narrowest face.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, action="append",
                        help=f"folder to search for videos; repeat for several "
                             f"(default: {DEFAULT_ROOTS[0]})")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="measure only the first N videos (smoke test)")
    parser.add_argument("--detail", type=Path, action="append",
                        help="print one video's samples, read front to back; repeat for "
                             "several (writes nothing)")
    args = parser.parse_args()

    if args.detail:
        landmarker = make_landmarker()
        try:
            for path in args.detail:
                detail(path, landmarker)
        finally:
            landmarker.close()
        return 0

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
    landmarker, one_face = make_landmarker(), make_landmarker(num_faces=1)
    rows, t0 = [], time.perf_counter()
    try:
        for i, (group, path) in enumerate(videos, 1):
            root = next(r for r in roots if path.is_relative_to(r))
            row = dict(group=group, file=str(path.relative_to(root)), error="")
            try:
                row.update(measure(path, landmarker, one_face))
                print(f"[{i}/{len(videos)}] {row['file']}: faces {row['faces']}/"
                      f"{row['read']}, second {row['second_samples']}, inside min "
                      f"{row['inside_min']:.3f}, {row['verdict'] or 'no face'}", flush=True)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[{i}/{len(videos)}] {row['file']}: {row['error']}", flush=True)
            rows.append(row)
    finally:
        landmarker.close()
        one_face.close()
    print(f"measured in {(time.perf_counter() - t0) / 60:.1f} min")

    summary = dict(
        settings=dict(n_detect_frames=config.N_DETECT_FRAMES, crop_padding=config.CROP_PADDING,
                      face_samples_warn=config.FACE_SAMPLES_WARN,
                      face_samples_reject=config.FACE_SAMPLES_REJECT,
                      second_face_min_width=config.SECOND_FACE_MIN_WIDTH,
                      face_outside_warn=config.FACE_OUTSIDE_WARN,
                      face_inside_reject=config.FACE_INSIDE_REJECT,
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