"""Capture-then-analyze orchestration for the live_vitals app.

A single face box is computed once per capture and held for its duration, which
matches how the Phase-3 models were trained: the preprocessing notebooks derived
one box per recording rather than tracking per frame.
"""
from dataclasses import dataclass, field

import numpy as np
import cv2

from .. import config
from ..preprocess.face_box import (
    make_landmarker, landmark_box, square_padded_box, box_clamp_report)
from ..preprocess.frames import crop_resize


@dataclass
class CaptureQuality:
    """Acquisition diagnostics reported alongside every reading."""
    width: int
    height: int
    source_fps: float
    effective_fps: float
    n_frames: int
    detections: int
    detections_attempted: int
    verdict: str
    crop_aspect: float
    frac_side_lost: float
    notes: list = field(default_factory=list)

    @property
    def usable(self):
        return self.verdict != "REJECT"


def decimation_stride(source_fps):
    """Returns the frame stride bringing a source rate near the training rate."""
    return max(1, int(round(source_fps / config.TARGET_FPS)))


def framing_verdict(box, width, height):
    """Classifies a capture's framing against the training contract."""
    report = box_clamp_report(box, width, height)
    notes = []
    if report["frac_side_lost"] > config.MAX_BOX_SIDE_LOST:
        verdict = "REJECT"
        notes.append("face box extends past the frame; sit further back")
    elif not report["aspect_in_training_range"]:
        verdict = "WARN"
        notes.append(f"crop aspect {report['expected_aspect']:.3f} is outside the "
                     f"training range; record in landscape")
    else:
        verdict = "ACCEPT"
    return verdict, report, notes


def rate_verdict(effective_fps):
    """Classifies an acquisition rate against the model's training rate.

    Rate is a hard part of the contract rather than an advisory note. Window
    confidence measures spectral concentration inside a window and cannot detect
    that the whole time base is wrong, so a low-rate capture can otherwise report
    a confident but incorrect value.
    """
    if not np.isfinite(effective_fps) or effective_fps < config.MIN_FPS_WARN:
        return "REJECT", (f"{effective_fps:.1f} fps is far below the "
                          f"{config.TARGET_FPS:.0f} fps the model was trained on; "
                          f"improve lighting or lower the capture resolution")
    if effective_fps < config.MIN_FPS_ACCEPT:
        return "WARN", (f"{effective_fps:.1f} fps is below the "
                        f"{config.TARGET_FPS:.0f} fps training rate; "
                        f"treat as indicative")
    return "ACCEPT", None


def worst_verdict(*verdicts):
    """Returns the most severe of the given verdicts."""
    for level in ("REJECT", "WARN"):
        if level in verdicts:
            return level
    return "ACCEPT"


def stable_box(frames_rgb, landmarker):
    """Returns the median square face box over sample frames, or None."""
    boxes = [b for b in (landmark_box(f, landmarker) for f in frames_rgb)
             if b is not None]
    if not boxes:
        return None, 0
    return square_padded_box(boxes), len(boxes)


def crops_from_video(video_path, landmarker=None):
    """Returns (clip, effective_fps, quality) for a recorded video."""
    landmarker = landmarker or make_landmarker()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_total <= 0:
        probe = cv2.VideoCapture(str(video_path))
        n_total = 0
        while probe.grab():
            n_total += 1
        probe.release()

    sample_at = sorted(set(np.linspace(0, max(n_total - 1, 0),
                                       config.N_DETECT_FRAMES).astype(int)))
    samples = []
    for index in sample_at:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, bgr = cap.read()
        if ok:
            samples.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    box, detections = stable_box(samples, landmarker)
    if box is None:
        cap.release()
        raise RuntimeError(f"no face detected in {video_path}")

    verdict, report, notes = framing_verdict(box, width, height)
    stride = decimation_stride(source_fps)
    effective_fps = source_fps / stride
    rate, rate_note = rate_verdict(effective_fps)
    if rate_note:
        notes.append(rate_note)
    verdict = worst_verdict(verdict, rate)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    clip, index = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if index % stride == 0:
            crop = crop_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), box)
            if crop is not None:
                clip.append(crop)
        index += 1
    cap.release()

    quality = CaptureQuality(
        width=width, height=height, source_fps=source_fps,
        effective_fps=effective_fps, n_frames=len(clip),
        detections=detections, detections_attempted=len(sample_at),
        verdict=verdict, crop_aspect=report["expected_aspect"],
        frac_side_lost=report["frac_side_lost"], notes=notes)
    return np.asarray(clip, dtype=np.uint8), effective_fps, quality


def crops_from_camera(camera, seconds, landmarker=None, on_progress=None):
    """Captures for a fixed duration, cropping with one frozen box.

    The box is derived from a short pre-roll and then held, so the crop is stable
    for the whole capture. Full frames are never retained or written.
    """
    landmarker = landmarker or make_landmarker()
    measured = camera.warm_up()
    rate, rate_note = rate_verdict(measured)
    if rate == "REJECT":
        raise RuntimeError(f"camera unusable: {rate_note}")
    width, height = camera.frame_size

    preroll = []
    for _ in range(config.N_DETECT_FRAMES):
        ok, rgb, _ = camera.read()
        if ok:
            preroll.append(rgb)
    box, detections = stable_box(preroll, landmarker)
    if box is None:
        raise RuntimeError("no face detected; check lighting and position")

    verdict, report, notes = framing_verdict(box, width, height)

    clip, stamps = [], []
    start = None
    while True:
        ok, rgb, stamp = camera.read()
        if not ok:
            continue
        start = start if start is not None else stamp
        elapsed = stamp - start
        if elapsed > seconds:
            break
        crop = crop_resize(rgb, box)
        if crop is not None:
            clip.append(crop)
            stamps.append(stamp)
        if on_progress:
            on_progress(elapsed, seconds)

    # Measured delivery rate, not the rate the device claims.
    if len(stamps) > 1:
        effective_fps = (len(stamps) - 1) / (stamps[-1] - stamps[0])
    else:
        effective_fps = float("nan")
    rate, rate_note = rate_verdict(effective_fps)
    if rate_note:
        notes.append(f"camera delivered {rate_note}")
    verdict = worst_verdict(verdict, rate)

    quality = CaptureQuality(
        width=width, height=height, source_fps=effective_fps,
        effective_fps=effective_fps, n_frames=len(clip),
        detections=detections, detections_attempted=config.N_DETECT_FRAMES,
        verdict=verdict, crop_aspect=report["expected_aspect"],
        frac_side_lost=report["frac_side_lost"], notes=notes)
    return np.asarray(clip, dtype=np.uint8), effective_fps, quality