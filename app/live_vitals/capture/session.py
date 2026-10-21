"""Capture-then-analyze orchestration for the live_vitals app.

A single face box is computed once per capture and held for its duration, which
matches how the Phase-3 models were trained: the preprocessing notebooks derived
one box per recording rather than tracking per frame.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
import time

import numpy as np
import cv2

from .. import config
from ..preprocess.face_box import (
    make_landmarker, face_boxes, square_padded_box, box_clamp_report, share_inside)
from ..preprocess.frames import crop_resize
from .motion import motion_verdict


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
    # The notes behind the verdict: why the capture was refused (REJECT) or flagged
    # (WARN). Empty when it was accepted; the other notes (frame timestamps, the
    # container's rate) inform without deciding anything.
    reasons: list = field(default_factory=list)
    # Head motion in the crop (capture/motion.py): the median in-band displacement in crop
    # pixels and the share of windows with a clear rhythm. NaN when the clip is too short to judge.
    motion_px: float = float("nan")
    motion_periodic: float = float("nan")
    # The face over the detection samples (face_verdict): how many samples showed a second
    # face at least SECOND_FACE_MIN_WIDTH of the subject's width, and the smallest share of
    # the subject's face inside the held box. NaN when not judged.
    second_face_samples: int = 0
    face_inside_min: float = float("nan")

    @property
    def usable(self):
        return self.verdict != "REJECT"


def decimation_stride(source_fps):
    """Returns the frame stride bringing a source rate near the training rate."""
    # A container that declares no rate, or NaN, would otherwise raise inside
    # int(round(...)) before rate_verdict ever gets to refuse the capture.
    if not np.isfinite(source_fps) or source_fps <= 0:
        return 1
    return max(1, int(round(source_fps / config.TARGET_FPS)))


def measured_rate(timestamps_ms, declared_fps):
    """Returns (fps, note): the rate the frames were actually delivered at.

    A container's declared rate is a claim, and the readout scales heart rate
    linearly with whatever rate it is handed -- a rate 4% high reports a heart
    rate 4% high, with full confidence and no other symptom. Browser
    MediaRecorder output is variable-rate and routinely declares an average it
    did not hold, and three MCD-rPPG recordings declare a rate their own frame
    count contradicts. The frames' presentation timestamps are the recording's
    own account of when they arrived, so they are preferred wherever the
    container provides them.
    """
    if len(timestamps_ms) > 1:
        span = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
        if span > 0:
            return (len(timestamps_ms) - 1) / span, None
    return declared_fps, ("frame timestamps unavailable; the rate is the "
                          "container's claim and has not been verified")


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
    if effective_fps > config.MAX_FPS_ACCEPT:
        return "WARN", (f"{effective_fps:.1f} fps is above the "
                        f"{config.TARGET_FPS:.0f} fps training rate, so each window "
                        f"spans less time than in training; treat as indicative")
    return "ACCEPT", None


def motion_fields(summary):
    """The CaptureQuality motion fields from a motion_verdict summary (None: not judged)."""
    if summary is None:
        return {}
    return dict(motion_px=summary["amp_median"], motion_periodic=summary["clear_frac"])


def worst_verdict(*verdicts):
    """Returns the most severe of the given verdicts."""
    for level in ("REJECT", "WARN"):
        if level in verdicts:
            return level
    return "ACCEPT"


def stable_box(frames_rgb, landmarker):
    """Returns (box, detections, faces) over sample frames.

    faces holds every face found in each frame, largest first (an empty list where none
    was). The subject is the largest face: box is the median square box of its landmark
    boxes, or None when no frame found a face, and detections how many frames found it.
    """
    faces = [face_boxes(frame, landmarker) for frame in frames_rgb]
    subjects = [boxes[0] for boxes in faces if boxes]
    if not subjects:
        return None, 0, faces
    return square_padded_box(subjects), len(subjects), faces


def second_faces(faces):
    """Counts the samples whose second face is at least SECOND_FACE_MIN_WIDTH as wide as
    the subject's. A smaller one - a face in a picture, someone far behind - is ignored."""
    count = 0
    for boxes in faces:
        if len(boxes) > 1:
            subject, other = boxes[0], boxes[1]
            count += int(other[2] - other[0]
                         >= config.SECOND_FACE_MIN_WIDTH * (subject[2] - subject[0]))
    return count


def face_verdict(faces, box):
    """Judges the face over a capture's sample frames; returns (verdict, gate_notes, summary).

    faces holds every face found in each sample, largest first, and box is the box held
    for the capture (stable_box). Three checks, each at its own level, so gate_notes pairs
    every note with its level:
      found    the share of samples that found the face: below FACE_SAMPLES_WARN warns,
               below FACE_SAMPLES_REJECT refuses. A sample without it caught the face
               turned away, covered or out of view, and the box rests on fewer samples.
      second   a second face at least SECOND_FACE_MIN_WIDTH of the subject's width warns.
               The box follows the larger face, but another person in view can move into
               the crop or change its light.
      outside  the share of the subject's face inside the held box, at each sample: more
               than FACE_OUTSIDE_WARN outside warns, less than FACE_INSIDE_REJECT inside
               refuses. The motion gate reads movement in the cardiac band only, so a face
               drifting out of the crop passes it.
    """
    n = len(faces)
    subjects = [boxes[0] for boxes in faces if boxes]
    gate_notes = []
    share_found = len(subjects) / max(n, 1)
    if share_found < config.FACE_SAMPLES_REJECT:
        gate_notes.append(("REJECT", f"the face was found in only {len(subjects)} of {n} "
                                     f"sample frames; keep your whole face in view"))
    elif share_found < config.FACE_SAMPLES_WARN:
        gate_notes.append(("WARN", f"the face was missed in {n - len(subjects)} of {n} "
                                   f"sample frames; treat as indicative"))
    second = second_faces(faces)
    if second:
        gate_notes.append(("WARN", f"a second face was in view in {second} of {n} sample "
                                   f"frames; only one person should face the camera"))
    inside = [share_inside(face, box) for face in subjects]
    inside_min = min(inside) if inside else float("nan")
    outside = sum(share < 1.0 - config.FACE_OUTSIDE_WARN for share in inside)
    if inside and inside_min < config.FACE_INSIDE_REJECT:
        gate_notes.append(("REJECT", f"the face moved out of the analysed area (at worst "
                                     f"{inside_min:.0%} of it inside); keep still and centred"))
    elif outside:
        gate_notes.append(("WARN", f"part of the face left the analysed area in {outside} of "
                                   f"{n} sample frames; treat as indicative"))
    verdict = worst_verdict(*(level for level, _ in gate_notes))
    return verdict, gate_notes, dict(second_face_samples=second, face_inside_min=inside_min)


def detection_positions(n_frames):
    """Returns the indices of the detection samples, spread evenly over n_frames."""
    return sorted(set(np.linspace(0, max(n_frames - 1, 0),
                                  config.N_DETECT_FRAMES).astype(int)))


@contextmanager
def opened(video_path):
    """Opens a video for reading, and releases it however the block is left.

    A read that raises would otherwise leave the file open until the capture object
    was collected, and on Windows an open file cannot be deleted: the web app's copy
    of an upload would outlive its request.
    """
    cap = cv2.VideoCapture(str(video_path))
    try:
        yield cap
    finally:
        cap.release()


def read_through(video_path, sample_at, max_frames=None):
    """Reads a video front to back: every frame's timestamp and the frames at sample_at.

    Returns (timestamps_ms, samples_rgb). Frames that are not sampled are only grabbed,
    never converted, so the pass costs little more than the decode itself. A video
    longer than max_frames raises ValueError when its next frame is reached.
    """
    wanted = set(sample_at)
    stamps, samples = [], []
    with opened(video_path) as cap:
        while cap.grab():
            if max_frames is not None and len(stamps) == max_frames:
                raise ValueError(f"{video_path} has more than {max_frames} frames")
            if len(stamps) in wanted:
                ok, bgr = cap.retrieve()
                if ok:
                    samples.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    return stamps, samples


def crops_from_video(video_path, landmarker=None, max_frames=None):
    """Returns (clip, effective_fps, quality) for a recorded video.

    The file is only ever read front to back, never seeked. A browser recording
    (MediaRecorder WebM) has no seek index and declares neither a frame rate nor
    a frame count: OpenCV reports ffmpeg's guess from the first timestamps as the
    rate (1000 fps, the 1 ms timebase, when they jitter) and derives a count from
    it, and a seek into the file can return the wrong frame or end the stream.
    So a first pass counts the frames and reads their timestamps, the detection
    samples are spread over the frames that exist, and the decimation stride
    follows the rate the frames were delivered at.

    With max_frames, a longer video is refused (ValueError) by the first pass,
    before any frame is cropped: a clip costs memory in proportion to its length,
    whatever the size of its file.
    """
    landmarker = landmarker or make_landmarker()
    with opened(video_path) as cap:
        if not cap.isOpened():
            raise FileNotFoundError(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        n_declared = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_at = detection_positions(n_declared)
    all_stamps, samples = read_through(video_path, sample_at, max_frames)
    count_held = len(all_stamps) == n_declared
    if not count_held:
        # The samples were placed by a count the file does not hold; a second pass
        # places them over the frames it has.
        sample_at = detection_positions(len(all_stamps))
        _, samples = read_through(video_path, sample_at, max_frames)

    box, detections, faces = stable_box(samples, landmarker)
    if box is None:
        raise RuntimeError(f"no face detected in {video_path}")

    verdict, report, notes = framing_verdict(box, width, height)
    # Each gate's notes beside its level, so the ones behind the final verdict can be
    # told apart from those that only inform.
    gate_notes = [(verdict, note) for note in notes]
    face, face_notes, face_summary = face_verdict(faces, box)
    notes.extend(note for _, note in face_notes)
    gate_notes.extend(face_notes)
    # The stride follows the delivered rate: from a guessed 1000 fps, a WebM capture
    # would keep one frame in 33.
    delivered_fps, _ = measured_rate(all_stamps, source_fps)
    stride = decimation_stride(delivered_fps)

    clip, stamps, index = [], [], 0
    with opened(video_path) as cap:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if index % stride == 0:
                crop = crop_resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), box)
                if crop is not None:
                    clip.append(crop)
                    stamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
            index += 1

    # The rate verdict waits for the decode: until the frames have been read
    # there is nothing to judge but the container's own claim about them.
    declared_fps = source_fps / stride
    effective_fps, stamp_note = measured_rate(stamps, declared_fps)
    if stamp_note:
        notes.append(stamp_note)
    # Only a container whose frame count held is held to its rate: a MediaRecorder
    # WebM miscounts because it declares neither, and its rate is only a guess.
    if count_held and declared_fps > 0 and np.isfinite(effective_fps):
        drift = abs(effective_fps - declared_fps) / declared_fps
        if drift > 0.02:
            notes.append(f"container declares {declared_fps:.2f} fps but "
                         f"delivered {effective_fps:.2f}; reading the declared "
                         f"rate would have biased the result by {100 * drift:.1f}%")
    rate, rate_note = rate_verdict(effective_fps)
    if rate_note:
        notes.append(rate_note)
        gate_notes.append((rate, rate_note))
    clip = np.asarray(clip, dtype=np.uint8)
    motion, motion_note, motion_summary = motion_verdict(clip, effective_fps)
    if motion_note:
        notes.append(motion_note)
        gate_notes.append((motion, motion_note))
    verdict = worst_verdict(verdict, face, rate, motion)

    quality = CaptureQuality(
        width=width, height=height, source_fps=source_fps,
        effective_fps=effective_fps, n_frames=len(clip),
        detections=detections, detections_attempted=len(sample_at),
        verdict=verdict, crop_aspect=report["expected_aspect"],
        frac_side_lost=report["frac_side_lost"], notes=notes,
        reasons=[note for level, note in gate_notes if level == verdict],
        **motion_fields(motion_summary), **face_summary)
    return clip, effective_fps, quality


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
    box, detections, faces = stable_box(preroll, landmarker)
    if box is None:
        raise RuntimeError("no face detected; check lighting and position")

    verdict, report, notes = framing_verdict(box, width, height)
    gate_notes = [(verdict, note) for note in notes]
    face, face_notes, face_summary = face_verdict(faces, box)
    notes.extend(note for _, note in face_notes)
    gate_notes.extend(face_notes)

    clip, stamps = [], []
    start = None
    last_delivery = time.perf_counter()
    while True:
        ok, rgb, stamp = camera.read()
        if not ok:
            # Only a delivered frame advances the clock that ends the capture, so
            # a camera that stops delivering would otherwise hold this loop forever.
            if time.perf_counter() - last_delivery > config.CAMERA_STALL_SECONDS:
                raise RuntimeError(f"camera delivered no frame for "
                                   f"{config.CAMERA_STALL_SECONDS:.0f} s; "
                                   f"capture abandoned")
            continue
        last_delivery = time.perf_counter()
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
        gate_notes.append((rate, notes[-1]))
    clip = np.asarray(clip, dtype=np.uint8)
    motion, motion_note, motion_summary = motion_verdict(clip, effective_fps)
    if motion_note:
        notes.append(motion_note)
        gate_notes.append((motion, motion_note))
    verdict = worst_verdict(verdict, face, rate, motion)

    quality = CaptureQuality(
        width=width, height=height, source_fps=effective_fps,
        effective_fps=effective_fps, n_frames=len(clip),
        detections=detections, detections_attempted=config.N_DETECT_FRAMES,
        verdict=verdict, crop_aspect=report["expected_aspect"],
        frac_side_lost=report["frac_side_lost"], notes=notes,
        reasons=[note for level, note in gate_notes if level == verdict],
        **motion_fields(motion_summary), **face_summary)
    return clip, effective_fps, quality