"""Face-box detection matching the Phase-3 preprocessing contract."""
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ..config import MP_MODEL, CROP_PADDING, TRAIN_ASPECT_RANGE


def make_landmarker(model_path=MP_MODEL, num_faces=2):
    """Creates a MediaPipe FaceLandmarker in single-image mode.

    It looks for two faces, so that a second person in view can be noticed; the subject
    is the larger of the two (landmark_box). A face alone in the frame gets the same
    landmarks whichever number is asked for.
    """
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=num_faces)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def face_boxes(frame_rgb, landmarker):
    """Returns every face's relative (x0, y0, x1, y1) landmark box, largest first."""
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
    boxes = []
    for pts in landmarker.detect(img).face_landmarks:
        xs = np.fromiter((p.x for p in pts), dtype=np.float64)
        ys = np.fromiter((p.y for p in pts), dtype=np.float64)
        boxes.append((xs.min(), ys.min(), xs.max(), ys.max()))
    return sorted(boxes, key=box_area, reverse=True)


def landmark_box(frame_rgb, landmarker):
    """Returns the subject's relative landmark box - the largest face's - or None."""
    boxes = face_boxes(frame_rgb, landmarker)
    return boxes[0] if boxes else None


def box_area(box):
    """The area of a relative (x0, y0, x1, y1) box."""
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def share_inside(face, box):
    """Returns the share of a face box's area that lies inside another box."""
    shared = box_area((max(face[0], box[0]), max(face[1], box[1]),
                       min(face[2], box[2]), min(face[3], box[3])))
    return float(shared / max(box_area(face), 1e-12))


def square_padded_box(boxes, padding=CROP_PADDING):
    """Reduces landmark boxes to one centered square box expanded by `padding`."""
    x0, y0, x1, y1 = np.median(np.asarray(boxes, dtype=np.float64), axis=0)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = max(x1 - x0, y1 - y0) * (1.0 + padding) / 2.0
    return (cx - half, cy - half, cx + half, cy + half)


def box_clamp_report(box, width, height):
    """Reports how far the padded box falls outside the frame.

    The contract box is square in relative coordinates, so on a non-square frame
    it is deliberately rectangular in pixels; training applied the same stretch.
    `distortion` therefore compares the kept region against that expected pixel
    aspect rather than against 1.0, and reads 1.0 whenever nothing was clamped.

    A clamped box yields a crop whose aspect differs from the training stretch,
    which puts the model input off-distribution.

    `frac_side_lost` is the share of the box's side that falls outside the frame, on
    the worse axis - the definition every Phase-3 store records as box_side_lost. Each
    axis is measured against its own side: the box is square in relative coordinates
    but not in pixels, so overflow at the top measured against the width would read
    3/4 of its size on a 4:3 frame, and 9/16 on a 16:9 one.
    """
    px = np.array([box[0] * width, box[1] * height, box[2] * width, box[3] * height])
    over = np.array([max(0.0, -px[0]), max(0.0, -px[1]),
                     max(0.0, px[2] - width), max(0.0, px[3] - height)])
    full_w, full_h = px[2] - px[0], px[3] - px[1]
    kept_w = min(px[2], width) - max(px[0], 0.0)
    kept_h = min(px[3], height) - max(px[1], 0.0)
    expected_aspect = float(full_w / max(full_h, 1e-9))
    kept_aspect = float(kept_w / max(kept_h, 1e-9))
    lost_w = (over[0] + over[2]) / max(full_w, 1e-9)
    lost_h = (over[1] + over[3]) / max(full_h, 1e-9)
    return dict(clamped=bool(over.any()),
                overflow_px=over,
                expected_aspect=expected_aspect,
                kept_aspect=kept_aspect,
                distortion=float(kept_aspect / max(expected_aspect, 1e-9)),
                aspect_in_training_range=bool(
                    TRAIN_ASPECT_RANGE[0] <= expected_aspect <= TRAIN_ASPECT_RANGE[1]),
                frac_side_lost=float(max(lost_w, lost_h)))