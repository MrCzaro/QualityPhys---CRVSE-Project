"""Face-box detection matching the Phase-3 preprocessing contract."""
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from ..config import MP_MODEL, CROP_PADDING


def make_landmarker(model_path=MP_MODEL):
    """Creates a MediaPipe FaceLandmarker in single-image mode."""
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE, num_faces=1)
    return mp_vision.FaceLandmarker.create_from_options(opts)


def landmark_box(frame_rgb, landmarker):
    """Returns the relative (x0, y0, x1, y1) landmark bounding box, or None."""
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
    res = landmarker.detect(img)
    if not res.face_landmarks:
        return None
    pts = res.face_landmarks[0]
    xs = np.fromiter((p.x for p in pts), dtype=np.float64)
    ys = np.fromiter((p.y for p in pts), dtype=np.float64)
    return (xs.min(), ys.min(), xs.max(), ys.max())


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
    """
    px = np.array([box[0] * width, box[1] * height, box[2] * width, box[3] * height])
    over = np.array([max(0.0, -px[0]), max(0.0, -px[1]),
                     max(0.0, px[2] - width), max(0.0, px[3] - height)])
    full_w, full_h = px[2] - px[0], px[3] - px[1]
    kept_w = min(px[2], width) - max(px[0], 0.0)
    kept_h = min(px[3], height) - max(px[1], 0.0)
    expected_aspect = float(full_w / max(full_h, 1e-9))
    kept_aspect = float(kept_w / max(kept_h, 1e-9))
    return dict(clamped=bool(over.any()),
                overflow_px=over,
                expected_aspect=expected_aspect,
                kept_aspect=kept_aspect,
                distortion=float(kept_aspect / max(expected_aspect, 1e-9)),
                frac_side_lost=float(over.sum() / max(full_w, 1e-9)))
