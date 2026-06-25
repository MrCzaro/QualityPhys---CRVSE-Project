"""
Face detection and rough ROI debug helpers for one browser-captured frame.

Current purpose:
    Decode one browser-submitted frame.
    Run MediaPipe Tasks Face Landmarker on the RGB image.
    Return simple JSON-safe face/landmark diagnostics.
    Return rough forehead / cheek ROI boxes and RGB summaries.

What this proves:
    Python backend can receive real camera pixels.
    Python backend can detect a face in the frame.
    Python backend can define approximate facial ROIs.
    Python backend can extract per-ROI RGB statistics.

What this does NOT do:
    Does not store frames.
    Does not extract a time-series rPPG signal yet.
    Does not compute POS/CHROM/GREEN yet.
    Does not run HR inference from camera frames yet.

Physiology:
    Forehead and cheeks are useful rPPG candidate regions because skin color
    changes over time can contain pulse-related blood-volume information.

Signal:
    This file only extracts one-frame RGB summaries. rPPG needs many frames:
        frame 1 ROI RGB
        frame 2 ROI RGB
        ...
        8 seconds of ROI RGB
        -> POS / CHROM / GREEN
        -> quality gate
        -> HR model

Limitation:
    ROIs here are rough bbox-proportional rectangles. They are a debug scaffold,
    not a final robust skin/ROI tracker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np

from backend.frame_debug import decode_image_data_url_to_rgb_array, summarize_rgb_frame


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FACE_LANDMARKER_MODEL_PATH = (
    APP_DIR / "models" / "mediapipe" / "face_landmarker.task"
)
_FACE_LANDMARKER_CACHE: dict[str, Any] = {}

def evaluate_roi_image_quality(roi_rgb: np.ndarray) -> dict[str, Any]:
    """
    Evaluate one-frame image quality for a candidate rPPG ROI.

    Parameters
    ----------
    roi_rgb:
        ROI RGB crop with shape (height, width, 3).

    Returns
    -------
    dict[str, Any]
        JSON-safe ROI quality diagnostic.

    Physiology:
        rPPG needs small color changes caused by pulsatile blood-volume changes.
        If pixels are clipped or too dark, those small changes can be lost.

    Signal:
        This function checks whether the ROI has enough usable intensity range.
        It does not detect pulse yet.

    Limitation:
        This is one-frame quality only. A real rPPG quality score also needs
        temporal stability across many frames.
    """

    if roi_rgb.size == 0:
        return {
            "status": "reject",
            "reasons": ["ROI crop is empty."],
            "metrics": {},
        }

    if roi_rgb.ndim != 3 or roi_rgb.shape[2] != 3:
        return {
            "status": "reject",
            "reasons": [f"Expected RGB ROI with shape (H, W, 3), got {roi_rgb.shape}."],
            "metrics": {},
        }

    roi_rgb = np.asarray(roi_rgb)

    if roi_rgb.dtype != np.uint8:
        roi_rgb = roi_rgb.astype(np.uint8)

    roi_float = roi_rgb.astype(np.float32)

    mean_rgb = roi_float.mean(axis=(0, 1))
    std_rgb = roi_float.std(axis=(0, 1))

    red = roi_rgb[:, :, 0]
    green = roi_rgb[:, :, 1]
    blue = roi_rgb[:, :, 2]

    red_saturated_fraction = float(np.mean(red >= 250))
    green_saturated_fraction = float(np.mean(green >= 250))
    blue_saturated_fraction = float(np.mean(blue >= 250))

    any_saturated_fraction = float(
        np.mean(
            (red >= 250)
            | (green >= 250)
            | (blue >= 250)
        )
    )

    red_dark_fraction = float(np.mean(red <= 10))
    green_dark_fraction = float(np.mean(green <= 10))
    blue_dark_fraction = float(np.mean(blue <= 10))

    any_dark_fraction = float(
        np.mean(
            (red <= 10)
            | (green <= 10)
            | (blue <= 10)
        )
    )

    green_mean = float(mean_rgb[1])
    green_std = float(std_rgb[1])
    red_mean = float(mean_rgb[0])
    blue_mean = float(mean_rgb[2])

    channel_mean_range = float(np.max(mean_rgb) - np.min(mean_rgb))

    reasons = []
    status = "ok"

    # Hard reject conditions.
    if any_saturated_fraction >= 0.10:
        status = "reject"
        reasons.append(
            f"Too many ROI pixels are saturated: {any_saturated_fraction:.3f} >= 0.100."
        )

    if red_saturated_fraction >= 0.30:
        status = "reject"
        reasons.append(
            f"Red channel is heavily saturated: {red_saturated_fraction:.3f} >= 0.300."
        )

    if green_saturated_fraction >= 0.20:
        status = "reject"
        reasons.append(
            f"Green channel is heavily saturated: {green_saturated_fraction:.3f} >= 0.200."
        )

    if any_dark_fraction >= 0.20:
        status = "reject"
        reasons.append(
            f"Too many ROI pixels are very dark: {any_dark_fraction:.3f} >= 0.200."
        )

    if green_mean < 20.0:
        status = "reject"
        reasons.append(
            f"Green mean is very low: {green_mean:.1f} < 20.0."
        )

    # Warning conditions.
    if status != "reject":
        if any_saturated_fraction >= 0.02:
            status = "warning"
            reasons.append(
                f"Some ROI pixels are saturated: {any_saturated_fraction:.3f} >= 0.020."
            )

        if red_saturated_fraction >= 0.05:
            status = "warning"
            reasons.append(
                f"Red channel has saturation risk: {red_saturated_fraction:.3f} >= 0.050."
            )

        if green_mean < 40.0:
            status = "warning"
            reasons.append(
                f"Green mean is low: {green_mean:.1f} < 40.0."
            )

        if green_mean > 220.0:
            status = "warning"
            reasons.append(
                f"Green mean is very high: {green_mean:.1f} > 220.0."
            )

        if any_dark_fraction >= 0.05:
            status = "warning"
            reasons.append(
                f"Some ROI pixels are very dark: {any_dark_fraction:.3f} >= 0.050."
            )

        if channel_mean_range > 120.0:
            status = "warning"
            reasons.append(
                f"RGB channel imbalance is high: range={channel_mean_range:.1f} > 120.0."
            )

    if len(reasons) == 0:
        reasons.append("ROI one-frame intensity quality looks acceptable.")

    return {
        "status": status,
        "reasons": reasons,
        "metrics": {
            "red_mean": red_mean,
            "green_mean": green_mean,
            "blue_mean": blue_mean,
            "red_std": float(std_rgb[0]),
            "green_std": green_std,
            "blue_std": float(std_rgb[2]),
            "red_saturated_fraction": red_saturated_fraction,
            "green_saturated_fraction": green_saturated_fraction,
            "blue_saturated_fraction": blue_saturated_fraction,
            "any_saturated_fraction": any_saturated_fraction,
            "red_dark_fraction": red_dark_fraction,
            "green_dark_fraction": green_dark_fraction,
            "blue_dark_fraction": blue_dark_fraction,
            "any_dark_fraction": any_dark_fraction,
            "channel_mean_range": channel_mean_range,
        },
    }


def get_mediapipe_status(
    model_path: Path | None = None,
) -> dict[str, Any]:
    """
    Return MediaPipe Tasks API and model-file status for debugging.

    Parameters
    ----------
    model_path:
        Optional model path. Defaults to app-local Face Landmarker model path.

    Returns
    -------
    dict[str, Any]
        JSON-safe MediaPipe status.
    """

    if model_path is None:
        model_path = DEFAULT_FACE_LANDMARKER_MODEL_PATH

    has_tasks = hasattr(mp, "tasks")
    has_image = hasattr(mp, "Image")
    has_image_format = hasattr(mp, "ImageFormat")

    return {
        "mediapipe_version": getattr(mp, "__version__", None),
        "package_file": getattr(mp, "__file__", None),
        "has_solutions_api": hasattr(mp, "solutions"),
        "has_tasks_api": has_tasks,
        "has_image_class": has_image,
        "has_image_format": has_image_format,
        "model_path": str(model_path),
        "model_exists": model_path.exists(),
    }


def _normalized_landmarks_to_pixel_array(
    landmarks,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Convert MediaPipe normalized landmarks into pixel coordinates.

    Parameters
    ----------
    landmarks:
        MediaPipe normalized landmark list.

    width:
        Image width in pixels.

    height:
        Image height in pixels.

    Returns
    -------
    np.ndarray
        Landmark array with shape (n_landmarks, 3), where columns are:
        x_px, y_px, z_relative.
    """

    points = []

    for landmark in landmarks:
        x_px = float(landmark.x) * float(width)
        y_px = float(landmark.y) * float(height)
        z_relative = float(landmark.z)

        points.append([x_px, y_px, z_relative])

    return np.asarray(points, dtype=np.float32)


def _compute_landmark_bbox(
    landmark_pixels: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    """
    Compute bounding box from landmark pixel coordinates.

    Parameters
    ----------
    landmark_pixels:
        Landmark array with shape (n_landmarks, 3).

    width:
        Image width in pixels.

    height:
        Image height in pixels.

    Returns
    -------
    dict[str, Any]
        JSON-safe bounding box and coverage information.
    """

    x_values = landmark_pixels[:, 0]
    y_values = landmark_pixels[:, 1]

    x_min = float(np.clip(np.min(x_values), 0, width - 1))
    x_max = float(np.clip(np.max(x_values), 0, width - 1))
    y_min = float(np.clip(np.min(y_values), 0, height - 1))
    y_max = float(np.clip(np.max(y_values), 0, height - 1))

    box_width = max(0.0, x_max - x_min)
    box_height = max(0.0, y_max - y_min)
    box_area = box_width * box_height
    image_area = float(width * height)

    return {
        "x_min": int(round(x_min)),
        "x_max": int(round(x_max)),
        "y_min": int(round(y_min)),
        "y_max": int(round(y_max)),
        "width": int(round(box_width)),
        "height": int(round(box_height)),
        "area_px": float(box_area),
        "area_fraction": float(box_area / image_area) if image_area > 0 else None,
    }


def _make_box_from_relative_face_bbox(
    face_bbox: dict[str, Any],
    image_width: int,
    image_height: int,
    x0_frac: float,
    x1_frac: float,
    y0_frac: float,
    y1_frac: float,
) -> dict[str, Any]:
    """
    Create an integer ROI box using relative coordinates inside the face bbox.

    Parameters
    ----------
    face_bbox:
        Face bounding box dictionary.

    image_width:
        Full image width.

    image_height:
        Full image height.

    x0_frac, x1_frac:
        Horizontal ROI range inside face bbox, from 0 to 1.

    y0_frac, y1_frac:
        Vertical ROI range inside face bbox, from 0 to 1.

    Returns
    -------
    dict[str, Any]
        JSON-safe ROI box.
    """

    face_x_min = int(face_bbox["x_min"])
    face_y_min = int(face_bbox["y_min"])
    face_width = int(face_bbox["width"])
    face_height = int(face_bbox["height"])

    x_min = face_x_min + int(round(face_width * x0_frac))
    x_max = face_x_min + int(round(face_width * x1_frac))
    y_min = face_y_min + int(round(face_height * y0_frac))
    y_max = face_y_min + int(round(face_height * y1_frac))

    x_min = int(np.clip(x_min, 0, image_width - 1))
    x_max = int(np.clip(x_max, 0, image_width))
    y_min = int(np.clip(y_min, 0, image_height - 1))
    y_max = int(np.clip(y_max, 0, image_height))

    box_width = max(0, x_max - x_min)
    box_height = max(0, y_max - y_min)
    box_area = float(box_width * box_height)
    image_area = float(image_width * image_height)

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "width": int(box_width),
        "height": int(box_height),
        "area_px": box_area,
        "area_fraction": float(box_area / image_area) if image_area > 0 else None,
    }


def _crop_rgb_with_bbox(
    rgb_array: np.ndarray,
    bbox: dict[str, Any],
) -> np.ndarray:
    """
    Crop RGB image using an integer bounding box.

    Parameters
    ----------
    rgb_array:
        RGB image array with shape (height, width, 3).

    bbox:
        Bounding box dictionary.

    Returns
    -------
    np.ndarray
        Cropped RGB image.
    """

    x_min = int(bbox["x_min"])
    x_max = int(bbox["x_max"])
    y_min = int(bbox["y_min"])
    y_max = int(bbox["y_max"])

    return rgb_array[y_min:y_max, x_min:x_max, :]


def _summarize_named_roi(
    rgb_array: np.ndarray,
    name: str,
    box: dict[str, Any],
) -> dict[str, Any]:
    """
    Crop, summarize, and quality-check one named ROI.

    Parameters
    ----------
    rgb_array:
        RGB image array.

    name:
        ROI name.

    box:
        ROI bounding box.

    Returns
    -------
    dict[str, Any]
        JSON-safe ROI summary with one-frame quality diagnostics.
    """

    crop = _crop_rgb_with_bbox(
        rgb_array=rgb_array,
        bbox=box,
    )

    if crop.size == 0:
        crop_summary = None
        quality = {
            "status": "reject",
            "reasons": ["Empty ROI crop."],
            "metrics": {},
        }
        usable = False
        reason = "Empty ROI crop."
    else:
        crop_summary = summarize_rgb_frame(crop)
        quality = evaluate_roi_image_quality(crop)
        usable = quality["status"] != "reject"
        reason = "ROI crop extracted and one-frame quality evaluated."

    return {
        "name": name,
        "usable": usable,
        "reason": reason,
        "box": box,
        "rgb_summary": crop_summary,
        "quality": quality,
    }


def compute_debug_face_rois(
    rgb_array: np.ndarray,
    face_bbox: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute rough forehead and cheek ROIs from the face bounding box.

    Parameters
    ----------
    rgb_array:
        RGB image array with shape (height, width, 3).

    face_bbox:
        Face bounding box dictionary.

    Returns
    -------
    dict[str, Any]
        JSON-safe ROI debug information.

    Notes
    -----
    These are intentionally simple bbox-relative ROIs.

    Current ROI naming uses image perspective:
        image_left_cheek:
            cheek region on the left side of the image.

        image_right_cheek:
            cheek region on the right side of the image.

    This avoids confusion between anatomical left/right and mirrored webcam view.

    Current tuning:
        forehead:
            upper central face, avoiding the hairline as much as possible.

        cheeks:
            upper-cheek / infraorbital region.

    This is still a scaffold. Later we can replace rectangles with landmark-based
    polygons or skin masks.

    Quality diagnostics:
        Each ROI receives a one-frame quality status:
            ok / warning / reject

        This checks intensity, darkness, and saturation risk. It does not detect
        pulse yet.
    """

    image_height, image_width, _ = rgb_array.shape

    roi_boxes = {
        "forehead": _make_box_from_relative_face_bbox(
            face_bbox=face_bbox,
            image_width=image_width,
            image_height=image_height,
            x0_frac=0.30,
            x1_frac=0.70,
            y0_frac=0.06,
            y1_frac=0.22,
        ),
        "image_left_cheek": _make_box_from_relative_face_bbox(
            face_bbox=face_bbox,
            image_width=image_width,
            image_height=image_height,
            x0_frac=0.10,
            x1_frac=0.34,
            y0_frac=0.40,
            y1_frac=0.62,
        ),
        "image_right_cheek": _make_box_from_relative_face_bbox(
            face_bbox=face_bbox,
            image_width=image_width,
            image_height=image_height,
            x0_frac=0.66,
            x1_frac=0.90,
            y0_frac=0.40,
            y1_frac=0.62,
        ),
    }

    roi_summaries = [
        _summarize_named_roi(
            rgb_array=rgb_array,
            name=name,
            box=box,
        )
        for name, box in roi_boxes.items()
    ]

    usable_roi_count = sum(1 for roi in roi_summaries if roi["usable"])

    status_counts = {
        "ok": 0,
        "warning": 0,
        "reject": 0,
    }

    for roi in roi_summaries:
        quality_status = roi["quality"]["status"]
        status_counts[quality_status] = status_counts.get(quality_status, 0) + 1

    if status_counts["reject"] > 0:
        overall_status = "warning"
        overall_reason = (
            "At least one ROI is rejected by one-frame intensity quality checks."
        )
    elif status_counts["warning"] > 0:
        overall_status = "warning"
        overall_reason = (
            "At least one ROI has one-frame intensity quality warnings."
        )
    else:
        overall_status = "ok"
        overall_reason = "All ROIs passed one-frame intensity quality checks."

    return {
        "method": "bbox_relative_rectangles_v3_with_quality_v1",
        "description": (
            "Rough forehead and image-perspective cheek rectangles derived from "
            "the detected face bbox. Cheek ROIs target the upper-cheek / "
            "infraorbital region. This is a debug scaffold, not final ROI logic."
        ),
        "roi_count": len(roi_summaries),
        "usable_roi_count": int(usable_roi_count),
        "quality_summary": {
            "overall_status": overall_status,
            "overall_reason": overall_reason,
            "status_counts": status_counts,
        },
        "rois": roi_summaries,
    }

def _create_face_landmarker_options(
    model_path: Path,
):
    """
    Create MediaPipe Tasks Face Landmarker options.

    Parameters
    ----------
    model_path:
        Path to face_landmarker.task.

    Returns
    -------
    tuple[Any, Any]
        FaceLandmarker class and FaceLandmarkerOptions object.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist.

    RuntimeError
        If the installed MediaPipe package does not expose the Tasks API.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            "Face Landmarker model file not found. "
            f"Expected path: {model_path}"
        )

    if not hasattr(mp, "tasks"):
        raise RuntimeError("Installed MediaPipe package does not expose mp.tasks.")

    if not hasattr(mp, "Image") or not hasattr(mp, "ImageFormat"):
        raise RuntimeError("Installed MediaPipe package does not expose mp.Image/ImageFormat.")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    return FaceLandmarker, options

def get_cached_face_landmarker(
    model_path: Path,
):
    """
    Return a cached MediaPipe Face Landmarker instance.

    Parameters
    ----------
    model_path:
        Path to face_landmarker.task.

    Returns
    -------
    Any
        Cached FaceLandmarker instance.

    Why this exists:
        Creating the Face Landmarker on every frame is expensive. During live
        sampling, repeated model/object initialization becomes a major bottleneck.

    Limitation:
        This simple cache is intended for local single-user demo use. For a
        production multi-user server, we would need a more careful resource and
        concurrency design.
    """

    cache_key = str(model_path.resolve())

    if cache_key in _FACE_LANDMARKER_CACHE:
        return _FACE_LANDMARKER_CACHE[cache_key]

    FaceLandmarker, options = _create_face_landmarker_options(model_path=model_path)
    landmarker = FaceLandmarker.create_from_options(options)

    _FACE_LANDMARKER_CACHE[cache_key] = landmarker

    return landmarker

def detect_face_landmarks_in_rgb_frame(
    rgb_array: np.ndarray,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run MediaPipe Tasks Face Landmarker on one RGB frame.

    Parameters
    ----------
    rgb_array:
        RGB image array with shape (height, width, 3).

    model_path:
        Optional path to face_landmarker.task.

    Returns
    -------
    dict[str, Any]
        JSON-safe face detection and ROI diagnostics.

    Notes
    -----
    MediaPipe can be strict about input layout and dtype.
    We force:
        - RGB image
        - uint8 dtype
        - C-contiguous memory layout

    Timing:
        This function reports approximate backend timings so we can identify
        whether MediaPipe face landmarking or ROI summary extraction is the
        bottleneck during live sampling.

    Performance:
        The Face Landmarker instance is cached and reused between requests.
    """

    from time import perf_counter

    function_start = perf_counter()

    if model_path is None:
        model_path = DEFAULT_FACE_LANDMARKER_MODEL_PATH

    mediapipe_status = get_mediapipe_status(model_path=model_path)

    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb_array.shape}.")

    input_prepare_start = perf_counter()

    if rgb_array.dtype != np.uint8:
        rgb_array = rgb_array.astype(np.uint8)

    rgb_array = np.ascontiguousarray(rgb_array)

    input_prepare_ms = (perf_counter() - input_prepare_start) * 1000.0

    height, width, channels = rgb_array.shape

    landmarker_fetch_start = perf_counter()

    landmarker = get_cached_face_landmarker(model_path=model_path)

    landmarker_fetch_ms = (perf_counter() - landmarker_fetch_start) * 1000.0

    mp_image_start = perf_counter()

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_array,
    )

    mp_image_ms = (perf_counter() - mp_image_start) * 1000.0

    landmarker_start = perf_counter()

    result = landmarker.detect(mp_image)

    face_landmarker_ms = (perf_counter() - landmarker_start) * 1000.0

    if len(result.face_landmarks) == 0:
        total_ms = (perf_counter() - function_start) * 1000.0

        return {
            "face_detected": False,
            "message": "No face landmarks detected in submitted frame.",
            "mediapipe": mediapipe_status,
            "image": {
                "width": int(width),
                "height": int(height),
                "channels": int(channels),
                "dtype": str(rgb_array.dtype),
                "c_contiguous": bool(rgb_array.flags["C_CONTIGUOUS"]),
            },
            "roi_debug": None,
            "timing_ms": {
                "input_prepare_ms": float(input_prepare_ms),
                "landmarker_fetch_ms": float(landmarker_fetch_ms),
                "mediapipe_image_ms": float(mp_image_ms),
                "face_landmarker_ms": float(face_landmarker_ms),
                "landmark_postprocess_ms": 0.0,
                "roi_debug_ms": 0.0,
                "total_face_debug_ms": float(total_ms),
            },
        }

    postprocess_start = perf_counter()

    face_landmarks = result.face_landmarks[0]

    landmark_pixels = _normalized_landmarks_to_pixel_array(
        landmarks=face_landmarks,
        width=width,
        height=height,
    )

    bbox = _compute_landmark_bbox(
        landmark_pixels=landmark_pixels,
        width=width,
        height=height,
    )

    face_crop = _crop_rgb_with_bbox(
        rgb_array=rgb_array,
        bbox=bbox,
    )

    if face_crop.size > 0:
        face_crop_summary = summarize_rgb_frame(face_crop)
    else:
        face_crop_summary = None

    landmark_postprocess_ms = (perf_counter() - postprocess_start) * 1000.0

    roi_start = perf_counter()

    roi_debug = compute_debug_face_rois(
        rgb_array=rgb_array,
        face_bbox=bbox,
    )

    roi_debug_ms = (perf_counter() - roi_start) * 1000.0
    total_ms = (perf_counter() - function_start) * 1000.0

    return {
        "face_detected": True,
        "message": "Face landmarks detected.",
        "mediapipe": mediapipe_status,
        "image": {
            "width": int(width),
            "height": int(height),
            "channels": int(channels),
            "dtype": str(rgb_array.dtype),
            "c_contiguous": bool(rgb_array.flags["C_CONTIGUOUS"]),
        },
        "face": {
            "landmark_count": int(len(face_landmarks)),
            "bbox": bbox,
            "face_crop_summary": face_crop_summary,
        },
        "roi_debug": roi_debug,
        "timing_ms": {
            "input_prepare_ms": float(input_prepare_ms),
            "landmarker_fetch_ms": float(landmarker_fetch_ms),
            "mediapipe_image_ms": float(mp_image_ms),
            "face_landmarker_ms": float(face_landmarker_ms),
            "landmark_postprocess_ms": float(landmark_postprocess_ms),
            "roi_debug_ms": float(roi_debug_ms),
            "total_face_debug_ms": float(total_ms),
        },
    }

def summarize_face_from_data_url_frame(
    image_data_url: str,
    model_path: Path | None = None,
) -> dict[str, Any]:
    """
    Decode one browser image data URL and run face landmark / ROI debug detection.

    Parameters
    ----------
    image_data_url:
        Browser canvas image encoded as a data URL.

    model_path:
        Optional path to face_landmarker.task.

    Returns
    -------
    dict[str, Any]
        JSON-safe face detection and ROI debug response.

    Privacy:
        The frame is decoded and processed in memory only.
        This function does not save image bytes or arrays to disk.

    Timing:
        The response includes coarse processing timings. These are not benchmark-
        grade measurements, but they are good enough to see where live sampling
        time is spent.
    """

    from time import perf_counter

    total_start = perf_counter()

    if model_path is None:
        model_path = DEFAULT_FACE_LANDMARKER_MODEL_PATH

    decode_start = perf_counter()

    rgb_array, decode_metadata = decode_image_data_url_to_rgb_array(image_data_url)

    decode_ms = (perf_counter() - decode_start) * 1000.0

    frame_summary_start = perf_counter()

    frame_summary = summarize_rgb_frame(rgb_array)

    frame_summary_ms = (perf_counter() - frame_summary_start) * 1000.0

    face_debug_start = perf_counter()

    face_summary = detect_face_landmarks_in_rgb_frame(
        rgb_array=rgb_array,
        model_path=model_path,
    )

    face_debug_total_ms = (perf_counter() - face_debug_start) * 1000.0

    total_ms = (perf_counter() - total_start) * 1000.0

    return {
        "status": "processed",
        "message": "Frame decoded, face detection, and ROI debug completed. Frame was not stored.",
        "frame": frame_summary,
        "decode": decode_metadata,
        "face_debug": face_summary,
        "timing_ms": {
            "decode_ms": float(decode_ms),
            "frame_summary_ms": float(frame_summary_ms),
            "face_debug_total_ms": float(face_debug_total_ms),
            "total_request_processing_ms": float(total_ms),
        },
    }