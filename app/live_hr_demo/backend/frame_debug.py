"""
Debug helpers for receiving one browser-captured frame.

Current purpose:
    Decode one browser canvas image sent as a data URL.
    Return basic image statistics.

What this proves:
    The browser can capture a camera frame.
    The frame can be sent to the Python backend.
    The backend can decode the frame into RGB pixels.

What this does NOT do:
    Does not store frames.
    Does not run face detection.
    Does not extract rPPG.
    Does not run model inference.
"""
from __future__ import annotations
import base64
from io import BytesIO
from typing import Any
import numpy as np
from PIL import Image


def split_data_url(image_data_url: str) -> tuple[str, str]:
    """
    Split a browser data URL into metadata and base64 payload.

    Parameters
    ----------
    image_data_url:
        Browser-generated data URL, for example:
        "data:image/jpeg;base64,/9j/..."

    Returns
    -------
    tuple[str, str]
        Metadata prefix and base64 payload.

    Raises
    ------
    ValueError
        If the input is not a valid base64 data URL.
    """

    if not isinstance(image_data_url, str):
        raise ValueError("image_data_url must be a string.")
    if "," not in image_data_url:
        raise ValueError("Invalid data URL: missing comma separator.")

    metadata, payload = image_data_url.split(",", 1)

    if not metadata.startswith("data:image/"):
        raise ValueError("Invalid data URL: expected image data URL.")
    if ";base64" not in metadata:
        raise ValueError("Invalid data URL: expected base64 image payload.")
    if len(payload) == 0:
        raise ValueError("Invalid data URL: empty base64 payload.")

    return metadata, payload


def decode_image_data_url_to_rgb_array(image_data_url: str) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Decode browser image data URL into an RGB NumPy array.

    Parameters
    ----------
    image_data_url:
        Browser canvas image encoded as a data URL.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        RGB image array with shape (height, width, 3) and metadata.

    Raises
    ------
    ValueError
        If decoding fails or image format is invalid.
    """
    metadata, payload = split_data_url(image_data_url)

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except Exception as exc:
        raise ValueError(f"Could not decode base64 image payload: {exc}") from exc
    try:
        image = Image.open(BytesIO(image_bytes))
        original_mode = image.mode
        image_rgb = image.convert("RGB")
    except Exception as exc:
        raise ValueError(f"Could not open image payload with Pillow: {exc}") from exc

    rgb_array = np.asarray(image_rgb)
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb_array.shape}.")

    decode_metadata = {
        "data_url_metadata": metadata,
        "encoded_size_bytes": len(image_bytes),
        "original_pillow_mode": original_mode,
    }

    return rgb_array, decode_metadata


def summarize_rgb_frame(rgb_array: np.ndarray) -> dict[str, Any]:
    """
    Summarize an RGB image array.

    Parameters
    ----------
    rgb_array:
        RGB image array with shape (height, width, 3).

    Returns
    -------
    dict[str, Any]
        JSON-safe image summary.
    """
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {rgb_array.shape}.")

    height, width, channels = rgb_array.shape
    mean_rgb = rgb_array.mean(axis=(0, 1))
    std_rgb = rgb_array.std(axis=(0, 1))
    min_rgb = rgb_array.min(axis=(0, 1))
    max_rgb = rgb_array.max(axis=(0, 1))

    return {
        "width": int(width),
        "height": int(height),
        "channels": int(channels),
        "dtype": str(rgb_array.dtype),
        "mean_rgb": {
            "r": float(mean_rgb[0]),
            "g": float(mean_rgb[1]),
            "b": float(mean_rgb[2]),
        },
        "std_rgb": {
            "r": float(std_rgb[0]),
            "g": float(std_rgb[1]),
            "b": float(std_rgb[2]),
        },
        "min_rgb": {
            "r": int(min_rgb[0]),
            "g": int(min_rgb[1]),
            "b": int(min_rgb[2]),
        },
        "max_rgb": {
            "r": int(max_rgb[0]),
            "g": int(max_rgb[1]),
            "b": int(max_rgb[2]),
        },
    }


def summarize_data_url_frame(image_data_url: str) -> dict[str, Any]:
    """
    Decode and summarize one browser-captured image data URL.

    Parameters
    ----------
    image_data_url:
        Browser canvas image encoded as a data URL.

    Returns
    -------
    dict[str, Any]
        JSON-safe debug response.

    Privacy:
        The frame is decoded in memory only.
        This function does not save image bytes or arrays to disk.
    """
    rgb_array, decode_metadata = decode_image_data_url_to_rgb_array(image_data_url)
    frame_summary = summarize_rgb_frame(rgb_array)

    return {
        "status": "received",
        "message": "Frame decoded in backend memory. Frame was not stored.",
        "frame": frame_summary,
        "decode": decode_metadata,
    }