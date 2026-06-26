"""
Live ROI model prediction helpers.

This module converts browser-collected ROI RGB samples into model-ready
POS/CHROM/GREEN channels and runs the experimental CRVSE HR prediction path.
"""

from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import torch
import numpy as np
from inference.window_inference import predict_hr_from_rppg_window
from rppg.live_methods import build_model_input_from_roi_series_payload


def make_live_roi_model_prediction_payload(payload: dict, model_bundle) -> dict:
    """
    Run experimental live model prediction from browser ROI samples.

    Parameters
    ----------
    payload:
        Browser-collected numeric ROI RGB sample payload.

    model_bundle:
        Loaded model bundle used for prediction.

    Returns
    -------
    dict
        JSON-safe-ish prediction payload before final serialization cleanup.

    Notes
    -----
    The function receives numeric ROI summaries only. It does not receive,
    store, or process raw image frames.
    """
    input_spec = model_bundle.model_spec["input"]
    target_frames = int(input_spec["target_frames"])
    window_seconds = float(input_spec["window_seconds"])
    model_input_result = build_model_input_from_roi_series_payload(
        payload=payload,
        target_length=target_frames,
        window_seconds=window_seconds,
    )

    if model_input_result.get("status") != "ok":
        return model_input_result
    model_input_np = model_input_result["_model_input"]

    # build_model_input_from_roi_series_payload returns:
    #   shape: (1, 3, target_frames)
    #
    # Channel order:
    #   0 = POS
    #   1 = CHROM
    #   2 = GREEN
    signals = {
        "pos": model_input_np[0, 0, :],
        "chrom": model_input_np[0, 1, :],
        "green": model_input_np[0, 2, :],
    }
    # The model consumes target_frames samples over window_seconds.
    fps = float(target_frames) / float(window_seconds)
    prediction_payload = predict_hr_from_rppg_window(
        signals=signals,
        fps=fps,
        bundle=model_bundle,
    )

    classical_analysis = model_input_result["classical_analysis"]
    return {
        "status": "ok",
        "message": "Experimental live ROI model prediction completed.",
        "model_prediction": prediction_payload,
        "model_input": {
            "input_shape": model_input_result["input_shape"],
            "channel_order": model_input_result["channel_order"],
            "target_length": model_input_result["target_length"],
            "source_sample_count": model_input_result["sample_count"],
            "source_duration_s": model_input_result["duration_s"],
            "source_estimated_fps": model_input_result["estimated_fps"],
            "model_target_frames": int(target_frames),
            "model_window_seconds": float(window_seconds),
            "model_assumed_fps_after_resampling": float(fps),
            "window_metadata": model_input_result["window_metadata"],
        },
        "classical_spectral_summary": {
            "green": classical_analysis["signals"]["green"]["spectral"],
            "pos": classical_analysis["signals"]["pos"]["spectral"],
            "chrom": classical_analysis["signals"]["chrom"]["spectral"],
        },
        "notes": [
            "This is experimental live model inference from browser-collected ROI RGB samples.",
            "Input was cropped to the latest model-duration window before resampling.",
            "Input was resampled to the model contract of 3 channels x target_frames samples.",
            "The model sees the resampled window as target_frames over model_window_seconds.",
            "This is not a medical measurement.",
        ],
    }


def make_json_safe_for_api(value: Any) -> Any:
    """
    Convert common Python, NumPy, Torch, and dataclass objects to JSON-safe values.

    Parameters
    ----------
    value:
        Object to convert before returning through a JSON API response.

    Returns
    -------
    Any
        JSON-serializable value.
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if is_dataclass(value):
        return make_json_safe_for_api(asdict(value))
    if hasattr(value, "model_dump"):
        return make_json_safe_for_api(value.model_dump())
    if isinstance(value, dict):
        return {str(key): make_json_safe_for_api(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe_for_api(item) for item in value]
    if hasattr(value, "__dict__"):
        return make_json_safe_for_api(vars(value))

    return str(value)