"""
Serialization helpers for live HR demo inference results.

Why this exists:
    Internal app code can use dataclasses.
    Browser/frontend responses need JSON-safe dictionaries.

Physiology:
    Not directly handled here.

Signal:
    Not directly handled here.

Limitation:
    This only formats already-computed results. It does not validate signal quality
    or run inference.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from inference.schemas import PredictionResult


def make_json_safe_value(value: Any) -> Any:
    """
    Convert common NumPy/Python values into JSON-safe values.

    Parameters
    ----------
    value:
        Any value from prediction metrics.

    Returns
    -------
    Any
        JSON-safe value.
    """

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): make_json_safe_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe_value(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe_value(item) for item in value]
    return value


def prediction_result_to_dict(result: PredictionResult) -> dict[str, Any]:
    """
    Convert PredictionResult into a JSON-safe dictionary.

    Parameters
    ----------
    result:
        PredictionResult from inference layer.

    Returns
    -------
    dict[str, Any]
        JSON-safe dictionary for UI/API responses.
    """
    return {
        "task": result.task,
        "model_hr_bpm": make_json_safe_value(result.value),
        "unit": result.unit,
        "model_name": result.model_name,
        "window_seconds": result.window_seconds,
        "target_frames": result.target_frames,
        "channel_names": list(result.channel_names),
        "quality": {
            "status": result.quality.status,
            "confidence": result.quality.confidence,
            "reasons": list(result.quality.reasons),
            "metrics": make_json_safe_value(result.quality.metrics),
        },
        "extra": make_json_safe_value(result.extra),
    }