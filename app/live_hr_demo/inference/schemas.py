"""
Prediction schemas for the live HR demo.

These dataclasses define clean objects that the rest of the app can use.

Why this exists:
    PyTorch returns tensors.
    Signal quality code returns internal diagnostic objects.
    The UI should receive readable prediction/result objects instead.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QualitySummary:
    """
    Quality summary for one inference window.

    Attributes
    ----------
    status:
        Window status, for example: accepted, rejected, not_available_yet.

    confidence:
        Confidence label, for example: good, moderate, rejected.

    reasons:
        Human-readable explanation strings.

    metrics:
        Numeric/debug metrics for UI display.
    """
    status: str = "not_available_yet"
    confidence: str = "not_available_yet"
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """
    Clean prediction result returned by the inference layer.

    Attributes
    ----------
    task:
        Name of the predicted vital/signature, currently "heart_rate".

    value:
        Numeric model prediction value. None if prediction was skipped.

    unit:
        Prediction unit, currently "bpm".

    model_name:
        Name from model_specs.yaml.

    window_seconds:
        Duration of signal window used by the model.

    target_frames:
        Number of samples expected by the model after resampling.

    channel_names:
        Input channel names, currently ["pos", "chrom", "green"].

    quality:
        Quality summary for this prediction/window.

    extra:
        Optional additional values, such as spectral HR estimate.
    """
    task: str
    value: float | None
    unit: str
    model_name: str
    window_seconds: float
    target_frames: int
    channel_names: list[str]
    quality: QualitySummary
    extra: dict[str, Any] = field(default_factory=dict)