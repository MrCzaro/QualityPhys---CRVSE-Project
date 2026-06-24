"""
Integrated window-level inference for the live HR demo.

This module connects:
    POS / CHROM / GREEN traces
    → signal quality evaluation
    → model window preparation
    → model HR prediction
    → clean PredictionResult


Signal:
    We first check whether the channels contain a plausible cardiac-band rhythm.
    If the window is acceptable, we prepare the tensor and run the model.

Limitation:
    This still assumes POS / CHROM / GREEN traces already exist.
    Video-to-rPPG extraction comes later.
"""
from __future__ import annotations
from typing import Any
import numpy as np

from inference.predictor import predict_hr_from_tensor
from inference.schemas import PredictionResult, QualitySummary
from models.loader import ModelBundle
from rppg.quality import WindowQualityResult, evaluate_window_quality
from rppg.windowing import WindowConfig, make_model_window_from_channels


def estimate_spectral_hr_from_quality(quality: WindowQualityResult) -> float | None:
    """
    Estimate one spectral HR value from accepted channel dominant BPMs.

    Current simple rule:
        median dominant BPM across finite channel estimates.

    Parameters
    ----------
    quality:
        Window quality result.

    Returns
    -------
    float | None
        Median spectral HR in BPM, or None if unavailable.
    """
    bpms: list[float] = []
    for sqi_result in quality.sqi_by_channel.values():
        bpm = float(sqi_result.dominant_bpm)
        if np.isfinite(bpm):
            bpms.append(bpm)

    if len(bpms) == 0:
        return None

    return float(np.median(bpms))


def make_quality_summary(quality: WindowQualityResult) -> QualitySummary:
    """
    Convert internal WindowQualityResult into UI/app-friendly QualitySummary.
    """
    return QualitySummary(
        status="accepted" if quality.accepted else "rejected",
        confidence=quality.confidence,
        reasons=list(quality.reasons),
        metrics=dict(quality.metrics),
    )


def predict_hr_from_rppg_window(signals: dict[str, np.ndarray], fps: float, bundle: ModelBundle) -> PredictionResult:
    """
    Evaluate quality and optionally predict HR from POS/CHROM/GREEN signals.

    Parameters
    ----------
    signals:
        Dictionary containing:
            pos
            chrom
            green

    fps:
        Sampling frequency of the signals in Hz.

    bundle:
        Loaded model bundle.

    Returns
    -------
    PredictionResult
        Complete window-level result.
    """
    model_spec = bundle.model_spec
    input_config = model_spec["input"]
    output_config = model_spec["output"]
    channel_names = tuple(input_config["channel_names"])

    quality = evaluate_window_quality(
        signals=signals,
        fps=fps,
        channel_names=channel_names,
        low_hz=float(model_spec["preprocessing"]["bandpass_low_hz"]),
        high_hz=float(model_spec["preprocessing"]["bandpass_high_hz"]),
    )

    spectral_hr_bpm = estimate_spectral_hr_from_quality(quality) if quality.accepted else None
    quality_summary = make_quality_summary(quality)
    base_result_kwargs: dict[str, Any] = {
        "task": model_spec["task"],
        "unit": output_config["unit"],
        "model_name": model_spec["name"],
        "window_seconds": float(input_config["window_seconds"]),
        "target_frames": int(input_config["target_frames"]),
        "channel_names": list(channel_names),
        "quality": quality_summary,
        "extra": {
            "spectral_hr_bpm": spectral_hr_bpm,
        },
    }

    if not quality.accepted:
        return PredictionResult(
            value=None,
            **base_result_kwargs,
        )

    window_config = WindowConfig(
        window_seconds=float(input_config["window_seconds"]),
        target_frames=int(input_config["target_frames"]),
        channel_names=channel_names,
        normalization=str(input_config["normalization"]),
    )

    x = make_model_window_from_channels(
        pos=signals["pos"],
        chrom=signals["chrom"],
        green=signals["green"],
        config=window_config,
    )

    prediction = predict_hr_from_tensor(x, bundle)[0]

    return PredictionResult(
        value=prediction.value,
        **base_result_kwargs,
    )