"""
Window-level quality evaluation for rPPG inference.

This module turns spectral SQI outputs into an accepted/rejected window decision.


Signal:
    POS / CHROM / GREEN should show a dominant frequency peak in the cardiac band.
    The different channels do not have to be perfect, but at least one or more
    should carry usable pulse information.

Limitation:
    This is not a medical validation layer. It is a signal-quality guardrail.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from rppg.sqi import SpectrumResult, summarize_multichannel_sqi


@dataclass
class WindowQualityResult:
    """
    Quality decision for one rPPG window.

    Attributes
    ----------
    accepted:
        Whether the window should be sent to the model / trusted for display.

    confidence:
        Simple confidence label: good, moderate, poor, rejected.

    reasons:
        Human-readable reasons for the decision.

    metrics:
        Numeric metrics useful for debugging and UI display.

    sqi_by_channel:
        Spectral SQI result for each channel.
    """
    accepted: bool
    confidence: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    sqi_by_channel: dict[str, SpectrumResult] = field(default_factory=dict)


def evaluate_window_quality(
    signals: dict[str, np.ndarray],
    fps: float,
    channel_names: tuple[str, ...] = ("pos", "chrom", "green"),
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    min_good_channels: int = 1,
    min_moderate_channels: int = 2,
    good_sqi_threshold: float = 0.50,
    moderate_sqi_threshold: float = 0.30,
    max_channel_bpm_spread: float = 20.0,
) -> WindowQualityResult:
    """
    Evaluate whether a POS/CHROM/GREEN rPPG window has usable multichannel support.
    """
    sqi_by_channel = summarize_multichannel_sqi(
        signals=signals,
        fps=fps,
        channel_names=channel_names,
        low_hz=low_hz,
        high_hz=high_hz,
    )

    reasons: list[str] = []
    metrics: dict[str, Any] = {}
    good_channels: list[str] = []
    moderate_or_good_channels: list[str] = []
    all_valid_bpms: list[float] = []
    supported_bpms: list[float] = []

    for channel_name, sqi_result in sqi_by_channel.items():
        sqi_value = float(sqi_result.sqi) if sqi_result.sqi is not None else 0.0
        bpm_value = (
            float(sqi_result.dominant_bpm)
            if sqi_result.dominant_bpm is not None
            else float("nan")
        )

        metrics[f"{channel_name}_sqi"] = sqi_value
        metrics[f"{channel_name}_dominant_bpm"] = bpm_value
        metrics[f"{channel_name}_status"] = sqi_result.status

        if np.isfinite(bpm_value):
            all_valid_bpms.append(bpm_value)

        if sqi_value >= good_sqi_threshold:
            good_channels.append(channel_name)

        if sqi_value >= moderate_sqi_threshold:
            moderate_or_good_channels.append(channel_name)
            if np.isfinite(bpm_value):
                supported_bpms.append(bpm_value)

    if len(supported_bpms) > 0:
        bpm_spread = float(np.max(supported_bpms) - np.min(supported_bpms))
    else:
        bpm_spread = float("nan")

    if len(all_valid_bpms) > 0:
        all_channel_bpm_spread = float(np.max(all_valid_bpms) - np.min(all_valid_bpms))
    else:
        all_channel_bpm_spread = float("nan")

    metrics["n_good_channels"] = len(good_channels)
    metrics["n_moderate_or_good_channels"] = len(moderate_or_good_channels)
    metrics["bpm_spread_across_channels"] = bpm_spread
    metrics["bpm_spread_across_all_detected_channels"] = all_channel_bpm_spread

    has_required_channel_support = len(moderate_or_good_channels) >= min_moderate_channels
    has_good_confidence = len(good_channels) >= min_good_channels
    has_consistent_channel_bpm = (
        np.isfinite(bpm_spread)
        and bpm_spread <= max_channel_bpm_spread
    )

    accepted = has_required_channel_support and has_consistent_channel_bpm

    if accepted:
        confidence = "good" if has_good_confidence else "moderate"

        reasons.append(
            f"Accepted: at least {min_moderate_channels} channel(s) have "
            f"moderate-or-better SQI ({moderate_or_good_channels})."
        )

        if has_good_confidence:
            reasons.append(
                f"Confidence good: at least {min_good_channels} channel(s) have "
                f"good SQI ({good_channels})."
            )
        else:
            reasons.append(
                "Confidence moderate: multichannel support is present, but no "
                "channel reached good SQI."
            )

        reasons.append(
            f"Supported-channel dominant BPM spread is acceptable "
            f"({bpm_spread:.1f} BPM <= {max_channel_bpm_spread:.1f} BPM)."
        )

    else:
        confidence = "rejected"

        if not has_required_channel_support:
            reasons.append(
                f"Rejected: fewer than {min_moderate_channels} channel(s) have "
                "moderate-or-better spectral support."
            )

        if has_required_channel_support and not has_consistent_channel_bpm:
            if np.isfinite(bpm_spread):
                reasons.append(
                    f"Rejected: supported-channel dominant BPM spread is too large "
                    f"({bpm_spread:.1f} BPM > {max_channel_bpm_spread:.1f} BPM)."
                )
            else:
                reasons.append(
                    "Rejected: supported-channel dominant BPM spread could not be computed."
                )

        if not has_required_channel_support:
            reasons.append(
                "A single good channel is not enough for model-window acceptance."
            )

    for channel_name, sqi_result in sqi_by_channel.items():
        reasons.append(
            f"{channel_name.upper()}: SQI={sqi_result.sqi:.3f}, "
            f"dominant={sqi_result.dominant_bpm:.1f} BPM, "
            f"status={sqi_result.status}."
        )

    return WindowQualityResult(
        accepted=accepted,
        confidence=confidence,
        reasons=reasons,
        metrics=metrics,
        sqi_by_channel=sqi_by_channel,
    )