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
    Evaluate whether a POS/CHROM/GREEN rPPG window is usable.

    Parameters
    ----------
    signals:
        Dictionary with POS / CHROM / GREEN signals.

    fps:
        Sampling frequency in Hz.

    channel_names:
        Channels to evaluate.

    low_hz:
        Lower cardiac-band frequency.

    high_hz:
        Upper cardiac-band frequency.

    min_good_channels:
        Candidate-accept window if at least this many channels are good.

    min_moderate_channels:
        Candidate-accept window if at least this many channels are moderate or better.

    good_sqi_threshold:
        SQI threshold for good channel.

    moderate_sqi_threshold:
        SQI threshold for moderate channel.

    max_channel_bpm_spread:
        Maximum allowed spread between valid channel dominant BPMs.

    Returns
    -------
    WindowQualityResult
        Accepted/rejected decision with reasons and metrics.
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
    valid_bpms: list[float] = []

    for channel_name, sqi_result in sqi_by_channel.items():
        metrics[f"{channel_name}_sqi"] = float(sqi_result.sqi)
        metrics[f"{channel_name}_dominant_bpm"] = float(sqi_result.dominant_bpm)
        metrics[f"{channel_name}_status"] = sqi_result.status

        if np.isfinite(sqi_result.dominant_bpm):
            valid_bpms.append(float(sqi_result.dominant_bpm))
        if sqi_result.sqi >= good_sqi_threshold:
            good_channels.append(channel_name)
        if sqi_result.sqi >= moderate_sqi_threshold:
            moderate_or_good_channels.append(channel_name)

    if len(valid_bpms) > 0:
        bpm_spread = float(np.max(valid_bpms) - np.min(valid_bpms))
    else:
        bpm_spread = float("nan")

    metrics["n_good_channels"] = len(good_channels)
    metrics["n_moderate_or_good_channels"] = len(moderate_or_good_channels)
    metrics["bpm_spread_across_channels"] = bpm_spread
    has_good_channel_support = len(good_channels) >= min_good_channels
    has_moderate_channel_support = len(moderate_or_good_channels) >= min_moderate_channels
    has_enough_spectral_support = (has_good_channel_support or has_moderate_channel_support)
    has_consistent_channel_bpm = (np.isfinite(bpm_spread) and bpm_spread <= max_channel_bpm_spread)
    accepted = has_enough_spectral_support and has_consistent_channel_bpm

    if accepted:
        if has_good_channel_support:
            confidence = "good"
            reasons.append(
                f"Accepted: at least {min_good_channels} channel(s) have good SQI "
                f"({good_channels})."
            )
        else:
            confidence = "moderate"
            reasons.append(
                f"Accepted: at least {min_moderate_channels} channel(s) have "
                f"moderate-or-better SQI ({moderate_or_good_channels})."
            )

        reasons.append(
            f"Channel dominant BPM spread is acceptable "
            f"({bpm_spread:.1f} BPM <= {max_channel_bpm_spread:.1f} BPM)."
        )

    else:
        confidence = "rejected"

        if not has_enough_spectral_support:
            reasons.append(
                "Rejected: not enough channels have a clear cardiac-band peak."
            )

        if not has_consistent_channel_bpm:
            if np.isfinite(bpm_spread):
                reasons.append(
                    f"Rejected: channel dominant BPM spread is too large "
                    f"({bpm_spread:.1f} BPM > {max_channel_bpm_spread:.1f} BPM)."
                )
            else:
                reasons.append(
                    "Rejected: channel dominant BPM spread could not be computed."
                )

        if has_enough_spectral_support and not has_consistent_channel_bpm:
            reasons.append(
                "Some channels had moderate/good SQI, but their dominant HR peaks "
                "did not agree with each other."
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