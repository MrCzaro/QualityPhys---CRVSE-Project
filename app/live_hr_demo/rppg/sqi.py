"""
Signal quality and spectrum utilities for rPPG signals.

This module estimates whether a pulse-like rPPG trace has a clear dominant
frequency in the cardiac band.

Physiology:
    Heart rate appears as a repeated rhythm. In frequency space, a stable pulse
    should create a visible peak around the heart-rate frequency.

Signal:
    Frequency in Hz can be converted to BPM:
        bpm = hz * 60

Limitation:
    Motion, lighting flicker, and face movement can also create spectral peaks.
    SQI is a useful warning signal, not a guarantee of correctness.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class SpectrumResult:
    """
    Frequency-domain summary for one signal.

    Attributes
    ----------
    freqs_hz:
        Frequency axis in Hz.

    power:
        Power spectrum values.

    cardiac_mask:
        Boolean mask for the cardiac frequency band.

    dominant_freq_hz:
        Dominant frequency inside cardiac band.

    dominant_bpm:
        Dominant frequency converted to BPM.

    sqi:
        Spectral quality index.

    status:
        Simple quality label: good, moderate, poor, invalid.

    reason:
        Human-readable explanation.
    """
    freqs_hz: np.ndarray
    power: np.ndarray
    cardiac_mask: np.ndarray
    dominant_freq_hz: float
    dominant_bpm: float
    sqi: float
    status: str
    reason: str


def compute_power_spectrum(signal: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided power spectrum using real FFT.

    Parameters
    ----------
    signal:
        1D signal.

    fps:
        Sampling frequency in Hz.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        freqs_hz, power
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}.")
    if len(signal) < 4:
        raise ValueError("Need at least 4 samples to compute spectrum.")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")

    signal = signal - np.mean(signal)
    window = np.hanning(len(signal)).astype(np.float32)
    windowed = signal * window
    fft_values = np.fft.rfft(windowed)
    power = np.abs(fft_values) ** 2
    freqs_hz = np.fft.rfftfreq(len(signal), d=1.0 / fps)

    return freqs_hz.astype(np.float32), power.astype(np.float32)

def _refine_peak_bin_by_parabolic_interpolation(
    power: np.ndarray,
    peak_idx: int,
    lower_idx: int,
    upper_idx: int,
    eps: float = 1e-20,
) -> float:
    """
    Refine a spectral peak location to sub-bin resolution.

    Parameters
    ----------
    power:
        Power spectrum values.

    peak_idx:
        Index of the maximum-power bin.

    lower_idx:
        Lowest bin index allowed by the analysis band.

    upper_idx:
        Highest bin index allowed by the analysis band.

    eps:
        Small constant preventing log(0).

    Returns
    -------
    float
        Fractional bin position. Returns peak_idx unchanged when refinement is
        not possible.

    Notes
    -----
    Fits a parabola through the log-power values at peak_idx - 1, peak_idx, and
    peak_idx + 1. For a Hann-windowed spectrum this recovers the true peak
    frequency far more accurately than taking the bin centre.

    Refinement is skipped when the peak sits on a band edge, when the curvature
    is degenerate, or when the computed offset falls outside the +/- 0.5 bin
    range that a genuine local maximum must satisfy.
    """

    if peak_idx <= lower_idx or peak_idx >= upper_idx:
        return float(peak_idx)

    left = float(np.log(float(power[peak_idx - 1]) + eps))
    centre = float(np.log(float(power[peak_idx]) + eps))
    right = float(np.log(float(power[peak_idx + 1]) + eps))

    denominator = left - 2.0 * centre + right

    if abs(denominator) < 1e-12:
        return float(peak_idx)

    offset = 0.5 * (left - right) / denominator

    if not np.isfinite(offset) or abs(offset) > 0.5:
        return float(peak_idx)

    return float(peak_idx) + offset


def estimate_spectral_sqi(
    signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    peak_neighborhood_bins: int = 1,
) -> SpectrumResult:
    """
    Estimate spectral SQI from a single rPPG trace.

    SQI definition:
        power near dominant cardiac-band peak / total cardiac-band power

    Parameters
    ----------
    signal:
        1D rPPG signal.

    fps:
        Sampling frequency in Hz.

    low_hz:
        Lower cardiac-band frequency.

    high_hz:
        Upper cardiac-band frequency.

    peak_neighborhood_bins:
        Number of neighboring FFT bins included around dominant peak.

    Returns
    -------
    SpectrumResult
        Spectrum and quality summary.

    Notes
    -----
    The dominant frequency is refined to sub-bin resolution by parabolic
    interpolation. SQI itself is computed from whole bins and is unchanged by
    that refinement, so existing SQI thresholds remain calibrated.
    """
    freqs_hz, power = compute_power_spectrum(signal=signal, fps=fps)
    cardiac_mask = (freqs_hz >= low_hz) & (freqs_hz <= high_hz)

    if not np.any(cardiac_mask):
        return SpectrumResult(
            freqs_hz=freqs_hz,
            power=power,
            cardiac_mask=cardiac_mask,
            dominant_freq_hz=float("nan"),
            dominant_bpm=float("nan"),
            sqi=0.0,
            status="invalid",
            reason="No FFT bins found inside cardiac band.",
        )

    cardiac_indices = np.where(cardiac_mask)[0]
    cardiac_power = power[cardiac_indices]
    total_cardiac_power = float(np.sum(cardiac_power))

    if total_cardiac_power <= 1e-12:
        return SpectrumResult(
            freqs_hz=freqs_hz,
            power=power,
            cardiac_mask=cardiac_mask,
            dominant_freq_hz=float("nan"),
            dominant_bpm=float("nan"),
            sqi=0.0,
            status="invalid",
            reason="Cardiac-band power is too low or flat.",
        )

    local_peak_idx = int(np.argmax(cardiac_power))
    peak_idx = int(cardiac_indices[local_peak_idx])
    left_idx = max(cardiac_indices[0], peak_idx - peak_neighborhood_bins)
    right_idx = min(cardiac_indices[-1], peak_idx + peak_neighborhood_bins)
    peak_power = float(np.sum(power[left_idx : right_idx + 1]))
    sqi = peak_power / total_cardiac_power

    refined_peak_bin = _refine_peak_bin_by_parabolic_interpolation(
        power=power,
        peak_idx=peak_idx,
        lower_idx=int(cardiac_indices[0]),
        upper_idx=int(cardiac_indices[-1]),
    )

    bin_width_hz = float(freqs_hz[1] - freqs_hz[0]) if len(freqs_hz) > 1 else 0.0
    dominant_freq_hz = float(refined_peak_bin * bin_width_hz)
    dominant_bpm = dominant_freq_hz * 60.0

    if sqi >= 0.50:
        status = "good"
        reason = "Clear dominant spectral peak in cardiac band."
    elif sqi >= 0.30:
        status = "moderate"
        reason = "Dominant spectral peak exists, but signal is not very clean."
    else:
        status = "poor"
        reason = "No strong dominant cardiac-band peak."

    return SpectrumResult(
        freqs_hz=freqs_hz,
        power=power,
        cardiac_mask=cardiac_mask,
        dominant_freq_hz=dominant_freq_hz,
        dominant_bpm=dominant_bpm,
        sqi=float(sqi),
        status=status,
        reason=reason,
    )


def summarize_multichannel_sqi(
    signals: dict[str, np.ndarray],
    fps: float,
    channel_names: tuple[str, ...] = ("pos", "chrom", "green"),
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> dict[str, SpectrumResult]:
    """
    Compute spectral SQI for multiple rPPG channels.

    Parameters
    ----------
    signals:
        Dictionary containing rPPG channels.

    fps:
        Sampling frequency in Hz.

    channel_names:
        Channels to summarize.

    low_hz:
        Lower cardiac-band frequency.

    high_hz:
        Upper cardiac-band frequency.

    Returns
    -------
    dict[str, SpectrumResult]
        One SpectrumResult per channel.
    """

    results: dict[str, SpectrumResult] = {}

    for channel_name in channel_names:
        if channel_name not in signals:
            raise ValueError(f"Missing signal channel: {channel_name}")

        results[channel_name] = estimate_spectral_sqi(
            signal=signals[channel_name],
            fps=fps,
            low_hz=low_hz,
            high_hz=high_hz,
        )

    return results