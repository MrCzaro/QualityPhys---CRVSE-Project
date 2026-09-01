"""Common interface for per-vital estimator plugins."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


from ..config import (MIN_CONFIDENCE, MIN_USABLE_WINDOWS, MIN_USABLE_FRACTION,
                      MIN_REPORTABLE_FRACTION, MAX_HR_SPREAD_BPM,
                      MAD_OUTLIER_K, MAD_FLOOR_BPM)

@dataclass(frozen=True)
class EstimatorResult:
    """One vital-sign reading with its confidence and status.

    `status` is never silently discarded: the UI reports it alongside the value
    so an unreliable capture is visible rather than hidden.
    """
    vital: str
    value: float
    unit: str
    confidence: float
    status: str
    waveform: Optional[np.ndarray] = None
    detail: dict = field(default_factory=dict)


class Estimator:
    """Base class for a frames-in, value-out vital-sign estimator."""

    name = "base"
    vital = "unknown"
    unit = ""

    def is_available(self) -> bool:
        """Reports whether the estimator's model assets are present."""
        raise NotImplementedError

    def estimate(self, frames_u8, fps) -> EstimatorResult:
        """Estimates one vital from a [T,72,72,3] uint8 face-crop sequence."""
        raise NotImplementedError




def aggregate_windows(vital, unit, rates, confidences, waves, fps,
                      n_attempted, extra=None):
    """Reduces per-window readings to one gated value, identically for every model.

    Per-window rates are combined by median rather than by stitching the
    waveforms, because separate windows carry no shared phase or scale and
    overlap-adding them can cancel a genuine pulse.

    This lives on the base rather than in each estimator so that two models
    compared on the same capture are gated on the same terms. Were they gated
    separately, a disagreement between them could be an artefact of different
    thresholds rather than a real difference between the signals, which would
    destroy the only thing a cross-check is for.

    `n_attempted` counts every window the caller tried, including those that
    produced no readable peak and so are absent from `rates`. Those windows still
    weigh on the usable fraction: removing them from the denominator instead
    would make an unreadable capture look progressively cleaner the less of it
    could be read.
    """
    report = dict(n_total=int(n_attempted),
                  n_no_peak=int(n_attempted) - len(rates),
                  fps=float(fps), **(extra or {}))

    if not rates:
        return EstimatorResult(vital, float("nan"), unit, 0.0, "no_estimate",
                               detail=dict(n_windows=0, usable_fraction=0.0,
                                           window_hr=[], window_confidence=[],
                                           window_kept=[], **report))

    rates = np.asarray(rates, dtype=float)
    confidences = np.asarray(confidences, dtype=float)

    # Two independent filters. Confidence removes motion-corrupted windows; MAD
    # removes gross outliers such as octave errors, which can carry a respectable
    # confidence and which no confidence threshold reliably catches.
    median_hr = float(np.median(rates))
    mad = max(float(np.median(np.abs(rates - median_hr))) * 1.4826, MAD_FLOOR_BPM)
    keep = ((confidences >= MIN_CONFIDENCE)
            & (np.abs(rates - median_hr) <= MAD_OUTLIER_K * mad))
    usable_fraction = float(keep.sum() / n_attempted)

    # Per-window values are always reported so a caller can inspect or plot the
    # windows without re-running the model, and without reaching past this
    # interface into a particular model's internals.
    report.update(window_hr=[float(x) for x in rates],
                  window_confidence=[float(x) for x in confidences],
                  window_kept=[bool(x) for x in keep],
                  usable_fraction=usable_fraction)

    # Every window is returned in the waveform, kept or not. A refused capture is
    # exactly when someone needs to see what the model actually produced, and the
    # per-window flags let a caller mark the rejected stretches.
    full_waveform = np.concatenate(waves) if waves else None

    # A capture that loses most of its windows has not measured anything.
    # Reporting a qualified value from one or two survivors is worse than
    # refusing: the median of a handful of noisy windows is itself noise.
    if keep.sum() < MIN_USABLE_WINDOWS or usable_fraction < MIN_REPORTABLE_FRACTION:
        return EstimatorResult(
            vital, float("nan"), unit,
            float(np.median(confidences[keep])) if keep.any() else 0.0,
            "insufficient_quality", waveform=full_waveform,
            detail=dict(n_windows=int(keep.sum()), **report))

    status = "ok" if usable_fraction >= MIN_USABLE_FRACTION else "degraded_capture"
    spread = float(np.percentile(rates[keep], 75) - np.percentile(rates[keep], 25))
    if status == "ok" and spread > MAX_HR_SPREAD_BPM:
        status = "unstable"

    return EstimatorResult(
        vital, float(np.median(rates[keep])), unit,
        float(np.median(confidences[keep])), status, waveform=full_waveform,
        detail=dict(n_windows=int(keep.sum()), spread_bpm=spread, **report))