"""Heart-rate estimator using classical spectral rPPG, as a model-free cross-check."""
import numpy as np

from ..config import CLIP_LEN, WINDOW_STRIDE, SPECTRAL_METHOD
from ..signal.hr import hr_from_bvp
from ..signal.spectral import METHODS, rgb_trace
from .base import Estimator, EstimatorResult, aggregate_windows


def _summarise(vital, unit, readings, fps, n_attempted):
    """One method's reading, under exactly the gates the reported value uses.

    Routed through `aggregate_windows` rather than re-applying MIN_CONFIDENCE by
    hand. Applying only the confidence gate here left the diagnostics table free
    to report a POS value the reported figure had already refused: 3 surviving
    windows of 16 passes MIN_CONFIDENCE but fails MIN_REPORTABLE_FRACTION, so the
    table showed 73.1 bpm beside a headline that said there was no reading. A
    cross-check whose methods are gated on different terms cannot tell a real
    disagreement from a threshold artefact, which is the only thing it is for.
    """
    rates, confidences = [], []
    for reading in readings:
        if np.isfinite(reading["hr_bpm"]):
            rates.append(reading["hr_bpm"])
            confidences.append(reading["confidence"])
    result = aggregate_windows(vital, unit, rates, confidences, [], fps,
                               n_attempted)
    return dict(hr_bpm=result.value, status=result.status,
                n_windows=(result.detail or {}).get("n_windows", 0),
                n_total=n_attempted)


class HRSpectral(Estimator):
    """Estimates HR from the per-frame mean colour of the face crop.

    Carries no learned parameters, so it fails on different things than a trained
    model does and its agreement with one is real corroboration. On the eight
    held-out UBFC subjects it reaches 1.29 bpm window MAE against a reference-BVP
    readout, within noise of the PhysNet checkpoint at 1.24. That says UBFC is an
    easy corpus, not that the model is redundant: the gap should open on motion,
    poor light and darker skin, where the classical projections are known to
    degrade, and exposing that is what this cross-check is for.
    """

    name = "hr_spectral"
    vital = "heart_rate"
    unit = "bpm"

    def __init__(self, method=SPECTRAL_METHOD):
        if method not in METHODS:
            raise KeyError(f"unknown method {method!r}; available: {sorted(METHODS)}")
        self.method = method

    def is_available(self):
        """Always true: the method is arithmetic, with no checkpoint to find."""
        return True

    def estimate(self, frames_u8, fps):
        frames_u8 = np.asarray(frames_u8)
        if len(frames_u8) < CLIP_LEN:
            return EstimatorResult(self.vital, float("nan"), self.unit, 0.0,
                                   "insufficient_frames",
                                   detail=dict(n_frames=len(frames_u8), need=CLIP_LEN))

        # The signals are built over the whole capture and only then read out per
        # window. POS and CHROM each adapt their projection on a short window of
        # their own, and confining them to one analysis window would truncate the
        # normalisation that makes them work.
        rgb = rgb_trace(frames_u8)
        signals = {name: fn(rgb, fps) for name, fn in METHODS.items()}
        starts = list(range(0, len(frames_u8) - CLIP_LEN + 1, WINDOW_STRIDE))
        readings = {name: [hr_from_bvp(signal[i:i + CLIP_LEN], fps) for i in starts]
                    for name, signal in signals.items()}

        primary = signals[self.method]
        rates, confidences, waves = [], [], []
        for start, reading in zip(starts, readings[self.method]):
            if np.isfinite(reading["hr_bpm"]):
                rates.append(reading["hr_bpm"])
                confidences.append(reading["confidence"])
                waves.append(primary[start:start + CLIP_LEN])

        # All three methods are summarised, not only the configured one. They fail
        # on different things, so the spread between them is a quality signal that
        # neither confidence nor window count reports.
        return aggregate_windows(
            self.vital, self.unit, rates, confidences, waves, fps, len(starts),
            extra=dict(method=self.method,
            method_hr={name: _summarise(self.vital, self.unit, rows, fps, len(starts))
                                  for name, rows in readings.items()}))