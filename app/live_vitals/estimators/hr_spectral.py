"""Heart-rate estimator using classical spectral rPPG, as a model-free cross-check."""
import numpy as np

from ..config import CLIP_LEN, WINDOW_STRIDE, MIN_CONFIDENCE, SPECTRAL_METHOD
from ..signal.hr import hr_from_bvp
from ..signal.spectral import METHODS, rgb_trace
from .base import Estimator, EstimatorResult, aggregate_windows


def _summarise(readings):
    """Median HR across one method's windows, under the shared confidence gate."""
    hr = np.array([r["hr_bpm"] for r in readings], dtype=float)
    confidence = np.array([r["confidence"] for r in readings], dtype=float)
    keep = np.isfinite(hr) & (confidence >= MIN_CONFIDENCE)
    return dict(hr_bpm=float(np.median(hr[keep])) if keep.any() else float("nan"),
                n_windows=int(keep.sum()), n_total=len(readings))


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
                       method_hr={name: _summarise(rows)
                                  for name, rows in readings.items()}))