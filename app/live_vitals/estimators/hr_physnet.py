"""Heart-rate estimator backed by the Phase-3 PhysNet v2 checkpoint."""
import numpy as np
import torch

from ..config import (CLIP_LEN, WINDOW_STRIDE, PHYSNET_CKPT,
                      MIN_CONFIDENCE, MIN_USABLE_WINDOWS, MIN_USABLE_FRACTION,
                      MAX_HR_SPREAD_BPM, MAD_OUTLIER_K, MAD_FLOOR_BPM)

from ..preprocess.frames import clip_to_tensor
from ..models.architectures.physnet import PhysNet
from ..signal.hr import hr_from_bvp
from .base import Estimator, EstimatorResult


class HRPhysNet(Estimator):
    """Estimates HR by reconstructing BVP over overlapping clips.

    Per-window heart rates are aggregated by median rather than by stitching the
    predicted waveforms, because separate windows carry no shared phase or scale
    and overlap-adding them can cancel a genuine pulse. The inter-quartile spread
    across windows is retained as a stability signal.
    """

    name = "hr_physnet_v2"
    vital = "heart_rate"
    unit = "bpm"

    def __init__(self, checkpoint=PHYSNET_CKPT, device="cpu"):
        self.checkpoint = checkpoint
        self.device = device
        self._model = None

    def is_available(self):
        return self.checkpoint.exists()

    def _load(self):
        if self._model is None:
            state = torch.load(self.checkpoint, map_location=self.device, weights_only=True)
            model = PhysNet(frames=CLIP_LEN)
            model.load_state_dict(state)
            model.eval().to(self.device)
            self._model = model
        return self._model

    def estimate(self, frames_u8, fps):
        frames_u8 = np.asarray(frames_u8)
        if len(frames_u8) < CLIP_LEN:
            return EstimatorResult(self.vital, float("nan"), self.unit, 0.0,
                                   "insufficient_frames",
                                   detail=dict(n_frames=len(frames_u8), need=CLIP_LEN))

        model = self._load()
        rates, confidences, waves = [], [], []
        for start in range(0, len(frames_u8) - CLIP_LEN + 1, WINDOW_STRIDE):
            tensor = clip_to_tensor(frames_u8[start:start + CLIP_LEN]).to(self.device)
            with torch.no_grad():
                bvp = model(tensor)[0].cpu().numpy()
            reading = hr_from_bvp(bvp, fps)
            if np.isfinite(reading["hr_bpm"]):
                rates.append(reading["hr_bpm"])
                confidences.append(reading["confidence"])
                waves.append(bvp)

        if not rates:
            return EstimatorResult(self.vital, float("nan"), self.unit, 0.0, "no_estimate")

        rates = np.asarray(rates)
        confidences = np.asarray(confidences)

        # Two independent filters. Confidence removes motion-corrupted windows;
        # MAD removes gross outliers such as octave errors, which can carry a
        # respectable confidence and which no confidence threshold reliably catches.
        median_hr = float(np.median(rates))
        mad = max(float(np.median(np.abs(rates - median_hr))) * 1.4826, MAD_FLOOR_BPM)
        keep = (confidences >= MIN_CONFIDENCE) & (np.abs(rates - median_hr) <= MAD_OUTLIER_K * mad)
        usable_fraction = float(keep.sum() / len(rates))

        if keep.sum() < MIN_USABLE_WINDOWS:
            used, status = np.ones_like(keep, dtype=bool), "low_confidence"
        else:
            used = keep
            status = "ok" if usable_fraction >= MIN_USABLE_FRACTION else "degraded_capture"

        spread = float(np.percentile(rates[used], 75) - np.percentile(rates[used], 25))
        confidence = float(np.median(confidences[used]))
        if status == "ok" and spread > MAX_HR_SPREAD_BPM:
            status = "unstable"

        return EstimatorResult(
            self.vital, float(np.median(rates[used])), self.unit, confidence, status,
            waveform=np.concatenate([w for w, k in zip(waves, used) if k]),
            detail=dict(n_windows=int(used.sum()), n_total=len(rates),
                        usable_fraction=usable_fraction,
                        spread_bpm=spread, fps=float(fps)))