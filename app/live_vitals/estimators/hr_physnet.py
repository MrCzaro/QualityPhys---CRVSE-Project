"""Heart-rate estimator backed by the Phase-3 PhysNet v2 checkpoint."""
import numpy as np
import torch

from ..config import (CLIP_LEN, WINDOW_STRIDE, PHYSNET_CKPT,
                      MIN_CONFIDENCE, MAX_HR_SPREAD_BPM)
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
        spread = float(np.percentile(rates, 75) - np.percentile(rates, 25))
        confidence = float(np.median(confidences))

        status = "ok"
        if spread > MAX_HR_SPREAD_BPM:
            status = "unstable"
        elif confidence < MIN_CONFIDENCE:
            status = "low_confidence"

        return EstimatorResult(
            self.vital, float(np.median(rates)), self.unit, confidence, status,
            waveform=np.concatenate(waves),
            detail=dict(n_windows=len(rates), spread_bpm=spread, fps=float(fps)))