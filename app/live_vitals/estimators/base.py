"""Common interface for per-vital estimator plugins."""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


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