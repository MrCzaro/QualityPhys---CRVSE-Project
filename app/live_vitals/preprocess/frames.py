"""Frame cropping and clip normalization matching the Phase-3 contract."""
import numpy as np
import cv2
import torch

from ..config import TARGET_SIZE


def crop_resize(frame_rgb, box, size=TARGET_SIZE):
    """Crops to a relative box clamped to the frame and resizes with INTER_AREA."""
    h, w = frame_rgb.shape[:2]
    x0 = max(0, int(box[0] * w)); y0 = max(0, int(box[1] * h))
    x1 = min(w, int(box[2] * w)); y1 = min(h, int(box[3] * h))
    if x1 <= x0 or y1 <= y0:
        return None
    return cv2.resize(frame_rgb[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)


def clip_to_tensor(frames_u8):
    """Converts [T,H,W,3] uint8 to a normalized [1,3,T,H,W] tensor.

    Applies the training-time normalization: a per-channel z-score computed
    over the whole clip.
    """
    f = np.asarray(frames_u8, dtype=np.float32)
    m = f.mean(axis=(0, 1, 2), keepdims=True)
    sd = f.std(axis=(0, 1, 2), keepdims=True) + 1e-6
    f = np.ascontiguousarray((f - m) / sd)
    return torch.from_numpy(f).permute(3, 0, 1, 2).float().unsqueeze(0)