"""Classical spectral rPPG signals: POS, CHROM and GREEN.

These recover a pulse from the per-frame mean colour of the face crop using
published linear projections rather than a learned model, so they share no
weights, no training data and no failure modes with the neural estimator.
Both paths read heart rate out through the same `signal.hr.hr_from_bvp`, so a
disagreement between them is a difference in the signal rather than in the
readout.

POS:   Wang et al., "Algorithmic Principles of Remote PPG", IEEE TBME 2017.
CHROM: de Haan & Jeanne, "Robust Pulse Rate from Chrominance-Based rPPG",
       IEEE TBME 2013.
"""
import numpy as np
from scipy.signal import butter, filtfilt

from ..config import (HR_LOW_HZ, HR_HIGH_HZ, SPECTRAL_ROI_FRACTION,
                      SPECTRAL_WINDOW_SECONDS, SPECTRAL_FILTER_ORDER)


def rgb_trace(frames_u8, roi_fraction=SPECTRAL_ROI_FRACTION):
    """Returns the [T,3] per-frame mean RGB of the centre of each face crop.

    The 72x72 crop is a sufficient source. INTER_AREA is an area average, so the
    crop mean is the full-resolution box mean, and averaging several hundred
    pixels puts uint8 quantisation well over an order of magnitude below the
    pulse modulation, which is a fraction of a percent of the mean intensity.

    Which part of the crop is used matters far more than its resolution.
    CROP_PADDING leaves the outer crop full of hair, shoulders and background
    whose brightness changes are not pulsatile, and averaging them in only adds
    noise. Measured across the eight held-out UBFC subjects, CHROM window MAE
    falls from 2.68 bpm over the whole crop to 1.29 bpm over the central 35%.
    """
    frames = np.asarray(frames_u8)
    height, width = frames.shape[1], frames.shape[2]
    dy = min(int(height * (1.0 - roi_fraction) / 2.0), (height - 1) // 2)
    dx = min(int(width * (1.0 - roi_fraction) / 2.0), (width - 1) // 2)
    centre = frames[:, dy:height - dy, dx:width - dx, :]
    return centre.reshape(len(centre), -1, 3).mean(axis=1).astype(np.float64)


def bandpass(x, fps, low_hz=HR_LOW_HZ, high_hz=HR_HIGH_HZ,
             order=SPECTRAL_FILTER_ORDER):
    """Zero-phase Butterworth bandpass over the cardiac band.

    Returns the signal untouched when it is too short for filtfilt's padding
    rather than raising, because a short window is the caller's decision to make
    and hr_from_bvp already refuses signals it cannot read.
    """
    x = np.asarray(x, dtype=np.float64)
    nyquist = 0.5 * float(fps)
    b, a = butter(int(order), [low_hz / nyquist, min(high_hz / nyquist, 0.99)],
                  btype="band")
    if len(x) <= 3 * max(len(a), len(b)):
        return x
    return filtfilt(b, a, x)


def pos(rgb, fps, window_seconds=SPECTRAL_WINDOW_SECONDS):
    """POS: per-window plane-orthogonal-to-skin projection, overlap-added.

    The sliding window is the algorithm, not an optimisation: normalising and
    projecting over a short window is what makes the projection invariant to the
    slow changes in skin tone and illumination that a whole-capture normalisation
    would leave in.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    n_frames = len(rgb)
    length = int(round(window_seconds * float(fps)))
    pulse = np.zeros(n_frames)
    if length < 8 or n_frames < length:
        return pulse

    for start in range(0, n_frames - length + 1):
        block = rgb[start:start + length]
        mean = block.mean(axis=0)
        if np.any(mean <= 1e-9):
            continue
        normalized = block / mean
        s1 = normalized[:, 1] - normalized[:, 2]
        s2 = normalized[:, 1] + normalized[:, 2] - 2.0 * normalized[:, 0]
        sd2 = s2.std()
        combined = s1 + (s1.std() / sd2) * s2 if sd2 > 1e-12 else s1
        sd = combined.std()
        if sd > 1e-12:
            pulse[start:start + length] += (combined - combined.mean()) / sd
    return bandpass(pulse, fps)


def chrom(rgb, fps, window_seconds=SPECTRAL_WINDOW_SECONDS):
    """CHROM: two chrominance projections combined to cancel specular reflection.

    The projections are filtered once over the whole capture rather than inside
    each window: a fourth-order filter settling over roughly a second cannot be
    trusted on a 48-sample window. The window still sets the mixing ratio, which
    is what adapts the method to the subject and the lighting.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    n_frames = len(rgb)
    length = int(round(window_seconds * float(fps)))
    length += length % 2
    pulse = np.zeros(n_frames)
    if length < 8 or n_frames < length:
        return pulse

    mean = rgb.mean(axis=0)
    normalized = rgb / np.where(mean > 1e-9, mean, 1.0)
    x_all = bandpass(3.0 * normalized[:, 0] - 2.0 * normalized[:, 1], fps)
    y_all = bandpass(1.5 * normalized[:, 0] + normalized[:, 1]
                     - 1.5 * normalized[:, 2], fps)

    taper = np.hanning(length)
    for start in range(0, n_frames - length + 1, max(1, length // 2)):
        x = x_all[start:start + length]
        y = y_all[start:start + length]
        combined = x - (x.std() / (y.std() + 1e-12)) * y
        sd = combined.std()
        if sd > 1e-12:
            pulse[start:start + length] += ((combined - combined.mean()) / sd) * taper
    return pulse


def green(rgb, fps):
    """Bandpassed mean green channel, inverted so systole reads as a rising edge.

    Kept for comparison rather than for use. It carries no specular rejection, so
    it follows illumination as readily as blood volume: on the held-out UBFC
    subjects it averages 8.8 bpm window MAE against CHROM's 1.3, and on one
    subject it locks onto a non-cardiac component 73 bpm away from the truth.
    """
    return bandpass(-(rgb[:, 1] - rgb[:, 1].mean()), fps)


# Ordered so the configured default is reported first in any diagnostic listing.
METHODS = {"chrom": chrom, "pos": pos, "green": green}