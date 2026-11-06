"""Respiratory rate from the motion of the torso in a whole-frame video stream.

Research code, connected to nothing in the app: no estimator, registry entry, page or capture
path uses it, and nothing it returns is a reading the app shows. It is readout A of NB_P3_39
(section 3), the readout that passed that notebook's development gate on VitalVideos-WW's
training split, carried here so that the one pre-registered test on the held-out split
(NB_P3_40) runs the code the app would run. Not a medical device.

The readout:

- **Region.** The rows below the padded face box, in grey (the mean of R, G and B), all
  columns. With fewer than MIN_TORSO_ROWS such rows, or fewer than MIN_FRAMES frames, the
  recording is not read.
- **Components.** Each pixel's series is detrended and band-passed to BREATH_BAND_HZ
  (Butterworth, zero-phase); the first N_COMPONENTS principal components of those series come
  from the frames' Gram matrix, the same series as a singular value decomposition of the pixel
  matrix gives, up to sign, at a fraction of its cost when pixels far outnumber frames.
- **Rate.** Each component is read with `resp_rate`; the reading is the rate of the component
  whose spectrum is most concentrated (`peak_share`) among those not refused. There is no
  confidence threshold: every rate the readout gives counts.

`resp_rate` is NB_P3_27's readout, the one that set the belt reference this readout is judged
against, carried over unchanged: a zero-padded Hann FFT, log-parabolic refinement, and a peak
on either edge of 6-42 breaths/min refused as drift or broadband noise rather than breathing.
It is not `signal.hr.hr_from_bvp`'s refinement, whose log guard, flatness threshold and edge
handling differ, so sharing that one would move the readings off NB_P3_39's.

Input: frames as [T, H, W, 3] of the whole frame (uint8 as stored, or any real type) and
`box_ry1`, the bottom of the padded face box as a fraction of the frame's height. The
development used VitalVideos' `frames_full`: the 960-wide, 1056-high video area-averaged
8-fold in each dimension, to 120 by 132, and over blocks of 6 frames, from 60 to 10 fps.
`fs` sets the filter and the spectra, so other rates run, but only that geometry at 10 fps
has been tested on real video.

The constants live here and not in `config.py`: `check_ubfc_regression` certifies every
uppercase constant there against the heart-rate baselines, and none of these belongs to them.
"""
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, detrend, sosfiltfilt

FULL_FS = 10.0  # frames_full: the whole frame at 10 fps
RESP_LOW_HZ, RESP_HIGH_HZ = 0.10, 0.70  # 6-42 breaths/min, NB_P3_27's band
FFT_PAD = 8  # zero padding before the FFT, as a multiple of the series' length
BREATH_BAND_HZ = (0.08, 1.0)  # the pixels' band-pass, a little wider than the band read
BREATH_FILTER_ORDER = 2
MIN_TORSO_ROWS = 8
MIN_FRAMES = 16  # resp_rate's own minimum; the band-pass pads 15 samples at each end
N_COMPONENTS = 5
SHARE_HALF_WIDTH_HZ = 0.03  # peak_share counts the power within this of the peak


class TorsoReading(NamedTuple):
    """One recording's reading. A refused recording reads NaN, NaN and -1."""
    rate_bpm: float  # breaths/min
    share: float  # the chosen component's peak concentration, 0-1
    component: int  # the chosen component's rank, 0 the largest
    rows: int  # rows in the torso region, 0 when too few remain to read


def resp_rate(x, fs, low=RESP_LOW_HZ, high=RESP_HIGH_HZ, pad=FFT_PAD):
    """Returns the dominant respiratory rate of a series in breaths/min, or NaN.

    NB_P3_27's readout, unchanged: a zero-padded FFT with log-parabolic peak refinement.
    A peak on either band edge is refused: it is slow drift or broadband noise rather than
    breathing, and the low edge is where drift parked several readings before the refusal
    was added.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 16 or not np.all(np.isfinite(x)) or x.std() < 1e-12:
        return np.nan
    x = (x - x.mean()) * np.hanning(x.size)
    n = x.size * pad
    power = np.abs(np.fft.rfft(x, n)) ** 2
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    band = np.where((freqs >= low) & (freqs <= high))[0]
    k = int(band[np.argmax(power[band])])
    if k in (band[0], band[-1]):
        return np.nan
    left, centre, right = np.log(power[k - 1:k + 2] + 1e-30)
    denom = left - 2.0 * centre + right
    offset = 0.5 * (left - right) / denom if abs(denom) > 1e-12 else 0.0
    offset = offset if abs(offset) <= 0.5 else 0.0
    return float((k + offset) * (freqs[1] - freqs[0]) * 60.0)


def peak_share(x, fs, pad=FFT_PAD):
    """Share of the 6-42 breaths/min power within SHARE_HALF_WIDTH_HZ of its peak, on
    resp_rate's spectrum: how concentrated a series' breathing band is, from 0 to 1."""
    x = np.asarray(x, dtype=np.float64)
    x = (x - x.mean()) * np.hanning(x.size)
    power = np.abs(np.fft.rfft(x, x.size * pad)) ** 2
    freqs = np.fft.rfftfreq(x.size * pad, 1.0 / fs)
    band = (freqs >= RESP_LOW_HZ) & (freqs <= RESP_HIGH_HZ)
    peak = freqs[band][np.argmax(power[band])]
    return float(power[band & (np.abs(freqs - peak) <= SHARE_HALF_WIDTH_HZ)].sum() / power[band].sum())


def breathing_band(x, fs=FULL_FS):
    """Linear detrend and the BREATH_BAND_HZ band-pass, zero-phase, along the first axis."""
    sos = butter(BREATH_FILTER_ORDER, [BREATH_BAND_HZ[0] / (fs / 2), BREATH_BAND_HZ[1] / (fs / 2)],
                 btype='band', output='sos')
    return sosfiltfilt(sos, detrend(x, axis=0), axis=0)


def torso_region(frames, box_ry1):
    """Grey rows from the bottom of the padded face box to the bottom of the frame, as
    [T, rows, W], or None when fewer than MIN_TORSO_ROWS remain."""
    grey = np.asarray(frames).astype(np.float64).mean(axis=3)
    top = int(np.ceil(box_ry1 * grey.shape[1]))
    return grey[:, top:, :] if grey.shape[1] - top >= MIN_TORSO_ROWS else None


def components(region, fs=FULL_FS, n=N_COMPONENTS):
    """The first n principal components of a region's breathing-band pixel series, as an
    [n, T] array, largest first: U x S of the band-passed, centred pixel matrix, taken from
    its Gram matrix."""
    X = breathing_band(region.reshape(len(region), -1), fs)
    X = X - X.mean(axis=0)
    values, vectors = np.linalg.eigh(X @ X.T)
    return np.stack([vectors[:, j] * np.sqrt(max(values[j], 0.0))
                     for j in np.argsort(values)[::-1][:n]])


def torso_rate(frames, box_ry1, fs=FULL_FS):
    """Readout A: the breathing rate of the torso below the face box, as a TorsoReading.

    Raises ValueError for frames that are not [T, H, W, 3], a box edge outside the frame,
    or a rate too low for the band-pass; a recording it cannot read comes back refused.
    """
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[3] != 3:
        raise ValueError(f'frames must be [T, H, W, 3], got shape {frames.shape}')
    if not 0.0 <= box_ry1 <= 1.0:
        raise ValueError(f'box_ry1 must be a fraction of the frame height, got {box_ry1}')
    if not fs > 2.0 * BREATH_BAND_HZ[1]:
        raise ValueError(f'fs must exceed {2.0 * BREATH_BAND_HZ[1]:g} Hz for the band-pass, got {fs}')
    region = torso_region(frames, box_ry1)
    if region is None:
        return TorsoReading(np.nan, np.nan, -1, 0)
    best = TorsoReading(np.nan, np.nan, -1, region.shape[1])
    if len(region) < MIN_FRAMES:
        return best
    for rank, series in enumerate(components(region, fs)):
        rate = resp_rate(series, fs)
        if np.isfinite(rate):
            share = peak_share(series, fs)
            if not np.isfinite(best.share) or share > best.share:
                best = TorsoReading(rate, share, rank, region.shape[1])
    return best