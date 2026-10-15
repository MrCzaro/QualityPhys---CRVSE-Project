"""Head motion in the face crop, and the periodic-motion capture gate.

The face box is frozen for the whole capture, so a moving head shows up as the crop's
content shifting from frame to frame. During exercise both estimators lock onto that
movement - pedalling, striding, rowing - and report its rhythm as a heart rate with full
confidence (ECG-Fitness, NB_P3_29). A capture whose crop moves too much, or moves
rhythmically by more than a resting head does, is therefore refused rather than read.

The measurement was written for the ECG-Fitness motion diagnostic, and the gate's
thresholds were set by calibrate_motion_gate.py on still recordings only, never on the
exercise data the gate is meant to catch. Both scripts import these functions, so the gate
measures exactly what they measured.
"""
import numpy as np

from .. import config
from ..signal.hr import hr_from_bvp
from ..signal.spectral import bandpass

NAN = float("nan")


def crop_shifts(frames):
    """Frame-to-frame shift of the crop's content (crop px), by cross-correlation.

    Plain cross-correlation of the mean-removed, Hann-weighted grey crop, peak refined to
    sub-pixel by a parabola on each axis. The Hann weight favours the centre, where the
    face is, over the padded border. OpenCV's phaseCorrelate was tried first: its
    whitening hands the decision to sensor noise wherever the crop has little texture,
    and on synthetic crops it returned shifts of thousands of pixels.
    """
    gray = np.asarray(frames, dtype=np.float64).mean(axis=3)
    n, h, w = gray.shape
    weight = np.outer(np.hanning(h), np.hanning(w))
    spectra = np.fft.rfft2((gray - gray.mean(axis=(1, 2), keepdims=True)) * weight)
    shift = np.zeros((n, 2))
    for lo in range(1, n, 256):  # in blocks, to bound memory
        hi = min(n, lo + 256)
        corr = np.fft.irfft2(spectra[lo:hi] * np.conj(spectra[lo - 1:hi - 1]), s=(h, w))
        corr = np.fft.fftshift(corr, axes=(1, 2))
        peak = corr.reshape(len(corr), -1).argmax(axis=1)
        iy, ix = np.unravel_index(peak, (h, w))
        iy, ix = np.clip(iy, 1, h - 2), np.clip(ix, 1, w - 2)
        k = np.arange(len(corr))
        shift[lo:hi, 0] = ix - w // 2 + _vertex(corr[k, iy, ix - 1], corr[k, iy, ix],
                                                 corr[k, iy, ix + 1])
        shift[lo:hi, 1] = iy - h // 2 + _vertex(corr[k, iy - 1, ix], corr[k, iy, ix],
                                                 corr[k, iy + 1, ix])
    return shift


def _vertex(left, centre, right):
    """Sub-sample offset of a parabola's vertex through three points; 0 if not a peak."""
    curvature = left - 2.0 * centre + right
    safe = np.where(np.abs(curvature) > 1e-12, curvature, 1.0)
    offset = np.where(np.abs(curvature) > 1e-12, 0.5 * (left - right) / safe, 0.0)
    return np.where(np.abs(offset) <= 0.5, offset, 0.0)


def window_rhythms(read_tracks, amp_tracks, amp_scale, fps, starts, source):
    """Per window: the rhythm (bpm), periodicity and channel of the most periodic channel
    in `read_tracks`, and the in-band rms displacement of `amp_tracks` times `amp_scale`.

    Every channel is band-passed over the whole recording with the app's cardiac band-pass,
    as POS is, then read on each analysis window with the app's own readout.
    """
    readable = {k: bandpass(v - v.mean(), fps) for k, v in read_tracks.items()}
    moving = [bandpass(v - v.mean(), fps) for v in amp_tracks]
    rows = []
    for s in starts:
        best_hr, best_conf, best_channel = NAN, -1.0, ""
        for channel, signal in readable.items():
            reading = hr_from_bvp(signal[s:s + config.CLIP_LEN], fps)
            if np.isfinite(reading["hr_bpm"]) and reading["confidence"] > best_conf:
                best_hr, best_conf, best_channel = (reading["hr_bpm"], reading["confidence"],
                                                    channel)
        amp = np.sqrt(sum(np.mean(v[s:s + config.CLIP_LEN] ** 2) for v in moving)) * amp_scale
        rows.append({"start": s, f"{source}_rhythm": best_hr,
                     f"{source}_periodicity": best_conf if best_channel else NAN,
                     f"{source}_channel": best_channel, f"{source}_amp": amp})
    return rows


def crop_motion(frames, fps):
    """Per-window crop rhythm, periodicity and in-band displacement, and the recording's
    summary; (None, None) if the recording is shorter than one analysis window.

    Over the app's analysis windows (CLIP_LEN frames, WINDOW_STRIDE apart):
      amp_median   median in-band (cardiac band) rms displacement of the crop, crop pixels
      amp_p90      90th percentile of the same over windows
      clear_frac   share of windows whose crop rhythm reaches MIN_CONFIDENCE periodicity;
                   a window with no readable rhythm counts as not clear
    The crop is read on its frame-to-frame shifts rather than their running sum: the
    registration error is independent from frame to frame, so the sum would turn it into a
    random walk whose power piles up at the bottom of the band. The displacement is measured
    on the running sum, which is where the head actually is.
    """
    n = len(frames)
    if n < config.CLIP_LEN:
        return None, None
    starts = list(range(0, n - config.CLIP_LEN + 1, config.WINDOW_STRIDE))
    shift = crop_shifts(frames)
    position = np.cumsum(shift, axis=0)
    rows = window_rhythms(dict(crop_x=shift[:, 0], crop_y=shift[:, 1]),
                          [position[:, 0], position[:, 1]], 1.0, fps, starts, "crop")
    amp = np.array([r["crop_amp"] for r in rows], dtype=float)
    periodicity = np.array([r["crop_periodicity"] for r in rows], dtype=float)
    clear = np.nan_to_num(periodicity, nan=0.0) >= config.MIN_CONFIDENCE
    rhythm = np.array([r["crop_rhythm"] for r in rows], dtype=float)
    summary = dict(n_frames=n, n_windows=len(rows),
                   amp_median=float(np.median(amp)), amp_p90=float(np.percentile(amp, 90)),
                   clear_frac=float(clear.mean()),
                   rhythm_when_clear=float(np.median(rhythm[clear])) if clear.any() else NAN)
    return rows, summary


def motion_verdict(frames, fps):
    """Classifies a capture's head motion; returns (verdict, note, summary).

    REJECT when the median in-band crop displacement reaches MOTION_REJECT_PX, or reaches
    MOTION_PERIODIC_PX with a clear rhythm in at least MOTION_PERIODIC_SHARE of the windows;
    ACCEPT otherwise. A capture shorter than one window, or without a usable rate, is not
    judged here (summary None): the rate gate and the estimator's own gates deal with it.
    """
    if not np.isfinite(fps) or fps <= 0:
        return "ACCEPT", None, None
    _, summary = crop_motion(frames, fps)
    if summary is None:
        return "ACCEPT", None, None
    amp, periodic = summary["amp_median"], summary["clear_frac"]
    if amp >= config.MOTION_REJECT_PX:
        return "REJECT", (f"the head moved {amp:.1f} px per window in the crop "
                          f"(limit {config.MOTION_REJECT_PX:g}); hold still during the "
                          f"capture"), summary
    if amp >= config.MOTION_PERIODIC_PX and periodic >= config.MOTION_PERIODIC_SHARE:
        return "REJECT", (f"the head moved rhythmically ({amp:.1f} px per window, regular "
                          f"in {periodic:.0%} of windows), as in exercise; that rhythm "
                          f"would be read as a pulse"), summary
    return "ACCEPT", None, summary