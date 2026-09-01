"""Heart-rate readout from a reconstructed BVP waveform."""
import numpy as np

from ..config import (HR_LOW_HZ, HR_HIGH_HZ, FFT_PAD_FACTOR, SUBHARMONIC_RATIO,
                      HARMONIC_GUARD_ENABLED)


def _refine_peak_parabolic(power, peak_idx, lower_idx, upper_idx, eps=1e-20):
    """Refines a spectral peak to sub-bin resolution via a log-parabola fit.

    Refinement is skipped when the peak sits on a band edge, when the curvature
    is degenerate, or when the offset falls outside the +/- 0.5 bin range a
    genuine local maximum must satisfy.
    """
    if peak_idx <= lower_idx or peak_idx >= upper_idx:
        return float(peak_idx)
    left = float(np.log(float(power[peak_idx - 1]) + eps))
    centre = float(np.log(float(power[peak_idx]) + eps))
    right = float(np.log(float(power[peak_idx + 1]) + eps))
    denominator = left - 2.0 * centre + right
    if abs(denominator) < 1e-12:
        return float(peak_idx)
    offset = 0.5 * (left - right) / denominator
    if not np.isfinite(offset) or abs(offset) > 0.5:
        return float(peak_idx)
    return float(peak_idx) + offset


def hr_from_bvp(bvp, fps, low_hz=HR_LOW_HZ, high_hz=HR_HIGH_HZ,
                pad_factor=FFT_PAD_FACTOR, harmonic_guard=HARMONIC_GUARD_ENABLED,
                subharmonic_ratio=SUBHARMONIC_RATIO):
    """Estimates HR (bpm) and a spectral-concentration confidence from a BVP clip.

    Zero-pads before the FFT and refines the peak parabolically, which recovers
    the sub-bin resolution a short clip cannot otherwise resolve. When
    `harmonic_guard` is set, a sufficiently strong sub-harmonic is preferred over
    the raw maximum, guarding against the octave error that appears when the
    fundamental is weaker than its first harmonic.
    """
    x = np.asarray(bvp, dtype=np.float64)
    x = x - x.mean()
    if x.size < 16 or not np.all(np.isfinite(x)) or x.std() < 1e-9:
        return dict(hr_bpm=float("nan"), confidence=0.0, status="degenerate")

    n_fft = int(x.size * max(1, pad_factor))
    power = np.abs(np.fft.rfft(x * np.hanning(x.size), n=n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(fps))
    band = np.where((freqs >= low_hz) & (freqs <= high_hz))[0]
    if band.size == 0 or power[band].sum() <= 0:
        return dict(hr_bpm=float("nan"), confidence=0.0, status="no_band")

    lower_idx, upper_idx = int(band[0]), int(band[-1])
    peak_idx = int(band[np.argmax(power[band])])
    chosen, status = peak_idx, "fundamental"

    if harmonic_guard:
        sub_hz = freqs[peak_idx] / 2.0
        if sub_hz >= low_hz:
            target = int(np.argmin(np.abs(freqs - sub_hz)))
            half_win = max(2, int(0.05 / (freqs[1] - freqs[0])))
            w0, w1 = max(lower_idx, target - half_win), min(upper_idx, target + half_win)
            if w1 > w0:
                local = int(w0 + np.argmax(power[w0:w1 + 1]))
                if power[local] >= subharmonic_ratio * power[peak_idx]:
                    chosen, status = local, "subharmonic_adopted"
    # An argmax sitting on a band edge is not a peak. The spectrum is still
    # rising as it leaves the search band, so the real maximum is outside it, and
    # what comes back is the edge frequency itself rather than a heart rate. This
    # is what a window with no cardiac content looks like when drift and motion
    # energy below the low cutoff dominate, and no confidence threshold catches
    # it: the power genuinely is concentrated at the edge.
    if chosen <= lower_idx or chosen >= upper_idx:
        return dict(hr_bpm=float("nan"), confidence=0.0, status="band_edge")

    refined = _refine_peak_parabolic(power, chosen, lower_idx, upper_idx)
    f_hz = float(np.interp(refined, np.arange(power.size), freqs))

    # Power within one unpadded-bin width of the peak, over total in-band power.
    # Integrating a neighbourhood keeps this meaningful under zero-padding.
    bin_hz = float(freqs[1] - freqs[0])
    lobe = max(1, int(round((float(fps) / x.size) / max(bin_hz, 1e-12))))
    c0, c1 = max(lower_idx, chosen - lobe), min(upper_idx, chosen + lobe)
    confidence = float(power[c0:c1 + 1].sum() / power[band].sum())

    return dict(hr_bpm=f_hz * 60.0, confidence=confidence, status=status)