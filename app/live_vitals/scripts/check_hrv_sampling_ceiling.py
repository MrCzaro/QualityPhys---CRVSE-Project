"""How much HRV error does the camera's frame rate impose, before any model?

HRV is a measurement of *when* beats happen, so it is limited by how finely beat
times can be resolved. A 30 fps camera places one sample every 33 ms, while a
resting RMSSD is typically 20-50 ms: the quantisation is the same size as the
quantity. This script measures that ceiling directly, using no model at all.

MCD-rPPG ships a 100 Hz contact PPG recorded simultaneously with the video, and a
500 Hz ECG recorded separately at the medical stand. The PPG is resampled down to
each candidate frame rate and beat times are recovered from it, so the only thing
that changes between rows is the sampling rate. Whatever error appears is a floor
that no rPPG model can beat, because the model's output is sampled at frame rate
too.

Two beat-time estimators are compared at each rate:

  nearest sample   the peak's integer index, which is what naive peak-finding gives
  parabolic        a 3-point parabola through the peak, which recovers sub-sample
                   position and costs nothing

The ECG column is not resampled. It is there to say what these subjects' HRV
actually is at a rate where quantisation is negligible, so the PPG reference can
be sanity-checked rather than trusted blindly.

Usage:
    python check_hrv_sampling_ceiling.py [--data-dir DIR] [--rates 60 30 25]

No frames are decoded and nothing is written.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

DEFAULT_DATA_DIR = Path(r"D:\QualityPhys\demo_data\mcd_rppg_subset")
SUBJECTS = ["4087", "4874", "5130", "6137", "8584", "9092"]
STATES = ["before", "after"]
DEFAULT_RATES = [100.0, 60.0, 30.0, 29.9, 25.0, 20.0, 15.0]
REFERENCE_HZ = 100.0
STAMP = "%Y-%m-%d %H:%M:%S.%f"


def load_ppg(path):
    """Returns (values, seconds) from a .PW file of 'value  timestamp' lines."""
    values, stamps = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        head, _, tail = line.partition("  ")
        values.append(float(head))
        stamps.append(tail.strip())
    t0 = datetime.strptime(stamps[0], STAMP)
    seconds = np.array([(datetime.strptime(s, STAMP) - t0).total_seconds()
                        for s in stamps])
    return np.asarray(values, dtype=float), seconds


def load_ecg(path, lead="II"):
    """Returns (values, sample_rate) for one ECG lead."""
    payload = json.load(open(path))
    leads = {entry["title"]: np.asarray(entry["values"], dtype=float)
             for entry in payload["data"]}
    return leads.get(lead, next(iter(leads.values()))), float(payload["frequency"])


def bandpass(x, fs, low_hz, high_hz, order=2):
    b, a = butter(order, [low_hz / (fs / 2), min(high_hz / (fs / 2), 0.99)],
                  btype="band")
    return filtfilt(b, a, np.asarray(x, dtype=float))


def refine(y, index):
    """Sub-sample peak position by fitting a parabola through three points.

    Returns the index unchanged at an array edge or where the curvature is
    degenerate, both of which mean there is no interior maximum to refine.
    """
    if index <= 0 or index >= len(y) - 1:
        return float(index)
    left, centre, right = float(y[index - 1]), float(y[index]), float(y[index + 1])
    denominator = left - 2.0 * centre + right
    if abs(denominator) < 1e-12:
        return float(index)
    offset = 0.5 * (left - right) / denominator
    return float(index) + (offset if abs(offset) <= 0.5 else 0.0)


def beat_times(y, fs, sub_sample, low_hz=0.7, high_hz=3.5):
    """Returns beat times in seconds from a pulse waveform."""
    filtered = bandpass(y, fs, low_hz, high_hz)
    peaks, _ = find_peaks(filtered, distance=max(1, int(0.3 * fs)),
                          height=0.2 * np.std(filtered))
    if sub_sample:
        peaks = np.array([refine(filtered, int(p)) for p in peaks], dtype=float)
    return np.asarray(peaks, dtype=float) / fs


def r_peak_times(y, fs):
    """Returns R-peak times via the Pan-Tompkins chain.

    Detecting on the raw ECG lets a tall T-wave register as a beat, which halves
    every RR interval it touches; isolating the QRS by its 5-15 Hz energy first
    avoids that.
    """
    band = bandpass(y - np.mean(y), fs, 5.0, 15.0)
    width = max(1, int(0.15 * fs))
    energy = np.convolve(np.diff(band, prepend=band[0]) ** 2,
                         np.ones(width) / width, mode="same")
    floor = energy[energy > np.percentile(energy, 90)]
    peaks, _ = find_peaks(energy, height=0.3 * np.median(floor),
                          distance=int(0.28 * fs))
    return np.asarray(peaks, dtype=float) / fs


def hrv(times, low_s=0.3, high_s=2.0):
    """Returns (RMSSD ms, SDNN ms, mean HR bpm, n) from a beat-time series.

    Intervals outside a plausible physiological range are dropped, and successive
    differences are taken only across intervals that survived as neighbours, so a
    single missed beat cannot manufacture a large difference.
    """
    rr = np.diff(np.asarray(times, dtype=float))
    ok = (rr > low_s) & (rr < high_s)
    rr = rr[ok]
    if len(rr) < 8:
        return float("nan"), float("nan"), float("nan"), len(rr)
    adjacent = ok[:-1] & ok[1:]
    diffs = np.diff(np.diff(np.asarray(times, dtype=float)))[adjacent]
    rmssd = float(np.sqrt(np.mean(diffs ** 2)) * 1000.0)
    return rmssd, float(np.std(rr, ddof=1) * 1000.0), float(60.0 / np.mean(rr)), len(rr)


def quantisation_rmssd_ms(fs):
    """RMSSD a perfectly regular pulse would show from sampling jitter alone.

    A beat time rounded to the nearest sample carries error uniform over one
    sample period, variance h^2/12. A successive difference of intervals combines
    three beat times as t[i] - 2t[i+1] + t[i+2], so its variance is 6h^2/12, and
    RMSSD adds that in quadrature with the true value.
    """
    h = 1.0 / float(fs)
    return float(np.sqrt(h * h / 2.0) * 1000.0)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    args = parser.parse_args()

    print("Beat timing is the whole of HRV, so this measures what sampling rate")
    print("alone costs. No model is involved; the same contact PPG is simply")
    print("resampled to each rate.\n")

    records, ecg_rows = [], []
    for subject in SUBJECTS:
        for state in STATES:
            folder = args.data_dir / subject
            ppg_path = folder / f"{subject}_{state}.PW"
            if not ppg_path.exists():
                print(f"  missing {ppg_path.name}, skipped")
                continue
            values, seconds = load_ppg(ppg_path)
            grid = np.arange(0.0, seconds[-1], 1.0 / REFERENCE_HZ)
            records.append((f"{subject}_{state}", np.interp(grid, seconds, values),
                            seconds[-1]))
            ecg_path = folder / f"{subject}_{state}.json"
            if ecg_path.exists():
                y, fs = load_ecg(ecg_path)
                ecg_rows.append(hrv(r_peak_times(y, fs))[:2])

    if not records:
        print("No recordings found. Pass --data-dir pointing at mcd_rppg_subset.")
        return 2
    print(f"{len(records)} recordings, {len(ecg_rows)} with ECG\n")

    if ecg_rows:
        e_rmssd = np.array([r[0] for r in ecg_rows], dtype=float)
        e_sdnn = np.array([r[1] for r in ecg_rows], dtype=float)
        print("500 Hz ECG, quantisation negligible -- what these subjects' HRV "
              "actually is:")
        print(f"  RMSSD median {np.nanmedian(e_rmssd):6.1f} ms   "
              f"range {np.nanmin(e_rmssd):.1f} - {np.nanmax(e_rmssd):.1f}")
        print(f"  SDNN  median {np.nanmedian(e_sdnn):6.1f} ms   "
              f"range {np.nanmin(e_sdnn):.1f} - {np.nanmax(e_sdnn):.1f}\n")

    reference = {}
    for name, wave, _ in records:
        reference[name] = hrv(beat_times(wave, REFERENCE_HZ, sub_sample=True))

    ref_rmssd = np.array([reference[n][0] for n, _, _ in records], dtype=float)
    ref_sdnn = np.array([reference[n][1] for n, _, _ in records], dtype=float)
    print(f"100 Hz contact PPG reference (sub-sample refined): "
          f"RMSSD median {np.nanmedian(ref_rmssd):.1f} ms, "
          f"SDNN median {np.nanmedian(ref_sdnn):.1f} ms\n")

    for sub_sample in (False, True):
        label = "parabolic sub-sample" if sub_sample else "nearest sample"
        print(f"--- beat time from the {label} ---")
        print(f"{'rate':>8} {'RMSSD':>8} {'err':>8} {'err %':>7} "
              f"{'SDNN':>8} {'err':>8} {'predicted':>10} {'HR err':>7}")
        for rate in args.rates:
            rmssd, sdnn, hr_err = [], [], []
            for name, wave, duration in records:
                grid = np.arange(0.0, duration, 1.0 / rate)
                source = np.arange(len(wave)) / REFERENCE_HZ
                low = np.interp(grid, source, wave)
                got = hrv(beat_times(low, rate, sub_sample))
                want = reference[name]
                rmssd.append((got[0], want[0]))
                sdnn.append((got[1], want[1]))
                hr_err.append(got[2] - want[2])
            got_r = np.array([g for g, _ in rmssd], dtype=float)
            want_r = np.array([w for _, w in rmssd], dtype=float)
            got_s = np.array([g for g, _ in sdnn], dtype=float)
            want_s = np.array([w for _, w in sdnn], dtype=float)
            err_r = np.nanmedian(np.abs(got_r - want_r))
            predicted = quantisation_rmssd_ms(rate) if not sub_sample else float("nan")
            print(f"{rate:7.1f}H {np.nanmedian(got_r):8.1f} {err_r:8.1f} "
                  f"{100 * err_r / np.nanmedian(want_r):6.0f}% "
                  f"{np.nanmedian(got_s):8.1f} "
                  f"{np.nanmedian(np.abs(got_s - want_s)):8.1f} "
                  f"{predicted:10.1f} "
                  f"{np.nanmedian(np.abs(np.array(hr_err, dtype=float))):7.2f}")
        print()

    print("Read the 30 Hz row against the ECG RMSSD above. If the error there is a")
    print("large fraction of the real value, frame rate caps video HRV no matter")
    print("how good the reconstruction is, and the 'predicted' column says how much")
    print("of that is pure rounding of beat times to the nearest frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())