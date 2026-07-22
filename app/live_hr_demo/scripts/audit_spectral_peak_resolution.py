"""
Spectral peak frequency resolution audit.

Measures how accurately estimate_spectral_sqi recovers a known synthetic pulse
frequency, across several window durations.

Run once before applying parabolic peak interpolation to record the baseline,
then again afterwards. The script labels which variant it is measuring.

This is a resolution experiment, not a smoke test, and is not part of
run_smoke_tests.py.
"""
from __future__ import annotations
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
APP_DIR = SCRIPT_DIR.parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import numpy as np

from rppg.sqi import estimate_spectral_sqi


def detect_variant() -> str:
    """Return a label describing which peak-estimation variant is installed."""
    try:
        from rppg.sqi import _refine_peak_bin_by_parabolic_interpolation  # noqa: F401
    except ImportError:
        return "bin-centre peak (baseline)"

    return "parabolic peak interpolation"


def measure_peak_error(
    fps: float,
    duration_s: float,
    noise_std: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """
    Measure absolute HR error against synthetic tones of known frequency.

    Parameters
    ----------
    fps:
        Sampling frequency in Hz.

    duration_s:
        Window duration in seconds.

    noise_std:
        Standard deviation of additive Gaussian noise.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        Bin width and error statistics in BPM.
    """
    n_samples = int(round(fps * duration_s))
    time_s = np.arange(n_samples) / fps
    rng = np.random.default_rng(seed)

    errors = []

    for true_bpm in np.arange(55.0, 101.0, 1.0):
        signal = np.sin(2.0 * np.pi * (true_bpm / 60.0) * time_s)
        signal = signal + rng.normal(0.0, noise_std, size=n_samples)

        result = estimate_spectral_sqi(
            signal=signal.astype(np.float32),
            fps=fps,
        )

        errors.append(abs(float(result.dominant_bpm) - float(true_bpm)))

    errors = np.asarray(errors, dtype=np.float64)

    return {
        "bin_width_bpm": 60.0 * fps / n_samples,
        "mean_abs_error_bpm": float(errors.mean()),
        "max_abs_error_bpm": float(errors.max()),
        "n_samples": float(n_samples),
    }


def main() -> None:
    """Run the resolution audit across representative window durations."""
    fps = 17.6

    print(f"variant : {detect_variant()}")
    print(f"fps     : {fps}")
    print("")
    print(f"{'window_s':>9} {'n':>5} {'bin_bpm':>8} {'mean_err':>9} {'max_err':>8}")

    for duration_s in (8.0, 12.0, 15.0, 30.0):
        stats = measure_peak_error(fps=fps, duration_s=duration_s)

        print(
            f"{duration_s:>9.1f} "
            f"{int(stats['n_samples']):>5d} "
            f"{stats['bin_width_bpm']:>8.2f} "
            f"{stats['mean_abs_error_bpm']:>9.3f} "
            f"{stats['max_abs_error_bpm']:>8.3f}"
        )


if __name__ == "__main__":
    main()