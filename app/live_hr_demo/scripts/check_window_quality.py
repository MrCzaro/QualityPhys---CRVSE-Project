"""
Window quality smoke test.

This script verifies the signal-quality gate used before window-level HR
prediction.

It checks two cases:

1. A clean synthetic rPPG pulse should be accepted with good confidence.
2. Random noise should be rejected because channel spectral peaks are unstable
   or disagree with each other.

This test does not run model inference. It only validates quality metrics,
quality decisions, and diagnostic reasons.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
APP_DIR = SCRIPT_DIR.parent
REPO_DIR = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from rppg.quality import evaluate_window_quality
from rppg.windowing import make_synthetic_rppg_channels


def make_noise_channels(duration_seconds: float, fps: float, seed: int = 123) -> dict[str, np.ndarray]:
    """
    Create random POS/CHROM/GREEN noise channels for a negative quality test.

    Parameters
    ----------
    duration_seconds:
        Length of the synthetic window in seconds.

    fps:
        Sampling rate used to construct the time axis.

    seed:
        Random seed for reproducible noise generation.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing ``time``, ``pos``, ``chrom``, and ``green`` arrays.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_seconds * fps))
    time = np.arange(n_samples, dtype=np.float32) / fps

    return {
        "time": time,
        "pos": rng.normal(0.0, 1.0, size=n_samples).astype(np.float32),
        "chrom": rng.normal(0.0, 1.0, size=n_samples).astype(np.float32),
        "green": rng.normal(0.0, 1.0, size=n_samples).astype(np.float32),
    }


def validate_quality_decisions(clean_quality, noise_quality) -> None:
    """
    Validate expected quality decisions for clean pulse and random noise.

    Parameters
    ----------
    clean_quality:
        Quality result for the clean synthetic pulse window.

    noise_quality:
        Quality result for the random-noise window.

    Raises
    ------
    ValueError
        If the clean pulse is rejected or random noise is accepted.
    """
    if not clean_quality.accepted:
        raise ValueError("Expected clean synthetic pulse to be accepted.")
    if clean_quality.confidence != "good":
        raise ValueError(
            f"Expected clean synthetic pulse confidence 'good', "
            f"got {clean_quality.confidence!r}."
        )
    if noise_quality.accepted:
        raise ValueError("Expected random noise to be rejected.")
    if noise_quality.confidence != "rejected":
        raise ValueError(
            f"Expected random noise confidence 'rejected', "
            f"got {noise_quality.confidence!r}."
        )


def print_quality_result(title: str, quality) -> None:
    """
    Print one quality result in a readable format.

    Parameters
    ----------
    title:
        Section title shown above the quality result.

    quality:
        Quality result returned by ``evaluate_window_quality``.
    """
    print(title)
    print("-" * 72)
    print(f"accepted: {quality.accepted}")
    print(f"confidence: {quality.confidence}")
    print()

    print("metrics:")
    for key, value in quality.metrics.items():
        print(f"{key}: {value}")

    print()
    print("reasons:")
    for reason in quality.reasons:
        print(f" - {reason}")

    print()


def main() -> None:
    """
    Run the window quality smoke test.
    """
    print("=" * 72)
    print("Window quality smoke test")
    print("=" * 72)
    print(f"App dir: {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    fps = 30.0
    duration_seconds = 8.0
    clean_signals = make_synthetic_rppg_channels(
        hr_bpm=72.0,
        duration_seconds=duration_seconds,
        fps=fps,
        noise_std=0.05,
        seed=42,
    )
    clean_quality = evaluate_window_quality(signals=clean_signals, fps=fps)
    noise_signals = make_noise_channels(duration_seconds=duration_seconds, fps=fps, seed=123)
    noise_quality = evaluate_window_quality(signals=noise_signals, fps=fps)
    validate_quality_decisions(clean_quality=clean_quality, noise_quality=noise_quality)

    print_quality_result("Clean synthetic pulse", clean_quality)
    print_quality_result("Random noise", noise_quality)
    print("=" * 72)
    print("PASS: window quality smoke test ran successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()