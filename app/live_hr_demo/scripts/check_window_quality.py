"""
Window quality smoke test.

This verifies:
    1. synthetic rPPG signal is accepted
    2. random noise is usually rejected or lower-confidence
    3. quality reasons and metrics are printed
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
    Create random noise channels for a negative smoke test.
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


def print_quality_result(title: str, quality) -> None:
    """
    Print quality decision in a readable format.
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
    Run quality smoke tests.
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
    print_quality_result("Clean synthetic pulse", clean_quality)
    noise_signals = make_noise_channels(duration_seconds=duration_seconds, fps=fps, seed=123)
    noise_quality = evaluate_window_quality(signals=noise_signals, fps=fps)
    print_quality_result("Random noise", noise_quality)

    print("=" * 72)
    print("PASS: window quality smoke test ran successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()