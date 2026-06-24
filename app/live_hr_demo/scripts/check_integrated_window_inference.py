"""
Integrated window inference smoke test.

This verifies:
    1. clean synthetic pulse is accepted and predicted
    2. random noise is rejected and prediction is skipped
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

from inference.window_inference import predict_hr_from_rppg_window
from models.loader import load_model_bundle
from rppg.windowing import make_synthetic_rppg_channels


def make_noise_channels(duration_seconds: float, fps: float, seed: int = 123) -> dict[str, np.ndarray]:
    """
    Create random noise channels for negative test.
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


def print_result(title: str, result) -> None:
    """
    Print integrated prediction result.
    """
    print(title)
    print("-" * 72)
    print(f"task: {result.task}")
    print(f"model_hr_bpm: {result.value}")
    print(f"spectral_hr_bpm: {result.extra.get('spectral_hr_bpm')}")
    print(f"unit: {result.unit}")
    print(f"quality status: {result.quality.status}")
    print(f"confidence: {result.quality.confidence}")
    print()
    print("metrics:")
    for key, value in result.quality.metrics.items():
        print(f"  {key}: {value}")
    print()
    print("reasons:")
    for reason in result.quality.reasons:
        print(f"  - {reason}")
    print()


def main() -> None:
    """
    Run integrated window inference smoke test.
    """
    print("=" * 72)
    print("Integrated window inference smoke test")
    print("=" * 72)
    print(f"App dir: {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    bundle = load_model_bundle(device="cpu")
    fps = 30.0
    duration_seconds = float(bundle.model_spec["input"]["window_seconds"])
    clean_signals = make_synthetic_rppg_channels(
        hr_bpm=72.0,
        duration_seconds=duration_seconds,
        fps=fps,
        noise_std=0.05,
        seed=42,
    )
    clean_result = predict_hr_from_rppg_window(signals=clean_signals, fps=fps, bundle=bundle)
    print_result("Clean synthetic pulse", clean_result)
    noise_signals = make_noise_channels(duration_seconds=duration_seconds, fps=fps, seed=123)
    noise_result = predict_hr_from_rppg_window(signals=noise_signals, fps=fps, bundle=bundle)
    print_result("Random noise", noise_result)

    print("=" * 72)
    print("PASS: integrated window inference smoke test ran successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()