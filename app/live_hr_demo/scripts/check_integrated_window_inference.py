"""
Integrated window inference smoke test.

This script verifies the full window-level inference path used before the live
camera app receives real ROI-derived rPPG signals.

It checks two cases:

1. A clean synthetic pulse window should pass signal-quality checks and produce
   a heart-rate prediction.

2. Random noise should fail signal-quality checks, and model prediction should
   be skipped.

The test is intentionally lightweight. It is not a model accuracy evaluation;
it only checks that quality gating, spectral analysis, and model inference are
connected correctly.
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
    Create random POS/CHROM/GREEN noise channels for a negative smoke test.

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


def print_result(title: str, result) -> None:
    """
    Print one integrated prediction result.

    Parameters
    ----------
    title:
        Section title shown above the result.

    result:
        Prediction result returned by ``predict_hr_from_rppg_window``.
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
    Run the integrated window inference smoke test.

    The test loads the CRVSE model bundle, evaluates a clean synthetic pulse
    window, evaluates a random-noise window, and prints both quality decisions.
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
    clean_signals = make_synthetic_rppg_channels(hr_bpm=72.0, duration_seconds=duration_seconds, 
                                                 fps=fps, noise_std=0.05, seed=42)
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