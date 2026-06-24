"""
Synthetic rPPG window prediction smoke test.

This verifies:
    1. synthetic POS / CHROM / GREEN traces can be generated
    2. traces can be converted to tensor [1, 3, 240]
    3. model predictor accepts the tensor
    4. PredictionResult is returned
"""

from __future__ import annotations
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
APP_DIR = SCRIPT_DIR.parent
REPO_DIR = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from inference.predictor import predict_hr_from_tensor
from models.loader import load_model_bundle
from rppg.windowing import (
    WindowConfig,
    make_model_window_from_channels,
    make_synthetic_rppg_channels,
)


def main() -> None:
    """
    Run synthetic window + model prediction smoke test.
    """
    print("=" * 72)
    print("Synthetic rPPG window prediction smoke test")
    print("=" * 72)
    print(f"App dir:  {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    synthetic_hr_bpm = 72.0
    bundle = load_model_bundle(device="cpu")
    input_config = bundle.model_spec["input"]
    window_config = WindowConfig(
        window_seconds=float(input_config["window_seconds"]),
        target_frames=int(input_config["target_frames"]),
        channel_names=tuple(input_config["channel_names"]),
        normalization=str(input_config["normalization"]),
    )
    signals = make_synthetic_rppg_channels(
        hr_bpm=synthetic_hr_bpm,
        duration_seconds=window_config.window_seconds,
        fps=30.0,
        noise_std=0.05,
        seed=42,
    )
    x = make_model_window_from_channels(
        pos=signals["pos"],
        chrom=signals["chrom"],
        green=signals["green"],
        config=window_config,
    )
    results = predict_hr_from_tensor(x, bundle)

    print("Synthetic input")
    print("-" * 72)
    print(f"synthetic_hr_bpm: {synthetic_hr_bpm}")
    print(f"raw samples: {len(signals['pos'])}")
    print(f"window_seconds: {window_config.window_seconds}")
    print(f"target_frames: {window_config.target_frames}")
    print(f"x.shape: {tuple(x.shape)}")
    print(f"x.mean: {float(x.mean()):.6f}")
    print(f"x.std: {float(x.std()):.6f}")
    print()

    print("Prediction result")
    print("-" * 72)

    for result in results:
        print(f"task: {result.task}")
        print(f"value: {result.value:.2f}")
        print(f"unit: {result.unit}")
        print(f"model_name: {result.model_name}")
        print(f"window_seconds: {result.window_seconds}")
        print(f"target_frames: {result.target_frames}")
        print(f"channel_names: {result.channel_names}")
        print(f"quality status: {result.quality.status}")
        print(f"quality reason: {result.quality.reasons}")
        print()

    print("=" * 72)
    print("PASS: synthetic window prediction smoke test ran successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()