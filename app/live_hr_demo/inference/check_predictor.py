"""
Predictor smoke test.

This verifies:
    1. model bundle loads
    2. predictor accepts tensor [1, 3, 240]
    3. predictor returns clean PredictionResult object
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
APP_DIR = SCRIPT_DIR.parent
REPO_DIR = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from inference.predictor import predict_hr_from_tensor
from models.loader import load_model_bundle


def main() -> None:
    """
    Run predictor smoke test.
    """
    print("=" * 72)
    print("Predictor smoke test")
    print("=" * 72)
    print(f"App dir:  {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    bundle = load_model_bundle(device="cpu")
    input_config = bundle.model_spec["input"]
    x = torch.randn(
        1,
        int(input_config["in_channels"]),
        int(input_config["target_frames"]),
    )
    results = predict_hr_from_tensor(x, bundle)

    print("Input")
    print("-" * 72)
    print(f"x.shape: {tuple(x.shape)}")
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
    print("PASS: predictor smoke test is valid")
    print("=" * 72)


if __name__ == "__main__":
    main()