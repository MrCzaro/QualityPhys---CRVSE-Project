"""
Serialization smoke test.

This verifies:
    1. integrated inference result can be converted to dict
    2. result can be JSON-encoded
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
APP_DIR = SCRIPT_DIR.parent
REPO_DIR = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from inference.serialization import prediction_result_to_dict
from inference.window_inference import predict_hr_from_rppg_window
from models.loader import load_model_bundle
from rppg.windowing import make_synthetic_rppg_channels


def main() -> None:
    """
    Run serialization smoke test.
    """
    print("=" * 72)
    print("Serialization smoke test")
    print("=" * 72)
    print(f"App dir: {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    bundle = load_model_bundle(device="cpu")
    fps = 30.0
    duration_seconds = float(bundle.model_spec["input"]["window_seconds"])
    signals = make_synthetic_rppg_channels(hr_bpm=72.0, duration_seconds=duration_seconds, fps=fps, noise_std=0.05, seed=42)
    result = predict_hr_from_rppg_window(signals=signals, fps=fps, bundle=bundle)
    result_dict = prediction_result_to_dict(result)
    result_json = json.dumps(result_dict, indent=2)

    print("Serialized result")
    print("-" * 72)
    print(result_json)
    print()
    print("=" * 72)
    print("PASS: serialization smoke test ran successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()