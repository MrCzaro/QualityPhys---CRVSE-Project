"""
Serialization smoke test for prediction results.

This script verifies that an integrated window-level prediction result can be
converted into a plain Python dictionary and encoded as JSON.

It checks the response shape used by app routes before returning prediction
results through a JSON API.

This is not an accuracy test. It only validates serialization compatibility.
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


def build_serialized_prediction_result() -> dict:
    """
    Run one synthetic prediction and serialize the result.

    Returns
    -------
    dict
        JSON-compatible prediction result dictionary.
    """
    bundle = load_model_bundle(device="cpu")
    fps = 30.0
    duration_seconds = float(bundle.model_spec["input"]["window_seconds"])
    signals = make_synthetic_rppg_channels(hr_bpm=72.0, duration_seconds=duration_seconds, fps=fps, noise_std=0.05, seed=42)
    result = predict_hr_from_rppg_window(signals=signals, fps=fps, bundle=bundle)
    result_dict = prediction_result_to_dict(result)

    if not isinstance(result_dict, dict):
        raise TypeError(f"Expected serialized prediction to be dict, got {type(result_dict)}")

    return result_dict


def validate_serialized_result(result_dict: dict) -> str:
    """
    Validate that a prediction dictionary can be JSON-encoded.

    Parameters
    ----------
    result_dict:
        Serialized prediction result dictionary.

    Returns
    -------
    str
        Pretty-printed JSON string.
    """
    required_keys = {
        "task",
        "model_hr_bpm",
        "unit",
        "model_name",
        "window_seconds",
        "target_frames",
        "channel_names",
        "quality",
        "extra",
    }

    missing_keys = required_keys.difference(result_dict.keys())

    if missing_keys:
        raise KeyError(f"Serialized prediction is missing keys: {sorted(missing_keys)}")

    return json.dumps(result_dict, indent=2)


def main() -> None:
    """
    Run the prediction serialization smoke test.
    """
    print("=" * 72)
    print("Serialization smoke test")
    print("=" * 72)
    print(f"App dir: {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    result_dict = build_serialized_prediction_result()
    result_json = validate_serialized_result(result_dict)

    print("Serialized result")
    print("-" * 72)
    print(result_json)
    print()
    print("=" * 72)
    print("PASS: serialization smoke test ran successfully")
    print("=" * 72)

if __name__ == "__main__":
    main()