"""
Model prediction fallback payload smoke test.

This verifies that the backend can build a graceful model-unavailable response
from numeric ROI samples without starting the browser or FastHTML server.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
APP_DIR = SCRIPT_DIR.parent
REPO_DIR = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from backend.api_routes import build_model_unavailable_prediction_payload


def make_synthetic_roi_samples(
    hr_bpm: float = 72.0,
    duration_seconds: float = 8.0,
    fps: float = 30.0,
) -> list[dict]:
    """
    Build synthetic browser-style ROI RGB samples.

    Returns numeric ROI summaries only, not frames.
    """

    roi_names = [
        "forehead",
        "image_left_cheek",
        "image_right_cheek",
    ]
    sample_count = int(duration_seconds * fps)
    frequency_hz = float(hr_bpm) / 60.0

    samples = []

    for index in range(sample_count):
        t_s = float(index) / float(fps)
        pulse = math.sin(2.0 * math.pi * frequency_hz * t_s)

        rois = {}

        for roi_index, roi_name in enumerate(roi_names):
            offset = float(roi_index) * 0.25
            rois[roi_name] = {
                "r": 100.0 + offset + 1.8 * pulse,
                "g": 105.0 + offset + 3.0 * pulse,
                "b": 95.0 + offset + 1.2 * pulse,
            }

        samples.append(
            {
                "t_s": t_s,
                "rois": rois,
            }
        )

    return samples


def validate_fallback_payload(payload: dict) -> str:
    """
    Validate the model-unavailable response shape and JSON compatibility.
    """

    if payload.get("status") != "model_unavailable":
        raise ValueError(f"Expected status model_unavailable, got {payload.get('status')}")

    if payload.get("model_available") is not False:
        raise ValueError(f"Expected model_available=False, got {payload.get('model_available')}")

    if payload.get("model_prediction") is not None:
        raise ValueError("Expected model_prediction to be None for unavailable model.")

    if payload.get("model_input") is not None:
        raise ValueError("Expected model_input to be None for unavailable model.")

    if payload.get("classical_analysis_status") != "ok":
        raise ValueError(
            "Expected fallback classical spectral analysis to be ok, got "
            f"{payload.get('classical_analysis_status')}"
        )

    summary = payload.get("classical_spectral_summary")

    if not isinstance(summary, dict):
        raise TypeError(f"Expected classical_spectral_summary dict, got {type(summary)}")

    missing_signals = {"green", "pos", "chrom"}.difference(summary.keys())

    if missing_signals:
        raise KeyError(f"Missing spectral fallback summaries: {sorted(missing_signals)}")

    for signal_name in ["green", "pos", "chrom"]:
        spectral = summary[signal_name]

        if not isinstance(spectral, dict):
            raise TypeError(f"Expected {signal_name} spectral summary dict, got {type(spectral)}")

        if spectral.get("dominant_bpm") is None:
            raise ValueError(f"Expected {signal_name} dominant_bpm to be available.")

    return json.dumps(payload, indent=2)


def main() -> None:
    print("=" * 72)
    print("Model prediction fallback payload smoke test")
    print("=" * 72)
    print(f"App dir: {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    samples = make_synthetic_roi_samples()
    payload = build_model_unavailable_prediction_payload(
        payload={"samples": samples},
        model_status={
            "available": False,
            "reason": "disabled_by_environment",
            "message": "Synthetic fallback smoke-test model status.",
            "exception_type": None,
        },
    )
    payload_json = validate_fallback_payload(payload)

    print("Fallback payload")
    print("-" * 72)
    print(payload_json)
    print()
    print("=" * 72)
    print("PASS: model prediction fallback payload smoke test ran successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()