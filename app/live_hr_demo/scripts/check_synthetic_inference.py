"""
Smoke-test synthetic rPPG inference outside the FastHTML app.

This script verifies that the model bundle can be loaded, synthetic
POS/CHROM/GREEN signals can be generated, and the prediction pipeline can
return a JSON-safe heart-rate result.

Run from the live demo directory:

    python scripts/check_synthetic_inference.py

or from the repository root:

    python app/live_hr_demo/scripts/check_synthetic_inference.py
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
LIVE_DEMO_DIR = SCRIPT_PATH.parents[1]

if str(LIVE_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(LIVE_DEMO_DIR))

from inference.serialization import prediction_result_to_dict
from inference.window_inference import predict_hr_from_rppg_window
from models.loader import load_model_bundle
from rppg.sqi import estimate_spectral_sqi
from rppg.windowing import make_synthetic_rppg_channels


def run_synthetic_inference_smoke_test(
    synthetic_hr_bpm: float = 72.0,
    noise_std: float = 0.05,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """
    Run one synthetic rPPG inference smoke test.

    Parameters
    ----------
    synthetic_hr_bpm:
        Heart rate used to generate the synthetic rPPG-like signal.

    noise_std:
        Standard deviation of synthetic noise added to the generated channels.

    seed:
        Random seed for reproducible synthetic signal generation.

    device:
        Torch device used when loading the model bundle.

    Returns
    -------
    dict
        JSON-safe prediction result with additional smoke-test metadata.
    """
    model_bundle = load_model_bundle(device=device)
    input_spec = model_bundle.model_spec["input"]
    preprocessing_spec = model_bundle.model_spec["preprocessing"]
    window_seconds = float(input_spec["window_seconds"])
    target_frames = int(input_spec["target_frames"])
    fps = float(target_frames) / float(window_seconds)
    signals = make_synthetic_rppg_channels(
        hr_bpm=synthetic_hr_bpm,
        duration_seconds=window_seconds,
        fps=fps,
        noise_std=noise_std,
        seed=seed,
    )
    prediction = predict_hr_from_rppg_window(
        signals=signals,
        fps=fps,
        bundle=model_bundle,
    )
    prediction_dict = prediction_result_to_dict(prediction)
    spectrum = estimate_spectral_sqi(
        signal=signals["pos"],
        fps=fps,
        low_hz=float(preprocessing_spec["bandpass_low_hz"]),
        high_hz=float(preprocessing_spec["bandpass_high_hz"]),
    )

    return {
        "status": "ok",
        "synthetic_hr_bpm": float(synthetic_hr_bpm),
        "noise_std": float(noise_std),
        "seed": int(seed),
        "model_name": model_bundle.model_spec["name"],
        "window_seconds": float(window_seconds),
        "target_frames": int(target_frames),
        "fps": float(fps),
        "prediction": prediction_dict,
        "spectral_check": {
            "dominant_bpm": float(spectrum.dominant_bpm),
            "sqi": float(spectrum.sqi),
            "status": str(spectrum.status),
        },
    }


def print_smoke_test_summary(result: dict) -> None:
    """
    Print a compact synthetic inference smoke-test summary.

    Parameters
    ----------
    result:
        Result returned by ``run_synthetic_inference_smoke_test``.
    """
    prediction = result["prediction"]
    spectral_check = result["spectral_check"]
    model_hr = prediction.get("model_hr_bpm")
    spectral_hr = prediction.get("extra", {}).get("spectral_hr_bpm")
    unit = prediction.get("unit", "bpm")

    print("Synthetic rPPG inference smoke test")
    print("=" * 72)
    print(f"Status: {result['status']}")
    print(f"Model: {result['model_name']}")
    print(f"Synthetic HR: {result['synthetic_hr_bpm']:.1f} bpm")
    print(f"Noise std: {result['noise_std']:.3f}")
    print(f"Window: {result['window_seconds']:.1f} s")
    print(f"Target frames: {result['target_frames']}")
    print(f"FPS: {result['fps']:.2f}")
    print()
    print(f"Model HR: {model_hr:.1f} {unit}" if model_hr is not None else "Model HR:            unavailable")
    print(f"Prediction spectral: {spectral_hr:.1f} {unit}" if spectral_hr is not None else "Prediction spectral: unavailable")
    print(f"Direct POS spectral: {spectral_check['dominant_bpm']:.1f} bpm")
    print(f"Direct POS SQI: {spectral_check['sqi']:.3f} / {spectral_check['status']}")
    print()
    print("Quality")
    print("-" * 72)
    print(f"{prediction['quality']['status']} / {prediction['quality']['confidence']}")

    for reason in prediction["quality"]["reasons"]:
        print(f"- {reason}")


def main() -> None:
    """
    Run the synthetic inference smoke test from the command line.
    """
    result = run_synthetic_inference_smoke_test()
    print_smoke_test_summary(result)


if __name__ == "__main__":
    main()