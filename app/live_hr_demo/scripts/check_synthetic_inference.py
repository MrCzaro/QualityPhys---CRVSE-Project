"""
Synthetic rPPG inference smoke test.

This script runs a compact positive-path check of the live-demo inference stack
outside the FastHTML app.

It verifies that:

1. The CRVSE model bundle loads.
2. Synthetic POS/CHROM/GREEN rPPG-like channels are generated.
3. Window-level prediction runs successfully.
4. The prediction result can be converted into a JSON-safe dictionary.
5. A direct POS spectral check detects a plausible heart-rate peak.

This is a deployment smoke test, not an accuracy evaluation.
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

from inference.serialization import prediction_result_to_dict
from inference.window_inference import predict_hr_from_rppg_window
from models.loader import load_model_bundle
from rppg.sqi import estimate_spectral_sqi
from rppg.windowing import make_synthetic_rppg_channels


def run_synthetic_inference_smoke_test(synthetic_hr_bpm: float = 72.0, noise_std: float = 0.05, seed: int = 42, device: str = "cpu") -> dict:
    """
    Run one synthetic rPPG inference smoke test.

    Parameters
    ----------
    synthetic_hr_bpm:
        Heart rate used to generate the synthetic rPPG-like signal.

    noise_std:
        Standard deviation of synthetic noise added to each generated channel.

    seed:
        Random seed for reproducible synthetic signal generation.

    device:
        Device used when loading the model bundle.

    Returns
    -------
    dict
        Prediction result and metadata for smoke-test reporting.
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


def validate_smoke_test_result(result: dict, bpm_tolerance: float = 10.0) -> None:
    """
    Validate the synthetic inference smoke-test result.

    Parameters
    ----------
    result:
        Result returned by ``run_synthetic_inference_smoke_test``.

    bpm_tolerance:
        Maximum allowed difference between synthetic HR and direct spectral HR.

    Raises
    ------
    ValueError
        If prediction or spectral checks return implausible values.
    """
    prediction = result["prediction"]
    spectral_check = result["spectral_check"]

    if result.get("status") != "ok":
        raise ValueError(f"Expected status 'ok', got {result.get('status')}")

    model_hr = prediction.get("model_hr_bpm")
    spectral_hr = prediction.get("extra", {}).get("spectral_hr_bpm")
    direct_spectral_hr = spectral_check.get("dominant_bpm")

    if model_hr is None:
        raise ValueError("Model HR is None for clean synthetic signal.")

    if spectral_hr is None:
        raise ValueError("Prediction spectral HR is None for clean synthetic signal.")

    if direct_spectral_hr is None:
        raise ValueError("Direct POS spectral HR is None.")

    direct_error = abs(float(direct_spectral_hr) - float(result["synthetic_hr_bpm"]))

    if direct_error > bpm_tolerance:
        raise ValueError(
            f"Direct spectral HR {direct_spectral_hr:.1f} bpm is too far from "
            f"synthetic HR {result['synthetic_hr_bpm']:.1f} bpm "
            f"(error {direct_error:.1f} > {bpm_tolerance:.1f})"
        )

    quality = prediction.get("quality", {})

    if quality.get("status") != "accepted":
        raise ValueError(f"Expected accepted quality for clean synthetic signal, got {quality.get('status')}")


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
    print(f"App dir: {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()
    print(f"Status: {result['status']}")
    print(f"Model: {result['model_name']}")
    print(f"Synthetic HR: {result['synthetic_hr_bpm']:.1f} bpm")
    print(f"Noise std: {result['noise_std']:.3f}")
    print(f"Window: {result['window_seconds']:.1f} s")
    print(f"Target frames: {result['target_frames']}")
    print(f"FPS: {result['fps']:.2f}")
    print()

    if model_hr is None:
        print("Model HR: unavailable")
    else:
        print(f"Model HR: {model_hr:.1f} {unit}")

    if spectral_hr is None:
        print("Prediction spectral: unavailable")
    else:
        print(f"Prediction spectral: {spectral_hr:.1f} {unit}")

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
    Run the synthetic rPPG inference smoke test.
    """
    result = run_synthetic_inference_smoke_test()
    validate_smoke_test_result(result=result, bpm_tolerance=10.0,)
    print_smoke_test_summary(result)

    print()
    print("=" * 72)
    print("PASS: synthetic rPPG inference smoke test ran successfully")
    print("=" * 72)

if __name__ == "__main__":
    main()