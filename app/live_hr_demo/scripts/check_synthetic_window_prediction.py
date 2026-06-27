"""
Synthetic model-window prediction smoke test.

This script verifies the low-level tensor prediction path used by the live HR
demo model wrapper.

It checks that:

1. Synthetic POS, CHROM, and GREEN rPPG-like channels can be generated.
2. The channels can be converted into the model input tensor shape ``(1, 3, 240)``.
3. The tensor predictor accepts the input.
4. A prediction result is returned.

This test does not run spectral SQI or quality gating. It validates the
model-ready tensor path only.
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
from rppg.windowing import WindowConfig, make_model_window_from_channels, make_synthetic_rppg_channels



def build_synthetic_model_window(bundle, synthetic_hr_bpm: float = 72.0):
    """
    Build one synthetic model-ready rPPG window.

    Parameters
    ----------
    bundle:
        Loaded model bundle.

    synthetic_hr_bpm:
        Heart rate used to generate synthetic rPPG-like channels.

    Returns
    -------
    tuple
        ``(signals, x, window_config)`` where ``x`` is the model input tensor.
    """
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

    return signals, x, window_config


def validate_model_window_prediction(x, results, expected_shape: tuple[int, int, int] = (1, 3, 240)) -> None:
    """
    Validate model-window tensor prediction outputs.

    Parameters
    ----------
    x:
        Model input tensor.

    results:
        Prediction results returned by ``predict_hr_from_tensor``.

    expected_shape:
        Expected model input tensor shape.

    Raises
    ------
    ValueError
        If tensor shape or prediction output is invalid.
    """
    if tuple(x.shape) != expected_shape:
        raise ValueError(f"Expected model input shape {expected_shape}, got {tuple(x.shape)}")
    if len(results) == 0:
        raise ValueError("Expected at least one prediction result, got zero.")

    for result in results:
        if result.task != "heart_rate":
            raise ValueError(f"Expected task 'heart_rate', got {result.task}")
        if result.value is None:
            raise ValueError("Expected prediction value, got None.")
        if result.unit != "bpm":
            raise ValueError(f"Expected unit 'bpm', got {result.unit}")


def print_prediction_results(results) -> None:
    """
    Print tensor prediction results.

    Parameters
    ----------
    results:
        Prediction results returned by ``predict_hr_from_tensor``.
    """
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


def main() -> None:
    """
    Run the synthetic model-window prediction smoke test.
    """
    print("=" * 72)
    print("Synthetic rPPG window prediction smoke test")
    print("=" * 72)
    print(f"App dir:  {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    synthetic_hr_bpm = 72.0
    bundle = load_model_bundle(device="cpu")
    signals, x, window_config = build_synthetic_model_window(bundle=bundle, synthetic_hr_bpm=synthetic_hr_bpm)
    results = predict_hr_from_tensor(x=x, bundle=bundle)
    validate_model_window_prediction(x=x, results=results, expected_shape=(1, 3, 240))

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

    print_prediction_results(results)

    print("=" * 72)
    print("PASS: synthetic window prediction smoke test ran successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()