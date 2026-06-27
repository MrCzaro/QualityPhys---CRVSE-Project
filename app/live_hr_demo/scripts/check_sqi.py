"""
Spectral SQI smoke test for synthetic rPPG channels.

This script verifies that the spectral signal-quality pipeline can analyze
synthetic POS, CHROM, and GREEN rPPG-like channels.

It checks that:

1. Synthetic multichannel rPPG signals are generated.
2. Spectral SQI is computed for each expected channel.
3. A dominant cardiac-band frequency is detected.
4. The detected dominant BPM is close to the expected synthetic HR.

This is a signal-processing smoke test, not a model inference test.
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

from rppg.sqi import summarize_multichannel_sqi
from rppg.windowing import make_synthetic_rppg_channels

def validate_sqi_results(
    sqi_results: dict,
    expected_channels: tuple[str, ...],
    expected_hr_bpm: float,
    tolerance_bpm: float = 10.0,
) -> None:
    """
    Validate spectral SQI results for expected rPPG channels.

    Parameters
    ----------
    sqi_results:
        Mapping from channel name to spectral SQI result.

    expected_channels:
        Channel names expected in the SQI result.

    expected_hr_bpm:
        Synthetic HR used to generate the test signal.

    tolerance_bpm:
        Maximum allowed absolute difference between detected and expected BPM.

    Raises
    ------
    KeyError
        If an expected channel is missing.

    ValueError
        If a channel has invalid SQI values or an implausible dominant BPM.
    """
    missing_channels = set(expected_channels).difference(sqi_results.keys())

    if missing_channels:
        raise KeyError(f"Missing SQI channels: {sorted(missing_channels)}")

    for channel_name in expected_channels:
        result = sqi_results[channel_name]

        if result.dominant_freq_hz is None:
            raise ValueError(f"{channel_name}: dominant_freq_hz is None")

        if result.dominant_bpm is None:
            raise ValueError(f"{channel_name}: dominant_bpm is None")

        if result.sqi is None:
            raise ValueError(f"{channel_name}: sqi is None")

        bpm_error = abs(float(result.dominant_bpm) - float(expected_hr_bpm))

        if bpm_error > tolerance_bpm:
            raise ValueError(
                f"{channel_name}: dominant BPM {result.dominant_bpm:.1f} is "
                f"too far from expected {expected_hr_bpm:.1f} bpm "
                f"(error {bpm_error:.1f} > {tolerance_bpm:.1f})"
            )


def print_sqi_results(sqi_results: dict) -> None:
    """
    Print spectral SQI results for each channel.

    Parameters
    ----------
    sqi_results:
        Mapping from channel name to spectral SQI result.
    """
    print("SQI results")
    print("-" * 72)

    for channel_name, result in sqi_results.items():
        print(f"channel: {channel_name}")
        print(f"dominant_hz: {result.dominant_freq_hz:.3f}")
        print(f"dominant_bpm: {result.dominant_bpm:.1f}")
        print(f"sqi: {result.sqi:.3f}")
        print(f"status: {result.status}")
        print(f"reason: {result.reason}")
        print()


def main() -> None:
    """
    Run the spectral SQI smoke test.
    """
    print("=" * 72)
    print("Spectral SQI smoke test")
    print("=" * 72)
    print(f"App dir:  {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    synthetic_hr_bpm = 72.0
    fps = 30.0
    duration_seconds = 8.0
    expected_channels = ("pos", "chrom", "green")
    signals = make_synthetic_rppg_channels(
        hr_bpm=synthetic_hr_bpm,
        duration_seconds=duration_seconds,
        fps=fps,
        noise_std=0.05,
        seed=42,
    )
    sqi_results = summarize_multichannel_sqi(
        signals=signals,
        fps=fps,
        channel_names=expected_channels,
        low_hz=0.7,
        high_hz=3.5,
    )
    validate_sqi_results(
        sqi_results=sqi_results,
        expected_channels=expected_channels,
        expected_hr_bpm=synthetic_hr_bpm,
        tolerance_bpm=10.0,
    )

    print("Synthetic signal")
    print("-" * 72)
    print(f"synthetic_hr_bpm: {synthetic_hr_bpm}")
    print(f"expected_hz: {synthetic_hr_bpm / 60.0:.3f}")
    print(f"duration_seconds: {duration_seconds}")
    print(f"fps: {fps}")
    print(f"n_samples: {len(signals['pos'])}")
    print()

    print_sqi_results(sqi_results)

    print("=" * 72)
    print("PASS: spectral SQI smoke test ran successfully")
    print("=" * 72)

if __name__ == "__main__":
    main()