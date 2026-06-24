"""
Spectral SQI smoke test.

This verifies:
    1. synthetic rPPG channels are generated
    2. power spectrum is computed
    3. dominant HR frequency is detected
    4. SQI status is returned for POS / CHROM / GREEN
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


def main() -> None:
    """
    Run spectral SQI smoke test.
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
        channel_names=("pos", "chrom", "green"),
        low_hz=0.7,
        high_hz=3.5,
    )

    print("Synthetic signal")
    print("-" * 72)
    print(f"synthetic_hr_bpm: {synthetic_hr_bpm}")
    print(f"expected_hz: {synthetic_hr_bpm / 60.0:.3f}")
    print(f"duration_seconds: {duration_seconds}")
    print(f"fps: {fps}")
    print(f"n_samples: {len(signals['pos'])}")
    print()
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

    print("=" * 72)
    print("PASS: spectral SQI smoke test ran successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()