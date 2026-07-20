"""
Run all live HR demo smoke tests.

This script executes the local smoke-test suite used to check that the app's
model contract, synthetic signal path, SQI logic, quality gating, serialization,
and integrated inference path still work after refactoring.

Run from the live demo directory:

    python scripts/run_smoke_tests.py

or from the scripts directory:

    python run_smoke_tests.py
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
APP_DIR = SCRIPT_DIR.parent
REPO_DIR = APP_DIR.parents[1]


SMOKE_TESTS = [
    "check_model_contract.py",
    "check_model_runtime_fallback.py",
    "check_model_prediction_fallback_payload.py",
    "check_roi_sample_api_contract.py",
    "check_synthetic_window_prediction.py",
    "check_sqi.py",
    "check_window_quality.py",
    "check_integrated_window_inference.py",
    "check_serialization.py",
    "check_synthetic_inference.py",
]


def run_one_smoke_test(script_name: str) -> tuple[bool, float]:
    """
    Run one smoke-test script.

    Parameters
    ----------
    script_name:
        File name of the smoke-test script located in the scripts directory.

    Returns
    -------
    tuple[bool, float]
        ``(passed, elapsed_seconds)``.
    """
    script_path = SCRIPT_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Smoke-test script not found: {script_path}")

    start_time = time.perf_counter()
    completed_process = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(SCRIPT_DIR),
        check=False,
    )

    elapsed_seconds = time.perf_counter() - start_time
    passed = completed_process.returncode == 0

    return passed, elapsed_seconds


def print_runner_header() -> None:
    """
    Print smoke-test runner header.
    """
    print("=" * 72)
    print("Live HR demo smoke-test runner")
    print("=" * 72)
    print(f"App dir: {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print(f"Scripts dir: {SCRIPT_DIR}")
    print(f"Python: {sys.executable}")
    print()


def main() -> None:
    """
    Run the live HR demo smoke-test suite.
    """

    print_runner_header()

    results = []
    total_start_time = time.perf_counter()

    for index, script_name in enumerate(SMOKE_TESTS, start=1):
        print("=" * 72)
        print(f"[{index}/{len(SMOKE_TESTS)}] Running {script_name}")
        print("=" * 72)

        try:
            passed, elapsed_seconds = run_one_smoke_test(script_name)
        except Exception as exc:
            passed = False
            elapsed_seconds = 0.0
            print(f"ERROR: {script_name} failed before execution: {exc}")

        results.append(
            {
                "script_name": script_name,
                "passed": passed,
                "elapsed_seconds": elapsed_seconds,
            }
        )

        status = "PASS" if passed else "FAIL"
        print()
        print(f"{status}: {script_name} ({elapsed_seconds:.2f} s)")
        print()

        if not passed:
            break

    total_elapsed_seconds = time.perf_counter() - total_start_time

    print("=" * 72)
    print("Smoke-test summary")
    print("=" * 72)

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(
            f"{status:4}  "
            f"{result['script_name']:<42} "
            f"{result['elapsed_seconds']:.2f} s"
        )

    print("-" * 72)
    print(f"Total elapsed: {total_elapsed_seconds:.2f} s")

    all_passed = all(result["passed"] for result in results)
    all_tests_ran = len(results) == len(SMOKE_TESTS)

    if all_passed and all_tests_ran:
        print("=" * 72)
        print("PASS: all live HR demo smoke tests passed")
        print("=" * 72)
        return

    print("=" * 72)
    print("FAIL: smoke-test suite stopped before all tests passed")
    print("=" * 72)

    raise SystemExit(1)


if __name__ == "__main__":
    main()