"""
Model runtime fallback smoke test.

This verifies that the demo-safe model loader can enter limited mode without
touching checkpoint files or breaking app startup.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
APP_DIR = SCRIPT_DIR.parent
REPO_DIR = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from models.runtime import MODEL_DISABLE_ENV_VAR, load_model_bundle_for_demo


def run_disabled_model_check() -> tuple[object | None, dict]:
    previous_value = os.environ.get(MODEL_DISABLE_ENV_VAR)
    os.environ[MODEL_DISABLE_ENV_VAR] = "1"

    try:
        return load_model_bundle_for_demo(device="cpu")
    finally:
        if previous_value is None:
            os.environ.pop(MODEL_DISABLE_ENV_VAR, None)
        else:
            os.environ[MODEL_DISABLE_ENV_VAR] = previous_value


def validate_disabled_model_result(model_bundle: object | None, model_status: dict) -> None:
    if model_bundle is not None:
        raise ValueError("Expected model bundle to be None when model loading is disabled.")

    if model_status.get("available") is not False:
        raise ValueError(f"Expected available=False, got {model_status.get('available')}")

    if model_status.get("reason") != "disabled_by_environment":
        raise ValueError(f"Unexpected fallback reason: {model_status.get('reason')}")

    if model_status.get("exception_type") is not None:
        raise ValueError(f"Expected no exception type, got {model_status.get('exception_type')}")

    message = str(model_status.get("message", ""))

    if MODEL_DISABLE_ENV_VAR not in message:
        raise ValueError(f"Expected message to mention {MODEL_DISABLE_ENV_VAR!r}: {message}")


def main() -> None:
    print("=" * 72)
    print("Model runtime fallback smoke test")
    print("=" * 72)
    print(f"App dir: {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    model_bundle, model_status = run_disabled_model_check()
    validate_disabled_model_result(model_bundle, model_status)

    print("Fallback status")
    print("-" * 72)
    print(model_status)
    print()
    print("=" * 72)
    print("PASS: model runtime fallback smoke test ran successfully")
    print("=" * 72)


if __name__ == "__main__":
    main()