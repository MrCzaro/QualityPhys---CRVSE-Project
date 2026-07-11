"""
Runtime model availability helpers for the live HR demo.

The strict model loader remains in models.loader. This module adds demo-safe
startup behavior so the app can run with spectral-only functionality when the
experimental model is unavailable.
"""

from __future__ import annotations

import os
from typing import Any

from models.loader import ModelBundle, load_model_bundle


MODEL_DISABLE_ENV_VAR = "QUALITYPHYS_DISABLE_MODEL"


def load_model_bundle_for_demo(device: str = "cpu") -> tuple[ModelBundle | None, dict[str, Any]]:
    """
    Load the experimental model for the demo without making app startup depend on it.

    Returns
    -------
    tuple
        ``(model_bundle, model_status)``. ``model_bundle`` is None when the
        experimental model is unavailable.
    """

    disable_value = os.environ.get(MODEL_DISABLE_ENV_VAR, "").strip().lower()

    if disable_value in {"1", "true", "yes", "on"}:
        return None, {
            "available": False,
            "reason": "disabled_by_environment",
            "message": f"Experimental CRVSE model loading is disabled by {MODEL_DISABLE_ENV_VAR}.",
            "exception_type": None,
        }

    try:
        model_bundle = load_model_bundle(device=device)
    except Exception as exc:
        return None, {
            "available": False,
            "reason": "load_failed",
            "message": str(exc),
            "exception_type": type(exc).__name__,
        }

    return model_bundle, {
        "available": True,
        "reason": "loaded",
        "message": "Experimental CRVSE model bundle loaded.",
        "exception_type": None,
    }