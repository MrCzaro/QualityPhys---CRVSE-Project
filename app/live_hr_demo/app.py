"""
FastHTML app shell for the QualityPhys live HR demo.

This module wires together the live camera-based rPPG demo page and its backend
API routes. The main app displays browser camera controls, ROI sampling,
live pulse-wave visualization, spectral HR estimation, and an experimental
CRVSE model HR comparison.

The app does not store camera frames. Debug frame and face routes process
submitted frames in memory, while live HR prediction routes operate on numeric
ROI RGB summaries collected in the browser.

Synthetic inference checks are kept outside the app in
``scripts/check_synthetic_inference.py``.
"""

from __future__ import annotations

from pathlib import Path
import sys

from fasthtml.common import *
from monsterui.all import *


APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from backend.api_routes import register_api_routes
from backend.ui_routes import register_ui_routes
from models.runtime import load_model_bundle_for_demo
from ui.live_demo_script import live_demo_script
from ui.live_demo_components import camera_preview_card


app, rt = fast_app(
    title="QualityPhys Live HR Demo",
    hdrs=(
        *Theme.blue.headers(),
        *Favicon(
            "/static/favicon.ico",
            "/static/favicon.ico",
        ),
    ),
    bodykw={
        "class": "bg-slate-100 text-slate-950",
    },
)

MODEL_BUNDLE, MODEL_STATUS = load_model_bundle_for_demo(device="cpu")

if not MODEL_STATUS["available"]:
    print(
        "WARNING: Experimental CRVSE model unavailable; "
        "serving live HR demo in limited spectral-only mode. "
        f"Reason: {MODEL_STATUS['message']}",
        file=sys.stderr,
    )
register_api_routes(
    rt=rt,
    model_bundle=MODEL_BUNDLE,
    model_status=MODEL_STATUS,
)
register_ui_routes(rt=rt)

@rt("/")
def index() -> FT:
    """
    Render the main live HR demo page.

    Returns
    -------
    FT
        FastHTML document containing the camera-based HR demo UI.

    Notes
    -----
    The page is a research demo, not a medical device. Camera access and
    frontend interaction are handled by ``live_demo_script()``.
    """

    return (
        Title("QualityPhys Live HR Demo"),
        Main(
            Div(
                H1(
                    "QualityPhys Live HR Demo",
                    cls="text-3xl font-bold text-slate-950",
                ),
                P(
                    "Research demo for camera-based heart-rate estimation from rPPG. "
                    "The primary HR estimate is based on spectral consensus from live "
                    "ROI color signals. The CRVSE model estimate is shown as an "
                    "experimental comparison.",
                    cls="mt-2 max-w-4xl text-slate-600",
                ),
                P(
                    "Not a medical device. Not for diagnosis or treatment decisions.",
                    cls="mt-2 font-medium text-red-700",
                ),
                cls="mb-8",
            ),
            camera_preview_card(),
            cls="mx-auto max-w-6xl px-6 py-10",
        ),
        live_demo_script(),
    )


serve()
