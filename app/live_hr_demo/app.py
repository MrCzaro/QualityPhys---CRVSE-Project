"""
Live HR demo app shell.

Current purpose:
    Show a synthetic backend inference result in the browser.
    Show synthetic rPPG waveform and power spectrum plots.
    Expose the same synthetic inference result through a real JSON endpoint.
    Demonstrate browser → backend → UI update with a refresh button.

What this proves:
    FastHTML page route works.
    Backend JSON endpoint works.
    Model bundle loads.
    Synthetic rPPG window runs through the full backend inference core.
    Serialized result can be displayed as HTML and returned as JSON.
    Browser JavaScript can fetch backend inference data and update the page.

What this does NOT do yet:
    No webcam.
    No real face video.
    No POS/CHROM/GREEN extraction from frames.
    No frame storage.
"""

from __future__ import annotations
from pathlib import Path
import sys
from fasthtml.common import *
from starlette.requests import Request
from starlette.responses import JSONResponse

APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from backend.face_debug import summarize_face_from_data_url_frame
from backend.frame_debug import summarize_data_url_frame
from backend.live_prediction import make_json_safe_for_api, make_live_roi_model_prediction_payload
from models.loader import load_model_bundle
from rppg.live_methods import analyze_roi_series_payload
from ui.live_demo_script import live_demo_script
from ui.live_demo_components import camera_preview_card

app, rt = fast_app(title="QualityPhys Live HR Demo")

MODEL_BUNDLE = load_model_bundle(device="cpu")



### Route handlers
@rt("/")
def index() -> FT:
    """
    Render the main live HR demo page.

    Returns
    -------
    FT
        FastHTML document containing the live camera-based HR demo.

    Notes
    -----
    Synthetic demo components and endpoints remain available in code for
    development checks, but they are not shown on the main page.
    """

    return Html(
        Head(
            Title("QualityPhys Live HR Demo"),
            Script(src="https://cdn.tailwindcss.com"),
        ),
        Body(
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
            cls="bg-slate-100",
        ),
    )




@rt("/api/debug-frame", methods=["POST"])
async def debug_frame_api(request: Request):
    """
    Receive one browser-captured frame and return basic decode diagnostics.

    Expected request JSON
    ---------------------
    {
        "image_data_url": "data:image/jpeg;base64,..."
    }

    Returns
    -------
    JSONResponse
        Image dimensions, channel count, dtype, RGB statistics, and decode metadata.

    Privacy:
        The frame is decoded in memory only.
        The frame is not stored on disk.
        No model inference is run here.

    Physiology:
        No physiology is estimated here.

    Signal:
        No rPPG signal is extracted here. This only confirms valid pixel transport.

    Limitation:
        This route processes one manually submitted frame, not a live stream.
    """

    try:
        payload = await request.json()
        image_data_url = payload.get("image_data_url")

        if image_data_url is None:
            return JSONResponse(
                {
                    "status": "error",
                    "message": "Missing required field: image_data_url",
                },
                status_code=400,
            )

        result = summarize_data_url_frame(image_data_url)

        return JSONResponse(result)

    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "message": str(exc),
            },
            status_code=400,
        )
    
@rt("/api/debug-face", methods=["POST"])
async def debug_face_api(request: Request):
    """
    Receive one browser-captured frame and run backend face landmark diagnostics.

    Expected request JSON
    ---------------------
    {
        "image_data_url": "data:image/jpeg;base64,..."
    }

    Returns
    -------
    JSONResponse
        Frame decode summary and face detection diagnostics.

    Privacy:
        The frame is decoded and processed in memory only.
        The frame is not stored on disk.
        No model inference is run here.

    Physiology:
        No physiology is estimated here.

    Signal:
        No rPPG signal is extracted here. This only confirms valid face geometry
        detection from real camera pixels.

    Limitation:
        This route processes one manually submitted frame, not a live stream.

    Debug behavior:
        If an error occurs, return the exception type and message in JSON so the
        browser can show the actual cause instead of only "HTTP 400".
    """

    try:
        payload = await request.json()
        image_data_url = payload.get("image_data_url")

        if image_data_url is None:
            return JSONResponse(
                {
                    "status": "error",
                    "stage": "request_validation",
                    "message": "Missing required field: image_data_url",
                },
                status_code=400,
            )

        result = summarize_face_from_data_url_frame(image_data_url)

        return JSONResponse(result)

    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "stage": "face_debug_api",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            status_code=400,
        )

@rt("/api/analyze-roi-series", methods=["POST"])
async def analyze_roi_series_api(request: Request):
    """
    Analyze browser-collected ROI RGB samples into candidate rPPG signals.

    Expected request JSON
    ---------------------
    {
        "samples": [
            {
                "t_s": 0.0,
                "rois": {
                    "forehead": {"r": ..., "g": ..., "b": ...},
                    "image_left_cheek": {"r": ..., "g": ..., "b": ...},
                    "image_right_cheek": {"r": ..., "g": ..., "b": ...}
                }
            }
        ]
    }

    Returns
    -------
    JSONResponse
        GREEN / POS / CHROM candidate signals and spectral summaries.

    Privacy:
        This route receives numeric ROI RGB summaries only.
        It does not receive or store image frames.

    Physiology:
        Candidate rPPG signals may contain pulse-related color variation.

    Signal:
        This route converts ROI RGB time series into GREEN, POS, and CHROM-style
        candidate signals.

    Limitation:
        This is not model inference and not a medical measurement.
    """

    try:
        payload = await request.json()
        result = analyze_roi_series_payload(payload)

        status_code = 200 if result.get("status") == "ok" else 400

        return JSONResponse(
            result,
            status_code=status_code,
        )

    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "stage": "analyze_roi_series_api",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            status_code=400,
        )   

@rt("/api/predict-live-roi-series", methods=["POST"])
async def predict_live_roi_series_api(request: Request):
    """
    Run experimental live model prediction from browser-collected ROI RGB samples.

    Expected request JSON
    ---------------------
    {
        "samples": [...]
    }

    Returns
    -------
    JSONResponse
        Model HR prediction plus classical spectral summaries.

    Notes
    -----
    This route receives numeric ROI RGB summaries only. Raw image frames are not
    sent to this endpoint.
    """

    try:
        payload = await request.json()

        result = make_live_roi_model_prediction_payload(
            payload=payload,
            model_bundle=MODEL_BUNDLE,
        )

        result = make_json_safe_for_api(result)

        status_code = 200 if result.get("status") == "ok" else 400

        return JSONResponse(
            result,
            status_code=status_code,
        )

    except Exception as exc:
        return JSONResponse(
            {
                "status": "error",
                "stage": "predict_live_roi_series_api",
                "exception_type": type(exc).__name__,
                "message": str(exc),
            },
            status_code=400,
        )

serve()