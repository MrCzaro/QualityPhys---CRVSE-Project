"""
API route registration for the live HR demo.

This module keeps backend API endpoints separate from the FastHTML app shell.
The routes support frame diagnostics, face/ROI diagnostics, ROI signal analysis,
and experimental live model prediction from browser-collected ROI RGB samples.
"""

from __future__ import annotations
from starlette.requests import Request
from starlette.responses import JSONResponse
from backend.face_debug import summarize_face_from_data_url_frame
from backend.frame_debug import summarize_data_url_frame
from backend.live_prediction import make_json_safe_for_api, make_live_roi_model_prediction_payload
from rppg.live_methods import analyze_roi_series_payload


def register_api_routes(rt, model_bundle) -> None:
    """
    Register backend API routes for the live HR demo.

    Parameters
    ----------
    rt:
        FastHTML route decorator returned by ``fast_app``.

    model_bundle:
        Loaded CRVSE model bundle used by the live prediction endpoint.
    """

    @rt("/api/debug-frame", methods=["POST"])
    async def debug_frame_api(request: Request):
        """
        Decode one browser-submitted frame and return basic image diagnostics.

        The frame is processed in memory only and is not stored.
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
                    "stage": "debug_frame_api",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
                status_code=400,
            )

    @rt("/api/debug-face", methods=["POST"])
    async def debug_face_api(request: Request):
        """
        Run face landmark and ROI diagnostics on one browser-submitted frame.

        The frame is processed in memory only and is not stored.
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

        This endpoint receives numeric ROI summaries only, not raw frames.
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
        Run experimental live model prediction from ROI RGB samples.

        This endpoint receives numeric ROI summaries only, not raw frames.
        """

        try:
            payload = await request.json()

            result = make_live_roi_model_prediction_payload(
                payload=payload,
                model_bundle=model_bundle,
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