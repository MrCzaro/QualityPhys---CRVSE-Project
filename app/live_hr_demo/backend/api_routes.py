"""
API route registration for the live HR demo.

This module keeps backend API endpoints separate from the FastHTML app shell.
The routes support frame diagnostics, face/ROI diagnostics, ROI signal analysis,
and experimental live model prediction from browser-collected ROI RGB samples.
"""
from __future__ import annotations
from starlette.requests import Request
from starlette.responses import JSONResponse
from backend.face_debug import summarize_face_from_data_url_frame, summarize_live_roi_sample_from_data_url_frame
from backend.frame_debug import summarize_data_url_frame
from backend.live_prediction import make_json_safe_for_api, make_live_roi_model_prediction_payload
from rppg.live_methods import analyze_roi_series_payload


def _classical_spectral_summary_from_payload(payload: dict) -> dict:
    """
    Compute the spectral summary that remains available without the model.
    """

    try:
        analysis = analyze_roi_series_payload(payload)
        analysis = make_json_safe_for_api(analysis)
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
            "summary": None,
        }

    if analysis.get("status") != "ok":
        return {
            "status": analysis.get("status", "error"),
            "message": analysis.get("message", "Classical spectral analysis was not available."),
            "summary": None,
        }

    signals = analysis.get("signals", {})

    return {
        "status": "ok",
        "message": "Classical spectral analysis completed.",
        "summary": {
            "green": signals.get("green", {}).get("spectral"),
            "pos": signals.get("pos", {}).get("spectral"),
            "chrom": signals.get("chrom", {}).get("spectral"),
        },
    }


def build_model_unavailable_prediction_payload(
    payload: dict,
    model_status: dict | None = None,
) -> dict:
    """
    Build a graceful API response when experimental model prediction is unavailable.

    The input payload contains numeric ROI summaries only, not raw frames.
    """

    classical_result = _classical_spectral_summary_from_payload(payload)

    return make_json_safe_for_api(
        {
            "status": "model_unavailable",
            "message": (
                "Experimental model prediction is unavailable due to a model "
                "loading issue. Spectral rPPG analysis remains available."
            ),
            "model_available": False,
            "model_load": model_status,
            "model_prediction": None,
            "model_input": None,
            "classical_analysis_status": classical_result["status"],
            "classical_analysis_message": classical_result["message"],
            "classical_spectral_summary": classical_result["summary"],
            "notes": [
                "The experimental CRVSE model prediction is unavailable.",
                "The browser camera workflow and classical spectral rPPG analysis can still be used.",
                "This is not a medical measurement.",
            ],
        }
    )


def register_api_routes(rt, model_bundle, model_status: dict | None = None) -> None:
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

    @rt("/api/roi-sample", methods=["POST"])
    async def roi_sample_api(request: Request):
        """
        Extract one compact live ROI sample from a browser-submitted frame.

        This endpoint is used by the main measurement loop. It returns numeric
        ROI summaries only and does not store the submitted frame.
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

            result = summarize_live_roi_sample_from_data_url_frame(image_data_url)

            return JSONResponse(result)

        except Exception as exc:
            return JSONResponse(
                {
                    "status": "error",
                    "stage": "roi_sample_api",
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
    async def analyze_roi_series_api(request: Request) -> JSONResponse:
        """Analyze browser-collected ROI RGB samples into candidate rPPG signals."""
        try:
            payload = await request.json()
            result = analyze_roi_series_payload(payload)
            result = make_json_safe_for_api(result)
            status_code = 200 if result.get("status") == "ok" else 400
            return JSONResponse(result, status_code=status_code)
        except Exception as exc:
            return JSONResponse(
                {
                    "status": "error",
                    "message": f"{type(exc).__name__}: {exc}",
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
            if model_bundle is None:
                return JSONResponse(
                    build_model_unavailable_prediction_payload(
                        payload=payload,
                        model_status=model_status,
                    ),
                    status_code=200,
                )
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