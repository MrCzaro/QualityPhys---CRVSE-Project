"""
Compact ROI sample API contract smoke test.

This test protects the browser sampling contract used by the main measurement
loop. It does not require a camera frame or a real MediaPipe face detection run;
instead it stubs the route-level ROI extractor and validates the JSON shape that
the browser expects from /api/roi-sample.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
APP_DIR = SCRIPT_DIR.parent
REPO_DIR = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import backend.api_routes as api_routes
from backend.face_debug import _compact_roi_debug_for_live_sample


class FakeRequest:
    """Minimal async request object for direct route-handler testing."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload

    async def json(self) -> dict[str, Any]:
        return self.payload


def make_route_registry():
    """Create a fake FastHTML route decorator and route registry."""
    routes = {}

    def rt(path: str, methods: list[str] | None = None):
        normalized_methods = tuple(methods or ["GET"])

        def decorator(route_function):
            routes[(path, normalized_methods)] = route_function
            return route_function

        return decorator

    return routes, rt


def make_full_face_summary_stub() -> dict[str, Any]:
    """Build a full debug-style face summary for compacting into ROI output."""
    return {
        "face_detected": True,
        "message": "Face landmarks detected.",
        "roi_debug": {
            "method": "bbox_relative_rectangles_v3_with_quality_v1",
            "roi_count": 3,
            "usable_roi_count": 3,
            "quality_summary": {
                "overall_status": "ok",
                "overall_reason": "All ROIs passed one-frame intensity quality checks.",
                "status_counts": {
                    "ok": 3,
                    "warning": 0,
                    "reject": 0,
                },
            },
            "rois": [
                {
                    "name": "forehead",
                    "usable": True,
                    "box": {"x_min": 10, "x_max": 30, "y_min": 10, "y_max": 20},
                    "rgb_summary": {
                        "mean_rgb": {"r": 190.0, "g": 140.0, "b": 120.0},
                        "std_rgb": {"r": 12.0, "g": 10.0, "b": 9.0},
                    },
                    "quality": {
                        "status": "ok",
                        "reasons": ["ROI one-frame intensity quality looks acceptable."],
                    },
                },
                {
                    "name": "image_left_cheek",
                    "usable": True,
                    "box": {"x_min": 5, "x_max": 20, "y_min": 30, "y_max": 45},
                    "rgb_summary": {
                        "mean_rgb": {"r": 170.0, "g": 120.0, "b": 95.0},
                        "std_rgb": {"r": 11.0, "g": 8.0, "b": 7.0},
                    },
                    "quality": {
                        "status": "ok",
                        "reasons": ["ROI one-frame intensity quality looks acceptable."],
                    },
                },
                {
                    "name": "image_right_cheek",
                    "usable": True,
                    "box": {"x_min": 40, "x_max": 55, "y_min": 30, "y_max": 45},
                    "rgb_summary": {
                        "mean_rgb": {"r": 175.0, "g": 125.0, "b": 100.0},
                        "std_rgb": {"r": 10.0, "g": 9.0, "b": 7.0},
                    },
                    "quality": {
                        "status": "ok",
                        "reasons": ["ROI one-frame intensity quality looks acceptable."],
                    },
                },
            ],
        },
    }


def fake_live_roi_sample_from_data_url_frame(image_data_url: str) -> dict[str, Any]:
    """Return a compact route payload without running image decode or MediaPipe."""
    if image_data_url != "data:image/jpeg;base64,fake":
        raise ValueError("Unexpected test image payload.")

    full_face_summary = make_full_face_summary_stub()
    compact_roi_debug = _compact_roi_debug_for_live_sample(full_face_summary)

    return {
        "status": "processed",
        "message": "Live ROI sample extracted. Frame was not stored.",
        "face_debug": {
            "face_detected": True,
            "message": "Face landmarks detected.",
            "roi_debug": compact_roi_debug,
        },
        "timing_ms": {
            "decode_ms": 1.0,
            "face_debug_total_ms": 12.0,
            "total_request_processing_ms": 13.0,
        },
    }


def response_json(response) -> dict[str, Any]:
    """Decode a Starlette JSONResponse body."""
    return json.loads(response.body.decode("utf-8"))


def validate_compact_roi_debug(roi_debug: dict[str, Any]) -> None:
    """Validate the compact ROI debug shape consumed by live_demo.js."""
    required_keys = {
        "method",
        "roi_count",
        "usable_roi_count",
        "quality_summary",
        "rois",
    }

    missing = required_keys.difference(roi_debug.keys())

    if missing:
        raise KeyError(f"Compact roi_debug is missing keys: {sorted(missing)}")

    rois = roi_debug["rois"]

    if not isinstance(rois, list):
        raise TypeError("roi_debug.rois must be a list.")

    expected_names = {"forehead", "image_left_cheek", "image_right_cheek"}
    actual_names = {roi.get("name") for roi in rois}

    if actual_names != expected_names:
        raise AssertionError(f"Unexpected ROI names: {sorted(actual_names)}")

    for roi in rois:
        if not isinstance(roi.get("usable"), bool):
            raise TypeError(f"ROI {roi.get('name')} usable must be bool.")

        mean_rgb = roi.get("rgb_summary", {}).get("mean_rgb")

        if not isinstance(mean_rgb, dict):
            raise TypeError(f"ROI {roi.get('name')} missing rgb_summary.mean_rgb.")

        for channel in ["r", "g", "b"]:
            value = mean_rgb.get(channel)

            if not isinstance(value, int | float):
                raise TypeError(f"ROI {roi.get('name')} mean RGB {channel} must be numeric.")

        quality_status = roi.get("quality", {}).get("status")

        if not isinstance(quality_status, str):
            raise TypeError(f"ROI {roi.get('name')} quality.status must be string.")


def validate_ok_payload(payload: dict[str, Any]) -> None:
    """Validate the successful /api/roi-sample response shape."""
    required_keys = {
        "status",
        "message",
        "face_debug",
        "timing_ms",
    }

    missing = required_keys.difference(payload.keys())

    if missing:
        raise KeyError(f"ROI sample payload is missing keys: {sorted(missing)}")

    if payload["status"] != "processed":
        raise AssertionError(f"Expected status='processed', got {payload['status']!r}")

    face_debug = payload["face_debug"]

    if face_debug.get("face_detected") is not True:
        raise AssertionError("Expected face_debug.face_detected to be True.")

    validate_compact_roi_debug(face_debug["roi_debug"])

    timing_ms = payload["timing_ms"]

    for key in ["decode_ms", "face_debug_total_ms", "total_request_processing_ms"]:
        value = timing_ms.get(key)

        if not isinstance(value, int | float):
            raise TypeError(f"timing_ms.{key} must be numeric.")


async def main_async() -> None:
    """Run the compact ROI sample API contract smoke test."""
    print("=" * 72)
    print("Compact ROI sample API contract smoke test")
    print("=" * 72)
    print(f"App dir: {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    compact_roi_debug = _compact_roi_debug_for_live_sample(make_full_face_summary_stub())

    if compact_roi_debug is None:
        raise AssertionError("Compact ROI debug helper returned None for a valid face summary.")

    validate_compact_roi_debug(compact_roi_debug)

    original_extractor = api_routes.summarize_live_roi_sample_from_data_url_frame

    try:
        api_routes.summarize_live_roi_sample_from_data_url_frame = (
            fake_live_roi_sample_from_data_url_frame
        )

        routes, rt = make_route_registry()
        api_routes.register_api_routes(rt, model_bundle=None, model_status=None)

        roi_route_key = ("/api/roi-sample", ("POST",))
        debug_face_route_key = ("/api/debug-face", ("POST",))

        if roi_route_key not in routes:
            raise KeyError("/api/roi-sample POST route was not registered.")

        if debug_face_route_key not in routes:
            raise KeyError("/api/debug-face POST route was not registered.")

        if routes[roi_route_key] is routes[debug_face_route_key]:
            raise AssertionError("/api/roi-sample and /api/debug-face must be separate handlers.")

        roi_sample_api = routes[roi_route_key]

        ok_response = await roi_sample_api(
            FakeRequest({"image_data_url": "data:image/jpeg;base64,fake"})
        )

        if ok_response.status_code != 200:
            raise AssertionError(f"Expected 200 OK, got {ok_response.status_code}")

        ok_payload = response_json(ok_response)
        validate_ok_payload(ok_payload)

        missing_field_response = await roi_sample_api(FakeRequest({}))

        if missing_field_response.status_code != 400:
            raise AssertionError(
                f"Expected missing-field response 400, got {missing_field_response.status_code}"
            )

        missing_field_payload = response_json(missing_field_response)

        if missing_field_payload.get("stage") != "request_validation":
            raise AssertionError("Missing-field response should identify request_validation stage.")

    finally:
        api_routes.summarize_live_roi_sample_from_data_url_frame = original_extractor

    print("Validated compact helper payload.")
    print("Validated /api/roi-sample success response.")
    print("Validated /api/roi-sample missing-field error response.")
    print("Validated /api/roi-sample remains separate from /api/debug-face.")
    print()
    print("=" * 72)
    print("PASS: compact ROI sample API contract smoke test ran successfully")
    print("=" * 72)


def main() -> None:
    """Run async smoke test entrypoint."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()