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
import time, torch, sys
from fasthtml.common import *
from starlette.requests import Request
from starlette.responses import JSONResponse

APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from backend.face_debug import summarize_face_from_data_url_frame
from backend.frame_debug import summarize_data_url_frame
from inference.serialization import prediction_result_to_dict
from inference.window_inference import predict_hr_from_rppg_window
from models.loader import load_model_bundle
from rppg.live_methods import analyze_roi_series_payload, build_model_input_from_roi_series_payload
from rppg.sqi import estimate_spectral_sqi
from rppg.windowing import make_synthetic_rppg_channels, zscore_1d
from ui.plots import render_spectrum_svg, render_waveform_svg


app, rt = fast_app(title="QualityPhys Live HR Demo")

MODEL_BUNDLE = load_model_bundle(device="cpu")


def make_synthetic_demo_payload(
    synthetic_hr_bpm: float = 72.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic rPPG inference payload for UI and API testing.

    Parameters
    ----------
    synthetic_hr_bpm:
        Synthetic pulse rate used to generate the fake POS/CHROM/GREEN traces.

    noise_std:
        Amount of synthetic noise added to the traces.

    seed:
        Random seed. Changing it creates slightly different synthetic traces.

    Physiology:
        We simulate a pulse-like rhythm at a chosen BPM.

    Signal:
        The synthetic POS/CHROM/GREEN traces go through the same quality and model
        inference path as future real rPPG traces.

    Limitation:
        Synthetic data is not a real webcam signal. This only tests backend,
        serialization, route, and visualization wiring.
    """
    fps = 30.0
    duration_seconds = float(MODEL_BUNDLE.model_spec["input"]["window_seconds"])
    signals = make_synthetic_rppg_channels(
        hr_bpm=synthetic_hr_bpm,
        duration_seconds=duration_seconds,
        fps=fps,
        noise_std=noise_std,
        seed=seed,
    )
    result = predict_hr_from_rppg_window(signals=signals, fps=fps, bundle=MODEL_BUNDLE)
    result_dict = prediction_result_to_dict(result)
    preprocessing_config = MODEL_BUNDLE.model_spec["preprocessing"]
    pos_for_display = zscore_1d(signals["pos"])
    spectrum = estimate_spectral_sqi(
        signal=signals["pos"],
        fps=fps,
        low_hz=float(preprocessing_config["bandpass_low_hz"]),
        high_hz=float(preprocessing_config["bandpass_high_hz"]),
    )
    waveform_svg = render_waveform_svg(
        time_s=signals["time"],
        signal=pos_for_display,
        title="Synthetic POS rPPG waveform",
    )
    spectrum_svg = render_spectrum_svg(
        freqs_hz=spectrum.freqs_hz,
        power=spectrum.power,
        low_hz=float(preprocessing_config["bandpass_low_hz"]),
        high_hz=float(preprocessing_config["bandpass_high_hz"]),
        dominant_freq_hz=spectrum.dominant_freq_hz,
        title="Synthetic POS power spectrum",
    )

    return {
        "result": result_dict,
        "waveform_svg": waveform_svg,
        "spectrum_svg": spectrum_svg,
    }


def metric_row(label: str, value) -> FT:
    """
    Render one metric row.
    """
    return Div(
        Span(label, cls="font-medium text-slate-700"),
        Span(str(value), cls="font-mono text-slate-900"),
        cls="flex justify-between gap-4 border-b border-slate-100 py-1",
    )


def result_card(result: dict) -> FT:
    """
    Render inference result as a simple card.
    """
    quality = result["quality"]
    metrics = quality["metrics"]
    extra = result["extra"]
    model_hr = result["model_hr_bpm"]
    spectral_hr = extra.get("spectral_hr_bpm")

    if model_hr is None:
        hr_display = "Unavailable"
    else:
        hr_display = f"{float(model_hr):.1f} {result['unit']}"

    if spectral_hr is None:
        spectral_display = "Unavailable"
    else:
        spectral_display = f"{float(spectral_hr):.1f} {result['unit']}"

    return Div(
        Div(
            H2("Synthetic inference result", cls="text-xl font-semibold"),
            P(
                "This result is generated from a synthetic 72 BPM rPPG-like signal. "
                "It proves the backend inference pipeline is connected to the UI.",
                cls="text-sm text-slate-600",
            ),
            A(
                "Open JSON endpoint",
                href="/api/synthetic-result",
                target="_blank",
                cls=(
                    "mt-3 inline-block rounded-lg border border-slate-300 "
                    "bg-white px-3 py-2 text-sm font-medium text-slate-700 "
                    "shadow-sm hover:bg-slate-50"
                ),
            ),
            cls="mb-4",
        ),
        Div(
            Div(
                Div("Model HR", cls="text-sm text-slate-500"),
                Div(hr_display, cls="text-4xl font-bold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
            ),
            Div(
                Div("Spectral HR", cls="text-sm text-slate-500"),
                Div(spectral_display, cls="text-4xl font-bold text-slate-900"),
                cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
            ),
            Div(
                Div("Quality", cls="text-sm text-slate-500"),
                Div(
                    f"{quality['status']} / {quality['confidence']}",
                    cls="text-2xl font-semibold text-slate-900",
                ),
                cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
            ),
            cls="grid gap-4 md:grid-cols-3",
        ),
        Div(
            H3("Quality metrics", cls="mt-6 mb-2 text-lg font-semibold"),
            Div(
                metric_row("POS SQI", f"{metrics.get('pos_sqi'):.3f}"),
                metric_row("CHROM SQI", f"{metrics.get('chrom_sqi'):.3f}"),
                metric_row("GREEN SQI", f"{metrics.get('green_sqi'):.3f}"),
                metric_row(
                    "BPM spread across channels",
                    f"{metrics.get('bpm_spread_across_channels'):.1f}",
                ),
                cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
            ),
        ),
        Div(
            H3("Accepted / rejected reasons", cls="mt-6 mb-2 text-lg font-semibold"),
            Ul(
                *[
                    Li(reason, cls="mb-1 text-sm text-slate-700")
                    for reason in quality["reasons"]
                ],
                cls="list-disc pl-5",
            ),
        ),
        cls="rounded-2xl border border-slate-200 bg-slate-50 p-6 shadow-sm",
    )


def plots_card(waveform_svg: str, spectrum_svg: str) -> FT:
    """
    Render waveform and spectrum plots.

    Note:
        Current spectrum plot is a developer diagnostic placeholder.
        We may later improve it, hide it in an advanced section, or replace it.
    """

    return Div(
        Div(
            H2("Developer signal diagnostics", cls="text-xl font-semibold"),
            P(
                "These plots show the synthetic POS rPPG trace and its power spectrum. "
                "Later, the same plot slots can display live camera-derived signals. "
                "The spectrum plot is currently a debug diagnostic, not final UI polish.",
                cls="text-sm text-slate-600",
            ),
            cls="mb-4",
        ),
        Div(
            Div(
                NotStr(waveform_svg),
                cls="rounded-xl border border-slate-200 bg-white p-3 shadow-sm",
            ),
            Div(
                NotStr(spectrum_svg),
                cls="rounded-xl border border-slate-200 bg-white p-3 shadow-sm",
            ),
            cls="grid gap-4 lg:grid-cols-2",
        ),
        cls="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-6 shadow-sm",
    )


def api_refresh_demo_card() -> FT:
    """
    Render a small frontend-backend refresh demo card.

    This proves the browser can call the backend endpoint and update UI fields
    without a full page reload.
    """

    return Div(
        Div(
            H2("API refresh demo", cls="text-xl font-semibold"),
            P(
                "This button calls /api/synthetic-result from the browser and updates "
                "the values below without reloading the page. This is the same pattern "
                "we will later use for live camera updates.",
                cls="text-sm text-slate-600",
            ),
            Button(
                "Refresh synthetic result",
                id="refresh-api-button",
                cls=(
                    "mt-4 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium "
                    "text-white shadow-sm hover:bg-slate-700"
                ),
            ),
            cls="mb-4",
        ),
        Div(
            Div(
                Div("API Model HR", cls="text-sm text-slate-500"),
                Div("Not loaded yet", id="api-model-hr", cls="text-2xl font-bold"),
                cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
            ),
            Div(
                Div("API Spectral HR", cls="text-sm text-slate-500"),
                Div("Not loaded yet", id="api-spectral-hr", cls="text-2xl font-bold"),
                cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
            ),
            Div(
                Div("API Quality", cls="text-sm text-slate-500"),
                Div("Not loaded yet", id="api-quality", cls="text-2xl font-bold"),
                cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
            ),
            cls="grid gap-4 md:grid-cols-3",
        ),
        Div(
            H3("Latest API reason", cls="mt-6 mb-2 text-lg font-semibold"),
            Div(
                "No API result loaded yet.",
                id="api-reason",
                cls="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-700 shadow-sm",
            ),
        ),
        cls="mt-6 rounded-2xl border border-slate-200 bg-slate-50 p-6 shadow-sm",
    )



def live_demo_script() -> FT:
    """
    Return JavaScript used by the API refresh demo and camera preview.

    Current browser behaviors:
        1. Refresh synthetic backend result through /api/synthetic-result.
        2. Start camera preview locally in the browser.
        3. Capture one local video frame into a canvas.
        4. Send one captured frame to backend for debug decoding.
        5. Send one captured frame to backend for face/ROI diagnostics.
        6. Draw backend face/ROI boxes over the captured canvas.
        7. Collect repeated ROI RGB samples for a short debug time series.
        8. Plot raw green-channel traces.
        9. Plot normalized green-channel traces.
        10. Stop camera stream cleanly.

    Privacy:
        Frames are only sent when the user clicks a backend-send button or starts
        ROI sampling.
        Backend debug routes decode/process each frame in memory and do not store it.
        Browser stores only numeric ROI RGB summaries in memory.

    Learning goal:
        Inspect whether real ROI RGB time series is stable enough before building
        POS / CHROM / GREEN extraction.
    """

    script = """
    let cameraStream = null;
    let hasCapturedFrame = false;
    let roiSamplingTimer = null;
    let roiSamples = [];
    let roiSamplingStartMs = null;
    let roiSamplingInFlight = false;

    const ROI_NAMES = ["forehead", "image_left_cheek", "image_right_cheek"];
    const ROI_COLORS = {
    forehead: "#7c3aed",
    image_left_cheek: "#16a34a",
    image_right_cheek: "#dc2626"
    };

    async function refreshSyntheticResult() {
    const button = document.getElementById("refresh-api-button");
    const modelHrEl = document.getElementById("api-model-hr");
    const spectralHrEl = document.getElementById("api-spectral-hr");
    const qualityEl = document.getElementById("api-quality");
    const reasonEl = document.getElementById("api-reason");

    button.disabled = true;
    button.innerText = "Refreshing...";

    try {
        const response = await fetch("/api/synthetic-result");

        if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        const modelHr = data.model_hr_bpm;
        const spectralHr = data.extra?.spectral_hr_bpm;
        const unit = data.unit ?? "bpm";

        modelHrEl.innerText = modelHr === null ? "Unavailable" : `${modelHr.toFixed(1)} ${unit}`;
        spectralHrEl.innerText = spectralHr === null ? "Unavailable" : `${spectralHr.toFixed(1)} ${unit}`;
        qualityEl.innerText = `${data.quality.status} / ${data.quality.confidence}`;

        const reasons = data.quality.reasons ?? [];
        reasonEl.innerText = reasons.length > 0 ? reasons[0] : "No reason returned.";
    } catch (error) {
        modelHrEl.innerText = "Error";
        spectralHrEl.innerText = "Error";
        qualityEl.innerText = "Error";
        reasonEl.innerText = `API call failed: ${error}`;
    } finally {
        button.disabled = false;
        button.innerText = "Refresh synthetic result";
    }
    }

    async function startCameraPreview() {
    const videoEl = document.getElementById("camera-video");
    const statusEl = document.getElementById("camera-status");
    const startButton = document.getElementById("start-camera-button");

    startButton.disabled = true;
    startButton.innerText = "Starting...";

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
        video: {
            width: { ideal: 640 },
            height: { ideal: 480 },
            frameRate: { ideal: 30 }
        },
        audio: false
        });

        videoEl.srcObject = cameraStream;

        statusEl.innerText =
        "Camera started. Preview is local in the browser. Frames are not sent to the backend automatically.";
    } catch (error) {
        statusEl.innerText = `Camera start failed: ${error}`;
    } finally {
        startButton.disabled = false;
        startButton.innerText = "Start camera";
    }
    }

    function captureOneFrame() {
    const videoEl = document.getElementById("camera-video");
    const canvasEl = document.getElementById("snapshot-canvas");
    const statusEl = document.getElementById("camera-status");

    if (!cameraStream) {
        statusEl.innerText = "Cannot capture frame: camera is not started.";
        return false;
    }

    const width = videoEl.videoWidth;
    const height = videoEl.videoHeight;

    if (width === 0 || height === 0) {
        statusEl.innerText = "Cannot capture frame yet: video dimensions are not ready.";
        return false;
    }

    canvasEl.width = width;
    canvasEl.height = height;

    const context = canvasEl.getContext("2d");
    context.drawImage(videoEl, 0, 0, width, height);

    hasCapturedFrame = true;

    statusEl.innerText =
        `Captured one local frame: ${width} x ${height}. Frame was not sent to backend.`;

    return true;
    }

    function getCapturedFrameDataUrl() {
    const canvasEl = document.getElementById("snapshot-canvas");

    if (!hasCapturedFrame) {
        throw new Error("Capture one frame first.");
    }

    return canvasEl.toDataURL("image/jpeg", 0.85);
    }

    async function parseJsonResponseEvenOnError(response) {
    let data = null;

    try {
        data = await response.json();
    } catch (error) {
        data = {
        status: "error",
        message: `Could not parse JSON response: ${error}`
        };
    }

    if (!response.ok) {
        const message = data.message ?? `HTTP ${response.status}`;
        const exceptionType = data.exception_type ?? "BackendError";
        throw new Error(`${response.status} ${exceptionType}: ${message}`);
    }

    return data;
    }

    async function sendCapturedFrameToBackend() {
    const statusEl = document.getElementById("camera-status");
    const debugEl = document.getElementById("backend-frame-debug");
    const sendButton = document.getElementById("send-frame-button");

    sendButton.disabled = true;
    sendButton.innerText = "Sending...";

    try {
        const imageDataUrl = getCapturedFrameDataUrl();

        const response = await fetch("/api/debug-frame", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            image_data_url: imageDataUrl
        })
        });

        const data = await parseJsonResponseEvenOnError(response);

        debugEl.innerText = JSON.stringify(data, null, 2);
        statusEl.innerText =
        `Backend decoded frame: ${data.frame.width} x ${data.frame.height}, ${data.frame.channels} channels. Frame was not stored.`;
    } catch (error) {
        debugEl.innerText = `Frame send failed: ${error}`;
        statusEl.innerText = `Frame send failed: ${error}`;
    } finally {
        sendButton.disabled = false;
        sendButton.innerText = "Send frame to backend";
    }
    }

    function formatRoiDisplayName(name) {
    return name;
    }

    function getRoiColor(name) {
    return ROI_COLORS[name] ?? "#2563eb";
    }

    function getLabelPosition(canvasEl, box, label, fontSize) {
    const padding = 4;
    const labelHeight = fontSize + 8;

    const context = canvasEl.getContext("2d");
    const labelWidth = context.measureText(label).width + 12;

    let labelX = box.x_min;
    let labelY = box.y_min - labelHeight - padding;

    if (labelY < 0) {
        labelY = box.y_max + padding;
    }

    if (labelY + labelHeight > canvasEl.height) {
        labelY = Math.max(0, box.y_min + padding);
    }

    if (labelX + labelWidth > canvasEl.width) {
        labelX = canvasEl.width - labelWidth - padding;
    }

    if (labelX < 0) {
        labelX = padding;
    }

    return {
        x: labelX,
        y: labelY,
        width: labelWidth,
        height: labelHeight
    };
    }

    function drawRectangleWithLabel(context, canvasEl, box, label, strokeStyle) {
    context.save();

    context.strokeStyle = strokeStyle;
    context.lineWidth = 3;
    context.strokeRect(box.x_min, box.y_min, box.width, box.height);

    const fontSize = 13;
    context.font = `${fontSize}px sans-serif`;

    const labelPosition = getLabelPosition(canvasEl, box, label, fontSize);

    context.fillStyle = "rgba(15, 23, 42, 0.88)";
    context.fillRect(
        labelPosition.x,
        labelPosition.y,
        labelPosition.width,
        labelPosition.height
    );

    context.fillStyle = "white";
    context.fillText(
        label,
        labelPosition.x + 6,
        labelPosition.y + fontSize + 2
    );

    context.restore();
    }

    function drawFaceAndRoiOverlay(data) {
    const canvasEl = document.getElementById("snapshot-canvas");
    const context = canvasEl.getContext("2d");

    const faceDebug = data.face_debug;

    if (!faceDebug || !faceDebug.face_detected) {
        return;
    }

    const faceBox = faceDebug.face?.bbox;
    const roiDebug = faceDebug.roi_debug;
    const rois = roiDebug?.rois ?? [];

    if (faceBox) {
        drawRectangleWithLabel(context, canvasEl, faceBox, "face", "#0f172a");
    }

    for (const roi of rois) {
        if (!roi.usable) {
        continue;
        }

        const label = formatRoiDisplayName(roi.name);
        const color = getRoiColor(roi.name);

        drawRectangleWithLabel(context, canvasEl, roi.box, label, color);
    }
    }

    function summarizeRoiMeans(data) {
    const rois = data.face_debug?.roi_debug?.rois ?? [];

    if (rois.length === 0) {
        return "No ROI summaries returned.";
    }

    const parts = [];

    for (const roi of rois) {
        const meanRgb = roi.rgb_summary?.mean_rgb;
        const qualityStatus = roi.quality?.status ?? "unknown";

        if (!meanRgb) {
        parts.push(`${roi.name}: no RGB summary`);
        continue;
        }

        parts.push(
        `${roi.name}: ${qualityStatus}, R=${meanRgb.r.toFixed(1)}, G=${meanRgb.g.toFixed(1)}, B=${meanRgb.b.toFixed(1)}`
        );
    }

    return parts.join(" | ");
    }

    async function detectFaceInBackend() {
    const statusEl = document.getElementById("camera-status");
    const debugEl = document.getElementById("backend-face-debug");
    const detectButton = document.getElementById("detect-face-button");

    detectButton.disabled = true;
    detectButton.innerText = "Detecting...";

    try {
        const imageDataUrl = getCapturedFrameDataUrl();

        const response = await fetch("/api/debug-face", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            image_data_url: imageDataUrl
        })
        });

        const data = await parseJsonResponseEvenOnError(response);

        debugEl.innerText = JSON.stringify(data, null, 2);

        const faceDetected = data.face_debug?.face_detected;
        const landmarkCount = data.face_debug?.face?.landmark_count;
        const bbox = data.face_debug?.face?.bbox;

        if (faceDetected) {
        drawFaceAndRoiOverlay(data);

        const roiSummary = summarizeRoiMeans(data);

        statusEl.innerText =
            `Backend detected face: ${landmarkCount} landmarks, bbox ${bbox.width} x ${bbox.height}. ` +
            `ROIs drawn. ${roiSummary}`;
        } else {
        statusEl.innerText = "Backend did not detect a face in this frame.";
        }
    } catch (error) {
        debugEl.innerText = `Face detection failed: ${error}`;
        statusEl.innerText = `Face detection failed: ${error}`;
    } finally {
        detectButton.disabled = false;
        detectButton.innerText = "Detect face + draw ROIs";
    }
    }

    function extractRoiSampleFromBackendResponse(data) {
    const nowMs = performance.now();
    const elapsedS = roiSamplingStartMs === null ? 0.0 : (nowMs - roiSamplingStartMs) / 1000.0;

    const faceDebug = data.face_debug;
    const roiDebug = faceDebug?.roi_debug;
    const rois = roiDebug?.rois ?? [];

    const sample = {
        t_s: elapsedS,
        face_detected: Boolean(faceDebug?.face_detected),
        roi_quality_overall: roiDebug?.quality_summary?.overall_status ?? "unknown",
        rois: {}
    };

    for (const roi of rois) {
        const meanRgb = roi.rgb_summary?.mean_rgb;

        if (!meanRgb) {
        continue;
        }

        sample.rois[roi.name] = {
        r: meanRgb.r,
        g: meanRgb.g,
        b: meanRgb.b,
        quality_status: roi.quality?.status ?? "unknown"
        };
    }

    return sample;
    }

    function getRoiGreenSeries(roiName) {
    return roiSamples
        .filter(sample => sample.rois[roiName] !== undefined)
        .map(sample => {
        return {
            t: sample.t_s,
            g: sample.rois[roiName].g,
            quality_status: sample.rois[roiName].quality_status
        };
        });
    }

    function zscoreSeries(series) {
    if (series.length === 0) {
        return [];
    }

    const values = series.map(point => point.g);
    const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / values.length;
    const std = Math.sqrt(variance);

    if (std < 1e-8) {
        return series.map(point => {
        return {
            t: point.t,
            z: 0.0,
            quality_status: point.quality_status
        };
        });
    }

    return series.map(point => {
        return {
        t: point.t,
        z: (point.g - mean) / std,
        quality_status: point.quality_status
        };
    });
    }

    function drawTracePlot(options) {
    const canvasEl = document.getElementById(options.canvasId);

    if (!canvasEl) {
        return;
    }

    const context = canvasEl.getContext("2d");
    const width = canvasEl.width;
    const height = canvasEl.height;

    context.clearRect(0, 0, width, height);

    context.fillStyle = "white";
    context.fillRect(0, 0, width, height);

    const paddingLeft = 58;
    const paddingRight = 20;
    const paddingTop = 28;
    const paddingBottom = 38;

    const plotX0 = paddingLeft;
    const plotX1 = width - paddingRight;
    const plotY0 = paddingTop;
    const plotY1 = height - paddingBottom;

    context.strokeStyle = "#cbd5e1";
    context.lineWidth = 1;

    context.beginPath();
    context.moveTo(plotX0, plotY1);
    context.lineTo(plotX1, plotY1);
    context.moveTo(plotX0, plotY0);
    context.lineTo(plotX0, plotY1);
    context.stroke();

    context.fillStyle = "#0f172a";
    context.font = "14px sans-serif";
    context.fillText(options.title, plotX0, 18);

    if (roiSamples.length < 2) {
        context.fillStyle = "#64748b";
        context.font = "13px sans-serif";
        context.fillText("Collect ROI samples to plot traces.", plotX0, 60);
        return;
    }

    const plottedSeries = {};
    let allYValues = [];

    for (const roiName of ROI_NAMES) {
        const rawSeries = getRoiGreenSeries(roiName);

        if (options.mode === "zscore") {
        plottedSeries[roiName] = zscoreSeries(rawSeries);
        allYValues = allYValues.concat(plottedSeries[roiName].map(point => point.z));
        } else {
        plottedSeries[roiName] = rawSeries;
        allYValues = allYValues.concat(plottedSeries[roiName].map(point => point.g));
        }
    }

    if (allYValues.length === 0) {
        return;
    }

    const tMin = roiSamples[0].t_s;
    const tMax = roiSamples[roiSamples.length - 1].t_s;
    const yMinRaw = Math.min(...allYValues);
    const yMaxRaw = Math.max(...allYValues);

    const yPad = Math.max(0.25, (yMaxRaw - yMinRaw) * 0.20);
    const yMin = yMinRaw - yPad;
    const yMax = yMaxRaw + yPad;

    function xScale(t) {
        if (tMax <= tMin) {
        return plotX0;
        }

        return plotX0 + (t - tMin) / (tMax - tMin) * (plotX1 - plotX0);
    }

    function yScale(yValue) {
        if (yMax <= yMin) {
        return (plotY0 + plotY1) / 2;
        }

        return plotY1 - (yValue - yMin) / (yMax - yMin) * (plotY1 - plotY0);
    }

    context.strokeStyle = "#e2e8f0";
    context.lineWidth = 1;

    for (let i = 1; i <= 3; i += 1) {
        const y = plotY0 + i / 4 * (plotY1 - plotY0);
        context.beginPath();
        context.moveTo(plotX0, y);
        context.lineTo(plotX1, y);
        context.stroke();
    }

    if (options.mode === "zscore") {
        const zeroY = yScale(0.0);
        context.strokeStyle = "#94a3b8";
        context.setLineDash([5, 5]);
        context.beginPath();
        context.moveTo(plotX0, zeroY);
        context.lineTo(plotX1, zeroY);
        context.stroke();
        context.setLineDash([]);
    }

    for (const roiName of ROI_NAMES) {
        const series = plottedSeries[roiName];

        if (series.length < 2) {
        continue;
        }

        context.strokeStyle = getRoiColor(roiName);
        context.lineWidth = 2;
        context.beginPath();

        for (let i = 0; i < series.length; i += 1) {
        const yValue = options.mode === "zscore" ? series[i].z : series[i].g;
        const x = xScale(series[i].t);
        const y = yScale(yValue);

        if (i === 0) {
            context.moveTo(x, y);
        } else {
            context.lineTo(x, y);
        }
        }

        context.stroke();
    }

    context.fillStyle = "#64748b";
    context.font = "11px sans-serif";
    context.fillText(`${tMin.toFixed(1)}s`, plotX0, height - 14);
    context.fillText(`${tMax.toFixed(1)}s`, plotX1 - 34, height - 14);
    context.fillText(`${options.yLabel} min ${yMinRaw.toFixed(2)}`, 8, plotY1);
    context.fillText(`${options.yLabel} max ${yMaxRaw.toFixed(2)}`, 8, plotY0 + 8);

    let legendX = plotX0 + 260;
    const legendY = 18;

    for (const roiName of ROI_NAMES) {
        context.fillStyle = getRoiColor(roiName);
        context.fillRect(legendX, legendY - 9, 10, 10);

        context.fillStyle = "#0f172a";
        context.font = "11px sans-serif";
        context.fillText(roiName, legendX + 14, legendY);

        legendX += 150;
    }
    }

    function drawRoiGreenTracePlot() {
    drawTracePlot({
        canvasId: "roi-green-trace-canvas",
        title: "Raw ROI green-channel traces",
        mode: "raw",
        yLabel: "G"
    });
    }

    function drawNormalizedRoiGreenTracePlot() {
    drawTracePlot({
        canvasId: "roi-green-normalized-trace-canvas",
        title: "Normalized ROI green-channel traces",
        mode: "zscore",
        yLabel: "z"
    });
    }

    function drawAllRoiPlots() {
    drawRoiGreenTracePlot();
    drawNormalizedRoiGreenTracePlot();
    }

    function summarizeCollectedRoiSamples() {
    const summaryEl = document.getElementById("roi-sampling-summary");

    if (roiSamples.length === 0) {
        summaryEl.innerText = "No ROI samples collected yet.";
        drawAllRoiPlots();
        return;
    }

    const firstT = roiSamples[0].t_s;
    const lastT = roiSamples[roiSamples.length - 1].t_s;
    const durationS = Math.max(0, lastT - firstT);

    const lines = [];

    lines.push(`samples: ${roiSamples.length}`);
    lines.push(`duration_s: ${durationS.toFixed(2)}`);
    lines.push("");

    for (const roiName of ROI_NAMES) {
        const values = roiSamples
        .map(sample => sample.rois[roiName])
        .filter(value => value !== undefined);

        if (values.length === 0) {
        lines.push(`${roiName}: no samples`);
        continue;
        }

        const latest = values[values.length - 1];

        const qualityCounts = {};

        for (const value of values) {
        qualityCounts[value.quality_status] = (qualityCounts[value.quality_status] ?? 0) + 1;
        }

        const greenValues = values.map(value => value.g);
        const greenMin = Math.min(...greenValues);
        const greenMax = Math.max(...greenValues);
        const greenRange = greenMax - greenMin;

        lines.push(
        `${roiName}: n=${values.length}, latest RGB=(${latest.r.toFixed(1)}, ${latest.g.toFixed(1)}, ${latest.b.toFixed(1)}), ` +
        `green_range=${greenRange.toFixed(2)}, quality_counts=${JSON.stringify(qualityCounts)}`
        );
    }

    lines.push("");
    lines.push("Note: this is raw ROI RGB only, not rPPG and not HR.");

    summaryEl.innerText = lines.join("\\n");

    drawAllRoiPlots();
    }
    function getOrCreateHiddenSamplingCanvas() {
        let canvasEl = document.getElementById("sampling-hidden-canvas");

        if (canvasEl) {
            return canvasEl;
        }

        canvasEl = document.createElement("canvas");
        canvasEl.id = "sampling-hidden-canvas";
        canvasEl.width = 640;
        canvasEl.height = 480;
        canvasEl.style.display = "none";

        document.body.appendChild(canvasEl);

        return canvasEl;
        }

    function captureFrameForSamplingDataUrl() {
        const videoEl = document.getElementById("camera-video");
        const statusEl = document.getElementById("camera-status");
        const canvasEl = getOrCreateHiddenSamplingCanvas();

        if (!cameraStream) {
            statusEl.innerText = "Cannot capture sampling frame: camera is not started.";
            return null;
        }

        const width = videoEl.videoWidth;
        const height = videoEl.videoHeight;

        if (width === 0 || height === 0) {
            statusEl.innerText = "Cannot capture sampling frame yet: video dimensions are not ready.";
            return null;
        }

        canvasEl.width = width;
        canvasEl.height = height;

        const context = canvasEl.getContext("2d");
        context.drawImage(videoEl, 0, 0, width, height);

        return canvasEl.toDataURL("image/jpeg", 0.85);
        }
    async function collectOneRoiSample() {
        const statusEl = document.getElementById("camera-status");

        if (roiSamplingInFlight) {
            return;
        }

        roiSamplingInFlight = true;

        try {
            const imageDataUrl = captureFrameForSamplingDataUrl();

            if (!imageDataUrl) {
            return;
            }

            const response = await fetch("/api/debug-face", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                image_data_url: imageDataUrl
            })
            });

            const data = await parseJsonResponseEvenOnError(response);
            const sample = extractRoiSampleFromBackendResponse(data);

            roiSamples.push(sample);

            summarizeCollectedRoiSamples();

            statusEl.innerText =
            `ROI sampling active. Collected ${roiSamples.length} sample(s). ` +
            "Frames are processed in memory and not stored.";
        } catch (error) {
            statusEl.innerText = `ROI sampling error: ${error}`;
        } finally {
            roiSamplingInFlight = false;
        }
        }

    function startRoiSampling() {
    const statusEl = document.getElementById("camera-status");
    const startButton = document.getElementById("start-roi-sampling-button");

    const samplingIntervalMs = 100;

    if (!cameraStream) {
        statusEl.innerText = "Cannot start ROI sampling: camera is not started.";
        return;
    }

    if (roiSamplingTimer !== null) {
        statusEl.innerText = "ROI sampling is already running.";
        return;
    }

    roiSamples = [];
    roiSamplingStartMs = performance.now();

    startButton.disabled = true;
    startButton.innerText = "Sampling...";

    statusEl.innerText =
        `ROI sampling started at ${samplingIntervalMs} ms interval. Hold still for about 8-10 seconds.`;

    summarizeCollectedRoiSamples();

    collectOneRoiSample();

    roiSamplingTimer = setInterval(() => {
        collectOneRoiSample();
    }, samplingIntervalMs);
    }

    function stopRoiSampling() {
    const statusEl = document.getElementById("camera-status");
    const startButton = document.getElementById("start-roi-sampling-button");

    if (roiSamplingTimer !== null) {
        clearInterval(roiSamplingTimer);
        roiSamplingTimer = null;
    }

    startButton.disabled = false;
    startButton.innerText = "Start ROI sampling";

    summarizeCollectedRoiSamples();

    statusEl.innerText =
        `ROI sampling stopped. Collected ${roiSamples.length} sample(s).`;
    }

    function clearRoiSamples() {
    const statusEl = document.getElementById("camera-status");

    if (roiSamplingTimer !== null) {
        clearInterval(roiSamplingTimer);
        roiSamplingTimer = null;
    }

    roiSamples = [];
    roiSamplingStartMs = null;
    roiSamplingInFlight = false;

    const startButton = document.getElementById("start-roi-sampling-button");
    startButton.disabled = false;
    startButton.innerText = "Start ROI sampling";

    summarizeCollectedRoiSamples();

    statusEl.innerText = "ROI samples cleared.";
    }
    async function analyzeRoiSeriesInBackend() {
    const statusEl = document.getElementById("camera-status");
    const outputEl = document.getElementById("roi-series-analysis-output");
    const analyzeButton = document.getElementById("analyze-roi-series-button");

    const greenSummaryEl = document.getElementById("green-signal-summary");
    const posSummaryEl = document.getElementById("pos-signal-summary");
    const chromSummaryEl = document.getElementById("chrom-signal-summary");

    function formatSignalSummary(signalName, signalData) {
        const spectral = signalData?.spectral;

        if (!spectral) {
        return `${signalName}: unavailable`;
        }

        const bpm = spectral.dominant_bpm;
        const sqi = spectral.sqi;
        const spectralStatus = spectral.status ?? "unknown";

        const bpmText = bpm === null || bpm === undefined ? "no BPM" : `${bpm.toFixed(1)} bpm`;
        const sqiText = sqi === null || sqi === undefined ? "no SQI" : `SQI ${sqi.toFixed(3)}`;

        return `${bpmText} / ${sqiText} / ${spectralStatus}`;
    }

    function setCompactSignalSummaries(data) {
        if (greenSummaryEl) {
        greenSummaryEl.innerText = formatSignalSummary("GREEN", data.signals?.green);
        }

        if (posSummaryEl) {
        posSummaryEl.innerText = formatSignalSummary("POS", data.signals?.pos);
        }

        if (chromSummaryEl) {
        chromSummaryEl.innerText = formatSignalSummary("CHROM", data.signals?.chrom);
        }
    }

    if (roiSamples.length < 20) {
        statusEl.innerText = "Collect at least 20 ROI samples before analysis.";
        return;
    }

    analyzeButton.disabled = true;
    analyzeButton.innerText = "Analyzing...";

    try {
        const response = await fetch("/api/analyze-roi-series", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            samples: roiSamples
        })
        });

        const data = await parseJsonResponseEvenOnError(response);

        setCompactSignalSummaries(data);

        const greenBpm = data.signals?.green?.spectral?.dominant_bpm;
        const posBpm = data.signals?.pos?.spectral?.dominant_bpm;
        const chromBpm = data.signals?.chrom?.spectral?.dominant_bpm;

        const greenSqi = data.signals?.green?.spectral?.sqi;
        const posSqi = data.signals?.pos?.spectral?.sqi;
        const chromSqi = data.signals?.chrom?.spectral?.sqi;

        const compactSummary = [
        `status: ${data.status}`,
        `samples: ${data.sample_count}`,
        `duration_s: ${data.duration_s.toFixed(2)}`,
        `estimated_fps: ${data.estimated_fps.toFixed(2)}`,
        "",
        `GREEN: ${greenBpm === null || greenBpm === undefined ? "none" : greenBpm.toFixed(1)} bpm / SQI ${greenSqi === null || greenSqi === undefined ? "none" : greenSqi.toFixed(3)} / ${data.signals?.green?.spectral?.status ?? "unknown"}`,
        `POS:   ${posBpm === null || posBpm === undefined ? "none" : posBpm.toFixed(1)} bpm / SQI ${posSqi === null || posSqi === undefined ? "none" : posSqi.toFixed(3)} / ${data.signals?.pos?.spectral?.status ?? "unknown"}`,
        `CHROM: ${chromBpm === null || chromBpm === undefined ? "none" : chromBpm.toFixed(1)} bpm / SQI ${chromSqi === null || chromSqi === undefined ? "none" : chromSqi.toFixed(3)} / ${data.signals?.chrom?.spectral?.status ?? "unknown"}`,
        "",
        "Full response:",
        JSON.stringify(data, null, 2)
        ].join("\\n");

        outputEl.innerText = compactSummary;

        statusEl.innerText =
        `ROI series analyzed. GREEN=${greenBpm?.toFixed(1) ?? "none"} BPM, ` +
        `POS=${posBpm?.toFixed(1) ?? "none"} BPM, ` +
        `CHROM=${chromBpm?.toFixed(1) ?? "none"} BPM.`;
    } catch (error) {
        outputEl.innerText = `ROI series analysis failed: ${error}`;
        statusEl.innerText = `ROI series analysis failed: ${error}`;

        if (greenSummaryEl) {
        greenSummaryEl.innerText = "Analysis failed";
        }

        if (posSummaryEl) {
        posSummaryEl.innerText = "Analysis failed";
        }

        if (chromSummaryEl) {
        chromSummaryEl.innerText = "Analysis failed";
        }
    } finally {
        analyzeButton.disabled = false;
        analyzeButton.innerText = "Analyze ROI series";
    }
    }
    function stopCameraPreview() {
    const videoEl = document.getElementById("camera-video");
    const statusEl = document.getElementById("camera-status");

    if (roiSamplingTimer !== null) {
        clearInterval(roiSamplingTimer);
        roiSamplingTimer = null;
    }

    const startSamplingButton = document.getElementById("start-roi-sampling-button");
    startSamplingButton.disabled = false;
    startSamplingButton.innerText = "Start ROI sampling";

    if (cameraStream) {
        const tracks = cameraStream.getTracks();

        for (const track of tracks) {
        track.stop();
        }

        cameraStream = null;
    }

    videoEl.srcObject = null;
    statusEl.innerText = "Camera stopped. Captured frame remains local in the canvas.";
    }

    async function runLiveModelPredictionInBackend() {
        const statusEl = document.getElementById("camera-status");
        const outputEl = document.getElementById("live-model-prediction-output");
        const runButton = document.getElementById("run-live-model-button");

        const modelHrEl = document.getElementById("live-model-hr-summary");
        const spectralConsensusEl = document.getElementById("spectral-consensus-summary");
        const modelDifferenceEl = document.getElementById("model-spectral-difference-summary");

        if (!window.livePredictionRuns) {
            window.livePredictionRuns = [];
        }

        function getValidNumber(value) {
            if (value === null || value === undefined) {
            return null;
            }

            const numberValue = Number(value);

            if (Number.isNaN(numberValue)) {
            return null;
            }

            return numberValue;
        }

        function mean(values) {
            if (values.length === 0) {
            return null;
            }

            return values.reduce((sum, value) => sum + value, 0) / values.length;
        }

        function formatBpm(value) {
            const numberValue = getValidNumber(value);

            if (numberValue === null) {
            return "none";
            }

            return `${numberValue.toFixed(1)} bpm`;
        }

        function formatSignedBpm(value) {
            const numberValue = getValidNumber(value);

            if (numberValue === null) {
            return "none";
            }

            const sign = numberValue >= 0 ? "+" : "";

            return `${sign}${numberValue.toFixed(1)} bpm`;
        }

        function formatNumber(value, digits = 2) {
            const numberValue = getValidNumber(value);

            if (numberValue === null) {
            return "none";
            }

            return numberValue.toFixed(digits);
        }

        function spectralBpmValues(data) {
            const greenBpm = getValidNumber(data.classical_spectral_summary?.green?.dominant_bpm);
            const posBpm = getValidNumber(data.classical_spectral_summary?.pos?.dominant_bpm);
            const chromBpm = getValidNumber(data.classical_spectral_summary?.chrom?.dominant_bpm);

            return [greenBpm, posBpm, chromBpm].filter(value => value !== null);
        }

        function summarizePrediction(data) {
            const modelHr = getValidNumber(data.model_prediction?.value);
            const consensus = mean(spectralBpmValues(data));
            const difference = modelHr !== null && consensus !== null ? modelHr - consensus : null;

            const green = data.classical_spectral_summary?.green;
            const pos = data.classical_spectral_summary?.pos;
            const chrom = data.classical_spectral_summary?.chrom;

            const windowMetadata = data.model_input?.window_metadata ?? {};

            const lines = [
            `status: ${data.status}`,
            `model_hr_bpm: ${formatBpm(modelHr)}`,
            `spectral_consensus_bpm: ${formatBpm(consensus)}`,
            `model_minus_spectral: ${formatSignedBpm(difference)}`,
            "",
            `GREEN: ${formatBpm(green?.dominant_bpm)} / SQI ${green?.sqi?.toFixed(3) ?? "none"} / ${green?.status ?? "unknown"}`,
            `POS:   ${formatBpm(pos?.dominant_bpm)} / SQI ${pos?.sqi?.toFixed(3) ?? "none"} / ${pos?.status ?? "unknown"}`,
            `CHROM: ${formatBpm(chrom?.dominant_bpm)} / SQI ${chrom?.sqi?.toFixed(3) ?? "none"} / ${chrom?.status ?? "unknown"}`,
            "",
            `original_duration_s: ${windowMetadata.original_duration_s?.toFixed(2) ?? "unknown"}`,
            `used_duration_s: ${windowMetadata.used_duration_s?.toFixed(2) ?? "unknown"}`,
            `used_samples: ${windowMetadata.used_sample_count ?? "unknown"}`,
            `source_fps: ${data.model_input?.source_estimated_fps?.toFixed(2) ?? "unknown"}`,
            "",
            "Full response:",
            JSON.stringify(data, null, 2)
            ];

            return {
            modelHr,
            consensus,
            difference,
            greenBpm: getValidNumber(green?.dominant_bpm),
            posBpm: getValidNumber(pos?.dominant_bpm),
            chromBpm: getValidNumber(chrom?.dominant_bpm),
            greenSqi: getValidNumber(green?.sqi),
            posSqi: getValidNumber(pos?.sqi),
            chromSqi: getValidNumber(chrom?.sqi),
            greenStatus: green?.status ?? "unknown",
            posStatus: pos?.status ?? "unknown",
            chromStatus: chrom?.status ?? "unknown",
            originalDurationS: getValidNumber(windowMetadata.original_duration_s),
            usedDurationS: getValidNumber(windowMetadata.used_duration_s),
            usedSamples: getValidNumber(windowMetadata.used_sample_count),
            sourceFps: getValidNumber(data.model_input?.source_estimated_fps),
            text: lines.join("\\n")
            };
        }

        function ensureRepeatabilityTableExists() {
            let container = document.getElementById("live-model-repeatability-container");

            if (container) {
            return container;
            }

            container = document.createElement("div");
            container.id = "live-model-repeatability-container";
            container.className = "mt-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm";

            container.innerHTML = `
            <div class="mb-2">
                <div class="text-sm font-semibold text-slate-900">Live prediction repeatability table</div>
                <div class="text-xs text-slate-600">
                Each row is one click of "Run live model prediction" using the current ROI sample buffer.
                </div>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full border-collapse text-xs">
                <thead>
                    <tr class="border-b border-slate-200 text-left text-slate-500">
                    <th class="py-2 pr-3">Run</th>
                    <th class="py-2 pr-3">Model HR</th>
                    <th class="py-2 pr-3">Spectral</th>
                    <th class="py-2 pr-3">Diff</th>
                    <th class="py-2 pr-3">GREEN SQI</th>
                    <th class="py-2 pr-3">POS SQI</th>
                    <th class="py-2 pr-3">CHROM SQI</th>
                    <th class="py-2 pr-3">Used s</th>
                    <th class="py-2 pr-3">Samples</th>
                    <th class="py-2 pr-3">FPS</th>
                    </tr>
                </thead>
                <tbody id="live-model-repeatability-table-body"></tbody>
                </table>
            </div>
            `;

            outputEl.insertAdjacentElement("afterend", container);

            return container;
        }

        function renderRepeatabilityTable() {
            ensureRepeatabilityTableExists();

            const tableBody = document.getElementById("live-model-repeatability-table-body");

            if (!tableBody) {
            return;
            }

            tableBody.innerHTML = "";

            for (const run of window.livePredictionRuns) {
            const row = document.createElement("tr");
            row.className = "border-b border-slate-100 text-slate-800";

            row.innerHTML = `
                <td class="py-2 pr-3">${run.runIndex}</td>
                <td class="py-2 pr-3 font-medium">${formatBpm(run.modelHr)}</td>
                <td class="py-2 pr-3">${formatBpm(run.consensus)}</td>
                <td class="py-2 pr-3">${formatSignedBpm(run.difference)}</td>
                <td class="py-2 pr-3">${formatNumber(run.greenSqi, 3)} / ${run.greenStatus}</td>
                <td class="py-2 pr-3">${formatNumber(run.posSqi, 3)} / ${run.posStatus}</td>
                <td class="py-2 pr-3">${formatNumber(run.chromSqi, 3)} / ${run.chromStatus}</td>
                <td class="py-2 pr-3">${formatNumber(run.usedDurationS, 2)}</td>
                <td class="py-2 pr-3">${run.usedSamples ?? "none"}</td>
                <td class="py-2 pr-3">${formatNumber(run.sourceFps, 2)}</td>
            `;

            tableBody.appendChild(row);
            }
        }

        function addPredictionRun(summary) {
            const runIndex = window.livePredictionRuns.length + 1;

            window.livePredictionRuns.push({
            runIndex: runIndex,
            modelHr: summary.modelHr,
            consensus: summary.consensus,
            difference: summary.difference,
            greenBpm: summary.greenBpm,
            posBpm: summary.posBpm,
            chromBpm: summary.chromBpm,
            greenSqi: summary.greenSqi,
            posSqi: summary.posSqi,
            chromSqi: summary.chromSqi,
            greenStatus: summary.greenStatus,
            posStatus: summary.posStatus,
            chromStatus: summary.chromStatus,
            originalDurationS: summary.originalDurationS,
            usedDurationS: summary.usedDurationS,
            usedSamples: summary.usedSamples,
            sourceFps: summary.sourceFps
            });

            renderRepeatabilityTable();
        }

        if (roiSamples.length < 20) {
            statusEl.innerText = "Collect at least 20 ROI samples before live model prediction.";
            return;
        }

        runButton.disabled = true;
        runButton.innerText = "Predicting...";

        try {
            const response = await fetch("/api/predict-live-roi-series", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                samples: roiSamples
            })
            });

            const data = await parseJsonResponseEvenOnError(response);
            const summary = summarizePrediction(data);

            if (modelHrEl) {
            modelHrEl.innerText = formatBpm(summary.modelHr);
            }

            if (spectralConsensusEl) {
            spectralConsensusEl.innerText = formatBpm(summary.consensus);
            }

            if (modelDifferenceEl) {
            modelDifferenceEl.innerText = formatSignedBpm(summary.difference);
            }

            outputEl.innerText = summary.text;

            addPredictionRun(summary);

            statusEl.innerText =
            `Live model prediction completed. Model=${formatBpm(summary.modelHr)}, ` +
            `spectral consensus=${formatBpm(summary.consensus)}.`;
        } catch (error) {
            outputEl.innerText = `Live model prediction failed: ${error}`;
            statusEl.innerText = `Live model prediction failed: ${error}`;

            if (modelHrEl) {
            modelHrEl.innerText = "Prediction failed";
            }

            if (spectralConsensusEl) {
            spectralConsensusEl.innerText = "Prediction failed";
            }

            if (modelDifferenceEl) {
            modelDifferenceEl.innerText = "Prediction failed";
            }
        } finally {
            runButton.disabled = false;
            runButton.innerText = "Run live model prediction";
        }
        }
    document.addEventListener("DOMContentLoaded", () => {
        const refreshButton = document.getElementById("refresh-api-button");
        const startCameraButton = document.getElementById("start-camera-button");
        const captureFrameButton = document.getElementById("capture-frame-button");
        const sendFrameButton = document.getElementById("send-frame-button");
        const detectFaceButton = document.getElementById("detect-face-button");
        const stopCameraButton = document.getElementById("stop-camera-button");
        const startRoiSamplingButton = document.getElementById("start-roi-sampling-button");
        const stopRoiSamplingButton = document.getElementById("stop-roi-sampling-button");
        const clearRoiSamplesButton = document.getElementById("clear-roi-samples-button");
        const analyzeRoiSeriesButton = document.getElementById("analyze-roi-series-button");
        const runLiveModelButton = document.getElementById("run-live-model-button");

        drawAllRoiPlots();

        if (refreshButton) {
            refreshButton.addEventListener("click", refreshSyntheticResult);
        }

        if (startCameraButton) {
            startCameraButton.addEventListener("click", startCameraPreview);
        }

        if (captureFrameButton) {
            captureFrameButton.addEventListener("click", captureOneFrame);
        }

        if (sendFrameButton) {
            sendFrameButton.addEventListener("click", sendCapturedFrameToBackend);
        }

        if (detectFaceButton) {
            detectFaceButton.addEventListener("click", detectFaceInBackend);
        }

        if (stopCameraButton) {
            stopCameraButton.addEventListener("click", stopCameraPreview);
        }

        if (startRoiSamplingButton) {
            startRoiSamplingButton.addEventListener("click", startRoiSampling);
        }

        if (stopRoiSamplingButton) {
            stopRoiSamplingButton.addEventListener("click", stopRoiSampling);
        }

        if (clearRoiSamplesButton) {
            clearRoiSamplesButton.addEventListener("click", clearRoiSamples);
        }

        if (analyzeRoiSeriesButton) {
            analyzeRoiSeriesButton.addEventListener("click", analyzeRoiSeriesInBackend);
        }

        if (runLiveModelButton) {
            runLiveModelButton.addEventListener("click", runLiveModelPredictionInBackend);
        }
        });
    """.strip()

    return Script(NotStr(script))

def camera_preview_card() -> FT:
    """
    Render the live camera-based HR estimation interface.

    The component contains the main demo panel and a collapsible diagnostics
    section. The main panel exposes camera controls, ROI sampling controls,
    the primary spectral HR estimate, the experimental CRVSE model estimate,
    and the pulse waveform canvas. The diagnostics section keeps development
    tools such as ROI overlays, candidate signal summaries, trace plots,
    repeatability results, and raw backend JSON responses.

    Returns
    -------
    FT
        FastHTML component tree for the live HR demo card.

    Notes
    -----
    This component only defines the page structure. Camera access, ROI sampling,
    backend requests, waveform drawing, and model prediction updates are handled
    by the JavaScript returned from ``live_demo_script()``.
    """
    button_primary = (
        "mt-4 rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium "
        "text-white shadow-sm hover:bg-slate-700"
    )

    button_secondary = (
        "mt-4 ml-2 rounded-lg border border-slate-300 bg-white px-4 py-2 "
        "text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50"
    )

    return Div(
        Div(
            H2("Live rPPG HR Demo", cls="text-2xl font-bold text-slate-900"),
            P(
                "Browser camera → face ROI sampling → rPPG signal extraction → "
                "spectral HR estimate with experimental CRVSE model comparison. "
                "Frames are processed in memory and are not stored.",
                cls="mt-2 text-sm text-slate-600",
            ),
            cls="mb-5",
        ),

        Div(
            Div(
                H3("Camera", cls="mb-3 text-lg font-semibold text-slate-900"),
                Video(
                    id="camera-video",
                    autoplay=True,
                    muted=True,
                    playsinline=True,
                    cls="w-full rounded-xl border border-slate-200 bg-black shadow-sm",
                ),
                Div(
                    Button(
                        "Start camera",
                        id="start-camera-button",
                        cls=button_primary,
                    ),
                    Button(
                        "Stop camera",
                        id="stop-camera-button",
                        cls=button_secondary,
                    ),
                    cls="flex flex-wrap items-center gap-0",
                ),
                Div(
                    Button(
                        "Start ROI sampling",
                        id="start-roi-sampling-button",
                        cls=(
                            "mt-3 rounded-lg bg-emerald-700 px-4 py-2 text-sm font-medium "
                            "text-white shadow-sm hover:bg-emerald-600"
                        ),
                    ),
                    Button(
                        "Stop ROI sampling",
                        id="stop-roi-sampling-button",
                        cls=button_secondary,
                    ),
                    Button(
                        "Clear ROI samples",
                        id="clear-roi-samples-button",
                        cls=button_secondary,
                    ),
                    cls="flex flex-wrap items-center gap-0",
                ),
                Div(
                    Button(
                        "Analyze ROI series",
                        id="analyze-roi-series-button",
                        cls=(
                            "mt-3 rounded-lg bg-indigo-700 px-4 py-2 text-sm font-medium "
                            "text-white shadow-sm hover:bg-indigo-600"
                        ),
                    ),
                    Button(
                        "Run live model prediction",
                        id="run-live-model-button",
                        cls=(
                            "mt-3 ml-2 rounded-lg bg-rose-700 px-4 py-2 text-sm font-medium "
                            "text-white shadow-sm hover:bg-rose-600"
                        ),
                    ),
                    cls="flex flex-wrap items-center gap-0",
                ),
                Div(
                    "Camera not started.",
                    id="camera-status",
                    cls=(
                        "mt-4 rounded-xl border border-slate-200 bg-white p-4 "
                        "text-sm text-slate-700 shadow-sm"
                    ),
                ),
                cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
            ),

            Div(
                H3("Pulse waveform", cls="mb-3 text-lg font-semibold text-slate-900"),
                P(
                    "Main demo waveform slot. Next step will wire this to a blue, "
                    "medical-style live rPPG pulse trace.",
                    cls="mb-2 text-sm text-slate-600",
                ),
                Canvas(
                    id="main-pulse-wave-canvas",
                    width="900",
                    height="280",
                    cls="w-full rounded-xl border border-slate-200 bg-white shadow-sm",
                ),
                Div(
                    Div(
                        Div(
                            "Estimated HR",
                            cls="text-xs font-semibold uppercase tracking-wide text-slate-500",
                        ),
                        Div(
                            "Not analyzed yet",
                            id="spectral-consensus-summary",
                            cls="mt-1 text-3xl font-bold text-slate-900",
                        ),
                        Div(
                            "Primary estimate: spectral consensus",
                            cls="mt-1 text-xs text-slate-500",
                        ),
                        cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
                    ),
                    Div(
                        Div(
                            "Model Estimated HR",
                            cls="text-xs font-semibold uppercase tracking-wide text-slate-500",
                        ),
                        Div(
                            "Not predicted yet",
                            id="live-model-hr-summary",
                            cls="mt-1 text-3xl font-bold text-slate-900",
                        ),
                        Div(
                            "Experimental CRVSE PhysFormer output",
                            cls="mt-1 text-xs text-rose-700",
                        ),
                        cls="rounded-xl border border-rose-100 bg-white p-4 shadow-sm",
                    ),
                    Div(
                        Div(
                            "Model - spectral",
                            cls="text-xs font-semibold uppercase tracking-wide text-slate-500",
                        ),
                        Div(
                            "Not predicted yet",
                            id="model-spectral-difference-summary",
                            cls="mt-1 text-3xl font-bold text-slate-900",
                        ),
                        Div(
                            "Agreement diagnostic",
                            cls="mt-1 text-xs text-slate-500",
                        ),
                        cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
                    ),
                    cls="mt-4 grid gap-3 md:grid-cols-3",
                ),
                cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
            ),
            cls="grid gap-5 lg:grid-cols-2",
        ),

        Details(
            Summary(
                "Advanced diagnostics",
                cls=(
                    "mt-6 cursor-pointer rounded-xl border border-slate-200 bg-white "
                    "p-4 text-sm font-semibold text-slate-800 shadow-sm"
                ),
            ),
            Div(
                Div(
                    H3("Frame capture and ROI overlay", cls="mb-2 text-lg font-semibold"),
                    P(
                        "Use this section to inspect face detection, ROI placement, "
                        "and backend frame decoding. This is debug UI, not main demo UI.",
                        cls="mb-3 text-sm text-slate-600",
                    ),
                    Div(
                        Button(
                            "Capture one frame",
                            id="capture-frame-button",
                            cls=(
                                "rounded-lg border border-slate-300 bg-white px-4 py-2 "
                                "text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50"
                            ),
                        ),
                        Button(
                            "Send frame to backend",
                            id="send-frame-button",
                            cls=(
                                "ml-2 rounded-lg border border-slate-300 bg-white px-4 py-2 "
                                "text-sm font-medium text-slate-700 shadow-sm hover:bg-slate-50"
                            ),
                        ),
                        Button(
                            "Detect face + draw ROIs",
                            id="detect-face-button",
                            cls=(
                                "ml-2 rounded-lg bg-blue-700 px-4 py-2 text-sm font-medium "
                                "text-white shadow-sm hover:bg-blue-600"
                            ),
                        ),
                        cls="mb-3 flex flex-wrap items-center gap-y-2",
                    ),
                    Canvas(
                        id="snapshot-canvas",
                        width="320",
                        height="240",
                        cls=(
                            "w-full max-w-sm rounded-xl border border-slate-200 "
                            "bg-white shadow-sm"
                        ),
                    ),
                    cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
                ),

                Div(
                    H3("Candidate rPPG signal summary", cls="mb-2 text-lg font-semibold"),
                    P(
                        "Backend GREEN / POS / CHROM spectral sanity check. "
                        "Main HR should use spectral consensus when channels agree.",
                        cls="mb-3 text-sm text-slate-600",
                    ),
                    Div(
                        Div(
                            Div(
                                "GREEN",
                                cls="text-xs font-semibold uppercase tracking-wide text-slate-500",
                            ),
                            Div(
                                "Not analyzed yet",
                                id="green-signal-summary",
                                cls="mt-1 text-sm font-medium text-slate-800",
                            ),
                            cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
                        ),
                        Div(
                            Div(
                                "POS",
                                cls="text-xs font-semibold uppercase tracking-wide text-slate-500",
                            ),
                            Div(
                                "Not analyzed yet",
                                id="pos-signal-summary",
                                cls="mt-1 text-sm font-medium text-slate-800",
                            ),
                            cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
                        ),
                        Div(
                            Div(
                                "CHROM",
                                cls="text-xs font-semibold uppercase tracking-wide text-slate-500",
                            ),
                            Div(
                                "Not analyzed yet",
                                id="chrom-signal-summary",
                                cls="mt-1 text-sm font-medium text-slate-800",
                            ),
                            cls="rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
                        ),
                        cls="grid gap-3 md:grid-cols-3",
                    ),
                    cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
                ),

                Div(
                    H3("ROI sampling summary", cls="mb-2 text-lg font-semibold"),
                    Pre(
                        "No ROI samples collected yet.",
                        id="roi-sampling-summary",
                        cls=(
                            "max-h-96 overflow-x-auto overflow-y-auto rounded-xl border "
                            "border-slate-200 bg-white p-4 text-xs text-slate-800 shadow-sm"
                        ),
                    ),
                    cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
                ),

                Div(
                    H3("Raw ROI green traces", cls="mb-2 text-lg font-semibold"),
                    P(
                        "Raw green-channel means from each ROI over time.",
                        cls="mb-2 text-sm text-slate-600",
                    ),
                    Canvas(
                        id="roi-green-trace-canvas",
                        width="900",
                        height="280",
                        cls="w-full rounded-xl border border-slate-200 bg-white shadow-sm",
                    ),
                    cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
                ),

                Div(
                    H3("Normalized ROI green traces", cls="mb-2 text-lg font-semibold"),
                    P(
                        "Z-score normalized ROI green traces for signal-shape inspection.",
                        cls="mb-2 text-sm text-slate-600",
                    ),
                    Canvas(
                        id="roi-green-normalized-trace-canvas",
                        width="900",
                        height="280",
                        cls="w-full rounded-xl border border-slate-200 bg-white shadow-sm",
                    ),
                    cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
                ),

                Div(
                    H3("Experimental model prediction details", cls="mb-2 text-lg font-semibold"),
                    Pre(
                        "No live model prediction run yet.",
                        id="live-model-prediction-output",
                        cls=(
                            "max-h-96 overflow-x-auto overflow-y-auto rounded-xl border "
                            "border-slate-200 bg-white p-4 text-xs text-slate-800 shadow-sm"
                        ),
                    ),
                    cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
                ),

                Div(
                    H3("Backend rPPG signal analysis JSON", cls="mb-2 text-lg font-semibold"),
                    Pre(
                        "No ROI series analyzed yet.",
                        id="roi-series-analysis-output",
                        cls=(
                            "max-h-96 overflow-x-auto overflow-y-auto rounded-xl border "
                            "border-slate-200 bg-white p-4 text-xs text-slate-800 shadow-sm"
                        ),
                    ),
                    cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
                ),

                Div(
                    H3("Backend frame debug response", cls="mb-2 text-lg font-semibold"),
                    Pre(
                        "No frame sent to backend yet.",
                        id="backend-frame-debug",
                        cls=(
                            "max-h-96 overflow-x-auto overflow-y-auto rounded-xl border "
                            "border-slate-200 bg-white p-4 text-xs text-slate-800 shadow-sm"
                        ),
                    ),
                    cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
                ),

                Div(
                    H3("Backend face / ROI debug response", cls="mb-2 text-lg font-semibold"),
                    Pre(
                        "No face detection request sent yet.",
                        id="backend-face-debug",
                        cls=(
                            "max-h-96 overflow-x-auto overflow-y-auto rounded-xl border "
                            "border-slate-200 bg-white p-4 text-xs text-slate-800 shadow-sm"
                        ),
                    ),
                    cls="rounded-2xl border border-slate-200 bg-slate-50 p-5 shadow-sm",
                ),

                cls="mt-3 grid gap-5",
            ),
        ),

        cls="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm",
    )

def get_model_input_fps_from_bundle(
    bundle,
    default_fps: float = 30.0,
) -> float:
    """
    Resolve model input FPS from the loaded model specification.

    Parameters
    ----------
    bundle:
        Loaded model bundle.

    default_fps:
        Fallback FPS if the config does not expose a clear sampling-rate key.

    Returns
    -------
    float
        Model input sampling rate.

    Why this exists:
        Different versions of model_specs.yaml may name the sampling rate
        differently. The synthetic demo already works, but this live route should
        not assume one exact key such as input.sampling_rate_hz.
    """

    model_spec = bundle.model_spec

    candidate_paths = [
        ("input", "sampling_rate_hz"),
        ("input", "fps"),
        ("input", "target_fps"),
        ("preprocessing", "sampling_rate_hz"),
        ("preprocessing", "fps"),
        ("preprocessing", "target_fps"),
        ("data", "sampling_rate_hz"),
        ("data", "fps"),
        ("data", "target_fps"),
    ]

    for section_name, key_name in candidate_paths:
        section = model_spec.get(section_name, {})

        if not isinstance(section, dict):
            continue

        value = section.get(key_name)

        if value is None:
            continue

        return float(value)

    return float(default_fps)

def make_live_roi_model_prediction_payload(payload: dict) -> dict:
    """
    Build model input from browser ROI samples and run experimental live prediction.

    Parameters
    ----------
    payload:
        Browser-collected numeric ROI RGB samples.

    Returns
    -------
    dict
        JSON-safe live model prediction payload.

    Privacy:
        This function receives numeric ROI RGB summaries only.
        It does not receive or store image frames.

    Physiology:
        The model receives candidate rPPG channels that may contain pulse-related
        color variation.

    Signal:
        ROI RGB samples are converted into POS / CHROM / GREEN candidate signals,
        cropped to the latest model-duration window, resampled to the model
        target length, and passed through the same prediction helper used by
        the synthetic demo.

    Limitation:
        This is experimental because the live sampler currently runs around
        10 Hz and is resampled to the model's 240-sample input contract.
    """

    input_spec = MODEL_BUNDLE.model_spec["input"]

    target_frames = int(input_spec["target_frames"])
    window_seconds = float(input_spec["window_seconds"])

    model_input_result = build_model_input_from_roi_series_payload(
        payload=payload,
        target_length=target_frames,
        window_seconds=window_seconds,
    )

    if model_input_result.get("status") != "ok":
        return model_input_result

    model_input_np = model_input_result["_model_input"]

    # build_model_input_from_roi_series_payload returns shape:
    #   (1, 3, target_frames)
    #
    # Channel order is:
    #   0 = pos
    #   1 = chrom
    #   2 = green
    #
    # predict_hr_from_rppg_window expects a dictionary of named channels,
    # same as the synthetic demo path.
    signals = {
        "pos": model_input_np[0, 0, :],
        "chrom": model_input_np[0, 1, :],
        "green": model_input_np[0, 2, :],
    }

    # Model spec says:
    #   target_frames = 240
    #   window_seconds = 8.0
    #
    # Therefore:
    #   fps = 240 / 8 = 30 Hz
    fps = float(target_frames) / float(window_seconds)

    prediction_payload = predict_hr_from_rppg_window(
        signals=signals,
        fps=fps,
        bundle=MODEL_BUNDLE,
    )

    classical_analysis = model_input_result["classical_analysis"]

    return {
        "status": "ok",
        "message": "Experimental live ROI model prediction completed.",
        "model_prediction": prediction_payload,
        "model_input": {
            "input_shape": model_input_result["input_shape"],
            "channel_order": model_input_result["channel_order"],
            "target_length": model_input_result["target_length"],
            "source_sample_count": model_input_result["sample_count"],
            "source_duration_s": model_input_result["duration_s"],
            "source_estimated_fps": model_input_result["estimated_fps"],
            "model_target_frames": int(target_frames),
            "model_window_seconds": float(window_seconds),
            "model_assumed_fps_after_resampling": float(fps),
            "window_metadata": model_input_result["window_metadata"],
        },
        "classical_spectral_summary": {
            "green": classical_analysis["signals"]["green"]["spectral"],
            "pos": classical_analysis["signals"]["pos"]["spectral"],
            "chrom": classical_analysis["signals"]["chrom"]["spectral"],
        },
        "notes": [
            "This is experimental live model inference from browser-collected ROI RGB samples.",
            "Input was cropped to the latest model-duration window before resampling.",
            "Input was resampled to the model contract of 3 channels x target_frames samples.",
            "The model sees the resampled window as target_frames over model_window_seconds.",
            "This is not a medical measurement.",
        ],
    }

def make_json_safe_for_api(value):
    """
    Convert common Python / NumPy / Torch / dataclass objects into JSON-safe values.

    Parameters
    ----------
    value:
        Any Python object that may appear in an API response.

    Returns
    -------
    Any
        JSON-serializable value.

    Why this exists:
        Model prediction helpers may return dataclasses such as PredictionResult,
        NumPy values, tensors, arrays, or nested dictionaries containing those
        objects. Starlette JSONResponse cannot serialize those directly.

    Limitation:
        Unknown custom objects are converted to strings as a last-resort fallback.
    """

    from dataclasses import asdict, is_dataclass
    from pathlib import Path

    import numpy as np

    try:
        import torch
    except ImportError:
        torch = None

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    if is_dataclass(value):
        return make_json_safe_for_api(asdict(value))

    if hasattr(value, "model_dump"):
        return make_json_safe_for_api(value.model_dump())

    if hasattr(value, "__dict__"):
        return make_json_safe_for_api(vars(value))

    if isinstance(value, dict):
        return {
            str(key): make_json_safe_for_api(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe_for_api(item)
            for item in value
        ]

    return str(value)

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

@rt("/api/synthetic-result")
def synthetic_result_api(request: Request):
    """
    Return synthetic inference result as JSON.

    This is the first real backend endpoint.

    It proves:
        Browser/client can request inference-like data from the backend.
        The serialized PredictionResult is JSON-safe.
        Later live endpoints can reuse the same response structure.

    Query parameters
    ----------------
    hr_bpm:
        Optional synthetic HR value. Default 72.

    noise_std:
        Optional synthetic noise level. Default 0.05.

    Limitation:
        This endpoint uses synthetic POS/CHROM/GREEN signals.
        It does not process webcam frames.
    """
    query = request.query_params
    synthetic_hr_bpm = float(query.get("hr_bpm", 72.0))
    noise_std = float(query.get("noise_std", 0.05))
    # Change the seed on each request so the refresh button produces slightly
    # different synthetic noise and therefore slightly different model outputs.
    seed = int(time.time() * 1000) % 1_000_000
    payload = make_synthetic_demo_payload(
        synthetic_hr_bpm=synthetic_hr_bpm,
        noise_std=noise_std,
        seed=seed,
    )
    return JSONResponse(payload["result"])


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

    Privacy:
        This route receives numeric ROI RGB summaries only.
        It does not receive or store image frames.

    Limitation:
        This is an experimental live demo route. The current browser sampler
        collects ROI summaries around 10 Hz and resamples the candidate signals
        to the model input length.
    """

    try:
        payload = await request.json()

        result = make_live_roi_model_prediction_payload(
            payload=payload,
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