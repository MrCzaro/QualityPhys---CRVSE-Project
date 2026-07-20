from fasthtml.common import *
from monsterui.all import *
from ui.result_components import (
    demo_button,
    diagnostic_card,
    main_panel_card,
    measurement_result_cards_placeholder,
    model_prediction_summary_placeholder,
    roi_analysis_summary_placeholder,
    signal_summary_cards_placeholder,
)



def demo_readiness_item(
    label: str,
    value_id: str,
    detail_id: str,
    value: str,
    detail: str,
) -> FT:
    """Render one desktop demo-readiness status row."""
    return Div(
        Div(label, cls="text-[11px] font-semibold uppercase tracking-wide text-slate-500"),
        Div(
            value,
            id=value_id,
            cls=(
                "mt-1 inline-flex rounded-full px-2.5 py-1 text-xs "
                "font-semibold text-slate-700"
            ),
            style="background: #e2e8f0; color: #334155;",
        ),
        Div(
            detail,
            id=detail_id,
            cls="mt-1 text-xs leading-snug text-slate-600",
        ),
        cls="rounded-lg border border-slate-200 bg-white p-3 shadow-sm",
    )


def demo_readiness_panel() -> FT:
    """Render a desktop-only readiness checklist for live demo operation."""
    return Div(
        Div(
            "Demo readiness",
            cls="text-xs font-semibold uppercase tracking-wide text-slate-500",
        ),
        Div(
            demo_readiness_item(
                "Browser",
                "readiness-secure-context-value",
                "readiness-secure-context-detail",
                "Checking",
                "Camera access requires HTTPS, localhost, or another secure context.",
            ),
            demo_readiness_item(
                "Camera",
                "readiness-camera-value",
                "readiness-camera-detail",
                "Not started",
                "Start the camera before measuring.",
            ),
            demo_readiness_item(
                "Samples",
                "readiness-samples-value",
                "readiness-samples-detail",
                "0 samples",
                "Collect at least 20 ROI samples before backend analysis.",
            ),
            demo_readiness_item(
                "Model FPS",
                "readiness-model-fps-value",
                "readiness-model-fps-detail",
                "Waiting",
                "Experimental model preprocessing needs at least 8 Hz source sampling.",
            ),
            demo_readiness_item(
                "Signal",
                "readiness-signal-value",
                "readiness-signal-detail",
                "Not analyzed",
                "Full-buffer spectral quality has not been computed yet.",
            ),
            demo_readiness_item(
                "Experimental model",
                "readiness-model-value",
                "readiness-model-detail",
                "Not predicted",
                "Model output is secondary; spectral HR remains primary.",
            ),
            cls="mt-3 grid gap-3 sm:grid-cols-2 xl:grid-cols-3",
        ),
        id="demo-readiness-panel",
        cls="mt-4 hidden rounded-xl border border-sky-100 bg-sky-50/60 p-4 shadow-sm md:block",
    )


def camera_preview_card() -> FT:
    """
    Render the live camera-based HR estimation interface.

    The component contains a clean main measurement panel and a collapsible
    diagnostics section. The main panel is intended for demo use, while the
    diagnostics section preserves manual controls for ROI sampling, backend
    frame checks, signal inspection, and model-prediction debugging.

    Returns
    -------
    FT
        FastHTML component tree for the live HR demo card.

    Notes
    -----
    This component only defines page structure. Camera access, sampling,
    backend requests, waveform drawing, and prediction updates are handled by
    ``live_demo_script()``.
    """

    return Div(
        Div(
            H2("Live rPPG HR Demo", cls="text-2xl font-bold text-slate-900"),
            P(
                "Camera-based research demo for heart-rate estimation from facial "
                "rPPG signals. Start the camera, then run a short measurement while "
                "holding still.",
                cls="mb-2 text-sm text-slate-600",
            ),
            cls="mb-5",
        ),

        Div(
            main_panel_card(
                "Camera",
                None,
                Video(
                    id="camera-video",
                    autoplay=True,
                    muted=True,
                    playsinline=True,
                    style="transform: scaleX(-1); transform-origin: center;",

                    cls="w-full rounded-xl border border-slate-200 bg-black shadow-sm",
                ),
                Div(
                    demo_button(
                        label="Start camera",
                        element_id="start-camera-button",
                        variant="primary",
                    ),
                    demo_button(
                        label="Start measurement",
                        element_id="start-measurement-button",
                        variant="measurement",
                    ),
                    demo_button(
                        label="Stop measurement",
                        element_id="stop-measurement-button",
                        variant="stop_measurement",
                    ),
                    demo_button(
                        label="Stop camera",
                        element_id="stop-camera-button",
                        variant="secondary",
                    ),
                    demo_button(
                        label="Clear",
                        element_id="clear-roi-samples-button",
                        variant="secondary",
                    ),
                    cls="mt-4 flex flex-wrap items-center gap-2",
                ),
                Div(
                    Div(
                        "Measurement status",
                        cls="text-xs font-semibold uppercase tracking-wide text-slate-500",
                    ),
                    Div(
                        "Ready.",
                        id="measurement-status-summary",
                        cls="mt-1 text-lg font-semibold text-slate-900",
                    ),
                    Div(
                        "Start the camera, then start a measurement while holding still.",
                        id="measurement-status-detail",
                        cls="mt-1 text-sm text-slate-600",
                    ),
                    Div(
                        Div(
                            id="measurement-progress-bar",
                            cls="h-2 rounded-full bg-emerald-600 transition-all duration-200",
                            style="width: 0%;",
                        ),
                        cls="mt-3 h-2 overflow-hidden rounded-full bg-slate-200",
                    ),
                    Div(
                        "0%",
                        id="measurement-progress-text",
                        cls="mt-1 text-xs text-slate-500",
                    ),
                    cls="mt-4 rounded-xl border border-emerald-100 bg-white p-4 shadow-sm",
                ),
                Div(
                    "Camera not started.",
                    id="camera-status",
                    cls=(
                        "mt-4 rounded-xl border border-slate-200 bg-white p-4 "
                        "text-sm text-slate-700 shadow-sm"
                    ),
                ),
                demo_readiness_panel(),
                Div(
                    Div(
                        "Measurement guidance",
                        cls="text-xs font-semibold uppercase tracking-wide text-slate-500",
                    ),
                    Ul(
                        Li("Use steady frontal face position.", cls="mb-1"),
                        Li("Avoid talking or large head movement.", cls="mb-1"),
                        Li(
                            "Wait for the signal to stabilize before trusting the estimate.",
                            cls="mb-1",
                        ),
                        cls="mt-2 list-disc pl-5 text-sm text-slate-600",
                    ),
                    cls="mt-4 rounded-xl border border-slate-200 bg-white p-4 shadow-sm",
                ),
            ),

            main_panel_card(
                "Pulse waveform",
                "Live monitor-style waveform from facial ROI color changes. "
                "The waveform is for signal visualization; HR is estimated after "
                "backend analysis.",
                Canvas(
                    id="main-pulse-wave-canvas",
                    width="900",
                    height="280",
                    cls="w-full rounded-xl border border-slate-200 bg-white shadow-sm",
                ),
                Div(
                    measurement_result_cards_placeholder(),
                    id="measurement-result-cards-container",
                ),
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
                diagnostic_card(
                    "Manual acquisition controls",
                    "Manual controls for testing the same steps used by the main "
                    "measurement flow. Keep these for development and debugging.",
                    Div(
                        "Start the camera to enable manual ROI sampling.",
                        id="advanced-manual-controls-status",
                        cls=(
                            "mb-3 rounded-lg border border-slate-200 bg-slate-50 "
                            "px-3 py-2 text-xs text-slate-600"
                        ),
                    ),
                    Div(
                        demo_button(
                            label="Start ROI sampling",
                            element_id="start-roi-sampling-button",
                            variant="measurement",
                        ),
                        demo_button(
                            label="Stop ROI sampling",
                            element_id="stop-roi-sampling-button",
                            variant="diagnostic",
                        ),
                        demo_button(
                            label="Analyze ROI series",
                            element_id="analyze-roi-series-button",
                            variant="analysis",
                        ),
                        demo_button(
                            label="Run live model prediction",
                            element_id="run-live-model-button",
                            variant="model",
                        ),
                        cls="flex flex-wrap items-center gap-2",
                    ),
                ),

                diagnostic_card(
                    "Frame capture and ROI overlay",
                    "Inspect face detection, ROI placement, and backend frame decoding. "
                    "This is debug UI, not main demo UI.",
                    Div(
                        "Start the camera to enable frame capture.",
                        id="advanced-frame-controls-status",
                        cls=(
                            "mb-3 rounded-lg border border-slate-200 bg-slate-50 "
                            "px-3 py-2 text-xs text-slate-600"
                        ),
                    ),
                    Div(
                        demo_button(
                            label="Capture one frame",
                            element_id="capture-frame-button",
                            variant="diagnostic",
                        ),
                        demo_button(
                            label="Send frame to backend",
                            element_id="send-frame-button",
                            variant="diagnostic",
                        ),
                        demo_button(
                            label="Detect face + draw ROIs",
                            element_id="detect-face-button",
                            variant="face",
                        ),
                        cls="mb-3 flex flex-wrap items-center gap-2",
                    ),
                    Canvas(
                        id="snapshot-canvas",
                        width="320",
                        height="240",
                        style="transform: scaleX(-1); transform-origin: center;",
                        cls=(
                            "w-full max-w-sm rounded-xl border border-slate-200 "
                            "bg-white shadow-sm"
                        ),
                    ),
                ),

                diagnostic_card(
                    "Candidate rPPG signal summary",
                    "Backend GREEN / POS / CHROM spectral sanity check. "
                    "Main HR should use spectral consensus when channels agree.",
                    Div(
                        signal_summary_cards_placeholder(),
                        id="signal-summary-cards-container",
                    ),
                ),

                diagnostic_card(
                    "ROI sampling summary",
                    None,
                    Pre(
                        "No ROI samples collected yet.",
                        id="roi-sampling-summary",
                        cls=(
                            "max-h-96 overflow-x-auto overflow-y-auto rounded-xl border "
                            "border-slate-200 bg-white p-4 text-xs text-slate-800 shadow-sm"
                        ),
                    ),
                ),

                diagnostic_card(
                    "Raw ROI green traces",
                    "Raw green-channel means from each ROI over time.",
                    Canvas(
                        id="roi-green-trace-canvas",
                        width="900",
                        height="280",
                        cls="w-full rounded-xl border border-slate-200 bg-white shadow-sm",
                    ),
                ),

                diagnostic_card(
                    "Normalized ROI green traces",
                    "Z-score normalized ROI green traces for signal-shape inspection.",
                    Canvas(
                        id="roi-green-normalized-trace-canvas",
                        width="900",
                        height="280",
                        cls="w-full rounded-xl border border-slate-200 bg-white shadow-sm",
                    ),
                ),

                diagnostic_card(
                    "Experimental model prediction details",
                    "Server-rendered summary of the CRVSE PhysFormer live prediction.",
                    Div(
                        model_prediction_summary_placeholder(),
                        id="live-model-prediction-output-container",
                    ),
                ),

                diagnostic_card(
                    "Backend rPPG signal analysis",
                    "Server-rendered summary of the analyzed ROI time series.",
                    Div(
                        roi_analysis_summary_placeholder(),
                        id="roi-series-analysis-output-container",
                    ),
                ),

                diagnostic_card(
                    "Backend frame debug response",
                    None,
                    Pre(
                        "No frame sent to backend yet.",
                        id="backend-frame-debug",
                        cls=(
                            "max-h-96 overflow-x-auto overflow-y-auto rounded-xl border "
                            "border-slate-200 bg-white p-4 text-xs text-slate-800 shadow-sm"
                        ),
                    ),
                ),

                diagnostic_card(
                    "Backend face / ROI debug response",
                    None,
                    Pre(
                        "No face detection request sent yet.",
                        id="backend-face-debug",
                        cls=(
                            "max-h-96 overflow-x-auto overflow-y-auto rounded-xl border "
                            "border-slate-200 bg-white p-4 text-xs text-slate-800 shadow-sm"
                        ),
                    ),
                ),

                cls="mt-3 grid gap-5",
            ),
        ),

        cls="mt-6 rounded-2xl border border-slate-200 bg-white p-6 shadow-sm",
    )

def mobile_demo_style() -> FT:
    """Render mobile-only presentation rules for the live demo page."""
    return Style(
        """
        @media (max-width: 767px) {
          html {
            color-scheme: light;
            background: #f8fafc !important;
          }

          body {
            background: #f8fafc !important;
            color: #0f172a !important;
          }

          main {
            padding: 1rem !important;
            background: #f8fafc !important;
          }

          #start-camera-button,
          #stop-camera-button,
          #clear-roi-samples-button,
          details {
            display: none !important;
          }

          #main-pulse-wave-canvas {
            display: none !important;
          }

          #mobile-camera-preview-frame {
            position: relative;
            width: 100%;
          }

          #camera-video {
            aspect-ratio: 4 / 3;
            object-fit: cover;
            border: 2px solid #cbd5e1 !important;
            border-radius: 1rem !important;
            background: #e2e8f0 !important;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
          }

          #mobile-camera-guidance-overlay {
            display: none;
            position: absolute;
            inset: 0;
            z-index: 5;
            align-items: center;
            justify-content: center;
            border-radius: 1rem;
            background: rgba(248, 250, 252, 0.68);
            backdrop-filter: blur(1px);
            color: #0f172a;
            padding: 1.25rem;
            text-align: center;
            pointer-events: none;
          }

          #mobile-camera-guidance-overlay[data-visible="true"] {
            display: flex;
          }

          #mobile-camera-guidance-primary {
            font-size: 1.5rem;
            line-height: 1.15;
            font-weight: 900;
            color: #0f172a;
          }

          #mobile-camera-guidance-secondary {
            margin-top: 0.5rem;
            font-size: 1rem;
            line-height: 1.35;
            font-weight: 700;
            color: #334155;
          }

          #start-measurement-button,
          #stop-measurement-button {
            width: 100%;
            min-height: 3.35rem;
            flex: 1 1 100%;
            border-radius: 9999px !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
            letter-spacing: 0 !important;
            opacity: 1 !important;
          }

          #start-measurement-button {
            background: #059669 !important;
            color: #ffffff !important;
            box-shadow: 0 10px 18px rgba(5, 150, 105, 0.22);
          }

          #start-measurement-button:disabled {
            background: #d1fae5 !important;
            color: #047857 !important;
            border: 1px solid #a7f3d0 !important;
            box-shadow: none !important;
          }

          #stop-measurement-button {
            background: #dc2626 !important;
            color: #ffffff !important;
            box-shadow: 0 10px 18px rgba(220, 38, 38, 0.18);
          }

          #stop-measurement-button:disabled {
            background: #f8fafc !important;
            color: #475569 !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: none !important;
          }

          #measurement-status-summary {
            font-size: 1.3rem !important;
            line-height: 1.25 !important;
            color: #0f172a !important;
          }

          #measurement-status-detail {
            font-size: 0.98rem !important;
            line-height: 1.45 !important;
            color: #334155 !important;
          }

          #measurement-progress-text {
            font-size: 1.05rem !important;
            font-weight: 800 !important;
            color: #047857 !important;
          }

          #measurement-progress-bar {
            background: #10b981 !important;
          }

          #camera-status {
            background: #ffffff !important;
            color: #334155 !important;
            border-color: #cbd5e1 !important;
          }

          #measurement-result-cards-container {
            scroll-margin-top: 1rem;
            color: #0f172a !important;
          }

          #measurement-result-cards-container * {
            color: #0f172a !important;
          }

          #measurement-result-cards-container > div > * {
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08) !important;
          }

          #measurement-quality-detail {
            color: #334155 !important;
          }
        }
        """
    )