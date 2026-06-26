from fasthtml.common import *


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
    by ``live_demo_script()``.
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
                "Live monitor-style rPPG waveform from facial ROI color changes. "
                "The waveform is for signal visualization; HR is estimated after backend analysis.",
                cls="mb-2 text-sm text-slate-600",
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
