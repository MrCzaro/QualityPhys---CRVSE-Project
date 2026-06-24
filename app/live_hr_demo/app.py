"""
Live HR demo app shell.

This is the first visible FastHTML app layer.

Current purpose:
    Show a synthetic backend inference result in the browser.
    Show synthetic rPPG waveform and power spectrum plots.

What this proves:
    FastHTML route works.
    Model bundle loads.
    Synthetic rPPG window runs through the full backend inference core.
    Serialized result can be displayed in HTML.
    Signal plots can be rendered in the UI.

What this does NOT do yet:
    No webcam.
    No real face video.
    No POS/CHROM/GREEN extraction from frames.
"""
from __future__ import annotations

from pathlib import Path
import sys
from fasthtml.common import *

APP_DIR = Path(__file__).resolve().parent

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from inference.serialization import prediction_result_to_dict
from inference.window_inference import predict_hr_from_rppg_window
from models.loader import load_model_bundle
from rppg.sqi import estimate_spectral_sqi
from rppg.windowing import make_synthetic_rppg_channels, zscore_1d
from ui.plots import render_spectrum_svg, render_waveform_svg


app, rt = fast_app(
    title="QualityPhys Live HR Demo",
)


MODEL_BUNDLE = load_model_bundle(device="cpu")


def make_synthetic_demo_payload() -> dict:
    """
    Generate synthetic rPPG inference payload for UI testing.

    Physiology:
        We simulate a pulse-like rhythm at 72 BPM.

    Signal:
        The synthetic POS/CHROM/GREEN traces go through the same quality and model
        inference path as future real rPPG traces.

    Limitation:
        Synthetic data is not a real webcam signal. This only tests app wiring and
        visualization.
    """
    fps = 30.0
    duration_seconds = float(MODEL_BUNDLE.model_spec["input"]["window_seconds"])
    signals = make_synthetic_rppg_channels(
        hr_bpm=72.0,
        duration_seconds=duration_seconds,
        fps=fps,
        noise_std=0.05,
        seed=42,
    )
    result = predict_hr_from_rppg_window(
        signals=signals,
        fps=fps,
        bundle=MODEL_BUNDLE,
    )
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
    """
    return Div(
        Div(
            H2("Signal diagnostics", cls="text-xl font-semibold"),
            P(
                "These plots show the synthetic POS rPPG trace and its power spectrum. "
                "Later, the same plot slots will display live camera-derived signals.",
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


@rt("/")
def index() -> FT:
    """
    Home page.
    """
    payload = make_synthetic_demo_payload()
    result = payload["result"]
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
                        "Current page uses synthetic data only. Live camera comes later.",
                        cls="mt-2 max-w-3xl text-slate-600",
                    ),
                    P(
                        "Not a medical device. Not for diagnosis or treatment decisions.",
                        cls="mt-2 font-medium text-red-700",
                    ),
                    cls="mb-8",
                ),
                result_card(result),
                plots_card(
                    waveform_svg=payload["waveform_svg"],
                    spectrum_svg=payload["spectrum_svg"],
                ),
                cls="mx-auto max-w-6xl px-6 py-10",
            ),
            cls="bg-slate-100",
        ),
    )


serve()