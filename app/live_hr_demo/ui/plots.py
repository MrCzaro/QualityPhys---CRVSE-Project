"""
Simple SVG plotting helpers for the live HR demo.

Why SVG first:
    It is easy to generate from Python.
    It does not require JavaScript chart libraries.
    It is good enough for the first FastHTML visual checkpoint.

Physiology:
    The pulse waveform represents optical blood-volume modulation, not ECG.

Signal:
    The waveform plot shows rPPG amplitude over time.
    The spectrum plot shows power across frequency and the dominant cardiac peak.

Limitation:
    These are simple static plots. For live real-time updates, we may later switch
    to Canvas or a JavaScript plotting library.
"""
from __future__ import annotations
import html
from dataclasses import dataclass
import numpy as np


@dataclass
class SvgPlotConfig:
    """
    Basic SVG plot configuration.
    """
    width: int = 720
    height: int = 220
    padding_left: int = 48
    padding_right: int = 20
    padding_top: int = 28
    padding_bottom: int = 38


def _safe_text(text: str) -> str:
    """
    Escape text for SVG/HTML.
    """
    return html.escape(str(text))


def _scale_values(values: np.ndarray, out_min: float, out_max: float) -> np.ndarray:
    """
    Scale numeric values into a display range.
    """

    values = np.asarray(values, dtype=np.float32)
    v_min = float(np.nanmin(values))
    v_max = float(np.nanmax(values))

    if not np.isfinite(v_min) or not np.isfinite(v_max):
        return np.full_like(values, fill_value=(out_min + out_max) / 2.0)
    if abs(v_max - v_min) < 1e-8:
        return np.full_like(values, fill_value=(out_min + out_max) / 2.0)

    scaled = (values - v_min) / (v_max - v_min)
    return out_min + scaled * (out_max - out_min)


def _polyline_points(x: np.ndarray, y: np.ndarray, config: SvgPlotConfig) -> str:
    """
    Convert x/y arrays into SVG polyline points.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}.")
    if len(x) < 2:
        raise ValueError("Need at least two points to draw a polyline.")

    plot_x_min = config.padding_left
    plot_x_max = config.width - config.padding_right
    plot_y_min = config.padding_top
    plot_y_max = config.height - config.padding_bottom
    sx = _scale_values(x, plot_x_min, plot_x_max)
    # SVG y-axis points downward, so invert the signal scale.
    sy = _scale_values(y, plot_y_max, plot_y_min)
    points = [f"{float(px):.2f},{float(py):.2f}" for px, py in zip(sx, sy)]
    return " ".join(points)


def render_waveform_svg(
    time_s: np.ndarray,
    signal: np.ndarray,
    title: str = "rPPG waveform",
    x_label: str = "Time [s]",
    y_label: str = "Normalized amplitude",
    config: SvgPlotConfig | None = None,
) -> str:
    """
    Render a simple time-domain waveform SVG.

    Parameters
    ----------
    time_s:
        Time axis in seconds.

    signal:
        1D signal values.

    title:
        Plot title.

    x_label:
        X-axis label.

    y_label:
        Y-axis label.

    config:
        Optional SVG plot config.

    Returns
    -------
    str
        SVG markup.
    """
    if config is None:
        config = SvgPlotConfig()

    time_s = np.asarray(time_s, dtype=np.float32)
    signal = np.asarray(signal, dtype=np.float32)
    points = _polyline_points(time_s, signal, config)

    x0 = config.padding_left
    x1 = config.width - config.padding_right
    y0 = config.padding_top
    y1 = config.height - config.padding_bottom

    t_min = float(np.nanmin(time_s))
    t_max = float(np.nanmax(time_s))

    svg = f"""
    <svg viewBox="0 0 {config.width} {config.height}" role="img" aria-label="{_safe_text(title)}" class="w-full">
    <rect x="0" y="0" width="{config.width}" height="{config.height}" rx="14" fill="white"></rect>

    <text x="{x0}" y="20" font-size="14" font-weight="700" fill="#0f172a">{_safe_text(title)}</text>

    <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#cbd5e1" stroke-width="1"></line>
    <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#cbd5e1" stroke-width="1"></line>

    <line x1="{x0}" y1="{(y0 + y1) / 2:.2f}" x2="{x1}" y2="{(y0 + y1) / 2:.2f}" stroke="#e2e8f0" stroke-width="1" stroke-dasharray="4 4"></line>

    <polyline points="{points}" fill="none" stroke="#0f172a" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"></polyline>

    <text x="{x0}" y="{config.height - 12}" font-size="11" fill="#64748b">{t_min:.1f}s</text>
    <text x="{x1 - 34}" y="{config.height - 12}" font-size="11" fill="#64748b">{t_max:.1f}s</text>
    <text x="{(x0 + x1) / 2 - 24:.2f}" y="{config.height - 12}" font-size="11" fill="#64748b">{_safe_text(x_label)}</text>

    <text x="12" y="{(y0 + y1) / 2:.2f}" font-size="11" fill="#64748b" transform="rotate(-90 12 {(y0 + y1) / 2:.2f})">{_safe_text(y_label)}</text>
    </svg>
    """.strip()

    return svg


def render_spectrum_svg(
    freqs_hz: np.ndarray,
    power: np.ndarray,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    dominant_freq_hz: float | None = None,
    title: str = "Power spectrum",
    config: SvgPlotConfig | None = None,
) -> str:
    """
    Render a simple frequency-domain power spectrum SVG.

    Parameters
    ----------
    freqs_hz:
        Frequency axis in Hz.

    power:
        Power spectrum values.

    low_hz:
        Lower cardiac-band boundary.

    high_hz:
        Upper cardiac-band boundary.

    dominant_freq_hz:
        Optional dominant frequency marker.

    title:
        Plot title.

    config:
        Optional SVG plot config.

    Returns
    -------
    str
        SVG markup.
    """

    if config is None:
        config = SvgPlotConfig()

    freqs_hz = np.asarray(freqs_hz, dtype=np.float32)
    power = np.asarray(power, dtype=np.float32)
    display_mask = (freqs_hz >= 0.0) & (freqs_hz <= 4.0)
    display_freqs = freqs_hz[display_mask]
    display_power = power[display_mask]

    if len(display_freqs) < 2:
        raise ValueError("Not enough spectrum points to render.")

    # Compress dynamic range slightly for nicer display.
    display_power = np.log1p(display_power)
    points = _polyline_points(display_freqs, display_power, config)

    x0 = config.padding_left
    x1 = config.width - config.padding_right
    y0 = config.padding_top
    y1 = config.height - config.padding_bottom

    def freq_to_x(freq: float) -> float:
        freq_min = 0.0
        freq_max = 4.0
        clipped = min(max(freq, freq_min), freq_max)
        return x0 + (clipped - freq_min) / (freq_max - freq_min) * (x1 - x0)

    cardiac_x0 = freq_to_x(low_hz)
    cardiac_x1 = freq_to_x(high_hz)
    dominant_line = ""

    if dominant_freq_hz is not None and np.isfinite(dominant_freq_hz):
        dom_x = freq_to_x(float(dominant_freq_hz))
        dom_bpm = float(dominant_freq_hz) * 60.0
        dominant_line = f"""
        <line x1="{dom_x:.2f}" y1="{y0}" x2="{dom_x:.2f}" y2="{y1}" stroke="#dc2626" stroke-width="2" stroke-dasharray="5 4"></line>
        <text x="{dom_x + 6:.2f}" y="{y0 + 14}" font-size="11" fill="#dc2626">{dom_bpm:.1f} BPM</text>
        """.rstrip()

    svg = f"""
    <svg viewBox="0 0 {config.width} {config.height}" role="img" aria-label="{_safe_text(title)}" class="w-full">
    <rect x="0" y="0" width="{config.width}" height="{config.height}" rx="14" fill="white"></rect>

    <text x="{x0}" y="20" font-size="14" font-weight="700" fill="#0f172a">{_safe_text(title)}</text>

    <rect x="{cardiac_x0:.2f}" y="{y0}" width="{cardiac_x1 - cardiac_x0:.2f}" height="{y1 - y0:.2f}" fill="#e0f2fe" opacity="0.65"></rect>
    <text x="{cardiac_x0 + 6:.2f}" y="{y1 - 8}" font-size="11" fill="#0369a1">cardiac band {low_hz:.1f}-{high_hz:.1f} Hz</text>

    <line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#cbd5e1" stroke-width="1"></line>
    <line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#cbd5e1" stroke-width="1"></line>

    <polyline points="{points}" fill="none" stroke="#0f172a" stroke-width="2.2" stroke-linejoin="round" stroke-linecap="round"></polyline>

    {dominant_line}

    <text x="{x0}" y="{config.height - 12}" font-size="11" fill="#64748b">0 Hz</text>
    <text x="{x1 - 28}" y="{config.height - 12}" font-size="11" fill="#64748b">4 Hz</text>
    <text x="{(x0 + x1) / 2 - 42:.2f}" y="{config.height - 12}" font-size="11" fill="#64748b">Frequency [Hz]</text>

    <text x="12" y="{(y0 + y1) / 2:.2f}" font-size="11" fill="#64748b" transform="rotate(-90 12 {(y0 + y1) / 2:.2f})">log power</text>
    </svg>
    """.strip()

    return svg