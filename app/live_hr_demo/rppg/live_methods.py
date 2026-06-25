"""
Live rPPG signal helpers for browser-collected ROI RGB samples.

Current purpose:
    Convert numeric ROI RGB samples into simple candidate rPPG signals:
        - GREEN
        - POS
        - CHROM

What this proves:
    Browser-collected ROI RGB samples can be analyzed by the Python backend.
    We can inspect classical rPPG candidate signals before model inference.

What this does NOT do:
    Does not process image frames.
    Does not run face detection.
    Does not run the neural model.
    Does not store frames or samples.

Physiology:
    Pulse-related blood-volume changes can create small periodic color changes
    in skin regions.

Signal:
    This module transforms ROI RGB time series into candidate pulse signals and
    estimates a simple spectral peak.

Limitation:
    This is an early live-demo implementation. POS/CHROM here are minimal,
    window-level implementations for inspection, not final production signal
    processing.
"""

from __future__ import annotations
from scipy.signal import resample
from typing import Any
import numpy as np
from rppg.sqi import estimate_spectral_sqi


def _safe_float(value: Any) -> float | None:
    """
    Convert a value to float if possible.

    Parameters
    ----------
    value:
        Input value.

    Returns
    -------
    float | None
        Float value, or None if conversion fails.
    """

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_roi_rgb_series(
    samples: list[dict[str, Any]],
    roi_names: list[str],
) -> dict[str, Any]:
    """
    Extract aligned ROI RGB arrays from browser-collected samples.

    Parameters
    ----------
    samples:
        Browser-side ROI samples. Each sample is expected to contain:
            t_s
            rois[roi_name].r/g/b

    roi_names:
        ROI names to extract.

    Returns
    -------
    dict[str, Any]
        JSON-safe extraction result plus NumPy arrays stored under private keys.

    Notes
    -----
    We keep only samples where all requested ROIs are present and numeric.
    """

    aligned_times = []
    roi_values = {roi_name: [] for roi_name in roi_names}

    dropped_count = 0

    for sample in samples:
        t_s = _safe_float(sample.get("t_s"))

        if t_s is None:
            dropped_count += 1
            continue

        rois = sample.get("rois", {})
        parsed_roi_values = {}

        sample_ok = True

        for roi_name in roi_names:
            roi = rois.get(roi_name)

            if roi is None:
                sample_ok = False
                break

            r = _safe_float(roi.get("r"))
            g = _safe_float(roi.get("g"))
            b = _safe_float(roi.get("b"))

            if r is None or g is None or b is None:
                sample_ok = False
                break

            parsed_roi_values[roi_name] = [r, g, b]

        if not sample_ok:
            dropped_count += 1
            continue

        aligned_times.append(t_s)

        for roi_name in roi_names:
            roi_values[roi_name].append(parsed_roi_values[roi_name])

    if len(aligned_times) == 0:
        return {
            "ok": False,
            "message": "No aligned ROI samples found.",
            "dropped_count": int(dropped_count),
            "_time_s": np.asarray([], dtype=np.float32),
            "_roi_rgb": {},
        }

    time_s = np.asarray(aligned_times, dtype=np.float32)

    roi_rgb = {
        roi_name: np.asarray(values, dtype=np.float32)
        for roi_name, values in roi_values.items()
    }

    return {
        "ok": True,
        "message": "Aligned ROI RGB samples extracted.",
        "sample_count": int(len(time_s)),
        "dropped_count": int(dropped_count),
        "_time_s": time_s,
        "_roi_rgb": roi_rgb,
    }


def estimate_sampling_rate_hz(time_s: np.ndarray) -> float | None:
    """
    Estimate sampling frequency from timestamps.

    Parameters
    ----------
    time_s:
        Time array in seconds.

    Returns
    -------
    float | None
        Estimated sampling frequency in Hz.
    """

    if len(time_s) < 3:
        return None

    diffs = np.diff(time_s)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return None

    median_dt = float(np.median(diffs))

    if median_dt <= 0:
        return None

    return float(1.0 / median_dt)


def combine_roi_rgb_by_mean(
    roi_rgb: dict[str, np.ndarray],
    roi_names: list[str],
) -> np.ndarray:
    """
    Combine multiple ROI RGB series by averaging them.

    Parameters
    ----------
    roi_rgb:
        Mapping from ROI name to RGB array with shape (n_samples, 3).

    roi_names:
        ROI names to include.

    Returns
    -------
    np.ndarray
        Combined RGB array with shape (n_samples, 3).

    Notes
    -----
    This is intentionally simple. Later we can weight ROIs by quality, reject bad
    ROIs, or combine POS/CHROM per ROI before averaging.
    """

    arrays = []

    for roi_name in roi_names:
        if roi_name in roi_rgb:
            arrays.append(roi_rgb[roi_name])

    if len(arrays) == 0:
        raise ValueError("No ROI RGB arrays available for combining.")

    stacked = np.stack(arrays, axis=0)

    return np.mean(stacked, axis=0)


def zscore_1d_numpy(
    signal: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Z-score normalize a 1D signal.

    Parameters
    ----------
    signal:
        1D signal.

    eps:
        Small constant to avoid division by zero.

    Returns
    -------
    np.ndarray
        Z-scored signal.
    """

    signal = np.asarray(signal, dtype=np.float32)

    mean = float(np.mean(signal))
    std = float(np.std(signal))

    if std < eps:
        return np.zeros_like(signal)

    return (signal - mean) / std


def normalize_rgb_by_channel_mean(
    rgb: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize RGB channels by their temporal channel means.

    Parameters
    ----------
    rgb:
        RGB array with shape (n_samples, 3).

    Returns
    -------
    np.ndarray
        Mean-normalized RGB array.
    """

    rgb = np.asarray(rgb, dtype=np.float32)

    channel_mean = np.mean(rgb, axis=0, keepdims=True)

    return rgb / (channel_mean + eps)


def make_green_signal(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Create simple GREEN rPPG candidate signal from RGB.

    Parameters
    ----------
    rgb:
        RGB array with shape (n_samples, 3).

    Returns
    -------
    np.ndarray
        Z-scored green-channel signal.
    """

    green = rgb[:, 1]

    return zscore_1d_numpy(green)


def make_pos_signal(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Create a minimal POS-style rPPG candidate signal.

    Parameters
    ----------
    rgb:
        RGB array with shape (n_samples, 3).

    Returns
    -------
    np.ndarray
        Z-scored POS-like signal.

    Notes
    -----
    This is a compact window-level POS approximation for live debugging.
    It is not yet a full overlap-window implementation.
    """

    rgb_norm = normalize_rgb_by_channel_mean(rgb)

    r = rgb_norm[:, 0]
    g = rgb_norm[:, 1]
    b = rgb_norm[:, 2]

    s1 = g - b
    s2 = g + b - 2.0 * r

    std_s1 = float(np.std(s1))
    std_s2 = float(np.std(s2))

    if std_s2 < 1e-8:
        pos = s1
    else:
        alpha = std_s1 / std_s2
        pos = s1 + alpha * s2

    return zscore_1d_numpy(pos)


def make_chrom_signal(
    rgb: np.ndarray,
) -> np.ndarray:
    """
    Create a minimal CHROM-style rPPG candidate signal.

    Parameters
    ----------
    rgb:
        RGB array with shape (n_samples, 3).

    Returns
    -------
    np.ndarray
        Z-scored CHROM-like signal.

    Notes
    -----
    This is a compact window-level CHROM approximation for live debugging.
    It is not yet a full overlap-window implementation.
    """

    rgb_norm = normalize_rgb_by_channel_mean(rgb)

    r = rgb_norm[:, 0]
    g = rgb_norm[:, 1]
    b = rgb_norm[:, 2]

    x = 3.0 * r - 2.0 * g
    y = 1.5 * r + g - 1.5 * b

    std_x = float(np.std(x))
    std_y = float(np.std(y))

    if std_y < 1e-8:
        chrom = x
    else:
        alpha = std_x / std_y
        chrom = x - alpha * y

    return zscore_1d_numpy(chrom)


def summarize_candidate_signal(
    signal: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> dict[str, Any]:
    """
    Summarize one candidate rPPG signal using spectral SQI.

    Parameters
    ----------
    signal:
        Candidate rPPG signal.

    fps:
        Sampling frequency in Hz.

    low_hz:
        Lower cardiac frequency bound.

    high_hz:
        Upper cardiac frequency bound.

    Returns
    -------
    dict[str, Any]
        JSON-safe signal summary.
    """

    signal = np.asarray(signal, dtype=np.float32)

    if len(signal) < 8:
        return {
            "ok": False,
            "message": "Signal too short for spectral summary.",
            "n_samples": int(len(signal)),
            "values": signal.tolist(),
            "spectral": None,
        }

    spectral = estimate_spectral_sqi(
        signal=signal,
        fps=fps,
        low_hz=low_hz,
        high_hz=high_hz,
    )

    return {
        "ok": True,
        "message": "Signal summarized.",
        "n_samples": int(len(signal)),
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal)),
        "min": float(np.min(signal)),
        "max": float(np.max(signal)),
        "values": signal.tolist(),
        "spectral": {
            "dominant_freq_hz": float(spectral.dominant_freq_hz)
            if spectral.dominant_freq_hz is not None
            else None,
            "dominant_bpm": float(spectral.dominant_bpm)
            if spectral.dominant_bpm is not None
            else None,
            "sqi": float(spectral.sqi)
            if spectral.sqi is not None
            else None,
            "status": spectral.status,
            "reason": spectral.reason,
        },
    }


def analyze_roi_series_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Analyze browser-collected ROI RGB samples.

    Parameters
    ----------
    payload:
        JSON payload containing:
            samples: list of browser ROI samples

        Optional:
            window_seconds: if provided, analyze only the latest N seconds.

    Returns
    -------
    dict[str, Any]
        JSON-safe analysis response.
    """

    samples = payload.get("samples", [])
    window_seconds = payload.get("window_seconds", None)

    if not isinstance(samples, list):
        return {
            "status": "error",
            "message": "Expected payload field 'samples' to be a list.",
        }

    window_metadata = None

    if window_seconds is not None:
        window_result = keep_latest_window_samples(
            samples=samples,
            window_seconds=float(window_seconds),
        )

        if not window_result["ok"]:
            return {
                "status": "error",
                "message": window_result["message"],
                "window_metadata": window_result,
            }

        samples = window_result["samples"]
        window_metadata = {
            "requested_window_seconds": float(window_seconds),
            "original_sample_count": window_result["original_sample_count"],
            "used_sample_count": window_result["used_sample_count"],
            "original_duration_s": window_result["original_duration_s"],
            "used_duration_s": window_result["used_duration_s"],
            "cutoff_s": window_result["cutoff_s"],
        }

    roi_names = [
        "forehead",
        "image_left_cheek",
        "image_right_cheek",
    ]

    extraction = extract_roi_rgb_series(
        samples=samples,
        roi_names=roi_names,
    )

    if not extraction["ok"]:
        return {
            "status": "error",
            "message": extraction["message"],
            "dropped_count": extraction["dropped_count"],
            "window_metadata": window_metadata,
        }

    time_s = extraction["_time_s"]
    roi_rgb = extraction["_roi_rgb"]

    fps = estimate_sampling_rate_hz(time_s)

    if fps is None:
        return {
            "status": "error",
            "message": "Could not estimate sampling rate from ROI sample timestamps.",
            "sample_count": int(len(time_s)),
            "window_metadata": window_metadata,
        }

    combined_rgb = combine_roi_rgb_by_mean(
        roi_rgb=roi_rgb,
        roi_names=roi_names,
    )

    green_signal = make_green_signal(combined_rgb)
    pos_signal = make_pos_signal(combined_rgb)
    chrom_signal = make_chrom_signal(combined_rgb)

    duration_s = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0

    return {
        "status": "ok",
        "message": "ROI time series analyzed into candidate rPPG signals.",
        "sample_count": int(len(time_s)),
        "duration_s": duration_s,
        "estimated_fps": float(fps),
        "roi_names_used": roi_names,
        "window_metadata": window_metadata,
        "time_s": time_s.tolist(),
        "signals": {
            "green": summarize_candidate_signal(
                signal=green_signal,
                fps=fps,
            ),
            "pos": summarize_candidate_signal(
                signal=pos_signal,
                fps=fps,
            ),
            "chrom": summarize_candidate_signal(
                signal=chrom_signal,
                fps=fps,
            ),
        },
        "notes": [
            "This is a classical signal extraction diagnostic, not model inference.",
            "Signals are z-scored candidate traces.",
            "Spectral HR is a rough sanity check and should not be treated as a medical measurement.",
        ],
    }

def resample_signal_to_length(
    signal: np.ndarray,
    target_length: int = 240,
) -> np.ndarray:
    """
    Resample a 1D signal to a fixed target length.

    Parameters
    ----------
    signal:
        Input 1D signal.

    target_length:
        Desired output length.

    Returns
    -------
    np.ndarray
        Resampled 1D signal with shape (target_length,).

    Why this exists:
        The trained CRVSE model expects a fixed-length 8-second input window:
            channels x time = 3 x 240

        Our live browser sampler currently produces a variable number of samples,
        usually around 100 samples over 10 seconds. For the first live demo
        experiment, we resample the candidate signals to the model contract.

    Limitation:
        Resampling does not create true missing high-frame-rate information.
        This is acceptable for an exploratory demo, not final validation.
    """

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}.")

    if len(signal) < 8:
        raise ValueError(f"Signal too short to resample: {len(signal)} samples.")

    resampled = resample(signal, target_length)

    return zscore_1d_numpy(resampled.astype(np.float32))

def keep_latest_window_samples(
    samples: list[dict[str, Any]],
    window_seconds: float,
) -> dict[str, Any]:
    """
    Keep only the latest time window from browser-collected ROI samples.

    Parameters
    ----------
    samples:
        Browser-collected ROI RGB samples. Each sample should contain t_s.

    window_seconds:
        Desired window length in seconds.

    Returns
    -------
    dict[str, Any]
        Dictionary containing filtered samples and window metadata.

    Why this exists:
        The model expects a fixed 8-second physiological window. If we collect
        10-12 seconds and compress the whole buffer into 240 samples, we distort
        the frequency content. For example, compressing 10.5 seconds into 8
        seconds can shift a 67 bpm spectral peak toward ~88 bpm.

    Limitation:
        This function assumes the browser timestamps are monotonic enough for
        local demo use.
    """

    if not isinstance(samples, list):
        return {
            "ok": False,
            "message": "Expected samples to be a list.",
            "samples": [],
            "original_sample_count": 0,
            "used_sample_count": 0,
            "original_duration_s": 0.0,
            "used_duration_s": 0.0,
        }

    valid_samples = []

    for sample in samples:
        if not isinstance(sample, dict):
            continue

        t_s = _safe_float(sample.get("t_s"))

        if t_s is None:
            continue

        sample_copy = dict(sample)
        sample_copy["t_s"] = t_s
        valid_samples.append(sample_copy)

    if len(valid_samples) == 0:
        return {
            "ok": False,
            "message": "No valid timestamped samples found.",
            "samples": [],
            "original_sample_count": len(samples),
            "used_sample_count": 0,
            "original_duration_s": 0.0,
            "used_duration_s": 0.0,
        }

    valid_samples = sorted(
        valid_samples,
        key=lambda sample: float(sample["t_s"]),
    )

    original_start_s = float(valid_samples[0]["t_s"])
    original_end_s = float(valid_samples[-1]["t_s"])
    original_duration_s = max(0.0, original_end_s - original_start_s)

    cutoff_s = original_end_s - float(window_seconds)

    filtered_samples = [
        sample
        for sample in valid_samples
        if float(sample["t_s"]) >= cutoff_s
    ]

    if len(filtered_samples) == 0:
        filtered_samples = valid_samples

    used_start_s = float(filtered_samples[0]["t_s"])
    used_end_s = float(filtered_samples[-1]["t_s"])
    used_duration_s = max(0.0, used_end_s - used_start_s)

    return {
        "ok": True,
        "message": "Latest sample window selected.",
        "samples": filtered_samples,
        "original_sample_count": int(len(valid_samples)),
        "used_sample_count": int(len(filtered_samples)),
        "original_duration_s": float(original_duration_s),
        "used_duration_s": float(used_duration_s),
        "window_seconds": float(window_seconds),
        "cutoff_s": float(cutoff_s),
    }

def build_model_input_from_roi_series_payload(
    payload: dict[str, Any],
    target_length: int = 240,
    window_seconds: float = 8.0,
) -> dict[str, Any]:
    """
    Build model-ready POS / CHROM / GREEN tensor data from ROI RGB samples.

    Parameters
    ----------
    payload:
        Browser-collected ROI RGB sample payload.

    target_length:
        Number of time samples expected by the model.

    window_seconds:
        Latest source-data window to use before resampling.

    Returns
    -------
    dict[str, Any]
        JSON-safe metadata plus private NumPy array under "_model_input".

    Model contract:
        Output model input has shape:
            (1, 3, target_length)

        Channel order:
            0 = POS
            1 = CHROM
            2 = GREEN

    Why this matters:
        The model expects an 8-second window. If the user collects 10-12 seconds
        and we resample the whole buffer to 240 samples, we compress time and
        artificially increase apparent frequency. Therefore we first crop to the
        latest 8 seconds and only then resample.

    Limitation:
        The input is still resampled from browser-collected ROI RGB summaries.
        This is not yet a true 30 fps frame-level preprocessing path.
    """

    samples = payload.get("samples", [])

    latest_window = keep_latest_window_samples(
        samples=samples,
        window_seconds=window_seconds,
    )

    if not latest_window["ok"]:
        return {
            "status": "error",
            "message": latest_window["message"],
            "window_metadata": latest_window,
        }

    cropped_payload = dict(payload)
    cropped_payload["samples"] = latest_window["samples"]
    cropped_payload["window_seconds"] = None

    analysis_payload = {
        "samples": latest_window["samples"],
    }

    analysis = analyze_roi_series_payload(analysis_payload)

    if analysis.get("status") != "ok":
        return {
            "status": "error",
            "message": "Could not analyze latest ROI window before model input construction.",
            "analysis": analysis,
            "window_metadata": latest_window,
        }

    roi_names = [
        "forehead",
        "image_left_cheek",
        "image_right_cheek",
    ]

    extraction = extract_roi_rgb_series(
        samples=latest_window["samples"],
        roi_names=roi_names,
    )

    if not extraction["ok"]:
        return {
            "status": "error",
            "message": extraction["message"],
            "dropped_count": extraction["dropped_count"],
            "window_metadata": latest_window,
        }

    time_s = extraction["_time_s"]
    roi_rgb = extraction["_roi_rgb"]

    combined_rgb = combine_roi_rgb_by_mean(
        roi_rgb=roi_rgb,
        roi_names=roi_names,
    )

    green_signal = make_green_signal(combined_rgb)
    pos_signal = make_pos_signal(combined_rgb)
    chrom_signal = make_chrom_signal(combined_rgb)

    pos_240 = resample_signal_to_length(
        signal=pos_signal,
        target_length=target_length,
    )
    chrom_240 = resample_signal_to_length(
        signal=chrom_signal,
        target_length=target_length,
    )
    green_240 = resample_signal_to_length(
        signal=green_signal,
        target_length=target_length,
    )

    model_input = np.stack(
        [
            pos_240,
            chrom_240,
            green_240,
        ],
        axis=0,
    ).astype(np.float32)

    model_input = model_input[None, :, :]

    used_duration_s = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0

    return {
        "status": "ok",
        "message": "Model input built from latest live ROI window.",
        "sample_count": int(len(time_s)),
        "duration_s": used_duration_s,
        "estimated_fps": analysis["estimated_fps"],
        "target_length": int(target_length),
        "window_seconds": float(window_seconds),
        "channel_order": ["pos", "chrom", "green"],
        "input_shape": list(model_input.shape),
        "window_metadata": {
            "requested_window_seconds": float(window_seconds),
            "original_sample_count": latest_window["original_sample_count"],
            "used_sample_count": latest_window["used_sample_count"],
            "original_duration_s": latest_window["original_duration_s"],
            "used_duration_s": latest_window["used_duration_s"],
            "cutoff_s": latest_window["cutoff_s"],
        },
        "classical_analysis": analysis,
        "_model_input": model_input,
    }