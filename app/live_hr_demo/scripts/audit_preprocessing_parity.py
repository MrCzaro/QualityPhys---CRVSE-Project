from __future__ import annotations
import argparse
from email import parser
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import butter, filtfilt


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
APP_DIR = REPO_ROOT / "app" / "live_hr_demo"
DEFAULT_H5_PATH = REPO_ROOT / "Data" / "rppg_ensemble" / "ubfc_rppg_ensemble.h5"

sys.path.insert(0, str(APP_DIR))

from rppg.live_methods import (  
    make_chrom_signal,
    make_green_signal,
    make_pos_signal,
    resample_signal_to_length,
)
from rppg.sqi import estimate_spectral_sqi, compute_power_spectrum 
from inference.predictor import predict_hr_from_tensor  
from inference.window_inference import predict_hr_from_rppg_window  
from models.loader import load_model_bundle  
from rppg.windowing import WindowConfig, make_model_window_from_channels  

CHANNELS = ("pos", "chrom", "green")


def zscore(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a zero-mean, unit-variance signal, or zeros for flat input."""
    signal = np.asarray(signal, dtype=np.float64)
    std = float(np.std(signal))
    if std < eps:
        return np.zeros_like(signal, dtype=np.float32)
    return ((signal - float(np.mean(signal))) / std).astype(np.float32)


def bandpass_filter(signal: np.ndarray, fps: float, low_hz: float = 0.7, high_hz: float = 3.5, order: int = 4) -> np.ndarray:
    """Apply the notebook/training heart-rate bandpass filter to one signal."""
    signal = np.asarray(signal, dtype=np.float64)
    nyquist = 0.5 * float(fps)
    if high_hz >= nyquist:
        raise ValueError(f"high_hz={high_hz} must be below Nyquist frequency {nyquist:.3f}.")
    b, a = butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    min_len = 3 * max(len(a), len(b))
    if len(signal) <= min_len:
        raise ValueError(f"Signal too short for filtfilt bandpass: len={len(signal)}, required>{min_len}")
    return filtfilt(b, a, signal).astype(np.float32)


def training_green_from_roi_rgb(roi_rgb: np.ndarray, fps: float) -> np.ndarray:
    """Build the training-style GREEN signal from ROI RGB data."""
    green_by_roi = roi_rgb[:, :, 1].astype(np.float64)
    signal = np.mean(green_by_roi, axis=1)
    return zscore(bandpass_filter(signal, fps))


def training_pos_from_roi_rgb(roi_rgb: np.ndarray, fps: float) -> np.ndarray:
    """Build the training-style POS signal by computing POS per ROI first."""
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)
    roi_signals: list[np.ndarray] = []

    for roi_index in range(roi_rgb.shape[1]):
        rgb = roi_rgb[:, roi_index, :].astype(np.float64)
        normalized = rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-8)
        projected = projection @ normalized.T
        s0 = projected[0]
        s1 = projected[1]
        alpha = np.std(s0) / (np.std(s1) + 1e-8)
        roi_signals.append(s0 + alpha * s1)

    signal = np.mean(np.stack(roi_signals, axis=0), axis=0)
    return zscore(bandpass_filter(signal, fps))


def training_chrom_from_roi_rgb(roi_rgb: np.ndarray, fps: float) -> np.ndarray:
    """Build the training-style CHROM signal by computing CHROM per ROI first."""
    roi_signals: list[np.ndarray] = []

    for roi_index in range(roi_rgb.shape[1]):
        rgb = roi_rgb[:, roi_index, :].astype(np.float64)
        normalized = rgb / (np.mean(rgb, axis=0, keepdims=True) + 1e-8)
        r = normalized[:, 0]
        g = normalized[:, 1]
        b = normalized[:, 2]
        x = 3.0 * r - 2.0 * g
        y = 1.5 * r + g - 1.5 * b
        alpha = np.std(x) / (np.std(y) + 1e-8)
        roi_signals.append(x - alpha * y)

    signal = np.mean(np.stack(roi_signals, axis=0), axis=0)
    return zscore(bandpass_filter(signal, fps))


def build_training_style_signals(roi_rgb: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    """Compute POS, CHROM, and GREEN using the training-style ROI pipeline."""
    return {
        "pos": training_pos_from_roi_rgb(roi_rgb, fps),
        "chrom": training_chrom_from_roi_rgb(roi_rgb, fps),
        "green": training_green_from_roi_rgb(roi_rgb, fps),
    }


def build_current_live_style_signals(roi_rgb_window: np.ndarray, target_frames: int) -> dict[str, np.ndarray]:
    """Compute POS, CHROM, and GREEN using the current live-demo pipeline."""
    combined_rgb = np.mean(roi_rgb_window.astype(np.float64), axis=1)

    raw = {
        "pos": make_pos_signal(combined_rgb),
        "chrom": make_chrom_signal(combined_rgb),
        "green": make_green_signal(combined_rgb),
    }

    return {
        channel: resample_signal_to_length(raw[channel], target_frames)
        for channel in CHANNELS
    }


def model_ready(signal: np.ndarray, target_frames: int) -> np.ndarray:
    """Resample and normalize one signal to the model input length."""
    return resample_signal_to_length(np.asarray(signal, dtype=np.float64), target_frames)


def iter_recordings(h5: h5py.File):
    """Yield recording groups that contain ROI RGB data."""
    subjects = h5.get("subjects")
    if subjects is None:
        return

    for subject_name in sorted(subjects.keys()):
        recordings = subjects[subject_name].get("recordings")
        if recordings is None:
            continue

        for recording_name in sorted(recordings.keys()):
            group = recordings[recording_name]
            if "roi_rgb" in group:
                yield subject_name, recording_name, group


def select_recording(
    h5: h5py.File,
    subject: str | None,
    recording: str | None,
) -> tuple[str, str, h5py.Group]:
    """Select a requested recording or the first recording with ROI RGB data."""
    if subject is not None and recording is not None:
        return subject, recording, h5["subjects"][subject]["recordings"][recording]

    for subject_name, recording_name, group in iter_recordings(h5):
        return subject_name, recording_name, group

    raise RuntimeError("No recording with roi_rgb found in this HDF5 file.")


def corr(left: np.ndarray, right: np.ndarray) -> float:
    """Return Pearson correlation for two equal-length signals."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) != len(right):
        raise ValueError(f"Length mismatch: {len(left)} vs {len(right)}")
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return float("nan")

    return float(np.corrcoef(left, right)[0, 1])


def mae(left: np.ndarray, right: np.ndarray) -> float:
    """Return mean absolute error between two signals."""
    return float(np.mean(np.abs(np.asarray(left) - np.asarray(right))))


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    """Return root mean squared error between two signals."""
    diff = np.asarray(left) - np.asarray(right)
    return float(np.sqrt(np.mean(diff * diff)))


def spectral_summary(signal: np.ndarray, fps: float) -> tuple[float, float]:
    """Return dominant BPM and spectral SQI for one model-ready signal."""
    sqi = estimate_spectral_sqi(signal, fps=fps)
    return float(sqi.dominant_bpm), float(sqi.sqi)


def print_candidate_table(
    reference: dict[str, np.ndarray],
    candidates: dict[str, dict[str, np.ndarray]],
    signal_fps: float,
) -> None:
    """Print reference-vs-candidate waveform and spectral comparisons."""

    print("\nReference: stored HDF5 training signal")
    print("--------------------------------------")
    print(
        f"{'candidate':<24} {'channel':<8} {'corr':>8} {'mae':>9} {'rmse':>9} "
        f"{'ref_bpm':>9} {'cand_bpm':>9} {'bpm_diff':>9} {'ref_sqi':>9} {'cand_sqi':>9}"
    )

    for candidate_name, candidate in candidates.items():
        for channel in CHANNELS:
            ref_signal = reference[channel]
            candidate_signal = candidate[channel]

            ref_bpm, ref_sqi = spectral_summary(ref_signal, signal_fps)
            cand_bpm, cand_sqi = spectral_summary(candidate_signal, signal_fps)

            print(
                f"{candidate_name:<24} "
                f"{channel:<8} "
                f"{corr(ref_signal, candidate_signal):8.4f} "
                f"{mae(ref_signal, candidate_signal):9.4f} "
                f"{rmse(ref_signal, candidate_signal):9.4f} "
                f"{ref_bpm:9.2f} "
                f"{cand_bpm:9.2f} "
                f"{abs(ref_bpm - cand_bpm):9.2f} "
                f"{ref_sqi:9.4f} "
                f"{cand_sqi:9.4f}"
            )

def make_model_tensor_from_signals(signals: dict[str, np.ndarray], bundle):
    """Convert POS/CHROM/GREEN candidate signals into the active model tensor."""
    input_config = bundle.model_spec["input"]
    window_config = WindowConfig(
        window_seconds=float(input_config["window_seconds"]),
        target_frames=int(input_config["target_frames"]),
        channel_names=tuple(input_config["channel_names"]),
        normalization=str(input_config["normalization"]),
    )
    return make_model_window_from_channels(
        pos=signals["pos"],
        chrom=signals["chrom"],
        green=signals["green"],
        config=window_config,
    )


def run_direct_model_prediction(signals: dict[str, np.ndarray], bundle) -> float | None:
    """Run the neural model directly, without the app quality gate."""
    model_input = make_model_tensor_from_signals(signals=signals, bundle=bundle)
    prediction = predict_hr_from_tensor(model_input, bundle)[0]
    if prediction.value is None:
        return None
    return float(prediction.value)


def run_quality_gated_prediction(signals: dict[str, np.ndarray], bundle) -> dict[str, object]:
    """Run the current app window-inference path, including quality gating."""
    input_config = bundle.model_spec["input"]
    fps = float(input_config["target_frames"]) / float(input_config["window_seconds"])
    prediction = predict_hr_from_rppg_window(signals=signals, fps=fps, bundle=bundle)
    return {
        "model_hr": None if prediction.value is None else float(prediction.value),
        "quality_status": prediction.quality.status,
        "quality_confidence": prediction.quality.confidence,
        "spectral_hr": prediction.extra.get("spectral_hr_bpm"),
    }


def print_model_comparison_table(
    candidates: dict[str, dict[str, np.ndarray]],
    device: str,
    target_hr_bpm: float | None,
) -> None:
    """Print model predictions, target error, and quality-gate diagnostics."""
    print("\nModel comparison")
    print("----------------")
    print(f"Loading model on device: {device}")
    bundle = load_model_bundle(device=device)

    print(f"Model: {bundle.model_spec['name']}")
    print(f"Checkpoint: {bundle.checkpoint_path}")
    print(f"Target HR: {target_hr_bpm if target_hr_bpm is not None else 'missing'}")
    print()
    print(
        f"{'candidate':<24} "
        f"{'direct_hr':>10} "
        f"{'direct_err':>11} "
        f"{'gated_hr':>10} "
        f"{'gated_err':>10} "
        f"{'spectral_hr':>12} "
        f"{'quality':>12} "
        f"{'confidence':>14}"
    )

    rejected_details: list[tuple[str, list[str]]] = []
    for candidate_name, signals in candidates.items():
        direct_hr = run_direct_model_prediction(signals=signals, bundle=bundle)
        gated = run_quality_gated_prediction(signals=signals, bundle=bundle)
        gated_hr = gated["model_hr"]
        spectral_hr = gated["spectral_hr"]

        direct_err = (None if direct_hr is None or target_hr_bpm is None else abs(float(direct_hr) - float(target_hr_bpm)))
        gated_err = (None if gated_hr is None or target_hr_bpm is None else abs(float(gated_hr) - float(target_hr_bpm)))

        direct_text = "None" if direct_hr is None else f"{direct_hr:.2f}"
        direct_err_text = "None" if direct_err is None else f"{direct_err:.2f}"
        gated_text = "None" if gated_hr is None else f"{gated_hr:.2f}"
        gated_err_text = "None" if gated_err is None else f"{gated_err:.2f}"
        spectral_text = "None" if spectral_hr is None else f"{float(spectral_hr):.2f}"

        print(
            f"{candidate_name:<24} "
            f"{direct_text:>10} "
            f"{direct_err_text:>11} "
            f"{gated_text:>10} "
            f"{gated_err_text:>10} "
            f"{spectral_text:>12} "
            f"{str(gated['quality_status']):>12} "
            f"{str(gated['quality_confidence']):>14}"
        )

        if gated["quality_status"] == "rejected":
            quality_result = predict_hr_from_rppg_window(
                signals=signals,
                fps=float(bundle.model_spec["input"]["target_frames"])
                / float(bundle.model_spec["input"]["window_seconds"]),
                bundle=bundle,
            )
            rejected_details.append((candidate_name, quality_result.quality.reasons))

    if rejected_details:
        print("\nRejected quality reasons")
        print("------------------------")
        for candidate_name, reasons in rejected_details:
            print(f"{candidate_name}:")
            for reason in reasons:
                print(f"  - {reason}")


def top_cardiac_peaks(
    signal: np.ndarray,
    fps: float,
    top_n: int = 6,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
    min_separation_bpm: float = 7.5,
) -> list[dict[str, float]]:
    """Return separated high-power peaks inside the cardiac frequency band."""
    freqs_hz, power = compute_power_spectrum(signal=signal, fps=fps)
    cardiac_mask = (freqs_hz >= low_hz) & (freqs_hz <= high_hz)
    cardiac_indices = np.where(cardiac_mask)[0]

    if len(cardiac_indices) == 0:
        return []

    total_power = float(np.sum(power[cardiac_indices]))
    if total_power <= 1e-12:
        return []

    local_peak_indices: list[int] = []

    for idx in cardiac_indices:
        left_power = power[idx - 1] if idx > 0 else -np.inf
        right_power = power[idx + 1] if idx + 1 < len(power) else -np.inf

        if power[idx] >= left_power and power[idx] >= right_power:
            local_peak_indices.append(int(idx))

    if len(local_peak_indices) == 0:
        local_peak_indices = [int(idx) for idx in cardiac_indices]

    ordered = sorted(
        local_peak_indices,
        key=lambda idx: float(power[idx]),
        reverse=True,
    )

    selected: list[int] = []
    min_separation_hz = float(min_separation_bpm) / 60.0

    for idx in ordered:
        if all(abs(float(freqs_hz[idx] - freqs_hz[other])) >= min_separation_hz for other in selected):
            selected.append(idx)

        if len(selected) >= top_n:
            break

    peaks = []

    for idx in selected:
        peaks.append(
            {
                "bpm": float(freqs_hz[idx] * 60.0),
                "power_pct": float(100.0 * power[idx] / total_power),
            }
        )

    return peaks


def target_frequency_power(
    signal: np.ndarray,
    fps: float,
    target_hr_bpm: float | None,
    low_hz: float = 0.7,
    high_hz: float = 3.5,
) -> dict[str, float] | None:
    """Return nearest FFT-bin power percentage at the target HR."""
    if target_hr_bpm is None:
        return None

    freqs_hz, power = compute_power_spectrum(signal=signal, fps=fps)
    cardiac_mask = (freqs_hz >= low_hz) & (freqs_hz <= high_hz)
    cardiac_indices = np.where(cardiac_mask)[0]

    if len(cardiac_indices) == 0:
        return None

    total_power = float(np.sum(power[cardiac_indices]))
    if total_power <= 1e-12:
        return None

    target_hz = float(target_hr_bpm) / 60.0
    nearest_idx = int(cardiac_indices[np.argmin(np.abs(freqs_hz[cardiac_indices] - target_hz))])

    return {
        "nearest_bpm": float(freqs_hz[nearest_idx] * 60.0),
        "power_pct": float(100.0 * power[nearest_idx] / total_power),
    }


def print_peak_diagnostics(
    candidates: dict[str, dict[str, np.ndarray]],
    signal_fps: float,
    target_hr_bpm: float | None,
    top_n: int,
) -> None:
    """Print top cardiac-band peaks for each candidate and channel."""
    print("\nTop spectral peaks")
    print("------------------")

    for candidate_name, signals in candidates.items():
        print(f"\n{candidate_name}")

        for channel in CHANNELS:
            peaks = top_cardiac_peaks(
                signal=signals[channel],
                fps=signal_fps,
                top_n=top_n,
            )
            target_power = target_frequency_power(
                signal=signals[channel],
                fps=signal_fps,
                target_hr_bpm=target_hr_bpm,
            )

            peak_text = ", ".join(
                f"{peak['bpm']:.1f} bpm ({peak['power_pct']:.1f}%)"
                for peak in peaks
            )

            if target_power is None:
                target_text = "target bin: missing"
            else:
                target_text = (
                    f"target bin: {target_power['nearest_bpm']:.1f} bpm "
                    f"({target_power['power_pct']:.1f}%)"
                )

            print(f"  {channel:<6} peaks: {peak_text}")
            print(f"         {target_text}")


def make_zero_signal_like(reference: np.ndarray) -> np.ndarray:
    """Return a zero-valued signal with the same shape as the reference signal."""
    return np.zeros_like(np.asarray(reference, dtype=np.float32))


def build_channel_ablation_candidates(signals: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """Build channel-ablation variants for one POS/CHROM/GREEN signal set."""
    zero = make_zero_signal_like(signals["pos"])

    return {
        "all_channels": {
            "pos": signals["pos"],
            "chrom": signals["chrom"],
            "green": signals["green"],
        },
        "pos_only": {
            "pos": signals["pos"],
            "chrom": zero,
            "green": zero,
        },
        "chrom_only": {
            "pos": zero,
            "chrom": signals["chrom"],
            "green": zero,
        },
        "green_only": {
            "pos": zero,
            "chrom": zero,
            "green": signals["green"],
        },
        "pos_chrom": {
            "pos": signals["pos"],
            "chrom": signals["chrom"],
            "green": zero,
        },
        "pos_green": {
            "pos": signals["pos"],
            "chrom": zero,
            "green": signals["green"],
        },
        "chrom_green": {
            "pos": zero,
            "chrom": signals["chrom"],
            "green": signals["green"],
        },
    }


def print_channel_ablation_table(
    base_name: str,
    signals: dict[str, np.ndarray],
    device: str,
    target_hr_bpm: float | None,
) -> None:
    """Print direct model predictions for channel-ablation variants."""
    print("\nChannel ablation")
    print("----------------")
    print(f"Base candidate: {base_name}")
    print(f"Loading model on device: {device}")
    bundle = load_model_bundle(device=device)
    ablations = build_channel_ablation_candidates(signals)

    print(f"Model:      {bundle.model_spec['name']}")
    print(f"Checkpoint: {bundle.checkpoint_path}")
    print(f"Target HR:  {target_hr_bpm if target_hr_bpm is not None else 'missing'}")
    print()
    print(
        f"{'variant':<18} "
        f"{'direct_hr':>10} "
        f"{'direct_err':>11}"
    )

    for variant_name, variant_signals in ablations.items():
        direct_hr = run_direct_model_prediction(
            signals=variant_signals,
            bundle=bundle,
        )

        direct_err = (
            None
            if direct_hr is None or target_hr_bpm is None
            else abs(float(direct_hr) - float(target_hr_bpm))
        )

        direct_text = "None" if direct_hr is None else f"{direct_hr:.2f}"
        err_text = "None" if direct_err is None else f"{direct_err:.2f}"

        print(
            f"{variant_name:<18} "
            f"{direct_text:>10} "
            f"{err_text:>11}"
        )

def parse_seconds_csv(text: str) -> list[float]:
    """Parse a comma-separated list of seconds into floats."""
    values: list[float] = []
    for part in text.split(","):
        stripped = part.strip()
        if stripped:
            values.append(float(stripped))
    if not values:
        raise ValueError("Expected at least one buffer duration.")

    return values


def print_buffer_context_sweep(
    group,
    roi_rgb_full: np.ndarray,
    stored_full: dict[str, np.ndarray],
    fps: float,
    analysis_end_second: float,
    window_seconds: float,
    target_frames: int,
    buffer_seconds_values: list[float],
    device: str,
) -> None:
    """Compare training-style model input as buffer context length changes."""
    analysis_end_frame = int(round(float(analysis_end_second) * fps))
    window_frames = int(round(float(window_seconds) * fps))
    analysis_start_frame = analysis_end_frame - window_frames

    if analysis_end_frame > len(roi_rgb_full):
        raise ValueError(
            f"analysis_end_frame={analysis_end_frame} exceeds recording length "
            f"{len(roi_rgb_full)}."
        )

    if analysis_start_frame < 0:
        raise ValueError(
            f"Analysis window starts before frame 0. "
            f"analysis_end_second={analysis_end_second}, window_seconds={window_seconds}."
        )

    target_info = get_window_target_hr(
        group=group,
        analysis_start=analysis_start_frame,
        analysis_end=analysis_end_frame,
    )
    target_hr_bpm = target_info["selected_target"]

    stored_reference = {
        channel: model_ready(
            stored_full[channel][analysis_start_frame:analysis_end_frame],
            target_frames,
        )
        for channel in CHANNELS
    }

    bundle = load_model_bundle(device=device)
    stored_reference_direct_hr = run_direct_model_prediction(
        signals=stored_reference,
        bundle=bundle,
    )
    stored_reference_direct_err = (
        None
        if stored_reference_direct_hr is None or target_hr_bpm is None
        else abs(float(stored_reference_direct_hr) - float(target_hr_bpm))
    )
    print("\nBuffer context sweep")
    print("--------------------")
    print(f"Analysis frames: {analysis_start_frame}:{analysis_end_frame}")
    print(f"Analysis seconds: {analysis_start_frame / fps:.3f}:{analysis_end_frame / fps:.3f}")
    print(f"Window HR mean: {target_info['window_mean']}")
    print(f"Window HR median: {target_info['window_median']}")
    print(f"Window HR min/max: {target_info['window_min']} / {target_info['window_max']}")
    print(f"Selected target HR: {target_info['selected_target']} ({target_info['selected_source']})")
    print(f"Stored reference direct HR: {stored_reference_direct_hr:.2f} (err {stored_reference_direct_err:.2f})")
    print()
    print(
        f"{'req_buf_s':>9} "
        f"{'actual_s':>9} "
        f"{'pos_corr':>9} "
        f"{'chrom_corr':>10} "
        f"{'green_corr':>10} "
        f"{'direct_hr':>10} "
        f"{'direct_err':>11}"
    )

    for requested_buffer_seconds in buffer_seconds_values:
        buffer_frames = int(round(float(requested_buffer_seconds) * fps))
        buffer_start_frame = max(0, analysis_end_frame - buffer_frames)
        actual_buffer_frames = analysis_end_frame - buffer_start_frame

        if actual_buffer_frames < window_frames:
            print(
                f"{requested_buffer_seconds:9.1f} "
                f"{actual_buffer_frames / fps:9.3f} "
                f"{'SKIP':>9} "
                f"{'SKIP':>10} "
                f"{'SKIP':>10} "
                f"{'SKIP':>10} "
                f"{'SKIP':>11}"
            )
            continue

        roi_rgb_buffer = roi_rgb_full[buffer_start_frame:analysis_end_frame]
        buffer_signals = build_training_style_signals(roi_rgb_buffer, fps)

        relative_window_start = len(roi_rgb_buffer) - window_frames
        candidate = {
            channel: model_ready(
                buffer_signals[channel][relative_window_start:],
                target_frames,
            )
            for channel in CHANNELS
        }

        direct_hr = run_direct_model_prediction(
            signals=candidate,
            bundle=bundle,
        )

        direct_err = (
            None
            if direct_hr is None or target_hr_bpm is None
            else abs(float(direct_hr) - float(target_hr_bpm))
        )

        direct_text = "None" if direct_hr is None else f"{direct_hr:.2f}"
        err_text = "None" if direct_err is None else f"{direct_err:.2f}"

        print(
            f"{requested_buffer_seconds:9.1f} "
            f"{actual_buffer_frames / fps:9.3f} "
            f"{corr(stored_reference['pos'], candidate['pos']):9.4f} "
            f"{corr(stored_reference['chrom'], candidate['chrom']):10.4f} "
            f"{corr(stored_reference['green'], candidate['green']):10.4f} "
            f"{direct_text:>10} "
            f"{err_text:>11}"
        )


def get_window_target_hr(group, analysis_start: int, analysis_end: int) -> dict[str, float | None]:
    """Return global and window-local HR targets when available."""
    global_hr = group.attrs.get("hr_mean", None)
    global_hr = None if global_hr is None else float(global_hr)

    if "hr_continuous" not in group:
        return {
            "global_mean": global_hr,
            "window_mean": None,
            "window_median": None,
            "window_min": None,
            "window_max": None,
            "selected_target": global_hr,
            "selected_source": "global_hr_mean",
        }

    hr = group["hr_continuous"][analysis_start:analysis_end].astype(np.float32)
    hr = hr[np.isfinite(hr)]

    if len(hr) == 0:
        return {
            "global_mean": global_hr,
            "window_mean": None,
            "window_median": None,
            "window_min": None,
            "window_max": None,
            "selected_target": global_hr,
            "selected_source": "global_hr_mean",
        }

    window_mean = float(np.mean(hr))
    window_median = float(np.median(hr))

    return {
        "global_mean": global_hr,
        "window_mean": window_mean,
        "window_median": window_median,
        "window_min": float(np.min(hr)),
        "window_max": float(np.max(hr)),
        "selected_target": window_mean,
        "selected_source": "hr_continuous_window_mean",
    }


def print_future_context_sweep(
    group,
    roi_rgb_full: np.ndarray,
    stored_full: dict[str, np.ndarray],
    fps: float,
    analysis_end_second: float,
    window_seconds: float,
    target_frames: int,
    pre_context_seconds: float,
    future_context_values: list[float],
    device: str,
) -> None:
    """Compare training-style signals as future filter context changes."""
    analysis_end_frame = int(round(float(analysis_end_second) * fps))
    window_frames = int(round(float(window_seconds) * fps))
    analysis_start_frame = analysis_end_frame - window_frames

    if analysis_end_frame > len(roi_rgb_full):
        raise ValueError(
            f"analysis_end_frame={analysis_end_frame} exceeds recording length "
            f"{len(roi_rgb_full)}."
        )

    if analysis_start_frame < 0:
        raise ValueError(
            f"Analysis window starts before frame 0. "
            f"analysis_end_second={analysis_end_second}, window_seconds={window_seconds}."
        )

    target_info = get_window_target_hr(
        group=group,
        analysis_start=analysis_start_frame,
        analysis_end=analysis_end_frame,
    )
    target_hr_bpm = target_info["selected_target"]

    stored_reference = {
        channel: model_ready(
            stored_full[channel][analysis_start_frame:analysis_end_frame],
            target_frames,
        )
        for channel in CHANNELS
    }

    bundle = load_model_bundle(device=device)

    stored_reference_direct_hr = run_direct_model_prediction(
        signals=stored_reference,
        bundle=bundle,
    )
    stored_reference_direct_err = (
        None
        if stored_reference_direct_hr is None or target_hr_bpm is None
        else abs(float(stored_reference_direct_hr) - float(target_hr_bpm))
    )
    full_signals = build_training_style_signals(roi_rgb_full, fps)
    full_recording_candidate = {
        channel: model_ready(
            full_signals[channel][analysis_start_frame:analysis_end_frame],
            target_frames,
        )
        for channel in CHANNELS
    }

    full_recording_direct_hr = run_direct_model_prediction(
        signals=full_recording_candidate,
        bundle=bundle,
    )
    full_recording_direct_err = (
        None
        if full_recording_direct_hr is None or target_hr_bpm is None
        else abs(float(full_recording_direct_hr) - float(target_hr_bpm))
    )

    print("\nFuture context sweep")
    print("--------------------")
    print(f"Analysis frames: {analysis_start_frame}:{analysis_end_frame}")
    print(f"Analysis seconds: {analysis_start_frame / fps:.3f}:{analysis_end_frame / fps:.3f}")
    print(f"Window HR mean: {target_info['window_mean']}")
    print(f"Window HR median: {target_info['window_median']}")
    print(f"Window HR min/max: {target_info['window_min']} / {target_info['window_max']}")
    print(f"Selected target HR: {target_info['selected_target']} ({target_info['selected_source']})")
    print(
        "Stored reference direct HR: "
        f"{stored_reference_direct_hr:.2f} "
        f"(err {stored_reference_direct_err:.2f})"
    )
    print()
    print(
        f"{'future_s':>9} "
        f"{'actual_pre':>10} "
        f"{'actual_future':>13} "
        f"{'pos_corr':>9} "
        f"{'chrom_corr':>10} "
        f"{'green_corr':>10} "
        f"{'direct_hr':>10} "
        f"{'direct_err':>11}"
    )

    full_recording_hr_text = (
        "None"
        if full_recording_direct_hr is None
        else f"{full_recording_direct_hr:.2f}"
    )
    full_recording_err_text = (
        "None"
        if full_recording_direct_err is None
        else f"{full_recording_direct_err:.2f}"
    )

    print(
        f"{'full':>9} "
        f"{'full':>10} "
        f"{'full':>13} "
        f"{corr(stored_reference['pos'], full_recording_candidate['pos']):9.4f} "
        f"{corr(stored_reference['chrom'], full_recording_candidate['chrom']):10.4f} "
        f"{corr(stored_reference['green'], full_recording_candidate['green']):10.4f} "
        f"{full_recording_hr_text:>10} "
        f"{full_recording_err_text:>11}"
    )
    pre_context_frames = int(round(float(pre_context_seconds) * fps))

    for future_context_seconds in future_context_values:
        future_context_frames = int(round(float(future_context_seconds) * fps))

        buffer_start_frame = max(0, analysis_start_frame - pre_context_frames)
        buffer_end_frame = min(
            len(roi_rgb_full),
            analysis_end_frame + future_context_frames,
        )

        actual_pre_seconds = (analysis_start_frame - buffer_start_frame) / fps
        actual_future_seconds = (buffer_end_frame - analysis_end_frame) / fps

        roi_rgb_buffer = roi_rgb_full[buffer_start_frame:buffer_end_frame]
        buffer_signals = build_training_style_signals(roi_rgb_buffer, fps)

        relative_start = analysis_start_frame - buffer_start_frame
        relative_end = analysis_end_frame - buffer_start_frame

        candidate = {
            channel: model_ready(
                buffer_signals[channel][relative_start:relative_end],
                target_frames,
            )
            for channel in CHANNELS
        }

        direct_hr = run_direct_model_prediction(
            signals=candidate,
            bundle=bundle,
        )

        direct_err = (
            None
            if direct_hr is None or target_hr_bpm is None
            else abs(float(direct_hr) - float(target_hr_bpm))
        )

        direct_text = "None" if direct_hr is None else f"{direct_hr:.2f}"
        err_text = "None" if direct_err is None else f"{direct_err:.2f}"

        print(
            f"{future_context_seconds:9.1f} "
            f"{actual_pre_seconds:10.3f} "
            f"{actual_future_seconds:13.3f} "
            f"{corr(stored_reference['pos'], candidate['pos']):9.4f} "
            f"{corr(stored_reference['chrom'], candidate['chrom']):10.4f} "
            f"{corr(stored_reference['green'], candidate['green']):10.4f} "
            f"{direct_text:>10} "
            f"{err_text:>11}"
        )


def print_analysis_window_sweep(
    group,
    roi_rgb_full: np.ndarray,
    stored_full: dict[str, np.ndarray],
    fps: float,
    analysis_end_seconds_values: list[float],
    window_seconds: float,
    local_buffer_seconds: float,
    target_frames: int,
    device: str,
) -> None:
    """Compare stored-reference and local-buffer model behavior across windows."""
    bundle = load_model_bundle(device=device)
    window_frames = int(round(float(window_seconds) * fps))
    local_buffer_frames = int(round(float(local_buffer_seconds) * fps))

    print("\nAnalysis window sweep")
    print("---------------------")
    print(f"Window seconds: {window_seconds:.3f}")
    print(f"Local buffer seconds: {local_buffer_seconds:.3f}")
    print()
    print(
        f"{'end_s':>7} "
        f"{'frames':>13} "
        f"{'hr_mean':>9} "
        f"{'hr_med':>9} "
        f"{'hr_min':>9} "
        f"{'hr_max':>9} "
        f"{'stored_hr':>10} "
        f"{'stored_err':>11} "
        f"{'local_hr':>10} "
        f"{'local_err':>10} "
        f"{'pos_r':>7} "
        f"{'chrom_r':>8} "
        f"{'green_r':>8}"
    )

    for analysis_end_second in analysis_end_seconds_values:
        analysis_end_frame = int(round(float(analysis_end_second) * fps))
        analysis_start_frame = analysis_end_frame - window_frames

        if analysis_start_frame < 0 or analysis_end_frame > len(roi_rgb_full):
            print(
                f"{analysis_end_second:7.1f} "
                f"{'SKIP':>13} "
                f"{'SKIP':>9} "
                f"{'SKIP':>9} "
                f"{'SKIP':>9} "
                f"{'SKIP':>9} "
                f"{'SKIP':>10} "
                f"{'SKIP':>11} "
                f"{'SKIP':>10} "
                f"{'SKIP':>10} "
                f"{'SKIP':>7} "
                f"{'SKIP':>8} "
                f"{'SKIP':>8}"
            )
            continue

        target_info = get_window_target_hr(
            group=group,
            analysis_start=analysis_start_frame,
            analysis_end=analysis_end_frame,
        )
        target_hr_bpm = target_info["selected_target"]

        stored_reference = {
            channel: model_ready(
                stored_full[channel][analysis_start_frame:analysis_end_frame],
                target_frames,
            )
            for channel in CHANNELS
        }

        stored_hr = run_direct_model_prediction(
            signals=stored_reference,
            bundle=bundle,
        )
        stored_err = (
            None
            if stored_hr is None or target_hr_bpm is None
            else abs(float(stored_hr) - float(target_hr_bpm))
        )

        buffer_end_frame = analysis_end_frame
        buffer_start_frame = max(0, buffer_end_frame - local_buffer_frames)
        actual_buffer_frames = buffer_end_frame - buffer_start_frame

        if actual_buffer_frames < window_frames:
            local_hr = None
            local_err = None
            pos_corr = float("nan")
            chrom_corr = float("nan")
            green_corr = float("nan")
        else:
            roi_rgb_buffer = roi_rgb_full[buffer_start_frame:buffer_end_frame]
            local_signals_raw = build_training_style_signals(roi_rgb_buffer, fps)

            relative_start = len(roi_rgb_buffer) - window_frames
            local_candidate = {
                channel: model_ready(
                    local_signals_raw[channel][relative_start:],
                    target_frames,
                )
                for channel in CHANNELS
            }

            local_hr = run_direct_model_prediction(
                signals=local_candidate,
                bundle=bundle,
            )
            local_err = (
                None
                if local_hr is None or target_hr_bpm is None
                else abs(float(local_hr) - float(target_hr_bpm))
            )

            pos_corr = corr(stored_reference["pos"], local_candidate["pos"])
            chrom_corr = corr(stored_reference["chrom"], local_candidate["chrom"])
            green_corr = corr(stored_reference["green"], local_candidate["green"])

        stored_hr_text = "None" if stored_hr is None else f"{stored_hr:.2f}"
        stored_err_text = "None" if stored_err is None else f"{stored_err:.2f}"
        local_hr_text = "None" if local_hr is None else f"{local_hr:.2f}"
        local_err_text = "None" if local_err is None else f"{local_err:.2f}"

        print(
            f"{analysis_end_second:7.1f} "
            f"{analysis_start_frame:5d}:{analysis_end_frame:<7d} "
            f"{target_info['window_mean']:9.2f} "
            f"{target_info['window_median']:9.2f} "
            f"{target_info['window_min']:9.2f} "
            f"{target_info['window_max']:9.2f} "
            f"{stored_hr_text:>10} "
            f"{stored_err_text:>11} "
            f"{local_hr_text:>10} "
            f"{local_err_text:>10} "
            f"{pos_corr:7.3f} "
            f"{chrom_corr:8.3f} "
            f"{green_corr:8.3f}"
        )


def main() -> None:
    """Run the preprocessing parity experiment, with optional model comparison."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare stored training signals against current live-style and "
            "training-style candidate live preprocessing paths."
        )
    )
    parser.add_argument("--h5-path", type=Path, default=DEFAULT_H5_PATH)
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--recording", type=str, default=None)
    parser.add_argument("--start-second", type=float, default=0.0)
    parser.add_argument("--buffer-seconds", type=float, default=12.0)
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--target-frames", type=int, default=240)
    parser.add_argument("--run-model", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--print-peaks", action="store_true")
    parser.add_argument("--top-peaks", type=int, default=6)
    parser.add_argument("--sweep-buffer-context", action="store_true")
    parser.add_argument("--analysis-end-second", type=float, default=None)
    parser.add_argument("--sweep-buffer-seconds", type=str, default="8,10,12,16,20,30")
    parser.add_argument("--sweep-future-context", action="store_true")
    parser.add_argument("--pre-context-seconds", type=float, default=20.0)
    parser.add_argument("--future-context-seconds", type=str, default="0,1,2,4,8,12,20")
    parser.add_argument("--sweep-analysis-windows", action="store_true")
    parser.add_argument("--analysis-end-seconds", type=str, default="12,20,30,40,50")
    parser.add_argument("--local-buffer-seconds", type=float, default=12.0)
    parser.add_argument(
        "--run-channel-ablation",
        choices=("stored_reference", "current_live", "training_window_8s", "training_buffer_latest8"),
        default=None,
    )
    args = parser.parse_args()

    h5_path = args.h5_path.resolve()
    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        subject_name, recording_name, group = select_recording(
            h5=h5,
            subject=args.subject,
            recording=args.recording,
        )

        roi_rgb_full = group["roi_rgb"][:].astype(np.float32)
        fps = float(group.attrs.get("fps", 30.0))

        buffer_start = int(round(args.start_second * fps))
        requested_buffer_frames = int(round(args.buffer_seconds * fps))
        window_frames = int(round(args.window_seconds * fps))

        if buffer_start >= len(roi_rgb_full):
            raise ValueError(
                f"Buffer start frame {buffer_start} exceeds recording length "
                f"{len(roi_rgb_full)}."
            )

        buffer_end = min(buffer_start + requested_buffer_frames, len(roi_rgb_full))
        buffer_frames = buffer_end - buffer_start

        if buffer_frames < window_frames:
            raise ValueError(
                f"Buffer too short for requested analysis window: "
                f"buffer_frames={buffer_frames}, window_frames={window_frames}."
            )

        analysis_end = buffer_end
        analysis_start = analysis_end - window_frames

        target_info = get_window_target_hr(
            group=group,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
        )
        
        roi_rgb_window = roi_rgb_full[analysis_start:analysis_end]
        roi_rgb_buffer = roi_rgb_full[buffer_start:buffer_end]

        stored_full = {
            channel: group[f"rppg_{channel}"][:].astype(np.float32)
            for channel in CHANNELS
        }

        stored_reference = {
            channel: model_ready(
                stored_full[channel][analysis_start:analysis_end],
                args.target_frames,
            )
            for channel in CHANNELS
        }

        full_recomputed = build_training_style_signals(roi_rgb_full, fps)
        full_recomputed_window = {
            channel: model_ready(
                full_recomputed[channel][analysis_start:analysis_end],
                args.target_frames,
            )
            for channel in CHANNELS
        }

        current_live = build_current_live_style_signals(
            roi_rgb_window=roi_rgb_window,
            target_frames=args.target_frames,
        )

        training_window_raw = build_training_style_signals(roi_rgb_window, fps)
        training_window = {
            channel: model_ready(training_window_raw[channel], args.target_frames)
            for channel in CHANNELS
        }

        training_buffer_raw = build_training_style_signals(roi_rgb_buffer, fps)
        relative_window_start = len(roi_rgb_buffer) - window_frames
        training_buffer = {
            channel: model_ready(
                training_buffer_raw[channel][relative_window_start:],
                args.target_frames,
            )
            for channel in CHANNELS
        }

        signal_fps = args.target_frames / args.window_seconds

        print("\nROI preprocessing parity audit")
        print("==============================")
        print(f"HDF5: {h5_path}")
        print(f"Subject: {subject_name}")
        print(f"Recording: {recording_name}")
        print(f"FPS: {fps:.3f}")
        print(f"ROI shape: {roi_rgb_full.shape}")
        print(f"Buffer frames: {buffer_start}:{buffer_end}")
        print(f"Analysis frames: {analysis_start}:{analysis_end}")
        print(f"Buffer seconds: {buffer_frames / fps:.3f}")
        print(f"Analysis seconds: {window_frames / fps:.3f}")
        print(f"Model FPS: {signal_fps:.3f}")
        print(f"Attrs HR: {group.attrs.get('hr_mean', 'missing')}")
        print(f"Window HR mean: {target_info['window_mean']}")
        print(f"Window HR median: {target_info['window_median']}")
        print(f"Window HR min/max: {target_info['window_min']} / {target_info['window_max']}")
        print(f"Selected target HR: {target_info['selected_target']} ({target_info['selected_source']})")
        print(f"Attrs SQI: {group.attrs.get('sqi', 'missing')}")

        candidates = {
            "full_recomputed_sanity": full_recomputed_window,
            "current_live": current_live,
            "training_window_8s": training_window,
            "training_buffer_latest8": training_buffer,
        }

        print_candidate_table(reference=stored_reference, candidates=candidates, signal_fps=signal_fps)

        if args.run_model:
            target_hr_bpm = target_info["selected_target"]

            model_candidates = {
                "stored_reference": stored_reference,
                "current_live": current_live,
                "training_window_8s": training_window,
                "training_buffer_latest8": training_buffer,
            }

            print_model_comparison_table(candidates=model_candidates, device=args.device, target_hr_bpm=target_hr_bpm)

        if args.run_channel_ablation is not None:
            target_hr_bpm = target_info["selected_target"]

            ablation_sources = {
                "stored_reference": stored_reference,
                "current_live": current_live,
                "training_window_8s": training_window,
                "training_buffer_latest8": training_buffer,
            }

            print_channel_ablation_table(
                base_name=args.run_channel_ablation,
                signals=ablation_sources[args.run_channel_ablation],
                device=args.device,
                target_hr_bpm=target_hr_bpm,
            )

        if args.print_peaks:
            peak_candidates = {
                "stored_reference": stored_reference,
                "current_live": current_live,
                "training_window_8s": training_window,
                "training_buffer_latest8": training_buffer,
            }

            target_hr_bpm = target_info["selected_target"]

            print_peak_diagnostics(
                candidates=peak_candidates,
                signal_fps=signal_fps,
                target_hr_bpm=target_hr_bpm,
                top_n=args.top_peaks,
            )

        if args.sweep_buffer_context:
            target_hr_bpm = target_info["selected_target"]

            analysis_end_second = args.analysis_end_second
            if analysis_end_second is None:
                analysis_end_second = float(analysis_end) / float(fps)

            print_buffer_context_sweep(
                group=group,
                roi_rgb_full=roi_rgb_full,
                stored_full=stored_full,
                fps=fps,
                analysis_end_second=analysis_end_second,
                window_seconds=args.window_seconds,
                target_frames=args.target_frames,
                buffer_seconds_values=parse_seconds_csv(args.sweep_buffer_seconds),
                device=args.device,
            )

        if args.sweep_future_context:
            analysis_end_second = args.analysis_end_second
            if analysis_end_second is None:
                analysis_end_second = float(analysis_end) / float(fps)

            print_future_context_sweep(
                group=group,
                roi_rgb_full=roi_rgb_full,
                stored_full=stored_full,
                fps=fps,
                analysis_end_second=analysis_end_second,
                window_seconds=args.window_seconds,
                target_frames=args.target_frames,
                pre_context_seconds=args.pre_context_seconds,
                future_context_values=parse_seconds_csv(args.future_context_seconds),
                device=args.device,
            )

        if args.sweep_analysis_windows:
            print_analysis_window_sweep(
                group=group,
                roi_rgb_full=roi_rgb_full,
                stored_full=stored_full,
                fps=fps,
                analysis_end_seconds_values=parse_seconds_csv(args.analysis_end_seconds),
                window_seconds=args.window_seconds,
                local_buffer_seconds=args.local_buffer_seconds,
                target_frames=args.target_frames,
                device=args.device,
            )
            
        print("\nHow to read this")
        print("----------------")
        print(
            "full_recomputed_sanity should be corr=1.0. If it is not, the "
            "script is not reconstructing the stored training preprocessing."
        )
        print(
            "current_live is the existing app-style path: average ROI RGB first, "
            "then minimal POS/CHROM/GREEN on the selected window."
        )
        print(
            "training_window_8s tests whether training-style preprocessing works "
            "when only the exact 8-second live window is available."
        )
        print(
            "training_buffer_latest8 tests whether a longer collection buffer "
            "helps by giving the bandpass filter context before cropping the "
            "latest 8 seconds."
        )
        print(
            "--run-model additionally compares raw model output and the current "
            "quality-gated app inference path for each candidate signal set."
        )


if __name__ == "__main__":
    main()