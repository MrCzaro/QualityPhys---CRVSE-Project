from __future__ import annotations

"""
Build a window-level audit table for live-compatible CRVSE preprocessing.

The script compares stored training tensors, reconstructed full-recording tensors,
compact live-style tensors, source-FPS local-buffer tensors, and simulated low-FPS
local-buffer tensors.

It writes:
- a detailed per-window CSV
- an aggregated summary CSV
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
APP_DIR = REPO_ROOT / "app" / "live_hr_demo"
sys.path.insert(0, str(APP_DIR))

from inference.predictor import predict_hr_from_tensor
from models.loader import load_model_bundle
from rppg.sqi import estimate_spectral_sqi
from rppg.windowing import WindowConfig, make_model_window_from_channels
from scripts.audit_preprocessing_parity import (
    build_current_live_style_signals,
    build_training_style_signals,
    corr,
    model_ready,
)

DATASETS = {
    "mcd_rppg": REPO_ROOT / "Data" / "rppg_ensemble" / "mcd_rppg_ensemble.h5",
    "ubfc_rppg": REPO_ROOT / "Data" / "rppg_ensemble" / "ubfc_rppg_ensemble.h5",
    "ubfc_phys": REPO_ROOT / "Data" / "rppg_ensemble" / "ubfc_phys_ensemble.h5",
    "ecg_fitness": REPO_ROOT / "Data" / "rppg_ensemble" / "ecg_fitness_ensemble.h5",
}

DATASET_SQI_THRESHOLDS = {
    "mcd_rppg": 0.10,
    "ubfc_rppg": 0.07,
    "ubfc_phys": 0.05,
    "ecg_fitness": 0.07,
}

CHANNELS = ("pos", "chrom", "green")
WINDOW_SEC = 8.0
TARGET_FRAMES = 240
SEED = 42


def parse_float_csv(text: str) -> list[float]:
    """Parse a comma-separated list of floats."""
    values = []
    for raw in text.split(","):
        raw = raw.strip()
        if raw:
            values.append(float(raw))
    return values


def zfloat(value) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def mean_or_none(values: list[float | None]) -> float | None:
    """Return the mean of available values, or None if empty."""
    values = [v for v in values if v is not None]
    return float(sum(values) / len(values)) if values else None


def median_or_none(values: list[float | None]) -> float | None:
    """Return the median of available values, or None if empty."""
    values = [v for v in values if v is not None]
    return float(statistics.median(values)) if values else None


def label_stability_bucket(hr_range: float | None) -> str:
    """Classify a window by HR range inside the label window."""
    if hr_range is None:
        return "unknown"
    if hr_range < 10.0:
        return "stable_lt_10_bpm"
    if hr_range < 25.0:
        return "moderate_10_25_bpm"
    if hr_range < 50.0:
        return "unstable_25_50_bpm"
    return "highly_unstable_ge_50_bpm"


def collect_subject_keys() -> list[str]:
    """Collect dataset-prefixed subject keys from all ensemble HDF5 files."""
    keys = []
    for dataset_name, h5_path in DATASETS.items():
        with h5py.File(h5_path, "r") as h5:
            for subject_id in h5["subjects"].keys():
                keys.append(f"{dataset_name}__{subject_id}")
    return sorted(keys)


def build_subject_split_map() -> dict[str, str]:
    """Reproduce the training notebook subject split."""
    keys = np.asarray(collect_subject_keys())
    rng = np.random.default_rng(SEED)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    split_map = {}
    for key in keys[:n_train]:
        split_map[str(key)] = "train"
    for key in keys[n_train : n_train + n_val]:
        split_map[str(key)] = "val"
    for key in keys[n_train + n_val :]:
        split_map[str(key)] = "test"

    return split_map


def estimate_channel(signal: np.ndarray, fps: float) -> dict:
    """Estimate dominant BPM and SQI for one rPPG channel."""
    result = estimate_spectral_sqi(signal=signal, fps=fps)
    return {
        "bpm": zfloat(result.dominant_bpm),
        "sqi": zfloat(result.sqi),
        "status": result.status,
    }


def spectral_consensus(signals: dict[str, np.ndarray], fps: float) -> dict:
    """Compute per-channel spectral summaries and a median consensus HR."""
    channels = {
        channel: estimate_channel(signals[channel], fps=fps)
        for channel in CHANNELS
    }

    usable = [
        row["bpm"]
        for row in channels.values()
        if row["bpm"] is not None and row["sqi"] is not None and row["sqi"] >= 0.30
    ]

    consensus = float(np.median(usable)) if usable else None
    spread = float(max(usable) - min(usable)) if len(usable) >= 2 else None

    return {
        "channels": channels,
        "consensus_bpm": consensus,
        "spread_bpm": spread,
    }


def make_tensor(signals: dict[str, np.ndarray], bundle):
    """Convert POS, CHROM, and GREEN signals to the active model tensor."""
    input_config = bundle.model_spec["input"]
    config = WindowConfig(
        window_seconds=float(input_config["window_seconds"]),
        target_frames=int(input_config["target_frames"]),
        channel_names=tuple(input_config["channel_names"]),
        normalization=str(input_config["normalization"]),
    )
    return make_model_window_from_channels(
        pos=signals["pos"],
        chrom=signals["chrom"],
        green=signals["green"],
        config=config,
    )


def predict_model_hr(signals: dict[str, np.ndarray], bundle) -> float:
    """Run direct CRVSE model inference for one candidate signal set."""
    tensor = make_tensor(signals=signals, bundle=bundle)
    return float(predict_hr_from_tensor(tensor, bundle)[0].value)


def get_window_target(group, start: int, end: int) -> dict:
    """Return window-local HR label statistics."""
    if "hr_continuous" not in group:
        hr_mean = zfloat(group.attrs.get("hr_mean", np.nan))
        return {
            "mean": hr_mean,
            "median": None,
            "min": None,
            "max": None,
            "range": None,
            "nan_frac": None,
            "stability_bucket": "unknown",
        }

    hr = group["hr_continuous"][start:end].astype(np.float32)
    nan_frac = float(np.mean(np.isnan(hr)))
    hr = hr[np.isfinite(hr)]

    if len(hr) == 0:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "range": None,
            "nan_frac": nan_frac,
            "stability_bucket": "unknown",
        }

    hr_min = float(np.min(hr))
    hr_max = float(np.max(hr))
    hr_range = hr_max - hr_min

    return {
        "mean": float(np.mean(hr)),
        "median": float(np.median(hr)),
        "min": hr_min,
        "max": hr_max,
        "range": hr_range,
        "nan_frac": nan_frac,
        "stability_bucket": label_stability_bucket(hr_range),
    }


def select_evenly(items: list, max_count: int) -> list:
    """Select up to max_count items evenly across a list."""
    if len(items) <= max_count:
        return items
    indices = np.linspace(0, len(items) - 1, num=max_count, dtype=int)
    return [items[int(index)] for index in indices]


def eligible_recording_ids(h5: h5py.File, dataset_name: str) -> list[tuple[str, str]]:
    """List SQI-passing recordings that contain ROI RGB data."""
    sqi_gate = DATASET_SQI_THRESHOLDS[dataset_name]
    selected = []

    for subject_id in sorted(h5["subjects"].keys()):
        recordings = h5["subjects"][subject_id]["recordings"]
        for recording_id in sorted(recordings.keys()):
            group = recordings[recording_id]
            if "roi_rgb" not in group:
                continue
            if float(group.attrs.get("sqi_ensemble", 1.0)) < sqi_gate:
                continue
            selected.append((subject_id, recording_id))

    return selected


def select_windows(group, fps: float, windows_per_recording: int) -> list[tuple[int, int]]:
    """Select evenly spaced 8-second windows from one recording."""
    n_frames = int(group["roi_rgb"].shape[0])
    window_frames = int(round(WINDOW_SEC * fps))

    if n_frames < window_frames:
        return []

    candidate_ends = np.linspace(
        window_frames,
        n_frames,
        num=windows_per_recording + 2,
        dtype=int,
    )[1:-1]

    windows = []
    for end in candidate_ends:
        start = int(end - window_frames)
        if start >= 0 and end <= n_frames:
            windows.append((start, int(end)))

    return windows


def simulate_roi_sampling(
    roi_rgb_full: np.ndarray,
    source_fps: float,
    start_frame: int,
    end_frame: int,
    simulated_fps: float,
) -> np.ndarray:
    """Downsample ROI RGB frames to a simulated live sampling rate."""
    start_s = start_frame / source_fps
    end_s = end_frame / source_fps

    if simulated_fps <= 0:
        raise ValueError(f"simulated_fps must be positive, got {simulated_fps}.")

    sample_times = np.arange(start_s, end_s, 1.0 / simulated_fps)
    if len(sample_times) < 2 or sample_times[-1] < end_s - (0.75 / simulated_fps):
        sample_times = np.append(sample_times, end_s - (1.0 / source_fps))

    frame_indices = np.rint(sample_times * source_fps).astype(int)
    frame_indices = np.clip(frame_indices, start_frame, end_frame - 1)
    frame_indices = np.unique(frame_indices)

    return roi_rgb_full[frame_indices].astype(np.float32)


def build_source_buffer_candidate(
    roi_rgb_full: np.ndarray,
    start: int,
    end: int,
    source_fps: float,
    buffer_seconds: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build a local-buffer candidate at original source FPS."""
    window_frames = end - start
    buffer_frames = int(round(buffer_seconds * source_fps))
    buffer_start = max(0, end - buffer_frames)

    roi_rgb_buffer = roi_rgb_full[buffer_start:end]
    buffer_raw = build_training_style_signals(roi_rgb_buffer, source_fps)
    relative_start = max(0, len(roi_rgb_buffer) - window_frames)

    signals = {
        channel: model_ready(buffer_raw[channel][relative_start:], TARGET_FRAMES)
        for channel in CHANNELS
    }

    return signals, {
        "preprocessing_status": "ok",
        "preprocessing_error": None,
        "model_buffer_seconds": len(roi_rgb_buffer) / source_fps,
        "simulated_fps": None,
        "sample_count": int(len(roi_rgb_buffer)),
        "window_sample_count": int(window_frames),
    }


def build_simulated_buffer_candidate(
    roi_rgb_full: np.ndarray,
    start: int,
    end: int,
    source_fps: float,
    buffer_seconds: float,
    simulated_fps: float,
) -> tuple[dict[str, np.ndarray] | None, dict]:
    """Build a local-buffer candidate after simulated live-rate sampling."""
    end_s = end / source_fps
    buffer_start_s = max(0.0, end_s - buffer_seconds)
    buffer_start_frame = int(round(buffer_start_s * source_fps))
    buffer_start_frame = max(0, min(buffer_start_frame, end - 1))

    roi_rgb_buffer = simulate_roi_sampling(
        roi_rgb_full=roi_rgb_full,
        source_fps=source_fps,
        start_frame=buffer_start_frame,
        end_frame=end,
        simulated_fps=simulated_fps,
    )

    metadata = {
        "model_buffer_seconds": len(roi_rgb_buffer) / simulated_fps,
        "simulated_fps": simulated_fps,
        "sample_count": int(len(roi_rgb_buffer)),
        "window_sample_count": int(round(WINDOW_SEC * simulated_fps)),
    }

    try:
        buffer_raw = build_training_style_signals(roi_rgb_buffer, simulated_fps)
        window_samples = int(round(WINDOW_SEC * simulated_fps))
        if len(roi_rgb_buffer) < window_samples:
            raise ValueError(
                f"Too few simulated samples for model window: "
                f"{len(roi_rgb_buffer)} < {window_samples}."
            )

        relative_start = len(roi_rgb_buffer) - window_samples
        signals = {
            channel: model_ready(buffer_raw[channel][relative_start:], TARGET_FRAMES)
            for channel in CHANNELS
        }

    except ValueError as exc:
        metadata["preprocessing_status"] = "failed"
        metadata["preprocessing_error"] = str(exc)
        return None, metadata

    metadata["preprocessing_status"] = "ok"
    metadata["preprocessing_error"] = None
    return signals, metadata


def build_candidates(
    group,
    start: int,
    end: int,
    source_fps: float,
    buffer_seconds: float,
    simulated_fps_values: list[float],
) -> dict[str, tuple[dict[str, np.ndarray] | None, dict]]:
    """Build all preprocessing candidates for one audit window."""
    roi_rgb_full = group["roi_rgb"][:].astype(np.float32)

    stored_full = {
        channel: group[f"rppg_{channel}"][:].astype(np.float32)
        for channel in CHANNELS
    }

    stored_reference = {
        channel: model_ready(stored_full[channel][start:end], TARGET_FRAMES)
        for channel in CHANNELS
    }

    full_recomputed_raw = build_training_style_signals(roi_rgb_full, source_fps)
    full_recomputed = {
        channel: model_ready(full_recomputed_raw[channel][start:end], TARGET_FRAMES)
        for channel in CHANNELS
    }

    current_live = build_current_live_style_signals(
        roi_rgb_window=roi_rgb_full[start:end],
        target_frames=TARGET_FRAMES,
    )

    candidates = {
        "stored_reference": (
            stored_reference,
            {
                "preprocessing_status": "ok",
                "preprocessing_error": None,
                "model_buffer_seconds": None,
                "simulated_fps": None,
                "sample_count": int(end - start),
                "window_sample_count": int(end - start),
            },
        ),
        "full_recomputed": (
            full_recomputed,
            {
                "preprocessing_status": "ok",
                "preprocessing_error": None,
                "model_buffer_seconds": None,
                "simulated_fps": None,
                "sample_count": int(len(roi_rgb_full)),
                "window_sample_count": int(end - start),
            },
        ),
        "current_live_compact": (
            current_live,
            {
                "preprocessing_status": "ok",
                "preprocessing_error": None,
                "model_buffer_seconds": WINDOW_SEC,
                "simulated_fps": None,
                "sample_count": int(end - start),
                "window_sample_count": int(end - start),
            },
        ),
    }

    source_buffer, source_meta = build_source_buffer_candidate(
        roi_rgb_full=roi_rgb_full,
        start=start,
        end=end,
        source_fps=source_fps,
        buffer_seconds=buffer_seconds,
    )
    candidates["training_buffer_source_fps"] = (source_buffer, source_meta)

    for simulated_fps in simulated_fps_values:
        signals, metadata = build_simulated_buffer_candidate(
            roi_rgb_full=roi_rgb_full,
            start=start,
            end=end,
            source_fps=source_fps,
            buffer_seconds=buffer_seconds,
            simulated_fps=simulated_fps,
        )
        mode = f"training_buffer_sim_{simulated_fps:g}fps"
        candidates[mode] = (signals, metadata)

    return candidates


def make_failed_metrics_row(
    base_row: dict,
    reference: dict[str, np.ndarray],
    metadata: dict,
) -> dict:
    """Create an audit row for a failed preprocessing candidate."""
    row = dict(base_row)
    row.update(metadata)
    row.update(
        {
            "model_hr": None,
            "model_abs_error": None,
            "spectral_consensus_hr": None,
            "spectral_abs_error": None,
            "spectral_spread_bpm": None,
        }
    )

    for channel in CHANNELS:
        row[f"{channel}_bpm"] = None
        row[f"{channel}_sqi"] = None
        row[f"{channel}_corr_vs_stored"] = None

    return row


def make_metrics_row(
    base_row: dict,
    signals: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    metadata: dict,
    bundle,
) -> dict:
    """Create an audit row with spectral and model metrics."""
    signal_fps = TARGET_FRAMES / WINDOW_SEC
    spectral = spectral_consensus(signals=signals, fps=signal_fps)
    model_hr = predict_model_hr(signals=signals, bundle=bundle)
    target_hr = base_row["target_hr_mean"]

    row = dict(base_row)
    row.update(metadata)
    row.update(
        {
            "model_hr": model_hr,
            "model_abs_error": None
            if target_hr is None
            else abs(model_hr - target_hr),
            "spectral_consensus_hr": spectral["consensus_bpm"],
            "spectral_abs_error": None
            if target_hr is None or spectral["consensus_bpm"] is None
            else abs(spectral["consensus_bpm"] - target_hr),
            "spectral_spread_bpm": spectral["spread_bpm"],
        }
    )

    for channel in CHANNELS:
        row[f"{channel}_bpm"] = spectral["channels"][channel]["bpm"]
        row[f"{channel}_sqi"] = spectral["channels"][channel]["sqi"]
        row[f"{channel}_corr_vs_stored"] = corr(reference[channel], signals[channel])

    return row


def build_detail_rows(args) -> list[dict]:
    """Build detailed audit rows across datasets, windows, and modes."""
    split_map = build_subject_split_map()
    bundle = load_model_bundle(device=args.device)
    rows = []

    simulated_fps_values = parse_float_csv(args.simulated_fps)

    for dataset_name, h5_path in DATASETS.items():
        windows_collected = 0

        with h5py.File(h5_path, "r") as h5:
            recording_ids = eligible_recording_ids(h5, dataset_name)
            recording_ids = select_evenly(recording_ids, args.max_recordings_per_dataset)

            for subject_id, recording_id in recording_ids:
                group = h5["subjects"][subject_id]["recordings"][recording_id]
                source_fps = float(group.attrs.get("fps", 30.0))
                subject_key = f"{dataset_name}__{subject_id}"

                for start, end in select_windows(
                    group=group,
                    fps=source_fps,
                    windows_per_recording=args.windows_per_recording,
                ):
                    if windows_collected >= args.max_windows_per_dataset:
                        break

                    target = get_window_target(group, start=start, end=end)
                    candidates = build_candidates(
                        group=group,
                        start=start,
                        end=end,
                        source_fps=source_fps,
                        buffer_seconds=args.buffer_seconds,
                        simulated_fps_values=simulated_fps_values,
                    )
                    reference = candidates["stored_reference"][0]

                    for mode, (signals, metadata) in candidates.items():
                        base_row = {
                            "dataset": dataset_name,
                            "subject_key": subject_key,
                            "split": split_map.get(subject_key, "unknown"),
                            "subject_id": subject_id,
                            "recording_id": recording_id,
                            "window_index_in_dataset": windows_collected,
                            "start_s": start / source_fps,
                            "end_s": end / source_fps,
                            "source_fps": source_fps,
                            "preprocessing_mode": mode,
                            "target_hr_mean": target["mean"],
                            "target_hr_median": target["median"],
                            "target_hr_min": target["min"],
                            "target_hr_max": target["max"],
                            "target_hr_range": target["range"],
                            "target_nan_frac": target["nan_frac"],
                            "label_stability_bucket": target["stability_bucket"],
                        }

                        if signals is None:
                            row = make_failed_metrics_row(
                                base_row=base_row,
                                reference=reference,
                                metadata=metadata,
                            )
                        else:
                            row = make_metrics_row(
                                base_row=base_row,
                                signals=signals,
                                reference=reference,
                                metadata=metadata,
                                bundle=bundle,
                            )

                        rows.append(row)

                    windows_collected += 1

                if windows_collected >= args.max_windows_per_dataset:
                    break

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise RuntimeError("No rows to write.")

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(rows: list[dict]) -> list[dict]:
    """Aggregate detail rows by dataset, split, mode, FPS, and label stability."""
    groups = {}

    for row in rows:
        key = (
            row["dataset"],
            row["split"],
            row["preprocessing_mode"],
            row["simulated_fps"],
            row["label_stability_bucket"],
        )
        groups.setdefault(key, []).append(row)

    summary_rows = []

    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        dataset, split, mode, simulated_fps, stability_bucket = key

        model_errors = [zfloat(row["model_abs_error"]) for row in group_rows]
        spectral_errors = [zfloat(row["spectral_abs_error"]) for row in group_rows]

        summary_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "preprocessing_mode": mode,
                "simulated_fps": simulated_fps,
                "label_stability_bucket": stability_bucket,
                "rows": len(group_rows),
                "ok_rows": sum(1 for row in group_rows if row["preprocessing_status"] == "ok"),
                "failed_rows": sum(1 for row in group_rows if row["preprocessing_status"] != "ok"),
                "model_mae_mean": mean_or_none(model_errors),
                "model_mae_median": median_or_none(model_errors),
                "spectral_mae_mean": mean_or_none(spectral_errors),
                "spectral_mae_median": median_or_none(spectral_errors),
                "pos_corr_mean": mean_or_none(
                    [zfloat(row["pos_corr_vs_stored"]) for row in group_rows]
                ),
                "chrom_corr_mean": mean_or_none(
                    [zfloat(row["chrom_corr_vs_stored"]) for row in group_rows]
                ),
                "green_corr_mean": mean_or_none(
                    [zfloat(row["green_corr_vs_stored"]) for row in group_rows]
                ),
                "target_hr_range_mean": mean_or_none(
                    [zfloat(row["target_hr_range"]) for row in group_rows]
                ),
                "spectral_spread_mean": mean_or_none(
                    [zfloat(row["spectral_spread_bpm"]) for row in group_rows]
                ),
            }
        )

    return summary_rows


def parse_args():
    """Parse command-line options for the audit run."""
    parser = argparse.ArgumentParser(
        description="Build a live-compatible CRVSE model-vs-spectral audit table."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "Data" / "audit_live_compatible_window_table.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "Data" / "audit_live_compatible_window_summary.csv",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--buffer-seconds", type=float, default=12.0)
    parser.add_argument("--simulated-fps", type=str, default="5,7.5,10,15,30")
    parser.add_argument("--max-recordings-per-dataset", type=int, default=12)
    parser.add_argument("--windows-per-recording", type=int, default=4)
    parser.add_argument("--max-windows-per-dataset", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    """Run the audit and write detail and summary CSV files."""
    args = parse_args()

    rows = build_detail_rows(args)
    summary_rows = build_summary_rows(rows)

    write_csv(args.output_csv, rows)
    write_csv(args.summary_csv, summary_rows)

    print(f"Wrote detail rows: {len(rows)} -> {args.output_csv}")
    print(f"Wrote summary rows: {len(summary_rows)} -> {args.summary_csv}")


if __name__ == "__main__":
    main()