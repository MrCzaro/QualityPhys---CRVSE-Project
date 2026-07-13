from __future__ import annotations

"""
Run frozen CRVSE PhysFormer predictions on the live fine-tuning manifest.

This script does not train or fine-tune anything. It measures the current checkpoint
on the exact manifest that will define later notebook-based fine-tuning work.

Default modes:
- stored_reference
- training_buffer_source_fps
- training_buffer_sim_15fps
- training_buffer_sim_10fps
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
APP_DIR = REPO_ROOT / "app" / "live_hr_demo"
sys.path.insert(0, str(APP_DIR))

from inference.predictor import predict_hr_from_tensor
from models.loader import load_model_bundle
from rppg.sqi import estimate_spectral_sqi
from rppg.windowing import WindowConfig, make_model_window_from_channels
from scripts.audit_preprocessing_parity import (
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

CHANNELS = ("pos", "chrom", "green")
WINDOW_SECONDS = 8.0
TARGET_FRAMES = 240


def parse_modes(text: str) -> list[str]:
    """Parse comma-separated preprocessing mode names."""
    return [item.strip() for item in text.split(",") if item.strip()]


def zfloat(value) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if math.isfinite(value) else None


def parse_bool(value) -> bool:
    """Parse bool-like CSV values."""
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def mean_or_none(values: list[float | None]) -> float | None:
    """Return mean of available values."""
    values = [value for value in values if value is not None]
    return float(sum(values) / len(values)) if values else None


def median_or_none(values: list[float | None]) -> float | None:
    """Return median of available values."""
    values = [value for value in values if value is not None]
    return float(statistics.median(values)) if values else None


def percentile_or_none(values: list[float | None], percentile: float) -> float | None:
    """Return a percentile using linear interpolation."""
    values = sorted(value for value in values if value is not None)

    if not values:
        return None

    index = (len(values) - 1) * percentile
    low = int(math.floor(index))
    high = int(math.ceil(index))

    if low == high:
        return float(values[low])

    return float(values[low] * (high - index) + values[high] * (index - low))


def load_manifest_rows(path: Path, max_eval_windows: int | None) -> list[dict]:
    """Load evaluation rows from the fine-tuning manifest."""
    rows = []

    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if not parse_bool(row["include_in_eval"]):
                continue

            rows.append(row)

            if max_eval_windows is not None and len(rows) >= max_eval_windows:
                break

    return rows


def group_rows_by_recording(rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    """Group manifest rows by dataset, subject, and recording."""
    grouped = defaultdict(list)

    for row in rows:
        key = (row["dataset"], row["subject_id"], row["recording_id"])
        grouped[key].append(row)

    return dict(grouped)


def estimate_channel(signal: np.ndarray, fps: float) -> dict:
    """Estimate dominant BPM and SQI for one signal channel."""
    result = estimate_spectral_sqi(signal=signal, fps=fps)
    return {
        "bpm": zfloat(result.dominant_bpm),
        "sqi": zfloat(result.sqi),
        "status": result.status,
    }


def spectral_consensus(signals: dict[str, np.ndarray], fps: float) -> dict:
    """Compute per-channel spectral summaries and median consensus HR."""
    channel_rows = {
        channel: estimate_channel(signals[channel], fps=fps)
        for channel in CHANNELS
    }

    usable = [
        row["bpm"]
        for row in channel_rows.values()
        if row["bpm"] is not None and row["sqi"] is not None and row["sqi"] >= 0.30
    ]

    consensus = float(np.median(usable)) if usable else None
    spread = float(max(usable) - min(usable)) if len(usable) >= 2 else None

    return {
        "channels": channel_rows,
        "consensus_bpm": consensus,
        "spread_bpm": spread,
    }


def make_tensor(signals: dict[str, np.ndarray], bundle) -> torch.Tensor:
    """Convert POS, CHROM, and GREEN signals into the model tensor."""
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


def build_stored_reference(
    stored_signals: dict[str, np.ndarray],
    start: int,
    end: int,
) -> dict[str, np.ndarray]:
    """Build model-ready signals from stored HDF5 rPPG channels."""
    return {
        channel: model_ready(stored_signals[channel][start:end], TARGET_FRAMES)
        for channel in CHANNELS
    }


def build_source_buffer_candidate(
    roi_rgb_full: np.ndarray,
    start: int,
    end: int,
    source_fps: float,
    buffer_seconds: float,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build a live-compatible local-buffer candidate at source FPS."""
    window_frames = end - start
    buffer_frames = int(source_fps * buffer_seconds)
    buffer_start = max(0, end - buffer_frames)

    roi_rgb_buffer = roi_rgb_full[buffer_start:end]
    buffer_signals = build_training_style_signals(roi_rgb_buffer, source_fps)

    relative_start = max(0, len(roi_rgb_buffer) - window_frames)

    signals = {
        channel: model_ready(buffer_signals[channel][relative_start:], TARGET_FRAMES)
        for channel in CHANNELS
    }

    metadata = {
        "preprocessing_status": "ok",
        "preprocessing_error": None,
        "model_buffer_seconds": len(roi_rgb_buffer) / source_fps,
        "simulated_fps": None,
        "sample_count": int(len(roi_rgb_buffer)),
        "window_sample_count": int(window_frames),
    }

    return signals, metadata


def simulate_roi_sampling(
    roi_rgb_full: np.ndarray,
    source_fps: float,
    start_frame: int,
    end_frame: int,
    simulated_fps: float,
) -> np.ndarray:
    """Sample stored ROI RGB frames at a simulated live rate."""
    start_s = start_frame / source_fps
    end_s = end_frame / source_fps

    sample_times = np.arange(start_s, end_s, 1.0 / simulated_fps)

    if len(sample_times) < 2 or sample_times[-1] < end_s - (0.75 / simulated_fps):
        sample_times = np.append(sample_times, end_s - (1.0 / source_fps))

    frame_indices = np.rint(sample_times * source_fps).astype(int)
    frame_indices = np.clip(frame_indices, start_frame, end_frame - 1)
    frame_indices = np.unique(frame_indices)

    return roi_rgb_full[frame_indices].astype(np.float32)


def build_simulated_buffer_candidate(
    roi_rgb_full: np.ndarray,
    start: int,
    end: int,
    source_fps: float,
    buffer_seconds: float,
    simulated_fps: float,
) -> tuple[dict[str, np.ndarray] | None, dict]:
    """Build a local-buffer candidate at simulated live FPS."""
    end_s = end / source_fps
    buffer_start_s = max(0.0, end_s - buffer_seconds)
    buffer_start_frame = int(source_fps * buffer_start_s)
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
        "window_sample_count": int(WINDOW_SECONDS * simulated_fps),
    }

    try:
        buffer_signals = build_training_style_signals(roi_rgb_buffer, simulated_fps)
        window_samples = int(WINDOW_SECONDS * simulated_fps)

        if len(roi_rgb_buffer) < window_samples:
            raise ValueError(
                f"Too few simulated samples for model window: "
                f"{len(roi_rgb_buffer)} < {window_samples}."
            )

        relative_start = len(roi_rgb_buffer) - window_samples

        signals = {
            channel: model_ready(buffer_signals[channel][relative_start:], TARGET_FRAMES)
            for channel in CHANNELS
        }

    except ValueError as exc:
        metadata["preprocessing_status"] = "failed"
        metadata["preprocessing_error"] = str(exc)
        return None, metadata

    metadata["preprocessing_status"] = "ok"
    metadata["preprocessing_error"] = None
    return signals, metadata


def parse_simulated_mode(mode: str) -> float | None:
    """Return simulated FPS encoded in a mode name, if present."""
    prefix = "training_buffer_sim_"
    suffix = "fps"

    if not mode.startswith(prefix) or not mode.endswith(suffix):
        return None

    return float(mode[len(prefix) : -len(suffix)])


def build_candidate(
    mode: str,
    roi_rgb_full: np.ndarray,
    stored_signals: dict[str, np.ndarray],
    start: int,
    end: int,
    source_fps: float,
    buffer_seconds: float,
) -> tuple[dict[str, np.ndarray] | None, dict]:
    """Build one preprocessing candidate for one manifest window."""
    if mode == "stored_reference":
        return build_stored_reference(stored_signals, start, end), {
            "preprocessing_status": "ok",
            "preprocessing_error": None,
            "model_buffer_seconds": None,
            "simulated_fps": None,
            "sample_count": int(end - start),
            "window_sample_count": int(end - start),
        }

    if mode == "training_buffer_source_fps":
        return build_source_buffer_candidate(
            roi_rgb_full=roi_rgb_full,
            start=start,
            end=end,
            source_fps=source_fps,
            buffer_seconds=buffer_seconds,
        )

    simulated_fps = parse_simulated_mode(mode)
    if simulated_fps is not None:
        return build_simulated_buffer_candidate(
            roi_rgb_full=roi_rgb_full,
            start=start,
            end=end,
            source_fps=source_fps,
            buffer_seconds=buffer_seconds,
            simulated_fps=simulated_fps,
        )

    raise ValueError(f"Unsupported preprocessing mode: {mode}")


def base_prediction_row(row: dict, mode: str, metadata: dict) -> dict:
    """Create the shared prediction row fields."""
    return {
        "dataset": row["dataset"],
        "subject_key": row["subject_key"],
        "split": row["split"],
        "window_role": row["window_role"],
        "subject_id": row["subject_id"],
        "recording_id": row["recording_id"],
        "start_frame": row["start_frame"],
        "end_frame": row["end_frame"],
        "start_s": row["start_s"],
        "end_s": row["end_s"],
        "source_fps": row["source_fps"],
        "preprocessing_mode": mode,
        "target_hr_mean": row["target_hr_mean"],
        "target_hr_median": row["target_hr_median"],
        "target_hr_range": row["target_hr_range"],
        "label_stability_bucket": row["label_stability_bucket"],
        "manifest_reason": row["reason"],
        **metadata,
    }


def make_failed_prediction_row(row: dict, mode: str, metadata: dict) -> dict:
    """Create a prediction row for a preprocessing failure."""
    output = base_prediction_row(row, mode, metadata)

    output.update(
        {
            "model_hr": None,
            "model_abs_error": None,
            "spectral_consensus_hr": None,
            "spectral_abs_error": None,
            "spectral_spread_bpm": None,
        }
    )

    for channel in CHANNELS:
        output[f"{channel}_bpm"] = None
        output[f"{channel}_sqi"] = None
        output[f"{channel}_corr_vs_stored"] = None

    return output


def make_prediction_row(
    row: dict,
    mode: str,
    signals: dict[str, np.ndarray],
    stored_reference: dict[str, np.ndarray],
    metadata: dict,
    model_hr: float,
) -> dict:
    """Create a prediction row with model and spectral metrics."""
    target_hr = zfloat(row["target_hr_mean"])
    spectral_fps = TARGET_FRAMES / WINDOW_SECONDS
    spectral = spectral_consensus(signals=signals, fps=spectral_fps)

    output = base_prediction_row(row, mode, metadata)
    output.update(
        {
            "model_hr": model_hr,
            "model_abs_error": None if target_hr is None else abs(model_hr - target_hr),
            "spectral_consensus_hr": spectral["consensus_bpm"],
            "spectral_abs_error": None
            if target_hr is None or spectral["consensus_bpm"] is None
            else abs(spectral["consensus_bpm"] - target_hr),
            "spectral_spread_bpm": spectral["spread_bpm"],
        }
    )

    for channel in CHANNELS:
        output[f"{channel}_bpm"] = spectral["channels"][channel]["bpm"]
        output[f"{channel}_sqi"] = spectral["channels"][channel]["sqi"]
        output[f"{channel}_corr_vs_stored"] = corr(
            stored_reference[channel],
            signals[channel],
        )

    return output


def flush_prediction_batch(
    pending_tensors: list[torch.Tensor],
    pending_payloads: list[dict],
    bundle,
    output_rows: list[dict],
) -> None:
    """Run a batch of tensors and append prediction rows."""
    if not pending_tensors:
        return

    batch = torch.cat(pending_tensors, dim=0)
    predictions = predict_hr_from_tensor(batch, bundle)

    for payload, prediction in zip(pending_payloads, predictions):
        output_rows.append(
            make_prediction_row(
                row=payload["manifest_row"],
                mode=payload["mode"],
                signals=payload["signals"],
                stored_reference=payload["stored_reference"],
                metadata=payload["metadata"],
                model_hr=float(prediction.value),
            )
        )

    pending_tensors.clear()
    pending_payloads.clear()


def build_prediction_rows(args) -> list[dict]:
    """Run frozen model predictions for manifest evaluation rows."""
    modes = parse_modes(args.modes)
    manifest_rows = load_manifest_rows(args.manifest_csv, args.max_eval_windows)
    grouped_rows = group_rows_by_recording(manifest_rows)

    bundle = load_model_bundle(device=args.device)
    output_rows = []
    pending_tensors = []
    pending_payloads = []

    processed_windows = 0

    for (dataset_name, subject_id, recording_id), rows in grouped_rows.items():
        h5_path = DATASETS[dataset_name]

        with h5py.File(h5_path, "r") as h5:
            group = h5["subjects"][subject_id]["recordings"][recording_id]
            roi_rgb_full = group["roi_rgb"][:].astype(np.float32)
            source_fps = float(group.attrs.get("fps", 30.0))

            stored_signals = {
                channel: group[f"rppg_{channel}"][:].astype(np.float32)
                for channel in CHANNELS
            }

            for row in rows:
                start = int(row["start_frame"])
                end = int(row["end_frame"])

                stored_reference = build_stored_reference(
                    stored_signals=stored_signals,
                    start=start,
                    end=end,
                )

                for mode in modes:
                    signals, metadata = build_candidate(
                        mode=mode,
                        roi_rgb_full=roi_rgb_full,
                        stored_signals=stored_signals,
                        start=start,
                        end=end,
                        source_fps=source_fps,
                        buffer_seconds=args.buffer_seconds,
                    )

                    if signals is None:
                        output_rows.append(
                            make_failed_prediction_row(
                                row=row,
                                mode=mode,
                                metadata=metadata,
                            )
                        )
                        continue

                    pending_tensors.append(make_tensor(signals=signals, bundle=bundle))
                    pending_payloads.append(
                        {
                            "manifest_row": row,
                            "mode": mode,
                            "signals": signals,
                            "stored_reference": stored_reference,
                            "metadata": metadata,
                        }
                    )

                    if len(pending_tensors) >= args.batch_size:
                        flush_prediction_batch(
                            pending_tensors=pending_tensors,
                            pending_payloads=pending_payloads,
                            bundle=bundle,
                            output_rows=output_rows,
                        )

                processed_windows += 1
                if processed_windows % args.progress_every == 0:
                    print(f"Processed manifest windows: {processed_windows}")

    flush_prediction_batch(
        pending_tensors=pending_tensors,
        pending_payloads=pending_payloads,
        bundle=bundle,
        output_rows=output_rows,
    )

    return output_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV."""
    if not rows:
        raise RuntimeError("No rows to write.")

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary_rows(rows: list[dict]) -> list[dict]:
    """Aggregate prediction rows for review."""
    groups = defaultdict(list)

    for row in rows:
        key = (
            row["dataset"],
            row["split"],
            row["window_role"],
            row["preprocessing_mode"],
            row["simulated_fps"],
            row["label_stability_bucket"],
        )
        groups[key].append(row)

    summary_rows = []

    for key, group_rows in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        dataset, split, role, mode, simulated_fps, stability = key

        model_errors = [zfloat(row["model_abs_error"]) for row in group_rows]
        spectral_errors = [zfloat(row["spectral_abs_error"]) for row in group_rows]

        summary_rows.append(
            {
                "dataset": dataset,
                "split": split,
                "window_role": role,
                "preprocessing_mode": mode,
                "simulated_fps": simulated_fps,
                "label_stability_bucket": stability,
                "rows": len(group_rows),
                "ok_rows": sum(
                    1 for row in group_rows if row["preprocessing_status"] == "ok"
                ),
                "failed_rows": sum(
                    1 for row in group_rows if row["preprocessing_status"] != "ok"
                ),
                "model_mae_mean": mean_or_none(model_errors),
                "model_mae_median": median_or_none(model_errors),
                "model_mae_p90": percentile_or_none(model_errors, 0.90),
                "spectral_mae_mean": mean_or_none(spectral_errors),
                "spectral_mae_median": median_or_none(spectral_errors),
                "spectral_mae_p90": percentile_or_none(spectral_errors, 0.90),
                "target_hr_range_mean": mean_or_none(
                    [zfloat(row["target_hr_range"]) for row in group_rows]
                ),
                "spectral_spread_mean": mean_or_none(
                    [zfloat(row["spectral_spread_bpm"]) for row in group_rows]
                ),
                "pos_corr_mean": mean_or_none(
                    [zfloat(row["pos_corr_vs_stored"]) for row in group_rows]
                ),
                "chrom_corr_mean": mean_or_none(
                    [zfloat(row["chrom_corr_vs_stored"]) for row in group_rows]
                ),
                "green_corr_mean": mean_or_none(
                    [zfloat(row["green_corr_vs_stored"]) for row in group_rows]
                ),
            }
        )

    return summary_rows


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Run frozen CRVSE baseline predictions on the live manifest."
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=REPO_ROOT / "Data" / "live_finetune_manifest.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "Data" / "live_finetune_frozen_baseline_predictions.csv",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=REPO_ROOT / "Data" / "live_finetune_frozen_baseline_summary.csv",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="stored_reference,training_buffer_source_fps,training_buffer_sim_15fps,training_buffer_sim_10fps",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--buffer-seconds", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-eval-windows", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    """Run the frozen baseline and write prediction and summary CSVs."""
    args = parse_args()

    rows = build_prediction_rows(args)
    summary_rows = build_summary_rows(rows)

    write_csv(args.output_csv, rows)
    write_csv(args.summary_csv, summary_rows)

    print(f"Wrote prediction rows: {len(rows)} -> {args.output_csv}")
    print(f"Wrote summary rows:    {len(summary_rows)} -> {args.summary_csv}")


if __name__ == "__main__":
    main()