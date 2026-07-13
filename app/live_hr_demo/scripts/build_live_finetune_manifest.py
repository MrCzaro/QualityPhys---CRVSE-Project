from __future__ import annotations

"""
Build a live-compatible fine-tuning manifest for CRVSE.

This script does not train a model and does not write tensor files. It creates a
CSV manifest that defines which windows belong to the proposed fine-tuning
experiment, which windows are evaluation-only, and why.

Main policy:
- MCD-rPPG, UBFC-rPPG, and UBFC-Phys may be used for fine-tuning.
- ECG-Fitness is excluded from first-pass fine-tuning.
- ECG-Fitness remains in evaluation as an out-of-domain high-HR/exercise stress test.
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
APP_DIR = REPO_ROOT / "app" / "live_hr_demo"
sys.path.insert(0, str(APP_DIR))

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

TRAINING_DATASETS = {"mcd_rppg", "ubfc_rppg", "ubfc_phys"}
OUT_OF_DOMAIN_DATASETS = {"ecg_fitness"}

SEED = 42
WINDOW_SEC = 8.0
STRIDE_SEC = 4.0
BUFFER_SEC = 12.0
TARGET_FRAMES = 240
HR_MIN = 40.0
HR_MAX = 180.0
MAX_NAN_FRAC = 0.10


def zfloat(value) -> float | None:
    """Return a finite float or None."""
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    return value if np.isfinite(value) else None


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
    subject_keys = []

    for dataset_name, h5_path in DATASETS.items():
        with h5py.File(h5_path, "r") as h5:
            for subject_id in h5["subjects"].keys():
                subject_keys.append(f"{dataset_name}__{subject_id}")

    return sorted(subject_keys)


def build_subject_split_map() -> dict[str, str]:
    """Reproduce the training notebooks' subject-level split."""
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


def get_window_target(group, start: int, end: int) -> dict:
    """Return HR label statistics for one window."""
    if "hr_continuous" not in group:
        hr_mean = zfloat(group.attrs.get("hr_mean", np.nan))
        return {
            "target_hr_mean": hr_mean,
            "target_hr_median": None,
            "target_hr_min": None,
            "target_hr_max": None,
            "target_hr_range": None,
            "target_nan_frac": None,
            "label_stability_bucket": "unknown",
        }

    hr = group["hr_continuous"][start:end].astype(np.float32)
    nan_frac = float(np.mean(np.isnan(hr)))
    valid_hr = hr[np.isfinite(hr)]

    if len(valid_hr) == 0:
        return {
            "target_hr_mean": None,
            "target_hr_median": None,
            "target_hr_min": None,
            "target_hr_max": None,
            "target_hr_range": None,
            "target_nan_frac": nan_frac,
            "label_stability_bucket": "unknown",
        }

    hr_min = float(np.min(valid_hr))
    hr_max = float(np.max(valid_hr))
    hr_range = hr_max - hr_min

    return {
        "target_hr_mean": float(np.mean(valid_hr)),
        "target_hr_median": float(np.median(valid_hr)),
        "target_hr_min": hr_min,
        "target_hr_max": hr_max,
        "target_hr_range": hr_range,
        "target_nan_frac": nan_frac,
        "label_stability_bucket": label_stability_bucket(hr_range),
    }


def signal_window_is_valid(group, start: int, end: int) -> tuple[bool, str]:
    """Check that stored POS, CHROM, and GREEN windows are usable."""
    for channel in ("pos", "chrom", "green"):
        dataset_name = f"rppg_{channel}"

        if dataset_name not in group:
            return False, f"missing_{dataset_name}"

        signal = group[dataset_name][start:end].astype(np.float32)

        if np.any(np.isnan(signal)):
            return False, f"{dataset_name}_contains_nan"

        if float(np.std(signal)) < 1e-6:
            return False, f"{dataset_name}_flat"

    return True, "ok"


def window_policy(
    dataset_name: str,
    split: str,
    target: dict,
    signal_ok: bool,
    signal_reason: str,
) -> dict:
    """Decide whether a window is trainable, eval-only, or excluded."""
    target_hr = target["target_hr_mean"]
    target_nan_frac = target["target_nan_frac"]

    if target_hr is None:
        return {
            "window_role": "excluded",
            "include_in_finetune": False,
            "include_in_eval": False,
            "reason": "missing_target_hr",
        }

    if not (HR_MIN <= target_hr <= HR_MAX):
        return {
            "window_role": "excluded",
            "include_in_finetune": False,
            "include_in_eval": False,
            "reason": "target_hr_out_of_range",
        }

    if target_nan_frac is not None and target_nan_frac > MAX_NAN_FRAC:
        return {
            "window_role": "excluded",
            "include_in_finetune": False,
            "include_in_eval": False,
            "reason": "too_many_missing_hr_labels",
        }

    if not signal_ok:
        return {
            "window_role": "excluded",
            "include_in_finetune": False,
            "include_in_eval": False,
            "reason": signal_reason,
        }

    if dataset_name in OUT_OF_DOMAIN_DATASETS:
        return {
            "window_role": "ood_eval",
            "include_in_finetune": False,
            "include_in_eval": True,
            "reason": "ecg_fitness_reserved_as_ood_stress_test",
        }

    if dataset_name not in TRAINING_DATASETS:
        return {
            "window_role": "eval_only",
            "include_in_finetune": False,
            "include_in_eval": True,
            "reason": "dataset_not_in_first_finetune_set",
        }

    if split == "train":
        return {
            "window_role": "finetune_train",
            "include_in_finetune": True,
            "include_in_eval": True,
            "reason": "eligible_training_window",
        }

    if split == "val":
        return {
            "window_role": "finetune_val",
            "include_in_finetune": False,
            "include_in_eval": True,
            "reason": "validation_window",
        }

    return {
        "window_role": "finetune_test",
        "include_in_finetune": False,
        "include_in_eval": True,
        "reason": "test_window",
    }


def iter_manifest_rows() -> list[dict]:
    """Build manifest rows for all eligible HDF5 windows."""
    split_map = build_subject_split_map()
    rows = []

    for dataset_name, h5_path in DATASETS.items():
        sqi_gate = DATASET_SQI_THRESHOLDS[dataset_name]

        with h5py.File(h5_path, "r") as h5:
            for subject_id in sorted(h5["subjects"].keys()):
                subject_key = f"{dataset_name}__{subject_id}"
                split = split_map.get(subject_key, "unknown")
                recordings = h5["subjects"][subject_id]["recordings"]

                for recording_id in sorted(recordings.keys()):
                    group = recordings[recording_id]

                    if "roi_rgb" not in group:
                        continue

                    recording_sqi = float(group.attrs.get("sqi_ensemble", 1.0))
                    if recording_sqi < sqi_gate:
                        continue

                    fps = float(group.attrs.get("fps", 30.0))
                    n_frames = int(group["roi_rgb"].shape[0])
                    window_frames = int(fps * WINDOW_SEC)
                    stride_frames = int(fps * STRIDE_SEC)

                    if n_frames < window_frames:
                        continue

                    for start in range(0, n_frames - window_frames + 1, stride_frames):
                        end = start + window_frames

                        target = get_window_target(group, start=start, end=end)
                        signal_ok, signal_reason = signal_window_is_valid(
                            group,
                            start=start,
                            end=end,
                        )

                        policy = window_policy(
                            dataset_name=dataset_name,
                            split=split,
                            target=target,
                            signal_ok=signal_ok,
                            signal_reason=signal_reason,
                        )

                        rows.append(
                            {
                                "dataset": dataset_name,
                                "subject_key": subject_key,
                                "split": split,
                                "subject_id": subject_id,
                                "recording_id": recording_id,
                                "start_frame": start,
                                "end_frame": end,
                                "start_s": start / fps,
                                "end_s": end / fps,
                                "source_fps": fps,
                                "window_seconds": WINDOW_SEC,
                                "stride_seconds": STRIDE_SEC,
                                "model_buffer_seconds": BUFFER_SEC,
                                "target_frames": TARGET_FRAMES,
                                "recording_sqi_ensemble": recording_sqi,
                                "dataset_sqi_gate": sqi_gate,
                                **target,
                                "signal_window_ok": signal_ok,
                                "signal_window_reason": signal_reason,
                                **policy,
                            }
                        )

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to CSV."""
    if not rows:
        raise RuntimeError("No manifest rows were generated.")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> None:
    """Print a compact manifest summary."""
    total = len(rows)
    trainable = sum(1 for row in rows if row["include_in_finetune"])
    eval_rows = sum(1 for row in rows if row["include_in_eval"])

    print(f"Total manifest rows: {total}")
    print(f"Fine-tuning rows:    {trainable}")
    print(f"Evaluation rows:     {eval_rows}")
    print()

    print("Rows by dataset and role:")
    counts = {}
    for row in rows:
        key = (row["dataset"], row["window_role"])
        counts[key] = counts.get(key, 0) + 1

    for key in sorted(counts):
        dataset_name, role = key
        print(f"  {dataset_name:<12} {role:<16} {counts[key]:>6}")

    print()
    print("Rows by label stability:")
    stability_counts = {}
    for row in rows:
        key = row["label_stability_bucket"]
        stability_counts[key] = stability_counts.get(key, 0) + 1

    for key in sorted(stability_counts):
        print(f"  {key:<28} {stability_counts[key]:>6}")


def parse_args():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(
        description="Build a CRVSE live-compatible fine-tuning manifest."
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=REPO_ROOT / "Data" / "live_finetune_manifest.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Build and write the fine-tuning manifest."""
    args = parse_args()
    rows = iter_manifest_rows()
    write_csv(args.output_csv, rows)
    summarize(rows)
    print()
    print(f"Wrote manifest: {args.output_csv}")


if __name__ == "__main__":
    main()