"""
Model prediction variance and tracking by simulated acquisition FPS.

Consumes the detail CSV written by audit_live_compatible_window_table.py and
asks a different question than that script's summary.

That script reports how large the error is. This script reports whether the
model is still responding to its input at all.

Discriminating statistics per preprocessing mode:

    slope        OLS slope of model_hr on target_hr.
                 ~1.0 means the model tracks the target.
                 ~0.0 means the model emits a near-constant value.

    std_ratio    pred_std / target_std. Fraction of the target's dynamic
                 range that survives in the predictions.

    pearson_r    Linear correlation between prediction and target.

    mae vs       Predict-mean baseline, the same test used in the MCD
    baseline     transfer-learning triage. If model MAE is not clearly below
                 the baseline, the model is adding nothing.

Windows are restricted to those that succeeded in every compared mode, so the
comparison is not confounded by which windows failed at low FPS.

This is an analysis script, not a smoke test. It is not part of
run_smoke_tests.py.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
APP_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = APP_DIR.parents[1]


def parse_float(value: str | None) -> float | None:
    """Parse a CSV field into a float, returning None when not usable."""
    if value is None:
        return None

    text = str(value).strip()

    if text == "" or text.lower() in {"none", "nan"}:
        return None

    try:
        parsed = float(text)
    except ValueError:
        return None

    if not np.isfinite(parsed):
        return None

    return parsed


def window_key(row: dict) -> tuple:
    """Build a stable identifier for one analysis window."""
    return (
        row.get("dataset", ""),
        row.get("subject_id", ""),
        row.get("recording_id", ""),
        f"{parse_float(row.get('start_s')) or 0.0:.3f}",
    )


def mode_label(row: dict) -> str:
    """Return the preprocessing mode, annotated with simulated FPS when set."""
    mode = str(row.get("preprocessing_mode", "unknown"))
    simulated_fps = parse_float(row.get("simulated_fps"))

    if simulated_fps is None:
        return mode

    return f"{mode} @ {simulated_fps:g} Hz"


def load_usable_rows(
    csv_path: Path,
    split: str,
    datasets: set[str] | None,
) -> list[dict]:
    """Load detail rows that have both a model prediction and a target."""
    usable = []

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("preprocessing_status") != "ok":
                continue

            if split != "all" and row.get("split") != split:
                continue

            if datasets is not None and row.get("dataset") not in datasets:
                continue

            model_hr = parse_float(row.get("model_hr"))
            target_hr = parse_float(row.get("target_hr_mean"))

            if model_hr is None or target_hr is None:
                continue

            row["_model_hr"] = model_hr
            row["_target_hr"] = target_hr
            row["_mode"] = mode_label(row)
            row["_key"] = window_key(row)

            usable.append(row)

    return usable


def restrict_to_common_windows(rows: list[dict]) -> tuple[list[dict], int]:
    """
    Keep only windows present in every mode.

    Returns
    -------
    tuple[list[dict], int]
        Filtered rows, and the number of common windows retained.
    """
    modes = {row["_mode"] for row in rows}
    keys_by_mode = {
        mode: {row["_key"] for row in rows if row["_mode"] == mode}
        for mode in modes
    }

    if not keys_by_mode:
        return [], 0

    common_keys = set.intersection(*keys_by_mode.values())
    filtered = [row for row in rows if row["_key"] in common_keys]

    return filtered, len(common_keys)


def summarize_mode(rows: list[dict]) -> dict:
    """Compute tracking and variance statistics for one preprocessing mode."""
    predictions = np.asarray([row["_model_hr"] for row in rows], dtype=np.float64)
    targets = np.asarray([row["_target_hr"] for row in rows], dtype=np.float64)

    target_std = float(np.std(targets))
    pred_std = float(np.std(predictions))
    target_variance = float(np.var(targets))

    if target_variance > 1e-9:
        covariance = float(np.mean((targets - targets.mean()) * (predictions - predictions.mean())))
        slope = covariance / target_variance
    else:
        slope = float("nan")

    if target_std > 1e-9 and pred_std > 1e-9:
        pearson_r = float(np.corrcoef(targets, predictions)[0, 1])
    else:
        pearson_r = float("nan")

    model_mae = float(np.mean(np.abs(predictions - targets)))
    baseline_mae = float(np.mean(np.abs(targets - targets.mean())))

    return {
        "n": len(rows),
        "target_mean": float(targets.mean()),
        "target_std": target_std,
        "pred_mean": float(predictions.mean()),
        "pred_std": pred_std,
        "std_ratio": pred_std / target_std if target_std > 1e-9 else float("nan"),
        "slope": slope,
        "pearson_r": pearson_r,
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "beats_baseline": model_mae < baseline_mae,
    }


def format_row(mode: str, stats: dict) -> str:
    """Format one summary line."""
    return (
        f"{mode:<34} "
        f"{stats['n']:>5d} "
        f"{stats['target_std']:>9.2f} "
        f"{stats['pred_mean']:>9.2f} "
        f"{stats['pred_std']:>8.2f} "
        f"{stats['std_ratio']:>8.3f} "
        f"{stats['slope']:>7.3f} "
        f"{stats['pearson_r']:>7.3f} "
        f"{stats['model_mae']:>8.2f} "
        f"{stats['baseline_mae']:>9.2f} "
        f"{'yes' if stats['beats_baseline'] else 'NO':>5}"
    )


def main() -> None:
    """Run the prediction variance analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT / "Data" / "audit_live_compatible_window_table.csv",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="Comma-separated dataset names, or 'all'.",
    )
    parser.add_argument(
        "--common-windows-only",
        action="store_true",
        default=True,
        help="Restrict to windows that succeeded in every mode.",
    )
    args = parser.parse_args()

    datasets = None

    if args.datasets != "all":
        datasets = {name.strip() for name in args.datasets.split(",") if name.strip()}

    rows = load_usable_rows(
        csv_path=args.input_csv,
        split=args.split,
        datasets=datasets,
    )

    if not rows:
        raise RuntimeError(
            f"No usable rows in {args.input_csv} for split={args.split}."
        )

    common_count = None

    if args.common_windows_only:
        rows, common_count = restrict_to_common_windows(rows)

        if not rows:
            raise RuntimeError("No windows succeeded across every mode.")

    modes = sorted({row["_mode"] for row in rows})

    print(f"input   : {args.input_csv}")
    print(f"split   : {args.split}")
    print(f"datasets: {args.datasets}")

    if common_count is not None:
        print(f"windows : {common_count} common across {len(modes)} modes")

    print("")
    print(
        f"{'mode':<34} {'n':>5} {'tgt_std':>9} {'pred_mean':>9} "
        f"{'pred_sd':>8} {'sd_ratio':>8} {'slope':>7} {'r':>7} "
        f"{'mae':>8} {'base_mae':>9} {'beats':>5}"
    )

    for mode in modes:
        mode_rows = [row for row in rows if row["_mode"] == mode]
        print(format_row(mode, summarize_mode(mode_rows)))

    print("")
    print("Reading the table:")
    print("  slope near 1.0  -> model tracks the target")
    print("  slope near 0.0  -> model emits a near-constant value")
    print("  sd_ratio -> 0   -> prediction dynamic range has collapsed")
    print("  beats = NO      -> model is worse than predicting the target mean")


if __name__ == "__main__":
    main()