"""
Offline calibration audit for the CRVSE PhysFormer live prediction.

The 2026-07-21 variance audit showed the model tracks its input (slope 0.635 at
20 Hz) but shrinks predictions toward a training corpus mean near 88 bpm. That
produces a positive bias for resting subjects.

This script fits candidate corrections on TRAIN subjects and evaluates them on
held-out VALIDATION and TEST subjects. Nothing here modifies the app.

Corrections tested:

    none      raw model output

    offset    subtract the mean signed bias measured on train subjects.
              Removes average bias, leaves shrinkage intact.

    linear    invert the fitted relationship pred = a + b * reference:
                  corrected = (pred - a) / b
              Removes shrinkage, but amplifies prediction noise by 1/b.

Reported per split:

    mae         mean absolute error
    bias        mean signed error, positive means over-prediction
    p90         90th percentile absolute error, matching the NB13 policy
    slope       OLS slope after correction; 1.0 means calibrated

IMPORTANT STATISTICAL CAVEAT

Shrinkage is not simply a defect. For a noisy estimator it is close to
MSE-optimal, so removing it can INCREASE MAE even while making the estimator
better calibrated. Do not select a correction on MAE alone. For a single-user
demo, a calibrated slope may matter more than population MAE, because the user
cares about their own reading rather than population error.

This is an analysis script, not a smoke test.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
APP_DIR = SCRIPT_PATH.parent.parent
REPO_ROOT = APP_DIR.parents[1]

DEMO_REFERENCE_HRS = (55.0, 60.0, 65.0, 70.0, 80.0, 100.0)


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

    return parsed if np.isfinite(parsed) else None


def load_rows(
    csv_path: Path,
    mode: str,
    datasets: set[str] | None,
) -> dict[str, list[tuple[float, float]]]:
    """
    Load (reference_hr, model_hr) pairs grouped by split.

    Parameters
    ----------
    csv_path:
        Detail CSV from audit_live_compatible_window_table.py.

    mode:
        Preprocessing mode to analyse.

    datasets:
        Dataset names to include, or None for all.

    Returns
    -------
    dict[str, list[tuple[float, float]]]
        Mapping from split name to (reference, prediction) pairs.
    """
    by_split: dict[str, list[tuple[float, float]]] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("preprocessing_status") != "ok":
                continue

            if row.get("preprocessing_mode") != mode:
                continue

            if datasets is not None and row.get("dataset") not in datasets:
                continue

            prediction = parse_float(row.get("model_hr"))
            reference = parse_float(row.get("target_hr_mean"))

            if prediction is None or reference is None:
                continue

            by_split.setdefault(row.get("split", "unknown"), []).append(
                (reference, prediction)
            )

    return by_split


def fit_correction(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """
    Fit offset and linear corrections on one set of pairs.

    Returns
    -------
    dict[str, float]
        Fitted intercept, slope, and mean signed bias.
    """
    reference = np.asarray([item[0] for item in pairs], dtype=np.float64)
    prediction = np.asarray([item[1] for item in pairs], dtype=np.float64)

    reference_variance = float(np.var(reference))

    if reference_variance > 1e-9:
        covariance = float(
            np.mean((reference - reference.mean()) * (prediction - prediction.mean()))
        )
        slope = covariance / reference_variance
    else:
        slope = float("nan")

    intercept = float(prediction.mean() - slope * reference.mean())

    return {
        "intercept": intercept,
        "slope": slope,
        "bias": float(np.mean(prediction - reference)),
        "n": float(len(pairs)),
    }


def apply_correction(
    prediction: np.ndarray,
    correction: str,
    fit: dict[str, float],
) -> np.ndarray:
    """Apply one named correction to raw model predictions."""
    if correction == "none":
        return prediction

    if correction == "offset":
        return prediction - fit["bias"]

    if correction == "linear":
        slope = fit["slope"]

        if not np.isfinite(slope) or abs(slope) < 1e-6:
            return np.full_like(prediction, np.nan)

        return (prediction - fit["intercept"]) / slope

    raise ValueError(f"Unknown correction: {correction}")


def evaluate(
    pairs: list[tuple[float, float]],
    correction: str,
    fit: dict[str, float],
) -> dict[str, float]:
    """Evaluate one correction on one split."""
    reference = np.asarray([item[0] for item in pairs], dtype=np.float64)
    raw = np.asarray([item[1] for item in pairs], dtype=np.float64)

    prediction = apply_correction(prediction=raw, correction=correction, fit=fit)

    if not np.all(np.isfinite(prediction)):
        return {
            "n": float(len(pairs)),
            "mae": float("nan"),
            "bias": float("nan"),
            "p90": float("nan"),
            "slope": float("nan"),
        }

    errors = prediction - reference
    reference_variance = float(np.var(reference))

    if reference_variance > 1e-9:
        covariance = float(
            np.mean((reference - reference.mean()) * (prediction - prediction.mean()))
        )
        slope = covariance / reference_variance
    else:
        slope = float("nan")

    return {
        "n": float(len(pairs)),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "p90": float(np.percentile(np.abs(errors), 90)),
        "slope": slope,
    }


def main() -> None:
    """Fit calibrations on train subjects and report held-out behaviour."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=REPO_ROOT / "Data" / "audit_live_compatible_window_table.csv",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="training_buffer_sim_20fps",
        help="Preprocessing mode to analyse.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="mcd_rppg,ubfc_rppg,ubfc_phys",
        help="Comma-separated dataset names, or 'all'.",
    )
    args = parser.parse_args()

    datasets = None

    if args.datasets != "all":
        datasets = {name.strip() for name in args.datasets.split(",") if name.strip()}

    by_split = load_rows(
        csv_path=args.input_csv,
        mode=args.mode,
        datasets=datasets,
    )

    if "train" not in by_split:
        raise RuntimeError(
            f"No train-split rows found for mode {args.mode!r}. "
            f"Available splits: {sorted(by_split)}"
        )

    fit = fit_correction(by_split["train"])

    print(f"input    : {args.input_csv}")
    print(f"mode     : {args.mode}")
    print(f"datasets : {args.datasets}")
    print("")
    print("Fitted on TRAIN subjects:")
    print(f"  n         : {int(fit['n'])}")
    print(f"  intercept : {fit['intercept']:.3f}")
    print(f"  slope     : {fit['slope']:.4f}")
    print(f"  bias      : {fit['bias']:+.3f} bpm")

    if np.isfinite(fit["slope"]) and abs(fit["slope"]) < 0.2:
        print("  WARNING: slope below 0.2; linear inversion will amplify noise ~5x")

    print("")
    print(
        f"{'split':<12} {'correction':<10} {'n':>5} "
        f"{'mae':>8} {'bias':>9} {'p90':>8} {'slope':>7}"
    )

    for split_name in ("train", "validation", "test"):
        if split_name not in by_split:
            continue

        for correction in ("none", "offset", "linear"):
            stats = evaluate(
                pairs=by_split[split_name],
                correction=correction,
                fit=fit,
            )

            print(
                f"{split_name:<12} {correction:<10} {int(stats['n']):>5d} "
                f"{stats['mae']:>8.2f} {stats['bias']:>+9.2f} "
                f"{stats['p90']:>8.2f} {stats['slope']:>7.3f}"
            )

    print("")
    print("What the demo would show for a given true HR:")
    print(f"{'true_hr':>8} {'none':>8} {'offset':>8} {'linear':>8}")

    for reference_hr in DEMO_REFERENCE_HRS:
        raw = fit["intercept"] + fit["slope"] * reference_hr
        raw_array = np.asarray([raw], dtype=np.float64)

        values = [
            float(apply_correction(raw_array, correction, fit)[0])
            for correction in ("none", "offset", "linear")
        ]

        print(
            f"{reference_hr:>8.0f} {values[0]:>8.1f} "
            f"{values[1]:>8.1f} {values[2]:>8.1f}"
        )

    print("")
    print("Reading the table:")
    print("  slope -> 1.0 means calibrated across the HR range")
    print("  bias  -> 0.0 means no average over- or under-prediction")
    print("  linear correction may raise MAE while improving slope; that is expected")


if __name__ == "__main__":
    main()