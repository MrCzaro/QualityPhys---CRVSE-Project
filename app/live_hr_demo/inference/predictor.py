"""
Prediction utilities for the live HR demo.
This module wraps model inference in a clean function.
Current input contract:
    x.shape = (batch, 3, 240)

Current output:
    PredictionResult with HR in BPM.

Signal:
    The model expects three normalized channels:
        POS, CHROM, GREEN

Limitation:
    This predictor assumes the input tensor is already correctly preprocessed.
    It does not extract rPPG from video. That comes later.
"""

from __future__ import annotations
import torch

from inference.schemas import PredictionResult, QualitySummary
from models.loader import ModelBundle


def validate_model_input(x: torch.Tensor, bundle: ModelBundle) -> None:
    """
    Validate that input tensor matches the active model spec.

    Parameters
    ----------
    x:
        Input tensor.

    bundle:
        Loaded model bundle.

    Raises
    ------
    ValueError
        If the tensor shape does not match the model contract.
    """
    input_config = bundle.model_spec["input"]
    expected_channels = int(input_config["in_channels"])
    expected_frames = int(input_config["target_frames"])

    if x.ndim != 3:
        raise ValueError(f"Expected tensor with shape (batch, channels, frames), got {tuple(x.shape)}.")

    _, channels, frames = x.shape

    if channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} channels, got {channels}. "
            f"Expected channel names: {input_config.get('channel_names')}"
        )

    if frames != expected_frames:
        raise ValueError(
            f"Expected {expected_frames} frames, got {frames}."
        )


def predict_hr_from_tensor(x: torch.Tensor, bundle: ModelBundle) -> list[PredictionResult]:
    """
    Predict HR from a batch of model-ready tensors.

    Parameters
    ----------
    x:
        Tensor with shape (batch, channels, frames).

    bundle:
        Loaded model bundle.

    Returns
    -------
    list[PredictionResult]
        One result per batch item.
    """

    validate_model_input(x, bundle)

    x = x.to(bundle.device)

    with torch.inference_mode():
        y = bundle.model(x)

    y = y.detach().cpu().float()

    model_spec = bundle.model_spec
    input_config = model_spec["input"]
    output_config = model_spec["output"]

    results: list[PredictionResult] = []

    for value in y.tolist():
        result = PredictionResult(
            task=model_spec["task"],
            value=float(value),
            unit=output_config["unit"],
            model_name=model_spec["name"],
            window_seconds=float(input_config["window_seconds"]),
            target_frames=int(input_config["target_frames"]),
            channel_names=list(input_config["channel_names"]),
            quality=QualitySummary(
                status="not_available_yet",
                confidence="not_available_yet",
                reasons=[
                    "Dummy/model-ready tensor inference only.",
                    "No video signal quality has been computed yet.",
                ],
                metrics={},
            ),
        )

        results.append(result)

    return results