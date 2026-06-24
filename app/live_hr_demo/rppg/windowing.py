"""
Windowing utilities for rPPG model input preparation.

This module prepares POS / CHROM / GREEN traces for model inference.

Current model contract:
    input shape = (batch, 3, 240)

Current channel order:
    0 = POS
    1 = CHROM
    2 = GREEN


Signal:
    The model does not consume raw video. It consumes fixed-length signal windows.

Limitation:
    This module assumes POS / CHROM / GREEN signals already exist.
    It does not extract them from face video yet.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class WindowConfig:
    """
    Configuration for converting rPPG traces into model-ready tensors.

    Attributes
    ----------
    window_seconds:
        Duration of the signal window in seconds.

    target_frames:
        Number of samples expected by the model after resampling.

    channel_names:
        Channel order expected by the model.

    normalization:
        Normalization mode. Current training used per-window z-score.
    """
    window_seconds: float = 8.0
    target_frames: int = 240
    channel_names: tuple[str, ...] = ("pos", "chrom", "green")
    normalization: str = "per_window_zscore"


def zscore_1d(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalize one 1D signal.

    Parameters
    ----------
    signal:
        Input signal with shape (time,).

    eps:
        Small value preventing division by zero.

    Returns
    -------
    np.ndarray
        Normalized signal with shape (time,).
    """

    signal = np.asarray(signal, dtype=np.float32)
    mean = float(np.mean(signal))
    std = float(np.std(signal))

    if std < eps:
        return np.zeros_like(signal, dtype=np.float32)

    return ((signal - mean) / (std + eps)).astype(np.float32)


def resample_1d_linear(signal: np.ndarray, target_frames: int) -> np.ndarray:
    """
    Resample a 1D signal to a fixed number of frames using linear interpolation.

    Parameters
    ----------
    signal:
        Input signal with shape (time,).

    target_frames:
        Number of output samples.

    Returns
    -------
    np.ndarray
        Resampled signal with shape (target_frames,).
    """

    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {signal.shape}.")

    if len(signal) < 2:
        raise ValueError("Need at least 2 samples to resample a signal.")

    old_x = np.linspace(0.0, 1.0, num=len(signal), dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_frames, dtype=np.float32)

    resampled = np.interp(new_x, old_x, signal)

    return resampled.astype(np.float32)


def make_model_window_from_channels(
    pos: np.ndarray,
    chrom: np.ndarray,
    green: np.ndarray,
    config: WindowConfig,
) -> torch.Tensor:
    """
    Convert POS / CHROM / GREEN traces into a model-ready tensor.

    Parameters
    ----------
    pos:
        POS rPPG trace.

    chrom:
        CHROM rPPG trace.

    green:
        GREEN rPPG trace.

    config:
        Windowing configuration.

    Returns
    -------
    torch.Tensor
        Tensor with shape (1, 3, target_frames).
    """

    channels = {
        "pos": pos,
        "chrom": chrom,
        "green": green,
    }

    prepared_channels: list[np.ndarray] = []

    for channel_name in config.channel_names:
        if channel_name not in channels:
            raise ValueError(f"Unsupported channel name: {channel_name}")

        signal = channels[channel_name]
        signal = resample_1d_linear(signal, config.target_frames)

        if config.normalization == "per_window_zscore":
            signal = zscore_1d(signal)
        else:
            raise ValueError(f"Unsupported normalization: {config.normalization}")

        prepared_channels.append(signal)

    stacked = np.stack(prepared_channels, axis=0).astype(np.float32)

    return torch.from_numpy(stacked).unsqueeze(0)


def make_synthetic_rppg_channels(
    hr_bpm: float = 72.0,
    duration_seconds: float = 8.0,
    fps: float = 30.0,
    noise_std: float = 0.05,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Create synthetic POS / CHROM / GREEN-like pulse traces.

    This is only for software testing and learning.

    Parameters
    ----------
    hr_bpm:
        Synthetic heart rate in beats per minute.

    duration_seconds:
        Duration of generated signal.

    fps:
        Sampling frequency.

    noise_std:
        Standard deviation of additive Gaussian noise.

    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys: pos, chrom, green, time.
    """

    rng = np.random.default_rng(seed)
    n_samples = int(round(duration_seconds * fps))
    time = np.arange(n_samples, dtype=np.float32) / float(fps)
    heart_hz = hr_bpm / 60.0
    base = np.sin(2.0 * np.pi * heart_hz * time)

    # Slightly different amplitudes/phases mimic different rPPG methods.
    pos = 1.00 * base + rng.normal(0.0, noise_std, size=n_samples)
    chrom = 0.85 * np.sin(2.0 * np.pi * heart_hz * time + 0.15) + rng.normal(
        0.0,
        noise_std,
        size=n_samples,
    )
    green = 0.70 * np.sin(2.0 * np.pi * heart_hz * time - 0.10) + rng.normal(
        0.0,
        noise_std,
        size=n_samples,
    )

    return {
        "time": time.astype(np.float32),
        "pos": pos.astype(np.float32),
        "chrom": chrom.astype(np.float32),
        "green": green.astype(np.float32),
    }