"""
Model loading utilities for the live HR demo.

This module turns the trained checkpoint + model_specs.yaml into a reusable
app component.

Why this exists:
    The app should not hardcode model architecture settings in app.py.
    The model contract lives in configs/model_specs.yaml.
    The checkpoint stores trained weights.
    This loader connects both.

Current model contract:
    input shape: (batch, 3, 240)
    channels: POS, CHROM, GREEN
    output shape: (batch,)
    output unit: BPM
"""

from __future__ import annotations
import torch, yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from models.architectures.crvse_physformer import CRVSEPhysFormer


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_SPECS_PATH = APP_DIR / "configs" / "model_specs.yaml"
DEFAULT_CHECKPOINT_DIR = APP_DIR / "models" / "checkpoints"


@dataclass
class ModelBundle:
    """
    Loaded model plus its configuration.

    Attributes
    ----------
    model:
        PyTorch model ready for inference.

    model_spec:
        Dictionary loaded from model_specs.yaml for the active model.

    checkpoint:
        Raw checkpoint dictionary loaded from the .pt file.

    checkpoint_path:
        Resolved path to the checkpoint file.

    device:
        Device used for inference.
    """

    model: CRVSEPhysFormer
    model_spec: dict[str, Any]
    checkpoint: dict[str, Any]
    checkpoint_path: Path
    device: torch.device


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load YAML file as a dictionary.

    Parameters
    ----------
    path:
        Path to YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.
    """

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        raise ValueError(f"YAML file is empty: {path}")

    if not isinstance(data, dict):
        raise TypeError(f"YAML root must be a dictionary, got {type(data)}.")

    return data


def get_model_spec(config: dict[str, Any], model_name: str | None = None) -> dict[str, Any]:
    """
    Select one model spec from model_specs.yaml.

    Parameters
    ----------
    config:
        Full parsed YAML config.

    model_name:
        Optional model name. If None, the first model is selected.

    Returns
    -------
    dict[str, Any]
        Selected model specification.
    """

    models = config.get("models")

    if not isinstance(models, list) or len(models) == 0:
        raise ValueError("Config must contain a non-empty list under key 'models'.")

    if model_name is None:
        selected = models[0]
    else:
        matching = [model for model in models if model.get("name") == model_name]

        if len(matching) == 0:
            available = [model.get("name") for model in models]
            raise ValueError(f"Model {model_name!r} not found. Available models: {available}")

        selected = matching[0]

    if not isinstance(selected, dict):
        raise TypeError("Selected model spec must be a dictionary.")

    return selected


def resolve_checkpoint_path(model_spec: dict[str, Any],checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR) -> Path:
    """
    Resolve checkpoint path from model spec.

    Parameters
    ----------
    model_spec:
        Selected model specification.

    checkpoint_dir:
        Directory containing model checkpoint files.

    Returns
    -------
    Path
        Full path to checkpoint.
    """

    checkpoint_file = model_spec.get("checkpoint_file")

    if not checkpoint_file:
        raise ValueError("Model spec is missing 'checkpoint_file'.")

    checkpoint_path = checkpoint_dir / checkpoint_file

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}\n"
            f"Expected checkpoint directory: {checkpoint_dir}"
        )

    return checkpoint_path


def build_crvse_physformer_from_spec(model_spec: dict[str, Any]) -> CRVSEPhysFormer:
    """
    Build CRVSEPhysFormer from model_specs.yaml.

    Parameters
    ----------
    model_spec:
        Selected model specification.

    Returns
    -------
    CRVSEPhysFormer
        Model architecture without loaded trained weights yet.
    """

    architecture = model_spec.get("architecture")

    if architecture != "CRVSEPhysFormer":
        raise ValueError(
            f"Unsupported architecture: {architecture!r}. "
            "This loader currently supports only 'CRVSEPhysFormer'."
        )

    input_config = model_spec.get("input", {})
    architecture_params = model_spec.get("architecture_params", {})

    model = CRVSEPhysFormer(
        in_channels=int(input_config["in_channels"]),
        cnn_channels=int(architecture_params["cnn_channels"]),
        freq_channels=int(architecture_params["freq_channels"]),
        n_heads=int(architecture_params["n_heads"]),
        n_layers=int(architecture_params["n_layers"]),
        dim_feedforward=int(architecture_params["dim_feedforward"]),
        dropout=float(architecture_params["dropout"]),
        hr_min=float(architecture_params["hr_min"]),
        hr_max=float(architecture_params["hr_max"]),
        target_frames=int(input_config["target_frames"]),
        max_positional_length=int(architecture_params.get("max_positional_length", 300)),
    )

    return model


def validate_checkpoint_against_spec(checkpoint: dict[str, Any], model_spec: dict[str, Any]) -> None:
    """
    Validate simple checkpoint metadata against model_specs.yaml.

    This does not prove the model is clinically useful.
    It only checks that config and checkpoint agree on basic input contract.
    """

    input_config = model_spec.get("input", {})

    expected_input_mode = input_config.get("input_mode")
    expected_in_channels = int(input_config.get("in_channels"))

    checkpoint_input_mode = checkpoint.get("input_mode")
    checkpoint_in_channels = int(checkpoint.get("in_channels"))

    if checkpoint_input_mode != expected_input_mode:
        raise ValueError(
            f"input_mode mismatch: config={expected_input_mode!r}, "
            f"checkpoint={checkpoint_input_mode!r}"
        )

    if checkpoint_in_channels != expected_in_channels:
        raise ValueError(
            f"in_channels mismatch: config={expected_in_channels}, "
            f"checkpoint={checkpoint_in_channels}"
        )

    if "model_state" not in checkpoint:
        raise KeyError("Checkpoint is missing required key: 'model_state'.")


def load_model_bundle(
    model_name: str | None = None,
    config_path: Path = DEFAULT_MODEL_SPECS_PATH,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    device: str | torch.device = "cpu",
) -> ModelBundle:
    """
    Load model config, checkpoint, architecture, and trained weights.

    Parameters
    ----------
    model_name:
        Optional model name from model_specs.yaml.
        If None, first model in the config is used.

    config_path:
        Path to model_specs.yaml.

    checkpoint_dir:
        Directory containing checkpoint files.

    device:
        Inference device, usually "cpu" for the demo.

    Returns
    -------
    ModelBundle
        Loaded model and related metadata.
    """

    device = torch.device(device)

    config = load_yaml(config_path)
    model_spec = get_model_spec(config, model_name=model_name)
    checkpoint_path = resolve_checkpoint_path(model_spec, checkpoint_dir=checkpoint_dir)

    # This checkpoint is trusted because it is your own training artifact.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint should be a dictionary, got {type(checkpoint)}.")

    validate_checkpoint_against_spec(checkpoint, model_spec)

    model = build_crvse_physformer_from_spec(model_spec)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()

    return ModelBundle(
        model=model,
        model_spec=model_spec,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def predict_dummy_hr(bundle: ModelBundle) -> torch.Tensor:
    """
    Run dummy inference using the model's configured input shape.

    This is only a software contract test.
    It does not represent a physiological signal.
    """
    input_config = bundle.model_spec["input"]
    in_channels = int(input_config["in_channels"])
    target_frames = int(input_config["target_frames"])
    dummy = torch.randn(1, in_channels, target_frames, device=bundle.device)

    with torch.inference_mode():
        prediction = bundle.model(dummy)

    return prediction