"""
Model contract checker for the live HR demo.

This script verifies:
    1. model_specs.yaml exists
    2. checkpoint exists
    3. checkpoint metadata matches config
    4. CRVSEPhysFormer rebuilds outside the notebook
    5. state_dict loads strictly
    6. dummy input [1, 3, 240] runs
    7. output shape is [1]
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
APP_DIR = SCRIPT_DIR.parent
REPO_DIR = APP_DIR.parents[1]

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from models.architectures.crvse_physformer import count_trainable_parameters
from models.loader import load_model_bundle, predict_dummy_hr


def main() -> None:
    """
    Run model contract check and print a readable report.
    """

    print("=" * 72)
    print("CRVSE PhysFormer model contract check")
    print("=" * 72)
    print(f"App dir:  {APP_DIR}")
    print(f"Repo dir: {REPO_DIR}")
    print()

    bundle = load_model_bundle(device="cpu")

    model_spec = bundle.model_spec
    checkpoint = bundle.checkpoint

    print("Model spec")
    print("-" * 72)
    print(f"Name: {model_spec.get('name')}")
    print(f"Display name: {model_spec.get('display_name')}")
    print(f"Architecture: {model_spec.get('architecture')}")
    print(f"Checkpoint: {bundle.checkpoint_path}")
    print()

    print("Checkpoint metadata")
    print("-" * 72)
    print(f"input_mode: {checkpoint.get('input_mode')}")
    print(f"in_channels: {checkpoint.get('in_channels')}")
    print(f"best_n_epochs: {checkpoint.get('best_n_epochs')}")
    print(f"best_val_mae: {checkpoint.get('best_val_mae'):.4f}")
    print(f"best_params: {checkpoint.get('best_params')}")
    print()

    print("Model contract")
    print("-" * 72)
    input_config = model_spec["input"]

    expected_shape = (
        1,
        int(input_config["in_channels"]),
        int(input_config["target_frames"]),
    )

    print(f"Expected dummy input shape: {expected_shape}")
    print(f"Trainable parameters: {count_trainable_parameters(bundle.model):,}")
    print()

    print("Running dummy inference")
    print("-" * 72)

    prediction = predict_dummy_hr(bundle)

    print(f"Output shape: {tuple(prediction.shape)}")
    print(f"Output value: {prediction.detach().cpu().numpy().tolist()}")
    print()

    if tuple(prediction.shape) != (1,):
        raise ValueError(f"Expected output shape (1,), got {tuple(prediction.shape)}")

    print("=" * 72)
    print("PASS: model contract is valid")
    print("=" * 72)


if __name__ == "__main__":
    main()