"""
Model contract smoke test for the live HR demo.

This script verifies that the configured CRVSE PhysFormer model can be rebuilt
outside the training notebook, loaded from its checkpoint, and executed with the
expected live-demo input shape.

The test checks:

1. The model bundle loads from the app configuration.
2. The model specification and checkpoint metadata are available.
3. The expected input shape is ``(1, 3, 240)``.
4. The checkpoint state is compatible with the model architecture.
5. A dummy input runs through the model and returns one HR value.

This is a deployment-readiness check, not an accuracy test.
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


def print_section(title: str) -> None:
    """
    Print a formatted section header.

    Parameters
    ----------
    title:
        Section title to display.
    """
    print(title)
    print("-" * 72)


def validate_model_contract(bundle) -> tuple[int, int, int]:
    """
    Validate the loaded model bundle against the live-demo input contract.

    Parameters
    ----------
    bundle:
        Loaded model bundle returned by ``load_model_bundle``.

    Returns
    -------
    tuple[int, int, int]
        Expected dummy input shape.

    Raises
    ------
    ValueError
        If the model specification does not match the expected live-demo
        multichannel input contract.
    """
    model_spec = bundle.model_spec
    input_config = model_spec["input"]
    expected_shape = (1, int(input_config["in_channels"]), int(input_config["target_frames"]))

    if expected_shape != (1, 3, 240):
        raise ValueError(f"Expected live-demo input shape (1, 3, 240), got {expected_shape}")

    checkpoint_path = Path(bundle.checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")

    return expected_shape


def main() -> None:
    """
    Run the CRVSE PhysFormer model contract smoke test.
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

    print_section("Model spec")
    print(f"Name: {model_spec.get('name')}")
    print(f"Display name: {model_spec.get('display_name')}")
    print(f"Architecture: {model_spec.get('architecture')}")
    print(f"Checkpoint: {bundle.checkpoint_path}")
    print()

    print_section("Checkpoint metadata")
    print(f"input_mode: {checkpoint.get('input_mode')}")
    print(f"in_channels: {checkpoint.get('in_channels')}")
    print(f"best_n_epochs: {checkpoint.get('best_n_epochs')}")

    best_val_mae = checkpoint.get("best_val_mae")

    if best_val_mae is None:
        print("best_val_mae: None")
    else:
        print(f"best_val_mae: {float(best_val_mae):.4f}")

    print(f"best_params: {checkpoint.get('best_params')}")
    print()

    print_section("Model contract")

    expected_shape = validate_model_contract(bundle)

    print(f"Expected dummy input shape: {expected_shape}")
    print(f"Trainable parameters: {count_trainable_parameters(bundle.model):,}")
    print()

    print_section("Running dummy inference")

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