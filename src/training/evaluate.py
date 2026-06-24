"""
Model evaluation script.

This module defines an entrypoint for evaluating a trained model on a
held-out test dataset. It loads the best checkpoint saved during
training and computes metrics such as accuracy on the test set.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models import create_model
from src.core.config import get_default_config, get_torch_device
from .constants import NUM_CLASSES
from .dataset import create_test_dataset  # To be implemented in dataset.py.
from .metrics import accuracy
from .utils import load_checkpoint
from .transforms import get_test_transform


def main() -> None:
    """
    Execute the evaluation workflow.

    This function builds the test dataset and data loader, reconstructs
    the model, loads the saved checkpoint, and then computes accuracy on
    the test set.
    """
    # Define directories consistent with the training script.
    data_dir = Path("data")
    output_dir = Path("artifacts")

    # Recreate the configuration used during training.
    config = get_default_config(data_dir=data_dir, output_dir=output_dir)
    device = get_torch_device(config)

    # Create the test transforms and dataset.
    test_transform = get_test_transform()
    test_dataset = create_test_dataset(
        data_dir=config.data_dir,
        transforms=test_transform,
    )

    # Wrap the test dataset in a data loader.
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Reconstruct the model architecture.
    model = create_model(
        backbone_name="resnet18",
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout_p=0.0,
    )
    model.to(device)

    # Load the best checkpoint from disk.
    checkpoint_path = config.checkpoint_path()
    load_checkpoint(model=model, path=checkpoint_path, map_location=device)

    # Switch the model to evaluation mode.
    model.eval()

    total_acc = 0.0
    num_batches = 0

    # Disable gradient computation for efficiency during evaluation.
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass to obtain logits.
            logits = model(inputs)

            # Compute accuracy for this batch.
            batch_acc = accuracy(logits=logits, targets=targets)

            total_acc += batch_acc
            num_batches += 1

    # Handle the case where the test loader is empty.
    if num_batches == 0:
        print("No test batches available.")
        return

    # Compute average accuracy across all batches.
    avg_acc = total_acc / num_batches
    print(f"Test accuracy: {avg_acc:.4f}")


if __name__ == "__main__":
    main()
