"""
Training entrypoint script.

This module defines a `main` function that orchestrates the training
process: it builds the dataset and data loaders, constructs the model,
configures the optimizer and loss function, and then invokes the trainer
to run the training loop.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models import create_model
from src.core.config import get_default_config
from src.training.constants import NUM_CLASSES
from src.training.dataset import (
    create_train_val_datasets,
)
from src.training.transforms import get_train_transform, get_val_transform
from .trainer import Trainer
from .utils import set_random_seed


def main() -> None:
    """
    Execute the training workflow.

    This function wires together the configuration, data pipeline, model
    construction, and training loop. It is designed to be called from a
    CLI or a notebook and keeps the high-level training logic in one
    place.
    """
    # Define directories for data and outputs. In a more advanced setup,
    # these could be read from environment variables or a shared config
    # module under `core/`.
    data_dir = Path(
        "data"
    )  # Directory where MedMNIST / PathMNIST files reside.[web:10]
    output_dir = Path("artifacts")  # Directory for checkpoints and logs.

    # Construct a default training configuration object.
    config = get_default_config(data_dir=data_dir, output_dir=output_dir)

    # Apply the random seed to improve reproducibility across runs.
    set_random_seed(config.seed)

    # Instantiate train and validation transforms.
    train_transforms = get_train_transform()
    val_transforms = get_val_transform()

    # Build the training and validation datasets using helper functions.
    train_dataset, val_dataset = create_train_val_datasets(
        data_dir=config.data_dir,
        transforms_train=train_transforms,
        transforms_val=val_transforms,
    )

    # Wrap the datasets in data loaders with the configured batch size
    # and number of worker processes.
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # Construct the classification model using the model factory.
    model = create_model(
        backbone_name="resnet18",
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout_p=0.0,
    )

    # Create the loss function. Cross-entropy is appropriate for
    # multi-class classification with integer labels.
    criterion = nn.CrossEntropyLoss()

    # Create an optimizer. Adam provides a good default for many tasks.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
    )

    # Construct the trainer object that encapsulates the training loop.
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # Run the training loop and save checkpoints as appropriate.
    trainer.train()


if __name__ == "__main__":
    # Allow the script to be run directly via `python -m src.training.train`.
    main()
