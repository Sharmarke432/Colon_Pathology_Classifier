"""
Training configuration utilities.

This module defines configuration objects and helper functions used to
control the behavior of the training and evaluation scripts. The goal is
to provide a single source of truth for parameters such as learning rate,
batch size, and device selection so that experiments are easy to
reproduce and compare.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from src.training.constants import (
    DATASET_NAME,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    NUM_CLASSES,
)


DeviceType = Literal["cpu"]  # Restrict the device type to CPU for now.


@dataclass
class TrainingConfig:
    """
    Configuration for a single training run.

    This dataclass captures the key hyperparameters and paths used during
    training. Instances can be created directly in code or loaded from a
    configuration file at a later stage to support more advanced
    workflows.
    """

    # Filesystem paths
    data_dir: Path  # Directory where the MedMNIST / PathMNIST data is stored.[web:10]
    output_dir: Path  # Directory where checkpoints and logs will be written.

    # Data-related parameters
    dataset_name: str = DATASET_NAME  # Name of the dataset being used.
    num_classes: int = NUM_CLASSES  # Number of output classes.

    # Training hyperparameters
    batch_size: int = DEFAULT_BATCH_SIZE  # Batch size for the data loader.
    num_workers: int = (
        DEFAULT_NUM_WORKERS  # Number of worker processes for data loading.
    )
    num_epochs: int = DEFAULT_NUM_EPOCHS  # Number of epochs to train for.
    learning_rate: float = DEFAULT_LEARNING_RATE  # Initial learning rate.

    # Device configuration
    device: DeviceType = (
        "cpu"  # Device identifier, constrained to "cpu" for reproducibility.
    )

    # Miscellaneous settings
    seed: int = 42  # Random seed used for reproducibility.[web:16]
    checkpoint_name: str = "best_model.pt"  # Filename for the best model checkpoint.

    model_name: str = "resnet18"

    def checkpoint_path(self) -> Path:
        """
        Compute the full path to the checkpoint file.

        Returns
        -------
        pathlib.Path
            Path pointing to the checkpoint file inside the output
            directory.
        """
        return self.output_dir / self.checkpoint_name


def get_default_config(
    data_dir: Path, output_dir: Path, model_name: str = "resnet18"
) -> TrainingConfig:
    """
    Construct a default `TrainingConfig` for CPU-only training.

    This helper function centralizes the creation of a configuration
    object with sensible defaults so that the training script can obtain
    a ready-to-use configuration with minimal boilerplate.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory containing the downloaded MedMNIST data.
    output_dir : pathlib.Path
        Directory where training outputs such as checkpoints and metrics
        will be stored.

    Returns
    -------
    TrainingConfig
        A configuration instance populated with default values.
    """
    # For now, always select the CPU device. This avoids any dependency
    # on CUDA and ensures that training can run on machines without a
    # GPU.[web:23]
    device: DeviceType = "cpu"

    config = TrainingConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        device=device,
        model_name=model_name,
    )

    return config


def get_torch_device(config: TrainingConfig) -> torch.device:
    """
    Convert the configuration's device type into a `torch.device`.

    This function separates the configuration-level representation of
    the device (a simple string literal) from the PyTorch-specific
    `torch.device` object used when constructing models and tensors.

    Parameters
    ----------
    config : TrainingConfig
        The configuration object that specifies which device should be
        used.

    Returns
    -------
    torch.device
        The corresponding PyTorch device object.
    """
    return torch.device(config.device)
