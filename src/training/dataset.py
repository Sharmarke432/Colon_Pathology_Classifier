"""
Dataset utilities for PathMNIST.

This module defines helpers to construct training, validation, and test
datasets for the PathMNIST benchmark. The initial implementation can be
simple and rely directly on the MedMNIST API, and can be extended later
to support custom npz files or additional metadata.
"""

from pathlib import Path
from typing import Tuple

from medmnist import PathMNIST  # Make sure medmnist is installed.[web:72][web:75]
from torch.utils.data import Dataset
from torchvision import transforms as T


def create_train_val_datasets(
    data_dir: Path,
    transforms_train: T.Compose,
    transforms_val: T.Compose,
) -> Tuple[Dataset, Dataset]:
    """
    Create PathMNIST training and validation datasets.

    This helper uses the official MedMNIST splits for PathMNIST. It
    expects that the data has already been downloaded to `data_dir`,
    otherwise MedMNIST will download it automatically when
    `download=True` is specified.

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory where MedMNIST will store the downloaded dataset files.
    transforms_train : torchvision.transforms.Compose
        Transform pipeline applied to training images.
    transforms_val : torchvision.transforms.Compose
        Transform pipeline applied to validation images.

    Returns
    -------
    tuple[Dataset, Dataset]
        A pair `(train_dataset, val_dataset)` ready to be wrapped in
        PyTorch `DataLoader` objects.
    """
    # Build training dataset using the official "train" split.
    train_dataset = PathMNIST(
        root=str(data_dir),
        split="train",
        transform=transforms_train,
        download=True,
    )

    # Build validation dataset using the official "val" split.
    val_dataset = PathMNIST(
        root=str(data_dir),
        split="val",
        transform=transforms_val,
        download=True,
    )

    return train_dataset, val_dataset
