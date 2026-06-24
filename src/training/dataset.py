"""
Dataset utilities for PathMNIST.

This module defines helpers to construct training, validation, and test
datasets for the PathMNIST benchmark. The initial implementation can be
simple and rely directly on the MedMNIST API, and can be extended later
to support custom npz files or additional metadata.[web:72][web:75]
"""

from pathlib import Path
from typing import Tuple

import torch
from medmnist import PathMNIST
from torch.utils.data import Dataset
from torchvision import transforms as T


class PathMNISTWrapper(Dataset):
    """
    Thin wrapper around medmnist.PathMNIST that ensures labels are
    returned as 1D integer class indices suitable for CrossEntropyLoss.[web:72][web:75]

    The underlying MedMNIST dataset can return labels with extra
    dimensions (for example shape (1,)), which causes errors when used
    directly with `nn.CrossEntropyLoss`. This wrapper normalizes the
    label shape and dtype.
    """

    def __init__(
        self,
        split: str,
        root: Path,
        transform: T.Compose | None = None,
        download: bool = False,
    ) -> None:
        """
        Initialize the wrapped PathMNIST dataset.

        Parameters
        ----------
        split : str
            Which data split to load: "train", "val", or "test".[web:72]
        root : pathlib.Path
            Directory where MedMNIST stores the dataset files.
        transform : torchvision.transforms.Compose, optional
            Transform pipeline applied to each image.
        download : bool, optional
            Whether to download the dataset if it is not already
            present in `root`.
        """
        self._dataset = PathMNIST(
            root=str(root),
            split=split,
            transform=transform,
            download=download,
        )

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset.
        """
        return len(self._dataset)

    def __getitem__(self, index: int):
        """
        Retrieve a single sample from the dataset.

        The returned label is converted to a 1D tensor of dtype
        `torch.long` so that it is compatible with CrossEntropyLoss.[web:91]
        """
        # The underlying dataset returns (image, label). The label may be
        # a NumPy array or tensor with an extra dimension.
        img, label = self._dataset[index]

        # Convert label to a tensor if it is not already one.
        label_tensor = torch.as_tensor(label)

        # Remove any singleton dimensions, e.g. shape (1,) -> scalar.
        label_tensor = label_tensor.squeeze()

        # Ensure the label is a single integer index.
        label_tensor = label_tensor.to(dtype=torch.long)

        return img, label_tensor


def create_train_val_datasets(
    data_dir: Path,
    transforms_train: T.Compose,
    transforms_val: T.Compose,
) -> Tuple[Dataset, Dataset]:
    """
    Create PathMNIST training and validation datasets.

    This helper uses the official MedMNIST splits for PathMNIST through
    the PathMNISTWrapper, which fixes label shapes for compatibility
    with CrossEntropyLoss.[web:72][web:75]

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
    train_dataset = PathMNISTWrapper(
        split="train",
        root=data_dir,
        transform=transforms_train,
        download=True,
    )

    val_dataset = PathMNISTWrapper(
        split="val",
        root=data_dir,
        transform=transforms_val,
        download=True,
    )

    return train_dataset, val_dataset


def create_test_dataset(
    data_dir: Path,
    transforms: T.Compose,
) -> Dataset:
    """
    Create PathMNIST test dataset using the official MedMNIST split.[web:72][web:75]

    Parameters
    ----------
    data_dir : pathlib.Path
        Directory where MedMNIST stores the dataset files.
    transforms : torchvision.transforms.Compose
        Transform pipeline applied to test images.

    Returns
    -------
    torch.utils.data.Dataset
        Test dataset ready to be wrapped in a DataLoader.
    """
    test_dataset = PathMNISTWrapper(
        split="test",
        root=data_dir,
        transform=transforms,
        download=True,
    )
    return test_dataset
