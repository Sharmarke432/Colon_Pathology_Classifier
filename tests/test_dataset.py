"""
Tests for the PathMNIST dataset utilities.

These tests verify that the PathMNISTWrapper and dataset factory
functions return samples and batches with the expected shapes and
dtypes, so they are compatible with CrossEntropyLoss.
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.training.dataset import (
    PathMNISTWrapper,
    create_train_val_datasets,
    create_test_dataset,
)
from src.training.transforms import (
    get_train_transform,
    get_val_transform,
    get_test_transform,
)
from src.training.constants import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH


def _get_one_batch(dataset, batch_size: int = 8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    return images, labels


def test_pathmnist_wrapper_single_sample_shapes_and_types(tmp_path: Path) -> None:
    """
    PathMNISTWrapper should return a 3xHxW float32 image tensor and a
    scalar long label in [0, NUM_CLASSES).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(
        parents=True, exist_ok=True
    )  # Ensure the directory exists for download.

    dataset = PathMNISTWrapper(
        split="train",
        root=str(data_dir),
        transform=get_train_transform(),
        download=True,
    )

    image, label = dataset[0]

    # Image: 3 x H x W, float32
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert image.dtype == torch.float32

    # Label: scalar long class index in range
    assert isinstance(label, torch.Tensor)
    assert label.dim() == 0
    assert label.dtype == torch.long
    value = int(label.item())
    assert 0 <= value < NUM_CLASSES


def test_train_val_datasets_batch_shapes(tmp_path: Path) -> None:
    """
    Batches from train/val datasets should be of shape:
    images: (B, 3, H, W), labels: (B,) with long dtype.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)  # NEW

    train_tf = get_train_transform()
    val_tf = get_val_transform()

    train_ds, val_ds = create_train_val_datasets(
        data_dir=str(data_dir),
        transforms_train=train_tf,
        transforms_val=val_tf,
    )

    train_images, train_labels = _get_one_batch(train_ds, batch_size=8)
    val_images, val_labels = _get_one_batch(val_ds, batch_size=8)

    for images, labels in (
        (train_images, train_labels),
        (val_images, val_labels),
    ):
        assert images.shape[1:] == (3, IMAGE_HEIGHT, IMAGE_WIDTH)
        assert images.dtype == torch.float32

        assert labels.shape[0] == images.shape[0]
        assert labels.shape[1:] == ()  # (B,) not (B,1)
        assert labels.dtype == torch.long
        assert torch.all((labels >= 0) & (labels < NUM_CLASSES))


def test_test_dataset_compatible_with_val(tmp_path: Path) -> None:
    """
    Test dataset should have the same image/label shape and dtypes as
    the validation dataset, so it can plug into the same evaluation code.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)  # NEW

    val_tf = get_val_transform()
    test_tf = get_test_transform()

    _, val_ds = create_train_val_datasets(
        data_dir=str(data_dir),
        transforms_train=val_tf,
        transforms_val=val_tf,
    )
    test_ds = create_test_dataset(
        data_dir=str(data_dir),
        transforms=test_tf,
    )

    val_img, val_lbl = val_ds[0]
    test_img, test_lbl = test_ds[0]

    # Image shape/dtype
    assert val_img.shape == test_img.shape == (3, IMAGE_HEIGHT, IMAGE_WIDTH)
    assert val_img.dtype == test_img.dtype == torch.float32

    # Label shape/dtype/range
    assert val_lbl.dim() == test_lbl.dim() == 0
    assert val_lbl.dtype == test_lbl.dtype == torch.long
    assert 0 <= int(val_lbl.item()) < NUM_CLASSES
    assert 0 <= int(test_lbl.item()) < NUM_CLASSES
