# src/training/transforms.py

"""
Image transform utilities for PathMNIST.

This module defines torchvision transform pipelines used during
training, validation, and testing. The mean and standard deviation
values are precomputed from the PathMNIST training split so that
normalization matches the dataset distribution.
"""

from torchvision import transforms as T

from .constants import IMAGE_HEIGHT, IMAGE_WIDTH

# PathMNIST mean/std computed from the training split.
# Keeping these as module-level constants makes them easy to reuse and
# also documents the normalization scheme used for the project.[web:4][web:94]
MEAN = [0.7406, 0.5331, 0.7059]
STD = [0.1270, 0.1542, 0.1196]


def get_train_transform() -> T.Compose:
    """
    Build the transform pipeline used for training images.

    This pipeline applies a small set of spatial and color augmentations
    suitable for pathology patches, followed by conversion to tensor and
    normalization using the precomputed PathMNIST statistics.

    Returns
    -------
    torchvision.transforms.Compose
        A composition of transforms applied to each training image.
    """
    return T.Compose(
        [
            # Ensure a fixed spatial size. This is mostly defensive in
            # case you later swap in a different MedMNIST size.[web:75]
            T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            # Lightweight augmentations to improve robustness while
            # keeping computation cheap on CPU.
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2),
            # Convert to tensor in [0, 1].
            T.ToTensor(),
            # Normalize using PathMNIST-specific mean and std so that the
            # model sees well-scaled inputs.[web:4][web:94]
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def get_val_transform() -> T.Compose:
    """
    Build the transform pipeline used for validation images.

    The validation pipeline avoids data augmentation to keep evaluation
    deterministic and comparable across runs but still applies the same
    resizing and normalization as training.

    Returns
    -------
    torchvision.transforms.Compose
        A composition of transforms applied to each validation image.
    """
    return T.Compose(
        [
            T.Resize((IMAGE_HEIGHT, IMAGE_WIDTH)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def get_test_transform() -> T.Compose:
    """
    Build the transform pipeline used for test images.

    This is identical to the validation transform so that test metrics
    reflect performance under the same preprocessing used for
    validation.[web:93]

    Returns
    -------
    torchvision.transforms.Compose
        A composition of transforms applied to each test image.
    """
    return get_val_transform()
