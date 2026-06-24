"""
Training-related constants.

This module contains simple constant values that are shared across the
training code, such as dataset names, image dimensions, and default
hyperparameters. Keeping these values in one place makes it easier to
update the configuration later without hunting through multiple files.
"""

# The canonical name of the dataset used in this project. For MedMNIST,
# PathMNIST is the dataset that contains colorectal histology patches.[web:10]
DATASET_NAME: str = "pathmnist"

# The number of classes in the PathMNIST dataset. According to the
# MedMNIST documentation there are nine tissue classes.[web:10]
NUM_CLASSES: int = 9

# The default image size for MedMNIST datasets is 28x28 pixels.[web:10]
IMAGE_HEIGHT: int = 28
IMAGE_WIDTH: int = 28

# Class label range (inclusive). PathMNIST labels are typically encoded
# as integers from 0 to NUM_CLASSES - 1.[web:10]
MIN_LABEL: int = 0
MAX_LABEL: int = NUM_CLASSES - 1

# Default batch size chosen to be small so that CPU-only training remains
# responsive even on modest hardware.
DEFAULT_BATCH_SIZE: int = 64

# Default number of data loader workers. Starting from zero avoids
# multiprocessing issues on Windows and simplifies debugging; this can be
# increased later once the pipeline is stable.[web:19]
DEFAULT_NUM_WORKERS: int = 0

# Default number of training epochs. This value is intentionally small so
# that initial experiments complete quickly and allow the rest of the
# pipeline (such as evaluation and inference) to be exercised.
DEFAULT_NUM_EPOCHS: int = 5

# Default learning rate for the optimizer. This is a conservative value
# that works reasonably well for small classification models such as
# ResNet18 on MedMNIST-like datasets.[web:21]
DEFAULT_LEARNING_RATE: float = 1e-3
