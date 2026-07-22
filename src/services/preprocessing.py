"""
Image preprocessing utilities for inference.
"""

from io import BytesIO

import torch
from PIL import Image

from src.training.transforms import get_test_transform


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load an uploaded image from raw bytes and convert it to RGB.
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Could not decode uploaded image.") from exc

    return image


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """
    Convert a PIL image into a batched tensor ready for model inference.
    """
    transform = get_test_transform()
    tensor = transform(image)
    return tensor.unsqueeze(0)
