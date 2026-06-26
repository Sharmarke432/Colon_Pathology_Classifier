"""
Model package initialization.

This module exposes the main factory function that other parts of the
codebase (for example the training and inference layers) can use to
construct classification models without depending on backend-specific
details such as the exact torchvision import paths.
"""

from .model_factory import (
    create_model,
)  # Import the factory function so that users can call models.create_model directly.

__all__ = [
    "create_model"
]  # Explicitly define what is exported when using `from models import *`.
