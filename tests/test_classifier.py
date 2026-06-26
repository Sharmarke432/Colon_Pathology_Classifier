# tests/test_classifier.py

"""
Unit tests for the LinearClassifier head.

These tests verify that the classifier maps feature vectors to logits
with the correct shape and dtype.
"""

import torch

from src.models.classifier import LinearClassifier
from src.training.constants import NUM_CLASSES


def test_linear_classifier_output_shape_and_dtype() -> None:
    """
    LinearClassifier should map (B, in_features) -> (B, NUM_CLASSES)
    with float32 logits.
    """
    batch_size = 5
    in_features = 128

    head = LinearClassifier(
        in_features=in_features,
        num_classes=NUM_CLASSES,
        dropout_p=0.5,
    )
    head.eval()

    features = torch.randn(batch_size, in_features)

    with torch.no_grad():
        logits = head(features)

    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (batch_size, NUM_CLASSES)
    assert logits.dtype == torch.float32
