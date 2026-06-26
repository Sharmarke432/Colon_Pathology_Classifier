"""
Simple tests for the model factory and classifier.

These tests validate that the model can be instantiated and run on
dummy inputs without raising errors, and that the output logits have
the expected shape and dtype.
"""

import torch

from src.models import create_model
from src.training.constants import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH


def test_create_model_forward_pass() -> None:
    """
    Ensure the model factory builds a model that can process a batch
    of images and produce logits with the expected shape.
    """
    # Construct the model using the default backbone.
    model = create_model(
        backbone_name="resnet18",
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout_p=0.0,
    )

    model.eval()

    # Create a dummy batch of images: batch_size x channels x height x width.
    dummy_input = torch.randn(4, 3, IMAGE_HEIGHT, IMAGE_WIDTH)

    # Run a forward pass.
    with torch.no_grad():
        logits = model(dummy_input)

    # Check that the batch dimension is preserved.
    assert logits.shape[0] == dummy_input.shape[0]

    # Check that the number of output units matches the class count.
    assert logits.shape[1] == NUM_CLASSES

    # Check type/dtype invariants.
    assert isinstance(logits, torch.Tensor)
    assert logits.dtype == torch.float32
