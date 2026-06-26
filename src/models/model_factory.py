"""
Model factory utilities.

This module defines functions that construct classification models for
the PathMNIST dataset. The factory hides backbone-specific details and
exposes a simple interface to the training and inference layers so that
you can swap architectures (for example from ResNet18 to EfficientNet-B0)
without changing the rest of the pipeline.
"""

from typing import Literal

from torch import nn
from torchvision import models as tv_models

from .classifier import LinearClassifier


BackboneName = Literal[
    "resnet18"
]  # You can add more options such as "efficientnet_b0" later.


def create_backbone(
    name: str,
    pretrained: bool = False,
) -> nn.Module:
    """
    Create a feature extractor backbone.

    Parameters
    ----------
    name : BackboneName
        The name of the backbone architecture to construct.
        Currently only "resnet18" is supported in order to keep the
        CPU-only training setup lightweight and reproducible.
    pretrained : bool, optional
        Whether to load ImageNet pretrained weights where available.
        Defaults to False to avoid implicit downloads and to make
        offline training possible.

    Returns
    -------
    nn.Module
        A backbone model whose final classification layer will be
        replaced by a custom classifier head.
    """
    if name == "resnet18":
        # Construct a ResNet18 from torchvision. This model is relatively
        # small and well-suited for CPU training on modest hardware.[web:18]
        backbone = tv_models.resnet18(
            weights=None if not pretrained else tv_models.ResNet18_Weights.DEFAULT
        )
    else:
        # Raise a clear error when an unsupported backbone is requested
        # so that configuration issues are detected early.
        raise ValueError(f"Unsupported backbone name: {name}")

    return backbone


def _replace_resnet_classifier(
    backbone: tv_models.ResNet,
    num_classes: int,
    dropout_p: float = 0.0,
) -> nn.Module:
    """
    Replace the final fully connected layer of a ResNet model.

    This helper function extracts the input feature dimension from the
    existing `fc` layer and constructs a new `LinearClassifier` with the
    desired number of output classes.

    Parameters
    ----------
    backbone : torchvision.models.ResNet
        The ResNet model whose classifier will be replaced.
    num_classes : int
        The desired number of output classes.
    dropout_p : float, optional
        Optional dropout probability for the classifier head.

    Returns
    -------
    nn.Module
        The modified ResNet model with an updated classification head.
    """
    # Read the number of input features from the existing fully connected layer.
    in_features = backbone.fc.in_features

    # Construct a new classifier head with the requested number of classes.
    classifier = LinearClassifier(
        in_features=in_features,
        num_classes=num_classes,
        dropout_p=dropout_p,
    )

    # Replace the original `fc` layer with our custom classifier head.
    backbone.fc = classifier

    return backbone


def create_model(
    backbone_name: str = "resnet18",
    num_classes: int = 9,
    pretrained: bool = False,
    dropout_p: float = 0.0,
) -> nn.Module:
    """
    Create a full classification model for PathMNIST.

    This function constructs the chosen backbone and attaches an
    appropriate classifier head. It serves as the main entry point for
    the training and inference code when a new model instance is needed.

    Parameters
    ----------
    backbone_name : BackboneName, optional
        The architecture to use for the feature extractor. Defaults to
        "resnet18" for a lightweight CPU-friendly setup.
    num_classes : int, optional
        The number of tissue classes in the dataset. Defaults to 9 which
        matches the PathMNIST configuration.[web:10]
    pretrained : bool, optional
        Whether to start from ImageNet-pretrained weights. Defaults to
        False to avoid hidden downloads and to keep local runs
        reproducible offline.
    dropout_p : float, optional
        Dropout probability in the classifier head. Defaults to 0.0 for
        simplicity.

    Returns
    -------
    nn.Module
        A model ready to be trained or used for inference.
    """
    # Create the backbone using a dedicated helper function.
    backbone = create_backbone(name=backbone_name, pretrained=pretrained)

    # For now, we only know how to replace the classifier of ResNet-based
    # architectures. If more backbones are introduced later, this logic
    # can be extended with additional conditionals or a registry.
    if isinstance(backbone, tv_models.ResNet):
        model = _replace_resnet_classifier(
            backbone=backbone,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )
    else:
        # Fail fast if a backbone type is not yet supported by the
        # classifier replacement logic, to avoid silent misconfigurations.
        raise TypeError(
            f"Unsupported backbone type for classifier replacement: {type(backbone)}"
        )

    return model
