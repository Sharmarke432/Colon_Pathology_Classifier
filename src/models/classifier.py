"""
Classifier head definitions.

This module defines simple classifier heads that can be attached to a
feature extractor backbone such as ResNet18 or EfficientNet-B0. The goal
is to keep the classification logic reusable and decoupled from the
backbone definitions so that different architectures can share the same
output head implementation.
"""

from typing import Optional

import torch
from torch import nn


class LinearClassifier(nn.Module):
    """
    A simple linear classifier head.

    This module takes an input feature vector and maps it directly to
    class logits using a single fully connected layer. It can optionally
    apply dropout before the linear layer to improve regularization for
    small datasets such as PathMNIST.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        dropout_p: float = 0.0,
    ) -> None:
        """
        Initialize the classifier head.

        Parameters
        ----------
        in_features : int
            The dimensionality of the input feature vector produced by
            the backbone network.
        num_classes : int
            The number of output classes for the classification task,
            for example nine classes in the PathMNIST dataset.[web:10]
        dropout_p : float, optional
            The dropout probability applied before the linear layer.
            Defaults to 0.0, which means no dropout is used. This keeps
            the classifier lightweight and CPU-friendly by default.
        """
        super().__init__()

        # Conditionally create a dropout layer
        # if the requested probability is greater than zero.
        self.dropout: Optional[nn.Dropout] = (
            nn.Dropout(p=dropout_p) if dropout_p > 0.0 else None
        )

        # Define the final linear layer that maps features to logits.
        # The `bias` term is enabled by default because it is common in
        # classification heads and the computational cost is negligible
        # for CPU-only training on small models.
        self.fc = nn.Linear(in_features, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_features) representing
            the features produced by the backbone.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, num_classes) containing
            raw class logits for downstream loss functions such as
            `torch.nn.CrossEntropyLoss`.
        """
        # Optionally apply dropout
        # if it was configured during initialization.
        if self.dropout is not None:
            x = self.dropout(x)

        # Apply the linear layer to obtain class logits.
        logits = self.fc(x)

        return logits
