"""
Metric computation utilities.

This module implements basic evaluation metrics for classification
tasks, such as accuracy. The functions are designed to operate on
PyTorch tensors so that they can be easily integrated into training and
evaluation loops.
"""

import torch


def accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    """
    Compute classification accuracy.

    This function compares the predicted class indices (obtained by
    taking the argmax over the logits) with the ground-truth targets and
    returns the fraction of correct predictions.

    Parameters
    ----------
    logits : torch.Tensor
        Tensor of shape (batch_size, num_classes) containing raw class
        scores produced by the model.
    targets : torch.Tensor
        Tensor of shape (batch_size,) containing integer class labels.

    Returns
    -------
    float
        The accuracy value between 0.0 and 1.0 representing the
        proportion of correctly classified samples in the batch.
    """
    # Obtain the predicted class indices by selecting the maximum logit
    # along the class dimension.
    preds = torch.argmax(logits, dim=1)

    # Compute the number of correct predictions by comparing with the
    # ground-truth labels.
    correct = (preds == targets).sum().item()

    # Avoid division by zero by handling empty batches explicitly.
    total = targets.numel()
    if total == 0:
        return 0.0

    return correct / total
