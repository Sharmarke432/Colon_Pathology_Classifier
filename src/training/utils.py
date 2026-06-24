"""
General training utilities.

This module contains helper functions that are broadly useful across the
training pipeline, including reproducibility helpers (random seeding),
checkpoint saving/loading, and simple logging wrappers.
"""

import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from torch import nn


def set_random_seed(seed: int) -> None:
    """
    Set the random seed for common libraries.

    This function configures the random number generators for Python's
    built-in `random` module, NumPy, and PyTorch so that training runs
    become more reproducible.

    Parameters
    ----------
    seed : int
        The seed value to apply to all supported libraries.
    """
    # Configure the Python standard library's random generator.
    random.seed(seed)

    # Configure NumPy's global random generator.
    np.random.seed(seed)

    # Configure PyTorch's CPU random generator.
    torch.manual_seed(seed)

    # Disable certain sources of nondeterminism in PyTorch where
    # possible. Note that full determinism is difficult to guarantee,
    # but these settings improve consistency across runs.[web:16]
    torch.use_deterministic_algorithms(False)


def save_checkpoint(
    model: nn.Module,
    path: Path,
    extra_state: Dict[str, Any] | None = None,
) -> None:
    """
    Save a model checkpoint to disk.

    This function serializes the model's state dictionary and any
    optional extra state information (such as epoch number or metrics)
    into a single file using `torch.save`.

    Parameters
    ----------
    model : torch.nn.Module
        The model whose parameters should be saved.
    path : pathlib.Path
        Filesystem path where the checkpoint will be written.
    extra_state : dict, optional
        Additional metadata to include in the checkpoint file.
    """
    # Ensure the parent directory exists before attempting to write the file.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Prepare the checkpoint payload with at least the model's state dict.
    checkpoint: Dict[str, Any] = {
        "model_state_dict": model.state_dict(),
    }

    # Optionally merge extra state information into the checkpoint.
    if extra_state is not None:
        checkpoint.update(extra_state)

    # Serialize the checkpoint to disk using PyTorch's built-in helper.
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: Path,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """
    Load a model checkpoint from disk.

    This function deserializes a previously saved checkpoint file and
    loads the model parameters into the provided model instance. It also
    returns any remaining metadata stored in the checkpoint so that the
    caller can inspect values such as the last epoch or best metrics.

    Parameters
    ----------
    model : torch.nn.Module
        The model into which the checkpoint parameters will be loaded.
    path : pathlib.Path
        Filesystem path pointing to the checkpoint file.
    map_location : str or torch.device, optional
        Device mapping to use when loading the checkpoint. Defaults to
        "cpu" so that GPU-based checkpoints can be safely loaded on CPU
        machines.

    Returns
    -------
    dict
        A dictionary containing the checkpoint data, excluding the
        "model_state_dict" entry which has already been applied to the
        model.
    """
    # Load the full checkpoint dictionary from disk.
    checkpoint = torch.load(path, map_location=map_location)

    # Extract and load the stored model parameters.
    state_dict = checkpoint.pop("model_state_dict")
    model.load_state_dict(state_dict)

    # Return any remaining metadata to the caller.
    return checkpoint
