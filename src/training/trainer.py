"""
Training loop implementation.

This module defines a simple CPU-first training loop that can be used to
optimize classification models on the PathMNIST dataset. The design
prioritizes clarity and reproducibility over maximum performance so that
the code remains easy to understand and extend.
"""

from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import accuracy
from .utils import save_checkpoint
from src.core.config import TrainingConfig, get_torch_device

# ^^^ adjust this import to where your TrainingConfig actually lives


class Trainer:
    """
    A lightweight trainer for classification models.

    This class encapsulates the training logic, including forward and
    backward passes, metric computation, and checkpointing. It is
    intended to be used by the `train.py` entrypoint script.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
    ) -> None:
        """
        Initialize the trainer with its dependencies.

        Parameters
        ----------
        model : torch.nn.Module
            The classification model to be trained.
        optimizer : torch.optim.Optimizer
            Optimizer used to update model parameters.
        criterion : torch.nn.Module
            Loss function (e.g., `nn.CrossEntropyLoss`).
        train_loader : torch.utils.data.DataLoader
            DataLoader providing training batches.
        val_loader : torch.utils.data.DataLoader
            DataLoader providing validation batches.
        config : TrainingConfig
            Configuration object with hyperparameters and paths.
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        # Resolve and store the PyTorch device (CPU in this project).
        self.device = get_torch_device(config)

        # Move model parameters to the chosen device.
        self.model.to(self.device)

        # Track the best validation accuracy observed so far.
        self.best_val_accuracy: float = 0.0

    def train(self) -> None:
        """
        Run the full training loop.

        This method iterates over epochs, runs a training epoch followed
        by a validation epoch, and saves a checkpoint whenever the
        validation accuracy improves.
        """
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch  # for logging
            train_loss, train_acc = self._run_epoch(
                data_loader=self.train_loader,
                training=True,
            )

            val_loss, val_acc = self._run_epoch(
                data_loader=self.val_loader,
                training=False,
            )

            # If validation accuracy improved, save a new checkpoint.
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc

                extra_state: Dict[str, Any] = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                }

                save_checkpoint(
                    model=self.model,
                    path=self.config.checkpoint_path(),
                    extra_state=extra_state,
                )

            print(
                f"Epoch {epoch+1}/{self.config.num_epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

    def _run_epoch(
        self,
        data_loader: DataLoader,
        training: bool,
    ) -> tuple[float, float]:
        """
        Run one epoch over a given DataLoader.

        Depending on the `training` flag, this either performs parameter
        updates (training) or a pure evaluation pass (validation).

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            DataLoader to iterate over.
        training : bool
            If True, enable gradient computation and update parameters.
            If False, run in evaluation mode without updates.

        Returns
        -------
        tuple[float, float]
            Average loss and accuracy across the epoch.
        """
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        # Use `torch.no_grad()` during evaluation to save memory and
        # computation; use normal gradient tracking during training.
        context = torch.enable_grad if training else torch.no_grad

        with context():
            for inputs, targets in data_loader:
                # Move inputs to the configured device.
                inputs = inputs.to(self.device)

                # Ensure targets have shape (batch_size,) and dtype long.
                # This handles MedMNIST labels that come in as (B, 1) or
                # with extra dimensions, which would otherwise break
                # CrossEntropyLoss.
                targets = targets.view(-1).to(self.device, dtype=torch.long)

                if training:
                    # Reset gradients before each optimization step.
                    self.optimizer.zero_grad()

                # Forward pass.
                logits = self.model(inputs)

                # Compute loss.
                loss = self.criterion(logits, targets)

                if training:
                    # Backward pass and parameter update.
                    loss.backward()
                    self.optimizer.step()

                # Compute accuracy for this batch.
                batch_acc = accuracy(logits=logits, targets=targets)

                log_every = 10  # print every 10 batches

                if training and (num_batches % log_every == 0):
                    print(
                        f"[epoch={self.current_epoch} "
                        f"batch={num_batches}] "
                        f"loss={loss.item():.4f} acc={batch_acc:.3f}"
                    )
                total_loss += loss.item()
                total_acc += batch_acc
                num_batches += 1

        if num_batches == 0:
            return 0.0, 0.0

        avg_loss = total_loss / num_batches
        avg_acc = total_acc / num_batches

        return avg_loss, avg_acc
