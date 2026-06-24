"""
Tests for the training utilities and Trainer class.

These tests run a very small training loop on synthetic data to confirm
that the Trainer integrates correctly with the model and optimizer.
"""

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import create_model
from src.core.config import TrainingConfig, get_torch_device
from src.training.train import Trainer
from src.training.utils import set_random_seed
from src.training.constants import NUM_CLASSES, IMAGE_HEIGHT, IMAGE_WIDTH


def test_trainer_runs_single_epoch(tmp_path: Path) -> None:
    """
    Verify that the Trainer can run a single epoch on synthetic data
    without raising errors and that it saves a checkpoint.
    """
    # Set a seed for reproducibility.
    set_random_seed(0)

    # Create a small synthetic dataset.
    num_samples = 16
    inputs = torch.randn(num_samples, 3, IMAGE_HEIGHT, IMAGE_WIDTH)
    targets = torch.randint(
        low=0,
        high=NUM_CLASSES,
        size=(num_samples,),
    )

    dataset = TensorDataset(inputs, targets)

    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Create a lightweight configuration pointing to the temporary directory.
    config = TrainingConfig(
        data_dir=Path("data"),
        output_dir=tmp_path,
        num_classes=NUM_CLASSES,
        batch_size=8,
        num_workers=0,
        num_epochs=1,
        learning_rate=1e-3,
        device="cpu",
    )

    device = get_torch_device(config)

    # Build the model, loss function, and optimizer.
    model = create_model(
        backbone_name="resnet18",
        num_classes=NUM_CLASSES,
        pretrained=False,
        dropout_p=0.0,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Instantiate the trainer and run training.
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    trainer.train()

    # Confirm that a checkpoint file was created.
    assert config.checkpoint_path().exists()
