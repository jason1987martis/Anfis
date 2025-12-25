"""Training utilities for the GPU-enabled ANFIS implementation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from dataset_loader import BRFSSDatasetLoader
from model import ANFISNetwork


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_accuracy: list[float] = field(default_factory=list)
    val_accuracy: list[float] = field(default_factory=list)


def train_anfis(
    model: ANFISNetwork,
    dataloaders: Dict[str, DataLoader],
    *,
    epochs: int = 25,
    optimizer: Optional[torch.optim.Optimizer] = None,
    loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    device: Optional[torch.device] = None,
    classification: bool = True,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    threshold: float = 0.5,
) -> TrainingHistory:
    """
    Train the ANFIS model using GPU acceleration when available.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = TrainingHistory()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in dataloaders["train"]:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

            if classification:
                preds = (outputs >= threshold).float()
                correct += (preds == targets).sum().item()
                total += targets.numel()

        mean_train_loss = train_loss / len(dataloaders["train"].dataset)
        train_acc = correct / total if classification and total > 0 else float("nan")

        history.train_loss.append(mean_train_loss)
        history.train_accuracy.append(train_acc)

        val_loss, val_acc = _evaluate(
            model,
            dataloaders.get("val"),
            loss_fn,
            device,
            classification=classification,
            threshold=threshold,
        )

        history.val_loss.append(val_loss)
        history.val_accuracy.append(val_acc)

        if scheduler is not None:
            scheduler.step(val_loss or mean_train_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {mean_train_loss:.4f} "
            f"Val Loss: {val_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Acc: {val_acc:.4f}"
        )

    return history


@torch.no_grad()
def _evaluate(
    model: ANFISNetwork,
    dataloader: Optional[DataLoader],
    loss_fn: Callable[[Tensor, Tensor], Tensor],
    device: torch.device,
    *,
    classification: bool,
    threshold: float,
) -> tuple[float, float]:
    if dataloader is None:
        return float("nan"), float("nan")

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        total_loss += loss.item() * inputs.size(0)

        if classification:
            preds = (outputs >= threshold).float()
            correct += (preds == targets).sum().item()
            total += targets.numel()

    mean_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / total if classification and total > 0 else float("nan")
    return mean_loss, accuracy


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPU-enabled ANFIS on the BRFSS dataset.")
    parser.add_argument("--data", required=True, help="Path to the Kaggle CSV export.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--mf-count", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--normalization",
        choices=("standard", "minmax", "none"),
        default="standard",
    )
    parser.add_argument("--sample", type=int, default=None, help="Optional subsample size for quick experiments.")
    parser.add_argument("--target", type=str, default=BRFSSDatasetLoader.DEFAULT_TARGET)
    args = parser.parse_args()

    loader = BRFSSDatasetLoader(
        args.data,
        normalization=args.normalization,
        sample_size=args.sample,
        target_column=args.target,
    )
    splits = loader.load()
    dataloaders = splits.to_dataloaders(batch_size=args.batch_size)

    input_dim = splits.train.tensors[0].shape[1]
    model = ANFISNetwork(input_dim=input_dim, mf_count=args.mf_count)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.initialize_memberships(splits.train.tensors[0].to(device))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = train_anfis(
        model,
        dataloaders,
        epochs=args.epochs,
        optimizer=optimizer,
        device=device,
        classification=True,
    )

    print("Training complete.")
    print(f"Final validation loss: {history.val_loss[-1]:.4f}")
    print(f"Final validation accuracy: {history.val_accuracy[-1]:.4f}")


if __name__ == "__main__":
    main()

