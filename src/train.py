"""Train a model with Adam, cross-entropy, and early stopping on validation loss."""

from __future__ import annotations

import copy
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    correct = 0
    n = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if train:
            loss.backward()
            optimizer.step()
        batch = labels.size(0)
        total_loss += loss.item() * batch
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += batch
    return total_loss / n, correct / n


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    seed: int = 0,
    patience: int = 5,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict:
    """Train one model, restore the best validation-loss weights, and return history."""
    device = device or get_device()
    set_seed(seed)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    wait = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        with torch.no_grad():
            val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  epoch {epoch:02d}/{epochs}  "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}",
            flush=True,
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  early stopping at epoch {epoch} (best epoch {best_epoch})", flush=True)
                break

    model.load_state_dict(best_state)
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": best_state,
                "lr": lr,
                "seed": seed,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "best_val_acc": best_val_acc,
            },
            checkpoint_path,
        )

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "epochs_ran": len(history["val_loss"]),
        "device": str(device),
        "lr": lr,
        "seed": seed,
    }


def plot_history(history: dict, out_path: str | Path, title: str) -> Path:
    """Save train/val loss and accuracy curves."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-entropy")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
