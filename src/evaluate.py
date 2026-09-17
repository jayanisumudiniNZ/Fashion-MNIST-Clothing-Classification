"""Test-set metrics: accuracy, macro-F1, per-class scores, confusion matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from src.data import CLASS_NAMES


def collect_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Return true labels and predicted labels for a loader."""
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            y_true.append(labels.numpy())
            y_pred.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(y_true), np.concatenate(y_pred)


def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    """Return overall and per-class metrics."""
    y_true, y_pred = collect_predictions(model, loader, device)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        zero_division=0,
    )
    per_class = []
    for i, name in enumerate(CLASS_NAMES):
        per_class.append(
            {
                "class_id": i,
                "class_name": name,
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
        )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": per_class,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def save_confusion_matrix(
    y_true,
    y_pred,
    out_path,
    class_names=None,
    title: str = "CNN confusion matrix (test set)",
) -> np.ndarray:
    """Save a labelled confusion-matrix figure and return the count matrix."""
    class_names = list(class_names or CLASS_NAMES)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    threshold = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return cm


def pair_confusion(cm: np.ndarray, true_name: str, pred_name: str) -> int:
    """Count true class ``true_name`` predicted as ``pred_name``."""
    true_idx = CLASS_NAMES.index(true_name)
    pred_idx = CLASS_NAMES.index(pred_name)
    return int(cm[true_idx, pred_idx])
