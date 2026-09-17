"""MLP baseline and CNN architectures for Fashion-MNIST."""

from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected baseline.

    Input is a ``(N, 1, 28, 28)`` batch, flattened to 784 inside ``forward``.
    Architecture: 784 -> 256 -> ReLU -> Dropout -> 128 -> ReLU -> Dropout -> 10 logits.
    """

    def __init__(self, dropout: float = 0.5, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN(nn.Module):
    """Small CNN used as the main COMP813 model.

    Two convolutional blocks, then a 128-unit classifier:

    * conv 3x3, 32 filters, ReLU, 2x2 max-pool
    * conv 3x3, 64 filters, ReLU, 2x2 max-pool
    * flatten -> 128 -> ReLU -> Dropout -> 10 logits

    ``conv2`` is the last convolutional layer (used later for Grad-CAM).
    """

    def __init__(self, dropout: float = 0.5, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=False)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.last_conv_layer = "conv2"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
