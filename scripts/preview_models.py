"""Sanity-check MLP and CNN forward passes on one Fashion-MNIST batch."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data import get_dataloaders
from src.models import CNN, MLP, count_parameters


def _check(model, images, name: str) -> None:
    model.eval()
    with torch.no_grad():
        logits = model(images)
    n_params = count_parameters(model)
    print(f"{name}")
    print(f"  parameters={n_params:,}")
    print(f"  input={tuple(images.shape)}  logits={tuple(logits.shape)}")
    assert logits.shape == (images.size(0), 10), f"{name} must output (N, 10)"


def main() -> None:
    splits = get_dataloaders(batch_size=128, augment=False)
    images, _ = next(iter(splits.train_loader))
    _check(MLP(), images, "MLP")
    _check(CNN(), images, "CNN")
    print("Both models accept (N, 1, 28, 28) and return 10-class logits.")


if __name__ == "__main__":
    main()
