"""Load Fashion-MNIST, split train/val/test, preprocess, and apply training-only augmentation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

VAL_FRACTION = 0.1
MAX_SHIFT_PIXELS = 2
CROP_PADDING = 4
IMAGE_SIZE = 28
AUG_FLIP_SHIFT = "flip_shift"
AUG_CROP = "crop"


@dataclass
class FashionMNISTSplits:
    """Train / validation / test loaders plus training-set normalisation stats."""

    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    mean: float
    std: float
    n_train: int
    n_val: int
    n_test: int
    class_names: list[str]
    augment: bool
    aug_kind: str


def _default_data_dir(data_dir: str | Path | None) -> Path:
    if data_dir is None:
        return Path(__file__).resolve().parents[1] / "data"
    return Path(data_dir)


def _stratified_train_val_indices(targets, val_fraction: float, seed: int):
    targets = np.asarray(targets)
    indices = np.arange(len(targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=targets,
        random_state=seed,
    )
    return train_idx.tolist(), val_idx.tolist()


def _train_mean_std(raw_dataset: datasets.FashionMNIST, train_idx: list[int]):
    """Mean and std of pixels in [0, 1], using the training split only."""
    images = raw_dataset.data[train_idx].float() / 255.0
    return float(images.mean()), float(images.std())


def _train_transforms(mean: float, std: float, augment: bool, aug_kind: str = AUG_FLIP_SHIFT):
    geometric = []
    if augment:
        if aug_kind == AUG_CROP:
            geometric = [
                transforms.RandomCrop(IMAGE_SIZE, padding=CROP_PADDING, fill=0),
            ]
        else:
            translate = (MAX_SHIFT_PIXELS / IMAGE_SIZE, MAX_SHIFT_PIXELS / IMAGE_SIZE)
            geometric = [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(degrees=0, translate=translate, fill=0),
            ]
    return transforms.Compose(
        geometric
        + [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )


def _eval_transforms(mean: float, std: float):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )


def get_dataloaders(
    batch_size: int = 128,
    augment: bool = False,
    aug_kind: str = AUG_FLIP_SHIFT,
    data_dir: str | Path | None = None,
    num_workers: int = 0,
    seed: int = 42,
    shuffle_seed: int | None = None,
) -> FashionMNISTSplits:
    """Return train, validation, and test dataloaders.

    Validation is a stratified 10% hold-out from the official training set.
    ``seed`` controls that split so it stays fixed across training runs.
    ``shuffle_seed`` only changes training-batch order (defaults to ``seed``).
    Training-only augmentation when ``augment=True``:
    * ``flip_shift``: horizontal flip (p=0.5) and at most 2-pixel shift
    * ``crop``: random crop after 4-pixel padding (isolated from flip/shift)
    Validation and test are never augmented. Normalisation uses the
    training-split mean and std.
    """
    if aug_kind not in {AUG_FLIP_SHIFT, AUG_CROP}:
        raise ValueError(f"Unknown aug_kind: {aug_kind}")
    kind = aug_kind if augment else "none"
    data_path = _default_data_dir(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    raw_train = datasets.FashionMNIST(root=str(data_path), train=True, download=True)
    train_idx, val_idx = _stratified_train_val_indices(
        raw_train.targets, VAL_FRACTION, seed
    )
    mean, std = _train_mean_std(raw_train, train_idx)

    train_ds = datasets.FashionMNIST(
        root=str(data_path),
        train=True,
        download=False,
        transform=_train_transforms(mean, std, augment, aug_kind),
    )
    val_ds = datasets.FashionMNIST(
        root=str(data_path),
        train=True,
        download=False,
        transform=_eval_transforms(mean, std),
    )
    test_ds = datasets.FashionMNIST(
        root=str(data_path),
        train=False,
        download=True,
        transform=_eval_transforms(mean, std),
    )

    generator = torch.Generator().manual_seed(
        seed if shuffle_seed is None else shuffle_seed
    )
    train_loader = DataLoader(
        Subset(train_ds, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return FashionMNISTSplits(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        mean=mean,
        std=std,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_ds),
        class_names=list(CLASS_NAMES),
        augment=augment,
        aug_kind=kind,
    )


def denormalize(images: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Undo training-set normalisation for plotting."""
    return images * std + mean


def plot_training_samples(
    splits: FashionMNISTSplits,
    out_path: str | Path,
    n_images: int = 12,
) -> Path:
    """Save a grid of training images with class names."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    images, labels = next(iter(splits.train_loader))
    n_images = min(n_images, images.size(0))
    cols = 6
    rows = int(np.ceil(n_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    shown = denormalize(images[:n_images], splits.mean, splits.std).clamp(0, 1)

    for i, ax in enumerate(axes):
        if i < n_images:
            ax.imshow(shown[i].squeeze().cpu().numpy(), cmap="gray")
            ax.set_title(CLASS_NAMES[int(labels[i])], fontsize=9)
        ax.axis("off")

    title = "Training samples"
    if splits.aug_kind == AUG_CROP:
        title += f" (random crop, pad={CROP_PADDING})"
    elif splits.aug_kind == AUG_FLIP_SHIFT:
        title += " (with flip + shift)"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
