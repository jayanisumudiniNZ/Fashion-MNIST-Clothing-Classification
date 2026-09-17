"""Download Fashion-MNIST, print split sizes, and save a sample image grid."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_dataloaders, plot_training_samples


def _print_splits(splits, title: str) -> None:
    print(title)
    print(f"  train={splits.n_train}  val={splits.n_val}  test={splits.n_test}")
    print(f"  train mean={splits.mean:.4f}  std={splits.std:.4f}")
    print(f"  augmentation={splits.augment}")
    images, labels = next(iter(splits.train_loader))
    print(f"  batch images={tuple(images.shape)}  labels={tuple(labels.shape)}")


def main() -> None:
    figures = ROOT / "figures"

    plain = get_dataloaders(batch_size=128, augment=False)
    _print_splits(plain, "Without augmentation (MLP / CNN baseline)")
    plot_training_samples(plain, figures / "sample_train.png")
    print(f"  saved {figures / 'sample_train.png'}")

    augmented = get_dataloaders(batch_size=128, augment=True)
    _print_splits(augmented, "With training augmentation (CNN + aug)")
    plot_training_samples(augmented, figures / "sample_train_aug.png")
    print(f"  saved {figures / 'sample_train_aug.png'}")


if __name__ == "__main__":
    main()
