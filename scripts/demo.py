"""Short runnable demo for the 2-5 minute COMP813 video.

Shows a 1-epoch training run (allowed when full training is long), then
reloads the saved full CNN and prints representative test results.
Does not overwrite results/metrics.csv from the 3-seed experiments.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data import CLASS_NAMES, get_dataloaders
from src.evaluate import evaluate_model, pair_confusion, save_confusion_matrix
from src.gradcam import save_gradcam_grid
from src.models import CNN, MLP, count_parameters
from src.train import get_device, train_model

METRICS_CSV = ROOT / "results" / "metrics.csv"
RUNS_CSV = ROOT / "results" / "runs.csv"


def _best_cnn_row() -> dict:
    with RUNS_CSV.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["run"] == "cnn"]
    if not rows:
        raise FileNotFoundError("Run Step 3 first so results/runs.csv exists.")
    return max(rows, key=lambda row: float(row["test_accuracy"]))


def _print_metrics_table() -> None:
    print("\nFull 3-seed test results (already saved from Step 3):")
    print(f"{'run':<10} {'accuracy':>12} {'macro-F1':>12}")
    with METRICS_CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            acc = float(row["accuracy_mean"]) * 100
            f1 = float(row["macro_f1_mean"]) * 100
            acc_std = float(row["accuracy_std"]) * 100
            print(
                f"{row['run']:<10} {acc:6.2f}%±{acc_std:.2f}  {f1:6.2f}%"
            )


def _open_figures(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if not existing:
        return
    try:
        subprocess.run(["open", *existing], check=False)
    except OSError:
        print("Open the PNG files in figures/ in Preview.")


def main() -> None:
    device = get_device()
    print("=" * 64)
    print("COMP813 demo: Fashion-MNIST MLP vs CNN + Grad-CAM")
    print("=" * 64)
    print(f"Device: {device}")
    print("No camera. Official 28x28 test images only.\n")

    print("1) Load data (54,000 train / 6,000 val / 10,000 test)")
    splits = get_dataloaders(batch_size=128, augment=False, seed=42)
    images, labels = next(iter(splits.train_loader))
    print(
        f"   train={splits.n_train} val={splits.n_val} test={splits.n_test}  "
        f"batch={tuple(images.shape)}"
    )
    print(f"   example label: {CLASS_NAMES[int(labels[0])]}")

    print("\n2) Models")
    mlp, cnn = MLP(), CNN()
    print(f"   MLP parameters: {count_parameters(mlp):,}")
    print(f"   CNN parameters: {count_parameters(cnn):,}")

    print("\n3) Small training run (1 epoch) — full training was 30 epochs")
    demo_cnn = CNN()
    demo_result = train_model(
        demo_cnn,
        splits.train_loader,
        splits.val_loader,
        epochs=1,
        lr=3e-4,
        seed=0,
        patience=1,
        device=device,
        checkpoint_path=ROOT / "results" / "checkpoints" / "demo_one_epoch.pt",
    )
    print(
        f"   1-epoch val acc={demo_result['best_val_acc']:.4f}  "
        "(full CNN test acc is ~92.2%)"
    )

    print("\n4) Reload the saved full CNN and score the 10,000 test images")
    best = _best_cnn_row()
    ckpt = ROOT / best["checkpoint"]
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    full_cnn = CNN()
    full_cnn.load_state_dict(payload["model_state"])
    full_cnn.to(device)
    metrics = evaluate_model(full_cnn, splits.test_loader, device)
    print(
        f"   checkpoint seed={best['seed']}  "
        f"test acc={metrics['accuracy']:.4f}  macro-F1={metrics['macro_f1']:.4f}"
    )

    cm_path = ROOT / "figures" / "confusion_cnn.png"
    cm = save_confusion_matrix(
        metrics["y_true"],
        metrics["y_pred"],
        cm_path,
        title=f"CNN (no aug, seed {best['seed']}) on 10,000 test images",
    )
    print(
        f"   Shirt → T-shirt/top: {pair_confusion(cm, 'Shirt', 'T-shirt/top')}  "
        f"T-shirt/top → Shirt: {pair_confusion(cm, 'T-shirt/top', 'Shirt')}"
    )

    print("\n5) Grad-CAM on predicted class")
    gradcam_path = ROOT / "figures" / "gradcam_grid.png"
    save_gradcam_grid(
        full_cnn,
        splits.test_loader,
        gradcam_path,
        mean=splits.mean,
        std=splits.std,
        device=device,
    )
    print(f"   saved {gradcam_path.name}")

    _print_metrics_table()
    print("\nOpening result figures...")
    _open_figures(
        [
            ROOT / "figures" / "sample_train.png",
            cm_path,
            gradcam_path,
            ROOT / "figures" / "lr_sweep.png",
        ]
    )
    print("Demo finished.")


if __name__ == "__main__":
    main()
