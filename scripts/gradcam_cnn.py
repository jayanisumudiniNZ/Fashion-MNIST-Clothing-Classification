"""Grad-CAM overlays for the best CNN checkpoint."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.data import get_dataloaders
from src.gradcam import save_gradcam_grid
from src.models import CNN
from src.train import get_device

RUNS_CSV = ROOT / "results" / "runs.csv"


def _best_cnn_row() -> dict:
    with RUNS_CSV.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["run"] == "cnn"]
    if not rows:
        raise FileNotFoundError("No CNN rows in results/runs.csv. Run Step 3 first.")
    return max(rows, key=lambda row: float(row["test_accuracy"]))


def main() -> None:
    device = get_device()
    best = _best_cnn_row()
    checkpoint = ROOT / best["checkpoint"]
    print(f"Device: {device}")
    print(f"Best CNN seed={best['seed']}  test acc={float(best['test_accuracy']):.4f}")
    print(f"Checkpoint: {checkpoint}")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CNN()
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()

    splits = get_dataloaders(batch_size=128, augment=False, seed=42)
    out_path = ROOT / "figures" / "gradcam_grid.png"
    save_gradcam_grid(
        model,
        splits.test_loader,
        out_path,
        mean=splits.mean,
        std=splits.std,
        device=device,
    )
    print(f"Saved {out_path}")
    print("Heatmaps use the predicted class. Check whether they sit on garment parts, not background.")


if __name__ == "__main__":
    main()
