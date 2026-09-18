"""Adebayo-style model-parameter randomisation checks for Grad-CAM.

No retraining. Load the best unaugmented CNN, keep the same five test images
and the trained model's predicted class, then reinitialise layers from the
output side (classifier, then conv2+classifier, then the whole network).
If Grad-CAM depends on learned weights, the maps should collapse.
"""

from __future__ import annotations

import copy
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data import CLASS_NAMES, get_dataloaders
from src.gradcam import (
    collect_gradcam_cases,
    generate_gradcam,
    overlay_heatmap,
    to_display_image,
)
from src.models import CNN
from src.train import get_device, set_seed

RUNS_CSV = ROOT / "results" / "runs.csv"


def _best_cnn_row() -> dict:
    with RUNS_CSV.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["run"] == "cnn"]
    if not rows:
        raise FileNotFoundError("No CNN rows in results/runs.csv. Run Step 3 first.")
    return max(rows, key=lambda row: float(row["test_accuracy"]))


def _reinit_module(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(child.weight, a=5**0.5)
            if child.bias is not None:
                fan_in = child.weight.size(1) * (
                    child.weight[0][0].numel() if child.weight.dim() == 4 else 1
                )
                if isinstance(child, nn.Linear):
                    fan_in = child.in_features
                bound = 1 / max(fan_in, 1) ** 0.5
                nn.init.uniform_(child.bias, -bound, bound)


def _randomize(model: CNN, level: str) -> CNN:
    cloned = copy.deepcopy(model)
    if level == "trained":
        return cloned
    if level in {"rand_fc", "rand_conv2", "rand_all"}:
        _reinit_module(cloned.classifier)
    if level in {"rand_conv2", "rand_all"}:
        _reinit_module(cloned.conv2)
    if level == "rand_all":
        _reinit_module(cloned.conv1)
    cloned.eval()
    return cloned


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)
    rx = x.argsort().argsort().astype(np.float64)
    ry = y.argsort().argsort().astype(np.float64)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main() -> None:
    set_seed(0)
    device = get_device()
    best = _best_cnn_row()
    checkpoint = ROOT / best["checkpoint"]
    print(f"Device: {device}")
    print(f"Best CNN seed={best['seed']}  test acc={float(best['test_accuracy']):.4f}")

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    trained = CNN()
    trained.load_state_dict(payload["model_state"])
    trained.to(device)
    trained.eval()

    splits = get_dataloaders(batch_size=128, augment=False, seed=42)
    cases = collect_gradcam_cases(trained, splits.test_loader, device)

    levels = [
        ("trained", "Trained"),
        ("rand_fc", "Rand FC"),
        ("rand_conv2", "Rand conv2+FC"),
        ("rand_all", "Rand all"),
    ]
    models = {key: _randomize(trained, key).to(device) for key, _ in levels}

    corr_rows = []
    fig, axes = plt.subplots(5, 5, figsize=(14, 12.5))
    for col, case in enumerate(cases):
        gray = to_display_image(case["image"], splits.mean, splits.std)
        axes[0, col].imshow(gray, cmap="gray")
        axes[0, col].set_title(case["title"], fontsize=9)
        axes[0, col].axis("off")

        trained_cam, _ = generate_gradcam(
            models["trained"], case["image"], class_idx=case["pred"], device=device
        )
        for row_idx, (key, label) in enumerate(levels, start=1):
            cam, _ = generate_gradcam(
                models[key], case["image"], class_idx=case["pred"], device=device
            )
            axes[row_idx, col].imshow(overlay_heatmap(gray, cam))
            axes[row_idx, col].axis("off")
            if col == 0:
                axes[row_idx, col].set_ylabel(label, fontsize=8)
            if key != "trained":
                corr_rows.append(
                    {
                        "case": case["title"],
                        "level": key,
                        "spearman_vs_trained": _spearman(trained_cam, cam),
                        "true": CLASS_NAMES[case["true"]],
                        "pred": CLASS_NAMES[case["pred"]],
                    }
                )

    axes[0, 0].set_ylabel("Image")
    fig.suptitle(
        "Adebayo model-randomisation check: Grad-CAM for the trained predicted class"
    )
    fig.tight_layout()
    out_fig = ROOT / "figures" / "gradcam_sanity.png"
    fig.savefig(out_fig, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_fig}")

    out_csv = ROOT / "results" / "gradcam_sanity.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case", "level", "spearman_vs_trained", "true", "pred"],
        )
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"Saved {out_csv}")
    print("Mean Spearman correlation vs trained map:")
    for key, label in levels[1:]:
        vals = [row["spearman_vs_trained"] for row in corr_rows if row["level"] == key]
        print(f"  {label:16s}  {sum(vals) / len(vals):.3f}")


if __name__ == "__main__":
    main()
