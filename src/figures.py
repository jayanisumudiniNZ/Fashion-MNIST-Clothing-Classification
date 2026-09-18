"""Report figures that can be regenerated from CSVs, histories, and checkpoints."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
HISTORIES = ROOT / "results" / "histories"
LATEX_FIGURES = ROOT.parent / "cvpr2017AuthorKit" / "latex" / "figures"

REPORT_FIGURES = [
    "sample_train.png",
    "sample_train_aug.png",
    "sample_train_crop.png",
    "method_pipeline.png",
    "lr_sweep.png",
    "curves_mlp_seed0.png",
    "curves_cnn_seed0.png",
    "curves_cnn_aug_seed0.png",
    "curves_cnn_crop_seed0.png",
    "per_class_f1.png",
    "confusion_cnn.png",
    "gradcam_grid.png",
    "gradcam_sanity.png",
]


def save_history(history: dict, run: str, seed: int) -> Path:
    HISTORIES.mkdir(parents=True, exist_ok=True)
    path = HISTORIES / f"{run}_seed{seed}.json"
    path.write_text(json.dumps(history, indent=2))
    return path


def load_history(run: str, seed: int) -> dict:
    path = HISTORIES / f"{run}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Train that run or restore results/histories/."
        )
    return json.loads(path.read_text())


def plot_lr_sweep(rows: list[dict], out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    labels = [f"{float(row['lr']):g}" for row in rows]
    accs = [float(row["best_val_acc"]) for row in rows]
    ax.bar(labels, accs, color="#4C78A8")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("CNN learning-rate sweep (no augmentation)")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_lr_sweep_from_csv(csv_path: Path, out_path: Path) -> Path:
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"No rows in {csv_path}")
    return plot_lr_sweep(rows, out_path)


def plot_method_pipeline(out_path: Path) -> Path:
    """Matched-training pipeline used in the Methods section."""
    fig, ax = plt.subplots(figsize=(13.6, 3.6))
    ax.set_xlim(0, 13.6)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    ax.set_title(
        "Method pipeline: matched MLP vs CNN training, then CNN-only Grad-CAM",
        fontsize=11,
        pad=8,
    )

    boxes = [
        (0.15, 1.25, 2.15, 1.15, "Fashion-MNIST\n$28\\times 28$ grayscale\n10 classes"),
        (2.7, 1.25, 2.2, 1.15, "Preprocess\n$[0,1]$ + train\nmean/std"),
        (5.3, 2.15, 2.55, 1.05, "MLP baseline\n$784\\rightarrow 256\\rightarrow 128\\rightarrow 10$"),
        (5.3, 0.4, 2.55, 1.15, "CNN (main model)\nconv32, pool, conv64, pool\nFC $128\\rightarrow 10$"),
        (8.25, 1.25, 2.25, 1.15, "Softmax / argmax\n10-class logits"),
        (10.85, 2.15, 2.5, 1.05, "Test metrics\nacc, macro-$F_1$, CM"),
        (10.85, 0.45, 2.5, 1.1, "Grad-CAM on conv2\n(CNN only)"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                linewidth=1.1,
                edgecolor="#4C78A8",
                facecolor="#D6E6F5",
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color="#4C4C4C",
            )
        )

    arrow(2.3, 1.82, 2.7, 1.82)
    arrow(4.9, 1.82, 5.3, 2.67)
    arrow(4.9, 1.82, 5.3, 0.97)
    arrow(7.85, 2.67, 8.25, 1.95)
    arrow(7.85, 0.97, 8.25, 1.7)
    arrow(10.5, 1.95, 10.85, 2.67)
    arrow(10.5, 1.7, 10.85, 1.0)
    ax.text(6.55, 1.72, "same CE + Adam\nearly stopping", ha="center", va="center", fontsize=7, color="#555555")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def copy_report_figures(latex_dir: Path | None = None) -> list[Path]:
    dest = Path(latex_dir or LATEX_FIGURES)
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in REPORT_FIGURES:
        src = FIGURES / name
        if src.exists():
            target = dest / name
            shutil.copy2(src, target)
            copied.append(target)
    extra_curves = sorted(FIGURES.glob("curves_*.png"))
    for src in extra_curves:
        shutil.copy2(src, dest / src.name)
        copied.append(dest / src.name)
    return copied
