"""CNN learning-rate sweep, then MLP / CNN / CNN+aug reported runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, get_dataloaders
from src.evaluate import evaluate_model
from src.models import CNN, MLP
from src.train import get_device, plot_history, train_model

LR_CANDIDATES = [1e-3, 3e-4, 1e-4]
SEEDS = [0, 1, 2]
SPLIT_SEED = 42
SWEEP_SEED = 0

RUNS = [
    {"name": "mlp", "model": "mlp", "augment": False},
    {"name": "cnn", "model": "cnn", "augment": False},
    {"name": "cnn_aug", "model": "cnn", "augment": True},
]


def _build_model(name: str):
    if name == "mlp":
        return MLP()
    if name == "cnn":
        return CNN()
    raise ValueError(f"Unknown model: {name}")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_lr_sweep(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    labels = [f"{row['lr']:g}" for row in rows]
    accs = [row["best_val_acc"] for row in rows]
    ax.bar(labels, accs, color="#4C78A8")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Best validation accuracy")
    ax.set_title("CNN learning-rate sweep (no augmentation)")
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_lr_sweep(args, device) -> float:
    print(f"\n=== CNN learning-rate sweep on validation (seed={SWEEP_SEED}) ===")
    splits = get_dataloaders(
        batch_size=args.batch_size,
        augment=False,
        seed=SPLIT_SEED,
        shuffle_seed=SWEEP_SEED,
    )
    rows = []
    for lr in args.lrs:
        print(f"\nSweep CNN lr={lr:g}")
        result = train_model(
            CNN(),
            splits.train_loader,
            splits.val_loader,
            epochs=args.epochs,
            lr=lr,
            seed=SWEEP_SEED,
            patience=args.patience,
            device=device,
        )
        rows.append(
            {
                "lr": lr,
                "best_val_loss": result["best_val_loss"],
                "best_val_acc": result["best_val_acc"],
                "best_epoch": result["best_epoch"],
                "epochs_ran": result["epochs_ran"],
            }
        )

    best = max(rows, key=lambda row: (row["best_val_acc"], -row["best_val_loss"]))
    best_lr = float(best["lr"])
    _write_csv(
        ROOT / "results" / "lr_sweep.csv",
        rows,
        ["lr", "best_val_loss", "best_val_acc", "best_epoch", "epochs_ran"],
    )
    _plot_lr_sweep(rows, ROOT / "figures" / "lr_sweep.png")
    payload = {
        "best_lr": best_lr,
        "best_val_acc": best["best_val_acc"],
        "best_val_loss": best["best_val_loss"],
        "candidates": rows,
    }
    (ROOT / "results" / "best_lr.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSelected learning rate: {best_lr:g} (val acc={best['best_val_acc']:.4f})")
    return best_lr


def run_reported_experiments(args, device, lr: float) -> None:
    print(f"\n=== Reported runs with lr={lr:g} ===")
    run_rows = []
    per_class_rows = []
    ckpt_dir = ROOT / "results" / "checkpoints"

    for spec in RUNS:
        for seed in args.seeds:
            print(f"\nRun {spec['name']}  seed={seed}  augment={spec['augment']}")
            splits = get_dataloaders(
                batch_size=args.batch_size,
                augment=spec["augment"],
                seed=SPLIT_SEED,
                shuffle_seed=seed,
            )
            model = _build_model(spec["model"])
            ckpt = ckpt_dir / f"{spec['name']}_seed{seed}.pt"
            result = train_model(
                model,
                splits.train_loader,
                splits.val_loader,
                epochs=args.epochs,
                lr=lr,
                seed=seed,
                patience=args.patience,
                device=device,
                checkpoint_path=ckpt,
            )
            test_metrics = evaluate_model(model, splits.test_loader, device)
            plot_history(
                result["history"],
                ROOT / "figures" / f"curves_{spec['name']}_seed{seed}.png",
                title=f"{spec['name']} seed={seed} lr={lr:g}",
            )
            run_rows.append(
                {
                    "run": spec["name"],
                    "model": spec["model"],
                    "augment": spec["augment"],
                    "seed": seed,
                    "lr": lr,
                    "best_epoch": result["best_epoch"],
                    "best_val_loss": result["best_val_loss"],
                    "best_val_acc": result["best_val_acc"],
                    "test_accuracy": test_metrics["accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "checkpoint": str(ckpt.relative_to(ROOT)),
                }
            )
            for row in test_metrics["per_class"]:
                per_class_rows.append(
                    {
                        "run": spec["name"],
                        "seed": seed,
                        **row,
                    }
                )

    _write_csv(
        ROOT / "results" / "runs.csv",
        run_rows,
        [
            "run",
            "model",
            "augment",
            "seed",
            "lr",
            "best_epoch",
            "best_val_loss",
            "best_val_acc",
            "test_accuracy",
            "test_macro_f1",
            "checkpoint",
        ],
    )
    _write_csv(
        ROOT / "results" / "per_class.csv",
        per_class_rows,
        ["run", "seed", "class_id", "class_name", "precision", "recall", "f1", "support"],
    )

    summary_rows = []
    for spec in RUNS:
        subset = [row for row in run_rows if row["run"] == spec["name"]]
        acc = [row["test_accuracy"] for row in subset]
        f1 = [row["test_macro_f1"] for row in subset]
        summary_rows.append(
            {
                "run": spec["name"],
                "model": spec["model"],
                "augment": spec["augment"],
                "lr": lr,
                "n_seeds": len(subset),
                "accuracy_mean": statistics.mean(acc),
                "accuracy_std": statistics.stdev(acc) if len(acc) > 1 else 0.0,
                "macro_f1_mean": statistics.mean(f1),
                "macro_f1_std": statistics.stdev(f1) if len(f1) > 1 else 0.0,
            }
        )
    _write_csv(
        ROOT / "results" / "metrics.csv",
        summary_rows,
        [
            "run",
            "model",
            "augment",
            "lr",
            "n_seeds",
            "accuracy_mean",
            "accuracy_std",
            "macro_f1_mean",
            "macro_f1_std",
        ],
    )
    print("\nTest-set summary (mean over seeds):")
    for row in summary_rows:
        print(
            f"  {row['run']:8s}  acc={row['accuracy_mean']:.4f}±{row['accuracy_std']:.4f}  "
            f"macro-F1={row['macro_f1_mean']:.4f}±{row['macro_f1_std']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 3 training experiments")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument(
        "--lrs",
        type=float,
        nargs="+",
        default=LR_CANDIDATES,
        help="CNN validation sweep candidates",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Skip the sweep and use this learning rate for reported runs",
    )
    parser.add_argument(
        "--skip-sweep",
        action="store_true",
        help="Skip the CNN learning-rate sweep (requires --lr or results/best_lr.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")

    if args.skip_sweep or args.lr is not None:
        if args.lr is not None:
            lr = args.lr
        else:
            payload = json.loads((ROOT / "results" / "best_lr.json").read_text())
            lr = float(payload["best_lr"])
        print(f"Using learning rate {lr:g} without a new sweep")
    else:
        lr = run_lr_sweep(args, device)

    run_reported_experiments(args, device, lr)
    print("\nWrote results/metrics.csv, results/per_class.csv, and figures/curves_*.png")


if __name__ == "__main__":
    main()
