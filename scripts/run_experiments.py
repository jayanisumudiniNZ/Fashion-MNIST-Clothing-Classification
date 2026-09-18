"""CNN learning-rate sweep, then four reported runs.

The four systems are trained separately so architecture (MLP vs CNN) is not
mixed with augmentation, and so crop is a fourth isolated CNN run rather
than a replacement of flip+shift.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import CLASS_NAMES, get_dataloaders
from src.evaluate import evaluate_model
from src.figures import plot_lr_sweep, save_history
from src.models import CNN, MLP
from src.train import get_device, plot_history, train_model

LR_CANDIDATES = [1e-3, 3e-4, 1e-4]
SEEDS = [0, 1, 2]
SPLIT_SEED = 42
SWEEP_SEED = 0

RUNS = [
    {"name": "mlp", "model": "mlp", "augment": False, "aug_kind": "flip_shift"},
    {"name": "cnn", "model": "cnn", "augment": False, "aug_kind": "flip_shift"},
    {"name": "cnn_aug", "model": "cnn", "augment": True, "aug_kind": "flip_shift"},
    {"name": "cnn_crop", "model": "cnn", "augment": True, "aug_kind": "crop"},
]


def _build_model(name: str):
    if name == "mlp":
        return MLP()
    if name == "cnn":
        return CNN()
    raise ValueError(f"Unknown model: {name}")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _merge_by_run(existing: list[dict], new_rows: list[dict]) -> list[dict]:
    trained = {row["run"] for row in new_rows}
    kept = [row for row in existing if row["run"] not in trained]
    merged = kept + new_rows
    merged.sort(key=lambda row: (row["run"], int(row.get("seed", 0))))
    return merged


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_lr_sweep(rows: list[dict], out_path: Path) -> None:
    plot_lr_sweep(rows, out_path)


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
    selected = [spec for spec in RUNS if spec["name"] in args.runs]
    if not selected:
        raise ValueError(f"No matching runs for {args.runs}. Choose from {[s['name'] for s in RUNS]}")
    print(f"\n=== Reported runs with lr={lr:g}: {', '.join(s['name'] for s in selected)} ===")
    run_rows = []
    per_class_rows = []
    ckpt_dir = ROOT / "results" / "checkpoints"

    for spec in selected:
        for seed in args.seeds:
            print(
                f"\nRun {spec['name']}  seed={seed}  "
                f"augment={spec['augment']}  aug_kind={spec['aug_kind']}"
            )
            splits = get_dataloaders(
                batch_size=args.batch_size,
                augment=spec["augment"],
                aug_kind=spec["aug_kind"],
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
            save_history(result["history"], spec["name"], seed)
            run_rows.append(
                {
                    "run": spec["name"],
                    "model": spec["model"],
                    "augment": spec["augment"],
                    "aug_kind": spec["aug_kind"] if spec["augment"] else "none",
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

    run_fields = [
        "run",
        "model",
        "augment",
        "aug_kind",
        "seed",
        "lr",
        "best_epoch",
        "best_val_loss",
        "best_val_acc",
        "test_accuracy",
        "test_macro_f1",
        "checkpoint",
    ]
    existing_runs = _read_csv(ROOT / "results" / "runs.csv")
    if existing_runs and set(args.runs) != {spec["name"] for spec in RUNS}:
        for row in existing_runs:
            row.setdefault("aug_kind", "none" if row.get("augment") in {"False", "false", False} else "flip_shift")
        run_rows = _merge_by_run(existing_runs, run_rows)
    _write_csv(ROOT / "results" / "runs.csv", run_rows, run_fields)

    existing_pc = _read_csv(ROOT / "results" / "per_class.csv")
    if existing_pc and set(args.runs) != {spec["name"] for spec in RUNS}:
        per_class_rows = _merge_by_run(existing_pc, per_class_rows)
    _write_csv(
        ROOT / "results" / "per_class.csv",
        per_class_rows,
        ["run", "seed", "class_id", "class_name", "precision", "recall", "f1", "support"],
    )

    summary_rows = []
    run_names = sorted({row["run"] for row in run_rows})
    for name in run_names:
        subset = [row for row in run_rows if row["run"] == name]
        acc = [float(row["test_accuracy"]) for row in subset]
        f1 = [float(row["test_macro_f1"]) for row in subset]
        spec = next((item for item in RUNS if item["name"] == name), None)
        augment = subset[0]["augment"]
        if spec is not None:
            augment = spec["augment"]
        if str(augment).lower() == "false":
            aug_kind = "none"
        elif spec is not None:
            aug_kind = spec["aug_kind"]
        else:
            aug_kind = subset[0].get("aug_kind", "none")
        summary_rows.append(
            {
                "run": name,
                "model": spec["model"] if spec is not None else subset[0]["model"],
                "augment": augment,
                "aug_kind": aug_kind,
                "lr": lr,
                "n_seeds": len(subset),
                "accuracy_mean": statistics.mean(acc),
                "accuracy_std": statistics.stdev(acc) if len(acc) > 1 else 0.0,
                "macro_f1_mean": statistics.mean(f1),
                "macro_f1_std": statistics.stdev(f1) if len(f1) > 1 else 0.0,
            }
        )
    order = {spec["name"]: i for i, spec in enumerate(RUNS)}
    summary_rows.sort(key=lambda row: order.get(row["run"], 99))
    _write_csv(
        ROOT / "results" / "metrics.csv",
        summary_rows,
        [
            "run",
            "model",
            "augment",
            "aug_kind",
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
        "--runs",
        nargs="+",
        default=[spec["name"] for spec in RUNS],
        choices=[spec["name"] for spec in RUNS],
        help="Which reported runs to train. Default is all four. "
        "Use --runs cnn_crop to add the crop ablation without retraining the others.",
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
