"""Evaluate the best CNN on the official 10,000-image test set."""

from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.data import CLASS_NAMES, get_dataloaders
from src.evaluate import evaluate_model, pair_confusion, save_confusion_matrix
from src.models import CNN
from src.train import get_device

FOCUS_CLASSES = ["T-shirt/top", "Pullover", "Coat", "Shirt"]
RUNS_CSV = ROOT / "results" / "runs.csv"
PER_CLASS_CSV = ROOT / "results" / "per_class.csv"


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _best_cnn_row(rows: list[dict]) -> dict:
    cnn_rows = [row for row in rows if row["run"] == "cnn"]
    if not cnn_rows:
        raise FileNotFoundError("No CNN rows in results/runs.csv. Run Step 3 first.")
    return max(cnn_rows, key=lambda row: float(row["test_accuracy"]))


def _load_cnn(checkpoint: Path, device: torch.device) -> CNN:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CNN()
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarise_per_class(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["run"], row["class_name"])].append(row)

    summary = []
    for (run, class_name), items in grouped.items():
        f1 = [float(item["f1"]) for item in items]
        precision = [float(item["precision"]) for item in items]
        recall = [float(item["recall"]) for item in items]
        summary.append(
            {
                "run": run,
                "class_id": items[0]["class_id"],
                "class_name": class_name,
                "n_seeds": len(items),
                "f1_mean": statistics.mean(f1),
                "f1_std": statistics.stdev(f1) if len(f1) > 1 else 0.0,
                "precision_mean": statistics.mean(precision),
                "recall_mean": statistics.mean(recall),
            }
        )
    summary.sort(key=lambda row: (row["run"], int(row["class_id"])))
    return summary


def main() -> None:
    device = get_device()
    run_rows = _read_csv(RUNS_CSV)
    best = _best_cnn_row(run_rows)
    checkpoint = ROOT / best["checkpoint"]
    print(f"Device: {device}")
    print(
        f"Best CNN: {best['run']} seed={best['seed']}  "
        f"test acc={float(best['test_accuracy']):.4f}  "
        f"macro-F1={float(best['test_macro_f1']):.4f}"
    )
    print(f"Checkpoint: {checkpoint}")

    splits = get_dataloaders(batch_size=128, augment=False, seed=42)
    model = _load_cnn(checkpoint, device)
    metrics = evaluate_model(model, splits.test_loader, device)
    print(
        f"Reloaded test accuracy={metrics['accuracy']:.4f}  "
        f"macro-F1={metrics['macro_f1']:.4f}"
    )

    cm_path = ROOT / "figures" / "confusion_cnn.png"
    cm = save_confusion_matrix(
        metrics["y_true"],
        metrics["y_pred"],
        cm_path,
        title=f"CNN (no aug, seed {best['seed']}) on 10,000 test images",
    )
    np.savetxt(
        ROOT / "results" / "confusion_cnn.csv",
        cm,
        delimiter=",",
        fmt="%d",
        header=",".join(CLASS_NAMES),
        comments="",
    )
    print(f"Saved {cm_path}")

    shirt_as_top = pair_confusion(cm, "Shirt", "T-shirt/top")
    top_as_shirt = pair_confusion(cm, "T-shirt/top", "Shirt")
    shirt_as_coat = pair_confusion(cm, "Shirt", "Coat")
    coat_as_shirt = pair_confusion(cm, "Coat", "Shirt")
    pullover_as_coat = pair_confusion(cm, "Pullover", "Coat")
    coat_as_pullover = pair_confusion(cm, "Coat", "Pullover")
    print("Off-diagonal confusions (true → predicted):")
    print(f"  Shirt → T-shirt/top: {shirt_as_top}")
    print(f"  T-shirt/top → Shirt: {top_as_shirt}")
    print(f"  Shirt → Coat: {shirt_as_coat}")
    print(f"  Coat → Shirt: {coat_as_shirt}")
    print(f"  Pullover → Coat: {pullover_as_coat}")
    print(f"  Coat → Pullover: {coat_as_pullover}")

    per_class_rows = _read_csv(PER_CLASS_CSV)
    summary = _summarise_per_class(per_class_rows)
    _write_csv(
        ROOT / "results" / "per_class_summary.csv",
        summary,
        [
            "run",
            "class_id",
            "class_name",
            "n_seeds",
            "f1_mean",
            "f1_std",
            "precision_mean",
            "recall_mean",
        ],
    )

    print("\nMean test F1 over 3 seeds (focus classes):")
    print(f"{'class':<12}  {'mlp':>8}  {'cnn':>8}  {'cnn_aug':>8}  {'cnn_crop':>8}")
    for name in FOCUS_CLASSES:
        values = {
            row["run"]: row["f1_mean"]
            for row in summary
            if row["class_name"] == name
        }
        crop = values.get("cnn_crop")
        crop_s = f"{crop:.4f}" if crop is not None else "   n/a"
        print(
            f"{name:<12}  {values['mlp']:.4f}  {values['cnn']:.4f}  "
            f"{values['cnn_aug']:.4f}  {crop_s}"
        )


if __name__ == "__main__":
    main()
