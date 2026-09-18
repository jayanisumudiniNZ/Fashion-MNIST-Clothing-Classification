"""Regenerate every report figure from saved CSVs, histories, and checkpoints.

Does not retrain the 3-seed experiments. Requires:
  results/metrics.csv, results/runs.csv, results/per_class.csv,
  results/lr_sweep.csv, results/histories/*.json, and CNN checkpoints.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import get_dataloaders, plot_training_samples
from src.figures import (
    REPORT_FIGURES,
    copy_report_figures,
    load_history,
    plot_lr_sweep_from_csv,
    plot_method_pipeline,
)
from src.train import plot_history

CURVE_RUNS = [
    (run, seed)
    for run in ("mlp", "cnn", "cnn_aug", "cnn_crop")
    for seed in (0, 1, 2)
]


def _run_script(name: str) -> None:
    script = ROOT / "scripts" / name
    print(f"\n=== {name} ===")
    runpy.run_path(str(script), run_name="__main__")


def _samples() -> None:
    print("\n=== sample grids ===")
    figures = ROOT / "figures"
    plain = get_dataloaders(batch_size=128, augment=False, seed=42)
    plot_training_samples(plain, figures / "sample_train.png")
    flip = get_dataloaders(batch_size=128, augment=True, aug_kind="flip_shift", seed=42)
    plot_training_samples(flip, figures / "sample_train_aug.png")
    crop = get_dataloaders(batch_size=128, augment=True, aug_kind="crop", seed=42)
    plot_training_samples(crop, figures / "sample_train_crop.png")
    print("  saved sample_train.png, sample_train_aug.png, sample_train_crop.png")


def _pipeline_and_tables() -> None:
    print("\n=== method pipeline and LR sweep ===")
    plot_method_pipeline(ROOT / "figures" / "method_pipeline.png")
    print("  saved method_pipeline.png")
    plot_lr_sweep_from_csv(
        ROOT / "results" / "lr_sweep.csv",
        ROOT / "figures" / "lr_sweep.png",
    )
    print("  saved lr_sweep.png")


def _curves() -> None:
    print("\n=== training curves from results/histories ===")
    for run, seed in CURVE_RUNS:
        history = load_history(run, seed)
        out = ROOT / "figures" / f"curves_{run}_seed{seed}.png"
        plot_history(history, out, title=f"{run} seed={seed} lr=0.0003")
        print(f"  saved {out.name}")


def main() -> None:
    _samples()
    _pipeline_and_tables()
    _curves()
    _run_script("evaluate_test.py")
    _run_script("gradcam_cnn.py")
    _run_script("gradcam_sanity.py")
    copied = copy_report_figures()
    print(f"\nCopied {len(copied)} figures to {ROOT.parent / 'cvpr2017AuthorKit' / 'latex' / 'figures'}")
    missing = [name for name in REPORT_FIGURES if not (ROOT / "figures" / name).exists()]
    if missing:
        raise SystemExit(f"Missing report figures: {missing}")
    print("All report figures were written.")


if __name__ == "__main__":
    main()
