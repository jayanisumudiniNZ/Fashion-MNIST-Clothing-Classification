# Fashion-MNIST clothing classification

Individual COMP813 project: compare an MLP and a small CNN on Fashion-MNIST, ablate simple geometric augmentation, and inspect CNN predictions with Grad-CAM.

No camera is required. Evaluation uses the official 10,000-image Fashion-MNIST test set.

## Results (3 seeds, official test set)

| Run | Model | Augmentation | Accuracy (%) | Macro-F1 (%) |
|-----|-------|--------------|--------------|--------------|
| A | MLP | off | 88.88 ± 0.15 | 88.83 ± 0.17 |
| B | CNN | off | **92.08 ± 0.23** | **92.05 ± 0.23** |
| C | CNN | horizontal flip + ≤2 px shift | 91.38 ± 0.28 | 91.36 ± 0.32 |

Selected CNN learning rate: **3e-4** (validation accuracy 93.28%). Best CNN checkpoint: seed 2, **92.24%** test accuracy.

Ablation answers from these runs (not from papers):

1. The CNN beats the MLP by about 3.2 points under the same training recipe.
2. Flip + shift does **not** raise overall accuracy (92.08% → 91.38%).
3. Shirt remains the hardest class (CNN F1 0.769 vs CNN+aug 0.746). Shirt → T-shirt/top: 95; T-shirt/top → Shirt: 85.

Write-up: `results/ablation.md`. Report figures live in `figures/` and are copied into `../cvpr2017AuthorKit/latex/figures/`.

## Setup

Python 3.10+ recommended. On Apple Silicon the training code uses MPS when available.

```bash
cd code
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

```text
code/
  requirements.txt
  README.md
  src/
    data.py          # load, stratified val split, augment
    models.py        # MLP and CNN
    train.py         # Adam, early stopping
    evaluate.py      # accuracy, F1, confusion matrix
    gradcam.py       # conv2 heatmaps
  scripts/
    preview_data.py
    preview_models.py
    run_experiments.py
    evaluate_test.py
    gradcam_cnn.py
    demo.py          # 1-epoch video demo
  figures/           # PNG plots for the report
  results/           # csv / json tables (not the large .pt files)
```

## Reproduce

Work from `code/` with the virtual environment activated. Fashion-MNIST downloads into `code/data/` on first run.

### 1. Preview data

```bash
python scripts/preview_data.py
```

Prints split sizes and training-set mean/std, and writes `figures/sample_train.png` and `figures/sample_train_aug.png`.

### 2. Preview models

```bash
python scripts/preview_models.py
```

Prints parameter counts (MLP 235,146; CNN 421,642) and checks that both models map `(N, 1, 28, 28)` to 10 logits.

### 3. Train (learning-rate sweep + 3 seeds)

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiments.py
```

This:

1. Sweeps CNN learning rates `1e-3`, `3e-4`, `1e-4` on the **validation** set (no augmentation)
2. Reuses the best LR for MLP, CNN, and CNN+aug, each with 3 seeds
3. Writes `results/metrics.csv`, `results/runs.csv`, `results/per_class.csv`, `results/lr_sweep.csv`, and `figures/curves_*.png`

Skip the sweep if you already have a chosen LR:

```bash
python scripts/run_experiments.py --skip-sweep --lr 0.0003
```

Do **not** re-run this before the video demo: it overwrites the 3-seed results and takes a long time.

### 4. Evaluate the best CNN

```bash
python scripts/evaluate_test.py
```

Reloads the best CNN checkpoint, scores the 10,000-image test set, and writes:

- `figures/confusion_cnn.png`
- `results/confusion_cnn.csv`
- `results/per_class_summary.csv`

### 5. Grad-CAM

```bash
python scripts/gradcam_cnn.py
```

Writes `figures/gradcam_grid.png` for the best CNN: correct top/shoe/bag examples and Shirt ↔ T-shirt/top errors.

### 6. Video demo (2–5 min MP4)

```bash
python -u scripts/demo.py
```

Runs a **1-epoch** training pass (allowed instead of full training), then reloads the saved CNN, reprints test metrics, and opens the confusion-matrix and Grad-CAM figures. Timed script: `../VIDEO_DEMO.md`.

`demo.py` does not overwrite `results/metrics.csv`.

## Training recipe (all three runs)

- Official 60,000/10,000 split; 10% stratified validation from train
- Pixels scaled to `[0, 1]`, then normalised with **train-split** mean/std
- Cross-entropy, Adam, batch size 128, max 30 epochs, early stopping (patience 5) on validation loss
- Augmentation on CNN training only: horizontal flip `p=0.5` and a shift of at most 2 pixels; never on val/test
- Grad-CAM on the last convolutional layer (`conv2`); not applied to the MLP

## What not to zip

Do **not** include these in the submission archive:

- `.venv/`
- `data/` (Fashion-MNIST download cache)
- `results/checkpoints/` and `*.pt` files
- `__pycache__/`

Keep `results/*.csv`, `results/*.json`, `results/ablation.md`, and `figures/*.png`.
