# Fashion-MNIST clothing classification

Individual COMP813 project: compare an MLP and a small CNN on Fashion-MNIST, ablate two geometric augmentations (flip+shift, and a separate crop run), inspect CNN predictions with Grad-CAM, and run an Adebayo-style parameter-randomisation check.

No camera is required. Evaluation uses the official 10,000-image Fashion-MNIST test set.

## Results (3 seeds, official test set)

| Run | Model | Augmentation | Accuracy (%) | Macro-F1 (%) |
|-----|-------|--------------|--------------|--------------|
| A | MLP | off | 88.88 ± 0.15 | 88.83 ± 0.17 |
| B | CNN | off | **92.08 ± 0.23** | **92.05 ± 0.23** |
| C | CNN | horizontal flip + ≤2 px shift | 91.38 ± 0.28 | 91.36 ± 0.32 |
| D | CNN | random crop, pad=4 | 90.13 ± 0.36 | 90.05 ± 0.43 |

Selected CNN learning rate: **3e-4** (validation accuracy 93.28%). Best CNN checkpoint: seed 2, **92.24%** test accuracy.

Ablation answers from these runs (not from papers):

1. The CNN beats the MLP by about 3.2 points under the same training recipe.
2. Flip + shift does **not** raise overall accuracy (92.08% → 91.38%). Isolated crop is worse (90.13%).
3. Shirt remains the hardest class (CNN F1 0.769 vs flip/shift 0.746 vs crop 0.697).
4. Grad-CAM maps collapse under cascading weight randomisation (mean Spearman 0.165 / 0.074 / 0.085 vs the trained map).

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
    data.py          # load, stratified val split, flip_shift / crop
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
    gradcam_sanity.py   # Adebayo parameter randomisation
    demo.py             # 1-epoch video demo
  figures/
  results/
```

## Reproduce

Work from `code/` with the virtual environment activated.

### 1. Preview data

```bash
python scripts/preview_data.py
```

Writes `figures/sample_train.png`, `figures/sample_train_aug.png`, and `figures/sample_train_crop.png`.

### 2. Preview models

```bash
python scripts/preview_models.py
```

### 3. Train

Full four-run experiment (overwrites all seeds; slow):

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiments.py
```

Train **only** the crop ablation, keeping existing MLP / CNN / CNN+aug rows:

```bash
PYTHONUNBUFFERED=1 python -u scripts/run_experiments.py --skip-sweep --lr 0.0003 --runs cnn_crop
```

Do **not** re-run the full command before the video demo: it overwrites the 3-seed results.

### 4. Evaluate the best unaugmented CNN

```bash
python scripts/evaluate_test.py
```

### 5. Grad-CAM

```bash
python scripts/gradcam_cnn.py
python scripts/gradcam_sanity.py
```

The second script does **not** retrain. It randomises the saved CNN from the top down and writes `figures/gradcam_sanity.png` plus `results/gradcam_sanity.csv`.

### 6. Video demo (2–5 min MP4)

```bash
python -u scripts/demo.py
```

`demo.py` does not overwrite `results/metrics.csv`. Timed script: `../VIDEO_DEMO.md`.

## Training recipe

- Official 60,000/10,000 split; 10% stratified validation from train
- Pixels scaled to `[0, 1]`, then normalised with **train-split** mean/std
- Cross-entropy, Adam, batch size 128, max 30 epochs, early stopping (patience 5)
- Two CNN-only augmentations, never on val/test:
  - `flip_shift`: horizontal flip `p=0.5` and a shift of at most 2 pixels
  - `crop`: `RandomCrop(28, padding=4)`
- Grad-CAM on `conv2`; Adebayo check randomises classifier → conv2 → all weights
- Not applied to the MLP

## What not to zip

- `.venv/`
- `data/`
- `results/checkpoints/` and `*.pt`
- `__pycache__/`

Keep `results/*.csv`, `results/*.json`, `results/ablation.md`, and `figures/*.png`.
