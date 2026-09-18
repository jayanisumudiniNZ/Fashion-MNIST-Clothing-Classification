# Ablation answers from this project's code (Step 6)

All numbers are from `metrics.csv`, `per_class_summary.csv`, `confusion_cnn.csv`, and `gradcam_sanity.csv`. They are **not** copied from papers.

## 1. Does the CNN beat the MLP?

**Yes.** On the official 10,000-image test set, mean over 3 seeds:

| Model | Accuracy (%) | Macro-F1 (%) |
|-------|--------------|--------------|
| MLP (no aug) | 88.88 ± 0.15 | 88.83 ± 0.17 |
| CNN (no aug) | **92.08 ± 0.23** | **92.05 ± 0.23** |

The CNN is about **3.2 percentage points** more accurate. Training used the same split, Adam, batch size 128, early stopping, and the same learning rate (3e-4).

## 2. Does augmentation raise overall accuracy?

**No.** Two isolated CNN policies both lowered accuracy:

| Model | Accuracy (%) | Macro-F1 (%) |
|-------|--------------|--------------|
| CNN (no aug) | **92.08 ± 0.23** | **92.05 ± 0.23** |
| CNN + flip/shift | 91.38 ± 0.28 | 91.36 ± 0.32 |
| CNN + crop (pad 4) | 90.13 ± 0.36 | 90.05 ± 0.43 |

Crop is a **fourth run**, not a replacement of flip/shift.

## 3. Does either policy reduce Shirt ↔ T-shirt errors?

**No.** Mean Shirt F1: MLP 0.699, CNN **0.769**, flip/shift 0.746, crop 0.697. Crop returns Shirt to about the MLP level.

Best CNN (no aug, seed 2): Shirt → T-shirt **95**, T-shirt → Shirt **85**.

## 4. Do Grad-CAM maps depend on trained weights (Adebayo-style)?

**Yes, they collapse when weights are randomised** (same five images, trained predicted class):

| Randomisation | Mean Spearman vs trained map |
|---------------|------------------------------|
| Classifier only | 0.165 |
| conv2 + classifier | 0.074 |
| All weights | 0.085 |

No extra training. Label-randomisation (train on shuffled labels) was not run.
