# Ablation answers from this project's code (Step 6)

All numbers are from `metrics.csv`, `per_class_summary.csv`, and `confusion_cnn.csv`. They are **not** copied from papers.

## 1. Does the CNN beat the MLP?

**Yes.** On the official 10,000-image test set, mean over 3 seeds:

| Model | Accuracy (%) | Macro-F1 (%) |
|-------|--------------|--------------|
| MLP (no aug) | 88.88 ± 0.15 | 88.83 ± 0.17 |
| CNN (no aug) | **92.08 ± 0.23** | **92.05 ± 0.23** |

The CNN is about **3.2 percentage points** more accurate. Training used the same split, Adam, batch size 128, early stopping, and the same learning rate (3e-4). The gain is therefore from convolution, not from a different optimiser.

## 2. Does augmentation raise overall accuracy?

**No, not in this setup.** Horizontal flip (p=0.5) plus a shift of at most 2 pixels **lowered** mean test accuracy:

| Model | Accuracy (%) | Macro-F1 (%) |
|-------|--------------|--------------|
| CNN (no aug) | **92.08 ± 0.23** | **92.05 ± 0.23** |
| CNN + flip/shift | 91.38 ± 0.28 | 91.36 ± 0.32 |

That is a drop of about **0.7 points**. The 1-epoch demo run is not used here; these figures are from the 30-epoch, 3-seed experiments.

## 3. Does augmentation reduce Shirt ↔ T-shirt errors, or only help easy classes?

**It does not reduce the hard-class errors.** Mean test F1 over 3 seeds:

| Class | MLP | CNN | CNN+aug |
|-------|-----|-----|---------|
| T-shirt/top | 0.846 | **0.872** | 0.870 |
| Pullover | 0.805 | **0.878** | 0.871 |
| Coat | 0.816 | **0.876** | 0.861 |
| Shirt | 0.699 | **0.769** | 0.746 |
| Trouser (easy) | 0.979 | 0.988 | 0.990 |
| Bag (easy) | 0.973 | 0.984 | 0.985 |

Shirt remains the weakest class. Augmentation slightly **hurts** Shirt (0.769 → 0.746) and Coat, while easy classes stay high either way.

On the best CNN (no aug, seed 2), the confusion matrix still shows:

- Shirt → T-shirt/top: **95**
- T-shirt/top → Shirt: **85**

So remaining error is overlapping upper-body labels, not a lack of flip/shift.
