"""Grad-CAM heatmaps from the last convolutional layer of the CNN.

Do not apply this to the MLP: it has no spatial feature maps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.data import CLASS_NAMES, denormalize


def generate_gradcam(
    model: nn.Module,
    image: torch.Tensor,
    class_idx: int | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, int]:
    """Return a 28x28 Grad-CAM heatmap and the class index used.

    ``image`` is a single example of shape ``(1, 28, 28)`` or ``(1, 1, 28, 28)``.
    Gradients are taken at ``model.conv2``, the last convolutional layer.
    """
    if not hasattr(model, "conv2"):
        raise TypeError("Grad-CAM requires a CNN with a conv2 layer.")

    device = device or next(model.parameters()).device
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    handle_f = model.conv2.register_forward_hook(forward_hook)
    handle_b = model.conv2.register_full_backward_hook(backward_hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(image)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        logits[0, class_idx].backward()
    finally:
        handle_f.remove()
        handle_b.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture conv2 maps.")

    acts = activations[0]
    grads = gradients[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)
    cam = cam.squeeze().detach().cpu().numpy()
    cam = cam - cam.min()
    vmax = cam.max()
    if vmax > 0:
        cam = cam / vmax
    return cam.astype(np.float32), int(class_idx)


def _to_display_image(image: torch.Tensor, mean: float, std: float) -> np.ndarray:
    if image.dim() == 4:
        image = image.squeeze(0)
    shown = denormalize(image.detach().cpu(), mean, std).clamp(0, 1)
    return shown.squeeze().numpy()


def overlay_heatmap(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a grayscale 28x28 image with a jet heatmap."""
    heat = plt.get_cmap("jet")(cam)[..., :3]
    rgb = np.stack([gray, gray, gray], axis=-1)
    return (1.0 - alpha) * rgb + alpha * heat


def to_display_image(image: torch.Tensor, mean: float, std: float) -> np.ndarray:
    return _to_display_image(image, mean, std)


def collect_gradcam_cases(model: nn.Module, loader: DataLoader, device: torch.device) -> list[dict]:
    """Pick correct tops/shoes/bag examples and Shirt ↔ T-shirt errors."""
    wanted = {
        "correct_top": None,
        "correct_shoe": None,
        "correct_bag": None,
        "error_shirt_as_top": None,
        "error_top_as_shirt": None,
    }
    top_ids = {0, 6}
    shoe_ids = {7, 9}
    bag_id = 8
    shirt_id = 6
    tshirt_id = 0

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            for i in range(images.size(0)):
                true = int(labels[i].item())
                pred = int(preds[i].item())
                conf = float(probs[i, pred].item())
                sample = {
                    "image": images[i].detach().cpu(),
                    "true": true,
                    "pred": pred,
                    "conf": conf,
                }
                if (
                    wanted["correct_top"] is None
                    and true in top_ids
                    and pred == true
                ):
                    wanted["correct_top"] = sample
                elif (
                    wanted["correct_shoe"] is None
                    and true in shoe_ids
                    and pred == true
                ):
                    wanted["correct_shoe"] = sample
                elif (
                    wanted["correct_bag"] is None
                    and true == bag_id
                    and pred == true
                ):
                    wanted["correct_bag"] = sample
                elif (
                    wanted["error_shirt_as_top"] is None
                    and true == shirt_id
                    and pred == tshirt_id
                ):
                    wanted["error_shirt_as_top"] = sample
                elif (
                    wanted["error_top_as_shirt"] is None
                    and true == tshirt_id
                    and pred == shirt_id
                ):
                    wanted["error_top_as_shirt"] = sample
                if all(value is not None for value in wanted.values()):
                    return _ordered_cases(wanted)
    missing = [key for key, value in wanted.items() if value is None]
    raise RuntimeError(f"Could not find Grad-CAM cases: {missing}")


def _ordered_cases(wanted: dict) -> list[dict]:
    titles = {
        "correct_top": "Correct top",
        "correct_shoe": "Correct shoe",
        "correct_bag": "Correct bag",
        "error_shirt_as_top": "Error: Shirt → T-shirt/top",
        "error_top_as_shirt": "Error: T-shirt/top → Shirt",
    }
    cases = []
    for key, title in titles.items():
        sample = wanted[key]
        sample["title"] = title
        cases.append(sample)
    return cases


def save_gradcam_grid(
    model: nn.Module,
    test_loader: DataLoader,
    out_path: str | Path,
    mean: float,
    std: float,
    device: torch.device | None = None,
) -> Path:
    """Save overlays for correct predictions and Shirt / T-shirt errors."""
    device = device or next(model.parameters()).device
    cases = collect_gradcam_cases(model, test_loader, device)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6.2))
    for col, case in enumerate(cases):
        cam, used_class = generate_gradcam(
            model, case["image"], class_idx=case["pred"], device=device
        )
        gray = _to_display_image(case["image"], mean, std)
        overlay = overlay_heatmap(gray, cam)
        true_name = CLASS_NAMES[case["true"]]
        pred_name = CLASS_NAMES[case["pred"]]
        subtitle = f"true {true_name}\npred {pred_name} ({case['conf']:.2f})"

        axes[0, col].imshow(gray, cmap="gray")
        axes[0, col].set_title(case["title"], fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(overlay)
        axes[1, col].set_title(subtitle, fontsize=8)
        axes[1, col].axis("off")
        _ = used_class

    axes[0, 0].set_ylabel("Image")
    axes[1, 0].set_ylabel("Grad-CAM")
    fig.suptitle("CNN Grad-CAM on Fashion-MNIST test images (predicted class)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path
