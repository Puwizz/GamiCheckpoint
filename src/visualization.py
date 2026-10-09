"""
visualization.py — Pipeline Visual Inspection & Error Analysis
==============================================================
Module for rendering and saving:
1. Training Input Sanity Check Grid (crops & augmentations as seen by the model)
2. Evaluation Prediction Check Grid (highlighting misclassifications in red vs correct in green)
"""

import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from step_config import STEP_CLASSES, STEP_NAMES


def denormalize_image(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Convert a normalized (3, H, W) float tensor back to (H, W, 3) uint8 numpy RGB image."""
    mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
    std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
    img = tensor.detach().cpu().numpy() * std + mean
    img = np.clip(img, 0.0, 1.0)
    img = (img.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return img


def save_train_samples_grid(dataset, output_path, classes=STEP_CLASSES, samples_per_class=3):
    """
    Generate an organized sanity check grid: len(classes) rows x samples_per_class columns.
    Shows the exact crops, orientations, and augmentations fed into MobileNetV2.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group dataset sample indices by class
    class_indices = {c: [] for c in classes}
    for idx, (_, label_idx) in enumerate(dataset.samples):
        cls_name = dataset.classes[label_idx]
        if cls_name in class_indices:
            class_indices[cls_name].append(idx)

    num_rows = len(classes)
    num_cols = samples_per_class

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3.6, num_rows * 3.4))
    fig.suptitle("Training Input Sanity Check (AutoCrop + Augmentation)", fontsize=14, fontweight="bold", y=0.995)

    axes = np.asarray(axes).reshape(num_rows, num_cols)

    for row_idx, cls_name in enumerate(classes):
        indices = class_indices[cls_name]
        step_desc = STEP_NAMES.get(cls_name, "")
        
        # Pick evenly spaced samples from this class
        if samples_per_class <= 1:
            chosen = [indices[0]] if indices else []
        elif len(indices) >= samples_per_class:
            chosen = [indices[int(i * (len(indices) - 1) / (samples_per_class - 1))] for i in range(samples_per_class)]
        else:
            chosen = indices + [indices[-1]] * (samples_per_class - len(indices)) if indices else []

        for col_idx in range(num_cols):
            ax = axes[row_idx, col_idx]
            if col_idx < len(chosen):
                sample_idx = chosen[col_idx]
                img_tensor, _ = dataset[sample_idx]
                img_rgb = denormalize_image(img_tensor)
                ax.imshow(img_rgb)
                ax.set_title(f"{cls_name}: {step_desc}\n[Sample {col_idx + 1}]", fontsize=9, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No sample", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved training input sanity check to {output_path}")


def render_prediction_grid(misclassified_samples, correct_samples, output_path, classes=STEP_CLASSES, total_images=0, total_errors=0, num_samples=16):
    """
    Render a 4x4 prediction check grid with Error Analysis Priority.
    Misclassified images are shown in RED; Correct predictions are shown in GREEN.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort errors by confidence descending (confident mistakes are most important to inspect)
    sorted_errors = sorted(misclassified_samples, key=lambda x: x.get("confidence", 0), reverse=True)

    # Priority: take errors first, then fill with correct predictions
    selected_items = sorted_errors[:num_samples]
    remaining = num_samples - len(selected_items)
    if remaining > 0 and correct_samples:
        step = max(1, len(correct_samples) // remaining)
        for i in range(0, len(correct_samples), step):
            selected_items.append(correct_samples[i])
            if len(selected_items) >= num_samples:
                break

    grid_size = int(math.ceil(math.sqrt(num_samples)))  # 4x4 for 16
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3.6, grid_size * 3.7))
    error_rate = (total_errors / max(1, total_images)) * 100.0 if total_images else 0.0
    fig.suptitle(
        f"Prediction Inspection: {output_path.stem}\n(Total Images: {total_images} | Errors: {total_errors} [{error_rate:.1f}%])",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, ax in enumerate(axes_flat):
        if idx < len(selected_items):
            item = selected_items[idx]
            img_rgb = denormalize_image(item["tensor"])
            ax.imshow(img_rgb)

            is_error = item["true_idx"] != item["pred_idx"]
            true_name = classes[item["true_idx"]]
            pred_name = classes[item["pred_idx"]]
            conf = item.get("confidence", 0.0)

            if is_error:
                title = f"MISCLASSIFIED\nTrue: {true_name} | Pred: {pred_name}\nConf: {conf:.1f}%"
                color = "red"
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(3.0)
            else:
                title = f"CORRECT: {true_name}\nConf: {conf:.1f}%"
                color = "darkgreen"
                for spine in ax.spines.values():
                    spine.set_edgecolor("green")
                    spine.set_linewidth(1.8)

            ax.set_title(title, fontsize=8.5, fontweight="bold", color=color)
        else:
            ax.text(0.5, 0.5, "-", ha="center", va="center")
            ax.axis("off")
            continue

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[SUCCESS] Saved prediction check ({total_errors} errors found) to {output_path}")
