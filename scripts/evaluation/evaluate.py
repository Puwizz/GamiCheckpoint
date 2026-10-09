import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root and src to python path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from model import StepModel
from preprocessing import val_transform
from step_config import STEP_CLASSES
from visualization import render_prediction_grid


def parse_args():
    root = ROOT
    parser = argparse.ArgumentParser(description="Evaluate the six-class step detector")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "step6" / "processed")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--model",
        type=Path,
        default=root / "checkpoints" / "origami_step6_best.pth",
    )
    parser.add_argument("--plots-dir", type=Path, default=root / "outputs" / "plots")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4, help="Number of DataLoader workers")
    return parser.parse_args()


def compute_classification_metrics(labels, predictions, target_names):
    n_classes = len(target_names)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for label, pred in zip(labels, predictions):
        if 0 <= label < n_classes and 0 <= pred < n_classes:
            matrix[label, pred] += 1

    total = len(labels)
    correct = sum(1 for label, pred in zip(labels, predictions) if label == pred)
    accuracy = (correct / total) if total > 0 else 0.0

    report = {}
    macro_prec, macro_rec, macro_f1 = 0.0, 0.0, 0.0
    weighted_prec, weighted_rec, weighted_f1 = 0.0, 0.0, 0.0

    for i, name in enumerate(target_names):
        tp = int(matrix[i, i])
        support = int(matrix[i, :].sum())
        pred_count = int(matrix[:, i].sum())
        prec = (tp / pred_count) if pred_count > 0 else 0.0
        rec = (tp / support) if support > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        report[name] = {
            "precision": prec,
            "recall": rec,
            "f1-score": f1,
            "support": support,
        }
        macro_prec += prec
        macro_rec += rec
        macro_f1 += f1
        weighted_prec += prec * support
        weighted_rec += rec * support
        weighted_f1 += f1 * support

    report["accuracy"] = accuracy
    report["macro avg"] = {
        "precision": macro_prec / n_classes if n_classes > 0 else 0.0,
        "recall": macro_rec / n_classes if n_classes > 0 else 0.0,
        "f1-score": macro_f1 / n_classes if n_classes > 0 else 0.0,
        "support": total,
    }
    report["weighted avg"] = {
        "precision": weighted_prec / total if total > 0 else 0.0,
        "recall": weighted_rec / total if total > 0 else 0.0,
        "f1-score": weighted_f1 / total if total > 0 else 0.0,
        "support": total,
    }

    lines = [f"{'':>12} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}\n"]
    for name in target_names:
        r = report[name]
        lines.append(f"{name:>12} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f} {r['support']:>10d}")
    lines.append("")
    lines.append(f"{'accuracy':>12} {'':>10} {'':>10} {accuracy:>10.2f} {total:>10d}")
    r = report["macro avg"]
    lines.append(f"{'macro avg':>12} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f} {r['support']:>10d}")
    r = report["weighted avg"]
    lines.append(f"{'weighted avg':>12} {r['precision']:>10.2f} {r['recall']:>10.2f} {r['f1-score']:>10.2f} {r['support']:>10d}")
    text_report = "\n".join(lines)

    return accuracy, matrix, report, text_report


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.ImageFolder(args.data_dir / args.split, transform=val_transform)
    expected_classes = list(STEP_CLASSES)
    if dataset.classes != expected_classes:
        raise ValueError(f"Expected {expected_classes}, got {dataset.classes}")

    model = StepModel.load_checkpoint(args.model, device=device, expected_classes=expected_classes)
    predictions = []
    labels = []
    misclassified_by_class = defaultdict(list)
    correct_by_class = defaultdict(list)
    max_samples_per_class = 6

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    with torch.no_grad():
        progress = tqdm(loader, desc=f"Evaluating {args.split}", leave=False)
        for inputs, targets in progress:
            outputs = model(inputs.to(device))
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, dim=1)
            preds_list = preds.cpu().tolist()
            targets_list = targets.tolist()
            predictions.extend(preds_list)
            labels.extend(targets_list)

            for b in range(inputs.size(0)):
                true_c = targets_list[b]
                pred_c = preds_list[b]
                is_err = pred_c != true_c
                sample_info = {
                    "tensor": inputs[b].cpu(),
                    "true_idx": true_c,
                    "pred_idx": pred_c,
                    "confidence": confs[b].item() * 100.0,
                }
                if is_err and len(misclassified_by_class[true_c]) < max_samples_per_class:
                    misclassified_by_class[true_c].append(sample_info)
                elif not is_err and len(correct_by_class[true_c]) < max_samples_per_class:
                    correct_by_class[true_c].append(sample_info)

    misclassified_samples = [s for samples in misclassified_by_class.values() for s in samples]
    correct_samples = [s for samples in correct_by_class.values() for s in samples]

    accuracy, matrix, report, text_report = compute_classification_metrics(
        labels, predictions, expected_classes
    )
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.plots_dir / f"step6_{args.split}_report.json"
    report_path.write_text(
        json.dumps(
            {
                "accuracy": accuracy,
                "split": args.split,
                "count": len(dataset),
                "classes": expected_classes,
                "model": str(args.model.resolve()),
                "data_dir": str(args.data_dir.resolve()),
                "confusion_matrix": matrix.tolist(),
                "report": report,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=expected_classes,
        yticklabels=expected_classes,
        title=f"Six-class step detector ({args.split})",
        ylabel="Actual step",
        xlabel="Predicted step",
    )
    thresh = matrix.max() / 2.0 if matrix.max() > 0 else 1.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )
    fig.tight_layout()
    matrix_path = args.plots_dir / f"step6_{args.split}_confusion_matrix.png"
    fig.savefig(matrix_path, dpi=200)
    plt.close(fig)

    pred_grid_path = args.plots_dir / f"step6_{args.split}_prediction_samples.png"
    total_errors = sum(1 for p, l in zip(predictions, labels) if p != l)
    render_prediction_grid(
        misclassified_samples,
        correct_samples,
        pred_grid_path,
        classes=expected_classes,
        total_images=len(dataset),
        total_errors=total_errors,
        num_samples=16,
    )

    print(f"\n[{args.split.upper()} EVALUATION COMPLETED]")
    print(f"Accuracy: {accuracy:.2%} ({len(dataset)} images)")
    print(text_report)

    critical_alerts = []
    for cls_name in expected_classes:
        cls_recall = report.get(cls_name, {}).get("recall", 1.0)
        if cls_recall < 0.30:
            critical_alerts.append((cls_name, cls_recall))

    if critical_alerts:
        print("!" * 62)
        print("  [CRITICAL ALERT] Low-Performing Steps (Recall < 30%):")
        for c_name, c_rec in critical_alerts:
            print(f"   -> {c_name}: Recall = {c_rec:.1%} (Model severely struggles here)")
        print("!" * 62 + "\n")

    print(f"Report: {report_path}")
    print(f"Confusion matrix: {matrix_path}")
    print(f"Prediction samples: {pred_grid_path}")


if __name__ == "__main__":
    main()
