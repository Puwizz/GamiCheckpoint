import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from model import create_model
from preprocessing import train_transform, val_transform
from step_config import STEP_CLASSES
from visualization import save_train_samples_grid


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def step_scheduler(scheduler, val_loss=None):
    """Step scheduler cleanly without branching on scheduler type in training loop."""
    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    running_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        progress = tqdm(loader, desc="Training" if training else "Validation", leave=False)
        for images, labels in progress:
            images = images.to(device)
            labels = labels.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += predictions.eq(labels).sum().item()
            progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100 * correct / total:.1f}%")

    return running_loss / total, 100 * correct / total




def plot_history(history, output_path):
    fig, (loss_axis, accuracy_axis) = plt.subplots(1, 2, figsize=(12, 4))
    loss_axis.plot(history["train_loss"], label="Train")
    loss_axis.plot(history["val_loss"], label="Validation")
    loss_axis.set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    loss_axis.legend()
    loss_axis.grid(True, alpha=0.3)
    accuracy_axis.plot(history["train_acc"], label="Train")
    accuracy_axis.plot(history["val_acc"], label="Validation")
    accuracy_axis.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    accuracy_axis.legend()
    accuracy_axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def load_domain_backbone(model, checkpoint_path):
    """Reuse origami feature weights from the 12-class baseline model."""
    if checkpoint_path is None:
        return False
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        print(f"[WARN] Baseline checkpoint not found: {checkpoint_path}")
        return False

    result = model.load_backbone_weights(checkpoint_path)
    if result.unexpected_keys:
        raise ValueError(f"Unexpected baseline keys: {result.unexpected_keys}")
    print(f"Loaded origami backbone tensors from {checkpoint_path.name}")
    return True


def parse_args():
    root = ROOT
    parser = argparse.ArgumentParser(description="Train the six-class step detector")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "step6" / "processed")
    parser.add_argument("--checkpoints-dir", type=Path, default=root / "checkpoints")
    parser.add_argument("--plots-dir", type=Path, default=root / "outputs" / "plots")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization to prevent overfitting")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fine-tune", action="store_true", help="Train the full backbone")
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["cosine", "plateau"],
        default="cosine",
        help="Learning rate scheduler (cosine annealing or reduce on plateau)",
    )
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        default=root / "checkpoints" / "origami_model_best.pth",
        help="Optional 12-class checkpoint used only to initialize MobileNet features",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError("Step dataset missing. Run: python scripts/prepare_dataset.py")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    expected_classes = list(STEP_CLASSES)
    if train_dataset.classes != expected_classes or val_dataset.classes != expected_classes:
        raise ValueError(
            f"Expected {expected_classes}, got train={train_dataset.classes}, "
            f"val={val_dataset.classes}"
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    args.plots_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 58)
    print("GamiSenpai six-class step training")
    print(f"Device={device} train={len(train_dataset)} val={len(val_dataset)}")
    print(f"Classes={train_dataset.classes}")
    print("=" * 58)

    sanity_grid_path = args.plots_dir / "train_batch_sample.png"
    save_train_samples_grid(train_dataset, sanity_grid_path, classes=expected_classes, samples_per_class=3)

    model = create_model(
        num_classes=len(expected_classes),
        pretrained=not args.backbone_checkpoint.is_file(),
        freeze_backbone=not args.fine_tune,
    )
    load_domain_backbone(model, args.backbone_checkpoint)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.02,
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = -1.0
    best_val_loss = float("inf")
    best_path = args.checkpoints_dir / "origami_step6_best.pth"

    for epoch in range(1, args.epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        step_scheduler(scheduler, val_loss)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        print(
            f"Epoch {epoch:02d}/{args.epochs} (lr={current_lr:.6f}): "
            f"train loss={train_loss:.4f} acc={train_acc:.1f}% | "
            f"val loss={val_loss:.4f} acc={val_acc:.1f}%"
        )
        is_better = val_acc > best_val_acc or (
            val_acc == best_val_acc and val_loss < best_val_loss
        )
        if is_better:
            best_val_acc = val_acc
            best_val_loss = val_loss
            model.save_checkpoint(
                best_path,
                classes=expected_classes,
                epoch=epoch,
                metadata={"val_acc": val_acc, "history": history},
            )
            print(f"  Saved best checkpoint: {best_path}")

    final_path = args.checkpoints_dir / "origami_step6_final.pth"
    model.save_checkpoint(
        final_path,
        classes=expected_classes,
        epoch=args.epochs,
        metadata={"val_acc": history["val_acc"][-1], "history": history},
    )
    plot_history(history, args.plots_dir / "step6_training_history.png")
    metrics_path = args.plots_dir / "step6_training_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "history": history,
                "training_config": {
                    "data_dir": str(args.data_dir.resolve()),
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "seed": args.seed,
                    "fine_tune": args.fine_tune,
                    "scheduler": args.scheduler,
                    "backbone_checkpoint": str(args.backbone_checkpoint.resolve()),
                    "train_count": len(train_dataset),
                    "val_count": len(val_dataset),
                    "classes": expected_classes,
                },
                "best_model": str(best_path.resolve()),
                "final_model": str(final_path.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Training complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()
