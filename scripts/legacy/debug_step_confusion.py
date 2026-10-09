import os
import sys

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(os.path.join(ROOT, "src"))

from model import create_model
from preprocessing import val_transform


DATA_DIR = os.path.join(ROOT, "data", "processed", "val")
MODEL_PATH = os.path.join(ROOT, "checkpoints", "origami_model_final.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_split():
    base = os.path.join(ROOT, "data", "processed")
    for split in ["train", "val"]:
        split_dir = os.path.join(base, split)
        print(f"[{split}]")
        for class_name in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            count = sum(
                1 for name in os.listdir(class_dir)
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            )
            print(f"  {class_name}: {count}")
        print()


def load_model(classes):
    model = create_model(
        num_classes=len(classes),
        pretrained=False,
        freeze_backbone=False,
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model


def evaluate_confusion():
    dataset = datasets.ImageFolder(DATA_DIR, transform=val_transform)
    classes = dataset.classes
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    classes = checkpoint.get("classes", classes)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = load_model(classes)

    labels = []
    preds = []
    paths = [path for path, _ in dataset.samples]

    with torch.no_grad():
        for inputs, target in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            predicted = outputs.argmax(1).cpu().tolist()
            preds.extend(predicted)
            labels.extend(target.tolist())

    cm = confusion_matrix(labels, preds)

    print("Classes:")
    for index, class_name in enumerate(classes):
        print(f"  {index}: {class_name}")
    print()

    print("Per-class recall / top confusion:")
    for index, class_name in enumerate(classes):
        total = int(cm[index].sum())
        correct = int(cm[index, index])
        recall = (correct / total * 100) if total else 0.0
        row = cm[index].copy()
        row[index] = 0
        if row.sum() > 0:
            confused_index = int(row.argmax())
            confused_name = classes[confused_index]
            confused_count = int(row[confused_index])
        else:
            confused_name = "-"
            confused_count = 0
        print(
            f"  {class_name}: recall={recall:.1f}% ({correct}/{total}), "
            f"top_confusion={confused_name} ({confused_count})"
        )
    print()

    name_to_index = {name: idx for idx, name in enumerate(classes)}
    targets = [
        "Chk1_Positive",
        "Chk2_Positive",
        "Chk1_Negative",
        "Chk2_Negative",
    ]
    print("Focused rows:")
    for target_name in targets:
        if target_name not in name_to_index:
            continue
        index = name_to_index[target_name]
        top_indices = np.argsort(cm[index])[::-1][:4]
        row_summary = [(classes[i], int(cm[index, i])) for i in top_indices]
        print(f"  {target_name}: {row_summary}")
    print()

    for target_name in ["Chk2_Positive", "Chk1_Positive"]:
        if target_name not in name_to_index:
            continue
        target_index = name_to_index[target_name]
        print(f"Misclassified samples for {target_name}:")
        shown = 0
        for path, label, pred in zip(paths, labels, preds):
            if label == target_index and pred != label:
                print(f"  {os.path.basename(path)} -> {classes[pred]}")
                shown += 1
                if shown >= 10:
                    break
        if shown == 0:
            print("  none")
        print()


if __name__ == "__main__":
    count_split()
    evaluate_confusion()
