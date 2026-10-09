"""
utils.py — Utility Functions
=============================
ฟังก์ชันช่วยเหลือทั่วไป
"""

import os
import shutil
import random
import stat
import sys
import time
from pathlib import Path
from typing import Union


def safe_rmtree(directory: Union[str, Path]) -> None:
    """Safely remove directory tree on Windows, handling ReadOnly and OneDrive attributes."""
    dir_path = Path(directory)
    if not dir_path.exists():
        return

    def _remove_readonly(func, path, excinfo):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            time.sleep(0.1)
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

    if sys.version_info >= (3, 12):
        def _on_exc(func, path, exc):
            _remove_readonly(func, path, None)
        shutil.rmtree(dir_path, onexc=_on_exc)
    else:
        shutil.rmtree(dir_path, onerror=_remove_readonly)


def split_dataset(source_dir, output_dir, train_ratio=0.8, seed=42):
    """
    แบ่ง dataset เป็น train/val sets
    
    Args:
        source_dir: โฟลเดอร์ข้อมูลดิบ (จาก data_collection.py)
        output_dir: โฟลเดอร์เก็บข้อมูลที่แบ่งแล้ว
        train_ratio: สัดส่วนของ training set (0-1)
        seed: random seed สำหรับ reproducibility
    """
    random.seed(seed)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    print(f"[INFO] Splitting dataset: {source_dir}")
    print(f"  Train ratio: {train_ratio:.0%}")

    total_train = 0
    total_val = 0

    # วนทุก class (subdirectory)
    for class_name in sorted(os.listdir(source_dir)):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # หารูปทั้งหมดใน class
        images = [f for f in os.listdir(class_path)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        random.shuffle(images)

        # แบ่ง train/val
        if train_ratio > 0.0 and len(images) > 0:
            split_idx = max(1, int(len(images) * train_ratio))
        else:
            split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # คัดลอกไฟล์
        for img in train_images:
            dest_dir = os.path.join(train_dir, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(os.path.join(class_path, img), os.path.join(dest_dir, img))

        for img in val_images:
            dest_dir = os.path.join(val_dir, class_name)
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(os.path.join(class_path, img), os.path.join(dest_dir, img))

        total_train += len(train_images)
        total_val += len(val_images)

        print(f"  {class_name}: {len(train_images)} train / {len(val_images)} val")

    print(f"\n[DONE] Total: {total_train} train / {total_val} val")
    print(f"  Output: {output_dir}")


def count_dataset(data_dir):
    """
    นับจำนวนภาพในแต่ละ class
    
    Args:
        data_dir: โฟลเดอร์ dataset
    
    Returns:
        dict: {class_name: count}
    """
    counts = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            counts[class_name] = len(images)
    return counts


def print_dataset_summary(data_dir):
    """แสดงสรุปข้อมูลใน dataset"""
    print(f"\nDataset Summary: {data_dir}")
    print("-" * 40)

    counts = count_dataset(data_dir)
    total = 0
    for class_name, count in counts.items():
        bar = "█" * (count // 5) + "░" * max(0, 20 - count // 5)
        print(f"  {class_name:30s} {bar} {count}")
        total += count

    print("-" * 40)
    print(f"  {'Total':30s}   {total}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print_dataset_summary(sys.argv[1])
    else:
        print("Usage: python utils.py <dataset_directory>")
        print("Example: python utils.py data/raw/crane")
