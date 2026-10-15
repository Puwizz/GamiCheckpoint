"""
auto_label_dataset.py — Auto-annotate existing dataset to YOLO format
=====================================================================
Reads images from ImageFolder format (data/step6/five_session/processed or custom),
runs foreground paper detection (with fallbacks) to compute bounding boxes,
and exports into standard YOLO format:
  data/yolo_dataset/
    images/train/, val/, test/
    labels/train/, val/, test/
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from tqdm import tqdm
import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from preprocessing import crop_to_foreground

CLASS_MAP = {
    "Chk1": 0,
    "Chk2": 1,
    "Chk3": 2,
    "Chk4": 3,
    "Chk5": 4,
    "Chk6": 5,
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ImageFolder dataset to YOLO format with auto-annotations.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "data" / "step6" / "five_session" / "processed",
        help="Path to source dataset containing train/val/test folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "yolo_dataset",
        help="Path to export YOLO dataset.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        default=True,
        help="Copy images to target folder (default True).",
    )
    return parser.parse_args()


def convert_split(split_name: str, source_split_dir: Path, output_dir: Path, copy_images: bool = True):
    out_img_dir = output_dir / "images" / split_name
    out_lbl_dir = output_dir / "labels" / split_name
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    annotated_images = 0
    fallback_full_frame = 0

    classes_found = [c for c in sorted(os.listdir(source_split_dir)) if c in CLASS_MAP]
    print(f"\nProcessing split '{split_name}' — found classes: {classes_found}")

    for class_name in classes_found:
        class_id = CLASS_MAP[class_name]
        class_dir = source_split_dir / class_name
        image_paths = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]

        desc = f"  {class_name} ({len(image_paths)} images)"
        for img_path in tqdm(image_paths, desc=desc):
            total_images += 1
            im = cv2.imread(str(img_path))
            if im is None:
                continue

            h, w = im.shape[:2]
            _, meta = crop_to_foreground(im)

            if meta.get("applied"):
                x1, y1, x2, y2 = meta["bbox"]
                annotated_images += 1
            else:
                # Fallback to centered 80% box if crop_to_foreground failed completely
                fallback_full_frame += 1
                bw, bh = int(w * 0.8), int(h * 0.8)
                x1 = int((w - bw) / 2)
                y1 = int((h - bh) / 2)
                x2 = x1 + bw
                y2 = y1 + bh

            # Normalize to YOLO format: cx, cy, nw, nh (0.0 to 1.0)
            box_w = max(1, x2 - x1)
            box_h = max(1, y2 - y1)
            cx = (x1 + box_w / 2.0) / w
            cy = (y1 + box_h / 2.0) / h
            nw = box_w / float(w)
            nh = box_h / float(h)

            cx = min(1.0, max(0.0, cx))
            cy = min(1.0, max(0.0, cy))
            nw = min(1.0, max(0.001, nw))
            nh = min(1.0, max(0.001, nh))

            unique_stem = f"{class_name}_{img_path.stem}"
            dest_img_path = out_img_dir / f"{unique_stem}{img_path.suffix}"
            dest_lbl_path = out_lbl_dir / f"{unique_stem}.txt"

            if copy_images and not dest_img_path.exists():
                shutil.copy2(img_path, dest_img_path)

            with open(dest_lbl_path, "w", encoding="utf-8") as f:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    print(f"Finished {split_name}: {total_images} total, {annotated_images} foreground crops, {fallback_full_frame} fallbacks.")


def main():
    args = parse_args()
    if not args.source_dir.is_dir():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    for split in ["train", "val", "test"]:
        split_dir = args.source_dir / split
        if not split_dir.is_dir():
            print(f"[WARN] Split directory missing: {split_dir}")
            continue
        convert_split(split, split_dir, args.output_dir, copy_images=args.copy_images)

    print(f"\n[SUCCESS] YOLO dataset ready at: {args.output_dir}")


if __name__ == "__main__":
    main()
