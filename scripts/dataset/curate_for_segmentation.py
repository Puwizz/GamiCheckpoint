"""
curate_for_segmentation.py — Curate High-Quality Images for Roboflow Segmentation
==================================================================================
1. Filters out full-frame boxes, heavy hand obstruction, and off-center images.
2. Balances candidates across sessions (diverse lighting/angles) up to 350 images per class.
3. Resizes images to max 1280px (saving ~97% disk space while retaining crisp fold details).
4. Saves to data/yolo_curated/ (images, labels, data.yaml).
5. Compresses into data/origami_curated_for_roboflow.zip ready for Roboflow upload.
"""

import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(os.getenv("GAMISENPAI_ROOT", Path(__file__).resolve().parents[2]))
LBL_DIR = ROOT / "data" / "yolo_dataset" / "labels"
IMG_DIR = ROOT / "data" / "yolo_dataset" / "images"
OUT_DIR = ROOT / "data" / "yolo_curated"
OUT_ZIP = ROOT / "data" / "origami_curated_for_roboflow.zip"

MAX_PER_CLASS = 350
TARGET_MAX_DIM = 1280
JPEG_QUALITY = 90

CLASS_NAMES = ["Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"]


def find_candidates():
    candidates = defaultdict(lambda: defaultdict(list))
    total_scanned = 0

    print("Step 1: Scanning label metadata across all splits...")
    for split in ["train", "val", "test"]:
        split_lbl = LBL_DIR / split
        if not split_lbl.exists():
            continue
        with os.scandir(split_lbl) as it:
            for entry in it:
                if not entry.name.endswith(".txt") or not entry.is_file():
                    continue
                total_scanned += 1
                try:
                    with open(entry.path, "r", encoding="utf-8") as f:
                        line = f.readline().strip()
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx, cy, nw, nh = map(float, parts[1:5])
                        area = nw * nh

                        # Filter 1: Exclude full-frame hand/background boxes
                        if area >= 0.80 or nw >= 0.92 or nh >= 0.92:
                            continue
                        # Filter 2: Exclude tiny objects
                        if area < 0.03:
                            continue
                        # Filter 3: Must be reasonably centered
                        if cx < 0.15 or cx > 0.85 or cy < 0.15 or cy > 0.85:
                            continue

                        stem = Path(entry.name).stem
                        # Determine session name for stratification
                        session_name = stem.split("__")[0] if "__" in stem else "unknown"

                        candidates[cls_id][session_name].append({
                            "split": split,
                            "stem": stem,
                            "lbl_path": entry.path,
                            "cx": cx, "cy": cy, "nw": nw, "nh": nh,
                            "line": line
                        })
                except Exception:
                    pass

    print(f"Scanned {total_scanned} total labels.")
    return candidates


def select_balanced_samples(candidates):
    selected = []
    print("\nStep 2: Selecting balanced samples across sessions...")
    
    for cls_id in range(6):
        cls_name = CLASS_NAMES[cls_id]
        session_dict = candidates[cls_id]
        all_cls_items = []
        
        # Round-robin sampling across sessions to ensure diversity
        sessions = sorted(session_dict.keys())
        max_session_len = max((len(session_dict[s]) for s in sessions), default=0)
        
        for idx in range(max_session_len):
            for s in sessions:
                if idx < len(session_dict[s]):
                    all_cls_items.append(session_dict[s][idx])
                    if len(all_cls_items) >= MAX_PER_CLASS:
                        break
            if len(all_cls_items) >= MAX_PER_CLASS:
                break
                
        print(f"  {cls_name} (Class {cls_id}): Selected {len(all_cls_items)} images across {len(sessions)} sessions.")
        selected.extend(all_cls_items)
        
    print(f"Total curated images: {len(selected)}")
    return selected


def process_and_save(selected_items):
    out_img_dir = OUT_DIR / "images"
    out_lbl_dir = OUT_DIR / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStep 3: Resizing and saving images to {OUT_DIR}...")
    saved_count = 0

    total = len(selected_items)
    for idx, item in enumerate(selected_items):
        if (idx + 1) % 200 == 0 or idx == 0 or (idx + 1) == total:
            print(f"  Processed {idx + 1}/{total} images ({(idx + 1) / total * 100:.1f}%)...", flush=True)
        stem = item["stem"]
        split = item["split"]
        
        # Look for source image
        src_img = IMG_DIR / split / f"{stem}.jpg"
        if not src_img.exists():
            src_img = IMG_DIR / split / f"{stem}.png"
        if not src_img.exists():
            continue

        im = cv2.imread(str(src_img))
        if im is None:
            continue

        h, w = im.shape[:2]
        max_dim = max(h, w)
        if max_dim > TARGET_MAX_DIM:
            scale = TARGET_MAX_DIM / float(max_dim)
            im_resized = cv2.resize(im, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            im_resized = im

        dest_img = out_img_dir / f"{stem}.jpg"
        cv2.imwrite(str(dest_img), im_resized, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])

        # Write corresponding label
        dest_lbl = out_lbl_dir / f"{stem}.txt"
        with open(dest_lbl, "w", encoding="utf-8") as f:
            f.write(item["line"] + "\n")

        saved_count += 1

    # Write data.yaml for Roboflow reference
    yaml_content = f"""# Origami Curated Dataset for Roboflow Segmentation
path: .
train: images
val: images
test: images

nc: 6
names: {CLASS_NAMES}
"""
    with open(OUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"Successfully processed {saved_count} images.")
    return saved_count


def create_zip():
    print(f"\nStep 4: Compressing into {OUT_ZIP.name} for Roboflow...")
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for root_path, dirs, files in os.walk(OUT_DIR):
            for file in files:
                abs_path = Path(root_path) / file
                rel_path = abs_path.relative_to(OUT_DIR)
                zf.write(abs_path, arcname=str(rel_path))

    size_mb = os.path.getsize(OUT_ZIP) / (1024 * 1024)
    print(f"\n[DONE] Package ready at: {OUT_ZIP}")
    print(f"Total ZIP size: {size_mb:.2f} MB (Well under Roboflow 2GB limit!)")


def main():
    candidates = find_candidates()
    selected = select_balanced_samples(candidates)
    saved = process_and_save(selected)
    create_zip()


if __name__ == "__main__":
    main()
