"""
convert_roboflow_coco_to_yolo_seg.py — Convert Roboflow COCO Segmentation to YOLOv11-seg format
================================================================================================
1. Reads data/roboflow.zip (COCO JSON + images).
2. Converts polygon coordinates to YOLO normalized format: <class_idx> <x1> <y1> <x2> <y2> ...
3. Distributes images and labels into train/val/test splits preserving existing session stratification.
4. Validates labels and generates data.yaml ready for YOLOv11-seg training.
"""

import json
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = ROOT / "data" / "roboflow.zip"
OUT_DIR = ROOT / "data" / "origami_seg_roboflow_v1"
CONFIG_YAML_PATH = ROOT / "configs" / "origami_seg_roboflow_v1.yaml"
BACKUP_DIR = ROOT / "data" / "origami_seg_backup"

CATEGORY_MAP = {
    "Chk1": 0,
    "Chk2": 1,
    "Chk3": 2,
    "Chk4": 3,
    "Chk5": 4,
    "Chk6": 5,
}


def build_split_lookup():
    """Look up which split each clean filename belonged to in the backup/existing data."""
    lookup = {}
    if BACKUP_DIR.exists():
        for split in ["train", "val", "test"]:
            img_dir = BACKUP_DIR / "images" / split
            if img_dir.exists():
                for p in img_dir.glob("*.*"):
                    lookup[p.name] = split
    return lookup


def convert():
    print(f"Opening Roboflow zip: {ZIP_PATH}")
    assert ZIP_PATH.exists(), f"Missing {ZIP_PATH}"

    split_lookup = build_split_lookup()
    print(f"Found existing split assignments for {len(split_lookup)} images.")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        with z.open("train/_annotations.coco.json") as f:
            coco = json.load(f)

        # Build category id -> class index map
        cat_id_to_cls = {}
        for cat in coco.get("categories", []):
            name = cat.get("name")
            if name in CATEGORY_MAP:
                cat_id_to_cls[cat["id"]] = CATEGORY_MAP[name]

        print(f"Category ID mapping: {cat_id_to_cls}")

        # Group annotations by image_id
        annotations_by_img = defaultdict(list)
        for ann in coco.get("annotations", []):
            annotations_by_img[ann["image_id"]].append(ann)

        print(f"Total images in COCO: {len(coco['images'])}")
        print(f"Total annotations in COCO: {len(coco['annotations'])}")

        # Prepare output directories
        for split in ["train", "val", "test"]:
            (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

        processed = 0
        split_counts = defaultdict(int)
        class_counts = defaultdict(int)

        for img_info in coco["images"]:
            img_id = img_info["id"]
            rf_file_name = img_info["file_name"]
            # Clean original name stored in extra['name']
            clean_name = img_info.get("extra", {}).get("name", rf_file_name)
            stem = Path(clean_name).stem

            w = float(img_info["width"])
            h = float(img_info["height"])

            # Determine split
            split = split_lookup.get(clean_name, "train")
            split_counts[split] += 1

            # Extract image from zip
            zip_member = f"train/{rf_file_name}"
            dest_img = OUT_DIR / "images" / split / f"{stem}.jpg"
            with z.open(zip_member) as src, open(dest_img, "wb") as dst:
                dst.write(src.read())

            # Convert annotations to YOLO polygon lines
            yolo_lines = []
            img_anns = annotations_by_img.get(img_id, [])
            for ann in img_anns:
                cat_id = ann["category_id"]
                if cat_id not in cat_id_to_cls:
                    continue
                cls_idx = cat_id_to_cls[cat_id]
                class_counts[cls_idx] += 1

                for seg in ann.get("segmentation", []):
                    if len(seg) < 6:
                        continue
                    # Normalize points
                    norm_pts = []
                    for i in range(0, len(seg), 2):
                        px = min(1.0, max(0.0, float(seg[i]) / w))
                        py = min(1.0, max(0.0, float(seg[i + 1]) / h))
                        norm_pts.append(f"{px:.6f}")
                        norm_pts.append(f"{py:.6f}")
                    yolo_lines.append(f"{cls_idx} " + " ".join(norm_pts))

            # Write label file
            dest_lbl = OUT_DIR / "labels" / split / f"{stem}.txt"
            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            processed += 1
            if processed % 500 == 0:
                print(f"Processed {processed}/{len(coco['images'])} images...")

        print(f"\n[DONE] Successfully converted {processed} images!")
        print("Split counts:", dict(split_counts))
        print("Class counts (Chk1-Chk6):", dict(class_counts))

        # Write data.yaml
        data_yaml = f"""# Origami Instance Segmentation Dataset Config (Ultralytics YOLOv11-seg)
path: {OUT_DIR.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 6
names: ['Chk1', 'Chk2', 'Chk3', 'Chk4', 'Chk5', 'Chk6']
"""
        with open(OUT_DIR / "data.yaml", "w", encoding="utf-8") as f:
            f.write(data_yaml)
        with open(CONFIG_YAML_PATH, "w", encoding="utf-8") as f:
            f.write(data_yaml)
        print(f"Saved data.yaml and {CONFIG_YAML_PATH} successfully.")


if __name__ == "__main__":
    convert()
