"""
convert_roboflow_v3_to_yolo_seg.py
===================================
Converts GamiSenpai.v3i.coco-segmentation.zip (5,889 augmented images)
into YOLOv11-seg instance segmentation format in data/origami_seg_roboflow_v3.

Split strategy (correct ML practice):
  - TRAIN : Augmented images from v3 zip (3x per original = 4,116 images)
  - VAL   : Original (non-augmented) images from v1 (393 images)
  - TEST  : Original (non-augmented) images from v1 (198 images)

This prevents data leakage and avoids inflated mAP from evaluating on augmented data.
"""

import json
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIP_PATH = ROOT / "data" / "GamiSenpai.v3i.coco-segmentation.zip"
OUT_DIR = ROOT / "data" / "origami_seg_roboflow_v3"
CONFIG_YAML_PATH = ROOT / "configs" / "origami_seg_roboflow_v3.yaml"
V1_DIR = ROOT / "data" / "origami_seg_roboflow_v1"

CATEGORY_MAP = {
    "Chk1": 0,
    "Chk2": 1,
    "Chk3": 2,
    "Chk4": 3,
    "Chk5": 4,
    "Chk6": 5,
}


def build_v1_train_lookup():
    """Return set of clean image names that belong to v1 TRAIN split (do not include in our v3 train)."""
    # exclude val/test originals from the augmented train set
    val_names = set()
    test_names = set()
    if V1_DIR.exists():
        for p in (V1_DIR / "images" / "val").glob("*.*"):
            val_names.add(p.name)
        for p in (V1_DIR / "images" / "test").glob("*.*"):
            test_names.add(p.name)
    return val_names, test_names


def copy_v1_split(split: str):
    """Copy original (non-augmented) images and labels from v1 val/test into v3 output."""
    src_img_dir = V1_DIR / "images" / split
    src_lbl_dir = V1_DIR / "labels" / split
    dst_img_dir = OUT_DIR / "images" / split
    dst_lbl_dir = OUT_DIR / "labels" / split

    count = 0
    for img_path in src_img_dir.glob("*.*"):
        shutil.copy2(img_path, dst_img_dir / img_path.name)
        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)
        count += 1
    return count


def main():
    print(f"Opening Roboflow v3 zip: {ZIP_PATH}")
    assert ZIP_PATH.exists(), f"Missing {ZIP_PATH}"
    assert V1_DIR.exists(), f"Missing v1 dir: {V1_DIR}"

    # Create output directories
    for split in ["train", "val", "test"]:
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Step 1: Copy Val/Test from v1 (original, no augmentation) ──────────
    print("\n[Step 1] Copying original val/test from v1...")
    n_val = copy_v1_split("val")
    n_test = copy_v1_split("test")
    print(f"  Copied val : {n_val} images")
    print(f"  Copied test: {n_test} images")

    # ── Step 2: Build lookup of val/test clean names to EXCLUDE from v3 train ─
    val_names, test_names = build_v1_train_lookup()
    exclude_from_train = val_names | test_names
    print(f"\n[Step 2] Excluding {len(exclude_from_train)} original val/test images from augmented train.")

    # ── Step 3: Extract augmented TRAIN from v3 zip ─────────────────────────
    print("\n[Step 3] Extracting augmented train images from v3 zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        with z.open("train/_annotations.coco.json") as f:
            coco = json.load(f)

        cat_id_to_cls = {}
        for cat in coco.get("categories", []):
            name = cat.get("name")
            if name in CATEGORY_MAP:
                cat_id_to_cls[cat["id"]] = CATEGORY_MAP[name]

        print(f"  Category ID mapping: {cat_id_to_cls}")

        annotations_by_img = defaultdict(list)
        for ann in coco.get("annotations", []):
            annotations_by_img[ann["image_id"]].append(ann)

        processed = 0
        skipped = 0
        class_counts = defaultdict(int)

        for img_info in coco["images"]:
            img_id = img_info["id"]
            rf_file_name = img_info["file_name"]
            clean_name = img_info.get("extra", {}).get("name", rf_file_name)

            # Skip augmented versions of val/test originals
            if clean_name in exclude_from_train:
                skipped += 1
                continue

            stem = Path(rf_file_name).stem
            w = float(img_info["width"])
            h = float(img_info["height"])

            # Extract image
            dest_img = OUT_DIR / "images" / "train" / f"{stem}.jpg"
            if not dest_img.exists():
                zip_member = f"train/{rf_file_name}"
                with z.open(zip_member) as src, open(dest_img, "wb") as dst:
                    dst.write(src.read())

            # Convert annotations
            yolo_lines = []
            for ann in annotations_by_img.get(img_id, []):
                cat_id = ann["category_id"]
                if cat_id not in cat_id_to_cls:
                    continue
                cls_idx = cat_id_to_cls[cat_id]
                class_counts[cls_idx] += 1

                for seg in ann.get("segmentation", []):
                    if len(seg) < 6:
                        continue
                    norm_pts = []
                    for i in range(0, len(seg), 2):
                        px = min(1.0, max(0.0, float(seg[i]) / w))
                        py = min(1.0, max(0.0, float(seg[i + 1]) / h))
                        norm_pts.append(f"{px:.6f}")
                        norm_pts.append(f"{py:.6f}")
                    yolo_lines.append(f"{cls_idx} " + " ".join(norm_pts))

            dest_lbl = OUT_DIR / "labels" / "train" / f"{stem}.txt"
            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed} train images (skipped {skipped} val/test)...")

    n_train = processed
    total = n_train + n_val + n_test

    print(f"\n[DONE] Dataset v3 created in {OUT_DIR}")
    print(f"  Train : {n_train} images  ({n_train/total*100:.1f}%)  [augmented 3x]")
    print(f"  Val   : {n_val} images   ({n_val/total*100:.1f}%)  [original, no augmentation]")
    print(f"  Test  : {n_test} images    ({n_test/total*100:.1f}%)  [original, no augmentation]")
    print(f"  Total : {total} images")
    print(f"  Class counts (Chk1-Chk6): {dict(class_counts)}")

    # Write configs
    data_yaml = f"""# Origami Instance Segmentation Dataset Config (Roboflow v3 Augmented Train Only)
# Train: 3x augmented | Val/Test: original non-augmented from v1
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
    print(f"\nSaved config at: {CONFIG_YAML_PATH}")


if __name__ == "__main__":
    main()



ZIP_PATH = ROOT / "data" / "GamiSenpai.v3i.coco-segmentation.zip"
OUT_DIR = ROOT / "data" / "origami_seg_roboflow_v3"
CONFIG_YAML_PATH = ROOT / "configs" / "origami_seg_roboflow_v3.yaml"
V1_DIR = ROOT / "data" / "origami_seg_roboflow_v1"

CATEGORY_MAP = {
    "Chk1": 0,
    "Chk2": 1,
    "Chk3": 2,
    "Chk4": 3,
    "Chk5": 4,
    "Chk6": 5,
}


def build_split_lookup():
    lookup = {}
    if V1_DIR.exists():
        for split in ["train", "val", "test"]:
            img_dir = V1_DIR / "images" / split
            if img_dir.exists():
                for p in img_dir.glob("*.*"):
                    clean_name = p.name.split(".rf.")[0] + ".jpg" if ".rf." in p.name else p.name
                    lookup[clean_name] = split
                    lookup[p.name] = split
    return lookup


def main():
    print(f"Opening Roboflow v3 zip: {ZIP_PATH}")
    assert ZIP_PATH.exists(), f"Missing {ZIP_PATH}"

    split_lookup = build_split_lookup()
    print(f"Found existing split assignments for {len(split_lookup)} base images.")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        with z.open("train/_annotations.coco.json") as f:
            coco = json.load(f)

        cat_id_to_cls = {}
        for cat in coco.get("categories", []):
            name = cat.get("name")
            if name in CATEGORY_MAP:
                cat_id_to_cls[cat["id"]] = CATEGORY_MAP[name]

        print(f"Category ID mapping: {cat_id_to_cls}")

        annotations_by_img = defaultdict(list)
        for ann in coco.get("annotations", []):
            annotations_by_img[ann["image_id"]].append(ann)

        print(f"Total images in COCO: {len(coco['images'])}")
        print(f"Total annotations in COCO: {len(coco['annotations'])}")

        for split in ["train", "val", "test"]:
            (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

        processed = 0
        split_counts = defaultdict(int)
        class_counts = defaultdict(int)

        for img_info in coco["images"]:
            img_id = img_info["id"]
            rf_file_name = img_info["file_name"]
            clean_name = img_info.get("extra", {}).get("name", rf_file_name)

            stem = Path(rf_file_name).stem
            w = float(img_info["width"])
            h = float(img_info["height"])

            split = split_lookup.get(clean_name, "train")
            split_counts[split] += 1

            zip_member = f"train/{rf_file_name}"
            dest_img = OUT_DIR / "images" / split / f"{stem}.jpg"
            if not dest_img.exists():
                with z.open(zip_member) as src, open(dest_img, "wb") as dst:
                    dst.write(src.read())

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
                    norm_pts = []
                    for i in range(0, len(seg), 2):
                        px = min(1.0, max(0.0, float(seg[i]) / w))
                        py = min(1.0, max(0.0, float(seg[i + 1]) / h))
                        norm_pts.append(f"{px:.6f}")
                        norm_pts.append(f"{py:.6f}")
                    yolo_lines.append(f"{cls_idx} " + " ".join(norm_pts))

            dest_lbl = OUT_DIR / "labels" / split / f"{stem}.txt"
            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

            processed += 1
            if processed % 1000 == 0 or processed == len(coco["images"]):
                print(f"Processed {processed}/{len(coco['images'])} images...")

        print(f"\n[DONE] Successfully converted {processed} images to {OUT_DIR}!")
        print("Split counts:", dict(split_counts))
        print("Class counts (Chk1-Chk6):", dict(class_counts))

        data_yaml = f"""# Origami Instance Segmentation Dataset Config (Roboflow v3 Augmented)
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
        print(f"Saved dataset configs at: {CONFIG_YAML_PATH}")


if __name__ == "__main__":
    main()
