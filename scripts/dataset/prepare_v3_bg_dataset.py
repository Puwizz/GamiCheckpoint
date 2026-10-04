"""
prepare_v3_bg_dataset.py — Build Zero-Leakage Dataset v3 (Pre-Augmented + Backgrounds)
========================================================================================
Creates data/origami_seg_v3_bg_balanced/ by merging:
1. Roboflow Pre-Augmented Train set (data/origami_seg_v3_balanced_split/images/train: 3,296 images)
2. Authentic Train Backgrounds (data/origami_seg_v2_bg_balanced/images/train: 352 images)
3. Clean Ground-Truth Val set (data/origami_seg_v2_bg_balanced/images/val: 356 images)
4. Clean Ground-Truth Test set (data/origami_seg_v2_bg_balanced/images/test: 279 images)

Strict Zero-Leakage Protocol:
- Train (3,648 images):
    - Augmented Origami: session_001, session_002, session_003, session_005 (3,296 images)
    - Authentic Backgrounds: session_001 (28), 002 (42), 003 (41), 005 (45) + Hard Negatives (196) = 352 images
    - Background Ratio = 352 / 3,648 = 9.65% (Optimal within Ultralytics 0-10% standard)
- Val (356 images):
    - Clean Un-augmented Origami: session_004 (331 images)
    - Authentic Backgrounds: session_004 (25 images)
- Test (279 images):
    - Clean Un-augmented Origami: session_006 (258 images)
    - Authentic Backgrounds: session_006 (21 images)

Total Dataset: 4,283 images, 4,283 labels (1:1 verified, zero data leakage).
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
V3_BALANCED = ROOT / "data" / "origami_seg_v3_balanced_split"
V2_BG_DATASET = ROOT / "data" / "origami_seg_v2_bg_balanced"
TARGET_DATASET = ROOT / "data" / "origami_seg_v3_bg_balanced"
TARGET_YAML = ROOT / "configs" / "origami_seg_v3_bg_balanced.yaml"


def compute_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    print("=" * 70)
    print("BUILDING ZERO-LEAKAGE DATASET V3 (PRE-AUGMENTED + BACKGROUNDS)")
    print(f"V3 Balanced Source : {V3_BALANCED}")
    print(f"V2 BG Source       : {V2_BG_DATASET}")
    print(f"Target Dataset     : {TARGET_DATASET}")
    print("=" * 70)

    if not V3_BALANCED.exists():
        raise FileNotFoundError(f"V3 dataset {V3_BALANCED} does not exist!")
    if not V2_BG_DATASET.exists():
        raise FileNotFoundError(f"V2 BG dataset {V2_BG_DATASET} does not exist!")

    # 1. Clean and initialize target directory
    if TARGET_DATASET.exists():
        print(f"[INFO] Removing previous {TARGET_DATASET}...")
        shutil.rmtree(TARGET_DATASET)

    for split in ["train", "val", "test"]:
        (TARGET_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. Populate Train Set
    print("\n[STEP 1/3] Assembling Train Split (3,296 v3 Origami + 352 Backgrounds)...")
    train_origami_count = 0
    v3_train_imgs = V3_BALANCED / "images" / "train"
    v3_train_lbls = V3_BALANCED / "labels" / "train"

    for img_path in v3_train_imgs.glob("*.*"):
        if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            lbl_path = v3_train_lbls / f"{img_path.stem}.txt"
            if lbl_path.exists():
                copy_file(img_path, TARGET_DATASET / "images" / "train" / img_path.name)
                copy_file(lbl_path, TARGET_DATASET / "labels" / "train" / lbl_path.name)
                train_origami_count += 1
            else:
                print(f"[WARN] Missing label for train image: {img_path.name}")

    print(f"  -> Copied {train_origami_count} augmented origami images to train.")

    # Copy authentic train backgrounds from v2
    train_bg_count = 0
    v2_train_imgs = V2_BG_DATASET / "images" / "train"
    v2_train_lbls = V2_BG_DATASET / "labels" / "train"

    for lbl_path in v2_train_lbls.glob("*.txt"):
        if lbl_path.stat().st_size == 0:  # 0-byte background label
            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = v2_train_imgs / f"{lbl_path.stem}{ext}"
                if img_path.exists():
                    copy_file(img_path, TARGET_DATASET / "images" / "train" / img_path.name)
                    copy_file(lbl_path, TARGET_DATASET / "labels" / "train" / lbl_path.name)
                    train_bg_count += 1
                    break

    print(f"  -> Copied {train_bg_count} authentic background images to train.")
    total_train = train_origami_count + train_bg_count
    bg_ratio = (train_bg_count / total_train) * 100 if total_train > 0 else 0
    print(f"  => Total Train: {total_train} images ({bg_ratio:.2f}% Backgrounds)")

    # 3. Populate Val & Test Splits (Clean Ground-Truth from v2)
    print("\n[STEP 2/3] Assembling Clean Ground-Truth Val & Test Splits...")
    for split in ["val", "test"]:
        v2_imgs = V2_BG_DATASET / "images" / split
        v2_lbls = V2_BG_DATASET / "labels" / split
        copied = 0
        bgs = 0
        for img_path in v2_imgs.glob("*.*"):
            if img_path.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                lbl_path = v2_lbls / f"{img_path.stem}.txt"
                if lbl_path.exists():
                    copy_file(img_path, TARGET_DATASET / "images" / split / img_path.name)
                    copy_file(lbl_path, TARGET_DATASET / "labels" / split / lbl_path.name)
                    copied += 1
                    if lbl_path.stat().st_size == 0:
                        bgs += 1
        print(f"  => {split.upper()}: {copied} images ({bgs} Backgrounds, {(bgs/copied)*100:.2f}%)")

    # 4. Generate Dataset YAML
    print("\n[STEP 3/3] Generating Dataset Config YAML...")
    yaml_content = f"""# Origami Instance Segmentation Dataset v3 (Pre-Augmented + Backgrounds)
# Balanced 70/17/13 Zero-Leakage Session Split
# Train: session_001, session_002, session_003, session_005 + 352 Backgrounds ({total_train} imgs)
# Val:   session_004 clean daylight bedroom + 25 Backgrounds (356 imgs)
# Test:  session_006 clean glossy floor tiles + 21 Backgrounds (279 imgs)

path: {TARGET_DATASET.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: 6
names: ['Chk1', 'Chk2', 'Chk3', 'Chk4', 'Chk5', 'Chk6']
"""
    TARGET_YAML.write_text(yaml_content, encoding="utf-8")
    print(f"  -> Saved config: {TARGET_YAML}")

    # 5. Run Strict 4-Stage Zero-Leakage Audit
    print("\n" + "=" * 70)
    print("RUNNING 4-STAGE ZERO-LEAKAGE AUDIT ON DATASET V3")
    print("=" * 70)

    # Check 1: 1:1 Image and Label Alignment
    print(" [1/4] Checking 1:1 Image and Label Alignment...")
    for split in ["train", "val", "test"]:
        imgs = set(p.stem for p in (TARGET_DATASET / "images" / split).glob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"])
        lbls = set(p.stem for p in (TARGET_DATASET / "labels" / split).glob("*.txt"))
        assert imgs == lbls, f"Alignment mismatch in {split}: {len(imgs)} imgs vs {len(lbls)} lbls"
        print(f"       -> {split}: {len(imgs)} images, {len(lbls)} labels [OK]")

    # Check 2: Strict Session Isolation
    print("\n [2/4] Checking Strict Session Isolation...")
    val_files = [p.name for p in (TARGET_DATASET / "images" / "val").glob("*.*")]
    test_files = [p.name for p in (TARGET_DATASET / "images" / "test").glob("*.*")]
    train_files = [p.name for p in (TARGET_DATASET / "images" / "train").glob("*.*")]

    # Val must only contain session_004
    for f in val_files:
        if "session_" in f:
            assert "session_004" in f, f"Leakage detected in val: {f}"
    print("       -> Val set strictly isolated to session_004 [OK]")

    # Test must only contain session_006
    for f in test_files:
        if "session_" in f:
            assert "session_006" in f, f"Leakage detected in test: {f}"
    print("       -> Test set strictly isolated to session_006 [OK]")

    # Train must not contain session_004 or session_006
    for f in train_files:
        if "session_" in f:
            assert "session_004" not in f, f"Session 004 leaked into train: {f}"
            assert "session_006" not in f, f"Session 006 leaked into train: {f}"
    print("       -> Train set has zero session_004 / session_006 instances [OK]")

    # Check 3: Background Label Zero-Byte Verification
    print("\n [3/4] Verifying 0-Byte Background Labels...")
    total_bgs = 0
    for split in ["train", "val", "test"]:
        split_bgs = 0
        for lbl in (TARGET_DATASET / "labels" / split).glob("*.txt"):
            if "bg_" in lbl.name:
                assert lbl.stat().st_size == 0, f"Background label not 0 bytes: {lbl}"
                split_bgs += 1
        print(f"       -> {split}: {split_bgs} background labels verified at exactly 0 bytes [OK]")
        total_bgs += split_bgs
    assert total_bgs == 398, f"Expected 398 background images, got {total_bgs}"

    # Check 4: Cross-Split MD5 Hash Exclusivity
    print("\n [4/4] Checking Cross-Split MD5 Hash Exclusivity...")
    split_hashes: Dict[str, Set[str]] = {}
    for split in ["train", "val", "test"]:
        hashes = set()
        for p in (TARGET_DATASET / "images" / split).glob("*.*"):
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                hashes.add(compute_md5(p))
        split_hashes[split] = hashes

    tv_overlap = split_hashes["train"] & split_hashes["val"]
    tt_overlap = split_hashes["train"] & split_hashes["test"]
    vt_overlap = split_hashes["val"] & split_hashes["test"]

    assert len(tv_overlap) == 0, f"Hash collision Train <-> Val: {len(tv_overlap)}"
    assert len(tt_overlap) == 0, f"Hash collision Train <-> Test: {len(tt_overlap)}"
    assert len(vt_overlap) == 0, f"Hash collision Val <-> Test: {len(vt_overlap)}"
    total_unique = len(split_hashes["train"]) + len(split_hashes["val"]) + len(split_hashes["test"])
    print(f"       -> 0 Hash Collisions across splits (Total unique images: {total_unique}) [OK]")

    print("\n" + "=" * 70)
    print("SUCCESS: DATASET V3 (v3_bg_balanced) BUILT AND AUDITED WITH ZERO LEAKAGE!")
    print(f"Total Dataset Images: {total_unique}")
    print(f"Train: {len(split_hashes['train'])} | Val: {len(split_hashes['val'])} | Test: {len(split_hashes['test'])}")
    print(f"Config File: {TARGET_YAML}")
    print("=" * 70)


if __name__ == "__main__":
    main()
