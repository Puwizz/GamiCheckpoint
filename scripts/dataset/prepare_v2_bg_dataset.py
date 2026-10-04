"""
prepare_v2_bg_dataset.py — Build Zero-Leakage Dataset v2 with Background Images
==============================================================================
Creates data/origami_seg_v2_bg_balanced/ by merging:
1. Base balanced split (origami_seg_v1_balanced_split: 1,963 images)
2. Screened Background candidates (data/background_candidates: 398 images)

Strict Zero-Leakage Session Distribution:
- Train:
    - Base: session_001, session_002, session_003, session_005 (1,374 imgs)
    - BG: session_001 (28), session_002 (42), session_003 (41), session_005 (45)
    - Hard Negatives: bedroom_fan_bed_clothing (60), floor_clutter_cards_notes (70), laptop_keyboard_stickers (66)
    -> Total Train: 1,726 imgs (20.4% Background)
- Val:
    - Base: session_004 (331 imgs)
    - BG: session_004 (25 imgs)
    -> Total Val: 356 imgs (7.0% Background)
- Test:
    - Base: session_006 (258 imgs)
    - BG: session_006 (21 imgs)
    -> Total Test: 279 imgs (7.5% Background)

Total Dataset: 2,361 images, 2,361 labels (1:1 verified, zero session leakage).
"""

import hashlib
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
BASE_DATASET = ROOT / "data" / "origami_seg_v1_balanced_split"
BG_CANDIDATES = ROOT / "data" / "background_candidates"
TARGET_DATASET = ROOT / "data" / "origami_seg_v2_bg_balanced"
TARGET_YAML = ROOT / "configs" / "origami_seg_v2_bg_balanced.yaml"

# Background destination split mapping
BG_SPLIT_MAP: Dict[str, str] = {
    # Train
    "session_001_night_Xplor": "train",
    "session_002_noon_Xpolr": "train",
    "session_003_val_floor_night": "train",
    "session_005_test_orgnight_roomlight": "train",
    "bedroom_fan_bed_clothing": "train",
    "floor_clutter_cards_notes": "train",
    "laptop_keyboard_stickers": "train",
    # Val (Matches session_004 in Val set)
    "session_004_train_blkwin_daylight": "val",
    # Test (Matches session_006 in Test set)
    "session_006_noon_floor": "test",
}


def compute_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main():
    print("=" * 70)
    print("BUILDING ZERO-LEAKAGE DATASET V2 WITH BACKGROUND CANDIDATES")
    print(f"Base Dataset: {BASE_DATASET}")
    print(f"Background Source: {BG_CANDIDATES}")
    print(f"Target Dataset: {TARGET_DATASET}")
    print("=" * 70)

    if not BASE_DATASET.exists():
        raise FileNotFoundError(f"Base dataset {BASE_DATASET} does not exist!")
    if not BG_CANDIDATES.exists():
        raise FileNotFoundError(f"Background candidates {BG_CANDIDATES} does not exist!")

    # 1. Clean and initialize target directory
    if TARGET_DATASET.exists():
        shutil.rmtree(TARGET_DATASET)

    for split in ["train", "val", "test"]:
        (TARGET_DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (TARGET_DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 2. Copy Base Dataset (v1 balanced split)
    print("\n[STEP 1/3] Copying Base Dataset...")
    base_counts = {}
    for split in ["train", "val", "test"]:
        src_img_dir = BASE_DATASET / "images" / split
        src_lbl_dir = BASE_DATASET / "labels" / split
        dst_img_dir = TARGET_DATASET / "images" / split
        dst_lbl_dir = TARGET_DATASET / "labels" / split

        imgs = list(src_img_dir.glob("*.jpg"))
        for img in imgs:
            shutil.copy2(img, dst_img_dir / img.name)
            lbl = src_lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl_dir / lbl.name)

        base_counts[split] = len(imgs)
        print(f"  {split.upper()}: {len(imgs)} base images copied")

    # 3. Distribute Background Candidates
    print("\n[STEP 2/3] Distributing Background Candidates...")
    bg_counts = {"train": 0, "val": 0, "test": 0}
    for bg_folder in sorted(BG_CANDIDATES.glob("*/*")):
        img_dir = bg_folder / "images"
        lbl_dir = bg_folder / "labels"
        if not (img_dir.exists() and lbl_dir.exists()):
            continue

        folder_name = bg_folder.name
        if folder_name not in BG_SPLIT_MAP:
            print(f"  [WARN] Unknown folder {folder_name}, skipping")
            continue

        dest_split = BG_SPLIT_MAP[folder_name]
        dst_img_dir = TARGET_DATASET / "images" / dest_split
        dst_lbl_dir = TARGET_DATASET / "labels" / dest_split

        imgs = sorted(list(img_dir.glob("*.jpg")))
        copied_here = 0
        for img in imgs:
            lbl = lbl_dir / f"{img.stem}.txt"
            assert lbl.exists(), f"Missing label for {img}"
            assert lbl.stat().st_size == 0, f"Label {lbl} is not 0 bytes!"

            shutil.copy2(img, dst_img_dir / img.name)
            shutil.copy2(lbl, dst_lbl_dir / lbl.name)
            copied_here += 1

        bg_counts[dest_split] += copied_here
        print(f"  -> {folder_name} ({copied_here} imgs) -> {dest_split.upper()}")

    # 4. Generate Dataset YAML
    print("\n[STEP 3/3] Generating Dataset Configuration YAML...")
    yaml_content = f"""# Origami Instance Segmentation Dataset v2 (Balanced Session Split + Backgrounds)
# Base: v1 balanced split (1,963 images)
# Backgrounds: 398 authentic images (6 sessions + laptop stickers + floor clutter + bedroom)
#
# Train: 1,726 images (1,374 base + 352 BG, 20.4% BG)
# Val:     356 images (  331 base +  25 BG,  7.0% BG)
# Test:    279 images (  258 base +  21 BG,  7.5% BG)
# Total: 2,361 images (100% Zero Session Leakage)

path: {TARGET_DATASET.as_posix()}
train: images/train
val: images/val
test: images/test

nc: 6
names: ['Chk1', 'Chk2', 'Chk3', 'Chk4', 'Chk5', 'Chk6']
"""
    with open(TARGET_YAML, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    print(f"  YAML saved to {TARGET_YAML}")

    # 5. Rigorous 4-Stage Zero-Leakage Audit
    print("\n" + "=" * 70)
    print("RUNNING ZERO-LEAKAGE AUDIT ON DATASET V2")
    print("=" * 70)

    # Audit 1: 1:1 Image and Label match + 0-byte label verification
    total_imgs = 0
    total_bgs = 0
    for split in ["train", "val", "test"]:
        imgs = set(p.stem for p in (TARGET_DATASET / "images" / split).glob("*.jpg"))
        lbls = set(p.stem for p in (TARGET_DATASET / "labels" / split).glob("*.txt"))
        assert imgs == lbls, f"Mismatch in {split}: {len(imgs)} imgs vs {len(lbls)} lbls"

        zero_lbls = [p for p in (TARGET_DATASET / "labels" / split).glob("*.txt") if p.stat().st_size == 0]
        assert len(zero_lbls) == bg_counts[split], (
            f"0-byte label count mismatch in {split}: found {len(zero_lbls)} vs expected {bg_counts[split]}"
        )
        total_imgs += len(imgs)
        total_bgs += len(zero_lbls)
        print(f"  [1/4] {split.upper()}: {len(imgs)} imgs ({len(zero_lbls)} 0-byte BG) - 1:1 PASSED")

    # Audit 2: Mutual Session Exclusivity
    split_sessions: Dict[str, Set[str]] = {}
    for split in ["train", "val", "test"]:
        sessions = set()
        for p in (TARGET_DATASET / "images" / split).glob("*.jpg"):
            for s_id in ["session_001", "session_002", "session_003", "session_004", "session_005", "session_006"]:
                if s_id in p.name:
                    sessions.add(s_id)
        split_sessions[split] = sessions
        print(f"  [2/4] {split.upper()} Sessions: {sorted(list(sessions))}")

    # Verify Val is strictly session_004
    assert split_sessions["val"] == {"session_004"}, f"Val session leakage: {split_sessions['val']}"
    # Verify Test is strictly session_006
    assert split_sessions["test"] == {"session_006"}, f"Test session leakage: {split_sessions['test']}"
    # Verify Train has no session_004 or session_006
    assert "session_004" not in split_sessions["train"], "session_004 leaked into Train!"
    assert "session_006" not in split_sessions["train"], "session_006 leaked into Train!"
    print("  [2/4] Mutual Session Exclusivity: PASSED (ZERO SESSION LEAKAGE)")

    # Audit 3: Hash Uniqueness
    hashes: Dict[str, str] = {}
    leak_hashes = []
    for split in ["train", "val", "test"]:
        for p in (TARGET_DATASET / "images" / split).glob("*.jpg"):
            h = compute_md5(p)
            if h in hashes:
                leak_hashes.append((p.name, hashes[h]))
            else:
                hashes[h] = f"{split}/{p.name}"

    assert len(leak_hashes) == 0, f"Found {len(leak_hashes)} duplicate image hashes across splits!"
    print(f"  [3/4] MD5 Hash Uniqueness ({len(hashes)} unique images): PASSED")

    # Audit 4: Final Summary
    print("  [4/4] Background Ratio Analysis:")
    print(f"    - Train: {base_counts['train']} origami + {bg_counts['train']} BG = {base_counts['train']+bg_counts['train']} (BG: {bg_counts['train']/(base_counts['train']+bg_counts['train'])*100:.1f}%)")
    print(f"    - Val:   {base_counts['val']} origami + {bg_counts['val']} BG = {base_counts['val']+bg_counts['val']} (BG: {bg_counts['val']/(base_counts['val']+bg_counts['val'])*100:.1f}%)")
    print(f"    - Test:  {base_counts['test']} origami + {bg_counts['test']} BG = {base_counts['test']+bg_counts['test']} (BG: {bg_counts['test']/(base_counts['test']+bg_counts['test'])*100:.1f}%)")
    print(f"    - GRAND TOTAL: {total_imgs} images ({total_bgs} Backgrounds, 100% 0-byte labels)")
    print("=" * 70)
    print("[SUCCESS] DATASET V2 READY FOR RUN 10 TRAINING!")
    print("=" * 70)


if __name__ == "__main__":
    main()
