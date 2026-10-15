"""
audit_dataset_leakage.py — Standalone Zero-Leakage & Integrity Auditor
========================================================================
Run rigorous 4-stage audit on any dataset split folder:
1. 1:1 Image and Label alignment
2. Strict Session isolation across splits
3. 0-Byte Background Label verification
4. Cryptographic MD5 Hash exclusivity across splits (0 duplicate images)

Usage:
    python scripts/yolo/audit_dataset_leakage.py --dir data/origami_seg_v2_bg_balanced
    python scripts/yolo/audit_dataset_leakage.py --dir data/origami_seg_v3_bg_balanced
"""

import argparse
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description="Audit dataset for Data Leakage and Label Integrity.")
    parser.add_argument(
        "--dir",
        type=str,
        default=str(ROOT / "data" / "origami_seg_v2_bg_balanced"),
        help="Path to dataset directory containing images/ and labels/.",
    )
    return parser.parse_args()


def compute_md5(filepath: Path) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_dataset(dataset_dir: Path):
    print("=" * 70)
    print("ZERO-LEAKAGE AUDITOR")
    print(f"Auditing Dataset Target: {dataset_dir}")
    print("=" * 70)

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return False

    splits = ["train", "val", "test"]
    for s in splits:
        if not (dataset_dir / "images" / s).exists():
            print(f"[ERROR] Missing images/{s} split!")
            return False
        if not (dataset_dir / "labels" / s).exists():
            print(f"[ERROR] Missing labels/{s} split!")
            return False

    # Stage 1: 1:1 Image and Label Alignment
    print("\n[1/4] Checking 1:1 Image and Label Alignment...")
    for s in splits:
        imgs = set(p.stem for p in (dataset_dir / "images" / s).glob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"])
        lbls = set(p.stem for p in (dataset_dir / "labels" / s).glob("*.txt"))
        missing_lbls = imgs - lbls
        orphan_lbls = lbls - imgs
        if missing_lbls:
            print(f"  [FAIL] {len(missing_lbls)} images in '{s}' are missing label files!")
            return False
        if orphan_lbls:
            print(f"  [FAIL] {len(orphan_lbls)} label files in '{s}' have no matching image!")
            return False
        print(f"  -> {s.upper():<5}: {len(imgs)} images, {len(lbls)} labels [1:1 MATCH]")

    # Stage 2: Session Isolation
    print("\n[2/4] Checking Session Isolation Across Splits...")
    session_locations: Dict[str, Set[str]] = defaultdict(set)
    for s in splits:
        for p in (dataset_dir / "images" / s).glob("*.*"):
            m = re.match(r"^(session_\d+)", p.name)
            if m:
                session_locations[m.group(1)].add(s)

    leaks = {sess: sp for sess, sp in session_locations.items() if len(sp) > 1}
    if leaks:
        print(f"  [FAIL] Session Leakage Detected! Sessions appearing in multiple splits: {leaks}")
        return False
    
    for sess, sp in sorted(session_locations.items()):
        print(f"  -> {sess} strictly isolated to -> {list(sp)[0].upper()}")
    print("  => Result: PASSED (Zero Session Cross-Contamination)")

    # Stage 3: 0-Byte Background Verification
    print("\n[3/4] Verifying 0-Byte Background Labels...")
    bg_counts = {}
    for s in splits:
        bgs = 0
        for lbl in (dataset_dir / "labels" / s).glob("*.txt"):
            if "bg_" in lbl.name or lbl.stat().st_size == 0:
                if lbl.stat().st_size != 0:
                    print(f"  [FAIL] Background label is NOT 0 bytes: {lbl.name} (size: {lbl.stat().st_size} bytes)")
                    return False
                bgs += 1
        bg_counts[s] = bgs
        print(f"  -> {s.upper():<5}: {bgs} authentic background labels verified at exactly 0 bytes")
    total_bgs = sum(bg_counts.values())
    print(f"  => Result: PASSED ({total_bgs} total clean 0-byte negative samples)")

    # Stage 4: Cryptographic MD5 Hash Exclusivity
    print("\n[4/4] Checking Cross-Split MD5 Hash Exclusivity...")
    split_hashes: Dict[str, Dict[str, Path]] = {s: {} for s in splits}
    for s in splits:
        for p in (dataset_dir / "images" / s).glob("*.*"):
            if p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                h = compute_md5(p)
                split_hashes[s][h] = p

    train_hashes = set(split_hashes["train"].keys())
    val_hashes = set(split_hashes["val"].keys())
    test_hashes = set(split_hashes["test"].keys())

    train_val_collision = train_hashes & val_hashes
    train_test_collision = train_hashes & test_hashes
    val_test_collision = val_hashes & test_hashes

    if train_val_collision:
        print(f"  [FAIL] {len(train_val_collision)} identical images found between Train and Val!")
        for h in list(train_val_collision)[:3]:
            print(f"         Collision: {split_hashes['train'][h].name} == {split_hashes['val'][h].name}")
        return False

    if train_test_collision:
        print(f"  [FAIL] {len(train_test_collision)} identical images found between Train and Test!")
        return False

    if val_test_collision:
        print(f"  [FAIL] {len(val_test_collision)} identical images found between Val and Test!")
        return False

    total_unique = len(train_hashes) + len(val_hashes) + len(test_hashes)
    print(f"  -> 0 Hash Collisions across all splits!")
    print(f"  -> Total Unique Images: {total_unique}")
    print("  => Result: PASSED (100% Cryptographically Leak-Free)")

    print("\n" + "=" * 70)
    print("AUDIT SUMMARY: DATASET IS 100% CLEAN OF DATA LEAKAGE!")
    print("=" * 70)
    return True


def main():
    args = parse_args()
    target = Path(args.dir)
    passed = audit_dataset(target)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
