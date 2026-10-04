"""
split_roboflow_by_session.py — Zero-Leakage Session-Based Dataset Splitter
==========================================================================
Re-distributes Roboflow segmentation datasets (v1, v3, etc.) strictly by session
to eliminate Data Leakage (Session leakage, Video/Temporal leakage, and Hash leakage).

Default Session Mapping:
- Train: session_001, session_002, session_004
- Val  : session_003
- Test : session_005, session_006

Includes automated validation for:
1. 1:1 Image-to-Label alignment
2. Mutual session exclusivity across splits
3. Video sequence isolation (no frame from same video in multiple splits)
4. MD5 Hash uniqueness across splits
"""

import argparse
import hashlib
import os
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SESSION_MAP: Dict[str, str] = {
    "session_001": "train",
    "session_002": "train",
    "session_004": "train",
    "session_003": "val",
    "session_005": "test",
    "session_006": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-split Roboflow segmentation datasets strictly by session."
    )
    parser.add_argument(
        "--source",
        type=str,
        default=str(ROOT / "data" / "origami_seg_roboflow_v3"),
        help="Path to downloaded Roboflow dataset directory.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=str(ROOT / "data" / "origami_seg_v3_session_split"),
        help="Destination directory for clean session-split dataset.",
    )
    parser.add_argument(
        "--train-sessions",
        type=str,
        default="session_001,session_002,session_004",
        help="Comma-separated list of session IDs for train split.",
    )
    parser.add_argument(
        "--val-sessions",
        type=str,
        default="session_003",
        help="Comma-separated list of session IDs for val split.",
    )
    parser.add_argument(
        "--test-sessions",
        type=str,
        default="session_005,session_006",
        help="Comma-separated list of session IDs for test split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate the split without copying files.",
    )
    return parser.parse_args()


def audit_leakage(target_dir: Path) -> None:
    """Run rigorous 4-stage audit for data leakage in the target dataset."""
    print("\n" + "=" * 60)
    print("RUNNING ZERO-LEAKAGE AUDIT")
    print("=" * 60)

    splits = ["train", "val", "test"]

    # 1. 1:1 Image and Label match
    for s in splits:
        img_dir = target_dir / "images" / s
        lbl_dir = target_dir / "labels" / s
        if not img_dir.exists():
            continue
        imgs = set(p.stem for p in img_dir.glob("*.jpg"))
        lbls = set(p.stem for p in lbl_dir.glob("*.txt"))
        diff = imgs ^ lbls
        assert len(diff) == 0, f"Image/Label mismatch in {s}: {len(diff)} files!"
    print(" [1/4] Image and Label 1:1 Alignment: PASSED")

    # 2. Session isolation
    session_splits: Dict[str, Set[str]] = defaultdict(set)
    video_splits: Dict[str, Set[str]] = defaultdict(set)
    hash_splits: Dict[str, Set[str]] = defaultdict(set)

    for s in splits:
        img_dir = target_dir / "images" / s
        for img_path in img_dir.glob("*.jpg"):
            fname = img_path.name

            # Session check
            m_sess = re.match(r"^(session_\d+)", fname)
            if m_sess:
                session_splits[m_sess.group(1)].add(s)

            # Video check (extract session + run + video/image id)
            m_vid = re.match(r"^(session_\d+.*?(IMG|VID)_\d+_\d+)", fname)
            if m_vid:
                video_splits[m_vid.group(1)].add(s)

            # MD5 check
            with open(img_path, "rb") as fp:
                h = hashlib.md5(fp.read()).hexdigest()
            hash_splits[h].add(s)

    # Assert session isolation
    for sess, found_splits in session_splits.items():
        assert len(found_splits) == 1, f"Session Leakage detected: {sess} in {found_splits}"
    print(f" [2/4] Session Isolation ({len(session_splits)} sessions): PASSED (0 Leaks)")

    # Assert video isolation
    vid_leaks = [v for v, sp in video_splits.items() if len(sp) > 1]
    assert len(vid_leaks) == 0, f"Video Leakage detected in {len(vid_leaks)} videos!"
    print(f" [3/4] Temporal/Video Isolation ({len(video_splits)} sequences): PASSED (0 Leaks)")

    # Assert hash isolation
    hash_leaks = [h for h, sp in hash_splits.items() if len(sp) > 1]
    assert len(hash_leaks) == 0, f"Hash Leakage detected in {len(hash_leaks)} images!"
    print(f" [4/4] MD5 Hash Exclusivity ({len(hash_splits)} images): PASSED (0 Leaks)")

    print("=" * 60)
    print("ALL AUDIT CHECKS PASSED: DATASET IS 100% CLEAN OF DATA LEAKAGE!")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source)
    target_dir = Path(args.target)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    print(f"Source dataset: {source_dir}")
    print(f"Target dataset: {target_dir}")

    # Build session mapping from CLI arguments
    session_map: Dict[str, str] = {}
    for s_id in args.train_sessions.split(","):
        if s_id.strip():
            session_map[s_id.strip()] = "train"
    for s_id in args.val_sessions.split(","):
        if s_id.strip():
            session_map[s_id.strip()] = "val"
    for s_id in args.test_sessions.split(","):
        if s_id.strip():
            session_map[s_id.strip()] = "test"

    print("Session Mapping:")
    for sess, sp in sorted(session_map.items()):
        print(f"  {sess} -> {sp}")

    # Prepare target directories
    for s in ["train", "val", "test"]:
        (target_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / s).mkdir(parents=True, exist_ok=True)

    copied = 0
    split_counts: Counter = Counter()
    class_counts: Dict[str, Counter] = {s: Counter() for s in ["train", "val", "test"]}

    # Read from train, val, test subdirs in source
    for current_split in ["train", "val", "test"]:
        src_img_dir = source_dir / "images" / current_split
        src_lbl_dir = source_dir / "labels" / current_split

        if not src_img_dir.exists():
            continue

        for img_path in src_img_dir.glob("*.jpg"):
            fname = img_path.name
            m = re.match(r"^(session_\d+)", fname)
            if not m:
                continue

            session_id = m.group(1)
            target_split = session_map.get(session_id)
            if not target_split:
                print(f"[WARN] Unknown session '{session_id}' in {fname}, skipping.")
                continue

            dest_img = target_dir / "images" / target_split / fname
            lbl_name = img_path.stem + ".txt"
            src_lbl = src_lbl_dir / lbl_name
            dest_lbl = target_dir / "labels" / target_split / lbl_name

            if not args.dry_run:
                shutil.copy2(img_path, dest_img)
                if src_lbl.exists():
                    shutil.copy2(src_lbl, dest_lbl)
                    with open(src_lbl, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_counts[target_split][int(parts[0])] += 1

            copied += 1
            split_counts[target_split] += 1

    print(f"\nSuccessfully processed {copied} images:")
    for s in ["train", "val", "test"]:
        print(f"  - {s.upper():5s}: {split_counts[s]:4d} images | Classes: {dict(sorted(class_counts[s].items()))}")

    if not args.dry_run:
        audit_leakage(target_dir)


if __name__ == "__main__":
    main()
