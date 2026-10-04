"""
rebalance_yolo_split.py — Rebalance YOLO dataset images and labels
================================================================
Moves existing annotations and images between train/val to match
the updated configs/five_session_split.json without re-generating labels.
"""

import shutil
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
YOLO_DIR = ROOT / "data" / "yolo_dataset"

IMG_DIR = YOLO_DIR / "images"
LBL_DIR = YOLO_DIR / "labels"

# Target destinations based on session prefix
TARGET_MAP = {
    "session_001_night_Xplor": "train",
    "session_003_val_floor_night_roomlight": "train",
    "session_004_train_blkwin_daylight": "train",
    "session_002_noon_Xpolr": "val",
    "session_006_noon_floor": "val",
    "session_005_test_orgnight_roomlight": "test",
}


def main():
    print(f"Checking YOLO dataset in: {YOLO_DIR}")
    assert IMG_DIR.exists() and LBL_DIR.exists(), "Dataset directories missing!"

    moves = 0
    # Process images and matching labels
    for current_split in ["train", "val"]:
        img_split_dir = IMG_DIR / current_split
        lbl_split_dir = LBL_DIR / current_split

        for img_path in list(img_split_dir.iterdir()):
            if not img_path.is_file():
                continue
            sess_name = img_path.stem.split("__")[0]
            target_split = TARGET_MAP.get(sess_name)
            if not target_split:
                continue

            if target_split != current_split:
                # Target paths
                dst_img = IMG_DIR / target_split / img_path.name
                shutil.move(str(img_path), str(dst_img))

                # Also move corresponding label
                lbl_name = img_path.stem + ".txt"
                src_lbl = lbl_split_dir / lbl_name
                if src_lbl.exists():
                    dst_lbl = LBL_DIR / target_split / lbl_name
                    shutil.move(str(src_lbl), str(dst_lbl))

                moves += 1

    print(f"Successfully moved {moves} image-label pairs.")

    # Audit new counts
    print("\n--- New Dataset Counts ---")
    for split in ["train", "val", "test"]:
        imgs = list((IMG_DIR / split).glob("*.*"))
        lbls = list((LBL_DIR / split).glob("*.txt"))
        sessions = Counter([p.stem.split("__")[0] for p in imgs])
        print(f"[{split.upper()}] Images: {len(imgs)}, Labels: {len(lbls)}")
        for s, cnt in sorted(sessions.items()):
            print(f"   {s}: {cnt}")


if __name__ == "__main__":
    main()
