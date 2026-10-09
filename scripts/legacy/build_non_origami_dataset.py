"""
build_non_origami_dataset.py — สกัดและสร้างชุดข้อมูล Chk0_NonOrigami (Negative Class)
=====================================================================================
สกัดภาพและแบ่ง Split แบบ Zero Data Leakage:
1. คลิปทดสอบ 1: experiments/*14-13-19.mp4
   - ช่วง 0:02 - 0:22 (เป้สีดำ, โฟมเหลือง, เก้าอี้, ห้อง, ช่องว่างโต๊ะขาว)
   -> จัดเข้าชุด Train
2. คลิปทดสอบ 2: experiments/GamiSenpai - AI Origami Assistant and 2 Test 10-09-26.mp4
   - ช่วง 0:13 - 0:24 (แล็ปท็อป, บัตรโรงเรียนศรีราชา, คีย์บอร์ด, โต๊ะ)
   - ช่วง 0:31 - 0:48 (กระดาษเปล่าสีขาว A4 ที่ยังไม่ได้พับ)
   -> จัดเข้าชุด Val
3. คลิปทดสอบ 3: experiments/*After add Nogami class*.mp4
   - ช่วง 0:02 - 0:26 (พื้นห้อง, สมุดโน้ตสีน้ำตาล, มือเปล่า, ขาตั้งกล้อง)
   -> จัดเข้าชุด Test
4. Background crops บริสุทธิ์จากคลิปดิบ Sessions 3, 4, 5:
   - Session 4 (โต๊ะดำ daylight) -> จัดเข้าชุด Train
   - Session 3 (พื้นห้องสีเทา night) -> จัดเข้าชุด Val
   - Session 5 (โต๊ะส้ม night) -> จัดเข้าชุด Test

* Zero Data Leakage: แยกชุดข้อมูลตาม Video / Session Source ก่อนเสมอ
* Data Augmentation (Horizontal Flip) ทำเฉพาะในชุด Train เท่านั้น ห้ามทำกับ Val/Test
* ห้ามดึงภาพจากโฟลเดอร์ Rejected เพราะใน Rejected มีภาพโอริกามิจริงที่ถูกฟิลเตอร์ blur คัดออกปนอยู่!
"""

import glob
import os
from pathlib import Path
import random
import shutil
import sys
import cv2
import numpy as np

if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from utils import safe_rmtree

DATA_DIR = ROOT / "data" / "step6" / "five_session" / "processed"
EXPERIMENTS_DIR = ROOT / "experiments"

SEED = 42
random.seed(SEED)


def extract_clean_webcam_frames(video_path: Path, intervals, step=3):
    """สกัดเฟรมจาก webcam feed ในหน้าจอ Screen Recording โดยครอปเลี่ยง UI badges/borders"""
    extracted = []
    if not video_path.is_file():
        print(f"[WARN] Video not found: {video_path}")
        return extracted

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    for idx, (start_sec, end_sec, label_prefix) in enumerate(intervals):
        start_frame = int(start_sec * fps)
        end_frame = int(end_sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        current_frame = start_frame
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if (current_frame - start_frame) % step == 0:
                h, w = frame.shape[:2]
                if w == 1920 and h == 1140:
                    # Webcam กล้องอยู่ประมาณ y: 300..840, x: 240..1035
                    # ครอปหลบ UI status badge ด้านบน และกรอบ bounding box ขอบจอ
                    cam_crop = frame[360:800, 290:980]
                else:
                    cam_crop = frame

                if cam_crop.size > 0:
                    extracted.append((cam_crop, f"{label_prefix}_f{current_frame:05d}"))
            current_frame += 1

    cap.release()
    print(f"[INFO] Extracted {len(extracted)} clean frames from {video_path.name}")
    return extracted


def extract_pure_backgrounds_from_raw_videos(limit_per_session=50):
    """สกัดภาพพื้นผิวโต๊ะ/พื้นเปล่าจากคลิปดิบ Sessions 3, 4, 5 โดยครอปเฉพาะส่วนที่ไม่มีโอริกามิ
    แยกข้อมูลตาม session prefix เพื่อป้องกัน session leakage ข้าม split
    """
    extracted_by_session = {
        "raw_bg_blackdesk": [],  # Session 4 (Train)
        "raw_bg_floor": [],      # Session 3 (Val)
        "raw_bg_orangedesk": [],  # Session 5 (Test)
    }

    session_configs = [
        ("session_004_train_blkwin_daylight", 0.60, 0.95, 0.10, 0.70, "raw_bg_blackdesk"),
        ("session_003_val_floor_night_roomlight", 0.60, 0.95, 0.10, 0.70, "raw_bg_floor"),
        ("session_005_test_orgnight_roomlight", 0.65, 0.95, 0.10, 0.70, "raw_bg_orangedesk"),
    ]

    for sname, y1_r, y2_r, x1_r, x2_r, prefix in session_configs:
        vdir = ROOT / "data" / "sessions" / sname / "raw_data_video"
        vfiles = list(vdir.glob("**/*.mp4"))
        random.shuffle(vfiles)
        count = 0

        for v in vfiles:
            if count >= limit_per_session:
                break
            cap = cv2.VideoCapture(str(v))
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                y1, y2 = int(h * y1_r), int(h * y2_r)
                x1, x2 = int(w * x1_r), int(w * x2_r)
                bg_crop = frame[y1:y2, x1:x2]
                if bg_crop.size > 0:
                    extracted_by_session[prefix].append((bg_crop, f"{prefix}_{count:03d}_{v.stem}"))
                    count += 1
            cap.release()

        print(f"[INFO] Extracted {count} pure background crops from {sname}")

    return extracted_by_session


def partition_and_augment(train_raw, val_raw, test_raw, seed=42):
    """
    แบ่งชุดข้อมูลและทำ Data Augmentation เฉพาะ Train (Zero Leakage):
    - Train: ทำ Horizontal Flip เพื่อเพิ่มความหลากหลาย
    - Val & Test: คงสภาพข้อมูลดิบไว้ ห้าม augment หรือ flip เด็ดขาด
    - สุ่มสับลำดับภาพแยกเฉพาะภายในแต่ละ split
    """
    rng = random.Random(seed)

    train_augmented = []
    for img, name in train_raw:
        train_augmented.append((img, name))
        flipped = cv2.flip(img, 1)
        train_augmented.append((flipped, f"{name}_flip"))

    val_final = list(val_raw)
    test_final = list(test_raw)

    rng.shuffle(train_augmented)
    rng.shuffle(val_final)
    rng.shuffle(test_final)

    return {
        "train": train_augmented,
        "val": val_final,
        "test": test_final,
    }


def main():
    print("=" * 60)
    print("Building 100% Pure Chk0_NonOrigami dataset (Zero Leakage)")
    print("=" * 60)

    vdo1_matches = list(EXPERIMENTS_DIR.glob("*14-13-19.mp4"))
    vdo2_matches = list(EXPERIMENTS_DIR.glob("*Test 10-09-26.mp4"))
    vdo3_matches = list(EXPERIMENTS_DIR.glob("*After add Nogami class*.mp4"))

    # 1. วิดีโอที่ 1 (เป้สีดำ, โฟมเหลือง, เก้าอี้, ช่องว่างโต๊ะขาว) -> จัดเข้า Train
    train_vdo_samples = []
    if vdo1_matches:
        vdo1_path = vdo1_matches[0]
        intervals_vdo1 = [
            (2.0, 22.0, "vdo1_backpack_chair_desk"),
        ]
        train_vdo_samples = extract_clean_webcam_frames(vdo1_path, intervals_vdo1, step=3)

    # 2. วิดีโอที่ 2 (แล็ปท็อป, บัตรนักเรียน, คีย์บอร์ด, โต๊ะเปล่า) -> จัดเข้า Val
    # * ห้ามรวมช่วงกระดาษขาว flat_white_paper เพื่อไม่ให้โมเดลจำสับสนกับ Step 1!
    val_vdo_samples = []
    if vdo2_matches:
        vdo2_path = vdo2_matches[0]
        intervals_vdo2 = [
            (13.0, 24.0, "vdo2_desk_laptop_card"),
        ]
        val_vdo_samples = extract_clean_webcam_frames(vdo2_path, intervals_vdo2, step=3)

    # 3. วิดีโอที่ 3 (พื้นห้อง, สมุดโน้ตสีน้ำตาล, มือเปล่า, ขาตั้งกล้อง) -> จัดเข้า Test
    test_vdo_samples = []
    if vdo3_matches:
        vdo3_path = vdo3_matches[0]
        intervals_vdo3 = [
            (2.0, 26.0, "vdo3_room_brown_booklet_hand"),
        ]
        test_vdo_samples = extract_clean_webcam_frames(vdo3_path, intervals_vdo3, step=3)

    # 4. พื้นหลังโต๊ะ/พื้นเปล่าจาก raw videos แยกตาม Session อย่างเข้มงวด
    # Session 4 -> Train, Session 3 -> Val, Session 5 -> Test
    bg_by_session = extract_pure_backgrounds_from_raw_videos(limit_per_session=50)

    # รวมข้อมูลดิบตาม split โดยแยก session ขาดจากกันเด็ดขาด (Zero Leakage)
    train_raw = train_vdo_samples + bg_by_session["raw_bg_blackdesk"]
    val_raw = val_vdo_samples + bg_by_session["raw_bg_floor"]
    test_raw = test_vdo_samples + bg_by_session["raw_bg_orangedesk"]

    print(f"[INFO] Raw negative samples partitioned by source (zero leakage):")
    print(f"       - Train : {len(train_raw)} (vdo1 + session_004)")
    print(f"       - Val   : {len(val_raw)} (vdo2 + session_003)")
    print(f"       - Test  : {len(test_raw)} (vdo3 + session_005)")

    # 5. ทำ Data Augmentation (Horizontal Flip) เฉพาะชุด Train เท่านั้น!
    # ห้ามทำกับ Val/Test เด็ดขาด เพื่อให้การประเมินผลเป็นตัวแทนของภาพจริงที่ไม่เคยเห็น
    splits = partition_and_augment(train_raw, val_raw, test_raw, seed=SEED)

    print(f"[INFO] Final dataset breakdown (after train-only augmentation):")
    print(f"       - Train : {len(splits['train'])} ({len(train_raw)} original + {len(train_raw)} flipped)")
    print(f"       - Val   : {len(splits['val'])} (clean unaugmented)")
    print(f"       - Test  : {len(splits['test'])} (clean unaugmented)")

    if not any(len(s) > 0 for s in splits.values()):
        print("[WARN] No samples extracted. Check video files or data/sessions directories.")
        return

    # 6. ลบโฟลเดอร์ Chk0_NonOrigami เก่าทิ้งทั้งหมดเพื่อความสะอาด
    for split_name in ["train", "val", "test"]:
        out_dir = DATA_DIR / split_name / "Chk0_NonOrigami"
        safe_rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        out_dir = DATA_DIR / split_name / "Chk0_NonOrigami"
        print(f"  -> Writing {len(items)} clean images to {out_dir}...")
        for idx, (img, name) in enumerate(items):
            target_path = out_dir / f"non_origami_{idx:04d}_{name}.jpg"
            h, w = img.shape[:2]
            if max(h, w) > 512:
                scale = 512.0 / max(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(target_path), img)

    print("[SUCCESS] Pure Chk0_NonOrigami dataset generated successfully with zero cross-split leakage!")


if __name__ == "__main__":
    main()
