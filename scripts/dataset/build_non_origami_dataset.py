"""
build_non_origami_dataset.py — สกัดและสร้างชุดข้อมูล Chk0_NonOrigami (Negative Class)
=====================================================================================
สกัดภาพจาก:
1. คลิปทดสอบ 1: experiments/*14-13-19.mp4
   - ช่วง 0:02 - 0:22 (เป้สีดำ, โฟมเหลือง, เก้าอี้, ห้อง, ช่องว่างโต๊ะขาว)
2. คลิปทดสอบ 2: experiments/GamiSenpai - AI Origami Assistant and 2 Test 10-09-26.mp4
   - ช่วง 0:13 - 0:24 (แล็ปท็อป, บัตรโรงเรียนศรีราชา, คีย์บอร์ด, โต๊ะ)
   - ช่วง 0:31 - 0:48 (กระดาษเปล่าสีขาว A4 ที่ยังไม่ได้พับ)
3. Background crops บริสุทธิ์จากคลิปดิบ Session 3 (พื้นห้องสีเทา), Session 4 (โต๊ะดำ), Session 5 (โต๊ะส้ม)
   - สกัดจากครึ่งล่างของเฟรมที่ไม่มีกระดาษพับ
   
* ห้ามดึงภาพจากโฟลเดอร์ Rejected เพราะใน Rejected มีภาพโอริกามิจริงที่ถูกฟิลเตอร์ blur คัดออกปนอยู่!

แบ่งเข้า Train (60%), Val (20%), Test (20%)
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

ROOT = Path(__file__).resolve().parents[1]
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
    """สกัดภาพพื้นผิวโต๊ะ/พื้นเปล่าจากคลิปดิบ Sessions 3, 4, 5 โดยครอปเฉพาะส่วนที่ไม่มีโอริกามิ"""
    extracted = []

    session_configs = [
        ("session_003_val_floor_night_roomlight", 0.60, 0.95, 0.10, 0.70, "raw_bg_floor"),
        ("session_004_train_blkwin_daylight", 0.60, 0.95, 0.10, 0.70, "raw_bg_blackdesk"),
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
                    extracted.append((bg_crop, f"{prefix}_{count:03d}_{v.stem}"))
                    count += 1
            cap.release()

        print(f"[INFO] Extracted {count} pure background crops from {sname}")

    return extracted


def main():
    print("=" * 60)
    print("Building 100% Pure Chk0_NonOrigami dataset")
    print("=" * 60)

    vdo1_matches = list(EXPERIMENTS_DIR.glob("*14-13-19.mp4"))
    vdo2_matches = list(EXPERIMENTS_DIR.glob("*Test 10-09-26.mp4"))
    vdo3_matches = list(EXPERIMENTS_DIR.glob("*After add Nogami class*.mp4"))

    all_samples = []

    # 1. วิดีโอที่ 1 (เป้สีดำ, โฟมเหลือง, เก้าอี้, ช่องว่างโต๊ะขาว)
    if vdo1_matches:
        vdo1_path = vdo1_matches[0]
        intervals_vdo1 = [
            (2.0, 22.0, "vdo1_backpack_chair_desk"),
        ]
        all_samples.extend(extract_clean_webcam_frames(vdo1_path, intervals_vdo1, step=3))

    # 2. วิดีโอที่ 2 (แล็ปท็อป, บัตรนักเรียน, คีย์บอร์ด, โต๊ะเปล่า)
    # * ห้ามรวมช่วงกระดาษขาว flat_white_paper เพื่อไม่ให้โมเดลจำสับสนกับ Step 1!
    if vdo2_matches:
        vdo2_path = vdo2_matches[0]
        intervals_vdo2 = [
            (13.0, 24.0, "vdo2_desk_laptop_card"),
        ]
        all_samples.extend(extract_clean_webcam_frames(vdo2_path, intervals_vdo2, step=3))

    # 3. วิดีโอที่ 3 (พื้นห้อง, สมุดโน้ตสีน้ำตาล, มือเปล่า, ขาตั้งกล้อง)
    if vdo3_matches:
        vdo3_path = vdo3_matches[0]
        intervals_vdo3 = [
            (2.0, 26.0, "vdo3_room_brown_booklet_hand"),
        ]
        all_samples.extend(extract_clean_webcam_frames(vdo3_path, intervals_vdo3, step=3))

    # 4. พื้นหลังโต๊ะ/พื้นเปล่าจาก raw videos
    bg_samples = extract_pure_backgrounds_from_raw_videos(limit_per_session=50)
    all_samples.extend(bg_samples)

    print(f"[INFO] Total pure raw negative samples collected: {len(all_samples)}")

    # 4. เพิ่ม Data Augmentation แบบง่าย (Horizontal Flip) เพื่อความหลากหลาย
    augmented = []
    for img, name in all_samples:
        augmented.append((img, name))
        flipped = cv2.flip(img, 1)
        augmented.append((flipped, f"{name}_flip"))

    random.shuffle(augmented)
    total = len(augmented)
    print(f"[INFO] Total samples after augmentation: {total}")

    # 5. ลบโฟลเดอร์ Chk0_NonOrigami เก่าทิ้งทั้งหมดเพื่อความสะอาด
    for split_name in ["train", "val", "test"]:
        out_dir = DATA_DIR / split_name / "Chk0_NonOrigami"
        safe_rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # 6. แบ่ง 60% Train, 20% Val, 20% Test
    n_train = int(total * 0.60)
    n_val = int(total * 0.20)

    splits = {
        "train": augmented[:n_train],
        "val": augmented[n_train : n_train + n_val],
        "test": augmented[n_train + n_val :],
    }

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

    print("[SUCCESS] Pure Chk0_NonOrigami dataset generated successfully!")


if __name__ == "__main__":
    main()
