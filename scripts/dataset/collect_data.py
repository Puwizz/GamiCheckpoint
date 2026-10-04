"""
data_collection.py — เก็บข้อมูลภาพ Origami จากกล้อง
=====================================================
สคริปต์นี้ช่วยเก็บภาพจากกล้องสำหรับสร้าง dataset
เก็บภาพทีละขั้นตอน โดยแยกเป็น correct / incorrect

วิธีใช้:
  python src/data_collection.py

กดปุ่ม:
  c — ถ่ายภาพ (capture)
  n — ไปขั้นตอนถัดไป
  p — กลับขั้นตอนก่อนหน้า
  t — สลับ correct/incorrect
  q — ออก
"""

import cv2
import os
from datetime import datetime


class DataCollector:
    """เครื่องมือเก็บข้อมูลภาพ Origami"""

    def __init__(self, origami_name="crane", total_steps=6, save_dir=None):
        self.origami_name = origami_name
        self.total_steps = total_steps
        self.current_step = 1
        self.is_correct = True  # True = correct, False = incorrect
        self.capture_count = 0

        # ตั้ง directory เก็บข้อมูล
        if save_dir is None:
            base_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
        else:
            base_dir = save_dir

        self.base_dir = os.path.join(base_dir, origami_name)
        self._create_directories()

    def _create_directories(self):
        """สร้างโฟลเดอร์สำหรับทุกขั้นตอน"""
        for step in range(1, self.total_steps + 1):
            for label in ["correct", "incorrect"]:
                path = os.path.join(
                    self.base_dir,
                    f"step_{step:02d}",
                    label
                )
                os.makedirs(path, exist_ok=True)
        print(f"[INFO] Directories created at: {self.base_dir}")

    def get_save_path(self):
        """สร้าง path สำหรับบันทึกภาพ"""
        label = "correct" if self.is_correct else "incorrect"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{self.origami_name}_{timestamp}.jpg"
        return os.path.join(
            self.base_dir,
            f"step_{self.current_step:02d}",
            label,
            filename
        )

    def get_current_count(self):
        """นับจำนวนภาพที่เก็บในขั้นตอนปัจจุบัน"""
        label = "correct" if self.is_correct else "incorrect"
        path = os.path.join(
            self.base_dir,
            f"step_{self.current_step:02d}",
            label
        )
        if os.path.exists(path):
            return len([f for f in os.listdir(path) if f.endswith(('.jpg', '.png'))])
        return 0

    def draw_ui(self, frame):
        """วาด UI บนภาพ"""
        h, w = frame.shape[:2]
        display = frame.copy()

        # Header bar
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # ข้อมูล step
        step_names = {
            1: "Square Base",
            2: "Kite Folds",
            3: "Bird Base",
            4: "Neck & Tail Setup",
            5: "Head Fold",
            6: "Completed Crane"
        }
        step_name = step_names.get(self.current_step, "")
        cv2.putText(display, f"Step {self.current_step}/{self.total_steps}: {step_name}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Label (correct/incorrect)
        label = "CORRECT" if self.is_correct else "INCORRECT"
        color = (0, 255, 0) if self.is_correct else (0, 0, 255)
        cv2.putText(display, label, (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # จำนวนภาพ
        count = self.get_current_count()
        cv2.putText(display, f"Images: {count}", (w - 180, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Total captured
        cv2.putText(display, f"Total: {self.capture_count}", (w - 180, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Footer
        overlay2 = display.copy()
        cv2.rectangle(overlay2, (0, h - 35), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay2, 0.7, display, 0.3, 0, display)
        cv2.putText(display, "[C] Capture  [N/P] Step  [T] Toggle  [Q] Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        return display

    def run(self):
        """เริ่มเก็บข้อมูล"""
        print("=" * 50)
        print(f"  Data Collection: {self.origami_name}")
        print(f"  Total Steps: {self.total_steps}")
        print("=" * 50)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera!")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("[INFO] Camera ready. Start capturing!")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = self.draw_ui(frame)
            cv2.imshow("Data Collection", display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('c') or key == ord('C'):
                save_path = self.get_save_path()
                cv2.imwrite(save_path, frame)
                self.capture_count += 1
                label = "correct" if self.is_correct else "incorrect"
                print(f"[CAPTURED] Step {self.current_step} ({label}) "
                      f"-> {os.path.basename(save_path)}")
            elif key == ord('n') or key == ord('N'):
                if self.current_step < self.total_steps:
                    self.current_step += 1
                    print(f"[STEP] -> Step {self.current_step}")
            elif key == ord('p') or key == ord('P'):
                if self.current_step > 1:
                    self.current_step -= 1
                    print(f"[STEP] -> Step {self.current_step}")
            elif key == ord('t') or key == ord('T'):
                self.is_correct = not self.is_correct
                label = "correct" if self.is_correct else "incorrect"
                print(f"[LABEL] -> {label}")

        cap.release()
        cv2.destroyAllWindows()
        print(f"\n[DONE] Total images captured: {self.capture_count}")


def main():
    print("Origami Data Collection Tool")
    print("-" * 40)

    # ตั้งค่า — แก้ตรงนี้ตามที่ต้องการ
    origami_name = "crane"   # ชื่อ Origami (เช่น crane, boat, frog)
    total_steps = 6         # จำนวนขั้นตอนทั้งหมด

    collector = DataCollector(
        origami_name=origami_name,
        total_steps=total_steps
    )
    collector.run()


if __name__ == "__main__":
    main()
