"""
test_camera.py — ทดสอบกล้อง + OpenCV พื้นฐาน
==============================================
สคริปต์นี้ใช้ทดสอบว่า:
  1. กล้อง (Webcam) ทำงานได้
  2. OpenCV ติดตั้งถูกต้อง
  3. การประมวลผลภาพเบื้องต้นทำงานได้

วิธีใช้:
  python src/test_camera.py

กดปุ่ม:
  q — ออกจากโปรแกรม
  s — บันทึกภาพปัจจุบัน
  m — สลับโหมด (Original / Grayscale / Edges / Contours)
"""

import cv2
import numpy as np
import os
from datetime import datetime


def draw_info_overlay(frame, mode_name, fps):
    """วาด overlay แสดงข้อมูลบนภาพ"""
    h, w = frame.shape[:2]

    # พื้นหลังโปร่งแสงด้านบน
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # ข้อความ
    cv2.putText(frame, f"Mode: {mode_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

    # คำสั่งด้านล่าง
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "[Q] Quit  [S] Save  [M] Switch Mode", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def process_original(frame):
    """โหมด Original — แสดงภาพดิบจากกล้อง"""
    return frame.copy()


def process_grayscale(frame):
    """โหมด Grayscale — แปลงเป็นภาพขาวดำ"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def process_edges(frame):
    """โหมด Edges — ตรวจจับขอบด้วย Canny"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    # แสดงเป็นสีเขียวบนพื้นดำ
    result = np.zeros_like(frame)
    result[:, :, 1] = edges  # Green channel
    return result


def process_contours(frame):
    """โหมด Contours — ตรวจจับรูปร่างของวัตถุ"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # หา contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    result = frame.copy()

    # กรองเอาเฉพาะ contours ที่มีขนาดพอสมควร
    min_area = 500
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # วาด contour
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

            # วาด bounding box
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # แสดงพื้นที่
            cv2.putText(result, f"Area: {area:.0f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # แสดงจำนวน contours ที่พบ
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    cv2.putText(result, f"Objects: {len(valid_contours)}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    return result


def main():
    print("=" * 50)
    print("  Origami Checker AI — Camera Test")
    print("=" * 50)
    print()

    # เปิดกล้อง
    print("[INFO] Opening camera...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        print("  - Check if your webcam is connected")
        print("  - Check if another app is using the camera")
        return

    # ตั้งค่ากล้อง
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened: {actual_w}x{actual_h}")
    print()
    print("[CONTROLS]")
    print("  Q — Quit")
    print("  S — Save screenshot")
    print("  M — Switch mode (Original / Grayscale / Edges / Contours)")
    print()

    # โหมดการแสดงผล
    modes = [
        ("Original", process_original),
        ("Grayscale", process_grayscale),
        ("Edges (Canny)", process_edges),
        ("Contours", process_contours),
    ]
    current_mode = 0

    # สร้างโฟลเดอร์เก็บ screenshots
    screenshot_dir = os.path.join(os.path.dirname(__file__), "..", "data", "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    # FPS tracking
    prev_time = cv2.getTickCount()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame!")
            break

        # คำนวณ FPS
        current_time = cv2.getTickCount()
        time_diff = (current_time - prev_time) / cv2.getTickFrequency()
        if time_diff > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / time_diff)  # Smoothed FPS
        prev_time = current_time

        # ประมวลผลตามโหมดที่เลือก
        mode_name, process_fn = modes[current_mode]
        display = process_fn(frame)

        # วาด overlay
        display = draw_info_overlay(display, mode_name, fps)

        # แสดงผล
        cv2.imshow("Origami Checker AI - Camera Test", display)

        # จัดการ keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("[INFO] Quitting...")
            break
        elif key == ord('m') or key == ord('M'):
            current_mode = (current_mode + 1) % len(modes)
            print(f"[MODE] Switched to: {modes[current_mode][0]}")
        elif key == ord('s') or key == ord('S'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshot_dir, f"capture_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[SAVE] Screenshot saved: {filename}")

    # ปิดกล้อง
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released. Done!")


if __name__ == "__main__":
    main()
