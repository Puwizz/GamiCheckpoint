import os
from datetime import datetime
import cv2
import numpy as np


def draw_info_overlay(frame, mode_name, fps):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"Mode: {mode_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "[Q] Quit  [S] Save  [M] Switch Mode", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame


def process_original(frame):
    return frame.copy()


def process_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def process_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    result = np.zeros_like(frame)
    result[:, :, 1] = edges
    return result


def process_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = frame.copy()
    min_area = 500
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(result, f"Area: {area:.0f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    cv2.putText(result, f"Objects: {len(valid_contours)}", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    return result


def run_camera(device_index=0, max_frames=None):
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    modes = [
        ("Original", process_original),
        ("Grayscale", process_grayscale),
        ("Edges (Canny)", process_edges),
        ("Contours", process_contours),
    ]
    current_mode = 0
    screenshot_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    prev_time = cv2.getTickCount()
    fps = 0.0
    frames_processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_processed += 1
        current_time = cv2.getTickCount()
        time_diff = (current_time - prev_time) / cv2.getTickFrequency()
        if time_diff > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / time_diff)
        prev_time = current_time

        mode_name, process_fn = modes[current_mode]
        display = process_fn(frame)
        display = draw_info_overlay(display, mode_name, fps)

        cv2.imshow("Origami Checker AI - Camera Test", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('m'), ord('M')):
            current_mode = (current_mode + 1) % len(modes)
        elif key in (ord('s'), ord('S')):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(screenshot_dir, f"capture_{timestamp}.jpg")
            cv2.imwrite(filename, frame)

        if max_frames is not None and frames_processed >= max_frames:
            break

    cap.release()
    cv2.destroyAllWindows()
    return True


if __name__ == "__main__":
    run_camera()
