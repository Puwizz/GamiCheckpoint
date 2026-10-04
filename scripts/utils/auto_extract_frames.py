import cv2
import os
import argparse

ROTATION_MAP = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def extract_frames(video_path, output_dir, step_name, extract_interval=0.3):
    """Extract frames from video every `extract_interval` seconds."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_interval = int(round(fps * extract_interval))
    if frame_interval <= 0:
        frame_interval = 1

    orientation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META))
    rotate_code = ROTATION_MAP.get(orientation)

    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    extracted_count = 0

    print(f"[INFO] Processing {video_basename} (Interval: {extract_interval}s, Frame step: {frame_interval})")

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            ret, frame = cap.retrieve()
            if not ret or frame is None:
                frame_count += 1
                continue

            if rotate_code is not None:
                frame = cv2.rotate(frame, rotate_code)

            time_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            save_path = os.path.join(output_dir, f"{video_basename}_f{frame_count:06d}_time{time_ms:06d}.jpg")

            cv2.imwrite(save_path, frame)
            extracted_count += 1

        frame_count += 1

    cap.release()
    print(f"[SUCCESS] Extracted {extracted_count} frames to {output_dir}")
    return extracted_count


def main():
    parser = argparse.ArgumentParser(description="Auto extract frames from raw_data_video folders.")
    parser.add_argument("--session-dir", required=True, help="Path to session directory (e.g., data/sessions/s3)")
    parser.add_argument("--interval", type=float, default=0.3, help="Extract frame every X seconds (default: 0.3)")
    args = parser.parse_args()

    session_dir = args.session_dir
    candidates = ["raw_data_video", "raw_data_videos", "raw_date_videos"]
    raw_dir = None
    for cand in candidates:
        cand_path = os.path.join(session_dir, cand)
        if os.path.exists(cand_path):
            raw_dir = cand_path
            break

    if not raw_dir:
        print(f"[ERROR] raw_data_video directory not found in {session_dir}")
        return

    total_extracted = 0

    valid_steps = {"Step1", "Step2", "Step3", "Step4", "Step5", "Step6"}
    video_extensions = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

    for root, dirs, files in os.walk(raw_dir):
        folder_name = os.path.basename(root)
        if folder_name in valid_steps:
            output_step_path = os.path.join(session_dir, folder_name)
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    video_path = os.path.join(root, file)
                    total_extracted += extract_frames(video_path, output_step_path, folder_name, args.interval)

    print(f"\n[DONE] Total frames extracted: {total_extracted}")


if __name__ == "__main__":
    main()

