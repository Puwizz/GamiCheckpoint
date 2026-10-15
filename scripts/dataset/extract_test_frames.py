import os
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
CAPTURES_DIR = Path(os.getenv("GAMISENPAI_CAPTURES_DIR", ROOT / "data" / "captures"))
DEFAULT_OUT_DIR = ROOT / "outputs" / "audit_user_tests"
DEFAULT_VIDEOS = [
    "Test YOLO with color paper.mp4",
    "Test YOLO with white paper.mp4",
]


def extract_test_frames(captures_dir=CAPTURES_DIR, out_dir=DEFAULT_OUT_DIR, videos=None):
    captures_dir = Path(captures_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_list = videos if videos is not None else DEFAULT_VIDEOS

    total_extracted = 0
    for vname in video_list:
        vpath = captures_dir / vname
        if not vpath.exists():
            print(f"File not found: {vpath}")
            continue

        cap = cv2.VideoCapture(str(vpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / fps
        print(f"Video: {vname} | Total frames: {total_frames} | FPS: {fps:.1f} | Duration: {duration_sec:.1f}s")

        timestamps = [int(total_frames * r) for r in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]]
        v_stem = vpath.stem.replace(" ", "_")

        for idx, f_idx in enumerate(timestamps):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                out_file = out_dir / f"{v_stem}_frame_{idx+1}.jpg"
                cv2.imwrite(str(out_file), frame)
                total_extracted += 1

        cap.release()

    print(f"Extracted {total_extracted} frames to {out_dir}")
    return total_extracted


if __name__ == "__main__":
    extract_test_frames()
