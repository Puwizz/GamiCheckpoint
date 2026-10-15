import os
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[2]
CAPTURES_DIR = Path(os.getenv("GAMISENPAI_CAPTURES_DIR", ROOT / "data" / "captures"))
DEFAULT_VIDEO_PATH = Path(os.getenv("GAMISENPAI_COLOR_VIDEO_PATH", CAPTURES_DIR / "Test YOLO with color paper.mp4"))
DEFAULT_OUT_DIR = ROOT / "outputs" / "audit_user_tests" / "color_sequence"


def extract_color_sequence(vpath=DEFAULT_VIDEO_PATH, out_dir=DEFAULT_OUT_DIR, step_sec=4):
    vpath = Path(vpath)
    out_dir = Path(out_dir)
    if not vpath.exists():
        print(f"Video not found: {vpath}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(vpath))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 22.0
    step = max(1, int(fps * step_sec))

    extracted = 0
    for f_idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            sec = int(f_idx / fps)
            cv2.imwrite(str(out_dir / f"sec_{sec:02d}.jpg"), frame)
            extracted += 1

    cap.release()
    print(f"Extracted {extracted} frames from color video to {out_dir}")
    return extracted


if __name__ == "__main__":
    extract_color_sequence()
