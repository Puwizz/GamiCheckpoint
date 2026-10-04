import cv2
from pathlib import Path

vpath = Path(r"C:\Users\admin\Videos\Captures\Test YOLO with color paper.mp4")
out_dir = Path("outputs/audit_user_tests/color_sequence")
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(vpath))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 22.0

# Extract 1 frame every 4 seconds (approx every 90 frames)
for f_idx in range(0, total, int(fps * 4)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if ret:
        sec = int(f_idx / fps)
        cv2.imwrite(str(out_dir / f"sec_{sec:02d}.jpg"), frame)
cap.release()
print("Extracted sequence from color video")
