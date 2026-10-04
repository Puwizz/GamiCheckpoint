import cv2
from pathlib import Path

captures_dir = Path(r"C:\Users\admin\Videos\Captures")
out_dir = Path("outputs/audit_user_tests")
out_dir.mkdir(parents=True, exist_ok=True)

videos = [
    "Test YOLO with color paper.mp4",
    "Test YOLO with white paper.mp4"
]

for vname in videos:
    vpath = captures_dir / vname
    if not vpath.exists():
        print(f"File not found: {vpath}")
        continue
    
    cap = cv2.VideoCapture(str(vpath))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps
    print(f"Video: {vname} | Total frames: {total_frames} | FPS: {fps:.1f} | Duration: {duration_sec:.1f}s")
    
    # Sample 6 evenly spaced timestamps across duration
    timestamps = [int(total_frames * r) for r in [0.15, 0.30, 0.45, 0.60, 0.75, 0.90]]
    v_stem = vpath.stem.replace(" ", "_")
    
    for idx, f_idx in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if ret and frame is not None:
            out_file = out_dir / f"{v_stem}_frame_{idx+1}.jpg"
            cv2.imwrite(str(out_file), frame)
            
    cap.release()
print("Extracted frames to outputs/audit_user_tests")
