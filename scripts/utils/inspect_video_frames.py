import cv2
from pathlib import Path
import sys
sys.path.append('src')
from yolo_verifier import YoloStepVerifier

verifier = YoloStepVerifier('checkpoints/yolo_runs/origami_yolo_run1/weights/best.pt', conf_threshold=0.15)
img_paths = [
    Path("outputs/audit_user_tests/color_sequence/sec_19.jpg"),
    Path("outputs/audit_user_tests/color_sequence/sec_55.jpg"),
    Path("outputs/audit_user_tests/Test_YOLO_with_white_paper_frame_4.jpg")
]

for p in img_paths:
    if not p.exists():
        continue
    img = cv2.imread(str(p))
    # Crop the video feed area on the left of screen
    # Video preview box is roughly between y=[230:750], x=[100:520]
    h, w = img.shape[:2]
    # Let's run on full screenshot first to see where YOLO predicts
    res_raw = verifier.model.predict(img, conf=0.15, verbose=False)[0]
    print(f"--- Predictions for {p.name} (full frame) ---")
    for box in res_raw.boxes:
        cls_id = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        print(f"  Class: {verifier.classes[cls_id]} | Conf: {conf:.2f} | BBox: {[int(x) for x in box.xyxy[0].tolist()]}")
