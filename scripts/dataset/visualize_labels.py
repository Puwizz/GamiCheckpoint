"""
visualize_labels.py — Render sample bounding boxes on images to inspect auto-labels.
"""

import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

CLASS_NAMES = ["Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"]
ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "yolo_dataset" / "images" / "train"
LBL_DIR = ROOT / "data" / "yolo_dataset" / "labels" / "train"
OUT_DIR = ROOT / "outputs" / "plots" / "yolo_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Pick 6 random images
lbl_files = list(LBL_DIR.glob("*.txt"))
random.seed(42)
samples = random.sample(lbl_files, min(6, len(lbl_files)))

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, lbl_path in enumerate(samples):
    img_name = lbl_path.stem + ".jpg"
    img_path = IMG_DIR / img_name
    if not img_path.exists():
        continue
    
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    with open(lbl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            cls_id, cx, cy, nw, nh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            bw = int(nw * w)
            bh = int(nh * h)
            x1 = int((cx * w) - (bw / 2))
            y1 = int((cy * h) - (bh / 2))
            x2 = x1 + bw
            y2 = y1 + bh
            
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(2, int(w / 400)))
            cv2.putText(img, cls_name, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[idx].imshow(img_rgb)
    axes[idx].set_title(f"{img_name[:25]}... ({cls_name})")
    axes[idx].axis("off")

plt.tight_layout()
out_plot = OUT_DIR / "sample_boxes_audit.png"
plt.savefig(str(out_plot), dpi=150)
plt.close()
print(f"Sample audit visualization saved to: {out_plot}")
