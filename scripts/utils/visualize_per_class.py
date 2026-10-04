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

# Pick 1 sample per class Chk1 to Chk6
fig, axes = plt.subplots(2, 3, figsize=(16, 11))
axes = axes.flatten()

for idx, target_cls_id in enumerate(range(6)):
    matching_lbls = []
    for p in LBL_DIR.glob("*.txt"):
        with open(p, "r", encoding="utf-8") as f:
            first_line = f.readline()
            if first_line.startswith(str(target_cls_id) + " "):
                matching_lbls.append(p)
                if len(matching_lbls) > 10:
                    break
    
    if not matching_lbls:
        continue
    
    lbl_path = random.choice(matching_lbls)
    img_name = lbl_path.stem + ".jpg"
    img_path = IMG_DIR / img_name
    if not img_path.exists():
        continue
    
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    with open(lbl_path, "r", encoding="utf-8") as f:
        line = f.readline().strip().split()
        cls_id, cx, cy, nw, nh = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
        bw, bh = int(nw * w), int(nh * h)
        x1 = int((cx * w) - (bw / 2))
        y1 = int((cy * h) - (bh / 2))
        x2 = x1 + bw
        y2 = y1 + bh
        
        cls_name = CLASS_NAMES[cls_id]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(2, int(w / 350)))
        cv2.putText(img, cls_name, (x1, max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[idx].imshow(img_rgb)
    axes[idx].set_title(f"{cls_name} ({img_name[:20]}...)")
    axes[idx].axis("off")

plt.tight_layout()
out_plot = OUT_DIR / "per_class_audit.png"
plt.savefig(str(out_plot), dpi=150)
plt.close()
print(f"Per-class audit visualization saved to: {out_plot}")
