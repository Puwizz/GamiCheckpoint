from pathlib import Path
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"]
ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "yolo_dataset" / "images" / "train"
LBL_DIR = ROOT / "data" / "yolo_dataset" / "labels" / "train"
OUT_DIR = ROOT / "outputs" / "plots" / "yolo_audit"


def visualize_image_labels(img, lines, class_names=CLASS_NAMES):
    h, w = img.shape[:2]
    cls_name = "Background"
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

            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(2, int(w / 400)))
            cv2.putText(img, cls_name, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        elif len(parts) > 5:
            cls_id = int(parts[0])
            cls_name = class_names[cls_id] if cls_id < len(class_names) else str(cls_id)
            coords = np.array(list(map(float, parts[1:])), dtype=np.float32).reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            pts = coords.astype(np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=max(2, int(w / 400)))
            cv2.putText(img, cls_name, (int(pts[0][0]), max(30, int(pts[0][1]) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    return img, cls_name


def main(lbl_dir=LBL_DIR, img_dir=IMG_DIR, out_dir=OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    lbl_files = list(lbl_dir.glob("*.txt"))
    if not lbl_files:
        return None

    random.seed(42)
    samples = random.sample(lbl_files, min(6, len(lbl_files)))

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, lbl_path in enumerate(samples):
        img_name = lbl_path.stem + ".jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        annotated_img, cls_name = visualize_image_labels(img, lines)
        img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{img_name[:25]}... ({cls_name})")
        axes[idx].axis("off")

    plt.tight_layout()
    out_plot = out_dir / "sample_boxes_audit.png"
    plt.savefig(str(out_plot), dpi=150)
    plt.close()
    return out_plot


if __name__ == "__main__":
    main()
