import random
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ["Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"]
ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "data" / "yolo_dataset" / "images" / "train"
LBL_DIR = ROOT / "data" / "yolo_dataset" / "labels" / "train"
OUT_DIR = ROOT / "outputs" / "plots" / "yolo_audit"


def draw_detection(img, line, class_names=CLASS_NAMES):
    h, w = img.shape[:2]
    parts = line.strip().split() if isinstance(line, str) else line
    if not parts:
        return img, "Background"

    try:
        cls_id = int(parts[0])
    except ValueError:
        return img, "Unknown"

    cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)

    if len(parts) == 5:
        try:
            cx, cy, nw, nh = map(float, parts[1:])
        except ValueError:
            return img, cls_name
        bw, bh = int(nw * w), int(nh * h)
        x1 = int((cx * w) - (bw / 2))
        y1 = int((cy * h) - (bh / 2))
        x2 = x1 + bw
        y2 = y1 + bh
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(2, int(w / 350)))
        cv2.putText(img, cls_name, (x1, max(35, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)
    elif len(parts) > 5:
        try:
            raw_pts = list(map(float, parts[1:]))
            coords = np.array(raw_pts, dtype=np.float32).reshape(-1, 2)
        except ValueError:
            return img, cls_name
        coords[:, 0] *= w
        coords[:, 1] *= h
        pts = coords.astype(np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=max(2, int(w / 350)))
        if len(pts) > 0:
            cv2.putText(img, cls_name, (int(pts[0][0]), max(35, int(pts[0][1]) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

    return img, cls_name


def main(img_dir=IMG_DIR, lbl_dir=LBL_DIR, out_dir=OUT_DIR, class_names=CLASS_NAMES):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    axes = axes.flatten()
    has_rendered = False

    for idx, target_cls_id in enumerate(range(min(6, len(class_names)))):
        matching_lbls = []
        for p in lbl_dir.glob("*.txt"):
            with open(p, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if first_line.startswith(str(target_cls_id) + " "):
                    matching_lbls.append(p)
                    if len(matching_lbls) > 10:
                        break

        if not matching_lbls:
            axes[idx].axis("off")
            continue

        lbl_path = random.choice(matching_lbls)
        img_name = lbl_path.stem + ".jpg"
        img_path = img_dir / img_name
        if not img_path.exists():
            axes[idx].axis("off")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            axes[idx].axis("off")
            continue

        with open(lbl_path, "r", encoding="utf-8") as f:
            first_line = f.readline()

        img, cls_name = draw_detection(img, first_line, class_names=class_names)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[idx].imshow(img_rgb)
        axes[idx].set_title(f"{cls_name} ({img_name[:20]}...)")
        axes[idx].axis("off")
        has_rendered = True

    if not has_rendered:
        plt.close(fig)
        return None

    plt.tight_layout()
    out_plot = out_dir / "per_class_audit.png"
    plt.savefig(str(out_plot), dpi=150)
    plt.close(fig)
    print(f"Per-class audit visualization saved to: {out_plot}")
    return out_plot


if __name__ == "__main__":
    main()
