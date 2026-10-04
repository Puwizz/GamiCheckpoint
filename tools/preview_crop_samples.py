import os
import sys
import glob
import cv2
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from preprocessing import crop_to_foreground

def main():
    os.makedirs("outputs/plots", exist_ok=True)
    steps = ["Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"]
    
    fig, axes = plt.subplots(6, 2, figsize=(11, 20))
    plt.subplots_adjust(hspace=0.35, wspace=0.15)
    
    for i, step in enumerate(steps):
        # Pick sample image from processed dataset (Test split or Val split)
        pattern = os.path.join("data", "step6", "five_session", "processed", "*", step, "*.jpg")
        matches = glob.glob(pattern)
        if not matches:
            continue
        
        # Pick a representative sample
        img_path = matches[len(matches) // 2]
        session_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        
        frame_bgr = cv2.imread(img_path)
        h, w = frame_bgr.shape[:2]
        
        cropped_bgr, meta = crop_to_foreground(frame_bgr)
        ch, cw = cropped_bgr.shape[:2]
        
        # Draw bbox on copy of original
        orig_with_box = frame_bgr.copy()
        if meta.get("applied"):
            x1, y1, x2, y2 = meta["bbox"]
            cv2.rectangle(orig_with_box, (x1, y1), (x2, y2), (0, 255, 0), max(4, int(w * 0.008)))
            label = f"ROI BBox ({meta.get('method')})"
            cv2.putText(orig_with_box, label, (x1, max(30, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        orig_rgb = cv2.cvtColor(orig_with_box, cv2.COLOR_BGR2RGB)
        crop_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        
        # Left column: Original frame with BBox
        axes[i, 0].imshow(orig_rgb)
        axes[i, 0].set_title(f"{step}: Original ({w}x{h})\n[Session: {session_name}]", fontsize=10, fontweight="bold")
        axes[i, 0].axis("off")
        
        # Right column: Cropped frame that the model receives
        axes[i, 1].imshow(crop_rgb)
        method_str = meta.get("method", "none")
        axes[i, 1].set_title(f"{step}: Cropped ({cw}x{ch})\n[Method: {method_str}]", fontsize=10, fontweight="bold", color="darkgreen")
        axes[i, 1].axis("off")

    out_path = os.path.join("outputs", "plots", "crop_verification_preview.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"[SUCCESS] Saved comparison grid to {out_path}")

if __name__ == "__main__":
    main()
