"""
inference.py — Real-time Origami Prediction
============================================
ใช้ model ที่ train แล้วทำนายขั้นตอน Origami จากกล้องแบบ real-time

วิธีใช้:
  python src/inference.py
"""

import os
import sys
from pathlib import Path

# Add project root and src to python path for imports
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch

from model import create_model
from preprocessing import inference_transform


class OrigamiPredictor:
    """ทำนายขั้นตอน Origami แบบ real-time"""

    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # โหลด model
        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint['classes']
        num_classes = len(self.classes)

        self.model = create_model(num_classes=num_classes, pretrained=False,
                                  freeze_backbone=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"[INFO] Model loaded: {len(self.classes)} classes")
        print(f"[INFO] Classes: {self.classes}")

    def predict(self, frame):
        """
        ทำนายจากภาพ 1 frame
        
        Returns:
            class_name: ชื่อ class ที่ทำนาย
            confidence: ความมั่นใจ (0-1)
            all_probs: probability ทุก class
        """
        input_tensor = inference_transform(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)

        class_name = self.classes[predicted.item()]
        return class_name, confidence.item(), probs[0].cpu().numpy()

    def draw_prediction(self, frame, class_name, confidence, all_probs=None):
        """วาดผลทำนายบนภาพ"""
        h, w = frame.shape[:2]
        display = frame.copy()

        # Header
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)

        # Prediction
        if confidence > 0.7:
            color = (0, 255, 0)    # เขียว = มั่นใจ
            status = "CORRECT"
        elif confidence > 0.5:
            color = (0, 255, 255)  # เหลือง = ไม่แน่ใจ
            status = "UNCERTAIN"
        else:
            color = (0, 0, 255)    # แดง = ไม่ตรง
            status = "CHECK AGAIN"

        cv2.putText(display, f"{class_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(display, f"Confidence: {confidence:.0%} — {status}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Confidence bar
        bar_w = int(confidence * (w - 20))
        cv2.rectangle(display, (10, h - 30), (10 + bar_w, h - 10), color, -1)
        cv2.rectangle(display, (10, h - 30), (w - 10, h - 10), (100, 100, 100), 1)

        return display


def main():
    # === Config ===
    CHECKPOINTS_DIR = os.path.join(ROOT, "checkpoints")
    model_path = os.path.join(CHECKPOINTS_DIR, "origami_step6_best.pth")

    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        print("  Train the model first with: python scripts/train.py")
        return

    predictor = OrigamiPredictor(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera!")
        return

    print("[INFO] Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        class_name, confidence, all_probs = predictor.predict(frame)
        display = predictor.draw_prediction(frame, class_name, confidence)

        cv2.imshow("Origami Checker AI", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
