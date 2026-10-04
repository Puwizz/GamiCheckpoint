"""
src/yolo_verifier.py — YOLO Object Detection Adapter for Step Verification
==========================================================================
Provides the YoloDetectorAdapter satisfying the internal DetectorAdapter seam,
and YoloStepVerifier subclassing the unified StepVerifier domain module.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from inference import (
    Detection,
    DetectorAdapter,
    StepVerifier,
    VerificationResult,
)
from step_config import STEP_NAMES, step_number


class YoloDetectorAdapter(DetectorAdapter):
    """Adapter wrapping Ultralytics YOLO model behind the DetectorAdapter seam."""
    adapter_name = "yolo_detection"
    fusion_mode = "yolo-native"

    def __init__(
        self,
        model_path: str,
        classes: Tuple[str, ...] = ("Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"),
        conf_threshold: float = 0.55,
        min_area_ratio: float = 0.015,
        device: Optional[str] = None,
        half: Optional[bool] = None,
    ):
        import torch
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.classes = list(classes)
        self.conf_threshold = conf_threshold
        self.min_area_ratio = min_area_ratio

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = str(device)

        # Enable FP16 half precision on CUDA GPU
        self.is_half = False
        if half is not False and self.device.startswith("cuda") and torch.cuda.is_available():
            try:
                self.model.to(self.device)
                self.model.half()
                self.is_half = True
            except Exception as e:
                print(f"[WARN] Failed to enable FP16 half precision on GPU: {e}")
                self.is_half = False

    def detect(self, frame_bgr: np.ndarray, imgsz: int = 640) -> List[Detection]:
        """Execute YOLO inference and map detections into canonical Detection dataclasses."""
        detect_conf = max(0.25, self.conf_threshold - 0.15)
        results = self.model.predict(
            frame_bgr,
            conf=detect_conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes if len(results) > 0 else None
        if boxes is None or len(boxes) == 0:
            return []

        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w) if (h * w) > 0 else 1.0

        detections = []
        for b in boxes:
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            # Reject tiny spurious noise (< min_area_ratio of camera frame)
            bw = max(0, xyxy[2] - xyxy[0])
            bh = max(0, xyxy[3] - xyxy[1])
            if (bw * bh) / frame_area < self.min_area_ratio:
                continue

            c_idx = int(b.cls[0])
            c_name = self.classes[c_idx] if c_idx < len(self.classes) else f"Chk{c_idx + 1}"
            c_conf = float(b.conf[0])
            detections.append(Detection(
                class_name=c_name,
                confidence=c_conf,
                bbox=tuple(xyxy),
                cls_idx=c_idx,
            ))
        return detections


class YoloStepVerifier(StepVerifier):
    """
    StepVerifier adapter using YOLO Object Detection.
    Inherits from the deep StepVerifier domain module, ensuring 100% backward
    compatibility with callers, unit test mocks, and the frontend HUD.
    """

    def __init__(
        self,
        model_path: str,
        classes: Tuple[str, ...] = ("Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"),
        conf_threshold: float = 0.55,
        min_top_gap: float = 8.0,
        min_area_ratio: float = 0.015,
        imgsz: int = 640,
        device: Optional[str] = None,
        half: Optional[bool] = None,
    ):
        adapter = YoloDetectorAdapter(
            model_path=model_path,
            classes=classes,
            conf_threshold=conf_threshold,
            min_area_ratio=min_area_ratio,
            device=device,
            half=half,
        )
        super().__init__(
            detector=adapter,
            classes=classes,
            conf_threshold=conf_threshold,
            min_top_gap=min_top_gap,
            imgsz=imgsz,
            device=adapter.device,
        )
        # Expose attributes expected by legacy callers and mock test fixtures
        self.model = adapter.model
        self.device = adapter.device
        self.is_half = adapter.is_half
        self.conf_threshold = conf_threshold
