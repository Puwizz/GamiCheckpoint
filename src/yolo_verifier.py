"""
src/yolo_verifier.py — YOLO Object Detection Adapter for Step Verification
==========================================================================
Provides the YoloDetectorAdapter satisfying the internal DetectorAdapter seam,
and YoloStepVerifier subclassing the unified StepVerifier domain module.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

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

    @staticmethod
    def _refine_polygon(
        raw_poly: Optional[Any],
        mask_tensor: Optional[Any],
        frame_bgr: np.ndarray,
        bbox: List[int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Refines a segmentation polygon:
        1. When mask tensor data is present, applies skin-tone subtraction (to isolate origami
           from holding hands) and morphological closing, extracting the dominant external contour.
        2. Applies polygon smoothing (Douglas-Peucker approxPolyDP) to remove micro-staircase
           jaggedness from prototype mask upsampling, outputting clean, crisp polygonal vertices.
        """
        h, w = frame_bgr.shape[:2]

        # Case 1: Mask tensor data available (from YOLO model with segmentation head)
        if mask_tensor is not None and (
            isinstance(mask_tensor, np.ndarray) or (torch is not None and torch.is_tensor(mask_tensor))
        ):
            try:
                if torch is not None and torch.is_tensor(mask_tensor):
                    m_arr = mask_tensor.detach().cpu().numpy()
                else:
                    m_arr = np.asarray(mask_tensor)

                if m_arr.ndim == 3:
                    m_arr = m_arr[0]
                m_arr = (m_arr > 0.5).astype(np.uint8) * 255

                if m_arr.shape != (h, w):
                    m_arr = cv2.resize(m_arr, (w, h), interpolation=cv2.INTER_NEAREST)

                bx1, by1, bx2, by2 = bbox
                margin = 25
                rx1, ry1 = max(0, bx1 - margin), max(0, by1 - margin)
                rx2, ry2 = min(w, bx2 + margin), min(h, by2 + margin)

                # Skin detection in YCrCb color space
                ycrcb = cv2.cvtColor(frame_bgr[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2YCrCb)
                skin_roi = cv2.inRange(ycrcb, np.array([0, 133, 77]), np.array([255, 173, 127]))

                m_roi = m_arr[ry1:ry2, rx1:rx2]
                erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                core = cv2.erode(m_roi, erode_k)
                if cv2.countNonZero(core) > 50:
                    mean_skin = cv2.mean(skin_roi, mask=core)[0]
                else:
                    mean_skin = cv2.mean(skin_roi, mask=m_roi)[0]

                # If paper itself is not skin-colored (less than ~60% skin pixels in core), subtract holding hand
                if mean_skin < 160:
                    m_arr[ry1:ry2, rx1:rx2] = cv2.bitwise_and(m_roi, cv2.bitwise_not(skin_roi))

                # Morphological closing to seal small crease gaps between wings/facets
                close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                m_arr = cv2.morphologyEx(m_arr, cv2.MORPH_CLOSE, close_k)

                # Extract external contour of dominant component (eliminates criss-crossing lines)
                cnts, _ = cv2.findContours(m_arr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts:
                    main_cnt = max(cnts, key=cv2.contourArea)
                    if cv2.contourArea(main_cnt) >= 50:
                        peri = cv2.arcLength(main_cnt, True)
                        eps = max(1.5, 0.006 * peri)
                        approx = cv2.approxPolyDP(main_cnt, eps, True)
                        if len(approx) >= 3:
                            return [
                                (int(round(float(p[0][0]))), int(round(float(p[0][1]))))
                                for p in approx
                            ]
            except Exception:
                pass

        # Case 2: Fallback to raw polygon coordinates (e.g. from masks.xy or mock test fixtures)
        if raw_poly is not None and len(raw_poly) >= 3:
            pts = np.array(raw_poly, dtype=np.float32).reshape((-1, 1, 2))
            if len(raw_poly) > 10:
                peri = cv2.arcLength(pts, True)
                eps = max(1.5, 0.006 * peri)
                approx = cv2.approxPolyDP(pts, eps, True)
                if len(approx) >= 3:
                    return [
                        (int(round(float(p[0][0]))), int(round(float(p[0][1]))))
                        for p in approx
                    ]
            return [
                (int(round(float(pt[0]))), int(round(float(pt[1]))))
                for pt in raw_poly
            ]

        return None

    def detect(self, frame_bgr: np.ndarray, imgsz: int = 640) -> List[Detection]:
        """Execute YOLO inference and map detections into canonical Detection dataclasses."""
        detect_conf = max(0.25, self.conf_threshold - 0.15)
        results = self.model.predict(
            frame_bgr,
            conf=detect_conf,
            imgsz=imgsz,
            device=self.device,
            half=self.is_half,
            retina_masks=True,
            verbose=False,
        )
        boxes = results[0].boxes if len(results) > 0 else None
        if boxes is None or len(boxes) == 0:
            return []

        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w) if (h * w) > 0 else 1.0

        masks = getattr(results[0], "masks", None)
        masks_xy = getattr(masks, "xy", None) if masks is not None else None
        masks_data = getattr(masks, "data", None) if masks is not None else None

        detections = []
        for idx, b in enumerate(boxes):
            xyxy = [int(v) for v in b.xyxy[0].tolist()]
            # Reject tiny spurious noise (< min_area_ratio of camera frame)
            bw = max(0, xyxy[2] - xyxy[0])
            bh = max(0, xyxy[3] - xyxy[1])
            if (bw * bh) / frame_area < self.min_area_ratio:
                continue

            c_idx = int(b.cls[0])
            c_name = self.classes[c_idx] if c_idx < len(self.classes) else f"Chk{c_idx + 1}"
            c_conf = float(b.conf[0])

            raw_poly = masks_xy[idx] if (masks_xy is not None and idx < len(masks_xy)) else None
            mask_tensor = masks_data[idx] if (masks_data is not None and idx < len(masks_data)) else None
            poly_pts = self._refine_polygon(raw_poly, mask_tensor, frame_bgr, xyxy)

            detections.append(Detection(
                class_name=c_name,
                confidence=c_conf,
                bbox=tuple(xyxy),
                cls_idx=c_idx,
                polygon=poly_pts,
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
