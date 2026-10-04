"""
inference.py — Unified Step Verification and Inference Engine
==============================================================
Centralizes all real-time inference logic for Origami step classification:
- Camera hardware readiness assessment (Luma and Contrast gates)
- Foreground paper localization and cropping
- Dual-path inference (blended fusion between full-frame and foreground crop)
- Confidence and Top-Gap threshold gating
"""

import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from preprocessing import crop_to_foreground, tensor_transform
from step_config import STEP_CLASSES, STEP_NAMES, step_number


# Default gating thresholds
CAMERA_READY_MIN_LUMA = 40.0
CAMERA_READY_MIN_STD = 15.0
RECOGNITION_MIN_CONFIDENCE = 45.0
RECOGNITION_MIN_TOP_GAP = 8.0


def assess_frame_readiness(frame_bgr: np.ndarray) -> Tuple[bool, str, Dict[str, float]]:
    """
    Assess whether a camera frame is ready for inference.
    Rejects dark frames, empty frames, and low-contrast camera placeholder frames.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False, "empty_frame", {"mean_luma": 0.0, "contrast": 0.0, "dark_ratio": 1.0}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    mean_luma = float(np.mean(gray))
    contrast = float(np.std(gray))
    dark_ratio = float(np.mean(gray < 25))

    metrics = {
        "mean_luma": round(mean_luma, 2),
        "contrast": round(contrast, 2),
        "dark_ratio": round(dark_ratio, 3),
    }

    # Iriun and many virtual cameras output a mostly black placeholder before
    # the phone connects. Do not let the classifier guess from that screen.
    if mean_luma < 22 and dark_ratio > 0.70:
        return False, "camera_placeholder_or_too_dark", metrics

    if contrast < 6:
        return False, "frame_has_too_little_detail", metrics

    return True, "ok", metrics


@dataclass
class VerificationResult:
    """Structured result returned by StepVerifier."""
    frame_ready: bool
    recognized: bool
    recognition_reason: str
    prediction: str
    step: str
    step_number: Optional[int]
    step_name: str
    confidence: float
    top_gap: float
    recognition_thresholds: Dict[str, float]
    roi: Dict[str, Any]
    top_predictions: List[Dict[str, Any]]
    metrics: Optional[Dict[str, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary matching /api/predict response contract."""
        if not self.frame_ready:
            return {
                "frame_ready": False,
                "reason": self.recognition_reason,
                "metrics": self.metrics or {},
            }

        return {
            "frame_ready": self.frame_ready,
            "recognized": self.recognized,
            "recognition_reason": self.recognition_reason,
            "prediction": self.prediction,
            "step": self.step,
            "step_number": self.step_number,
            "step_name": self.step_name,
            "confidence": self.confidence,
            "top_gap": self.top_gap,
            "recognition_thresholds": self.recognition_thresholds,
            "roi": self.roi,
            "top_predictions": self.top_predictions,
        }


@dataclass
class Detection:
    """Normalized object detection candidate produced by any DetectorAdapter."""
    class_name: str
    confidence: float  # Score from 0.0 to 1.0 (or percentage)
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    cls_idx: Optional[int] = None


class DetectorAdapter:
    """Interface for pluggable model detector adapters."""
    adapter_name: str = "base_detector"
    fusion_mode: str = "native"

    def detect(self, frame_bgr: np.ndarray, imgsz: int = 1024) -> List[Detection]:
        raise NotImplementedError


class SyntheticDetectorAdapter(DetectorAdapter):
    """In-memory test double adapter for testing without ML models or GPU."""
    adapter_name = "synthetic"
    fusion_mode = "mock"

    def __init__(self, detections: Optional[List[Detection]] = None):
        self.detections = detections or []

    def detect(self, frame_bgr: np.ndarray, imgsz: int = 1024) -> List[Detection]:
        return self.detections


class ClassificationDetectorAdapter(DetectorAdapter):
    """Adapter for legacy MobileNetV2 dual-path classification model."""
    adapter_name = "dual_path_mobilenet"
    fusion_mode = "blended"

    def __init__(self, model: torch.nn.Module, device: Any, classes: List[str]):
        self.model = model
        self.device = device
        self.classes = classes
        self.model = self.model.to(self.device)
        self.model.eval()

    def detect(self, frame_bgr: np.ndarray, imgsz: int = 1024) -> List[Detection]:
        cropped_bgr, roi_meta = crop_to_foreground(frame_bgr)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cropped_rgb = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
        full_image = Image.fromarray(frame_rgb)
        cropped_image = Image.fromarray(cropped_rgb)

        original_tensor = tensor_transform(full_image).unsqueeze(0).to(self.device)
        cropped_tensor = tensor_transform(cropped_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            original_outputs = self.model(original_tensor)
            cropped_outputs = self.model(cropped_tensor)

            original_probabilities = F.softmax(original_outputs, dim=1)
            cropped_probabilities = F.softmax(cropped_outputs, dim=1)
            original_conf, original_idx = torch.max(original_probabilities, 1)
            cropped_conf, cropped_idx = torch.max(cropped_probabilities, 1)

            original_pred_class = self.classes[original_idx.item()]
            cropped_pred_class = self.classes[cropped_idx.item()]

            crop_weight = 0.0
            if roi_meta.get("applied"):
                if cropped_pred_class != "Chk0_NonOrigami":
                    crop_weight = 1.0
                    self.fusion_mode = "crop-priority"
                elif original_pred_class != "Chk0_NonOrigami":
                    crop_weight = 0.30
                    self.fusion_mode = "full-priority"
                else:
                    crop_weight = 0.50
                    self.fusion_mode = "blended"
            else:
                self.fusion_mode = "full-frame"

            outputs = ((1.0 - crop_weight) * original_outputs) + (crop_weight * cropped_outputs)
            if roi_meta.get("applied") and cropped_pred_class != "Chk0_NonOrigami":
                if "Chk0_NonOrigami" in self.classes:
                    chk0_idx = self.classes.index("Chk0_NonOrigami")
                    outputs[0, chk0_idx] = -100.0

            probabilities = F.softmax(outputs, dim=1)

            # Step 6 aspect ratio bias
            if roi_meta.get("applied") and roi_meta.get("bbox"):
                x1, y1, x2, y2 = roi_meta["bbox"]
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)
                aspect = bw / float(bh)
                chk5_idx = self.classes.index("Chk5") if "Chk5" in self.classes else -1
                chk6_idx = self.classes.index("Chk6") if "Chk6" in self.classes else -1
                if chk5_idx >= 0 and chk6_idx >= 0 and aspect > 1.25:
                    if probabilities[0, chk6_idx] > 0.15:
                        probabilities[0, chk6_idx] = probabilities[0, chk6_idx] * 1.35
                        probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)

        detections = []
        raw_bbox = tuple(roi_meta["bbox"]) if roi_meta.get("applied") and roi_meta.get("bbox") else None
        probs = probabilities[0].tolist()
        for idx, p in enumerate(probs):
            if idx < len(self.classes):
                detections.append(Detection(
                    class_name=self.classes[idx],
                    confidence=float(p),
                    bbox=raw_bbox,
                    cls_idx=idx,
                ))
        return detections


class StepVerifier:
    """
    Deep domain module for step prediction and verification.
    Owns the full pipeline from raw camera image to verified step decision behind a small interface.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        detector: Optional[DetectorAdapter] = None,
        device: Optional[Any] = None,
        classes: Tuple[str, ...] = STEP_CLASSES,
        min_confidence: float = RECOGNITION_MIN_CONFIDENCE,
        min_top_gap: float = RECOGNITION_MIN_TOP_GAP,
        step_names: Optional[Dict[str, str]] = None,
        conf_threshold: Optional[float] = None,
        imgsz: int = 1024,
    ):
        if conf_threshold is not None:
            self.min_confidence = round(conf_threshold * 100.0, 2) if conf_threshold <= 1.0 else round(conf_threshold, 2)
            self.conf_threshold = conf_threshold if conf_threshold <= 1.0 else conf_threshold / 100.0
        else:
            self.min_confidence = round(min_confidence, 2)
            self.conf_threshold = min_confidence / 100.0 if min_confidence > 1.0 else min_confidence

        self.min_top_gap = min_top_gap
        self.classes = list(classes)
        self.step_names = step_names or STEP_NAMES
        self.imgsz = imgsz
        self.detector = detector
        self.model = model
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.detector is None and self.model is not None:
            self.detector = ClassificationDetectorAdapter(
                model=self.model,
                device=self.device,
                classes=self.classes,
            )

    def verify_bgr(self, frame_bgr: np.ndarray) -> VerificationResult:
        """Verify an OpenCV BGR frame."""
        frame_ready, frame_reason, frame_metrics = assess_frame_readiness(frame_bgr)
        if not frame_ready:
            return VerificationResult(
                frame_ready=False,
                recognized=False,
                recognition_reason=frame_reason,
                prediction="",
                step="",
                step_number=None,
                step_name="",
                confidence=0.0,
                top_gap=0.0,
                recognition_thresholds={
                    "min_confidence": self.min_confidence,
                    "min_top_gap": self.min_top_gap,
                },
                roi={},
                top_predictions=[],
                metrics=frame_metrics,
            )

        if self.detector is None:
            raise RuntimeError("StepVerifier has no detector adapter configured.")

        detections = self.detector.detect(frame_bgr, imgsz=self.imgsz)
        h, w = frame_bgr.shape[:2]

        if not detections:
            return VerificationResult(
                frame_ready=True,
                recognized=False,
                recognition_reason="no_origami_detected",
                prediction="Chk0_NonOrigami",
                step="Chk0_NonOrigami",
                step_number=0,
                step_name=self.step_names.get("Chk0_NonOrigami", "Non-Origami / Background"),
                confidence=0.0,
                top_gap=0.0,
                recognition_thresholds={
                    "min_confidence": self.min_confidence,
                    "min_top_gap": self.min_top_gap,
                },
                roi={"applied": False, "method": getattr(self.detector, "adapter_name", "detector"), "bbox": [0, 0, w, h]},
                top_predictions=[],
                metrics=frame_metrics,
            )

        # Aggregate detections per class (highest confidence per unique class)
        class_max_conf: Dict[str, float] = {}
        class_best_det: Dict[str, Detection] = {}
        for d in detections:
            conf_val = float(d.confidence)
            if conf_val > class_max_conf.get(d.class_name, -1.0):
                class_max_conf[d.class_name] = conf_val
                class_best_det[d.class_name] = d

        sorted_classes = sorted(class_max_conf.items(), key=lambda item: item[1], reverse=True)
        best_cls_name, best_raw_conf = sorted_classes[0]
        best_det = class_best_det[best_cls_name]

        confidence_percent = round(best_raw_conf * 100.0 if best_raw_conf <= 1.0 else best_raw_conf, 2)

        if len(sorted_classes) >= 2:
            runner_up_raw = sorted_classes[1][1]
            runner_up_percent = round(runner_up_raw * 100.0 if runner_up_raw <= 1.0 else runner_up_raw, 2)
            top_gap = round(confidence_percent - runner_up_percent, 2)
        else:
            top_gap = confidence_percent

        top_predictions = []
        for c_name, conf_val in sorted_classes[:3]:
            try:
                num = step_number(c_name)
            except Exception:
                num = None
            conf_pct = round(conf_val * 100.0 if conf_val <= 1.0 else conf_val, 2)
            top_predictions.append({
                "class_name": c_name,
                "step_number": num,
                "step_name": self.step_names.get(c_name, ""),
                "confidence": conf_pct,
            })

        if best_cls_name == "Chk0_NonOrigami":
            recognized = False
            recognition_reason = "non_origami"
        else:
            recognized = (confidence_percent >= self.min_confidence and top_gap >= self.min_top_gap)
            if not recognized:
                if confidence_percent < self.min_confidence:
                    recognition_reason = "low_confidence"
                else:
                    recognition_reason = "ambiguous_step"
            else:
                recognition_reason = "ok"

        # ROI Bounding Box
        if best_det.bbox is not None and len(best_det.bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in best_det.bbox]
            x1, y1 = max(0, min(x1, w)), max(0, min(y1, h))
            x2, y2 = max(0, min(x2, w)), max(0, min(y2, h))
            bbox = [x1, y1, x2, y2]
        else:
            bbox = [0, 0, w, h]

        roi_data = {
            "applied": (best_cls_name != "Chk0_NonOrigami"),
            "method": getattr(self.detector, "adapter_name", "detector"),
            "bbox": bbox,
            "fusion_mode": getattr(self.detector, "fusion_mode", "native"),
            "confidence": confidence_percent,
        }

        try:
            primary_step_num = step_number(best_cls_name)
        except Exception:
            primary_step_num = None

        return VerificationResult(
            frame_ready=True,
            recognized=recognized,
            recognition_reason=recognition_reason,
            prediction=best_cls_name,
            step=best_cls_name,
            step_number=primary_step_num,
            step_name=self.step_names.get(best_cls_name, ""),
            confidence=confidence_percent,
            top_gap=top_gap,
            recognition_thresholds={
                "min_confidence": self.min_confidence,
                "min_top_gap": self.min_top_gap,
            },
            roi=roi_data,
            top_predictions=top_predictions,
            metrics=frame_metrics,
        )

    def verify_pil(self, image: Image.Image) -> VerificationResult:
        """Verify a PIL Image."""
        frame_rgb = np.array(image.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return self.verify_bgr(frame_bgr)

    def verify_bytes(self, contents: bytes) -> VerificationResult:
        """Verify raw image bytes (e.g. from HTTP upload)."""
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return self.verify_pil(image)
