"""
inference.py — Unified Step Verification and Inference Engine
==============================================================
Centralizes all real-time inference logic for Origami step classification:
- Camera hardware readiness assessment (Luma and Contrast gates)
- Confidence and Top-Gap threshold gating

NOTE: MobileNetV2 ClassificationDetectorAdapter has been removed.
The active verifier backend is YoloStepVerifier (see yolo_verifier.py).
Legacy MobileNet code is archived in scripts/legacy/.
"""

import io
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

from step_config import STEP_CLASSES, STEP_NAMES, step_number


# Default gating thresholds
CAMERA_READY_MIN_LUMA = 22.0
CAMERA_READY_DARK_RATIO = 0.70
CAMERA_READY_MIN_CONTRAST = 6.0
CAMERA_READY_DARK_PIXEL_MAX_LUMA = 25.0
RECOGNITION_MIN_CONFIDENCE = 45.0
RECOGNITION_MIN_TOP_GAP = 8.0


def assess_frame_readiness(frame_bgr: np.ndarray) -> Tuple[bool, str, Dict[str, float]]:
    """
    Assess whether a camera frame is ready for inference.
    Rejects dark frames, empty frames, and low-contrast camera placeholder frames.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return False, "empty_frame", {"mean_luma": 0.0, "contrast": 0.0, "dark_ratio": 1.0}

    if frame_bgr.ndim == 2:
        gray = frame_bgr
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 1:
        gray = frame_bgr.squeeze(2)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 3:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2GRAY)
    else:
        return False, "invalid_frame_format", {"mean_luma": 0.0, "contrast": 0.0, "dark_ratio": 1.0}

    mean_luma = float(np.mean(gray))
    contrast = float(np.std(gray))
    dark_ratio = float(np.mean(gray < CAMERA_READY_DARK_PIXEL_MAX_LUMA))

    metrics = {
        "mean_luma": round(mean_luma, 2),
        "contrast": round(contrast, 2),
        "dark_ratio": round(dark_ratio, 3),
    }

    if mean_luma < CAMERA_READY_MIN_LUMA and dark_ratio > CAMERA_READY_DARK_RATIO:
        return False, "camera_placeholder_or_too_dark", metrics

    if contrast < CAMERA_READY_MIN_CONTRAST:
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
    coaching_hint: Optional[str] = None
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
            "coaching_hint": self.coaching_hint,
        }


@dataclass
class Detection:
    """Normalized object detection candidate produced by any DetectorAdapter."""
    class_name: str
    confidence: float  # Score from 0.0 to 1.0 (or percentage)
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    cls_idx: Optional[int] = None
    polygon: Optional[List[Tuple[int, int]]] = None


class DetectorAdapter:
    """Interface for pluggable model detector adapters."""
    adapter_name: str = "base_detector"
    fusion_mode: str = "native"

    def detect(self, frame_bgr: np.ndarray, imgsz: int = 640) -> List[Detection]:
        raise NotImplementedError


class SyntheticDetectorAdapter(DetectorAdapter):
    """In-memory test double adapter for testing without ML models or GPU."""
    adapter_name = "synthetic"
    fusion_mode = "mock"

    def __init__(self, detections: Optional[List[Detection]] = None):
        self.detections = detections or []

    def detect(self, frame_bgr: np.ndarray, imgsz: int = 640) -> List[Detection]:
        return self.detections



class StepVerifier:
    """
    Deep domain module for step prediction and verification.
    Owns the full pipeline from raw camera image to verified step decision behind a small interface.
    """

    def __init__(
        self,
        detector: Optional[DetectorAdapter] = None,
        device: Optional[Any] = None,
        classes: Tuple[str, ...] = STEP_CLASSES,
        min_confidence: float = RECOGNITION_MIN_CONFIDENCE,
        min_top_gap: float = RECOGNITION_MIN_TOP_GAP,
        step_names: Optional[Dict[str, str]] = None,
        conf_threshold: Optional[float] = None,
        imgsz: int = 640,
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
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.detector is None:
            raise ValueError(
                "StepVerifier requires a DetectorAdapter. "
                "Pass detector=YoloStepVerifier(...) or detector=SyntheticDetectorAdapter(...)."
            )

    def build_coaching_hint(
        self,
        expected_step: Optional[int],
        recognized: bool,
        recognition_reason: str,
        primary_step_num: Optional[int],
        top_predictions: List[Dict[str, Any]],
    ) -> Optional[str]:
        if expected_step is None or expected_step <= 0:
            return None

        expected_name = self.step_names.get(f"Chk{expected_step}", f"Step {expected_step}")

        if recognition_reason in ("no_origami_detected", "non_origami"):
            return f"Place your origami paper clearly in the center of the camera to check Step {expected_step} ({expected_name})."

        if recognition_reason == "low_confidence":
            return f"Fold detected with low certainty. Hold the paper flat facing the camera under good lighting for Step {expected_step} ({expected_name})."

        if recognition_reason == "ambiguous_step":
            if len(top_predictions) >= 2:
                c1 = top_predictions[0].get("step_name") or top_predictions[0].get("class_name")
                c2 = top_predictions[1].get("step_name") or top_predictions[1].get("class_name")
                if c1 and c2:
                    return f"Your fold is halfway between {c1} and {c2}. Crease the edges firmly to match Step {expected_step} ({expected_name})."
            return f"Your fold looks ambiguous. Crease the edges firmly to match Step {expected_step} ({expected_name})."

        if recognized is True and primary_step_num is not None:
            if primary_step_num == expected_step:
                return f"Great! Step {expected_step} ({expected_name}) looks alignment-ready. Hold steady to confirm."
            if primary_step_num == expected_step - 1:
                prev_name = self.step_names.get(f"Chk{primary_step_num}", f"Step {primary_step_num}")
                return f"Still showing Step {primary_step_num} ({prev_name}). Perform the next fold to reach Step {expected_step} ({expected_name})."
            if primary_step_num < expected_step - 1:
                det_name = self.step_names.get(f"Chk{primary_step_num}", f"Step {primary_step_num}")
                return f"Detected earlier Step {primary_step_num} ({det_name}). Follow the guide to fold up to Step {expected_step} ({expected_name})."
            if primary_step_num > expected_step:
                det_name = self.step_names.get(f"Chk{primary_step_num}", f"Step {primary_step_num}")
                return f"Detected future Step {primary_step_num} ({det_name}), but Step {expected_step} ({expected_name}) must be verified first."

        return None

    def _no_origami_result(self, w: int, h: int, frame_metrics: Dict[str, float], expected_step: Optional[int] = None) -> VerificationResult:
        coaching = self.build_coaching_hint(
            expected_step=expected_step,
            recognized=False,
            recognition_reason="no_origami_detected",
            primary_step_num=0,
            top_predictions=[],
        )
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
            roi={"applied": False, "method": getattr(self.detector, "adapter_name", "detector"), "bbox": [0, 0, w, h], "polygon": None, "frame_size": [w, h]},
            top_predictions=[],
            coaching_hint=coaching,
            metrics=frame_metrics,
        )

    def verify_bgr(self, frame_bgr: np.ndarray, expected_step: Optional[int] = None) -> VerificationResult:
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
                coaching_hint=None,
                metrics=frame_metrics,
            )

        if self.detector is None:
            raise RuntimeError("StepVerifier has no detector adapter configured.")

        # Normalize 2D or 4-channel frames to 3-channel BGR for detector
        if frame_bgr.ndim == 2:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 1:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
        elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)

        detections = self.detector.detect(frame_bgr, imgsz=self.imgsz)
        h, w = frame_bgr.shape[:2]

        if not detections:
            return self._no_origami_result(w, h, frame_metrics, expected_step=expected_step)

        # Aggregate detections per class (highest confidence per unique class)
        class_max_conf: Dict[str, float] = {}
        class_best_det: Dict[str, Detection] = {}
        for d in detections:
            try:
                conf_val = float(d.confidence)
                if math.isnan(conf_val) or math.isinf(conf_val):
                    conf_val = 0.0
            except (TypeError, ValueError):
                conf_val = 0.0

            if conf_val > class_max_conf.get(d.class_name, -1.0):
                class_max_conf[d.class_name] = conf_val
                class_best_det[d.class_name] = d

        sorted_classes = sorted(class_max_conf.items(), key=lambda item: item[1], reverse=True)
        if not sorted_classes:
            return self._no_origami_result(w, h, frame_metrics)

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

        # ROI Polygon
        polygon = None
        if best_cls_name != "Chk0_NonOrigami" and getattr(best_det, "polygon", None):
            raw_poly = best_det.polygon
            if len(raw_poly) >= 3:
                polygon = [
                    [max(0, min(int(px), w)), max(0, min(int(py), h))]
                    for px, py in raw_poly
                ]

        roi_data = {
            "applied": (best_cls_name != "Chk0_NonOrigami"),
            "method": getattr(self.detector, "adapter_name", "detector"),
            "bbox": bbox,
            "polygon": polygon,
            "fusion_mode": getattr(self.detector, "fusion_mode", "native"),
            "confidence": confidence_percent,
            "frame_size": [w, h],
        }

        try:
            primary_step_num = step_number(best_cls_name)
        except Exception:
            primary_step_num = None

        coaching = self.build_coaching_hint(
            expected_step=expected_step,
            recognized=recognized,
            recognition_reason=recognition_reason,
            primary_step_num=primary_step_num,
            top_predictions=top_predictions,
        )

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
            coaching_hint=coaching,
            metrics=frame_metrics,
        )

    def verify_pil(self, image: Image.Image, expected_step: Optional[int] = None) -> VerificationResult:
        """Verify a PIL Image."""
        frame_rgb = np.array(image.convert("RGB"))
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return self.verify_bgr(frame_bgr, expected_step=expected_step)

    def verify_bytes(self, contents: bytes, expected_step: Optional[int] = None) -> VerificationResult:
        """Verify raw image bytes (e.g. from HTTP upload)."""
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return self.verify_pil(image, expected_step=expected_step)
