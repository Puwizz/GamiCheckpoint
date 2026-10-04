"""
tests/test_step_verifier.py
Unit tests for the deep StepVerifier domain module using synthetic adapters (zero GPU/Torch dependency).
"""
import numpy as np
import pytest
from src.inference import (
    StepVerifier,
    Detection,
    SyntheticDetectorAdapter,
    VerificationResult,
)


def create_bright_frame():
    """Create a realistic bright, textured test frame to pass assess_frame_readiness."""
    np.random.seed(42)
    return np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)


def test_step_verifier_with_synthetic_adapter_recognized():
    adapter = SyntheticDetectorAdapter(detections=[
        Detection(class_name="Chk3", confidence=0.88, bbox=(10, 20, 100, 120))
    ])
    verifier = StepVerifier(
        detector=adapter,
        classes=("Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"),
        conf_threshold=0.55,
        min_top_gap=8.0,
    )
    frame = create_bright_frame()
    result = verifier.verify_bgr(frame)

    assert result.frame_ready is True
    assert result.recognized is True
    assert result.prediction == "Chk3"
    assert result.step_number == 3
    assert result.confidence == 88.0
    assert result.top_gap == 88.0
    assert result.roi["bbox"] == [10, 20, 100, 120]


def test_step_verifier_ambiguous_step_gating():
    # Runner up is too close (80% vs 75% -> gap 5% < min 8%)
    adapter = SyntheticDetectorAdapter(detections=[
        Detection(class_name="Chk4", confidence=0.80, bbox=(10, 20, 100, 120)),
        Detection(class_name="Chk3", confidence=0.75, bbox=(15, 25, 95, 115)),
    ])
    verifier = StepVerifier(
        detector=adapter,
        classes=("Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"),
        conf_threshold=0.55,
        min_top_gap=8.0,
    )
    frame = create_bright_frame()
    result = verifier.verify_bgr(frame)

    assert result.frame_ready is True
    assert result.recognized is False
    assert result.recognition_reason == "ambiguous_step"
    assert result.prediction == "Chk4"
    assert result.confidence == 80.0
    assert result.top_gap == 5.0


def test_step_verifier_dark_frame_rejection():
    # Adapter should NOT even be called if frame is black
    adapter = SyntheticDetectorAdapter(detections=[
        Detection(class_name="Chk1", confidence=0.99)
    ])
    verifier = StepVerifier(
        detector=adapter,
        classes=("Chk1", "Chk2"),
    )
    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = verifier.verify_bgr(black_frame)

    assert result.frame_ready is False
    assert result.recognized is False
    assert result.recognition_reason == "camera_placeholder_or_too_dark"
