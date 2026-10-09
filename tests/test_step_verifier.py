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


def test_assess_frame_readiness_handles_grayscale_and_rgba():
    from src.inference import assess_frame_readiness

    # 2D grayscale array
    gray_2d = np.full((100, 100), 128, dtype=np.uint8)
    ready, reason, _ = assess_frame_readiness(gray_2d)
    assert reason != "invalid_frame_format"

    # 3D 1-channel array
    gray_3d = np.full((100, 100, 1), 128, dtype=np.uint8)
    ready, reason, _ = assess_frame_readiness(gray_3d)
    assert reason != "invalid_frame_format"

    # 4-channel BGRA array
    bgra_4d = np.full((100, 100, 4), 128, dtype=np.uint8)
    ready, reason, _ = assess_frame_readiness(bgra_4d)
    assert reason != "invalid_frame_format"


def test_step_verifier_handles_grayscale_and_rgba_frames():
    adapter = SyntheticDetectorAdapter(detections=[
        Detection(class_name="Chk1", confidence=0.95, bbox=(10, 20, 50, 60))
    ])
    verifier = StepVerifier(
        detector=adapter,
        classes=("Chk1", "Chk2"),
    )

    # 2D textured grayscale frame
    gray_frame = np.random.randint(100, 200, (480, 640), dtype=np.uint8)
    res_gray = verifier.verify_bgr(gray_frame)
    assert res_gray.frame_ready is True
    assert res_gray.recognized is True
    assert res_gray.prediction == "Chk1"

    # 4-channel BGRA frame
    bgra_frame = np.random.randint(100, 200, (480, 640, 4), dtype=np.uint8)
    res_bgra = verifier.verify_bgr(bgra_frame)
    assert res_bgra.frame_ready is True
    assert res_bgra.recognized is True
    assert res_bgra.prediction == "Chk1"


def test_auto_crop_foreground_handles_rgba_and_grayscale():
    from PIL import Image
    from src.preprocessing import AutoCropForeground

    cropper = AutoCropForeground()

    # RGBA image
    rgba_img = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
    res_rgba = cropper(rgba_img)
    assert isinstance(res_rgba, Image.Image)
    assert res_rgba.mode == "RGB"

    # Grayscale 'L' image
    gray_img = Image.new("L", (100, 100), 200)
    res_gray = cropper(gray_img)
    assert isinstance(res_gray, Image.Image)
    assert res_gray.mode == "RGB"

    # Palette 'P' image
    palette_img = Image.new("P", (100, 100))
    res_palette = cropper(palette_img)
    assert isinstance(res_palette, Image.Image)
    assert res_palette.mode == "RGB"


def test_step_verifier_handles_nan_confidence_without_index_error():
    """CORE-10: Verify that NaN detection confidences do not cause IndexError: list index out of range."""
    adapter = SyntheticDetectorAdapter(detections=[
        Detection(class_name="Chk1", confidence=float("nan"), bbox=(10, 20, 100, 120))
    ])
    verifier = StepVerifier(
        detector=adapter,
        classes=("Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"),
    )
    frame = create_bright_frame()
    result = verifier.verify_bgr(frame)

    assert result.frame_ready is True
    assert result.recognized is False
    assert result.confidence == 0.0
    assert result.prediction == "Chk1"


def test_step_verifier_handles_mixed_nan_and_valid_confidences():
    """CORE-10: Verify that valid confidences win over NaN confidences."""
    adapter = SyntheticDetectorAdapter(detections=[
        Detection(class_name="Chk1", confidence=float("nan"), bbox=(10, 20, 100, 120)),
        Detection(class_name="Chk2", confidence=0.92, bbox=(15, 25, 95, 115)),
    ])
    verifier = StepVerifier(
        detector=adapter,
        classes=("Chk1", "Chk2", "Chk3", "Chk4", "Chk5", "Chk6"),
        conf_threshold=0.55,
    )
    frame = create_bright_frame()
    result = verifier.verify_bgr(frame)

    assert result.frame_ready is True
    assert result.recognized is True
    assert result.prediction == "Chk2"
    assert result.confidence == 92.0

