"""
tests/test_yolo_verifier.py
Unit tests for YoloStepVerifier: top_gap calculation, ambiguity gating, and FP16 configuration.
"""
import sys
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from src.yolo_verifier import YoloStepVerifier


@pytest.fixture
def mock_yolo(monkeypatch):
    mock_module = MagicMock()
    mock_yolo_cls = MagicMock()
    mock_module.YOLO = mock_yolo_cls
    monkeypatch.setitem(sys.modules, "ultralytics", mock_module)
    yield mock_yolo_cls


def create_bright_frame():
    np.random.seed(42)
    frame = np.random.randint(100, 200, (480, 640, 3), dtype=np.uint8)
    return frame


def test_yolo_verifier_device_cpu_init(mock_yolo):
    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu")
    assert verifier.device == "cpu"
    assert verifier.is_half is False


def test_yolo_verifier_no_detection(mock_yolo):
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu", conf_threshold=0.40)
    frame = create_bright_frame()

    res = verifier.verify_bgr(frame)
    assert res.frame_ready is True
    assert res.recognized is False
    assert res.recognition_reason == "no_origami_detected"
    assert res.prediction == "Chk0_NonOrigami"
    assert res.confidence == 0.0
    assert res.top_gap == 0.0
    assert len(res.top_predictions) == 0


def test_yolo_verifier_single_detection_recognized(mock_yolo):
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    mock_box = MagicMock()
    mock_box.conf = [torch.tensor(0.85)]
    mock_box.cls = [torch.tensor(2)]
    mock_box.xyxy = [torch.tensor([10.0, 20.0, 100.0, 120.0])]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu", conf_threshold=0.40, min_top_gap=8.0)
    frame = create_bright_frame()

    res = verifier.verify_bgr(frame)
    assert res.recognized is True
    assert res.recognition_reason == "ok"
    assert res.prediction == "Chk3"
    assert res.step_number == 3
    assert res.confidence == 85.0
    assert res.top_gap == 85.0
    assert len(res.top_predictions) == 1
    assert res.top_predictions[0]["class_name"] == "Chk3"
    assert res.roi["applied"] is True
    assert res.roi["bbox"] == [10, 20, 100, 120]


def test_yolo_verifier_real_top_gap_multiple_classes(mock_yolo):
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    mock_box1 = MagicMock()
    mock_box1.conf = [torch.tensor(0.80)]
    mock_box1.cls = [torch.tensor(2)]
    mock_box1.xyxy = [torch.tensor([10.0, 20.0, 100.0, 120.0])]

    mock_box2 = MagicMock()
    mock_box2.conf = [torch.tensor(0.45)]
    mock_box2.cls = [torch.tensor(1)]
    mock_box2.xyxy = [torch.tensor([12.0, 22.0, 95.0, 115.0])]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box1, mock_box2]
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu", conf_threshold=0.40, min_top_gap=8.0)
    frame = create_bright_frame()

    res = verifier.verify_bgr(frame)
    assert res.recognized is True
    assert res.recognition_reason == "ok"
    assert res.prediction == "Chk3"
    assert res.confidence == 80.0
    assert res.top_gap == 35.0
    assert len(res.top_predictions) == 2
    assert res.top_predictions[0]["class_name"] == "Chk3"
    assert res.top_predictions[1]["class_name"] == "Chk2"


def test_yolo_verifier_ambiguous_step_gating(mock_yolo):
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    mock_box1 = MagicMock()
    mock_box1.conf = [torch.tensor(0.52)]
    mock_box1.cls = [torch.tensor(2)]
    mock_box1.xyxy = [torch.tensor([10.0, 20.0, 100.0, 120.0])]

    mock_box2 = MagicMock()
    mock_box2.conf = [torch.tensor(0.49)]
    mock_box2.cls = [torch.tensor(3)]
    mock_box2.xyxy = [torch.tensor([15.0, 25.0, 105.0, 125.0])]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box1, mock_box2]
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu", conf_threshold=0.40, min_top_gap=8.0)
    frame = create_bright_frame()

    res = verifier.verify_bgr(frame)
    assert res.recognized is False
    assert res.recognition_reason == "ambiguous_step"
    assert res.prediction == "Chk3"
    assert res.confidence == 52.0
    assert res.top_gap == 3.0


def test_yolo_verifier_low_confidence_gating(mock_yolo):
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    mock_box = MagicMock()
    mock_box.conf = [torch.tensor(0.25)]
    mock_box.cls = [torch.tensor(0)]
    mock_box.xyxy = [torch.tensor([10.0, 20.0, 100.0, 120.0])]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu", conf_threshold=0.40)
    frame = create_bright_frame()

    res = verifier.verify_bgr(frame)
    assert res.recognized is False
    assert res.recognition_reason == "low_confidence"
    assert res.confidence == 25.0


def test_yolo_verifier_filters_out_spurious_noise_tiny_boxes(mock_yolo):
    """Verify that detections below min_area_ratio are filtered out as noise."""
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    # Tiny 5x5 box on 480x640 frame = 25 / 307200 = 0.000081 < 0.015 min_area_ratio
    tiny_box = MagicMock()
    tiny_box.conf = [torch.tensor(0.95)]
    tiny_box.cls = [torch.tensor(0)]
    tiny_box.xyxy = [torch.tensor([10.0, 10.0, 15.0, 15.0])]

    mock_result = MagicMock()
    mock_result.boxes = [tiny_box]
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu", min_area_ratio=0.015)
    frame = create_bright_frame()

    res = verifier.verify_bgr(frame)
    assert res.frame_ready is True
    assert res.recognized is False
    assert res.recognition_reason == "no_origami_detected"
    assert res.prediction == "Chk0_NonOrigami"


def test_yolo_verifier_custom_figure_classes(mock_yolo):
    """Verify that YoloStepVerifier properly maps custom figure classes (e.g. 4 frog steps)."""
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    custom_classes = ("Frog1", "Frog2", "Frog3", "Frog4")
    mock_box = MagicMock()
    mock_box.conf = [torch.tensor(0.90)]
    mock_box.cls = [torch.tensor(1)]  # Index 1 -> "Frog2"
    mock_box.xyxy = [torch.tensor([50.0, 50.0, 250.0, 250.0])]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", classes=custom_classes, device="cpu", conf_threshold=0.50)
    frame = create_bright_frame()

    res = verifier.verify_bgr(frame)
    assert res.frame_ready is True
    assert res.recognized is True
    assert res.prediction == "Frog2"
    assert res.confidence == 90.0
