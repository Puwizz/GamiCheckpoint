"""
tests/test_yolo_verifier.py
Unit tests for YoloStepVerifier: top_gap calculation, ambiguity gating, and FP16 configuration.
"""
import cv2
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


def test_yolo_verifier_extracts_and_clamps_segmentation_polygon(mock_yolo):
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    mock_box = MagicMock()
    mock_box.conf = [torch.tensor(0.88)]
    mock_box.cls = [torch.tensor(1)]  # Chk2
    mock_box.xyxy = [torch.tensor([50.0, 40.0, 200.0, 300.0])]

    mock_masks = MagicMock()
    mock_masks.xy = [
        np.array([[-5.2, 40.4], [200.0, 40.0], [700.0, 300.0], [50.0, 500.0]], dtype=np.float32)
    ]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.masks = mock_masks
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu", conf_threshold=0.40)
    frame = create_bright_frame()  # 480h x 640w

    res = verifier.verify_bgr(frame)
    assert res.recognized is True
    assert res.roi["polygon"] == [[0, 40], [200, 40], [640, 300], [50, 480]]


def test_yolo_verifier_handles_missing_or_degenerate_masks(mock_yolo):
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    mock_box = MagicMock()
    mock_box.conf = [torch.tensor(0.85)]
    mock_box.cls = [torch.tensor(0)]
    mock_box.xyxy = [torch.tensor([10.0, 20.0, 120.0, 140.0])]

    mock_masks = MagicMock()
    mock_masks.xy = [np.array([[10.0, 20.0], [120.0, 140.0]], dtype=np.float32)]  # < 3 points

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.masks = mock_masks
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu", conf_threshold=0.40)
    res = verifier.verify_bgr(create_bright_frame())
    assert res.roi["polygon"] is None

    mock_result.masks = None
    res_no_mask = verifier.verify_bgr(create_bright_frame())
    assert res_no_mask.roi["polygon"] is None


def test_yolo_verifier_passes_retina_masks_true(mock_yolo):
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu")
    verifier.verify_bgr(create_bright_frame())

    assert mock_model_instance.predict.called
    kwargs = mock_model_instance.predict.call_args.kwargs
    assert kwargs.get("retina_masks") is True


def test_yolo_verifier_smooths_jagged_polygon(mock_yolo):
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    mock_box = MagicMock()
    mock_box.conf = [torch.tensor(0.85)]
    mock_box.cls = [torch.tensor(1)]
    mock_box.xyxy = [torch.tensor([50.0, 50.0, 300.0, 300.0])]

    # Create a staircase jagged triangle with 60 points
    jagged_pts = []
    # Bottom edge (50, 300) to (300, 300)
    for x in range(50, 301, 10):
        jagged_pts.append([float(x), 300.0 + (1.0 if x % 20 == 0 else -1.0)])
    # Right edge (300, 300) to (175, 50)
    for i, t in enumerate(np.linspace(0, 1, 25)):
        px = 300.0 * (1 - t) + 175.0 * t + (1.5 if i % 2 == 0 else -1.5)
        py = 300.0 * (1 - t) + 50.0 * t
        jagged_pts.append([px, py])
    # Left edge (175, 50) to (50, 300)
    for i, t in enumerate(np.linspace(0, 1, 25)):
        px = 175.0 * (1 - t) + 50.0 * t + (-1.5 if i % 2 == 0 else 1.5)
        py = 50.0 * (1 - t) + 300.0 * t
        jagged_pts.append([px, py])

    mock_masks = MagicMock()
    mock_masks.xy = [np.array(jagged_pts, dtype=np.float32)]
    mock_masks.data = None

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.masks = mock_masks
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu", conf_threshold=0.40)
    res = verifier.verify_bgr(create_bright_frame())

    assert res.recognized is True
    assert res.roi["polygon"] is not None
    # Raw had ~76 points; smoothed approxPolyDP should have <= 15 points
    assert len(res.roi["polygon"]) <= 15
    assert len(res.roi["polygon"]) >= 3


def test_yolo_verifier_refines_mask_with_skin_exclusion(mock_yolo):
    mock_model_instance = MagicMock()
    mock_yolo.return_value = mock_model_instance

    # Frame 480x640: White paper square at [100:250, 100:250], Skin region at [250:350, 200:300]
    frame = np.full((480, 640, 3), 128, dtype=np.uint8)  # gray background
    # White paper: BGR=(250, 250, 250)
    frame[100:250, 100:250] = (250, 250, 250)
    # Skin-colored hand: YCrCb (Y=160, Cr=150, Cb=100) -> approx BGR
    # Cr=150, Cb=100 in inRange([0, 133, 77], [255, 173, 127])
    skin_bgr = cv2.cvtColor(np.uint8([[[160, 150, 100]]]), cv2.COLOR_YCrCb2BGR)[0][0]
    frame[250:350, 200:300] = skin_bgr

    # Mock mask covering BOTH the white paper AND the skin region
    mask_arr = np.zeros((480, 640), dtype=np.uint8)
    mask_arr[100:250, 100:250] = 1
    mask_arr[250:350, 200:300] = 1  # hand region erroneously included in mask

    mock_box = MagicMock()
    mock_box.conf = [torch.tensor(0.90)]
    mock_box.cls = [torch.tensor(0)]
    mock_box.xyxy = [torch.tensor([100.0, 100.0, 300.0, 350.0])]

    mock_masks = MagicMock()
    mock_masks.data = torch.from_numpy(mask_arr).unsqueeze(0)
    mock_masks.xy = [np.array([[100, 100], [250, 100], [300, 350], [100, 350]], dtype=np.float32)]

    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_result.masks = mock_masks
    mock_model_instance.predict.return_value = [mock_result]

    verifier = YoloStepVerifier("dummy_weights.pt", device="cpu", conf_threshold=0.40)
    res = verifier.verify_bgr(frame)

    assert res.recognized is True
    poly = res.roi["polygon"]
    assert poly is not None
    # Maximum Y in the refined polygon should not extend deep into the hand region (> 260)
    max_y = max(pt[1] for pt in poly)
    assert max_y <= 260


