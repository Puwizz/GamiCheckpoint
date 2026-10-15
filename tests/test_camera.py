from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest
from scripts.utils.test_camera import (
    draw_info_overlay,
    process_contours,
    process_edges,
    process_grayscale,
    process_original,
    run_camera,
)


@pytest.fixture
def synthetic_frame():
    return np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)


def test_process_original(synthetic_frame):
    processed = process_original(synthetic_frame)
    assert processed.shape == (480, 640, 3)
    assert processed.dtype == np.uint8
    assert processed is not synthetic_frame
    assert np.array_equal(processed, synthetic_frame)


def test_process_grayscale(synthetic_frame):
    processed = process_grayscale(synthetic_frame)
    assert processed.shape == (480, 640, 3)
    assert np.array_equal(processed[:, :, 0], processed[:, :, 1])
    assert np.array_equal(processed[:, :, 1], processed[:, :, 2])


def test_process_edges():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)

    processed = process_edges(frame)
    assert processed.shape == (480, 640, 3)
    assert np.all(processed[:, :, 0] == 0)
    assert np.all(processed[:, :, 2] == 0)
    assert np.any(processed[:, :, 1] > 0)


def test_process_contours():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (300, 300), (255, 255, 255), -1)

    processed = process_contours(frame)
    assert processed.shape == (480, 640, 3)
    assert not np.array_equal(processed, frame)


def test_draw_info_overlay(synthetic_frame):
    frame = synthetic_frame.copy()
    overlay = draw_info_overlay(frame, "Original", 30.0)
    assert overlay.shape == (480, 640, 3)
    assert not np.array_equal(overlay, synthetic_frame)


def test_run_camera_device_open_failure(monkeypatch):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)

    success = run_camera(device_index=99)
    assert success is False


def test_run_camera_read_failure(monkeypatch):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (False, None)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)

    success = run_camera(device_index=0)
    assert success is True
    assert mock_cap.release.called


def test_run_camera_process_single_frame(monkeypatch):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args, **kwargs: mock_cap)
    monkeypatch.setattr(cv2, "imshow", lambda *args, **kwargs: None)
    monkeypatch.setattr(cv2, "waitKey", lambda *args, **kwargs: ord('q'))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda *args, **kwargs: None)

    success = run_camera(device_index=0, max_frames=1)
    assert success is True
    assert mock_cap.set.call_count >= 2
    assert mock_cap.release.called
