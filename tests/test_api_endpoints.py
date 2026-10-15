"""
tests/test_api_endpoints.py
Unit tests for FastAPI endpoints: registry caching, validation, error handling, and security.
"""
from unittest.mock import MagicMock
import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest
import app.app as app_module


def create_test_image_bytes():
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(app_module.state, "load", lambda: app_module.state.registry.load())
    monkeypatch.setattr(app_module.state.registry, "get_verifier", lambda *args, **kwargs: None)
    with TestClient(app_module.app) as c:
        yield c


def test_list_figures_endpoint(client):
    response = client.get("/api/figures")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    figure_ids = [f["id"] for f in data]
    assert "crane" in figure_ids
    assert "frog" in figure_ids


def test_list_figures_force_reload(client):
    response = client.get("/api/figures?reload=true")
    assert response.status_code == 200
    assert "crane" in app_module.state.registry


def test_get_figure_detail_success(client):
    response = client.get("/api/figures/crane")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "crane"
    assert "steps" in data
    assert len(data["steps"]) == 6


def test_get_figure_detail_not_found(client):
    response = client.get("/api/figures/nonexistent_origami")
    assert response.status_code == 404


def test_get_figure_detail_invalid_path_pattern(client):
    response = client.get("/api/figures/crane..bad!path")
    assert response.status_code == 422


def test_predict_coming_soon_figure(client):
    dummy_img = b"fake_image_bytes"
    response = client.post(
        "/api/predict?figure=frog",
        files={"file": ("test.jpg", dummy_img, "image/jpeg")},
    )
    assert response.status_code == 400
    assert response.json()["status"] == "coming_soon"


def test_predict_nonexistent_figure(client):
    dummy_img = b"fake_image_bytes"
    response = client.post(
        "/api/predict?figure=unknown_figure",
        files={"file": ("test.jpg", dummy_img, "image/jpeg")},
    )
    assert response.status_code == 404


def test_predict_empty_file(client):
    response = client.post(
        "/api/predict?figure=crane",
        files={"file": ("test.jpg", b"", "image/jpeg")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["error"].lower()


def test_predict_unsupported_media_type(client):
    response = client.post(
        "/api/predict?figure=crane",
        files={"file": ("test.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 400
    assert "unsupported media type" in response.json()["error"].lower()


def test_predict_payload_too_large(client):
    large_payload = b"x" * (app_module.MAX_UPLOAD_SIZE + 1024)
    response = client.post(
        "/api/predict?figure=crane",
        files={"file": ("large.jpg", large_payload, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed limit" in response.json()["error"].lower()


def test_predict_streaming_closes_file(client, monkeypatch):
    closed = False

    from starlette.datastructures import UploadFile

    orig_close = UploadFile.close

    async def mock_close(self):
        nonlocal closed
        closed = True
        return await orig_close(self)

    monkeypatch.setattr(UploadFile, "close", mock_close)

    response = client.post(
        "/api/predict?figure=crane",
        files={"file": ("test.jpg", b"dummy_content", "image/jpeg")},
    )
    assert closed is True


def test_predict_invalid_query_pattern(client):
    response = client.post(
        "/api/predict?figure=bad!figure@name",
        files={"file": ("test.jpg", b"dummy", "image/jpeg")},
    )
    assert response.status_code == 422


def test_cors_headers_allowed_origin(client):
    response = client.get("/api/figures", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert response.headers.get("access-control-allow-credentials") == "true"


def test_cors_preflight_options(client):
    response = client.options(
        "/api/predict",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert "POST" in response.headers.get("access-control-allow-methods", "")


def test_predict_model_not_loaded_503(client, monkeypatch):
    monkeypatch.setattr(app_module.state, "verifiers", {"crane": None})
    response = client.post(
        "/api/predict?figure=crane",
        files={"file": ("test.jpg", b"valid_image_bytes", "image/jpeg")},
    )
    assert response.status_code == 503
    assert "not loaded yet" in response.json()["error"].lower()


def test_predict_success_recognized(client, monkeypatch):
    mock_verifier = MagicMock()
    mock_res = MagicMock()
    mock_res.to_dict.return_value = {
        "frame_ready": True,
        "recognized": True,
        "recognition_reason": "ok",
        "prediction": "Chk1",
        "step": "Chk1",
        "step_number": 1,
        "step_name": "Initial Fold",
        "confidence": 92.5,
        "top_gap": 45.0,
        "recognition_thresholds": {"conf_threshold": 0.55, "min_top_gap": 8.0},
        "roi": {"applied": True, "bbox": [10, 10, 80, 80]},
        "top_predictions": [{"class_name": "Chk1", "confidence": 92.5}],
    }
    mock_verifier.verify_bytes.return_value = mock_res
    monkeypatch.setattr(app_module.state, "get_verifier", lambda fig: mock_verifier)

    dummy_img = create_test_image_bytes()
    response = client.post(
        "/api/predict?figure=crane",
        files={"file": ("frame.jpg", dummy_img, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["frame_ready"] is True
    assert data["recognized"] is True
    assert data["step"] == "Chk1"
    assert data["step_number"] == 1
    assert data["confidence"] == 92.5
    assert data["top_gap"] == 45.0
    assert "roi" in data


def test_predict_success_unrecognized_gap(client, monkeypatch):
    mock_verifier = MagicMock()
    mock_res = MagicMock()
    mock_res.to_dict.return_value = {
        "frame_ready": True,
        "recognized": False,
        "recognition_reason": "ambiguous_step",
        "prediction": "Chk1",
        "step": "Chk1",
        "step_number": 1,
        "step_name": "Initial Fold",
        "confidence": 55.0,
        "top_gap": 3.0,
        "recognition_thresholds": {"conf_threshold": 0.55, "min_top_gap": 8.0},
        "roi": {"applied": False, "bbox": None},
        "top_predictions": [
            {"class_name": "Chk1", "confidence": 55.0},
            {"class_name": "Chk2", "confidence": 52.0},
        ],
    }
    mock_verifier.verify_bytes.return_value = mock_res
    monkeypatch.setattr(app_module.state, "get_verifier", lambda fig: mock_verifier)

    dummy_img = create_test_image_bytes()
    response = client.post(
        "/api/predict?figure=crane",
        files={"file": ("frame.jpg", dummy_img, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["frame_ready"] is True
    assert data["recognized"] is False
    assert data["recognition_reason"] == "ambiguous_step"


def test_predict_success_frame_not_ready(client, monkeypatch):
    mock_verifier = MagicMock()
    mock_res = MagicMock()
    mock_res.to_dict.return_value = {
        "frame_ready": False,
        "reason": "frame_too_dark",
        "metrics": {"mean_luma": 10.0},
    }
    mock_verifier.verify_bytes.return_value = mock_res
    monkeypatch.setattr(app_module.state, "get_verifier", lambda fig: mock_verifier)

    dummy_img = create_test_image_bytes()
    response = client.post(
        "/api/predict?figure=crane",
        files={"file": ("frame.jpg", dummy_img, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["frame_ready"] is False
    assert data["reason"] == "frame_too_dark"


def test_consistent_error_response_schema_detail_and_error(client):
    res_404 = client.get("/api/figures/nonexistent_figure")
    assert res_404.status_code == 404
    data_404 = res_404.json()
    assert "detail" in data_404
    assert "error" in data_404
    assert data_404["detail"] == data_404["error"]

    res_pred_404 = client.post(
        "/api/predict?figure=nonexistent_figure",
        files={"file": ("test.jpg", b"dummy", "image/jpeg")},
    )
    assert res_pred_404.status_code == 404
    data_pred_404 = res_pred_404.json()
    assert "detail" in data_pred_404
    assert "error" in data_pred_404
    assert data_pred_404["detail"] == data_pred_404["error"]

    res_pred_400 = client.post(
        "/api/predict?figure=crane",
        files={"file": ("test.jpg", b"", "image/jpeg")},
    )
    assert res_pred_400.status_code == 400
    data_pred_400 = res_pred_400.json()
    assert "detail" in data_pred_400
    assert "error" in data_pred_400
    assert data_pred_400["detail"] == data_pred_400["error"]


def test_dynamic_verifier_loading_multi_figure(monkeypatch):
    mock_verifier = MagicMock()
    mock_verifier.classes = ("Chk1", "Chk2", "Chk3")

    monkeypatch.setitem(app_module.state.registry._registry, "custom_bird", {
        "id": "custom_bird",
        "status": "available",
        "total_steps": 3,
        "steps": [
            {"step_number": 1, "class_name": "Chk1"},
            {"step_number": 2, "class_name": "Chk2"},
            {"step_number": 3, "class_name": "Chk3"},
        ]
    })
    app_module.state.verifiers.pop("custom_bird", None)

    monkeypatch.setattr(app_module.state.registry, "get_verifier", lambda fig_id, device=None: mock_verifier if fig_id == "custom_bird" else None)

    verifier = app_module.state.get_verifier("custom_bird")
    assert verifier is mock_verifier
    assert "custom_bird" in app_module.state.verifiers
    assert app_module.state.verifiers["custom_bird"] is mock_verifier


def test_multi_figure_get_classes_from_registry():
    classes = app_module.state.get_classes("frog")
    assert classes == ["Chk1", "Chk2", "Chk3", "Chk4"]


def test_list_figures_reload_clears_verifier_cache(client):
    app_module.state.verifiers["cached_test"] = "dummy_obj"
    assert "cached_test" in app_module.state.verifiers

    response = client.get("/api/figures?reload=true")
    assert response.status_code == 200
    assert "cached_test" not in app_module.state.verifiers


