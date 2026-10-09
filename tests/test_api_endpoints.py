"""
tests/test_api_endpoints.py
Unit tests for FastAPI endpoints: registry caching, validation, error handling, and security.
"""
from fastapi.testclient import TestClient
import pytest
import app.app as app_module


@pytest.fixture
def client():
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
    # Path traversal / invalid characters should be rejected with 422
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
    # 11MB file exceeds 10MB limit
    large_payload = b"x" * (app_module.MAX_UPLOAD_SIZE + 1024)
    response = client.post(
        "/api/predict?figure=crane",
        files={"file": ("large.jpg", large_payload, "image/jpeg")},
    )
    assert response.status_code == 413
    assert "exceeds maximum allowed limit" in response.json()["error"].lower()


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
