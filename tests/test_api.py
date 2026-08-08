from io import BytesIO
import os

from fastapi.testclient import TestClient
from PIL import Image
import pytest

from src.api.main import app

client = TestClient(app)

checkpoint_missing = not os.path.exists("artifacts/best_model.pt")


def make_test_image() -> BytesIO:
    image = Image.new("RGB", (28, 28), color=(120, 80, 160))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body.get("status") in {"ok", "healthy"}


@pytest.mark.skipif(checkpoint_missing, reason="No trained checkpoint available")
def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200
    body = response.json()

    assert "model_name" in body
    assert "num_classes" in body
    assert body["image_height"] == 28
    assert body["image_width"] == 28


def test_frontend_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Pathology" in response.text


def test_static_css():
    response = client.get("/static/styles.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]


def test_static_javascript():
    response = client.get("/static/app.js")

    assert response.status_code == 200
    assert "fetch" in response.text


@pytest.mark.skipif(checkpoint_missing, reason="No trained checkpoint available")
def test_predict_valid_image():
    image = make_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test-image.png",
                image,
                "image/png",
            )
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert "filename" in body
    assert "predicted_class" in body
    assert "predicted_label" in body
    assert "confidence" in body
    assert "probabilities" in body

    assert 0 <= body["confidence"] <= 1
    assert len(body["probabilities"]) == 9


def test_predict_rejects_invalid_file():
    response = client.post(
        "/predict",
        files={
            "file": (
                "not-an-image.txt",
                b"this is not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code in {400, 415, 422}
