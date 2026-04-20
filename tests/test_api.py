import io
from unittest.mock import patch
from PIL import Image
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def _make_image_bytes(size=(100, 100), fmt="JPEG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(128, 64, 32)).save(buf, format=fmt)
    return buf.getvalue()


def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Age Prediction API" in response.json()["message"]


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_no_file():
    response = client.post("/predict")
    assert response.status_code == 422


def test_predict_valid_image():
    img_bytes = _make_image_bytes()
    response = client.post("/predict", files={"file": ("face.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 200
    body = response.json()
    assert "predicted_age" in body
    assert "confidence_interval" in body
    assert body["unit"] == "years"
    assert isinstance(body["predicted_age"], float)


def test_predict_response_format():
    img_bytes = _make_image_bytes()
    response = client.post("/predict", files={"file": ("face.png", img_bytes, "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"filename", "predicted_age", "confidence_interval", "face_crop", "unit", "device_used"}
    assert body["filename"] == "face.png"
    assert "low" in body["confidence_interval"]
    assert "high" in body["confidence_interval"]
    assert body["confidence_interval"]["low"] <= body["predicted_age"] <= body["confidence_interval"]["high"]


def test_predict_non_image_file():
    response = client.post("/predict", files={"file": ("doc.txt", b"hello world", "text/plain")})
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_predict_corrupted_image():
    response = client.post("/predict", files={"file": ("bad.jpg", b"\xff\xd8corrupted", "image/jpeg")})
    assert response.status_code == 500


def test_predict_oversized_file():
    large_data = b"x" * (10 * 1024 * 1024 + 1)
    response = client.post("/predict", files={"file": ("big.jpg", large_data, "image/jpeg")})
    assert response.status_code == 413
    assert "10MB" in response.json()["detail"]


def test_predict_no_face_detected():
    img_bytes = _make_image_bytes()
    with patch("api.main.face_cascade") as mock_cascade:
        mock_cascade.detectMultiScale.return_value = []
        response = client.post("/predict", files={"file": ("no_face.jpg", img_bytes, "image/jpeg")})
    assert response.status_code == 422
    assert "face" in response.json()["detail"].lower()
