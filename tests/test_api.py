from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_read_root():
    """Check if the API starts correctly"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Age Prediction API is running. Go to /docs for Swagger UI."
    }


def test_predict_no_file():
    """Check if the API correctly rejects empty requests"""
    response = client.post("/predict")
    assert response.status_code == 422
