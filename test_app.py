import json
import pytest
from app import app

@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"API Flask MLflow est en ligne" in response.data

def test_predict_success(client, monkeypatch):
    # Mock du modèle
    class MockModel:
        def predict(self, X):
            return ["positif"]

    # Remplacer le modèle par un mock
    from app import model
    monkeypatch.setattr("app.model", MockModel())

    response = client.post(
        "/predict",
        data=json.dumps({"text": "Ceci est un test"}),
        content_type="application/json"
    )

    assert response.status_code == 200
    data = json.loads(response.data)
    assert "prediction" in data
    assert data["prediction"] == "positif"

def test_predict_missing_field(client):
    response = client.post(
        "/predict",
        data=json.dumps({"wrong_field": "oops"}),
        content_type="application/json"
    )

    assert response.status_code == 400
    data = json.loads(response.data)
    assert "error" in data

