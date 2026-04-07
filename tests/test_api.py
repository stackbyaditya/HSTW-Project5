from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app

PAYLOAD = {
    "ip": 87540,
    "app": 12,
    "device": 1,
    "os": 13,
    "channel": 497,
    "click_time": "2017-11-07T09:30:38",
}


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_prediction_endpoint() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", json=PAYLOAD)

    assert response.status_code == 200

    body = response.json()
    assert set(body) == {"fraud_probability", "prediction", "label"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction"] in {0, 1}
    assert body["label"] in {"FRAUD", "LEGITIMATE"}
    assert body["label"] == ("FRAUD" if body["prediction"] == 1 else "LEGITIMATE")
