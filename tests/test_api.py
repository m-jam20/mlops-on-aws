from fastapi.testclient import TestClient

from churn.api import app

CUSTOMER = {"tenure": 5, "MonthlyCharges": 70.35, "TotalCharges": 351.75}


def test_health(model_file):
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_probability(model_file):
    with TestClient(app) as client:
        response = client.post("/predict", json=CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert isinstance(body["churn"], bool)


def test_predict_rejects_bad_input(model_file):
    with TestClient(app) as client:
        response = client.post("/predict", json={**CUSTOMER, "Contract": "Weekly"})
    assert response.status_code == 422  # validation error
