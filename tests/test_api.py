from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _payload() -> dict[str, object]:
    return {
        "age": 67,
        "sex": "F",
        "ward_type": "medical",
        "infection_site": "urinary",
        "pathogen": "E. coli",
        "antibiotic": "nitrofurantoin",
        "prior_antibiotic_exposure": 0,
        "comorbidity_score": 2,
        "local_resistance_rate": 0.12,
    }


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata_endpoint() -> None:
    response = client.get("/metadata")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "MedMLOps Treatment Response API"
    assert "feature_schema" in body
    assert "numeric_features" in body["feature_schema"]
    assert "not clinically validated" in body["warning"]


def test_model_info_endpoint() -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "model_exists" in body
    assert "model_path" in body
    assert "warning" in body


def test_predict_endpoint_rejects_invalid_age() -> None:
    payload = _payload()
    payload["age"] = 140
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_category() -> None:
    payload = _payload()
    payload["ward_type"] = "invalid"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_if_model_exists() -> None:
    if not Path("models/treatment_response_model.joblib").exists():
        pytest.skip("Model artifact does not exist. Run training before prediction test.")

    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability_of_susceptibility"] <= 1.0
    assert "recommendation_level" in body
    assert "explanation" in body
    assert "not clinically validated" in body["warning"]
