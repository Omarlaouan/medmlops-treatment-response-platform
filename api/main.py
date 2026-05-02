"""FastAPI service for treatment-response prediction."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explainability.explain_prediction import explain_input
from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET
from src.inference.predict import DISCLAIMER, predict_one

APP_VERSION = "1.0.0"
MODEL_PATH = Path("models/treatment_response_model.joblib")
MODEL_METADATA_PATH = Path("models/model_metadata.json")

VALID_VALUES = {
    "sex": ["M", "F"],
    "ward_type": ["emergency", "outpatient", "medical", "surgery", "ICU"],
    "infection_site": ["urinary", "respiratory", "bloodstream", "skin_soft_tissue"],
    "pathogen": [
        "E. coli",
        "Klebsiella pneumoniae",
        "Staphylococcus aureus",
        "Pseudomonas aeruginosa",
        "Enterococcus faecalis",
    ],
    "antibiotic": [
        "amoxicillin",
        "ceftriaxone",
        "ciprofloxacin",
        "gentamicin",
        "meropenem",
        "vancomycin",
        "nitrofurantoin",
    ],
}

app = FastAPI(title="MedMLOps Treatment Response API", version=APP_VERSION)


class PredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=95)
    sex: Literal["M", "F"]
    ward_type: Literal["emergency", "outpatient", "medical", "surgery", "ICU"]
    infection_site: Literal["urinary", "respiratory", "bloodstream", "skin_soft_tissue"]
    pathogen: Literal[
        "E. coli",
        "Klebsiella pneumoniae",
        "Staphylococcus aureus",
        "Pseudomonas aeruginosa",
        "Enterococcus faecalis",
    ]
    antibiotic: Literal[
        "amoxicillin",
        "ceftriaxone",
        "ciprofloxacin",
        "gentamicin",
        "meropenem",
        "vancomycin",
        "nitrofurantoin",
    ]
    prior_antibiotic_exposure: int = Field(..., ge=0, le=1)
    comorbidity_score: int = Field(..., ge=0, le=6)
    local_resistance_rate: float = Field(..., ge=0.0, le=1.0)


def _model_timestamp(path: Path) -> str | None:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def _load_model_metadata() -> dict[str, object] | None:
    if not MODEL_METADATA_PATH.exists():
        return None
    return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    """Service health check."""
    return {"status": "ok", "warning": DISCLAIMER}


@app.get("/metadata")
def metadata() -> dict[str, object]:
    """Return service-level metadata and request schema information."""
    return {
        "service": "MedMLOps Treatment Response API",
        "version": APP_VERSION,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_last_modified_utc": _model_timestamp(MODEL_PATH),
        "feature_schema": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target": TARGET,
            "valid_values": VALID_VALUES,
        },
        "warning": DISCLAIMER,
    }


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Return model artifact metadata when training has been run."""
    return {
        "model_exists": MODEL_PATH.exists(),
        "model_path": str(MODEL_PATH),
        "model_last_modified_utc": _model_timestamp(MODEL_PATH),
        "metadata": _load_model_metadata(),
        "warning": DISCLAIMER,
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, object]:
    """Predict synthetic susceptibility probability for one request."""
    payload = request.model_dump()
    try:
        prediction = predict_one(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "probability_of_susceptibility": prediction["probability_of_susceptibility"],
        "recommendation_level": prediction["recommendation_level"],
        "explanation": explain_input(payload),
        "warning": prediction["warning"],
    }
