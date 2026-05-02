"""FastAPI service for treatment-response prediction."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.explainability.explain_prediction import explain_input
from src.inference.predict import DISCLAIMER, predict_one


app = FastAPI(title="MedMLOps Treatment Response API")


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


@app.get("/health")
def health() -> dict[str, str]:
    """Service health check."""
    return {"status": "ok", "warning": DISCLAIMER}


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
