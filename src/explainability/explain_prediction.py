"""Lightweight explainability helpers for the baseline model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def get_global_feature_importance(
    model_path: str | Path = "models/treatment_response_model.joblib",
    top_n: int = 10,
) -> dict[str, list[dict[str, float | str]]]:
    """Return top positive and negative logistic-regression coefficients."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    pipeline = joblib.load(path)
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_[0]

    importance = pd.DataFrame({"feature": feature_names, "coefficient": coefficients})
    positive = (
        importance.sort_values("coefficient", ascending=False)
        .head(top_n)
        .to_dict(orient="records")
    )
    negative = (
        importance.sort_values("coefficient", ascending=True)
        .head(top_n)
        .to_dict(orient="records")
    )
    return {"top_positive_drivers": positive, "top_negative_drivers": negative}


def explain_input(input_dict: dict[str, Any]) -> dict[str, list[str]]:
    """Provide a simple local heuristic explanation for an input record."""
    negative_factors: list[str] = []
    positive_factors: list[str] = []
    contextual_factors: list[str] = []

    local_resistance_rate = float(input_dict.get("local_resistance_rate", 0.0))
    prior_exposure = int(input_dict.get("prior_antibiotic_exposure", 0))
    ward_type = str(input_dict.get("ward_type", ""))
    pathogen = str(input_dict.get("pathogen", "unknown pathogen"))
    antibiotic = str(input_dict.get("antibiotic", "unknown antibiotic"))

    if local_resistance_rate >= 0.30:
        negative_factors.append("High local resistance rate is expected to reduce susceptibility.")
    elif local_resistance_rate <= 0.15:
        positive_factors.append("Low local resistance rate is expected to support susceptibility.")

    if prior_exposure == 1:
        negative_factors.append("Prior antibiotic exposure is expected to reduce susceptibility.")

    if ward_type == "ICU":
        negative_factors.append("ICU setting is associated with higher resistance risk in this synthetic data.")

    if pathogen == "E. coli" and antibiotic == "nitrofurantoin":
        positive_factors.append("Nitrofurantoin is modeled as relatively active for urinary E. coli.")
    if antibiotic == "meropenem":
        positive_factors.append("Meropenem is modeled as generally high activity, with penalties for high-risk context.")
    if antibiotic == "vancomycin" and pathogen in {"Staphylococcus aureus", "Enterococcus faecalis"}:
        positive_factors.append("Vancomycin is modeled as relevant for this Gram-positive pathogen.")
    if antibiotic == "vancomycin" and pathogen in {
        "E. coli",
        "Klebsiella pneumoniae",
        "Pseudomonas aeruginosa",
    }:
        negative_factors.append("Vancomycin is modeled as not useful for Gram-negative pathogens.")
    if antibiotic == "amoxicillin" and pathogen in {
        "E. coli",
        "Klebsiella pneumoniae",
        "Pseudomonas aeruginosa",
        "Staphylococcus aureus",
    }:
        negative_factors.append("Amoxicillin is modeled as lower activity for this pathogen context.")

    contextual_factors.append(f"Pathogen-antibiotic context: {pathogen} treated with {antibiotic}.")

    return {
        "positive_factors": positive_factors or ["No strong positive heuristic factors identified."],
        "negative_factors": negative_factors or ["No strong negative heuristic factors identified."],
        "contextual_factors": contextual_factors,
    }
