"""Local inference utilities for treatment-response prediction."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.features.build_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


DISCLAIMER = (
    "This project is an educational prototype. It is not a medical device, "
    "not clinically validated, and must not be used for clinical decision-making."
)


def recommendation_from_probability(probability: float) -> str:
    """Map susceptibility probability to a demo recommendation label."""
    if probability >= 0.8:
        return "strong candidate"
    if probability >= 0.6:
        return "candidate"
    if probability >= 0.4:
        return "uncertain"
    return "not recommended"


@lru_cache(maxsize=4)
def load_model(model_path: str = "models/treatment_response_model.joblib") -> Any:
    """Load a persisted sklearn model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `python -m src.training.train` first."
        )
    return joblib.load(path)


def predict_one(
    input_dict: dict[str, Any],
    model_path: str = "models/treatment_response_model.joblib",
) -> dict[str, Any]:
    """Predict susceptibility probability for one input record."""
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing = [column for column in feature_columns if column not in input_dict]
    if missing:
        raise ValueError(f"Missing required prediction fields: {missing}")

    model = load_model(str(model_path))
    input_df = pd.DataFrame([{column: input_dict[column] for column in feature_columns}])
    probability = float(model.predict_proba(input_df)[:, 1][0])
    return {
        "probability_of_susceptibility": probability,
        "recommendation_level": recommendation_from_probability(probability),
        "warning": DISCLAIMER,
    }
