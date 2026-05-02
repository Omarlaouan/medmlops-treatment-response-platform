"""Feature definitions and preprocessing for treatment-response modeling."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "comorbidity_score",
    "local_resistance_rate",
    "prior_antibiotic_exposure",
]
CATEGORICAL_FEATURES = ["sex", "ward_type", "infection_site", "pathogen", "antibiotic"]
TARGET = "susceptible"


def get_preprocessor() -> ColumnTransformer:
    """Create the sklearn preprocessing transformer."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into model features and target."""
    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required modeling columns: {missing}")
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy(), df[TARGET].astype(int).copy()
