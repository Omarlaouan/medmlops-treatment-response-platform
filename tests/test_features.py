from __future__ import annotations

import pandas as pd

from src.features.build_features import get_preprocessor, split_features_target


def test_preprocessor_fit_transform_small_dataframe() -> None:
    df = pd.DataFrame(
        {
            "age": [45, 72, 61],
            "comorbidity_score": [1, 4, 2],
            "local_resistance_rate": [0.12, 0.41, 0.22],
            "prior_antibiotic_exposure": [0, 1, 0],
            "sex": ["F", "M", "F"],
            "ward_type": ["outpatient", "ICU", "medical"],
            "infection_site": ["urinary", "urinary", "urinary"],
            "pathogen": ["E. coli", "Enterococcus faecalis", "Klebsiella pneumoniae"],
            "antibiotic": ["nitrofurantoin", "vancomycin", "ceftriaxone"],
            "susceptible": [1, 0, 1],
        }
    )
    X, y = split_features_target(df)
    transformed = get_preprocessor().fit_transform(X, y)
    assert transformed.shape[0] == len(df)
    assert transformed.shape[1] > 4
