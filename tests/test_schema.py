from __future__ import annotations

import pandas as pd
import pytest

from src.data.validate_schema import validate_dataframe


def _valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["P0000001", "P0000002"],
            "age": [45, 72],
            "sex": ["F", "M"],
            "ward_type": ["outpatient", "ICU"],
            "infection_site": ["urinary", "urinary"],
            "pathogen": ["E. coli", "Enterococcus faecalis"],
            "antibiotic": ["nitrofurantoin", "vancomycin"],
            "prior_antibiotic_exposure": [0, 1],
            "comorbidity_score": [1, 4],
            "local_resistance_rate": [0.12, 0.41],
            "sample_date": ["2024-01-15", "2024-02-20"],
            "susceptible": [1, 0],
        }
    )


def test_validate_schema_passes_for_valid_dataframe() -> None:
    summary = validate_dataframe(_valid_dataframe())
    assert summary["status"] == "passed"
    assert summary["n_rows"] == 2


def test_validate_schema_fails_for_invalid_age() -> None:
    df = _valid_dataframe()
    df.loc[0, "age"] = 140
    with pytest.raises(ValueError, match="age"):
        validate_dataframe(df)
