"""Schema validation for synthetic AMR-like CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = [
    "patient_id",
    "age",
    "sex",
    "ward_type",
    "infection_site",
    "pathogen",
    "antibiotic",
    "prior_antibiotic_exposure",
    "comorbidity_score",
    "local_resistance_rate",
    "sample_date",
    "susceptible",
]

VALID_CATEGORIES = {
    "sex": {"M", "F"},
    "ward_type": {"emergency", "outpatient", "medical", "surgery", "ICU"},
    "infection_site": {"urinary", "respiratory", "bloodstream", "skin_soft_tissue"},
    "pathogen": {
        "E. coli",
        "Klebsiella pneumoniae",
        "Staphylococcus aureus",
        "Pseudomonas aeruginosa",
        "Enterococcus faecalis",
    },
    "antibiotic": {
        "amoxicillin",
        "ceftriaxone",
        "ciprofloxacin",
        "gentamicin",
        "meropenem",
        "vancomycin",
        "nitrofurantoin",
    },
}


def _invalid_values(series: pd.Series, valid_values: set[Any]) -> list[Any]:
    values = set(series.dropna().unique())
    return sorted(values - valid_values)


def validate_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Validate a dataframe and return a summary dictionary.

    Raises:
        ValueError: If required columns, ranges, categories, or target values are invalid.
    """
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    summary: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "missing_value_rate": df[REQUIRED_COLUMNS].isna().mean().round(4).to_dict(),
    }

    for column, valid_values in VALID_CATEGORIES.items():
        invalid = _invalid_values(df[column], valid_values)
        if invalid:
            raise ValueError(f"Column '{column}' contains invalid values: {invalid}")

    checks = {
        "age": df["age"].between(0, 120),
        "local_resistance_rate": df["local_resistance_rate"].between(0, 1),
        "prior_antibiotic_exposure": df["prior_antibiotic_exposure"].isin([0, 1]),
        "comorbidity_score": df["comorbidity_score"].between(0, 6),
        "susceptible": df["susceptible"].isin([0, 1]),
    }
    for column, mask in checks.items():
        if not bool(mask.all()):
            bad_count = int((~mask).sum())
            raise ValueError(f"Column '{column}' failed validation for {bad_count} rows.")

    sample_dates = pd.to_datetime(df["sample_date"], errors="coerce")
    if sample_dates.isna().any():
        raise ValueError("Column 'sample_date' contains invalid date strings.")

    summary["target_rate"] = float(df["susceptible"].mean())
    summary["date_min"] = str(sample_dates.min().date())
    summary["date_max"] = str(sample_dates.max().date())
    summary["status"] = "passed"
    return summary


def validate_csv(input_path: str | Path) -> dict[str, Any]:
    """Load and validate a CSV file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    return validate_dataframe(df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a synthetic AMR CSV schema.")
    parser.add_argument("--input", type=Path, required=True, help="Input CSV path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_csv(args.input)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
