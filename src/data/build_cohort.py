"""Build modeling cohorts from validated synthetic AMR data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def build_cohort(
    input_path: str | Path,
    output_path: str | Path,
    infection_site: str = "urinary",
    min_pair_count: int = 1,
) -> pd.DataFrame:
    """Filter raw records into a modeling cohort and save the result."""
    if min_pair_count < 1:
        raise ValueError("min_pair_count must be at least 1.")

    df = pd.read_csv(input_path)
    cohort = df.loc[df["infection_site"] == infection_site].copy()
    cohort = cohort.dropna(subset=["pathogen", "antibiotic", "susceptible"])

    if min_pair_count > 1 and not cohort.empty:
        pair_counts = cohort.groupby(["pathogen", "antibiotic"]).size().rename("pair_count")
        cohort = cohort.merge(pair_counts, on=["pathogen", "antibiotic"], how="left")
        cohort = cohort.loc[cohort["pair_count"] >= min_pair_count].drop(columns=["pair_count"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)
    return cohort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a modeling cohort.")
    parser.add_argument("--input", type=Path, required=True, help="Raw input CSV path.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/cohort.csv"))
    parser.add_argument("--infection-site", default="urinary")
    parser.add_argument("--min-pair-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cohort = build_cohort(
        input_path=args.input,
        output_path=args.output,
        infection_site=args.infection_site,
        min_pair_count=args.min_pair_count,
    )
    print(f"Saved cohort with {len(cohort):,} rows to {args.output}")


if __name__ == "__main__":
    main()
