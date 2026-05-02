"""Generate synthetic AMR-like treatment-response data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SEXES = ["M", "F"]
WARD_TYPES = ["emergency", "outpatient", "medical", "surgery", "ICU"]
INFECTION_SITES = ["urinary", "respiratory", "bloodstream", "skin_soft_tissue"]
PATHOGENS = [
    "E. coli",
    "Klebsiella pneumoniae",
    "Staphylococcus aureus",
    "Pseudomonas aeruginosa",
    "Enterococcus faecalis",
]
ANTIBIOTICS = [
    "amoxicillin",
    "ceftriaxone",
    "ciprofloxacin",
    "gentamicin",
    "meropenem",
    "vancomycin",
    "nitrofurantoin",
]


PAIR_ACTIVITY = {
    ("E. coli", "nitrofurantoin"): 0.88,
    ("E. coli", "ceftriaxone"): 0.72,
    ("E. coli", "ciprofloxacin"): 0.62,
    ("E. coli", "gentamicin"): 0.76,
    ("E. coli", "meropenem"): 0.94,
    ("E. coli", "amoxicillin"): 0.42,
    ("E. coli", "vancomycin"): 0.04,
    ("Klebsiella pneumoniae", "nitrofurantoin"): 0.42,
    ("Klebsiella pneumoniae", "ceftriaxone"): 0.55,
    ("Klebsiella pneumoniae", "ciprofloxacin"): 0.48,
    ("Klebsiella pneumoniae", "gentamicin"): 0.65,
    ("Klebsiella pneumoniae", "meropenem"): 0.90,
    ("Klebsiella pneumoniae", "amoxicillin"): 0.18,
    ("Klebsiella pneumoniae", "vancomycin"): 0.03,
    ("Staphylococcus aureus", "nitrofurantoin"): 0.20,
    ("Staphylococcus aureus", "ceftriaxone"): 0.35,
    ("Staphylococcus aureus", "ciprofloxacin"): 0.50,
    ("Staphylococcus aureus", "gentamicin"): 0.45,
    ("Staphylococcus aureus", "meropenem"): 0.35,
    ("Staphylococcus aureus", "amoxicillin"): 0.30,
    ("Staphylococcus aureus", "vancomycin"): 0.92,
    ("Pseudomonas aeruginosa", "nitrofurantoin"): 0.05,
    ("Pseudomonas aeruginosa", "ceftriaxone"): 0.08,
    ("Pseudomonas aeruginosa", "ciprofloxacin"): 0.45,
    ("Pseudomonas aeruginosa", "gentamicin"): 0.58,
    ("Pseudomonas aeruginosa", "meropenem"): 0.78,
    ("Pseudomonas aeruginosa", "amoxicillin"): 0.03,
    ("Pseudomonas aeruginosa", "vancomycin"): 0.02,
    ("Enterococcus faecalis", "nitrofurantoin"): 0.64,
    ("Enterococcus faecalis", "ceftriaxone"): 0.08,
    ("Enterococcus faecalis", "ciprofloxacin"): 0.38,
    ("Enterococcus faecalis", "gentamicin"): 0.30,
    ("Enterococcus faecalis", "meropenem"): 0.22,
    ("Enterococcus faecalis", "amoxicillin"): 0.72,
    ("Enterococcus faecalis", "vancomycin"): 0.88,
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_synthetic_amr(n_rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic AMR-like dataset with plausible treatment signal."""
    if n_rows <= 0:
        raise ValueError("n_rows must be positive.")

    rng = np.random.default_rng(seed)

    ward_type = rng.choice(WARD_TYPES, size=n_rows, p=[0.22, 0.28, 0.25, 0.17, 0.08])
    infection_site = rng.choice(INFECTION_SITES, size=n_rows, p=[0.48, 0.26, 0.12, 0.14])
    pathogen = rng.choice(PATHOGENS, size=n_rows, p=[0.40, 0.20, 0.18, 0.12, 0.10])

    antibiotics = []
    for site, bug in zip(infection_site, pathogen, strict=False):
        if site == "urinary" and bug == "E. coli":
            probs = [0.10, 0.14, 0.18, 0.10, 0.08, 0.02, 0.38]
        elif bug in {"Staphylococcus aureus", "Enterococcus faecalis"}:
            probs = [0.12, 0.10, 0.16, 0.12, 0.10, 0.30, 0.10]
        elif bug == "Pseudomonas aeruginosa":
            probs = [0.04, 0.04, 0.24, 0.24, 0.30, 0.02, 0.12]
        else:
            probs = [0.16, 0.22, 0.20, 0.15, 0.14, 0.02, 0.11]
        antibiotics.append(rng.choice(ANTIBIOTICS, p=probs))
    antibiotic = np.array(antibiotics)

    age = np.clip(rng.normal(loc=58, scale=18, size=n_rows).round(), 18, 95).astype(int)
    sex = rng.choice(SEXES, size=n_rows, p=[0.47, 0.53])
    prior_antibiotic_exposure = rng.binomial(1, p=0.28, size=n_rows)
    comorbidity_score = np.clip(rng.poisson(lam=1.8, size=n_rows), 0, 6).astype(int)

    base_resistance = rng.beta(a=2.2, b=5.0, size=n_rows)
    ward_resistance_adjustment = np.where(ward_type == "ICU", 0.16, 0.0)
    exposure_resistance_adjustment = prior_antibiotic_exposure * 0.08
    local_resistance_rate = np.clip(
        base_resistance + ward_resistance_adjustment + exposure_resistance_adjustment,
        0.02,
        0.95,
    )

    pair_base = np.array(
        [PAIR_ACTIVITY[(bug, drug)] for bug, drug in zip(pathogen, antibiotic, strict=False)]
    )

    logit = np.log(pair_base / (1 - pair_base))
    logit -= 2.1 * local_resistance_rate
    logit -= 0.75 * prior_antibiotic_exposure
    logit -= np.where(ward_type == "ICU", 0.75, 0.0)
    logit -= 0.16 * comorbidity_score
    logit -= np.maximum(age - 70, 0) * 0.012
    logit += np.where(infection_site == "urinary", 0.12, 0.0)
    logit += rng.normal(loc=0.0, scale=0.25, size=n_rows)

    probability = np.clip(_sigmoid(logit), 0.01, 0.99)
    susceptible = rng.binomial(1, probability, size=n_rows)

    start_date = np.datetime64("2023-01-01")
    end_date = np.datetime64("2025-12-31")
    day_offsets = rng.integers(0, int((end_date - start_date).astype(int)) + 1, size=n_rows)
    sample_dates = start_date + day_offsets.astype("timedelta64[D]")

    return pd.DataFrame(
        {
            "patient_id": [f"P{idx:07d}" for idx in range(1, n_rows + 1)],
            "age": age,
            "sex": sex,
            "ward_type": ward_type,
            "infection_site": infection_site,
            "pathogen": pathogen,
            "antibiotic": antibiotic,
            "prior_antibiotic_exposure": prior_antibiotic_exposure,
            "comorbidity_score": comorbidity_score,
            "local_resistance_rate": local_resistance_rate.round(3),
            "sample_date": pd.to_datetime(sample_dates).strftime("%Y-%m-%d"),
            "susceptible": susceptible,
        }
    )


def save_dataset(df: pd.DataFrame, output_path: str | Path) -> None:
    """Save generated data to CSV, creating parent directories if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic AMR-like data.")
    parser.add_argument("--n-rows", type=int, default=10_000, help="Number of rows to generate.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/synthetic_amr.csv"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_synthetic_amr(n_rows=args.n_rows, seed=args.seed)
    save_dataset(df, args.output)
    print(f"Generated {len(df):,} rows at {args.output}")


if __name__ == "__main__":
    main()
