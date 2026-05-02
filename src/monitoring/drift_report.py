"""Simple drift monitoring report for synthetic AMR cohorts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


NUMERIC_COLUMNS = ["age", "local_resistance_rate", "comorbidity_score"]
CATEGORICAL_COLUMNS = ["pathogen", "antibiotic", "ward_type"]
TARGET = "susceptible"


def _numeric_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float,
) -> tuple[list[dict[str, float | str]], list[str]]:
    rows: list[dict[str, float | str]] = []
    alerts: list[str] = []
    for column in NUMERIC_COLUMNS:
        ref_mean = float(reference[column].mean())
        cur_mean = float(current[column].mean())
        ref_std = float(reference[column].std())
        cur_std = float(current[column].std())
        mean_diff = cur_mean - ref_mean
        rows.append(
            {
                "column": column,
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "mean_difference": mean_diff,
                "reference_std": ref_std,
                "current_std": cur_std,
            }
        )
        if abs(mean_diff) > threshold:
            alerts.append(
                f"Numeric drift alert: {column} mean changed by {mean_diff:.3f}, above threshold {threshold:.3f}."
            )
    return rows, alerts


def _categorical_summary(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    alerts: list[str] = []
    for column in CATEGORICAL_COLUMNS:
        ref_dist = reference[column].value_counts(normalize=True).to_dict()
        cur_dist = current[column].value_counts(normalize=True).to_dict()
        ref_categories = set(ref_dist)
        cur_categories = set(cur_dist)
        new_categories = sorted(cur_categories - ref_categories)
        max_distribution_delta = 0.0
        for category in ref_categories | cur_categories:
            max_distribution_delta = max(
                max_distribution_delta,
                abs(float(cur_dist.get(category, 0.0)) - float(ref_dist.get(category, 0.0))),
            )
        rows.append(
            {
                "column": column,
                "reference_categories": sorted(ref_categories),
                "current_categories": sorted(cur_categories),
                "new_categories": new_categories,
                "max_distribution_delta": max_distribution_delta,
            }
        )
        if new_categories:
            alerts.append(f"New category alert in {column}: {new_categories}.")
    return rows, alerts


def generate_drift_report(
    reference_path: str | Path,
    current_path: str | Path,
    output_path: str | Path = "reports/drift_report.md",
    mean_threshold: float = 0.10,
    target_rate_threshold: float = 0.05,
) -> dict[str, Any]:
    """Compare reference and current data and write a markdown drift report."""
    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    numeric_rows, numeric_alerts = _numeric_summary(reference, current, mean_threshold)
    categorical_rows, categorical_alerts = _categorical_summary(reference, current)
    alerts = numeric_alerts + categorical_alerts

    target_summary: dict[str, float] | None = None
    if TARGET in reference.columns and TARGET in current.columns:
        reference_rate = float(reference[TARGET].mean())
        current_rate = float(current[TARGET].mean())
        target_delta = current_rate - reference_rate
        target_summary = {
            "reference_target_rate": reference_rate,
            "current_target_rate": current_rate,
            "target_rate_delta": target_delta,
        }
        if abs(target_delta) > target_rate_threshold:
            alerts.append(
                f"Target-rate drift alert: susceptible rate changed by {target_delta:.3f}, above threshold {target_rate_threshold:.3f}."
            )

    report = {
        "numeric_summary": numeric_rows,
        "categorical_summary": categorical_rows,
        "target_summary": target_summary,
        "alerts": alerts,
    }
    _write_markdown_report(report, output_path)
    return report


def _write_markdown_report(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Drift Report",
        "",
        "This report compares a reference training batch to a current synthetic cohort.",
        "",
        "## Numeric Drift",
        "",
        "| Column | Reference Mean | Current Mean | Mean Difference | Reference Std | Current Std |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["numeric_summary"]:
        lines.append(
            "| {column} | {reference_mean:.4f} | {current_mean:.4f} | {mean_difference:.4f} | {reference_std:.4f} | {current_std:.4f} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Categorical Drift",
            "",
            "| Column | New Categories | Max Distribution Delta |",
            "| --- | --- | ---: |",
        ]
    )
    for row in report["categorical_summary"]:
        lines.append(
            f"| {row['column']} | {row['new_categories']} | {row['max_distribution_delta']:.4f} |"
        )

    lines.extend(["", "## Target Rate", ""])
    if report["target_summary"] is None:
        lines.append("Target column unavailable in one of the compared datasets.")
    else:
        target = report["target_summary"]
        lines.extend(
            [
                "| Reference Susceptible Rate | Current Susceptible Rate | Delta |",
                "| ---: | ---: | ---: |",
                f"| {target['reference_target_rate']:.4f} | {target['current_target_rate']:.4f} | {target['target_rate_delta']:.4f} |",
            ]
        )

    lines.extend(["", "## Alerts", ""])
    if report["alerts"]:
        lines.extend(f"- {alert}" for alert in report["alerts"])
    else:
        lines.append("- No simple drift alerts triggered.")

    lines.extend(
        [
            "",
            "## Disclaimer",
            "",
            "This project is an educational prototype. It is not a medical device, not clinically validated, and must not be used for clinical decision-making.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simple drift report.")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/drift_report.md"))
    parser.add_argument("--mean-threshold", type=float, default=0.10)
    parser.add_argument("--target-rate-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_drift_report(
        reference_path=args.reference,
        current_path=args.current,
        output_path=args.output,
        mean_threshold=args.mean_threshold,
        target_rate_threshold=args.target_rate_threshold,
    )
    print(f"Generated drift report at {args.output}")
    if report["alerts"]:
        print("Alerts:")
        for alert in report["alerts"]:
            print(f"- {alert}")


if __name__ == "__main__":
    main()
