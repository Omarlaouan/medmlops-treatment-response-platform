"""Train the baseline treatment-response model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.split_data import stratified_train_test_split
from src.features.build_features import TARGET, get_preprocessor, split_features_target
from src.training.evaluate import evaluate_binary_classifier


def _write_evaluation_report(metrics: dict[str, float], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Report",
        "",
        "This report was generated from the held-out synthetic test split.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for metric, value in metrics.items():
        lines.append(f"| {metric} | {value:.4f} |")
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


def train_model(
    input_path: str | Path = "data/processed/cohort.csv",
    model_output: str | Path = "models/treatment_response_model.joblib",
    evaluation_output: str | Path = "reports/evaluation_report.md",
    predictions_output: str | Path = "data/processed/test_predictions.csv",
    reference_output: str | Path = "data/reference/reference_batch.csv",
) -> dict[str, float]:
    """Train, evaluate, and persist the baseline model."""
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Training input is empty.")

    train_df, test_df = stratified_train_test_split(df)
    X_train, y_train = split_features_target(train_df)
    X_test, y_test = split_features_target(test_df)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            (
                "model",
                LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary_classifier(y_test.to_numpy(), y_proba)

    model_path = Path(model_output)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)

    predictions_path = Path(predictions_output)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions = test_df.copy()
    test_predictions["predicted_probability_of_susceptibility"] = y_proba
    test_predictions["predicted_susceptible"] = (y_proba >= 0.5).astype(int)
    test_predictions.to_csv(predictions_path, index=False)

    reference_path = Path(reference_output)
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(reference_path, index=False)

    _write_evaluation_report(metrics, evaluation_output)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline treatment-response model.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/cohort.csv"))
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("models/treatment_response_model.joblib"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_model(input_path=args.input, model_output=args.model_output)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
