"""Train the baseline treatment-response model."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.split_data import stratified_train_test_split
from src.features.build_features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    get_preprocessor,
    split_features_target,
)
from src.training.evaluate import (
    calibration_summary,
    evaluate_binary_classifier,
    evaluate_slices,
    threshold_metrics,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MODEL_TYPE = "LogisticRegression"
MLFLOW_EXPERIMENT_NAME = "medmlops-treatment-response-platform"


def _format_float(value: Any) -> str:
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.4f}"


def _add_metric_table(lines: list[str], metrics: dict[str, float]) -> None:
    lines.extend(["| Metric | Value |", "| --- | ---: |"])
    for metric, value in metrics.items():
        lines.append(f"| {metric} | {_format_float(value)} |")


def _add_threshold_table(lines: list[str], rows: list[dict[str, float]]) -> None:
    lines.extend(
        [
            "| Threshold | Accuracy | Precision | Recall | F1 | Predicted Positive Rate |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {threshold:.2f} | {accuracy:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {predicted_positive_rate:.4f} |".format(
                **row
            )
        )


def _add_calibration_table(lines: list[str], rows: list[dict[str, float | int]]) -> None:
    lines.extend(
        [
            "| Probability Bin | Count | Mean Predicted Probability | Observed Susceptible Rate |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        bin_label = f"{row['bin_lower']:.1f}-{row['bin_upper']:.1f}"
        lines.append(
            f"| {bin_label} | {row['count']} | {_format_float(row['mean_predicted_probability'])} | {_format_float(row['observed_susceptible_rate'])} |"
        )


def _add_slice_tables(lines: list[str], slice_rows: dict[str, list[dict[str, Any]]]) -> None:
    for column, rows in slice_rows.items():
        lines.extend(
            [
                "",
                f"### {column}",
                "",
                "| Slice | Support | Low Support | Target Rate | ROC-AUC | Accuracy | Precision | Recall | F1 |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                "| {slice_value} | {support} | {low_support} | {target_rate} | {roc_auc} | {accuracy} | {precision} | {recall} | {f1} |".format(
                    slice_value=row["slice_value"],
                    support=row["support"],
                    low_support="yes" if row["low_support"] else "no",
                    target_rate=_format_float(row["target_rate"]),
                    roc_auc=_format_float(row["roc_auc"]),
                    accuracy=_format_float(row["accuracy"]),
                    precision=_format_float(row["precision"]),
                    recall=_format_float(row["recall"]),
                    f1=_format_float(row["f1"]),
                )
            )


def _write_evaluation_report(
    metrics: dict[str, float],
    threshold_rows: list[dict[str, float]],
    calibration_rows: list[dict[str, float | int]],
    slice_rows: dict[str, list[dict[str, Any]]],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Evaluation Report",
        "",
        "This report was generated from the held-out synthetic test split.",
        "",
        "## Global Metrics",
        "",
    ]
    _add_metric_table(lines, metrics)
    lines.extend(
        [
            "",
            "## Threshold Analysis",
            "",
            "Recommendation levels in the API are threshold-based demonstration labels. This table shows how standard classification metrics change across candidate thresholds.",
            "",
        ]
    )
    _add_threshold_table(lines, threshold_rows)
    lines.extend(
        [
            "",
            "## Calibration Summary",
            "",
            "Calibration is summarized by comparing average predicted probability to observed susceptible rate in probability bins. This is a synthetic-data diagnostic, not clinical validation.",
            "",
        ]
    )
    _add_calibration_table(lines, calibration_rows)
    lines.extend(["", "## Slice Evaluation", ""])
    _add_slice_tables(lines, slice_rows)
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- ROC-AUC may be `n/a` for small slices containing only one target class.",
            "- Brier score is included as a probability-quality metric; lower is better.",
            "- Slice metrics are included to demonstrate healthcare ML monitoring discipline, not to claim subgroup safety.",
            "",
            "## Disclaimer",
            "",
            "This project is an educational prototype. It is not a medical device, not clinically validated, and must not be used for clinical decision-making.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_evaluation_json(
    metrics: dict[str, float],
    threshold_rows: list[dict[str, float]],
    calibration_rows: list[dict[str, float | int]],
    slice_rows: dict[str, list[dict[str, Any]]],
    output_path: str | Path,
) -> None:
    _write_json(
        {
            "global_metrics": metrics,
            "threshold_metrics": threshold_rows,
            "calibration_summary": calibration_rows,
            "slice_metrics": slice_rows,
        },
        output_path,
    )


def _build_reference_profile(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "target_rate": float(df[TARGET].mean()) if TARGET in df.columns else None,
        "missingness": df.isna().mean().round(6).to_dict(),
        "numeric": {
            column: {
                "mean": float(df[column].mean()),
                "std": float(df[column].std()),
                "min": float(df[column].min()),
                "max": float(df[column].max()),
            }
            for column in NUMERIC_FEATURES
            if column in df.columns
        },
        "categorical": {
            column: df[column].value_counts(normalize=True).round(6).to_dict()
            for column in CATEGORICAL_FEATURES
            if column in df.columns
        },
    }


def _write_json(payload: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _maybe_log_mlflow(
    pipeline: Pipeline,
    metrics: dict[str, float],
    params: dict[str, Any],
    artifact_paths: list[Path],
) -> str | None:
    """Log a run to MLflow when the optional dependency is installed."""
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError:
        return None

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="baseline-logistic-regression") as run:
        mlflow.log_params(params)
        for metric, value in metrics.items():
            if isinstance(value, float) and math.isfinite(value):
                mlflow.log_metric(metric, value)
        for artifact_path in artifact_paths:
            if artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        return run.info.run_id


def _add_evaluation_slices(test_df: pd.DataFrame) -> pd.DataFrame:
    df = test_df.copy()
    df["comorbidity_bucket"] = pd.cut(
        df["comorbidity_score"],
        bins=[-1, 0, 2, 4, 6],
        labels=["0", "1-2", "3-4", "5-6"],
    ).astype(str)
    return df


def train_model(
    input_path: str | Path = "data/processed/cohort.csv",
    model_output: str | Path = "models/treatment_response_model.joblib",
    evaluation_output: str | Path = "reports/evaluation_report.md",
    evaluation_json_output: str | Path = "reports/evaluation_report.json",
    predictions_output: str | Path = "data/processed/test_predictions.csv",
    reference_output: str | Path = "data/reference/reference_batch.csv",
    reference_profile_output: str | Path = "data/reference/reference_profile.json",
    model_metadata_output: str | Path = "models/model_metadata.json",
) -> dict[str, float]:
    """Train, evaluate, and persist the baseline model."""
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError("Training input is empty.")

    train_df, test_df = stratified_train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
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
    threshold_rows = threshold_metrics(y_test.to_numpy(), y_proba)
    calibration_rows = calibration_summary(y_test.to_numpy(), y_proba)
    slice_eval_df = _add_evaluation_slices(test_df)
    slice_rows = evaluate_slices(
        slice_eval_df,
        y_test.to_numpy(),
        y_proba,
        slice_columns=[
            "ward_type",
            "pathogen",
            "antibiotic",
            "prior_antibiotic_exposure",
            "comorbidity_bucket",
        ],
    )

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

    _write_json(_build_reference_profile(train_df), reference_profile_output)
    _write_evaluation_report(
        metrics=metrics,
        threshold_rows=threshold_rows,
        calibration_rows=calibration_rows,
        slice_rows=slice_rows,
        output_path=evaluation_output,
    )
    _write_evaluation_json(
        metrics=metrics,
        threshold_rows=threshold_rows,
        calibration_rows=calibration_rows,
        slice_rows=slice_rows,
        output_path=evaluation_json_output,
    )

    metadata = {
        "model_name": "synthetic-treatment-response-baseline",
        "model_type": MODEL_TYPE,
        "model_path": str(model_path),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "n_rows": int(len(df)),
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "train_target_rate": float(y_train.mean()),
        "test_target_rate": float(y_test.mean()),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "target": TARGET,
        "metrics": metrics,
        "recommendation_thresholds": {
            "strong candidate": "p >= 0.8",
            "candidate": "0.6 <= p < 0.8",
            "uncertain": "0.4 <= p < 0.6",
            "not recommended": "p < 0.4",
        },
    }

    mlflow_run_id = _maybe_log_mlflow(
        pipeline=pipeline,
        metrics=metrics,
        params={
            "model_type": MODEL_TYPE,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_rows": len(df),
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "numeric_features": ",".join(NUMERIC_FEATURES),
            "categorical_features": ",".join(CATEGORICAL_FEATURES),
            "target": TARGET,
        },
        artifact_paths=[
            Path(evaluation_output),
            Path(evaluation_json_output),
            Path(predictions_output),
            Path(reference_profile_output),
            Path("reports/model_card.md"),
        ],
    )
    metadata["mlflow_run_id"] = mlflow_run_id
    _write_json(metadata, model_metadata_output)
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
