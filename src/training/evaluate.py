"""Reusable evaluation utilities for binary classifiers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLDS = [0.4, 0.5, 0.6, 0.8]


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Evaluate a binary classifier from true labels and positive-class probabilities."""
    y_true_array = np.asarray(y_true).astype(int)
    y_proba_array = np.asarray(y_proba).astype(float)
    y_pred = (y_proba_array >= threshold).astype(int)

    if len(np.unique(y_true_array)) < 2:
        roc_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true_array, y_proba_array))

    return {
        "roc_auc": roc_auc,
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true_array, y_proba_array)),
    }


def threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
) -> list[dict[str, float]]:
    """Evaluate classifier behavior across multiple decision thresholds."""
    y_true_array = np.asarray(y_true).astype(int)
    y_proba_array = np.asarray(y_proba).astype(float)
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        metrics = evaluate_binary_classifier(y_true_array, y_proba_array, threshold=threshold)
        y_pred = (y_proba_array >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "predicted_positive_rate": float(y_pred.mean()),
            }
        )
    return rows


def calibration_summary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    """Summarize observed outcome rates by predicted-probability bin."""
    y_true_array = np.asarray(y_true).astype(int)
    y_proba_array = np.asarray(y_proba).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float | int]] = []

    for index in range(n_bins):
        lower = bins[index]
        upper = bins[index + 1]
        if index == n_bins - 1:
            mask = (y_proba_array >= lower) & (y_proba_array <= upper)
        else:
            mask = (y_proba_array >= lower) & (y_proba_array < upper)

        count = int(mask.sum())
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "mean_predicted_probability": float(y_proba_array[mask].mean())
                if count
                else float("nan"),
                "observed_susceptible_rate": float(y_true_array[mask].mean())
                if count
                else float("nan"),
            }
        )
    return rows


def evaluate_slices(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    slice_columns: Sequence[str],
    threshold: float = 0.5,
    min_support: int = 30,
) -> dict[str, list[dict[str, Any]]]:
    """Compute per-slice metrics for categorical or bucketed columns.

    ROC-AUC is reported as NaN for slices that contain only one target class.
    """
    y_true_array = np.asarray(y_true).astype(int)
    y_proba_array = np.asarray(y_proba).astype(float)
    eval_df = df.copy()
    eval_df["_y_true"] = y_true_array
    eval_df["_y_proba"] = y_proba_array

    results: dict[str, list[dict[str, Any]]] = {}
    for column in slice_columns:
        if column not in eval_df.columns:
            continue

        rows: list[dict[str, Any]] = []
        for value, group in eval_df.groupby(column, dropna=False):
            support = int(len(group))
            metrics = evaluate_binary_classifier(
                group["_y_true"].to_numpy(),
                group["_y_proba"].to_numpy(),
                threshold=threshold,
            )
            rows.append(
                {
                    "slice_value": str(value),
                    "support": support,
                    "low_support": support < min_support,
                    "target_rate": float(group["_y_true"].mean()),
                    "roc_auc": metrics["roc_auc"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                }
            )

        results[column] = sorted(rows, key=lambda row: row["support"], reverse=True)
    return results
