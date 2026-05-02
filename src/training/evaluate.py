"""Reusable evaluation metrics for binary classifiers."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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
