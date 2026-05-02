from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.evaluate import calibration_summary, evaluate_slices, threshold_metrics


def test_threshold_metrics_returns_rows_for_each_threshold() -> None:
    rows = threshold_metrics(
        y_true=np.array([0, 0, 1, 1]),
        y_proba=np.array([0.1, 0.45, 0.7, 0.9]),
        thresholds=[0.4, 0.8],
    )
    assert [row["threshold"] for row in rows] == [0.4, 0.8]
    assert all("recall" in row for row in rows)


def test_calibration_summary_counts_all_rows() -> None:
    rows = calibration_summary(
        y_true=np.array([0, 1, 1]),
        y_proba=np.array([0.2, 0.6, 0.9]),
        n_bins=5,
    )
    assert sum(int(row["count"]) for row in rows) == 3


def test_evaluate_slices_handles_one_class_slice() -> None:
    df = pd.DataFrame({"ward_type": ["ICU", "ICU", "medical", "medical"]})
    rows = evaluate_slices(
        df=df,
        y_true=np.array([1, 1, 0, 1]),
        y_proba=np.array([0.8, 0.7, 0.2, 0.6]),
        slice_columns=["ward_type"],
        min_support=1,
    )
    assert "ward_type" in rows
    assert {row["slice_value"] for row in rows["ward_type"]} == {"ICU", "medical"}
