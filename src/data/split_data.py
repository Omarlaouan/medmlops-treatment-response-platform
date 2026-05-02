"""Reusable train/test split helper."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features.build_features import TARGET


def stratified_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return stratified train and test dataframes."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET],
    )
    return train_df.copy(), test_df.copy()
