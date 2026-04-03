from typing import List, Tuple

import numpy as np
import pandas as pd

from src.config import (
    TARGET_COL,
    TIMESTAMP_COL,
    GROUP_COLS,
    OPTIONAL_DROP_COLS,
    CATEGORICAL_COLS,
)
from src.utils import infer_existing_columns


def sort_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort the dataset by grouping columns and timestamp.
    """
    sort_cols = infer_existing_columns(df, GROUP_COLS + [TIMESTAMP_COL])
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    return df


def clean_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize target labels to string form and clean whitespace.
    """
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
    return df


def drop_rows_with_missing_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where target is missing.
    """
    if TARGET_COL in df.columns:
        df = df[df[TARGET_COL].notna()].copy()
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values:
    - categorical: mode or 'Unknown'
    - numeric: forward-fill within reactor, then median fallback

    This is a practical baseline strategy for process monitoring data.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [
        col for col in df.columns
        if col not in numeric_cols and col != TIMESTAMP_COL
    ]

    # Forward fill numeric values within reactor groups (if available)
    if all(col in df.columns for col in GROUP_COLS):
        for col in numeric_cols:
            df[col] = df.groupby(GROUP_COLS)[col].transform(lambda s: s.ffill())
    else:
        for col in numeric_cols:
            df[col] = df[col].ffill()

    # Median fallback for remaining numeric nulls
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    # Fill categorical columns
    for col in categorical_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[col] = df[col].fillna(fill_value)

    return df


def cast_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast selected columns to categorical/string-safe form.
    """
    df = df.copy()
    for col in infer_existing_columns(df, CATEGORICAL_COLS):
        df[col] = df[col].astype("category")
    return df


def remove_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that may directly leak future fault information into the classifier.
    """
    df = df.copy()
    leakage_cols = infer_existing_columns(df, OPTIONAL_DROP_COLS)
    if leakage_cols:
        df = df.drop(columns=leakage_cols)
    return df


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end preprocessing before feature engineering.
    """
    df = sort_time_series(df)
    df = clean_target_labels(df)
    df = drop_rows_with_missing_target(df)
    df = remove_leakage_columns(df)
    df = fill_missing_values(df)
    df = cast_categories(df)
    return df


def time_based_split(
    df: pd.DataFrame,
    test_size: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data in time order:
    - earliest rows -> train
    - latest rows -> test

    Assumes the data has already been sorted by time.
    """
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def get_feature_target_matrices(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split into X and y.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    return X, y
