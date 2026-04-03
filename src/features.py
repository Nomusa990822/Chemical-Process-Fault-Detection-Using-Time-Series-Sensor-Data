from typing import List

import numpy as np
import pandas as pd

from src.config import (
    GROUP_COLS,
    TIMESTAMP_COL,
    TS_FEATURE_CANDIDATES,
    LAG_STEPS,
    ROLLING_WINDOWS,
)
from src.utils import infer_existing_columns


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar/time-based features from timestamp.
    """
    df = df.copy()

    if TIMESTAMP_COL not in df.columns:
        return df

    ts = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["dayofweek"] = ts.dt.dayofweek
    df["month"] = ts.dt.month

    return df


def add_lag_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Add lag features by reactor/group.
    """
    df = df.copy()
    existing_cols = infer_existing_columns(df, columns)

    if not existing_cols:
        return df

    if all(col in df.columns for col in GROUP_COLS):
        for col in existing_cols:
            for lag in LAG_STEPS:
                df[f"{col}_lag_{lag}"] = df.groupby(GROUP_COLS)[col].shift(lag)
    else:
        for col in existing_cols:
            for lag in LAG_STEPS:
                df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df


def add_rolling_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Add rolling mean and std features by reactor/group.
    """
    df = df.copy()
    existing_cols = infer_existing_columns(df, columns)

    if not existing_cols:
        return df

    if all(col in df.columns for col in GROUP_COLS):
        for col in existing_cols:
            for window in ROLLING_WINDOWS:
                grouped = df.groupby(GROUP_COLS)[col]
                df[f"{col}_rollmean_{window}"] = (
                    grouped.transform(lambda s: s.rolling(window, min_periods=1).mean())
                )
                df[f"{col}_rollstd_{window}"] = (
                    grouped.transform(lambda s: s.rolling(window, min_periods=1).std())
                )
    else:
        for col in existing_cols:
            for window in ROLLING_WINDOWS:
                df[f"{col}_rollmean_{window}"] = df[col].rolling(window, min_periods=1).mean()
                df[f"{col}_rollstd_{window}"] = df[col].rolling(window, min_periods=1).std()

    return df


def add_rate_of_change_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Add first-difference features by reactor/group.
    """
    df = df.copy()
    existing_cols = infer_existing_columns(df, columns)

    if not existing_cols:
        return df

    if all(col in df.columns for col in GROUP_COLS):
        for col in existing_cols:
            df[f"{col}_diff_1"] = df.groupby(GROUP_COLS)[col].diff(1)
    else:
        for col in existing_cols:
            df[f"{col}_diff_1"] = df[col].diff(1)

    return df


def fill_feature_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NaNs introduced by lag/rolling/diff features.
    """
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(0)

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    """
    df = df.copy()

    df = add_time_features(df)
    df = add_lag_features(df, TS_FEATURE_CANDIDATES)
    df = add_rolling_features(df, TS_FEATURE_CANDIDATES)
    df = add_rate_of_change_features(df, TS_FEATURE_CANDIDATES)
    df = fill_feature_nans(df)

    return df
