from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from src.config import RAW_PARQUET_PATH, TIMESTAMP_COL
from src.utils import print_header


def load_data(
    filepath: Optional[Path] = None,
    columns: Optional[Sequence[str]] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Load the chemical process dataset from a Parquet file.

    Parameters
    ----------
    filepath : Path, optional
        Path to the parquet file. Defaults to config.RAW_PARQUET_PATH.
    columns : sequence of str, optional
        Columns to load.
    sample_frac : float, optional
        Optional sample fraction for faster experimentation.
    random_state : int
        Random seed for sampling.

    Returns
    -------
    pd.DataFrame
    """
    filepath = filepath or RAW_PARQUET_PATH

    if not Path(filepath).exists():
        raise FileNotFoundError(
            f"Parquet file not found at: {filepath}\n"
            "Run your setup_data.sh first to download and convert the dataset."
        )

    print_header("Loading dataset")
    df = pd.read_parquet(filepath, columns=columns)

    if TIMESTAMP_COL in df.columns:
        df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], errors="coerce")

    if sample_frac is not None:
        if not (0 < sample_frac <= 1):
            raise ValueError("sample_frac must be between 0 and 1.")
        df = df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

    print(f"Loaded shape: {df.shape}")
    return df


def basic_overview(df: pd.DataFrame):
    """
    Print a quick overview of the dataset.
    """
    print_header("Basic dataset overview")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nDtypes:")
    print(df.dtypes)
    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False).head(20))
