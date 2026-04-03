import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd


def ensure_directories(paths):
    """
    Create directories if they do not exist.
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def save_json(data: Dict[str, Any], filepath):
    """
    Save a dictionary to a JSON file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def save_dataframe(df: pd.DataFrame, filepath):
    """
    Save a DataFrame to CSV.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath):
    """
    Save a trained model/pipeline using joblib.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """
    Load a saved joblib model/pipeline.
    """
    return joblib.load(filepath)


def print_header(title: str):
    line = "=" * 70
    print(f"\n{line}\n{title}\n{line}")


def infer_existing_columns(df: pd.DataFrame, columns):
    """
    Return only columns that exist in the DataFrame.
    """
    return [col for col in columns if col in df.columns]
