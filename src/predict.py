import pandas as pd

from src.config import MODELS_DIR, MODEL_FILENAME, TARGET_COL
from src.features import build_features
from src.preprocess import prepare_base_dataframe
from src.utils import load_model


def predict_faults(df: pd.DataFrame) -> pd.DataFrame:
    """
    Load the trained model and generate fault predictions.
    """
    pipeline = load_model(MODELS_DIR / MODEL_FILENAME)

    df = prepare_base_dataframe(df)
    df = build_features(df)

    X = df.copy()
    if TARGET_COL in X.columns:
        X = X.drop(columns=[TARGET_COL])

    if "timestamp" in X.columns:
        X = X.drop(columns=["timestamp"])

    preds = pipeline.predict(X)

    result = df.copy()
    result["predicted_fault_type"] = preds
    return result
