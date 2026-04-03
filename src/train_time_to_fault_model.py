import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    RAW_PARQUET_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    OUTPUTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
)
from src.data_loader import load_data
from src.features import build_features
from src.preprocess import (
    sort_time_series,
    fill_missing_values,
    cast_categories,
    time_based_split,
)
from src.utils import ensure_directories, print_header, save_json, save_model


TARGET_REG_COL = "time_to_fault_min"
MODEL_NAME = "time_to_fault_regressor.joblib"
METRICS_NAME = "time_to_fault_metrics.json"


def main():
    ensure_directories([OUTPUTS_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR])

    print_header("Training Time-to-Fault Regressor")

    df = load_data(RAW_PARQUET_PATH)

    if TARGET_REG_COL not in df.columns:
        raise ValueError(f"{TARGET_REG_COL} not found in dataset.")

    # Keep only rows with a known time-to-fault
    df = df[df[TARGET_REG_COL].notna()].copy()

    df = sort_time_series(df)
    df = fill_missing_values(df)
    df = cast_categories(df)
    df = build_features(df)

    train_df, test_df = time_based_split(df, test_size=TEST_SIZE)

    X_train = train_df.drop(columns=[TARGET_REG_COL])
    y_train = train_df[TARGET_REG_COL]

    X_test = test_df.drop(columns=[TARGET_REG_COL])
    y_test = test_df[TARGET_REG_COL]

    if "timestamp" in X_train.columns:
        X_train = X_train.drop(columns=["timestamp"])
        X_test = X_test.drop(columns=["timestamp"])

    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=14,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "r2": float(r2_score(y_test, preds)),
    }

    print(f"MAE:  {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2:   {metrics['r2']:.4f}")

    save_json(metrics, METRICS_DIR / METRICS_NAME)
    save_model(pipeline, MODELS_DIR / MODEL_NAME)

    print("Time-to-fault model training complete.")


if __name__ == "__main__":
    main()
