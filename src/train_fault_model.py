import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    RAW_PARQUET_PATH,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE,
    MODEL_FILENAME,
    METRICS_FILENAME,
    PREDICTIONS_FILENAME,
    CONFUSION_MATRIX_FIG,
    FEATURE_IMPORTANCE_FIG,
    OUTPUTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
)
from src.data_loader import load_data, basic_overview
from src.evaluate import (
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
)
from src.features import build_features
from src.preprocess import (
    prepare_base_dataframe,
    time_based_split,
    get_feature_target_matrices,
)
from src.utils import (
    ensure_directories,
    infer_existing_columns,
    print_header,
    save_dataframe,
    save_json,
    save_model,
)


def build_preprocessor(X: pd.DataFrame):
    """
    Build a preprocessing transformer.
    - numeric features pass through
    - categorical features are one-hot encoded
    """
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    return preprocessor, categorical_cols, numeric_cols


def build_model():
    """
    Baseline tree-based classifier with class balancing.
    """
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=16,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return model


def main():
    ensure_directories([
        OUTPUTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        MODELS_DIR,
        PREDICTIONS_DIR,
    ])

    print_header("Chemical Process Fault Classification Pipeline")

    df = load_data(RAW_PARQUET_PATH)
    basic_overview(df)

    print_header("Preprocessing base dataframe")
    df = prepare_base_dataframe(df)

    print_header("Feature engineering")
    df = build_features(df)

    print_header("Time-based split")
    train_df, test_df = time_based_split(df, test_size=TEST_SIZE)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    X_train, y_train = get_feature_target_matrices(train_df, target_col=TARGET_COL)
    X_test, y_test = get_feature_target_matrices(test_df, target_col=TARGET_COL)

    # Remove timestamp after feature extraction
    drop_after_features = infer_existing_columns(X_train, ["timestamp"])
    if drop_after_features:
        X_train = X_train.drop(columns=drop_after_features)
        X_test = X_test.drop(columns=drop_after_features)

    preprocessor, categorical_cols, numeric_cols = build_preprocessor(X_train)
    classifier = build_model()

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    print_header("Training model")
    pipeline.fit(X_train, y_train)

    print_header("Generating predictions")
    y_pred = pipeline.predict(X_test)

    print_header("Evaluating model")
    metrics = compute_classification_metrics(y_test, y_pred)
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Weighted F1:       {metrics['weighted_f1']:.4f}")
    print(f"Macro F1:          {metrics['macro_f1']:.4f}")

    print_header("Saving predictions")
    pred_df = X_test.copy()
    pred_df[TARGET_COL] = y_test.values
    pred_df["predicted_fault_type"] = y_pred
    save_dataframe(pred_df, PREDICTIONS_DIR / PREDICTIONS_FILENAME)

    print_header("Saving metrics")
    save_json(metrics, METRICS_DIR / METRICS_FILENAME)

    print_header("Saving model")
    save_model(pipeline, MODELS_DIR / MODEL_FILENAME)

    print_header("Saving plots")
    labels = sorted(y_test.astype(str).unique().tolist())
    plot_confusion_matrix(
        y_true=y_test.astype(str),
        y_pred=pd.Series(y_pred).astype(str),
        labels=labels,
        save_path=FIGURES_DIR / CONFUSION_MATRIX_FIG,
    )

    # Feature names after preprocessing
    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    plot_feature_importance(
        model=pipeline,
        feature_names=feature_names.tolist(),
        save_path=FIGURES_DIR / FEATURE_IMPORTANCE_FIG,
        top_n=20,
    )

    print_header("Done")
    print("Artifacts saved to outputs/ successfully.")


if __name__ == "__main__":
    main()
