from pathlib import Path


# =========================================================
# Project Paths
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

RAW_PARQUET_PATH = DATA_DIR / "chemical_process.parquet"


# =========================================================
# Columns
# =========================================================
TARGET_COL = "fault_type"
TIMESTAMP_COL = "timestamp"
GROUP_COLS = ["reactor_id"]

# Candidate columns for supervised learning
DROP_COLS_BASE = [
    TARGET_COL,
    TIMESTAMP_COL,
]

# Optional columns that may exist
OPTIONAL_DROP_COLS = [
    "time_to_fault_min",   # regression target / leakage risk for classification
]

# Columns commonly useful as IDs / categories
CATEGORICAL_COLS = [
    "operating_regime",
]

# We keep reactor_id numeric by default, but you can make it categorical if you want.
ID_NUMERIC_COLS = [
    "reactor_id",
]

# Columns that often behave like setpoints / flags
POTENTIAL_CATEGORICAL_NUMERIC = [
    "temperature_setpoint_C",
    "pressure_setpoint_bar",
]


# =========================================================
# Feature Engineering
# =========================================================
LAG_STEPS = [1, 3, 5]
ROLLING_WINDOWS = [5, 15]

# Selected numeric columns for time-series features
TS_FEATURE_CANDIDATES = [
    "reactor_temp_C",
    "pressure_bar",
    "feed_flow_rate_kgph",
    "coolant_flow_rate_kgph",
    "agitator_speed_rpm",
    "reaction_rate",
    "conversion_pct",
    "selectivity_pct",
    "yield_pct",
    "vibration_mm_s",
    "motor_current_A",
    "power_consumption_kW",
    "efficiency_loss_pct",
]


# =========================================================
# Training
# =========================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20

MODEL_FILENAME = "fault_classifier.joblib"
METRICS_FILENAME = "fault_classification_metrics.json"
PREDICTIONS_FILENAME = "fault_predictions.csv"

CONFUSION_MATRIX_FIG = "confusion_matrix.png"
FEATURE_IMPORTANCE_FIG = "feature_importance.png"
CLASS_DISTRIBUTION_FIG = "class_distribution.png"
