"""
preprocessing.py
----------------
Data preprocessing pipeline for the Customer Churn Prediction System.
Handles missing values, encoding, scaling, and feature engineering.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

CATEGORICAL_COLS = [
    "gender", "partner", "dependents", "phone_service", "multiple_lines",
    "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "contract",
    "paperless_billing", "payment_method"
]

NUMERIC_COLS = ["tenure", "monthly_charges", "total_charges"]

TARGET_COL = "churn"

DROP_COLS = ["customer_id", "senior_citizen"]  # senior_citizen already binary


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing Functions
# ──────────────────────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from a CSV file."""
    logger.info(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Dataset loaded — shape: {df.shape}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing and malformed values.
    - total_charges can be blank strings (new customers with 0 tenure)
    - Fill numeric NaNs with median, categorical NaNs with mode
    """
    logger.info("Handling missing values …")
    df = df.copy()

    # total_charges sometimes stored as ' ' (space) — convert to NaN then fill
    if "total_charges" in df.columns:
        df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")
        missing_tc = df["total_charges"].isna().sum()
        if missing_tc > 0:
            logger.info(f"  Filling {missing_tc} missing total_charges with 0.0 (new customers)")
            df["total_charges"].fillna(0.0, inplace=True)

    # Numeric columns — fill with median
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isna().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"  Filled numeric NaN in '{col}' with median={median_val:.2f}")

    # Categorical columns — fill with mode
    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isna().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"  Filled categorical NaN in '{col}' with mode='{mode_val}'")

    logger.info("Missing value handling complete.")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert 'churn' column from Yes/No strings to 1/0 integer."""
    df = df.copy()
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0})
        logger.info(f"Target encoded — churn distribution:\n{df[TARGET_COL].value_counts().to_dict()}")
    return df


def encode_categoricals(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    Label-encode all categorical columns.

    Parameters
    ----------
    df       : input DataFrame
    encoders : dict of {col: LabelEncoder} — supply when transforming (not fitting)
    fit      : if True, fit new encoders; else use supplied encoders

    Returns
    -------
    df_encoded, encoders
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders.get(col)
            if le is None:
                raise ValueError(f"No encoder found for column '{col}'")
            # Handle unseen labels gracefully
            known_classes = set(le.classes_)
            df[col] = df[col].astype(str).apply(
                lambda x: x if x in known_classes else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    logger.info(f"Encoded {len(CATEGORICAL_COLS)} categorical columns.")
    return df, encoders


def scale_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    """
    Standard-scale numeric feature columns.

    Returns
    -------
    df_scaled, scaler
    """
    df = df.copy()
    cols_to_scale = [c for c in NUMERIC_COLS if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        logger.info(f"Fitted and applied StandardScaler to {cols_to_scale}")
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit=False.")
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])
        logger.info(f"Applied existing StandardScaler to {cols_to_scale}")

    return df, scaler


def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns not needed for training."""
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    logger.info(f"Dropped columns: {cols_to_drop}")
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering: add derived features to improve model performance.
    - avg_monthly_revenue  : total_charges / (tenure + 1)  — avoids div-by-zero
    - high_monthly_charges : 1 if monthly_charges > 75
    """
    df = df.copy()

    if "total_charges" in df.columns and "tenure" in df.columns:
        df["avg_monthly_revenue"] = df["total_charges"] / (df["tenure"] + 1)

    if "monthly_charges" in df.columns:
        # Use raw threshold before scaling; if already scaled this is approximate
        df["high_monthly_charges"] = (df["monthly_charges"] > 0).astype(int)

    logger.info("Feature engineering applied.")
    return df


def full_preprocessing_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    End-to-end preprocessing pipeline.

    Returns
    -------
    X_train, X_test, y_train, y_test, encoders, scaler, feature_names
    """
    logger.info("=== Starting full preprocessing pipeline ===")

    df = handle_missing_values(df)
    df = encode_target(df)
    df = drop_unused_columns(df)
    df = add_engineered_features(df)
    df, encoders = encode_categoricals(df, fit=True)

    # Separate features / target
    X = df.drop(columns=[TARGET_COL], errors="ignore")
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    X, scaler = scale_features(X, fit=True)

    # Train / test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        f"Pipeline complete — Train: {X_train.shape}, Test: {X_test.shape}, "
        f"Churn rate: {y.mean():.2%}"
    )

    return X_train, X_test, y_train, y_test, encoders, scaler, feature_names


def preprocess_single_record(record: dict, encoders: dict, scaler: StandardScaler) -> np.ndarray:
    """
    Transform a single customer record (dict) into the model's input format.
    Used at inference time by the FastAPI endpoint.
    """
    df = pd.DataFrame([record])
    df = handle_missing_values(df)
    df = add_engineered_features(df)
    df, _ = encode_categoricals(df, encoders=encoders, fit=False)

    # Align columns — drop target/unused, fill missing engineered cols with 0
    for col in DROP_COLS + [TARGET_COL]:
        df.drop(columns=[col], errors="ignore", inplace=True)

    df, _ = scale_features(df, scaler=scaler, fit=False)
    return df.values


def save_artifacts(encoders: dict, scaler: StandardScaler, feature_names: list, save_dir: str):
    """Persist preprocessing artifacts alongside the model."""
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(encoders, os.path.join(save_dir, "encoders.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(save_dir, "feature_names.pkl"))
    logger.info(f"Preprocessing artifacts saved to '{save_dir}'")


def load_artifacts(save_dir: str):
    """Load persisted preprocessing artifacts."""
    encoders     = joblib.load(os.path.join(save_dir, "encoders.pkl"))
    scaler       = joblib.load(os.path.join(save_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(save_dir, "feature_names.pkl"))
    logger.info(f"Preprocessing artifacts loaded from '{save_dir}'")
    return encoders, scaler, feature_names
