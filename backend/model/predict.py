"""
predict.py
----------
Inference module for the Customer Churn Prediction System.
Loads the trained model and preprocessing artifacts, then
returns churn predictions with probability scores.
"""

import os
import json
import logging
import numpy as np
import joblib

from preprocessing import load_artifacts, preprocess_single_record

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../.."))
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")


# ──────────────────────────────────────────────────────────────────────────────
# Predictor Class
# ──────────────────────────────────────────────────────────────────────────────

class ChurnPredictor:
    """
    Wraps the trained churn model and preprocessing artifacts.
    Designed for use by the FastAPI backend and Streamlit frontend.

    Usage
    -----
    predictor = ChurnPredictor()
    result = predictor.predict({
        "tenure": 12,
        "monthly_charges": 70,
        "contract": "Month-to-month",
        ...
    })
    # result → {"churn_prediction": "Yes", "probability": 0.87, ...}
    """

    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self.model        = None
        self.encoders     = None
        self.scaler       = None
        self.feature_names = None
        self.metadata     = {}
        self._load()

    def _load(self):
        """Load model and preprocessing artifacts from disk."""
        model_path = os.path.join(self.models_dir, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'. "
                "Please run train_model.py first."
            )

        self.model = joblib.load(model_path)
        self.encoders, self.scaler, self.feature_names = load_artifacts(self.models_dir)

        # Load metadata (if exists)
        meta_path = os.path.join(self.models_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.metadata = json.load(f)

        logger.info(
            f"ChurnPredictor loaded — model: {self.metadata.get('model_name', 'unknown')}"
        )

    def predict(self, record: dict) -> dict:
        """
        Predict churn for a single customer record.

        Parameters
        ----------
        record : dict
            Customer feature dictionary. Missing fields are filled with defaults.

        Returns
        -------
        dict with keys:
            churn_prediction : "Yes" or "No"
            probability      : float in [0, 1] (probability of churn)
            confidence       : "High" | "Medium" | "Low"
            model_used       : name of the model
        """
        # Apply defaults for optional fields not supplied by caller
        record = self._apply_defaults(record)

        try:
            X = preprocess_single_record(record, self.encoders, self.scaler)
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise ValueError(f"Could not preprocess input record: {e}")

        # Align feature count (pad with zeros if engineered features missing)
        expected_n = len(self.feature_names)
        if X.shape[1] < expected_n:
            pad = np.zeros((X.shape[0], expected_n - X.shape[1]))
            X = np.hstack([X, pad])
        elif X.shape[1] > expected_n:
            X = X[:, :expected_n]

        prob = float(self.model.predict_proba(X)[0][1])
        label = "Yes" if prob >= 0.5 else "No"
        confidence = (
            "High"   if prob >= 0.75 or prob <= 0.25 else
            "Medium" if prob >= 0.60 or prob <= 0.40 else
            "Low"
        )

        return {
            "churn_prediction": label,
            "probability":      round(prob, 4),
            "confidence":       confidence,
            "model_used":       self.metadata.get("model_name", "Unknown")
        }

    def predict_batch(self, records: list) -> list:
        """Predict churn for a list of customer record dicts."""
        return [self.predict(r) for r in records]

    @staticmethod
    def _apply_defaults(record: dict) -> dict:
        """Fill missing fields with sensible defaults to avoid KeyErrors."""
        defaults = {
            "gender":              "Male",
            "senior_citizen":      0,
            "partner":             "No",
            "dependents":          "No",
            "tenure":              0,
            "phone_service":       "Yes",
            "multiple_lines":      "No",
            "internet_service":    "Fiber optic",
            "online_security":     "No",
            "online_backup":       "No",
            "device_protection":   "No",
            "tech_support":        "No",
            "streaming_tv":        "No",
            "streaming_movies":    "No",
            "contract":            "Month-to-month",
            "paperless_billing":   "Yes",
            "payment_method":      "Electronic check",
            "monthly_charges":     70.0,
            "total_charges":       0.0
        }
        merged = {**defaults, **record}
        return merged

    @property
    def model_info(self) -> dict:
        """Return metadata about the loaded model."""
        return {
            "model_name":    self.metadata.get("model_name", "Unknown"),
            "version":       self.metadata.get("version", "v1"),
            "metrics":       self.metadata.get("metrics", {}),
            "best_params":   self.metadata.get("best_params", {}),
            "feature_count": len(self.feature_names) if self.feature_names else 0
        }


# ──────────────────────────────────────────────────────────────────────────────
# Singleton instance (used by FastAPI / Streamlit)
# ──────────────────────────────────────────────────────────────────────────────

_predictor_instance: ChurnPredictor = None


def get_predictor() -> ChurnPredictor:
    """Return the singleton ChurnPredictor, initialising it if needed."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ChurnPredictor()
    return _predictor_instance


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    predictor = get_predictor()
    print("\nModel info:", json.dumps(predictor.model_info, indent=2))

    # Example high-risk customer
    sample = {
        "tenure":           2,
        "monthly_charges":  85.0,
        "total_charges":    170.0,
        "contract":         "Month-to-month",
        "internet_service": "Fiber optic",
        "payment_method":   "Electronic check",
        "online_security":  "No",
        "tech_support":     "No",
        "paperless_billing":"Yes",
        "partner":          "No",
        "dependents":       "No"
    }

    result = predictor.predict(sample)
    print("\nPrediction result:")
    print(json.dumps(result, indent=2))
