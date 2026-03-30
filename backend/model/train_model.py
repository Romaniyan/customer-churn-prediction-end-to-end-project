"""
train_model.py
--------------
Model training, evaluation, hyperparameter tuning, and selection pipeline
for the Customer Churn Prediction System.
"""

import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for servers
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)

from preprocessing import (
    load_data, full_preprocessing_pipeline, save_artifacts
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "../../logs/training.log"), mode="a")
    ]
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../.."))
DATA_PATH   = os.path.join(PROJECT_DIR, "data", "dataset.csv")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR   = os.path.join(PROJECT_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "logs"), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Model Registry with hyperparameter grids
# ──────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=1000, random_state=42),
        "param_grid": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"]
        }
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=42),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [2, 5, 10],
            "criterion": ["gini", "entropy"]
        }
    },
    "Support Vector Machine": {
        "estimator": SVC(probability=True, random_state=42),
        "param_grid": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
            "gamma": ["scale", "auto"]
        }
    }
}


# ──────────────────────────────────────────────────────────────────────────────
# EDA Plots
# ──────────────────────────────────────────────────────────────────────────────

def run_eda(df_raw: pd.DataFrame):
    """Generate and save EDA visualizations."""
    logger.info("Running EDA …")
    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Churn distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    churn_counts = df_raw["churn"].value_counts()
    axes[0].pie(churn_counts, labels=churn_counts.index, autopct="%1.1f%%",
                colors=["#4CAF50", "#F44336"], startangle=90)
    axes[0].set_title("Churn Distribution")

    sns.countplot(data=df_raw, x="churn", palette={"No": "#4CAF50", "Yes": "#F44336"}, ax=axes[1])
    axes[1].set_title("Customer Count by Churn")
    axes[1].set_xlabel("Churn")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "churn_distribution.png"), dpi=150)
    plt.close()

    # 2. Churn by contract type
    fig, ax = plt.subplots(figsize=(9, 5))
    ct = df_raw.groupby(["contract", "churn"]).size().unstack(fill_value=0)
    ct.plot(kind="bar", color=["#4CAF50", "#F44336"], ax=ax, edgecolor="white")
    ax.set_title("Churn by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Count")
    ax.legend(title="Churn")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "churn_by_contract.png"), dpi=150)
    plt.close()

    # 3. Tenure distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(data=df_raw, x="tenure", hue="churn", multiple="stack",
                 palette={"No": "#4CAF50", "Yes": "#F44336"}, bins=24, ax=ax)
    ax.set_title("Tenure Distribution by Churn")
    ax.set_xlabel("Tenure (months)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "tenure_distribution.png"), dpi=150)
    plt.close()

    # 4. Monthly charges distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df_raw, x="churn", y="monthly_charges",
                palette={"No": "#4CAF50", "Yes": "#F44336"}, ax=ax)
    ax.set_title("Monthly Charges by Churn")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "monthly_charges_churn.png"), dpi=150)
    plt.close()

    # 5. Correlation heatmap (numeric only)
    numeric_df = df_raw.select_dtypes(include=[np.number]).copy()
    if "senior_citizen" in numeric_df.columns:
        numeric_df.drop(columns=["senior_citizen"], inplace=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()

    logger.info(f"EDA plots saved to '{PLOTS_DIR}'")


# ──────────────────────────────────────────────────────────────────────────────
# Training & Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """Compute evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1_score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc":   round(roc_auc_score(y_test, y_prob), 4) if y_prob is not None else None,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    logger.info(
        f"[{model_name}] Acc={metrics['accuracy']:.4f}  "
        f"Prec={metrics['precision']:.4f}  "
        f"Rec={metrics['recall']:.4f}  "
        f"F1={metrics['f1_score']:.4f}  "
        f"AUC={metrics['auc_roc']}"
    )
    return metrics


def plot_confusion_matrix(cm: list, model_name: str):
    """Save confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(np.array(cm), annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"], ax=ax)
    ax.set_title(f"Confusion Matrix — {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(PLOTS_DIR, f"cm_{safe_name}.png"), dpi=150)
    plt.close()


def plot_roc_curves(roc_data: dict):
    """Overlay ROC curves for all models in a single figure."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for (name, (fpr, tpr, auc_score)), color in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_score:.3f})", color=color, lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=150)
    plt.close()
    logger.info("ROC curves saved.")


def plot_feature_importance(model, feature_names: list, model_name: str):
    """Plot and save feature importance (Random Forest / Decision Tree only)."""
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]          # top-15

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(
        [feature_names[i] for i in reversed(indices)],
        importances[list(reversed(indices))],
        color="#2196F3"
    )
    ax.set_title(f"Feature Importance — {model_name}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    plt.savefig(os.path.join(PLOTS_DIR, f"feature_importance_{safe_name}.png"), dpi=150)
    plt.close()
    logger.info(f"Feature importance plot saved for {model_name}.")


def train_all_models(X_train, X_test, y_train, y_test, feature_names: list) -> dict:
    """
    Train each model with GridSearchCV, evaluate, and collect results.
    Returns a dict: {model_name: {"model": fitted_estimator, "metrics": {...}}}
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    roc_data = {}

    for name, cfg in MODEL_REGISTRY.items():
        logger.info(f"── Training: {name} ──")

        grid = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        logger.info(f"  Best params: {grid.best_params_}")

        # Cross-validation score
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="f1")
        logger.info(f"  CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        metrics = evaluate_model(best_model, X_test, y_test, name)
        metrics["cv_f1_mean"] = round(cv_scores.mean(), 4)
        metrics["cv_f1_std"]  = round(cv_scores.std(),  4)
        metrics["best_params"] = grid.best_params_

        results[name] = {"model": best_model, "metrics": metrics}

        # Confusion matrix plot
        plot_confusion_matrix(metrics["confusion_matrix"], name)

        # Feature importance (tree-based models)
        plot_feature_importance(best_model, feature_names, name)

        # ROC data
        if hasattr(best_model, "predict_proba"):
            y_prob = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_data[name] = (fpr, tpr, metrics["auc_roc"])

    plot_roc_curves(roc_data)
    return results


def plot_model_comparison(results: dict):
    """Bar chart comparing all models across key metrics."""
    metrics_to_compare = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    comparison = {
        name: {m: r["metrics"][m] for m in metrics_to_compare}
        for name, r in results.items()
    }
    df_cmp = pd.DataFrame(comparison).T

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df_cmp))
    width = 0.15
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    for i, (metric, color) in enumerate(zip(metrics_to_compare, colors)):
        ax.bar(x + i * width, df_cmp[metric], width, label=metric.upper(), color=color, alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(df_cmp.index, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Metrics")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=150)
    plt.close()
    logger.info("Model comparison chart saved.")


def select_best_model(results: dict) -> tuple:
    """
    Select the best model based on F1 score (primary) + AUC-ROC (tiebreaker).
    Returns (best_model_name, best_model_object, best_metrics)
    """
    best_name = max(
        results,
        key=lambda n: (results[n]["metrics"]["f1_score"], results[n]["metrics"]["auc_roc"] or 0)
    )
    best = results[best_name]
    logger.info(f"Best model selected: {best_name}  (F1={best['metrics']['f1_score']:.4f})")
    return best_name, best["model"], best["metrics"]


def save_model(model, model_name: str, metrics: dict, version: str = "v1"):
    """
    Persist the best model and its metadata.
    Saves:
      models/model.pkl          — primary model file (for API)
      models/model_{version}.pkl — versioned backup
      models/model_metadata.json — evaluation snapshot
    """
    # Primary (always overwrite latest)
    primary_path = os.path.join(MODELS_DIR, "model.pkl")
    joblib.dump(model, primary_path)

    # Versioned backup
    versioned_path = os.path.join(MODELS_DIR, f"model_{version}.pkl")
    joblib.dump(model, versioned_path)

    # Metadata
    meta = {
        "model_name": model_name,
        "version": version,
        "metrics": {k: v for k, v in metrics.items()
                    if k not in ("classification_report", "confusion_matrix")},
        "best_params": metrics.get("best_params", {})
    }
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Model saved → {primary_path}")
    logger.info(f"Versioned  → {versioned_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  Customer Churn — Model Training Pipeline ║")
    logger.info("╚══════════════════════════════════════════╝")

    # 1. Load raw data
    df_raw = load_data(DATA_PATH)

    # 2. EDA
    run_eda(df_raw)

    # 3. Preprocessing
    X_train, X_test, y_train, y_test, encoders, scaler, feature_names = \
        full_preprocessing_pipeline(df_raw)

    # 4. Train all models
    results = train_all_models(X_train, X_test, y_train, y_test, feature_names)

    # 5. Compare
    plot_model_comparison(results)

    # 6. Select best
    best_name, best_model, best_metrics = select_best_model(results)

    # 7. Save model + preprocessing artifacts
    save_model(best_model, best_name, best_metrics)
    save_artifacts(encoders, scaler, feature_names, MODELS_DIR)

    # 8. Print summary
    logger.info("\n" + "=" * 50)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 50)
    for name, data in results.items():
        m = data["metrics"]
        logger.info(
            f"{name:<28} Acc={m['accuracy']:.4f}  F1={m['f1_score']:.4f}  AUC={m['auc_roc']}"
        )
    logger.info(f"\n★  Best Model : {best_name}")
    logger.info(f"   F1 Score   : {best_metrics['f1_score']:.4f}")
    logger.info(f"   AUC-ROC    : {best_metrics['auc_roc']:.4f}")
    logger.info("=" * 50)

    return results, best_name, best_model


if __name__ == "__main__":
    main()
