"""
app.py
------
Streamlit Dashboard for the Customer Churn Prediction System.

Features
--------
- Manual single-customer prediction form
- Batch prediction via CSV upload
- Model performance metrics & charts
- Prediction history viewer
- EDA visualizations
"""

import os
import sys
import json
import logging
import requests
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Path setup — allow importing from backend directly
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
MODELS_DIR  = os.path.join(PROJECT_DIR, "models")
PLOTS_DIR   = os.path.join(PROJECT_DIR, "plots")

sys.path.insert(0, os.path.join(BACKEND_DIR, "model"))
sys.path.insert(0, os.path.join(BACKEND_DIR, "database"))

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem;
        color: white; text-align: center;
    }
    .main-header h1 { font-size: 2.2rem; margin: 0; }
    .main-header p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }

    /* Metric cards */
    .metric-card {
        background: white; border-radius: 10px;
        padding: 1.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #2a5298; margin-bottom: 1rem;
    }
    .metric-card h3 { margin: 0; color: #2a5298; font-size: 0.9rem; }
    .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1e3c72; }

    /* Prediction result */
    .pred-yes { background:#fff5f5; border:2px solid #fc8181; border-radius:10px;
                padding:1.5rem; text-align:center; }
    .pred-no  { background:#f0fff4; border:2px solid #68d391; border-radius:10px;
                padding:1.5rem; text-align:center; }
    .pred-yes h2 { color:#c53030; }
    .pred-no  h2 { color:#276749; }

    /* Sidebar */
    .css-1d391kg { background-color: #1e3c72; }
    [data-testid="stSidebar"] { background-color: #f8faff; }

    /* Section headers */
    .section-header { border-bottom: 2px solid #2a5298; padding-bottom: 0.4rem;
                      margin-bottom: 1rem; color: #1e3c72; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: load predictor (direct, no API call needed)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model …")
def load_predictor():
    """Cache the predictor so it's loaded only once per session."""
    try:
        from predict import get_predictor
        return get_predictor(), None
    except FileNotFoundError as e:
        return None, str(e)
    except Exception as e:
        return None, str(e)


@st.cache_resource
def load_db():
    try:
        import db_manager as db
        return db, None
    except Exception as e:
        return None, str(e)


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📉 Churn Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Single Prediction", "📂 Batch Prediction",
         "📊 Model Performance", "🗄️ Prediction History", "📈 EDA Charts"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("**Built with**")
    st.markdown("🐍 Python · Scikit-learn  \n🚀 FastAPI · Streamlit  \n🗃️ SQLite")
    st.markdown("---")
    st.caption("Portfolio project — CS / ML")


# ──────────────────────────────────────────────────────────────────────────────
# Banner
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📉 Customer Churn Prediction System</h1>
    <p>Predict which customers are at risk of leaving — powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Load resources
# ──────────────────────────────────────────────────────────────────────────────

predictor, pred_err = load_predictor()
db, db_err          = load_db()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Home
# ──────────────────────────────────────────────────────────────────────────────

if page == "🏠 Home":
    if pred_err:
        st.error(f"⚠️ Model not loaded: {pred_err}")
        st.info("Run `python backend/model/train_model.py` to train the model first.")
    else:
        info = predictor.model_info
        metrics = info.get("metrics", {})

        st.markdown("### 🤖 Model Status")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Best Model",   info.get("model_name", "—"))
        with c2:
            st.metric("Accuracy",     f"{metrics.get('accuracy', 0):.2%}")
        with c3:
            st.metric("F1 Score",     f"{metrics.get('f1_score', 0):.2%}")
        with c4:
            st.metric("AUC-ROC",      f"{metrics.get('auc_roc', 0):.4f}")

        st.markdown("---")
        st.markdown("### 🚀 Quick Start")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.info("**🔍 Single Prediction**\nEnter customer details manually to get instant churn probability.")
        with col_b:
            st.success("**📂 Batch Prediction**\nUpload a CSV to score thousands of customers at once.")
        with col_c:
            st.warning("**📊 Model Performance**\nExplore confusion matrices, ROC curves, and feature importance.")

        if db:
            try:
                stats = db.get_prediction_stats()
                st.markdown("---")
                st.markdown("### 📊 Database Statistics")
                d1, d2, d3, d4 = st.columns(4)
                d1.metric("Total Predictions", stats["total_predictions"])
                d2.metric("Churn Count",        stats["churn_count"])
                d3.metric("No-Churn Count",     stats["no_churn_count"])
                d4.metric("Overall Churn Rate", f"{stats['churn_rate']:.1%}")
            except Exception:
                pass


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Single Prediction
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🔍 Single Prediction":
    st.markdown("### 🔍 Single Customer Prediction")

    if pred_err:
        st.error(f"Model not available: {pred_err}")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("#### Customer Details")

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            customer_id    = st.text_input("Customer ID (optional)", placeholder="e.g. CUST-001")
            gender         = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with r1c2:
            partner    = st.selectbox("Partner",    ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure     = st.slider("Tenure (months)", 0, 72, 12)
        with r1c3:
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0, 0.5)
            total_charges   = st.number_input("Total Charges ($)",   0.0, 10000.0,
                                               round(tenure * monthly_charges, 2))

        st.markdown("#### Services")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            phone_service  = st.selectbox("Phone Service",  ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        with s2:
            internet_service  = st.selectbox("Internet Service",  ["Fiber optic", "DSL", "No"])
            online_security   = st.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        with s3:
            online_backup     = st.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        with s4:
            tech_support     = st.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
            streaming_tv     = st.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("#### Contract & Billing")
        b1, b2, b3 = st.columns(3)
        with b1:
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        with b2:
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        with b3:
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])

        submitted = st.form_submit_button("🚀 Predict Churn", use_container_width=True)

    if submitted:
        record = dict(
            customer_id=customer_id or None, gender=gender,
            senior_citizen=senior_citizen, partner=partner,
            dependents=dependents, tenure=float(tenure),
            phone_service=phone_service, multiple_lines=multiple_lines,
            internet_service=internet_service, online_security=online_security,
            online_backup=online_backup, device_protection=device_protection,
            tech_support=tech_support, streaming_tv=streaming_tv,
            streaming_movies=streaming_movies, contract=contract,
            paperless_billing=paperless_billing, payment_method=payment_method,
            monthly_charges=float(monthly_charges), total_charges=float(total_charges)
        )
        with st.spinner("Running prediction …"):
            try:
                result = predictor.predict(record)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        st.markdown("---")
        st.markdown("### 🎯 Prediction Result")
        prob = result["probability"]

        col_res, col_gauge = st.columns([1, 1])
        with col_res:
            css_class = "pred-yes" if result["churn_prediction"] == "Yes" else "pred-no"
            icon      = "⚠️" if result["churn_prediction"] == "Yes" else "✅"
            color     = "#c53030" if result["churn_prediction"] == "Yes" else "#276749"
            st.markdown(f"""
            <div class="{css_class}">
                <h2>{icon} Churn: {result['churn_prediction']}</h2>
                <p style="font-size:1.2rem; font-weight:600; color:{color}">
                    Probability: {prob:.1%}
                </p>
                <p>Confidence: <strong>{result['confidence']}</strong></p>
                <p style="font-size:0.85rem; color:#666">Model: {result['model_used']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col_gauge:
            fig, ax = plt.subplots(figsize=(4, 4))
            colors = ["#68d391" if prob < 0.5 else "#fc8181", "#e9ecef"]
            ax.pie([prob, 1 - prob], colors=colors, startangle=90,
                   wedgeprops={"width": 0.45})
            ax.text(0, 0, f"{prob:.0%}", ha="center", va="center",
                    fontsize=22, fontweight="bold",
                    color="#c53030" if prob >= 0.5 else "#276749")
            ax.set_title("Churn Probability", fontsize=11)
            st.pyplot(fig)
            plt.close()

        # Save to DB
        if db:
            try:
                cid = customer_id or f"DASH-{datetime.utcnow().strftime('%H%M%S')}"
                db.insert_prediction(
                    customer_id=cid,
                    churn_prediction=result["churn_prediction"],
                    probability=result["probability"],
                    confidence=result["confidence"],
                    model_used=result["model_used"],
                    input_features=record
                )
                st.success("✅ Prediction saved to database.")
            except Exception as e:
                st.warning(f"DB save failed: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Batch Prediction
# ──────────────────────────────────────────────────────────────────────────────

elif page == "📂 Batch Prediction":
    st.markdown("### 📂 Batch Prediction via CSV Upload")

    if pred_err:
        st.error(f"Model not available: {pred_err}")
        st.stop()

    st.info(
        "Upload a CSV file with customer data. "
        "Required columns: `tenure`, `monthly_charges`, `contract`, "
        "`internet_service`, `payment_method`"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        st.markdown(f"**{len(df)} customers loaded.**")
        st.dataframe(df.head(10), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", use_container_width=True):
            progress_bar = st.progress(0)
            results_list = []
            total = len(df)

            for i, (_, row) in enumerate(df.iterrows()):
                record = row.to_dict()
                try:
                    res = predictor.predict(record)
                except Exception:
                    res = {"churn_prediction": "Error", "probability": 0.0,
                           "confidence": "N/A", "model_used": "N/A"}

                results_list.append({
                    "customer_id":      record.get("customer_id", f"ROW-{i+1}"),
                    "churn_prediction": res["churn_prediction"],
                    "probability":      res["probability"],
                    "confidence":       res["confidence"]
                })

                if db and res["churn_prediction"] != "Error":
                    try:
                        db.insert_prediction(
                            customer_id=str(record.get("customer_id", f"ROW-{i+1}")),
                            churn_prediction=res["churn_prediction"],
                            probability=res["probability"],
                            confidence=res["confidence"],
                            model_used=res["model_used"],
                            input_features=record
                        )
                    except Exception:
                        pass

                progress_bar.progress((i + 1) / total)

            results_df = pd.DataFrame(results_list)
            st.markdown("---")
            st.markdown("### 📊 Batch Results")

            b1, b2, b3 = st.columns(3)
            churned = (results_df["churn_prediction"] == "Yes").sum()
            b1.metric("Total Customers", total)
            b2.metric("Predicted Churn", int(churned))
            b3.metric("Churn Rate",      f"{churned/total:.1%}")

            st.dataframe(
                results_df.style.applymap(
                    lambda v: "color: red; font-weight: bold" if v == "Yes" else
                              "color: green; font-weight: bold" if v == "No" else "",
                    subset=["churn_prediction"]
                ),
                use_container_width=True
            )

            csv_out = results_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results CSV", csv_out,
                               "churn_predictions.csv", "text/csv")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Model Performance
# ──────────────────────────────────────────────────────────────────────────────

elif page == "📊 Model Performance":
    st.markdown("### 📊 Model Performance Metrics")

    if pred_err:
        st.error(f"Model not available: {pred_err}")
        st.stop()

    info    = predictor.model_info
    metrics = info.get("metrics", {})

    # Key metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{metrics.get('accuracy',  0):.4f}")
    c2.metric("Precision", f"{metrics.get('precision', 0):.4f}")
    c3.metric("Recall",    f"{metrics.get('recall',    0):.4f}")
    c4.metric("F1 Score",  f"{metrics.get('f1_score',  0):.4f}")
    c5.metric("AUC-ROC",   f"{metrics.get('auc_roc',   0):.4f}")

    st.markdown("---")

    # Show saved plots if available
    def show_plot(filename: str, caption: str):
        path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
        else:
            st.info(f"Plot not found: {filename}. Train the model to generate it.")

    tab1, tab2, tab3, tab4 = st.tabs(["Confusion Matrix", "ROC Curves",
                                       "Feature Importance", "Model Comparison"])

    with tab1:
        model_name = info.get("model_name", "").lower().replace(" ", "_")
        show_plot(f"cm_{model_name}.png", "Confusion Matrix — Best Model")

    with tab2:
        show_plot("roc_curves.png", "ROC Curves — All Models")

    with tab3:
        show_plot(f"feature_importance_{model_name}.png",
                  "Feature Importance — Best Model")

    with tab4:
        show_plot("model_comparison.png", "Model Comparison — All Metrics")

    st.markdown("---")
    st.markdown("#### Best Model Details")
    st.json({
        "model":       info.get("model_name"),
        "version":     info.get("version"),
        "best_params": info.get("best_params"),
        "metrics":     metrics
    })


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Prediction History
# ──────────────────────────────────────────────────────────────────────────────

elif page == "🗄️ Prediction History":
    st.markdown("### 🗄️ Prediction History")

    if db_err or db is None:
        st.error(f"Database unavailable: {db_err}")
        st.stop()

    try:
        stats = db.get_prediction_stats()
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Total",         stats["total_predictions"])
        h2.metric("Churn",         stats["churn_count"])
        h3.metric("No Churn",      stats["no_churn_count"])
        h4.metric("Churn Rate",    f"{stats['churn_rate']:.1%}")

        records = db.get_all_predictions(limit=200)
        if records:
            hist_df = pd.DataFrame(records)
            st.dataframe(
                hist_df.style.applymap(
                    lambda v: "color: red; font-weight: bold" if v == "Yes" else
                              "color: green; font-weight: bold" if v == "No" else "",
                    subset=["churn_prediction"]
                ),
                use_container_width=True
            )
        else:
            st.info("No predictions stored yet. Make some predictions first!")
    except Exception as e:
        st.error(f"Could not load history: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: EDA Charts
# ──────────────────────────────────────────────────────────────────────────────

elif page == "📈 EDA Charts":
    st.markdown("### 📈 Exploratory Data Analysis")

    def show_eda(filename: str, caption: str):
        path = os.path.join(PLOTS_DIR, filename)
        if os.path.exists(path):
            st.image(path, caption=caption, use_column_width=True)
        else:
            st.info(f"Run training to generate: {filename}")

    col_left, col_right = st.columns(2)
    with col_left:
        show_eda("churn_distribution.png",      "Churn Distribution")
        show_eda("tenure_distribution.png",      "Tenure Distribution by Churn")
        show_eda("correlation_heatmap.png",      "Correlation Heatmap")
    with col_right:
        show_eda("churn_by_contract.png",        "Churn by Contract Type")
        show_eda("monthly_charges_churn.png",    "Monthly Charges by Churn")
