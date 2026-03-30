# 📉 Customer Churn Prediction System

> **Portfolio-grade Machine Learning project** — end-to-end churn prediction with a REST API, interactive dashboard, and SQLite persistence.  
> Built for: Final-year CS/AI students · Internship applications · Entry-level ML/DS roles · GitHub portfolios.

---

## 🗂️ Project Structure

```
customer_churn_project/
├── backend/
│   ├── main.py                  ← FastAPI REST API
│   ├── model/
│   │   ├── train_model.py       ← Model training & EDA pipeline
│   │   ├── predict.py           ← Inference engine
│   │   └── preprocessing.py    ← Data preprocessing pipeline
│   └── database/
│       └── db_manager.py       ← SQLite CRUD layer
├── frontend/
│   └── app.py                  ← Streamlit dashboard
├── data/
│   └── dataset.csv             ← Sample telecom dataset
├── models/                     ← Saved model artefacts (auto-created)
│   ├── model.pkl
│   ├── encoders.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── model_metadata.json
├── plots/                      ← EDA & evaluation charts (auto-created)
├── logs/                       ← Training logs (auto-created)
├── database/                   ← SQLite DB file (auto-created)
├── requirements.txt
├── Dockerfile
├── docker-entrypoint.sh
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | scikit-learn (LR, RF, DT, SVM) |
| Data Processing | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Database | SQLite |
| Model Persistence | Joblib |
| Containerization | Docker |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd backend/model
python train_model.py
```

**What this does:**
- Loads `data/dataset.csv`
- Runs full EDA — saves plots to `plots/`
- Trains Logistic Regression, Random Forest, Decision Tree, SVM
- Performs `GridSearchCV` hyperparameter tuning + 5-fold cross-validation
- Selects the best model (highest F1 + AUC-ROC)
- Saves `models/model.pkl`, `encoders.pkl`, `scaler.pkl`, `model_metadata.json`
- Logs everything to `logs/training.log`

### 3. Start the REST API

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API live at → **http://localhost:8000**  
Interactive docs → **http://localhost:8000/docs**

### 4. Launch the Dashboard

```bash
cd frontend
streamlit run app.py
```

Dashboard live at → **http://localhost:8501**

---

## 🐳 Docker (One-Command Deploy)

```bash
# Build image (trains model automatically)
docker build -t churn-predictor .

# Run container
docker run -p 8000:8000 -p 8501:8501 churn-predictor
```

| Service | URL |
|---|---|
| FastAPI REST API | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |

---

## 🔌 API Reference

### Health Check

```http
GET /
```
```json
{ "status": "ok", "version": "1.0.0", "timestamp": "2024-01-15T10:30:00" }
```

---

### Single Prediction

```http
POST /predict
Content-Type: application/json
```

**Request body:**
```json
{
  "tenure": 12,
  "monthly_charges": 70.0,
  "total_charges": 840.0,
  "contract": "Month-to-month",
  "internet_service": "Fiber optic",
  "payment_method": "Electronic check",
  "online_security": "No",
  "tech_support": "No",
  "paperless_billing": "Yes",
  "partner": "No",
  "dependents": "No",
  "gender": "Male",
  "senior_citizen": 0
}
```

**Response:**
```json
{
  "customer_id": "API-A1B2C3D4",
  "churn_prediction": "Yes",
  "probability": 0.8732,
  "confidence": "High",
  "model_used": "Random Forest",
  "timestamp": "2024-01-15T10:30:05"
}
```

---

### Batch Prediction

```http
POST /predict/batch
Content-Type: application/json
```

```json
{
  "customers": [
    { "tenure": 12, "monthly_charges": 70, "contract": "Month-to-month", "internet_service": "Fiber optic", "payment_method": "Electronic check" },
    { "tenure": 60, "monthly_charges": 45, "contract": "Two year",       "internet_service": "DSL",         "payment_method": "Bank transfer (automatic)" }
  ]
}
```

---

### Model Info

```http
GET /model/info
```

---

### Prediction History

```http
GET /predictions?limit=50
GET /predictions/stats
```

---

## 📊 Model Performance (Example)

After training on the sample dataset you can expect results like:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0.80 | 0.67 | 0.54 | 0.60 | 0.84 |
| **Random Forest** | **0.83** | **0.74** | **0.58** | **0.65** | **0.87** |
| Decision Tree | 0.76 | 0.58 | 0.55 | 0.56 | 0.73 |
| SVM | 0.81 | 0.70 | 0.53 | 0.60 | 0.85 |

> Results vary by random seed and dataset size. Train on the full Kaggle Telco dataset for stronger performance.

---

## 📁 Dataset Format

The CSV must contain these columns:

| Column | Type | Example Values |
|---|---|---|
| `customer_id` | string | C001 |
| `gender` | string | Male, Female |
| `senior_citizen` | int | 0, 1 |
| `partner` | string | Yes, No |
| `dependents` | string | Yes, No |
| `tenure` | int | 0 – 72 |
| `phone_service` | string | Yes, No |
| `multiple_lines` | string | Yes, No, No phone service |
| `internet_service` | string | DSL, Fiber optic, No |
| `online_security` | string | Yes, No, No internet service |
| `online_backup` | string | Yes, No, No internet service |
| `device_protection` | string | Yes, No, No internet service |
| `tech_support` | string | Yes, No, No internet service |
| `streaming_tv` | string | Yes, No, No internet service |
| `streaming_movies` | string | Yes, No, No internet service |
| `contract` | string | Month-to-month, One year, Two year |
| `paperless_billing` | string | Yes, No |
| `payment_method` | string | Electronic check, Mailed check, … |
| `monthly_charges` | float | 29.85 |
| `total_charges` | float | 1889.50 |
| `churn` | string | **Yes, No** |

💡 The full **IBM Telco Customer Churn** dataset (7,043 rows) is available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

---

## 🔬 Advanced Features

| Feature | Implementation |
|---|---|
| Hyperparameter Tuning | `GridSearchCV` (all 4 models) |
| Cross-Validation | `StratifiedKFold` (5 folds) |
| Feature Engineering | `avg_monthly_revenue`, `high_monthly_charges` |
| Feature Importance | Top-15 features (RF / DT) |
| Model Versioning | Versioned `.pkl` files + JSON metadata |
| Logging | `logging` module → `logs/training.log` |
| Error Handling | Try/except throughout; graceful API errors |
| Database | Full SQLite CRUD with WAL mode |
| Containerization | Dockerfile + entrypoint script |

---

## 🧪 cURL Examples

```bash
# Health check
curl http://localhost:8000/

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure":2,"monthly_charges":85,"contract":"Month-to-month","internet_service":"Fiber optic","payment_method":"Electronic check"}'

# Model info
curl http://localhost:8000/model/info

# Recent predictions
curl http://localhost:8000/predictions?limit=10

# Stats
curl http://localhost:8000/predictions/stats
```

---

## 📜 License

MIT License — free to use for personal projects, portfolios, and learning.

---

## 🙋 Contributing

1. Fork the repo  
2. Create a feature branch (`git checkout -b feature/awesome-addition`)  
3. Commit your changes  
4. Open a Pull Request

---

*Built as a portfolio demonstration of end-to-end ML engineering.*
