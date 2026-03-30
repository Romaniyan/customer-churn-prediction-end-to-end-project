"""
main.py
-------
FastAPI REST API for the Customer Churn Prediction System.

Endpoints
---------
GET  /              → health check
GET  /model/info    → model metadata & metrics
POST /predict       → single customer churn prediction
POST /predict/batch → batch churn predictions
GET  /predictions   → retrieve stored predictions
GET  /predictions/stats → aggregate prediction statistics
"""

import sys
import os
import logging
import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

# Allow imports from sibling directories
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "model"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "database"))

from predict import get_predictor
from db_manager import insert_prediction, get_all_predictions, get_prediction_stats

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-ready REST API for predicting customer churn. "
        "Uses a machine-learning model trained on historical telecom data."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS — allow all origins in development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ──────────────────────────────────────────────────────────────────────────────

class CustomerInput(BaseModel):
    """Input schema for a single customer churn prediction request."""

    # Identifiers (optional — auto-generated if omitted)
    customer_id: Optional[str] = Field(None, description="Unique customer identifier")

    # Demographics
    gender:         Optional[str] = Field("Male",  description="Male | Female")
    senior_citizen: Optional[int] = Field(0,       description="1 if senior citizen, else 0")
    partner:        Optional[str] = Field("No",    description="Yes | No")
    dependents:     Optional[str] = Field("No",    description="Yes | No")

    # Service usage
    tenure:           float = Field(..., ge=0,  description="Months with the company")
    phone_service:    Optional[str] = Field("Yes", description="Yes | No")
    multiple_lines:   Optional[str] = Field("No",  description="Yes | No | No phone service")
    internet_service: str  = Field(...,              description="DSL | Fiber optic | No")
    online_security:  Optional[str] = Field("No",  description="Yes | No | No internet service")
    online_backup:    Optional[str] = Field("No",  description="Yes | No | No internet service")
    device_protection:Optional[str] = Field("No",  description="Yes | No | No internet service")
    tech_support:     Optional[str] = Field("No",  description="Yes | No | No internet service")
    streaming_tv:     Optional[str] = Field("No",  description="Yes | No | No internet service")
    streaming_movies: Optional[str] = Field("No",  description="Yes | No | No internet service")

    # Contract & billing
    contract:          str   = Field(..., description="Month-to-month | One year | Two year")
    paperless_billing: Optional[str] = Field("Yes", description="Yes | No")
    payment_method:    str   = Field(..., description="Electronic check | Mailed check | Bank transfer (automatic) | Credit card (automatic)")

    # Charges
    monthly_charges: float = Field(..., ge=0, description="Monthly bill in USD")
    total_charges:   Optional[float] = Field(None, ge=0, description="Total billed in USD (auto-computed if omitted)")

    @validator("total_charges", pre=True, always=True)
    def default_total_charges(cls, v, values):
        if v is None:
            tenure = values.get("tenure", 0)
            monthly = values.get("monthly_charges", 0)
            return round(tenure * monthly, 2)
        return v

    class Config:
        schema_extra = {
            "example": {
                "tenure": 12,
                "monthly_charges": 70.5,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for a churn prediction."""
    customer_id:      str
    churn_prediction: str   # "Yes" | "No"
    probability:      float
    confidence:       str   # "High" | "Medium" | "Low"
    model_used:       str
    timestamp:        str


class BatchInput(BaseModel):
    customers: List[CustomerInput]


class BatchResponse(BaseModel):
    total:   int
    results: List[PredictionResponse]


class HealthResponse(BaseModel):
    status:    str
    version:   str
    timestamp: str


# ──────────────────────────────────────────────────────────────────────────────
# Startup — load model once
# ──────────────────────────────────────────────────────────────────────────────

predictor = None

@app.on_event("startup")
def startup_event():
    global predictor
    logger.info("Starting up — loading model …")
    try:
        predictor = get_predictor()
        logger.info(f"Model loaded: {predictor.model_info['model_name']}")
    except FileNotFoundError as e:
        logger.error(str(e))
        # API will start but /predict will return 503


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Health check endpoint — confirms the API is running."""
    return {
        "status":    "ok",
        "version":   "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/model/info", tags=["Model"])
def model_info():
    """Return metadata and performance metrics of the loaded model."""
    if predictor is None:
        raise HTTPException(503, detail="Model not loaded. Run train_model.py first.")
    return predictor.model_info


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_churn(customer: CustomerInput):
    """
    Predict churn probability for a single customer.

    - **tenure**: months the customer has been with the company  
    - **monthly_charges**: current monthly bill  
    - **contract**: contract type (Month-to-month | One year | Two year)  
    - **internet_service**: internet plan type  
    - **payment_method**: payment method used  
    """
    if predictor is None:
        raise HTTPException(503, detail="Model not available. Run train_model.py first.")

    # Auto-assign customer_id if not provided
    cid = customer.customer_id or f"API-{uuid.uuid4().hex[:8].upper()}"
    record = customer.dict()
    record["customer_id"] = cid

    try:
        result = predictor.predict(record)
    except ValueError as e:
        raise HTTPException(422, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected prediction error")
        raise HTTPException(500, detail="Internal prediction error.")

    timestamp = datetime.utcnow().isoformat()

    # Persist to database
    try:
        insert_prediction(
            customer_id      = cid,
            churn_prediction = result["churn_prediction"],
            probability      = result["probability"],
            confidence       = result["confidence"],
            model_used       = result["model_used"],
            input_features   = record
        )
    except Exception as e:
        logger.warning(f"DB write failed (prediction still returned): {e}")

    return PredictionResponse(
        customer_id      = cid,
        churn_prediction = result["churn_prediction"],
        probability      = result["probability"],
        confidence       = result["confidence"],
        model_used       = result["model_used"],
        timestamp        = timestamp
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(batch: BatchInput):
    """Predict churn for a batch of customers."""
    if predictor is None:
        raise HTTPException(503, detail="Model not available.")

    results = []
    for customer in batch.customers:
        cid = customer.customer_id or f"API-{uuid.uuid4().hex[:8].upper()}"
        record = customer.dict()
        record["customer_id"] = cid
        try:
            res = predictor.predict(record)
        except Exception as e:
            logger.warning(f"Batch prediction failed for {cid}: {e}")
            res = {"churn_prediction": "Error", "probability": 0.0,
                   "confidence": "N/A", "model_used": "N/A"}

        try:
            insert_prediction(
                customer_id      = cid,
                churn_prediction = res["churn_prediction"],
                probability      = res["probability"],
                confidence       = res.get("confidence", "N/A"),
                model_used       = res.get("model_used", "N/A"),
                input_features   = record
            )
        except Exception:
            pass

        results.append(PredictionResponse(
            customer_id      = cid,
            churn_prediction = res["churn_prediction"],
            probability      = res["probability"],
            confidence       = res.get("confidence", "N/A"),
            model_used       = res.get("model_used", "N/A"),
            timestamp        = datetime.utcnow().isoformat()
        ))

    return BatchResponse(total=len(results), results=results)


@app.get("/predictions", tags=["History"])
def get_predictions(limit: int = Query(50, ge=1, le=500)):
    """Retrieve the most recent stored predictions."""
    return get_all_predictions(limit=limit)


@app.get("/predictions/stats", tags=["History"])
def prediction_stats():
    """Return aggregate statistics about all stored predictions."""
    return get_prediction_stats()


# ──────────────────────────────────────────────────────────────────────────────
# Run directly (development only — use uvicorn in production)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
