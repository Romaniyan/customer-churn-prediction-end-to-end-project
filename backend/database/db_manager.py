"""
db_manager.py
-------------
SQLite database integration for the Customer Churn Prediction System.
Stores customer data, prediction results, and timestamps.
"""

import os
import sqlite3
import logging
import json
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Database path
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../.."))
DB_PATH = os.path.join(PROJECT_DIR, "database", "churn.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Connection helper
# ──────────────────────────────────────────────────────────────────────────────

@contextmanager
def get_connection():
    """Context manager that provides a thread-safe SQLite connection."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row          # enables dict-like row access
    conn.execute("PRAGMA journal_mode=WAL") # better concurrency
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"DB transaction error: {e}")
        raise
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────────────────
# Schema initialisation
# ──────────────────────────────────────────────────────────────────────────────

def init_db():
    """Create database tables if they don't already exist."""
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS customers (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id         TEXT,
                gender              TEXT,
                senior_citizen      INTEGER,
                partner             TEXT,
                dependents          TEXT,
                tenure              REAL,
                phone_service       TEXT,
                multiple_lines      TEXT,
                internet_service    TEXT,
                online_security     TEXT,
                online_backup       TEXT,
                device_protection   TEXT,
                tech_support        TEXT,
                streaming_tv        TEXT,
                streaming_movies    TEXT,
                contract            TEXT,
                paperless_billing   TEXT,
                payment_method      TEXT,
                monthly_charges     REAL,
                total_charges       REAL,
                created_at          TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id         TEXT,
                churn_prediction    TEXT NOT NULL,
                probability         REAL NOT NULL,
                confidence          TEXT,
                model_used          TEXT,
                input_features      TEXT,    -- JSON blob of raw input
                created_at          TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_customer
                ON predictions (customer_id);
            CREATE INDEX IF NOT EXISTS idx_predictions_created
                ON predictions (created_at);
        """)
    logger.info(f"Database initialised at: {DB_PATH}")


# ──────────────────────────────────────────────────────────────────────────────
# CRUD operations
# ──────────────────────────────────────────────────────────────────────────────

def insert_customer(record: dict) -> int:
    """
    Insert a customer record and return the new row ID.
    Accepts any subset of customer fields; missing ones default to NULL.
    """
    fields = [
        "customer_id", "gender", "senior_citizen", "partner", "dependents",
        "tenure", "phone_service", "multiple_lines", "internet_service",
        "online_security", "online_backup", "device_protection", "tech_support",
        "streaming_tv", "streaming_movies", "contract", "paperless_billing",
        "payment_method", "monthly_charges", "total_charges"
    ]
    columns  = [f for f in fields if f in record]
    values   = [record[c] for c in columns]
    placeholders = ", ".join(["?"] * len(columns))
    sql = f"INSERT INTO customers ({', '.join(columns)}) VALUES ({placeholders})"

    with get_connection() as conn:
        cursor = conn.execute(sql, values)
        row_id = cursor.lastrowid

    logger.debug(f"Customer inserted — row_id={row_id}")
    return row_id


def insert_prediction(
    customer_id: str,
    churn_prediction: str,
    probability: float,
    confidence: str,
    model_used: str,
    input_features: dict
) -> int:
    """Insert a prediction result and return the new row ID."""
    sql = """
        INSERT INTO predictions
            (customer_id, churn_prediction, probability, confidence, model_used, input_features)
        VALUES (?, ?, ?, ?, ?, ?)
    """
    with get_connection() as conn:
        cursor = conn.execute(sql, (
            customer_id,
            churn_prediction,
            probability,
            confidence,
            model_used,
            json.dumps(input_features)
        ))
        row_id = cursor.lastrowid

    logger.debug(f"Prediction stored — row_id={row_id}, customer={customer_id}, result={churn_prediction}")
    return row_id


def get_all_predictions(limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve the most recent predictions."""
    sql = """
        SELECT id, customer_id, churn_prediction, probability, confidence, model_used, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT ?
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_prediction_stats() -> Dict[str, Any]:
    """Return aggregate statistics for the predictions table."""
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        churned = conn.execute(
            "SELECT COUNT(*) FROM predictions WHERE churn_prediction='Yes'"
        ).fetchone()[0]
        avg_prob = conn.execute("SELECT AVG(probability) FROM predictions").fetchone()[0]
        recent = conn.execute("""
            SELECT created_at FROM predictions ORDER BY created_at DESC LIMIT 1
        """).fetchone()

    return {
        "total_predictions": total,
        "churn_count":       churned,
        "no_churn_count":    total - churned,
        "churn_rate":        round(churned / total, 4) if total else 0.0,
        "avg_probability":   round(avg_prob, 4) if avg_prob else 0.0,
        "last_prediction":   dict(recent)["created_at"] if recent else None
    }


def get_predictions_by_customer(customer_id: str) -> List[Dict[str, Any]]:
    """Retrieve all predictions for a specific customer ID."""
    sql = """
        SELECT * FROM predictions
        WHERE customer_id = ?
        ORDER BY created_at DESC
    """
    with get_connection() as conn:
        rows = conn.execute(sql, (customer_id,)).fetchall()
    return [dict(r) for r in rows]


def clear_predictions():
    """Delete all prediction records (useful for testing)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM predictions")
    logger.warning("All prediction records deleted.")


# ──────────────────────────────────────────────────────────────────────────────
# Auto-initialise on import
# ──────────────────────────────────────────────────────────────────────────────

init_db()
