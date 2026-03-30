#!/bin/bash
# docker-entrypoint.sh
# Start both FastAPI and Streamlit in background processes

set -e

echo "========================================"
echo "  Customer Churn Prediction System"
echo "========================================"

# Start FastAPI backend
echo "[1/2] Starting FastAPI backend on port 8000..."
cd /app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 &
FASTAPI_PID=$!

# Wait briefly for API to initialise
sleep 3

# Start Streamlit frontend
echo "[2/2] Starting Streamlit dashboard on port 8501..."
cd /app/frontend
streamlit run app.py \
  --server.port=8501 \
  --server.address=0.0.0.0 \
  --server.headless=true \
  --browser.gatherUsageStats=false &
STREAMLIT_PID=$!

echo ""
echo "✅  FastAPI  → http://localhost:8000"
echo "✅  API Docs → http://localhost:8000/docs"
echo "✅  Dashboard→ http://localhost:8501"
echo ""

# Keep container running; stop if either process dies
wait $FASTAPI_PID $STREAMLIT_PID
