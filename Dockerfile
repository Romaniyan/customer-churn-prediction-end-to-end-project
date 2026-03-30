# ──────────────────────────────────────────────────────────────────
# Dockerfile — Customer Churn Prediction System
# Multi-stage: trains the model, then starts both API & dashboard
# ──────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Metadata
LABEL maintainer="your-email@example.com"
LABEL description="Customer Churn Prediction System"
LABEL version="1.0.0"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /app

# Install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project source
COPY . .

# Create required directories
RUN mkdir -p models logs plots database

# Train the model at build time (bake model.pkl into the image)
RUN cd backend/model && python train_model.py

# Expose ports
EXPOSE 8000 8501

# Startup script
COPY docker-entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
