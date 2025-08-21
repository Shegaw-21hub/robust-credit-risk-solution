# ============== Build stage (optional if you need compilation) ==============
FROM python:3.10-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (add if needed for numpy/pandas speed or MLflow backends)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port for uvicorn
EXPOSE 8000

# Environment (override via docker-compose or runtime)
ENV MODEL_LOCAL_PATH="models/rf_model.pkl" \
    FEATURE_NAMES_PATH="models/feature_names.json" \
    APP_TITLE="Credit Risk API" \
    APP_VERSION="1.0.0"

# Run the FastAPI app
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
