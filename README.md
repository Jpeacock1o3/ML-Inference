This assignment was for my Data Intensive Computing Graduate Course

# ML Inference API

A production-grade REST API for serving real-time machine learning predictions, built with FastAPI and deployed via Docker. Designed for high concurrency, low-latency inference, and operational observability.

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI, Uvicorn (uvloop) |
| Database | MySQL 8 with async SQLAlchemy |
| Caching | Redis 7 (cache-aside pattern) |
| ML | scikit-learn, joblib |
| Containerization | Docker, Docker Compose |
| Monitoring | Prometheus, structured JSON logging |

## Features

- **Async inference** — CPU-bound model inference runs in a `ThreadPoolExecutor`, keeping the event loop free for 100+ concurrent requests
- **Redis caching** — predictions are cached by a SHA-256 hash of the input, eliminating redundant DB queries and model calls on repeated inputs
- **Optimized MySQL schema** — composite indexes on `(input_hash, model_version)` for cache lookups and `(model_version, created_at)` for time-range monitoring queries
- **Batch endpoint** — process up to 50 predictions in a single request using `asyncio.gather`
- **Observability** — Prometheus metrics at `/metrics`, structured JSON logs on every request, and a `/health` endpoint that independently checks the DB, cache, and model
- **Global error handling** — unhandled exceptions are caught, logged with context, and return a clean JSON response

## Project Structure

```
app/
├── main.py                    # App entrypoint, lifespan, error handler
├── core/
│   ├── config.py              # Environment-driven settings (Pydantic)
│   ├── logging.py             # JSON structured logging setup
│   └── middleware.py          # Per-request logging + Prometheus instrumentation
├── db/
│   ├── database.py            # Async engine, session factory, init/teardown
│   ├── models.py              # SQLAlchemy ORM models with indexes
│   └── cache.py               # Async Redis get/set/delete helpers
├── ml/
│   ├── model.py               # MLModel class — async predict, lazy loading
│   └── schemas.py             # Pydantic request/response schemas
├── services/
│   ├── prediction_service.py  # Cache-aside logic, DB persistence
│   └── monitoring.py          # Prometheus metric definitions
└── api/routes/
    ├── predictions.py         # Prediction endpoints
    └── health.py              # Health check + metrics endpoints
scripts/
├── train_model.py             # Train and serialize the demo model
└── init_db.sql                # MySQL DDL
tests/                         # pytest-asyncio test suite
```

## Getting Started

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) and Docker Compose

### Run with Docker

```bash
git clone <repo-url>
cd ML-Inference
docker-compose up --build
```

The API will be available at `http://localhost:8000`.

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# (Optional) Train and save the model — auto-loads a demo model if skipped
python scripts/train_model.py

# Start the server
uvicorn app.main:app --reload
```

A running MySQL and Redis instance are required for local development. Update `.env` with your connection details.

## API Reference

### `POST /api/v1/predictions/`

Run a single prediction.

**Request**
```json
{
  "features": [5.1, 3.5, 1.4, 0.2],
  "model_version": "v1"
}
```

**Response**
```json
{
  "request_id": "f3a2c1d0-...",
  "predicted_class": "setosa",
  "confidence": 0.98,
  "probabilities": {
    "setosa": 0.98,
    "versicolor": 0.01,
    "virginica": 0.01
  },
  "model_version": "v1",
  "inference_ms": 2.5,
  "cache_hit": false
}
```

### `POST /api/v1/predictions/batch`

Run up to 50 predictions in one request.

**Request**
```json
{
  "requests": [
    {"features": [5.1, 3.5, 1.4, 0.2]},
    {"features": [6.7, 3.0, 5.2, 2.3]}
  ]
}
```

### `GET /api/v1/predictions/history`

Retrieve past predictions with optional filtering.

| Query param | Type | Default | Description |
|---|---|---|---|
| `model_version` | string | — | Filter by model version |
| `limit` | int | 50 | Max results (1–200) |
| `offset` | int | 0 | Pagination offset |

### `GET /health`

Returns the status of the DB, cache, and model.

```json
{
  "status": "healthy",
  "db": "ok",
  "cache": "ok",
  "model": "ok",
  "version": "v1"
}
```

### `GET /metrics`

Prometheus metrics endpoint. Exposes request counts, latency histograms, prediction counts, and cache hit/miss counters.

### `GET /docs`

Interactive Swagger UI for exploring and testing all endpoints.

## Demo Model

The default model is a Random Forest classifier trained on the [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html). Features are `[sepal_length, sepal_width, petal_length, petal_width]` in centimeters, and the model predicts one of `setosa`, `versicolor`, or `virginica`.

To substitute your own model, serialize it with `joblib` in the following format and place it at the path configured in `MODEL_PATH`:

```python
import joblib

artifact = {
    "model": trained_sklearn_model,
    "version": "v2",
    "metadata": {
        "algorithm": "...",
        "feature_names": [...],
        "class_names": [...],
    },
}
joblib.dump(artifact, "models/classifier.joblib")
```

## Running Tests

```bash
pip install pytest pytest-asyncio httpx
pytest
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Environment name |
| `DB_HOST` | `localhost` | MySQL host |
| `DB_NAME` | `ml_inference` | Database name |
| `DB_USER` | `ml_user` | Database user |
| `DB_PASSWORD` | `ml_password` | Database password |
| `DB_POOL_SIZE` | `20` | SQLAlchemy connection pool size |
| `REDIS_HOST` | `localhost` | Redis host |
| `REDIS_TTL` | `300` | Cache TTL in seconds |
| `MODEL_PATH` | `./models/classifier.joblib` | Path to serialized model |
| `LOG_LEVEL` | `INFO` | Logging level |
