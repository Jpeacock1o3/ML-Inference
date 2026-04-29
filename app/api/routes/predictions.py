import asyncio
import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.ml.model import MLModel, get_model
from app.ml.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import get_prediction_history, run_prediction
from app.services.monitoring import PREDICTION_COUNT

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/predictions", tags=["predictions"])


@router.post("/", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(
    payload: PredictionRequest,
    db: AsyncSession = Depends(get_db),
    model: MLModel = Depends(get_model),
) -> PredictionResponse:
    try:
        return await run_prediction(payload, model, db)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        logger.error("Model inference failed", extra={"error": str(exc)}, exc_info=True)
        PREDICTION_COUNT.labels(model_version="unknown", status="error").inc()
        raise HTTPException(status_code=503, detail="Model unavailable")


@router.post("/batch", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
async def predict_batch(
    payload: BatchPredictionRequest,
    db: AsyncSession = Depends(get_db),
    model: MLModel = Depends(get_model),
) -> BatchPredictionResponse:
    start = time.perf_counter()
    tasks = [run_prediction(req, model, db) for req in payload.requests]
    try:
        results = await asyncio.gather(*tasks)
    except (ValueError, RuntimeError) as exc:
        logger.error("Batch inference failed", extra={"error": str(exc)}, exc_info=True)
        raise HTTPException(status_code=503, detail="Batch inference failed")

    return BatchPredictionResponse(
        results=list(results),
        total_ms=round((time.perf_counter() - start) * 1000, 3),
    )


@router.get("/history", status_code=status.HTTP_200_OK)
async def prediction_history(
    model_version: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    records = await get_prediction_history(db, model_version, limit, offset)
    return {
        "total": len(records),
        "limit": limit,
        "offset": offset,
        "results": [
            {
                "id": r.id,
                "request_id": r.request_id,
                "predicted_class": r.predicted_class,
                "confidence": r.confidence,
                "model_version": r.model_version,
                "inference_ms": r.inference_ms,
                "cache_hit": r.cache_hit,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in records
        ],
    }
