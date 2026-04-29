import logging
from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from sqlalchemy import text

from app.db.database import AsyncSessionLocal
from app.db.cache import get_redis
from app.ml.model import get_model
from app.ml.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    db_status = "ok"
    cache_status = "ok"
    model_status = "ok"

    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:
        logger.error("DB health check failed", extra={"error": str(exc)})
        db_status = "error"

    try:
        client = await get_redis()
        await client.ping()
    except Exception as exc:
        logger.warning("Cache health check failed", extra={"error": str(exc)})
        cache_status = "error"

    try:
        model = get_model()
        if model._model is None:
            model_status = "not_loaded"
    except Exception as exc:
        logger.error("Model health check failed", extra={"error": str(exc)})
        model_status = "error"

    overall = "healthy" if all(s == "ok" for s in (db_status, cache_status, model_status)) else "degraded"

    return HealthResponse(
        status=overall,
        db=db_status,
        cache=cache_status,
        model=model_status,
        version=get_model().version,
    )


@router.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
