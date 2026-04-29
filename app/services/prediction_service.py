import logging
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.cache import cache_get, cache_set
from app.db.models import Prediction
from app.ml.model import MLModel, compute_input_hash
from app.ml.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


async def run_prediction(
    request: PredictionRequest,
    model: MLModel,
    db: AsyncSession,
) -> PredictionResponse:
    request_id = str(uuid.uuid4())
    input_hash = compute_input_hash(request.features, model.version)
    cache_key = f"pred:{input_hash}"

    cached = await cache_get(cache_key)
    if cached:
        logger.info("Cache hit", extra={"input_hash": input_hash})
        return PredictionResponse(
            request_id=request_id,
            cache_hit=True,
            **{k: cached[k] for k in ("predicted_class", "confidence", "probabilities", "inference_ms")},
            model_version=model.version,
        )

    result = await model.predict(request.features)

    await cache_set(cache_key, result)

    record = Prediction(
        request_id=request_id,
        input_hash=input_hash,
        input_data={"features": request.features},
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_version=model.version,
        inference_ms=result["inference_ms"],
        cache_hit=False,
    )
    db.add(record)

    return PredictionResponse(
        request_id=request_id,
        predicted_class=result["predicted_class"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        model_version=model.version,
        inference_ms=result["inference_ms"],
        cache_hit=False,
    )


async def get_prediction_history(
    db: AsyncSession,
    model_version: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Prediction]:
    query = select(Prediction).order_by(Prediction.created_at.desc())
    if model_version:
        query = query.where(Prediction.model_version == model_version)
    query = query.limit(limit).offset(offset)
    result = await db.execute(query)
    return list(result.scalars().all())
