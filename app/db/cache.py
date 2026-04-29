import json
import logging
from typing import Any
import redis.asyncio as aioredis
from app.core.config import get_settings
from app.services.monitoring import CACHE_HITS, CACHE_MISSES

logger = logging.getLogger(__name__)
settings = get_settings()

_redis: aioredis.Redis | None = None


async def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _redis


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


async def cache_get(key: str) -> Any | None:
    try:
        client = await get_redis()
        raw = await client.get(key)
        if raw is not None:
            CACHE_HITS.labels(cache_type="prediction").inc()
            return json.loads(raw)
        CACHE_MISSES.labels(cache_type="prediction").inc()
        return None
    except Exception as exc:
        logger.warning("Cache get failed", extra={"key": key, "error": str(exc)})
        CACHE_MISSES.labels(cache_type="prediction").inc()
        return None


async def cache_set(key: str, value: Any, ttl: int | None = None) -> None:
    try:
        client = await get_redis()
        ttl = ttl if ttl is not None else settings.redis_ttl
        await client.setex(key, ttl, json.dumps(value))
    except Exception as exc:
        logger.warning("Cache set failed", extra={"key": key, "error": str(exc)})


async def cache_delete(key: str) -> None:
    try:
        client = await get_redis()
        await client.delete(key)
    except Exception as exc:
        logger.warning("Cache delete failed", extra={"key": key, "error": str(exc)})
