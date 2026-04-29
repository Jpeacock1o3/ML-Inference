import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.core.middleware import RequestLoggingMiddleware
from app.db.database import init_db, close_db
from app.db.cache import close_redis
from app.ml.model import get_model
from app.api.routes import predictions, health

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ML Inference API", extra={"env": settings.app_env})
    try:
        await init_db()
    except Exception as exc:
        logger.warning("DB init skipped (likely no DB in dev)", extra={"error": str(exc)})
    get_model()  # warm up model on startup
    logger.info("Startup complete")
    yield
    await close_db()
    await close_redis()
    logger.info("Shutdown complete")


app = FastAPI(
    title="ML Inference API",
    description="Real-time ML predictions via REST API",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(RequestLoggingMiddleware)

app.include_router(predictions.router)
app.include_router(health.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "Unhandled exception",
        extra={"path": request.url.path, "error": str(exc)},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )
