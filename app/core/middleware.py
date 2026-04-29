import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.logging import generate_request_id, request_id_var
from app.services.monitoring import REQUEST_COUNT, REQUEST_LATENCY

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = generate_request_id()
        token = request_id_var.set(request_id)
        start = time.perf_counter()

        logger.info(
            "Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
            },
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "Unhandled exception",
                extra={"request_id": request_id, "error": str(exc)},
                exc_info=True,
            )
            raise
        finally:
            latency = time.perf_counter() - start
            status = getattr(response, "status_code", 500)

            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status=status,
            ).inc()
            REQUEST_LATENCY.labels(
                method=request.method,
                endpoint=request.url.path,
            ).observe(latency)

            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status,
                    "latency_ms": round(latency * 1000, 2),
                },
            )
            request_id_var.reset(token)

        response.headers["X-Request-ID"] = request_id
        return response
