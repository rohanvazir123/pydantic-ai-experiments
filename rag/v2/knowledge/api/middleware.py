"""API middleware stack.

Applied in this order (outermost first):
  1. CorrelationID  — set X-Request-ID header; inject into contextvars for log correlation
  2. StructuredLog  — emit one JSON log line per request with latency, user_id, corpus_id
  3. AuditEmitter   — background task: INSERT INTO audit_events after every auth'd request
  4. CORS           — configured in app.py via FastAPI CORSMiddleware
  5. RateLimiter    — slowapi, configured in app.py

The X-Request-ID is the correlation key across logs, Langfuse trace, Prometheus,
and audit_events. It is returned in every API response as request_id.
"""

import logging
import time
import uuid as _uuid
from contextvars import ContextVar

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

# Per-request context variables — safe for concurrent async requests
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")
_user_id_var:    ContextVar[str] = ContextVar("user_id",    default="")
_tenant_id_var:  ContextVar[str] = ContextVar("tenant_id",  default="")


def get_request_id() -> str:
    return _request_id_var.get()


def get_user_id() -> str:
    return _user_id_var.get()


def get_tenant_id() -> str:
    return _tenant_id_var.get()


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Sets X-Request-ID on every request/response and injects into contextvars."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Use client-provided ID if present (for distributed tracing), else generate
        request_id = request.headers.get("X-Request-ID") or str(_uuid.uuid4())
        _request_id_var.set(request_id)

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class StructuredLogMiddleware(BaseHTTPMiddleware):
    """Emits one JSON-compatible structured log line per request."""

    async def dispatch(self, request: Request, call_next) -> Response:
        t0 = time.monotonic()
        response = await call_next(request)
        latency_ms = int((time.monotonic() - t0) * 1000)

        logger.info(
            "request",
            extra={
                "request_id": get_request_id(),
                "user_id":    get_user_id() or None,
                "tenant_id":  get_tenant_id() or None,
                "method":     request.method,
                "path":       request.url.path,
                "status":     response.status_code,
                "latency_ms": latency_ms,
            },
        )
        return response
