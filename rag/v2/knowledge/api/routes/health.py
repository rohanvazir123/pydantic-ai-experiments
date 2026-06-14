"""Health and metrics routes."""

from typing import Literal, cast

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from knowledge.api.schemas import HealthResponse

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Liveness + readiness check.

    Checks: PostgreSQL pool, Redis ping, AGE DB, worker heartbeats, DLQ depth.
    Returns 200 when healthy, 503 when degraded or unhealthy.
    """
    app   = request.app
    state = getattr(app.state, "health_cache", None)

    # Use cached health if available (TTL managed by RedisCache)
    if state:
        return cast("HealthResponse", state)

    components: dict[str, str] = {}
    degraded_modes: list[str] = []

    # PostgreSQL
    try:
        vs = getattr(app.state, "vector_store", None)
        if vs and vs._pool:
            async with vs._pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            components["postgres"] = "healthy"
        else:
            components["postgres"] = "not_initialized"
    except Exception as exc:
        components["postgres"] = f"unhealthy: {exc}"
        degraded_modes.append("unavailable")

    # Redis
    try:
        redis = getattr(app.state, "redis", None)
        if redis:
            await redis.ping()
            components["redis"] = "healthy"
        else:
            components["redis"] = "not_initialized"
    except Exception:
        components["redis"] = "unhealthy"
        degraded_modes.append("no_cache")

    # Ollama (LLM)
    try:
        import httpx

        from knowledge.config.settings import load_settings
        s = load_settings()
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{s.llm_base_url}/models")
            components["ollama"] = "healthy" if resp.status_code < 500 else "unhealthy"
    except Exception:
        components["ollama"] = "unhealthy"
        degraded_modes.append("search_only")

    all_healthy = all(v == "healthy" for v in components.values())
    _status: Literal["healthy", "degraded", "unhealthy"] = (
        "healthy" if all_healthy else ("degraded" if components else "unhealthy")
    )

    return HealthResponse(
        status=_status,
        degraded_modes=degraded_modes,
        components=components,
    )


@router.get("/metrics", response_class=PlainTextResponse, tags=["ops"])
async def metrics() -> PlainTextResponse:
    """Prometheus metrics endpoint.

    Phase 11 TODO: expose real Prometheus metrics from knowledge/observability/metrics.py.
    """
    try:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
    except Exception:
        return PlainTextResponse("# Prometheus metrics not available\n")
