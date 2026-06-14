"""Log viewer route — on-demand, not streaming (admin only)."""

import json
import uuid

from fastapi import APIRouter, Query, Request

from knowledge.api.middleware import get_request_id

router = APIRouter(prefix="/logs", tags=["logs"])

# Ascending severity rank — used for min-level filtering
_LEVEL_RANK: dict[str, int] = {
    "DEBUG":    0,
    "INFO":     1,
    "WARNING":  2,
    "ERROR":    3,
    "CRITICAL": 4,
}


@router.get("", response_model=dict)
async def get_logs(
    request: Request,
    level:      str | None = Query(default="DEBUG",  description="Minimum severity level"),
    levels:     str | None = Query(default=None,     description="Comma-separated list of exact levels to include (overrides `level`)"),
    service:    str | None = Query(default=None),
    corpus_id:  str | None = Query(default=None),
    request_id_filter: str | None = Query(default=None, alias="request_id"),
    limit:      int        = Query(default=200, ge=1, le=2000),
) -> dict:
    """Return recent log entries from the Redis ring buffer.

    Reads from knowledge:logs:recent (last 5,000 entries, 24h TTL).
    On-demand only — no streaming.

    Level filtering:
    - `level=INFO`  returns INFO, WARNING, ERROR, CRITICAL (minimum severity)
    - `levels=DEBUG,ERROR` returns only DEBUG and ERROR entries (exact match set)
    """
    redis      = getattr(request.app.state, "redis", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if redis is None:
        return {"request_id": request_id, "data": [], "error": None}

    # Build exact-match level set if caller specified individual levels
    exact_levels: set[str] | None = None
    if levels:
        exact_levels = {l.strip().upper() for l in levels.split(",") if l.strip()}

    min_rank = _LEVEL_RANK.get((level or "DEBUG").upper(), 0)

    # Fetch a generous buffer from Redis — filtering reduces the final count
    raw_entries = await redis.lrange("knowledge:logs:recent", 0, limit * 5 - 1)

    entries = []
    for raw in raw_entries:
        try:
            entry = json.loads(raw)
        except Exception:
            continue

        entry_level = entry.get("level", "").upper()

        # Level filter
        if exact_levels is not None:
            if entry_level not in exact_levels:
                continue
        else:
            if _LEVEL_RANK.get(entry_level, 0) < min_rank:
                continue

        # Field filters
        if service and entry.get("service") != service:
            continue
        if corpus_id and entry.get("corpus_id") != corpus_id:
            continue
        if request_id_filter and entry.get("request_id") != request_id_filter:
            continue

        entries.append(entry)
        if len(entries) >= limit:
            break

    return {"request_id": request_id, "data": entries, "error": None}
