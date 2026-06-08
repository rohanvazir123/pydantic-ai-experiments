"""Log viewer route — on-demand, not streaming (admin only)."""

import json
import uuid

from fastapi import APIRouter, Query, Request

from knowledge.api.middleware import get_request_id

router = APIRouter(prefix="/logs", tags=["logs"])


@router.get("", response_model=dict)
async def get_logs(
    request: Request,
    level:      str | None = Query(default="INFO"),
    service:    str | None = Query(default=None),
    corpus_id:  str | None = Query(default=None),
    request_id_filter: str | None = Query(default=None, alias="request_id"),
    limit:      int        = Query(default=100, ge=1, le=500),
) -> dict:
    """Return recent log entries from the Redis ring buffer.

    Reads from knowledge:logs:recent (last 5,000 entries, 24h TTL).
    On-demand only — no streaming.
    """
    redis      = getattr(request.app.state, "redis", None)
    request_id = get_request_id() or str(uuid.uuid4())

    if redis is None:
        return {"request_id": request_id, "data": [], "error": None}

    # Read from LPUSH ring buffer (newest first, since LPUSH prepends)
    raw_entries = await redis.lrange("knowledge:logs:recent", 0, limit * 3 - 1)

    entries = []
    for raw in raw_entries:
        try:
            entry = json.loads(raw)
        except Exception:
            continue

        # Apply filters
        if level and entry.get("level", "").upper() < level.upper():
            continue
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
