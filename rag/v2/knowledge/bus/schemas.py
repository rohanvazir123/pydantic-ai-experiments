"""Pydantic models for all Redis Streams message types.

Every message that flows through knowledge:ingest, knowledge:search,
knowledge:eval, or knowledge:events is one of these models.

Serialisation: model_dump_json() → XADD field; model_validate_json() on consume.
The `attempt` field on IngestJob / SearchRequest tracks retry depth — it is
incremented by _execute_with_retry before re-enqueueing on transient failure.
"""

from datetime import datetime, UTC
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


def _job_id() -> str:
    return str(uuid4())


class IngestJob(BaseModel):
    """Published to knowledge:ingest stream by the API on POST /ingest."""

    job_id:                  str      = Field(default_factory=_job_id)
    tenant_id:               str
    corpus_id:               str
    source_path:             str | None = None     # local folder / file path
    source_url:              str | None = None     # remote URL to download
    enable_graph_extraction: bool    = False
    mode:                    Literal["full", "incremental"] = "incremental"
    attempt:                 int     = 1           # incremented on retry; 1-indexed
    submitted_at:            datetime = Field(default_factory=_now)


class SearchRequest(BaseModel):
    """Published to knowledge:search stream for async / bulk search batches.

    Interactive queries skip the stream and call the retriever directly.
    """

    request_id:   str      = Field(default_factory=_job_id)
    tenant_id:    str
    corpus_ids:   list[str]
    query:        str
    k:            int      = 5
    callback_key: str | None = None  # Redis key to LPUSH the result when done
    attempt:      int      = 1
    submitted_at: datetime = Field(default_factory=_now)


class EvalJob(BaseModel):
    """Published to knowledge:eval stream by POST /evaluate/run."""

    run_id:      str = Field(default_factory=_job_id)
    corpus_id:   str
    tenant_id:   str
    model_tier:  Literal["small", "large"] = "small"
    search_type: Literal["hybrid", "semantic", "text"] = "hybrid"
    k:           int = 5
    baseline_run_id: str | None = None
    submitted_at: datetime = Field(default_factory=_now)


class WorkerEvent(BaseModel):
    """Published to knowledge:events stream by workers.

    The API's SSE job-progress endpoint filters this stream by job_id.
    """

    event_type: Literal["heartbeat", "job_started", "job_completed", "job_failed"]
    worker_id:  str
    job_id:     str | None = None    # None for heartbeat events
    tenant_id:  str | None = None
    corpus_id:  str | None = None
    progress:   int | None = None   # 0-100 percentage
    error:      str | None = None
    ts:         datetime   = Field(default_factory=_now)
