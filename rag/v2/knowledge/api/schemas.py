"""Pydantic request/response models for the knowledge API.

APIResponse[T] is the standard envelope:
  { "request_id": uuid, "data": T | null, "error": ErrorDetail | null, "cache_hit": str | null }

All error responses have data=null; all success responses have error=null.
Never return a bare string or untyped dict — always use this envelope.
"""

from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ── Error envelope ────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    code:          str
    message:       str
    details:       dict[str, Any] = Field(default_factory=dict)
    retry_after_s: int | None     = None
    doc_url:       str | None     = None


class APIResponse(BaseModel, Generic[T]):
    request_id: str
    data:       T | None          = None
    error:      ErrorDetail | None = None
    cache_hit:  str | None        = None   # "l2" | "l3" | None


# ── Auth ──────────────────────────────────────────────────────────────────────

class TokenRequest(BaseModel):
    email:    str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = 900          # seconds


class RefreshResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int = 900


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query:           str
    corpus_ids:      list[str]
    session_id:      str                           # required — UUID per conversation
    model_tier:      Literal["auto", "small", "large"] = "auto"
    message_history: list[dict[str, Any]] | None  = None


class ChatResponse(BaseModel):
    answer:                  str
    status:                  str
    confidence:              float | None          = None
    citations:               list[dict[str, Any]]  = Field(default_factory=list)
    low_confidence_warning:  bool                  = False
    pipeline_latency_ms:     dict[str, int]        = Field(default_factory=dict)
    estimated_cost_usd:      float                 = 0.0
    model_tier_used:         str                   = "small"
    prompt_tokens:           int                   = 0
    completion_tokens:       int                   = 0
    cache_hit:               str | None            = None
    request_id:              str                   = ""
    trace_url:               str | None            = None
    abstention_layer:        int | None            = None
    abstention_reason:       str | None            = None


# ── Search ────────────────────────────────────────────────────────────────────

class SearchRequest(BaseModel):
    query:           str
    corpus_ids:      list[str]
    k:               int           = Field(default=5, ge=1, le=50)
    search_type:     Literal["hybrid", "semantic", "text"] = "hybrid"
    include_graph:   bool          = False
    metadata_filter: dict[str, str] | None = None


class SearchResultItem(BaseModel):
    chunk_id:        str
    document_title:  str
    document_source: str
    content:         str
    confidence:      float | None = None
    excerpt:         str          = ""


class SearchResponse(BaseModel):
    results: list[SearchResultItem]
    query:   str
    k:       int


# ── Ingest ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    corpus_id:               str
    source_path:             str | None  = None
    source_url:              str | None  = None
    enable_graph_extraction: bool        = False
    mode:                    Literal["full", "incremental"] = "incremental"


class IngestJobResponse(BaseModel):
    job_id:      str
    status:      str = "queued"
    corpus_id:   str
    submitted_at: str


class JobStatusResponse(BaseModel):
    job_id:          str
    status:          str
    progress:        int          = 0
    corpus_id:       str          = ""
    chunks_ingested: int | None   = None
    error:           str | None   = None
    submitted_at:    str | None   = None
    completed_at:    str | None   = None


# ── Corpus ────────────────────────────────────────────────────────────────────

class CorpusInfo(BaseModel):
    id:                      str
    display_name:            str
    source_folders:          list[str]
    allowed_roles:           list[str]
    enable_graph_extraction: bool
    graph_ontology_path:     str | None = None


# ── Scheduler ─────────────────────────────────────────────────────────────────

class ScheduledJobRequest(BaseModel):
    name:                    str
    source_type:             Literal["local", "url", "s3", "gcs"]
    source_config:           dict[str, str]
    corpus_id:               str
    cron_expr:               str
    mode:                    Literal["full", "incremental"] = "incremental"
    enable_graph_extraction: bool = False


class ScheduledJobResponse(BaseModel):
    id:           str
    name:         str
    corpus_id:    str
    cron_expr:    str
    mode:         str
    is_active:    bool
    next_run_at:  str | None = None
    last_run_at:  str | None = None
    last_status:  str | None = None


# ── Evaluation ────────────────────────────────────────────────────────────────

class EvalRunRequest(BaseModel):
    corpus_id:       str
    k:               int   = Field(default=5, ge=1, le=20)
    model_tier:      Literal["small", "large"] = "small"
    search_type:     Literal["hybrid", "semantic", "text"] = "hybrid"
    baseline_run_id: str | None = None


class EvalRunResponse(BaseModel):
    run_id:       str
    corpus_id:    str
    status:       str
    sample_count: int = 0


# ── Feedback ──────────────────────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    request_id: str
    thumbs:     bool | None  = None
    rating:     int | None   = Field(default=None, ge=1, le=5)
    correction: str | None   = None
    tags:       list[str]    = Field(default_factory=list)


class SignalRequest(BaseModel):
    session_id:  str
    signal_type: Literal[
        "query_reformulation", "follow_up_question",
        "session_abandoned", "copy_action", "escalation",
    ]
    request_id:  str | None = None


# ── Memory ────────────────────────────────────────────────────────────────────

class ConversationSummary(BaseModel):
    id:          str
    session_id:  str
    title:       str | None  = None
    summary:     str | None  = None
    turn_count:  int         = 0
    last_turn_at: str | None = None


class MessageItem(BaseModel):
    id:               str
    role:             Literal["user", "assistant"]
    content:          str
    pipeline_status:  str | None = None
    cost_usd:         float | None = None
    created_at:       str | None = None


class MemoryItem(BaseModel):
    id:         str
    content:    str
    created_at: str | None = None


class AddMemoryRequest(BaseModel):
    content: str


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:         Literal["healthy", "degraded", "unhealthy"]
    degraded_modes: list[str]             = Field(default_factory=list)
    components:     dict[str, str]        = Field(default_factory=dict)
    dlq_depth:      int                   = 0
