# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Data models for the ingestion pipeline.

NOTE: SearchResult and Citation are defined here because they originate at
ingestion time (chunks carry the raw scores). They are consumed by retrieval,
agent, and API layers. If circular imports arise, move them to knowledge/models.py.
"""

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

# ── Chunking configuration ────────────────────────────────────────────────────

class ChunkingConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_tokens: int = 512


class IngestionConfig(BaseModel):
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    max_tokens: int = 512


# ── Chunk data (produced by the chunker) ─────────────────────────────────────

class ChunkData(BaseModel):
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = 0
    token_count: int = 0
    start_char: int = 0
    end_char: int = 0
    corpus_id: str = ""
    tenant_id: str = ""


# ── Docling conversion result ─────────────────────────────────────────────────

class ConversionResult(BaseModel):
    """Returned by DoclingProcessor.process()."""

    model_config = {"arbitrary_types_allowed": True}

    markdown: str
    docling_doc: Any | None = None    # DoclingDocument | None
    format: str = "unknown"            # "pdf", "docx", "audio", "markdown", "text"


# ── Search / retrieval models ─────────────────────────────────────────────────

class SearchResult(BaseModel):
    """One ranked chunk returned by the retriever.

    raw_score / raw_score_type describe the score from the search leg.
    confidence is populated by the CrossEncoder reranker — None until then.
    Citation.relevance_score maps to confidence, never raw_score.
    """

    chunk_id: UUID
    document_id: UUID
    document_title: str
    document_source: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    raw_score: float
    raw_score_type: Literal["cosine_similarity", "ts_rank", "rrf"]
    confidence: float | None = None    # set by CrossEncoder; None until reranked


class Citation(BaseModel):
    """A cited source chunk in the agent's response.

    relevance_score = SearchResult.confidence (post-rerank sigmoid, 0-1).
    Never set from raw_score — raw scores are not calibrated across search types.
    """

    chunk_id: UUID
    document_title: str
    document_source: str
    relevance_score: float             # = SearchResult.confidence
    excerpt: str                       # ≤ 200 chars of the supporting chunk


# ── Ingestion results ─────────────────────────────────────────────────────────

class IngestResult(BaseModel):
    """Returned by DocumentIngestionPipeline.run() per job."""

    job_id: str
    chunks_ingested: int = 0
    graph_entities: int = 0
    duration_s: float = 0.0
    skipped: bool = False              # True when fingerprint cache hit (no change)
    errors: list[str] = Field(default_factory=list)


class IngestionResult(BaseModel):
    """Per-document result (mirrors v1's IngestionResult for compatibility)."""

    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    errors: list[str] = Field(default_factory=list)


# ── Metadata filter (used by search / agent tools) ────────────────────────────

class MetadataFilter(BaseModel):
    """Optional filter applied during retrieval."""

    document_source: str | None = None
    metadata_eq: dict[str, str] = Field(default_factory=dict)
