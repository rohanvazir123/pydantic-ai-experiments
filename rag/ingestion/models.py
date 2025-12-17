"""Data models for document ingestion and retrieval."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


class IngestedDocument(BaseModel):
    """Represents a full document before chunking."""

    id: str
    text: str
    metadata: dict[str, Any] = {}


class DocumentChunk(BaseModel):
    """Represents a document chunk (Pydantic model version)."""

    id: str
    text: str
    metadata: dict[str, Any] = {}


@dataclass
class ChunkData:
    """Represents a document chunk with optional embedding (dataclass version)."""

    content: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, Any]
    token_count: int | None = None
    embedding: list[float] | None = None

    def __post_init__(self):
        """Calculate token count if not provided."""
        if self.token_count is None:
            # Rough estimation: ~4 characters per token
            self.token_count = len(self.content) // 4


@dataclass
class ChunkingConfig:
    """Configuration for document chunking."""

    chunk_size: int = 1000  # Target characters per chunk
    chunk_overlap: int = 200  # Character overlap between chunks
    max_chunk_size: int = 2000  # Maximum chunk size
    min_chunk_size: int = 100  # Minimum chunk size
    max_tokens: int = 512  # Maximum tokens for embedding models

    def __post_init__(self):
        """Validate configuration."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        if self.min_chunk_size <= 0:
            raise ValueError("Minimum chunk size must be positive")


@dataclass
class IngestionConfig:
    """Configuration for document ingestion pipeline."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_chunk_size: int = 2000
    max_tokens: int = 512


@dataclass
class IngestionResult:
    """Result of document ingestion."""

    document_id: str
    title: str
    chunks_created: int
    processing_time_ms: float
    errors: list[str] = field(default_factory=list)


class SearchResult(BaseModel):
    """Model for search results."""

    chunk_id: str = Field(..., description="MongoDB ObjectId of chunk as string")
    document_id: str = Field(..., description="Parent document ObjectId as string")
    content: str = Field(..., description="Chunk text content")
    similarity: float = Field(..., description="Relevance score (0-1)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    document_title: str = Field(..., description="Title from document lookup")
    document_source: str = Field(..., description="Source from document lookup")


class RetrievedChunk(DocumentChunk):
    """A document chunk with relevance score from retrieval."""

    score: float
