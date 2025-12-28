"""
Data models for document ingestion and retrieval.

Module: rag.ingestion.models
============================

This module defines all data models used throughout the RAG ingestion and
retrieval pipeline. Models use both Pydantic (for validation) and dataclasses
(for lightweight data containers).

Classes
-------
IngestedDocument(BaseModel)
    Full document before chunking.
    Attributes: id, text, metadata

DocumentChunk(BaseModel)
    Pydantic model for document chunks.
    Attributes: id, text, metadata

ChunkData (dataclass)
    Document chunk with optional embedding.
    Attributes:
        content: str          - Chunk text content
        index: int            - Position in document (0-indexed)
        start_char: int       - Start character offset
        end_char: int         - End character offset
        metadata: dict        - Chunk metadata
        token_count: int|None - Token count (auto-estimated if None)
        embedding: list|None  - Embedding vector (768 dims for nomic-embed-text)

ChunkingConfig (dataclass)
    Configuration for chunking behavior.
    Attributes:
        chunk_size: int       - Target chars per chunk (default: 1000)
        chunk_overlap: int    - Overlap between chunks (default: 200)
        max_chunk_size: int   - Maximum chunk size (default: 2000)
        min_chunk_size: int   - Minimum chunk size (default: 100)
        max_tokens: int       - Max tokens for embeddings (default: 512)

IngestionConfig (dataclass)
    Configuration for ingestion pipeline.
    Attributes: chunk_size, chunk_overlap, max_chunk_size, max_tokens

IngestionResult (dataclass)
    Result of document ingestion.
    Attributes:
        document_id: str      - MongoDB document ID
        title: str            - Document title
        chunks_created: int   - Number of chunks created
        processing_time_ms: float - Processing time
        errors: list[str]     - Any errors encountered

SearchResult(BaseModel)
    Search result with document context.
    Attributes:
        chunk_id: str         - Chunk MongoDB ObjectId
        document_id: str      - Parent document ObjectId
        content: str          - Chunk text
        similarity: float     - Relevance score (0-1)
        metadata: dict        - Chunk metadata
        document_title: str   - Parent document title
        document_source: str  - Parent document source

RetrievedChunk(DocumentChunk)
    Document chunk with retrieval score.
    Attributes: id, text, metadata, score

Usage
-----
    from rag.ingestion.models import ChunkData, ChunkingConfig, SearchResult

    # Create chunking config
    config = ChunkingConfig(chunk_size=500, chunk_overlap=100)

    # Create a chunk
    chunk = ChunkData(
        content="Sample text...",
        index=0,
        start_char=0,
        end_char=100,
        metadata={"source": "doc.pdf"}
    )

    # Create search result
    result = SearchResult(
        chunk_id="abc123",
        document_id="doc456",
        content="Found text...",
        similarity=0.95,
        document_title="My Document",
        document_source="doc.pdf"
    )
"""

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


if __name__ == "__main__":
    # Standalone test for models module
    print("=" * 60)
    print("RAG Models Module Test")
    print("=" * 60)

    # Test ChunkingConfig
    print("\n--- ChunkingConfig ---")
    config = ChunkingConfig(chunk_size=500, chunk_overlap=100, max_tokens=256)
    print(f"  chunk_size: {config.chunk_size}")
    print(f"  chunk_overlap: {config.chunk_overlap}")
    print(f"  max_tokens: {config.max_tokens}")

    # Test invalid config
    print("\n  Testing validation (overlap >= size should fail)...")
    try:
        invalid = ChunkingConfig(chunk_size=100, chunk_overlap=150)
        print("  ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  OK: Caught expected error: {e}")

    # Test ChunkData
    print("\n--- ChunkData ---")
    chunk = ChunkData(
        content="This is sample content for testing the chunk data model.",
        index=0,
        start_char=0,
        end_char=55,
        metadata={"source": "test.pdf", "page": 1},
    )
    print(f"  content: {chunk.content[:40]}...")
    print(f"  index: {chunk.index}")
    print(f"  token_count (auto): {chunk.token_count}")
    print(f"  embedding: {chunk.embedding}")

    # Add embedding
    chunk.embedding = [0.1] * 768
    print(f"  embedding (after set): [{chunk.embedding[0]}...] ({len(chunk.embedding)} dims)")

    # Test IngestionResult
    print("\n--- IngestionResult ---")
    result = IngestionResult(
        document_id="abc123",
        title="Test Document",
        chunks_created=5,
        processing_time_ms=123.45,
    )
    print(f"  document_id: {result.document_id}")
    print(f"  title: {result.title}")
    print(f"  chunks_created: {result.chunks_created}")
    print(f"  processing_time_ms: {result.processing_time_ms}")
    print(f"  errors: {result.errors}")

    # Test SearchResult
    print("\n--- SearchResult ---")
    search_result = SearchResult(
        chunk_id="chunk789",
        document_id="doc456",
        content="Found relevant content here...",
        similarity=0.87,
        metadata={"page": 3},
        document_title="Important Report",
        document_source="report.pdf",
    )
    print(f"  chunk_id: {search_result.chunk_id}")
    print(f"  document_title: {search_result.document_title}")
    print(f"  similarity: {search_result.similarity}")
    print(f"  content: {search_result.content[:30]}...")

    # Test JSON serialization
    print("\n--- JSON Serialization ---")
    json_str = search_result.model_dump_json(indent=2)
    print(f"  SearchResult as JSON:\n{json_str}")

    print("\n" + "=" * 60)
    print("Models test completed successfully!")
    print("=" * 60)
