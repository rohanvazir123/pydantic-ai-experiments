# Copyright 2024 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ingestion models and pipeline."""

import pytest

from rag.ingestion.models import (
    ChunkData,
    ChunkingConfig,
    IngestionConfig,
    IngestionResult,
    SearchResult,
)


class TestChunkData:
    """Test ChunkData dataclass."""

    def test_chunk_data_creation(self):
        """Test creating a ChunkData instance."""
        chunk = ChunkData(
            content="This is test content",
            index=0,
            start_char=0,
            end_char=20,
            metadata={"source": "test.md"},
        )
        assert chunk.content == "This is test content"
        assert chunk.index == 0
        assert chunk.start_char == 0
        assert chunk.end_char == 20
        assert chunk.metadata == {"source": "test.md"}

    def test_chunk_data_auto_token_count(self):
        """Test that token count is auto-calculated."""
        content = "a" * 100  # 100 characters
        chunk = ChunkData(
            content=content,
            index=0,
            start_char=0,
            end_char=100,
            metadata={},
        )
        # ~4 characters per token
        assert chunk.token_count == 25

    def test_chunk_data_with_explicit_token_count(self):
        """Test that explicit token count is preserved."""
        chunk = ChunkData(
            content="test content",
            index=0,
            start_char=0,
            end_char=12,
            metadata={},
            token_count=50,
        )
        assert chunk.token_count == 50

    def test_chunk_data_with_embedding(self):
        """Test creating ChunkData with embedding."""
        embedding = [0.1, 0.2, 0.3]
        chunk = ChunkData(
            content="test",
            index=0,
            start_char=0,
            end_char=4,
            metadata={},
            embedding=embedding,
        )
        assert chunk.embedding == embedding


class TestChunkingConfig:
    """Test ChunkingConfig dataclass."""

    def test_default_config(self):
        """Test default chunking configuration."""
        config = ChunkingConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_chunk_size == 2000
        assert config.min_chunk_size == 100
        assert config.max_tokens == 512

    def test_custom_config(self):
        """Test custom chunking configuration."""
        config = ChunkingConfig(
            chunk_size=500,
            chunk_overlap=100,
            max_chunk_size=1000,
            min_chunk_size=50,
            max_tokens=256,
        )
        assert config.chunk_size == 500
        assert config.chunk_overlap == 100

    def test_config_validation_overlap_too_large(self):
        """Test that overlap >= chunk_size raises error."""
        with pytest.raises(
            ValueError, match="Chunk overlap must be less than chunk size"
        ):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_config_validation_negative_min_chunk(self):
        """Test that non-positive min_chunk_size raises error."""
        with pytest.raises(ValueError, match="Minimum chunk size must be positive"):
            ChunkingConfig(min_chunk_size=0)


class TestIngestionConfig:
    """Test IngestionConfig dataclass."""

    def test_default_ingestion_config(self):
        """Test default ingestion configuration."""
        config = IngestionConfig()
        assert config.chunk_size == 1000
        assert config.chunk_overlap == 200
        assert config.max_chunk_size == 2000
        assert config.max_tokens == 512

    def test_custom_ingestion_config(self):
        """Test custom ingestion configuration."""
        config = IngestionConfig(
            chunk_size=800,
            chunk_overlap=150,
            max_chunk_size=1600,
            max_tokens=384,
        )
        assert config.chunk_size == 800
        assert config.max_tokens == 384


class TestIngestionResult:
    """Test IngestionResult dataclass."""

    def test_ingestion_result_success(self):
        """Test successful ingestion result."""
        result = IngestionResult(
            document_id="doc123",
            title="Test Document",
            chunks_created=10,
            processing_time_ms=150.5,
        )
        assert result.document_id == "doc123"
        assert result.title == "Test Document"
        assert result.chunks_created == 10
        assert result.processing_time_ms == 150.5
        assert result.errors == []

    def test_ingestion_result_with_errors(self):
        """Test ingestion result with errors."""
        result = IngestionResult(
            document_id="",
            title="Failed Document",
            chunks_created=0,
            processing_time_ms=50.0,
            errors=["File not found", "Invalid format"],
        )
        assert result.document_id == ""
        assert result.chunks_created == 0
        assert len(result.errors) == 2
        assert "File not found" in result.errors


class TestSearchResult:
    """Test SearchResult model."""

    def test_search_result_creation(self):
        """Test creating a SearchResult instance."""
        result = SearchResult(
            chunk_id="chunk123",
            document_id="doc456",
            content="This is the chunk content",
            similarity=0.85,
            metadata={"page": 1},
            document_title="My Document",
            document_source="docs/my_document.md",
        )
        assert result.chunk_id == "chunk123"
        assert result.document_id == "doc456"
        assert result.content == "This is the chunk content"
        assert result.similarity == 0.85
        assert result.metadata == {"page": 1}
        assert result.document_title == "My Document"
        assert result.document_source == "docs/my_document.md"

    def test_search_result_default_metadata(self):
        """Test SearchResult with default empty metadata."""
        result = SearchResult(
            chunk_id="chunk123",
            document_id="doc456",
            content="Content",
            similarity=0.9,
            document_title="Title",
            document_source="source.md",
        )
        assert result.metadata == {}
