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
    MetadataFilter,
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


class TestMetadataFilter:
    """Unit tests for MetadataFilter model and is_empty property."""

    # --- is_empty ---

    def test_default_filter_is_empty(self):
        assert MetadataFilter().is_empty is True

    def test_empty_dicts_still_empty(self):
        assert MetadataFilter(
            metadata_eq={}, metadata_in={}, metadata_gte={}, metadata_lte={}, document_sources=[]
        ).is_empty is True

    def test_metadata_eq_not_empty(self):
        assert MetadataFilter(metadata_eq={"quarter": "Q4"}).is_empty is False

    def test_metadata_in_not_empty(self):
        assert MetadataFilter(metadata_in={"quarter": ["Q3", "Q4"]}).is_empty is False

    def test_document_source_not_empty(self):
        assert MetadataFilter(document_source="rag/documents/report.md").is_empty is False

    def test_document_sources_not_empty(self):
        assert MetadataFilter(document_sources=["a.md", "b.md"]).is_empty is False

    def test_document_title_not_empty(self):
        assert MetadataFilter(document_title="Q4 2024 Earnings").is_empty is False

    # --- Q4 2024 scenario ---

    def test_q4_2024_filter_not_empty(self):
        f = MetadataFilter(metadata_eq={"quarter": "Q4", "year": "2024"})
        assert f.is_empty is False
        assert f.metadata_eq["quarter"] == "Q4"
        assert f.metadata_eq["year"] == "2024"

    def test_q4_2024_combined_with_doc_source(self):
        f = MetadataFilter(
            metadata_eq={"quarter": "Q4", "year": "2024"},
            document_source="rag/documents/earnings/amazon_q4_2024.md",
        )
        assert f.is_empty is False
        assert f.metadata_eq == {"quarter": "Q4", "year": "2024"}
        assert f.document_source == "rag/documents/earnings/amazon_q4_2024.md"

    def test_multi_quarter_in_filter(self):
        f = MetadataFilter(metadata_in={"quarter": ["Q3", "Q4"], "year": ["2024"]})
        assert f.is_empty is False
        assert "Q3" in f.metadata_in["quarter"]
        assert "Q4" in f.metadata_in["quarter"]

    def test_metadata_gte_not_empty(self):
        assert MetadataFilter(metadata_gte={"date": "2024-10-01"}).is_empty is False

    def test_metadata_lte_not_empty(self):
        assert MetadataFilter(metadata_lte={"date": "2024-12-31"}).is_empty is False

    def test_q4_2024_date_range_filter(self):
        f = MetadataFilter(
            metadata_gte={"date": "2024-10-01"},
            metadata_lte={"date": "2024-12-31"},
        )
        assert f.is_empty is False
        assert f.metadata_gte["date"] == "2024-10-01"
        assert f.metadata_lte["date"] == "2024-12-31"

    # --- field defaults ---

    def test_metadata_eq_defaults_to_empty_dict(self):
        assert MetadataFilter().metadata_eq == {}

    def test_metadata_in_defaults_to_empty_dict(self):
        assert MetadataFilter().metadata_in == {}

    def test_metadata_gte_defaults_to_empty_dict(self):
        assert MetadataFilter().metadata_gte == {}

    def test_metadata_lte_defaults_to_empty_dict(self):
        assert MetadataFilter().metadata_lte == {}

    def test_document_sources_defaults_to_empty_list(self):
        assert MetadataFilter().document_sources == []

    def test_document_source_defaults_to_none(self):
        assert MetadataFilter().document_source is None

    def test_document_title_defaults_to_none(self):
        assert MetadataFilter().document_title is None
