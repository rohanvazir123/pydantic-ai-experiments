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

"""Tests for PostgreSQL/pgvector vector store."""

import pytest
import pytest_asyncio

from rag.config.settings import load_settings
from rag.storage.vector_store.postgres import PostgresHybridStore


class TestPostgresConnection:
    """Test PostgreSQL connection and basic operations."""

    @pytest.fixture
    def settings(self):
        """Load settings fixture."""
        return load_settings()

    @pytest.fixture
    def store(self):
        """Create PostgresHybridStore fixture."""
        return PostgresHybridStore()

    def test_store_initialization(self, store):
        """Test that store initializes with correct settings."""
        assert store.settings is not None
        assert store.pool is None  # Not connected yet
        assert store._initialized is False

    def test_settings_have_database_url(self, settings):
        """Test that settings have database URL configured."""
        assert settings.database_url is not None
        assert len(settings.database_url) > 0

    def test_settings_have_table_names(self, settings):
        """Test that settings have table names configured."""
        assert settings.postgres_table_documents is not None
        assert settings.postgres_table_chunks is not None
        assert len(settings.postgres_table_documents) > 0
        assert len(settings.postgres_table_chunks) > 0


@pytest.mark.asyncio
class TestPostgresConnectionLive:
    """Test PostgreSQL connection - requires live PostgreSQL connection."""

    @pytest.fixture
    def settings(self):
        """Load settings fixture."""
        return load_settings()

    @pytest_asyncio.fixture
    async def connected_store(self):
        """Create and initialize a connected PostgresHybridStore."""
        store = PostgresHybridStore()
        await store.initialize()
        yield store
        await store.close()

    async def test_postgres_connection(self, connected_store):
        """Test that PostgreSQL connection is established."""
        assert connected_store._initialized is True
        assert connected_store.pool is not None

    async def test_tables_exist(self, connected_store, settings):
        """Test that required tables exist."""
        async with connected_store.pool.acquire() as conn:
            # Check documents table
            docs_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = $1
                )
            """, settings.postgres_table_documents)
            assert docs_exists, f"Documents table '{settings.postgres_table_documents}' not found"

            # Check chunks table
            chunks_exists = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables
                    WHERE table_name = $1
                )
            """, settings.postgres_table_chunks)
            assert chunks_exists, f"Chunks table '{settings.postgres_table_chunks}' not found"

    async def test_pgvector_extension_enabled(self, connected_store):
        """Test that pgvector extension is enabled."""
        async with connected_store.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_extension WHERE extname = 'vector'
                )
            """)
            assert result, "pgvector extension is not enabled"

    async def test_vector_index_exists(self, connected_store, settings):
        """Test that vector index exists on chunks table."""
        async with connected_store.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE tablename = $1
                    AND indexname = 'chunks_embedding_idx'
                )
            """, settings.postgres_table_chunks)
            assert result, "Vector index 'chunks_embedding_idx' not found on chunks table"

    async def test_text_search_index_exists(self, connected_store, settings):
        """Test that text search index exists on chunks table."""
        async with connected_store.pool.acquire() as conn:
            result = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM pg_indexes
                    WHERE tablename = $1
                    AND indexname = 'chunks_content_tsv_idx'
                )
            """, settings.postgres_table_chunks)
            assert result, "Text search index 'chunks_content_tsv_idx' not found on chunks table"

    async def test_get_document_count(self, connected_store):
        """Test getting document count."""
        count = await connected_store.get_document_count()
        assert isinstance(count, int)
        assert count >= 0

    async def test_get_chunk_count(self, connected_store):
        """Test getting chunk count."""
        count = await connected_store.get_chunk_count()
        assert isinstance(count, int)
        assert count >= 0


@pytest.mark.asyncio
class TestPostgresStoreOperations:
    """Test PostgreSQL store CRUD operations."""

    @pytest_asyncio.fixture
    async def connected_store(self):
        """Create and initialize a connected PostgresHybridStore."""
        store = PostgresHybridStore()
        await store.initialize()
        yield store
        await store.close()

    async def test_save_and_get_document(self, connected_store):
        """Test saving and retrieving a document."""
        # Save document
        doc_id = await connected_store.save_document(
            title="Test Document",
            source="test_doc.txt",
            content="This is test content for PostgreSQL store.",
            metadata={"test": True, "type": "test"}
        )
        assert doc_id is not None
        assert len(doc_id) > 0

        # Retrieve document
        doc = await connected_store.get_document_by_source("test_doc.txt")
        assert doc is not None
        assert doc["title"] == "Test Document"
        assert doc["content"] == "This is test content for PostgreSQL store."
        assert doc["metadata"]["test"] is True

        # Cleanup
        deleted = await connected_store.delete_document_and_chunks("test_doc.txt")
        assert deleted is True

        # Verify deletion
        doc = await connected_store.get_document_by_source("test_doc.txt")
        assert doc is None

    async def test_get_all_document_sources(self, connected_store):
        """Test getting all document sources."""
        sources = await connected_store.get_all_document_sources()
        assert isinstance(sources, list)

    async def test_get_document_hash(self, connected_store):
        """Test getting document hash."""
        # Save document with hash
        doc_id = await connected_store.save_document(
            title="Hash Test Doc",
            source="hash_test.txt",
            content="Content for hash test",
            metadata={"content_hash": "abc123hash"}
        )

        # Get hash
        hash_value = await connected_store.get_document_hash("hash_test.txt")
        assert hash_value == "abc123hash"

        # Cleanup
        await connected_store.delete_document_and_chunks("hash_test.txt")


@pytest.mark.asyncio
class TestPostgresSearchOperations:
    """Test PostgreSQL search operations - requires data in database."""

    @pytest_asyncio.fixture
    async def connected_store(self):
        """Create and initialize a connected PostgresHybridStore."""
        store = PostgresHybridStore()
        await store.initialize()
        yield store
        await store.close()

    async def test_semantic_search_empty_results(self, connected_store):
        """Test semantic search returns empty list when no data matches."""
        # Create a dummy embedding (768 dimensions)
        dummy_embedding = [0.0] * 768

        results = await connected_store.semantic_search(dummy_embedding, 5)
        assert isinstance(results, list)

    async def test_text_search_empty_results(self, connected_store):
        """Test text search returns empty list when no data matches."""
        results = await connected_store.text_search("xyznonexistentquery123", 5)
        assert isinstance(results, list)

    async def test_hybrid_search_empty_results(self, connected_store):
        """Test hybrid search returns empty list when no data matches."""
        dummy_embedding = [0.0] * 768

        results = await connected_store.hybrid_search(
            "xyznonexistentquery123",
            dummy_embedding,
            5
        )
        assert isinstance(results, list)


@pytest.mark.asyncio
class TestEmbeddingDimensionValidation:
    """Test embedding dimension validation."""

    @pytest.fixture
    def settings(self):
        """Load settings fixture."""
        return load_settings()

    async def test_embedding_dimension_is_positive(self, settings):
        """Test that embedding dimension is a positive integer."""
        assert settings.embedding_dimension > 0
        assert isinstance(settings.embedding_dimension, int)

    async def test_embedding_dimension_matches_model(self, settings):
        """Test that embedding dimension matches the configured model."""
        # Known dimensions for common models
        known_dimensions = {
            "nomic-embed-text": 768,
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }

        model_name = settings.embedding_model.lower()

        for model_key, expected_dim in known_dimensions.items():
            if model_key in model_name:
                assert settings.embedding_dimension == expected_dim, (
                    f"Model '{settings.embedding_model}' should have "
                    f"dimension {expected_dim}, but got {settings.embedding_dimension}"
                )
                break
