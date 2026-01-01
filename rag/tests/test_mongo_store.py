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

"""Tests for MongoDB vector store and index verification."""

import pytest
import pytest_asyncio

from rag.config.settings import load_settings
from rag.storage.vector_store.mongo import MongoHybridStore


class TestMongoDBConnection:
    """Test MongoDB connection and basic operations."""

    @pytest.fixture
    def settings(self):
        """Load settings fixture."""
        return load_settings()

    @pytest.fixture
    def store(self):
        """Create MongoHybridStore fixture."""
        return MongoHybridStore()

    def test_store_initialization(self, store):
        """Test that store initializes with correct settings."""
        assert store.settings is not None
        assert store.client is None  # Not connected yet
        assert store._initialized is False

    def test_settings_have_index_names(self, settings):
        """Test that settings have vector and text index names."""
        assert settings.mongodb_vector_index is not None
        assert settings.mongodb_text_index is not None
        assert len(settings.mongodb_vector_index) > 0
        assert len(settings.mongodb_text_index) > 0


@pytest.mark.asyncio
class TestMongoDBIndexVerification:
    """Test MongoDB index verification - requires live MongoDB connection."""

    @pytest.fixture
    def settings(self):
        """Load settings fixture."""
        return load_settings()

    @pytest_asyncio.fixture
    async def connected_store(self):
        """Create and initialize a connected MongoHybridStore."""
        store = MongoHybridStore()
        await store.initialize()
        yield store
        await store.close()

    async def test_mongodb_connection(self, connected_store):
        """Test that MongoDB connection is established."""
        assert connected_store._initialized is True
        assert connected_store.client is not None
        assert connected_store.db is not None

    async def test_vector_index_exists(self, connected_store, settings):
        """Test that vector search index exists on chunks collection."""
        collection = connected_store.db[settings.mongodb_collection_chunks]

        # List all search indexes on the collection
        # list_search_indexes() returns a coroutine that yields a cursor
        cursor = await collection.list_search_indexes()
        indexes = await cursor.to_list()

        # Find vector index
        vector_index_names = [idx.get("name") for idx in indexes]

        assert settings.mongodb_vector_index in vector_index_names, (
            f"Vector index '{settings.mongodb_vector_index}' not found. "
            f"Available indexes: {vector_index_names}. "
            "Please create the vector index in MongoDB Atlas UI."
        )

    async def test_vector_index_configuration(self, connected_store, settings):
        """Test that vector index has correct configuration."""
        collection = connected_store.db[settings.mongodb_collection_chunks]

        # Find the vector index
        cursor = await collection.list_search_indexes()
        indexes = await cursor.to_list()

        vector_index = None
        for index in indexes:
            if index.get("name") == settings.mongodb_vector_index:
                vector_index = index
                break

        if vector_index is None:
            pytest.skip(
                f"Vector index '{settings.mongodb_vector_index}' not found. "
                "Create it in MongoDB Atlas UI first."
            )

        # Check index definition
        definition = vector_index.get("latestDefinition", {})

        # Verify it's configured for vector search
        fields = definition.get("fields", [])
        embedding_field = None
        for field in fields:
            if field.get("path") == "embedding":
                embedding_field = field
                break

        assert embedding_field is not None, (
            "Vector index should have 'embedding' field configured"
        )

        # Verify embedding dimension matches settings
        if "numDimensions" in embedding_field:
            assert embedding_field["numDimensions"] == settings.embedding_dimension, (
                f"Vector index dimension ({embedding_field['numDimensions']}) "
                f"does not match settings ({settings.embedding_dimension})"
            )

        # Verify similarity metric (typically cosine or dotProduct)
        if "similarity" in embedding_field:
            assert embedding_field["similarity"] in [
                "cosine",
                "dotProduct",
                "euclidean",
            ], f"Unexpected similarity metric: {embedding_field['similarity']}"

    async def test_text_index_exists(self, connected_store, settings):
        """Test that text search index exists on chunks collection."""
        collection = connected_store.db[settings.mongodb_collection_chunks]

        # List all search indexes on the collection
        cursor = await collection.list_search_indexes()
        indexes = await cursor.to_list()

        # Find text index
        text_index_names = [idx.get("name") for idx in indexes]

        assert settings.mongodb_text_index in text_index_names, (
            f"Text index '{settings.mongodb_text_index}' not found. "
            f"Available indexes: {text_index_names}. "
            "Please create the text index in MongoDB Atlas UI."
        )

    async def test_collections_exist(self, connected_store, settings):
        """Test that required collections exist (skip if empty database)."""
        collection_names = await connected_store.db.list_collection_names()

        # Skip if database is empty (no ingestion has been run yet)
        if not collection_names:
            pytest.skip(
                "No collections found. Run document ingestion first to create collections."
            )

        assert settings.mongodb_collection_chunks in collection_names, (
            f"Chunks collection '{settings.mongodb_collection_chunks}' not found. "
            f"Available collections: {collection_names}"
        )
        assert settings.mongodb_collection_documents in collection_names, (
            f"Documents collection '{settings.mongodb_collection_documents}' not found. "
            f"Available collections: {collection_names}"
        )


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
