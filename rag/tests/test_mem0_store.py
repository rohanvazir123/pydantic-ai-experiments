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

"""Tests for Mem0 Store with PostgreSQL/pgvector backend.

These tests verify:
1. Mem0Store initialization and configuration
2. Database URL parsing
3. Memory CRUD operations
4. Search and context retrieval

Requirements:
    - PostgreSQL/Neon with pgvector extension
    - DATABASE_URL in .env
    - Ollama running for LLM and embeddings
    - MEM0_ENABLED=true in .env
"""

import logging

import pytest

from rag.config.settings import load_settings
from rag.memory.mem0_store import Mem0Store, create_mem0_store

logger = logging.getLogger(__name__)


class TestMem0StoreBasic:
    """Test basic Mem0Store initialization."""

    @pytest.fixture
    def settings(self):
        """Load settings fixture."""
        return load_settings()

    @pytest.fixture
    def store(self):
        """Create Mem0Store fixture."""
        return Mem0Store()

    def test_store_initialization(self, store):
        """Test that store initializes with correct settings."""
        assert store.settings is not None
        assert store._memory is None  # Not initialized until first use
        assert store._initialized is False

    def test_store_has_settings(self, store):
        """Test that store has access to settings."""
        assert store.settings.llm_model is not None
        assert store.settings.embedding_model is not None
        assert store.settings.embedding_dimension > 0

    def test_create_mem0_store_factory(self):
        """Test factory function creates store."""
        store = create_mem0_store()
        assert isinstance(store, Mem0Store)
        assert store.settings is not None


class TestMem0StoreDatabaseParsing:
    """Test DATABASE_URL parsing for pgvector."""

    @pytest.fixture
    def store(self):
        """Create Mem0Store fixture."""
        return Mem0Store()

    def test_parse_database_url_basic(self, store):
        """Test parsing a basic PostgreSQL URL."""
        # Temporarily set database_url
        original = store.settings.database_url
        store.settings.database_url = "postgresql://user:pass@localhost:5432/testdb"

        config = store._parse_database_url()

        assert config["user"] == "user"
        assert config["password"] == "pass"
        assert config["host"] == "localhost"
        assert config["port"] == 5432
        assert config["dbname"] == "testdb"

        store.settings.database_url = original

    def test_parse_database_url_with_sslmode(self, store):
        """Test parsing URL with query parameters."""
        original = store.settings.database_url
        store.settings.database_url = (
            "postgresql://user:pass@host.neon.tech:5432/neondb?sslmode=require"
        )

        config = store._parse_database_url()

        assert config["host"] == "host.neon.tech"
        assert config["dbname"] == "neondb"

        store.settings.database_url = original

    def test_parse_database_url_missing_raises_error(self, store):
        """Test that missing URL raises ValueError."""
        original = store.settings.database_url
        store.settings.database_url = ""

        with pytest.raises(ValueError, match="DATABASE_URL not configured"):
            store._parse_database_url()

        store.settings.database_url = original


class TestMem0StoreEnabled:
    """Test Mem0 enabled/disabled behavior."""

    def test_is_enabled_returns_setting(self):
        """Test is_enabled reflects settings."""
        store = Mem0Store()
        assert store.is_enabled() == store.settings.mem0_enabled

    def test_disabled_add_returns_empty(self):
        """Test add returns empty when disabled."""
        store = Mem0Store()
        # Temporarily disable
        original = store.settings.mem0_enabled
        store.settings.mem0_enabled = False

        result = store.add("test", user_id="test_user")
        assert result == {"results": []}

        store.settings.mem0_enabled = original

    def test_disabled_search_returns_empty(self):
        """Test search returns empty when disabled."""
        store = Mem0Store()
        original = store.settings.mem0_enabled
        store.settings.mem0_enabled = False

        result = store.search("query", user_id="test_user")
        assert result == []

        store.settings.mem0_enabled = original

    def test_disabled_get_all_returns_empty(self):
        """Test get_all returns empty when disabled."""
        store = Mem0Store()
        original = store.settings.mem0_enabled
        store.settings.mem0_enabled = False

        result = store.get_all(user_id="test_user")
        assert result == []

        store.settings.mem0_enabled = original

    def test_disabled_get_context_returns_empty(self):
        """Test get_context_string returns empty when disabled."""
        store = Mem0Store()
        original = store.settings.mem0_enabled
        store.settings.mem0_enabled = False

        result = store.get_context_string("query", user_id="test_user")
        assert result == ""

        store.settings.mem0_enabled = original


@pytest.mark.skipif(
    not load_settings().mem0_enabled,
    reason="Mem0 is disabled (MEM0_ENABLED != true)",
)
class TestMem0StoreIntegration:
    """Integration tests for Mem0Store with PostgreSQL.

    These tests require:
    - MEM0_ENABLED=true in .env
    - PostgreSQL/Neon with pgvector
    - Ollama running
    """

    TEST_USER_ID = "test_mem0_user_12345"

    @pytest.fixture
    def store(self):
        """Create Mem0Store fixture with cleanup."""
        store = Mem0Store()
        yield store
        # Cleanup test data
        try:
            store.delete_all(user_id=self.TEST_USER_ID)
        except Exception:
            pass

    def test_add_memory(self, store):
        """Test adding a memory."""
        result = store.add(
            "I am a software engineer working on AI projects",
            user_id=self.TEST_USER_ID,
            metadata={"source": "test"},
        )

        assert "results" in result or isinstance(result, dict)
        logger.info(f"Add result: {result}")

    def test_get_all_memories(self, store):
        """Test retrieving all memories for a user."""
        # First add some memories
        store.add(
            "I prefer Python over JavaScript",
            user_id=self.TEST_USER_ID,
        )

        memories = store.get_all(user_id=self.TEST_USER_ID)

        assert isinstance(memories, list)
        logger.info(f"Found {len(memories)} memories")

    def test_search_memories(self, store):
        """Test searching memories."""
        # Add a memory
        store.add(
            "I work on machine learning models",
            user_id=self.TEST_USER_ID,
        )

        # Search for related content
        results = store.search(
            "What does the user work on?",
            user_id=self.TEST_USER_ID,
            limit=3,
        )

        assert isinstance(results, list)
        logger.info(f"Search returned {len(results)} results")

    def test_get_context_string(self, store):
        """Test getting formatted context string."""
        # Add some memories
        store.add(
            "My favorite framework is FastAPI",
            user_id=self.TEST_USER_ID,
        )

        context = store.get_context_string(
            "web frameworks",
            user_id=self.TEST_USER_ID,
            limit=3,
        )

        assert isinstance(context, str)
        if context:
            assert "User Context" in context
        logger.info(f"Context: {context[:100]}..." if context else "Empty context")

    def test_delete_all_memories(self, store):
        """Test deleting all memories for a user."""
        # Add memories
        store.add(
            "Test memory to delete",
            user_id=self.TEST_USER_ID,
        )

        # Delete all
        store.delete_all(user_id=self.TEST_USER_ID)

        # Verify empty
        memories = store.get_all(user_id=self.TEST_USER_ID)
        assert len(memories) == 0


class TestMem0StoreContextFormatting:
    """Test context string formatting."""

    def test_empty_memories_returns_empty_string(self):
        """Test that no memories returns empty context."""
        store = Mem0Store()
        original_enabled = store.settings.mem0_enabled
        store.settings.mem0_enabled = False

        context = store.get_context_string("test", user_id="nobody")
        assert context == ""

        store.settings.mem0_enabled = original_enabled

    def test_history_disabled_returns_empty(self):
        """Test history returns empty when disabled."""
        store = Mem0Store()
        original = store.settings.mem0_enabled
        store.settings.mem0_enabled = False

        result = store.history("some_id")
        assert result == []

        store.settings.mem0_enabled = original
