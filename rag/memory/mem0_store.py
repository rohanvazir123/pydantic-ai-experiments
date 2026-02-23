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

"""
Mem0 memory store for persistent user context.

Module: rag.memory.mem0_store
=============================

This module provides a wrapper around Mem0 for storing and retrieving
user-specific memories to enable personalization in the RAG system.

Uses PostgreSQL with pgvector for vector storage (same as main RAG system).

Classes
-------
Mem0Store
    Wrapper for Mem0 Memory with RAG-specific configuration.

    Methods:
        __init__(settings: Settings | None = None)
            Initialize with optional settings override.

        add(text: str, user_id: str, metadata: dict | None = None) -> dict
            Add a memory for a user.

        search(query: str, user_id: str, limit: int = 5) -> list[dict]
            Search memories relevant to a query.

        get_all(user_id: str) -> list[dict]
            Get all memories for a user.

        delete(memory_id: str) -> None
            Delete a specific memory by ID.

        delete_all(user_id: str) -> None
            Delete all memories for a user.

        get_context_string(query: str, user_id: str, limit: int = 3) -> str
            Get formatted memory context for LLM prompt.

        is_enabled() -> bool
            Check if Mem0 is enabled in settings.

Functions
---------
create_mem0_store(settings: Settings | None = None) -> Mem0Store
    Factory function to create Mem0Store instance.

Usage
-----
    from rag.memory.mem0_store import Mem0Store, create_mem0_store

    # Create store
    store = create_mem0_store()

    # Add memory
    store.add(
        "User prefers concise answers",
        user_id="john_doe",
        metadata={"source": "preference"}
    )

    # Search memories
    memories = store.search("communication style", user_id="john_doe")

    # Get formatted context for LLM
    context = store.get_context_string("What is the PTO policy?", user_id="john_doe")

    # Delete all user memories
    store.delete_all(user_id="john_doe")
"""

import logging
from typing import Any

from rag.config.settings import Settings, load_settings

logger = logging.getLogger(__name__)


class Mem0Store:
    """
    Wrapper for Mem0 Memory with RAG-specific configuration.

    Uses Ollama for LLM and embeddings by default (same as RAG system).
    Memories are stored in PostgreSQL using pgvector (same as main RAG system).
    """

    def __init__(self, settings: Settings | None = None):
        """
        Initialize Mem0 store.

        Args:
            settings: Optional settings override (uses load_settings() if None)
        """
        self.settings = settings or load_settings()
        self._memory = None
        self._initialized = False

    def _parse_database_url(self) -> dict:
        """Parse DATABASE_URL into connection parameters for pgvector."""
        from urllib.parse import parse_qs, urlparse

        url = self.settings.database_url
        if not url:
            raise ValueError("DATABASE_URL not configured in settings")

        parsed = urlparse(url)

        # Extract connection parameters
        config = {
            "user": parsed.username or "",
            "password": parsed.password or "",
            "host": parsed.hostname or "localhost",
            "port": parsed.port or 5432,
        }

        # Get database name from path (remove leading /)
        dbname = parsed.path.lstrip("/") if parsed.path else ""

        # Handle query parameters (e.g., sslmode)
        query_params = parse_qs(parsed.query)

        # For Neon, handle the endpoint parameter
        if "options" in query_params:
            options = query_params["options"][0]
            config["options"] = options

        return {
            "dbname": dbname,
            **config,
        }

    def _get_memory(self):
        """Lazy initialize Mem0 Memory instance."""
        if self._memory is None:
            from mem0 import Memory

            # Get Ollama base URL without /v1 suffix
            ollama_base = (
                self.settings.llm_base_url.replace("/v1", "")
                if self.settings.llm_base_url
                else "http://localhost:11434"
            )

            # Parse database URL for pgvector config
            db_config = self._parse_database_url()

            # Configure Mem0 to use same LLM/embedder as RAG
            # and PostgreSQL with pgvector for vector store (same as RAG)
            config = {
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": self.settings.llm_model,
                        "ollama_base_url": ollama_base,
                    },
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": self.settings.embedding_model,
                        "ollama_base_url": ollama_base,
                    },
                },
                "vector_store": {
                    "provider": "pgvector",
                    "config": {
                        "collection_name": self.settings.mem0_collection_name,
                        "embedding_model_dims": self.settings.embedding_dimension,
                        "dbname": db_config["dbname"],
                        "user": db_config["user"],
                        "password": db_config["password"],
                        "host": db_config["host"],
                        "port": db_config["port"],
                    },
                },
                "version": "v1.1",
            }

            self._memory = Memory.from_config(config)
            self._initialized = True
            logger.info(
                f"Mem0 initialized with PostgreSQL/pgvector table: {self.settings.mem0_collection_name}"
            )

        return self._memory

    def is_enabled(self) -> bool:
        """Check if Mem0 is enabled in settings."""
        return self.settings.mem0_enabled

    def add(
        self,
        text: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
    ) -> dict:
        """
        Add a memory for a user.

        Args:
            text: The memory content to store
            user_id: User identifier for scoping the memory
            metadata: Optional metadata to store with the memory
            infer: If True, LLM extracts facts; if False, stores raw text

        Returns:
            Result dict with memory IDs and events
        """
        if not self.is_enabled():
            logger.debug("Mem0 disabled, skipping add")
            return {"results": []}

        memory = self._get_memory()
        result = memory.add(text, user_id=user_id, metadata=metadata, infer=infer)
        logger.info(
            f"Added memory for user {user_id}: {len(result.get('results', []))} items"
        )
        return result

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[dict]:
        """
        Search memories relevant to a query.

        Args:
            query: Search query
            user_id: User identifier to scope search
            limit: Maximum number of results

        Returns:
            List of memory dicts with 'id', 'memory', 'score' keys
        """
        if not self.is_enabled():
            logger.debug("Mem0 disabled, skipping search")
            return []

        memory = self._get_memory()
        results = memory.search(query, user_id=user_id, limit=limit)

        # Handle both dict and list returns from mem0
        if isinstance(results, dict):
            results = results.get("results", [])

        logger.info(f"Found {len(results)} memories for query: {query[:50]}...")
        return results

    def get_all(self, user_id: str) -> list[dict]:
        """
        Get all memories for a user.

        Args:
            user_id: User identifier

        Returns:
            List of all memory dicts for the user
        """
        if not self.is_enabled():
            return []

        memory = self._get_memory()
        results = memory.get_all(user_id=user_id)

        # Handle both dict and list returns
        if isinstance(results, dict):
            results = results.get("results", [])

        return results

    def delete(self, memory_id: str) -> None:
        """
        Delete a specific memory by ID.

        Args:
            memory_id: The memory ID to delete
        """
        if not self.is_enabled():
            return

        memory = self._get_memory()
        memory.delete(memory_id)
        logger.info(f"Deleted memory: {memory_id}")

    def delete_all(self, user_id: str) -> None:
        """
        Delete all memories for a user.

        Args:
            user_id: User identifier
        """
        if not self.is_enabled():
            return

        memory = self._get_memory()
        memory.delete_all(user_id=user_id)
        logger.info(f"Deleted all memories for user: {user_id}")

    def get_context_string(
        self,
        query: str,
        user_id: str,
        limit: int = 3,
    ) -> str:
        """
        Get formatted memory context for LLM prompt.

        Args:
            query: The current query to find relevant memories for
            user_id: User identifier
            limit: Maximum memories to include

        Returns:
            Formatted string with user context, or empty string if none
        """
        if not self.is_enabled():
            return ""

        memories = self.search(query, user_id=user_id, limit=limit)

        if not memories:
            return ""

        lines = ["User Context (from memory):"]
        for mem in memories:
            memory_text = mem.get("memory", "")
            if memory_text:
                lines.append(f"- {memory_text}")

        return "\n".join(lines)

    def history(self, memory_id: str) -> list[dict]:
        """
        Get the history of changes for a memory.

        Args:
            memory_id: The memory ID

        Returns:
            List of history entries
        """
        if not self.is_enabled():
            return []

        memory = self._get_memory()
        return memory.history(memory_id)


def create_mem0_store(settings: Settings | None = None) -> Mem0Store:
    """
    Factory function to create Mem0Store.

    Args:
        settings: Optional settings override

    Returns:
        Mem0Store instance
    """
    return Mem0Store(settings=settings)


if __name__ == "__main__":
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _logger = logging.getLogger(__name__)

    _logger.info("=" * 60)
    _logger.info("RAG Mem0 Store Module Test (PostgreSQL/pgvector)")
    _logger.info("=" * 60)

    # Check if enabled
    settings = load_settings()
    _logger.info(f"\nMem0 Enabled: {settings.mem0_enabled}")
    _logger.info(f"Table Name: {settings.mem0_collection_name}")
    _logger.info(f"Database URL: {settings.database_url[:30]}..." if settings.database_url else "Not set")
    _logger.info(f"LLM Model: {settings.llm_model}")
    _logger.info(f"Embedding Model: {settings.embedding_model}")

    if not settings.mem0_enabled:
        _logger.warning("\nMem0 is disabled. Set MEM0_ENABLED=true in .env to test.")
        _logger.info("Exiting without running tests.")
    else:
        # Create store
        store = create_mem0_store()
        _logger.info("\n[Mem0Store Created]")

        test_user = "test_user_123"

        # Clean up any existing test data
        _logger.info("\n--- Cleanup ---")
        store.delete_all(user_id=test_user)
        _logger.info(f"  Deleted existing memories for {test_user}")

        # Test adding memories
        _logger.info("\n--- Add Memories ---")
        result1 = store.add(
            "I am a senior engineer working on the ML team",
            user_id=test_user,
            metadata={"source": "user_intro"},
        )
        _logger.info(f"  Added intro: {result1}")

        result2 = store.add(
            "I prefer concise, technical answers without too much explanation",
            user_id=test_user,
            metadata={"source": "preference"},
        )
        _logger.info(f"  Added preference: {result2}")

        result3 = store.add(
            "Last time I asked about the PTO policy",
            user_id=test_user,
            metadata={"source": "history"},
        )
        _logger.info(f"  Added history: {result3}")

        # Test get_all
        _logger.info("\n--- Get All Memories ---")
        all_memories = store.get_all(user_id=test_user)
        _logger.info(f"  Total memories: {len(all_memories)}")
        for mem in all_memories:
            _logger.info(f"    - {mem.get('memory', mem)[:60]}...")

        # Test search
        _logger.info("\n--- Search Memories ---")
        search_queries = [
            "What team does the user work on?",
            "How should I format my answers?",
            "What did we discuss before?",
        ]
        for query in search_queries:
            results = store.search(query, user_id=test_user, limit=2)
            _logger.info(f"  Query: '{query}'")
            for r in results:
                score = r.get("score", "N/A")
                memory = r.get("memory", str(r))[:50]
                _logger.info(f"    [{score}] {memory}...")

        # Test get_context_string
        _logger.info("\n--- Context String ---")
        context = store.get_context_string(
            "What benefits do I get?",
            user_id=test_user,
            limit=3,
        )
        _logger.info(f"  Context for LLM:\n{context}")

        # Cleanup
        _logger.info("\n--- Final Cleanup ---")
        store.delete_all(user_id=test_user)
        _logger.info(f"  Cleaned up test user: {test_user}")

    _logger.info("\n" + "=" * 60)
    _logger.info("Mem0 store test completed!")
    _logger.info("=" * 60)
