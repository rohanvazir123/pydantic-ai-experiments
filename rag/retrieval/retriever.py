"""
Retrieval orchestrator for RAG system.

Module: rag.retrieval.retriever
===============================

This module provides the high-level retrieval interface for the RAG system.
It coordinates embedding generation and search operations with result caching.

Classes
-------
ResultCache
    LRU cache for search results with TTL expiration.

    Methods:
        __init__(max_size: int = 100, ttl_seconds: int = 300)
            Initialize cache with size limit and TTL.

        get(query, search_type, match_count) -> list[SearchResult] | None
            Get cached results if available and not expired.

        set(query, search_type, match_count, results) -> None
            Cache search results.

        stats() -> dict
            Return cache statistics.

        clear() -> None
            Clear all cached results.

Retriever
    Main retrieval interface orchestrating embeddings and search.

    Methods:
        __init__(store: MongoHybridStore | None, embedder: EmbeddingGenerator | None)
            Initialize with optional store and embedder (creates defaults).

        async retrieve(
            query: str,
            match_count: int | None = None,
            search_type: str = "hybrid",
            use_cache: bool = True
        ) -> list[SearchResult]
            Retrieve documents matching query with caching.

        async retrieve_as_context(
            query: str,
            match_count: int | None = None,
            search_type: str = "hybrid"
        ) -> str
            Retrieve and format results as LLM context string.

        get_cache_stats() -> dict (static)
            Return result cache statistics.

        clear_cache() -> None (static)
            Clear the result cache.

        async close() -> None
            Close store connection.

Module Attributes
-----------------
_result_cache: ResultCache
    Global result cache instance (shared across Retriever instances).

Search Types
------------
- "hybrid": Combined vector + text search with RRF (default)
- "semantic": Pure vector similarity search
- "text": Full-text keyword search

Usage
-----
    from rag.retrieval.retriever import Retriever
    from rag.storage.vector_store.mongo import MongoHybridStore

    # Create retriever
    store = MongoHybridStore()
    retriever = Retriever(store=store)

    # Retrieve documents
    results = await retriever.retrieve(
        query="What is RAG?",
        match_count=5,
        search_type="hybrid"
    )

    # Get formatted context for LLM
    context = await retriever.retrieve_as_context("employee benefits")

    # Check cache stats
    print(Retriever.get_cache_stats())

    # Cleanup
    await retriever.close()
"""

import hashlib
import logging
import time
from collections import OrderedDict

from rag.config.settings import load_settings
from rag.ingestion.embedder import EmbeddingGenerator
from rag.ingestion.models import SearchResult
from rag.storage.vector_store.mongo import MongoHybridStore

logger = logging.getLogger(__name__)


class ResultCache:
    """
    LRU cache for search results.

    Caches (query, search_type, match_count) -> search results to avoid
    redundant database searches for repeated queries.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        Initialize result cache.

        Args:
            max_size: Maximum number of results to cache (default: 100)
            ttl_seconds: Time-to-live in seconds (default: 300 = 5 minutes)
        """
        self._cache: OrderedDict[str, tuple[float, list[SearchResult]]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _get_key(self, query: str, search_type: str, match_count: int) -> str:
        """Generate cache key from query parameters."""
        key_str = f"{query}:{search_type}:{match_count}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:24]

    def get(
        self, query: str, search_type: str, match_count: int
    ) -> list[SearchResult] | None:
        """Get cached results if available and not expired."""
        key = self._get_key(query, search_type, match_count)
        if key in self._cache:
            timestamp, results = self._cache[key]
            # Check TTL
            if time.time() - timestamp < self._ttl:
                self._cache.move_to_end(key)
                self._hits += 1
                return results
            else:
                # Expired, remove from cache
                del self._cache[key]
        self._misses += 1
        return None

    def set(
        self,
        query: str,
        search_type: str,
        match_count: int,
        results: list[SearchResult],
    ) -> None:
        """Cache search results."""
        key = self._get_key(query, search_type, match_count)
        self._cache[key] = (time.time(), results)
        self._cache.move_to_end(key)

        # Evict oldest if over limit
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# Global result cache (shared across instances)
_result_cache = ResultCache(max_size=100, ttl_seconds=300)


class Retriever:
    """Orchestrates embedding and retrieval operations."""

    def __init__(
        self,
        store: MongoHybridStore | None = None,
        embedder: EmbeddingGenerator | None = None,
    ):
        """
        Initialize retriever.

        Args:
            store: Vector store instance (creates MongoHybridStore if not provided)
            embedder: Embedding generator (creates EmbeddingGenerator if not provided)
        """
        self.settings = load_settings()
        self.store = store or MongoHybridStore()
        self.embedder = embedder or EmbeddingGenerator()

    async def retrieve(
        self,
        query: str,
        match_count: int | None = None,
        search_type: str = "hybrid",
        use_cache: bool = True,
    ) -> list[SearchResult]:
        """
        Retrieve relevant documents for a query with optional caching.

        Args:
            query: Search query text
            match_count: Number of results to return (defaults to settings)
            search_type: Type of search - "semantic", "text", or "hybrid" (default)
            use_cache: Whether to use result cache (default: True)

        Returns:
            List of search results ordered by relevance
        """
        if match_count is None:
            match_count = self.settings.default_match_count

        # Check result cache first
        if use_cache:
            cached_results = _result_cache.get(query, search_type, match_count)
            if cached_results is not None:
                logger.info(
                    f"[CACHE HIT] Search results from cache ({len(cached_results)} results)"
                )
                return cached_results

        start_time = time.time()
        logger.info(
            f"[CACHE MISS] Retrieving for query: '{query}', type: {search_type}, count: {match_count}"
        )

        # Generate query embedding (uses its own cache)
        query_embedding = await self.embedder.embed_query(query)

        # Perform search based on type
        if search_type == "semantic":
            results = await self.store.semantic_search(query_embedding, match_count)
        elif search_type == "text":
            results = await self.store.text_search(query, match_count)
        else:  # hybrid (default)
            results = await self.store.hybrid_search(
                query, query_embedding, match_count
            )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Retrieved {len(results)} results in {elapsed_ms:.0f}ms")

        # Cache results
        if use_cache:
            _result_cache.set(query, search_type, match_count, results)

        return results

    @staticmethod
    def get_cache_stats() -> dict:
        """Get result cache statistics."""
        return _result_cache.stats()

    @staticmethod
    def clear_cache() -> None:
        """Clear the result cache."""
        _result_cache.clear()
        logger.info("Result cache cleared")

    async def retrieve_as_context(
        self, query: str, match_count: int | None = None, search_type: str = "hybrid"
    ) -> str:
        """
        Retrieve and format results as context string for LLM.

        Args:
            query: Search query text
            match_count: Number of results to return
            search_type: Type of search

        Returns:
            Formatted context string
        """
        results = await self.retrieve(query, match_count, search_type)

        if not results:
            return "No relevant information found in the knowledge base."

        # Format results
        response_parts = [f"Found {len(results)} relevant documents:\n"]

        for i, result in enumerate(results, 1):
            response_parts.append(
                f"\n--- Document {i}: {result.document_title} "
                f"(relevance: {result.similarity:.2f}) ---"
            )
            response_parts.append(result.content)

        return "\n".join(response_parts)

    async def close(self) -> None:
        """Close connections."""
        await self.store.close()


if __name__ == "__main__":
    import asyncio

    async def main():
        print("=" * 60)
        print("RAG Retriever Module Test")
        print("=" * 60)

        # Create retriever
        store = MongoHybridStore()
        retriever = Retriever(store=store)
        print("\n[Retriever Created]")
        print(f"  Default match count: {retriever.settings.default_match_count}")

        # Clear cache for testing
        Retriever.clear_cache()
        EmbeddingGenerator.clear_cache()
        print("  Caches cleared")

        # Test queries
        test_queries = [
            ("What does NeuralFlow AI do?", "hybrid"),
            ("employee benefits", "semantic"),
            ("PTO policy", "text"),
        ]

        for query, search_type in test_queries:
            print(f"\n--- Query: '{query}' ({search_type}) ---")

            # First call (cache miss)
            start = time.time()
            results = await retriever.retrieve(
                query=query,
                match_count=3,
                search_type=search_type,
                use_cache=True,
            )
            first_time = (time.time() - start) * 1000

            print(f"  Results: {len(results)}")
            print(f"  Time (miss): {first_time:.0f}ms")

            for i, r in enumerate(results):
                print(f"    [{i+1}] {r.document_title} (score: {r.similarity:.4f})")

            # Second call (cache hit)
            start = time.time()
            _ = await retriever.retrieve(
                query=query,
                match_count=3,
                search_type=search_type,
                use_cache=True,
            )
            second_time = (time.time() - start) * 1000
            print(f"  Time (hit): {second_time:.0f}ms")

        # Test retrieve_as_context
        print("\n--- Context Retrieval ---")
        context = await retriever.retrieve_as_context(
            query="What is the company mission?",
            match_count=2,
        )
        print(f"  Context length: {len(context)} chars")
        print(f"  Preview: {context[:200]}...")

        # Cache statistics
        print("\n--- Cache Statistics ---")
        result_stats = Retriever.get_cache_stats()
        embed_stats = EmbeddingGenerator.get_cache_stats()
        print(f"  Result cache: {result_stats}")
        print(f"  Embedding cache: {embed_stats}")

        # Cleanup
        await retriever.close()

        print("\n" + "=" * 60)
        print("Retriever test completed successfully!")
        print("=" * 60)

    asyncio.run(main())
