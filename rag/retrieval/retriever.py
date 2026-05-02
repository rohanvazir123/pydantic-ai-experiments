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
Retrieval orchestrator for RAG system.

Module: rag.retrieval.retriever
===============================

This module provides the high-level retrieval interface for the RAG system.
It coordinates embedding generation, optional HyDE query transformation,
search, optional reranking, and result caching.

Classes
-------
ResultCache
    LRU cache for search results with TTL expiration.

Retriever
    Main retrieval interface orchestrating embeddings and search.

    Methods:
        __init__(store, embedder, reranker, hyde)
            Initialize with optional components (all lazy-init from settings).

        async retrieve(
            query: str,
            match_count: int | None = None,
            search_type: str = "hybrid",
            use_cache: bool = True
        ) -> list[SearchResult]
            Retrieve documents. Pipeline:
              1. HyDE (if enabled): generate hypothetical doc, embed it
              2. Over-fetch from DB (if reranking)
              3. Search (semantic / text / hybrid)
              4. Rerank (if enabled) down to match_count
              5. Cache final results

        async retrieve_as_context(query, match_count, search_type) -> str
            Retrieve and format results as LLM context string.

Module Attributes
-----------------
_result_cache: ResultCache
    Global result cache (shared across Retriever instances).

Search Types
------------
- "hybrid": Combined vector + text search with RRF (default)
- "semantic": Pure vector similarity search
- "text": Full-text keyword search

Feature Flags (settings)
------------------------
- hyde_enabled: Use HyDE embedding instead of raw query embedding
- reranker_enabled: Rerank over-fetched results before returning
- reranker_type: "llm" or "cross_encoder"
- reranker_overfetch_factor: How many × match_count to fetch before reranking

Usage
-----
    from rag.retrieval.retriever import Retriever

    retriever = Retriever()
    results = await retriever.retrieve("What is RAG?", match_count=5)
    context = await retriever.retrieve_as_context("employee benefits")
    await retriever.close()
"""

import hashlib
import logging
import time
from collections import OrderedDict

from rag.config.settings import load_settings
from rag.ingestion.embedder import EmbeddingGenerator
from rag.ingestion.models import SearchResult
from rag.retrieval.query_processors import HyDEProcessor
from rag.retrieval.rerankers import BaseReranker, CrossEncoderReranker, LLMReranker
from rag.storage.vector_store.postgres import PostgresHybridStore

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
    """Orchestrates embedding, optional HyDE, search, and optional reranking."""

    def __init__(
        self,
        store: PostgresHybridStore | None = None,
        embedder: EmbeddingGenerator | None = None,
        reranker: BaseReranker | None = None,
        hyde: HyDEProcessor | None = None,
    ):
        """
        Initialize retriever.

        Args:
            store: Vector store (creates PostgresHybridStore if not provided)
            embedder: Embedding generator (creates EmbeddingGenerator if not provided)
            reranker: Optional reranker override (lazy-init from settings if None)
            hyde: Optional HyDE processor override (lazy-init from settings if None)
        """
        self.settings = load_settings()
        self.store = store or PostgresHybridStore()
        self.embedder = embedder or EmbeddingGenerator()
        self._reranker = reranker
        self._hyde = hyde

    def _get_hyde(self) -> HyDEProcessor:
        """Lazy-init HyDE processor from settings."""
        if self._hyde is None:
            self._hyde = HyDEProcessor(
                model=self.settings.llm_model,
                base_url=self.settings.llm_base_url,
                api_key=self.settings.llm_api_key,
                embedding_model=self.settings.embedding_model,
                embedding_base_url=self.settings.embedding_base_url,
            )
        return self._hyde

    def _get_reranker(self) -> BaseReranker:
        """Lazy-init reranker from settings."""
        if self._reranker is None:
            reranker_type = self.settings.reranker_type
            if reranker_type == "cross_encoder":
                model = self.settings.reranker_model or "BAAI/bge-reranker-base"
                self._reranker = CrossEncoderReranker(model_name=model)
            else:  # default: llm
                model = self.settings.reranker_model or self.settings.llm_model
                self._reranker = LLMReranker(
                    model=model,
                    base_url=self.settings.llm_base_url,
                    api_key=self.settings.llm_api_key,
                )
        return self._reranker

    async def retrieve(
        self,
        query: str,
        match_count: int | None = None,
        search_type: str = "hybrid",
        use_cache: bool = True,
    ) -> list[SearchResult]:
        """
        Retrieve relevant documents for a query.

        Pipeline:
          1. Check cache
          2. HyDE (if enabled): generate hypothetical doc, embed it
          3. Over-fetch from DB (if reranking enabled)
          4. Search (semantic / text / hybrid)
          5. Rerank (if enabled) and trim to match_count
          6. Cache and return

        Args:
            query: Search query text
            match_count: Number of results to return (defaults to settings)
            search_type: "semantic", "text", or "hybrid" (default)
            use_cache: Whether to use result cache (default: True)

        Returns:
            List of search results ordered by relevance
        """
        if match_count is None:
            match_count = self.settings.default_match_count

        # 1. Cache check
        if use_cache:
            cached_results = _result_cache.get(query, search_type, match_count)
            if cached_results is not None:
                logger.info(
                    f"[CACHE HIT] {len(cached_results)} results"
                )
                return cached_results

        start_time = time.time()
        logger.info(
            f"[RETRIEVE] query='{query}', type={search_type}, count={match_count}, "
            f"hyde={'on' if self.settings.hyde_enabled else 'off'}, "
            f"rerank={'on' if self.settings.reranker_enabled else 'off'}"
        )

        # 2. Query embedding — use HyDE if enabled
        if self.settings.hyde_enabled:
            hyde = self._get_hyde()
            hypothetical = await hyde.generate_hypothetical(query)
            # Use our existing embedder so caching/settings stay consistent
            query_embedding = await self.embedder.generate_embedding(hypothetical)
            logger.info("[HyDE] Using hypothetical-document embedding")
        else:
            query_embedding = await self.embedder.embed_query(query)

        # 3. Determine fetch count (over-fetch before reranking)
        fetch_count = match_count
        if self.settings.reranker_enabled:
            fetch_count = min(
                match_count * self.settings.reranker_overfetch_factor,
                self.settings.max_match_count,
            )

        # 4. Search
        if search_type == "semantic":
            results = await self.store.semantic_search(query_embedding, fetch_count)
        elif search_type == "text":
            results = await self.store.text_search(query, fetch_count)
        else:  # hybrid (default)
            results = await self.store.hybrid_search(query, query_embedding, fetch_count)

        # 5. Rerank
        if self.settings.reranker_enabled and results:
            reranker = self._get_reranker()
            results = await reranker.rerank(query, results, top_k=match_count)
            logger.info(
                f"[RERANK] {self.settings.reranker_type}: trimmed to {len(results)} results"
            )

        # 6. Relevance threshold guardrail — drop low-confidence chunks
        # Only applies to semantic search: cosine similarity is 0-1 and 0.40 is
        # a meaningful threshold there. RRF scores (~0.016) and ts_rank scores
        # are not calibrated to the same scale.
        threshold = self.settings.min_relevance_score
        if threshold > 0 and results and search_type == "semantic":
            before = len(results)
            results = [r for r in results if r.similarity >= threshold]
            dropped = before - len(results)
            if dropped:
                logger.warning(
                    "[GUARDRAIL] Relevance threshold %.2f dropped %d/%d chunk(s) "
                    "(max kept score: %.2f)",
                    threshold,
                    dropped,
                    before,
                    max((r.similarity for r in results), default=0.0),
                )

        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"[RETRIEVE] Done: {len(results)} results in {elapsed_ms:.0f}ms")

        # 6. Cache final results
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
            return "No relevant information found in the knowledge base for this query."

        # Format results — chunk_id is included so the agent can cite specific sources
        response_parts = [f"Found {len(results)} relevant document(s):\n"]

        for i, result in enumerate(results, 1):
            response_parts.append(
                f"\n--- Source [{result.chunk_id}] {result.document_title} "
                f"(relevance: {result.similarity:.2f}) ---"
            )
            response_parts.append(result.content)

        return "\n".join(response_parts)

    async def close(self) -> None:
        """Close connections."""
        await self.store.close()


if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _logger = logging.getLogger(__name__)

    async def main():
        _logger.info("=" * 60)
        _logger.info("RAG Retriever Module Test")
        _logger.info("=" * 60)

        # Create retriever
        store = PostgresHybridStore()
        retriever = Retriever(store=store)
        _logger.info("[Retriever Created]")
        _logger.info(f"  Default match count: {retriever.settings.default_match_count}")

        # Clear cache for testing
        Retriever.clear_cache()
        EmbeddingGenerator.clear_cache()
        _logger.info("  Caches cleared")

        # Test queries
        test_queries = [
            ("What does NeuralFlow AI do?", "hybrid"),
            ("employee benefits", "semantic"),
            ("PTO policy", "text"),
        ]

        for query, search_type in test_queries:
            _logger.info(f"--- Query: '{query}' ({search_type}) ---")

            # First call (cache miss)
            start = time.time()
            results = await retriever.retrieve(
                query=query,
                match_count=3,
                search_type=search_type,
                use_cache=True,
            )
            first_time = (time.time() - start) * 1000

            _logger.info(f"  Results: {len(results)}")
            _logger.info(f"  Time (miss): {first_time:.0f}ms")

            for i, r in enumerate(results):
                _logger.info(f"    [{i+1}] {r.document_title} (score: {r.similarity:.4f})")

            # Second call (cache hit)
            start = time.time()
            _ = await retriever.retrieve(
                query=query,
                match_count=3,
                search_type=search_type,
                use_cache=True,
            )
            second_time = (time.time() - start) * 1000
            _logger.info(f"  Time (hit): {second_time:.0f}ms")

        # Test retrieve_as_context
        _logger.info("--- Context Retrieval ---")
        context = await retriever.retrieve_as_context(
            query="What is the company mission?",
            match_count=2,
        )
        _logger.info(f"  Context length: {len(context)} chars")
        _logger.info(f"  Preview: {context[:200]}...")

        # Cache statistics
        _logger.info("--- Cache Statistics ---")
        result_stats = Retriever.get_cache_stats()
        embed_stats = EmbeddingGenerator.get_cache_stats()
        _logger.info(f"  Result cache: {result_stats}")
        _logger.info(f"  Embedding cache: {embed_stats}")

        # Cleanup
        await retriever.close()

        _logger.info("=" * 60)
        _logger.info("Retriever test completed successfully!")
        _logger.info("=" * 60)

    asyncio.run(main())
