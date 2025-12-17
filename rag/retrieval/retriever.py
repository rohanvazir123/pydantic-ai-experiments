"""Retrieval orchestrator for RAG system."""

import logging
from typing import List, Optional

from rag.config.settings import load_settings
from rag.ingestion.embedder import EmbeddingGenerator
from rag.ingestion.models import SearchResult
from rag.storage.vector_store.mongo import MongoHybridStore

logger = logging.getLogger(__name__)


class Retriever:
    """Orchestrates embedding and retrieval operations."""

    def __init__(
        self,
        store: Optional[MongoHybridStore] = None,
        embedder: Optional[EmbeddingGenerator] = None
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
        match_count: Optional[int] = None,
        search_type: str = "hybrid"
    ) -> List[SearchResult]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query text
            match_count: Number of results to return (defaults to settings)
            search_type: Type of search - "semantic", "text", or "hybrid" (default)

        Returns:
            List of search results ordered by relevance
        """
        if match_count is None:
            match_count = self.settings.default_match_count

        logger.info(f"Retrieving for query: '{query}', type: {search_type}, count: {match_count}")

        # Generate query embedding
        query_embedding = await self.embedder.embed_query(query)

        # Perform search based on type
        if search_type == "semantic":
            results = await self.store.semantic_search(query_embedding, match_count)
        elif search_type == "text":
            results = await self.store.text_search(query, match_count)
        else:  # hybrid (default)
            results = await self.store.hybrid_search(query, query_embedding, match_count)

        logger.info(f"Retrieved {len(results)} results")
        return results

    async def retrieve_as_context(
        self,
        query: str,
        match_count: Optional[int] = None,
        search_type: str = "hybrid"
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
