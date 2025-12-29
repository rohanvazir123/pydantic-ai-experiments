"""
Reranking implementations for improved retrieval relevance.

Module: rag.retrieval.rerankers
===============================

This module provides different reranking strategies to improve search result
quality by re-scoring initial retrieval results with more sophisticated models.

Classes
-------
BaseReranker (ABC)
    Abstract base class for all rerankers.

    Methods:
        async rerank(query, results, top_k) -> list[SearchResult]
            Rerank search results based on relevance to query.

CrossEncoderReranker(BaseReranker)
    Uses cross-encoder models for semantic reranking.
    Processes query-document pairs together for deep interaction.

    Recommended models:
        - BAAI/bge-reranker-large (best quality)
        - BAAI/bge-reranker-base (good balance)
        - cross-encoder/ms-marco-MiniLM-L-6-v2 (fastest)

    Methods:
        __init__(model_name: str = "BAAI/bge-reranker-base")
        async rerank(query, results, top_k) -> list[SearchResult]

    Requirements:
        pip install sentence-transformers

ColBERTReranker(BaseReranker)
    Uses ColBERT late interaction for token-level matching.
    Note: Simplified implementation - full ColBERT requires pre-indexing.

    Methods:
        __init__(model_name: str = "colbert-ir/colbertv2.0")
        async rerank(query, results, top_k) -> list[SearchResult]

    Requirements:
        pip install colbert-ai sentence-transformers

LLMReranker(BaseReranker)
    Uses LLM to judge relevance by scoring each document.
    Works with any OpenAI-compatible API (Ollama, OpenAI, etc.)

    Methods:
        __init__(model, base_url, api_key)
        async rerank(query, results, top_k) -> list[SearchResult]

Functions
---------
create_reranker(reranker_type: str, **kwargs) -> BaseReranker
    Factory function to create a reranker instance.
    Types: "cross_encoder", "colbert", "llm"

Usage
-----
    from rag.retrieval.rerankers import create_reranker, CrossEncoderReranker

    # Using factory function
    reranker = create_reranker("cross_encoder")
    reranked = await reranker.rerank(query, results, top_k=5)

    # Direct instantiation
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    reranked = await reranker.rerank("What is RAG?", search_results, top_k=3)

    # LLM reranker with Ollama
    llm_reranker = create_reranker(
        "llm",
        model="llama3.1:8b",
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    reranked = await llm_reranker.rerank(query, results, top_k=5)
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from rag.ingestion.models import SearchResult

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base class for rerankers."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Rerank search results based on relevance to query.

        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of search results
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """
    Reranker using cross-encoder models.

    Cross-encoders process query-document pairs together, allowing
    for deep interaction between query and document tokens.

    Recommended models:
    - BAAI/bge-reranker-large (best quality)
    - BAAI/bge-reranker-base (good balance)
    - cross-encoder/ms-marco-MiniLM-L-6-v2 (fastest)
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the cross-encoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore

                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for CrossEncoderReranker. "
                    "Install with: pip install sentence-transformers"
                ) from e
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Rerank results using cross-encoder.

        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of search results with updated similarity scores
        """
        if not results:
            return results

        model = self._load_model()

        # Create query-document pairs
        pairs = [(query, r.content) for r in results]

        # Score with cross-encoder
        scores = model.predict(pairs)

        # Pair results with scores and sort
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Create new results with updated scores
        reranked = []
        for result, score in scored_results[:top_k]:
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    similarity=float(score),  # Replace with reranker score
                    metadata={
                        **result.metadata,
                        "original_score": result.similarity,
                        "reranker": "cross_encoder",
                    },
                    document_title=result.document_title,
                    document_source=result.document_source,
                )
            )

        logger.info(f"Reranked {len(results)} results, returning top {len(reranked)}")
        return reranked


class ColBERTReranker(BaseReranker):
    """
    Reranker using ColBERT late interaction.

    ColBERT computes token-level similarity between query and document,
    enabling fine-grained matching while maintaining efficiency.
    """

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        """
        Initialize ColBERT reranker.

        Args:
            model_name: HuggingFace model name for ColBERT
        """
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        """Lazy load the ColBERT model."""
        if self._model is None:
            try:
                from colbert import Searcher  # type: ignore
                from colbert.infra import ColBERTConfig  # type: ignore

                logger.info(f"Loading ColBERT model: {self.model_name}")
                config = ColBERTConfig(checkpoint=self.model_name)
                self._model = Searcher(config=config)
            except ImportError as e:
                raise ImportError(
                    "colbert-ai is required for ColBERTReranker. "
                    "Install with: pip install colbert-ai"
                ) from e
        return self._model

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Rerank results using ColBERT.

        Note: This is a simplified implementation. Full ColBERT requires
        pre-indexing documents. For ad-hoc reranking, consider using
        the cross-encoder instead.
        """
        if not results:
            return results

        # For simplicity, fall back to cross-encoder behavior
        # Full ColBERT implementation requires document pre-indexing
        logger.warning(
            "ColBERT reranking requires pre-indexed documents. "
            "Using simplified scoring."
        )

        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("colbert-ir/colbertv2.0")

            # Encode query and documents
            query_embedding = model.encode(query)
            doc_embeddings = model.encode([r.content for r in results])

            # Compute similarities
            scores = np.dot(doc_embeddings, query_embedding)

            # Sort by score
            scored_results = list(zip(results, scores))
            scored_results.sort(key=lambda x: x[1], reverse=True)

            reranked = []
            for result, score in scored_results[:top_k]:
                reranked.append(
                    SearchResult(
                        chunk_id=result.chunk_id,
                        document_id=result.document_id,
                        content=result.content,
                        similarity=float(score),
                        metadata={
                            **result.metadata,
                            "original_score": result.similarity,
                            "reranker": "colbert",
                        },
                        document_title=result.document_title,
                        document_source=result.document_source,
                    )
                )

            return reranked

        except ImportError:
            logger.error("sentence-transformers required for ColBERT fallback")
            return results[:top_k]


class LLMReranker(BaseReranker):
    """
    Reranker using LLM to judge relevance.

    This reranker asks the LLM to rate the relevance of each document
    to the query and reorders based on those ratings.
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
    ):
        """
        Initialize LLM reranker.

        Args:
            model: LLM model name
            base_url: API base URL
            api_key: API key
        """
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Get or create the async OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        return self._client

    async def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Rerank results using LLM relevance judgments.

        Args:
            query: The search query
            results: List of search results to rerank
            top_k: Number of top results to return

        Returns:
            Reranked list of search results
        """
        if not results:
            return results

        client = self._get_client()

        # Score each document individually for reliability
        scored_results = []

        for result in results:
            score = await self._score_document(client, query, result.content)
            scored_results.append((result, score))

        # Sort by score
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Create new results with LLM scores
        reranked = []
        for result, score in scored_results[:top_k]:
            reranked.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    document_id=result.document_id,
                    content=result.content,
                    similarity=score,
                    metadata={
                        **result.metadata,
                        "original_score": result.similarity,
                        "reranker": "llm",
                    },
                    document_title=result.document_title,
                    document_source=result.document_source,
                )
            )

        logger.info(
            f"LLM reranked {len(results)} results, returning top {len(reranked)}"
        )
        return reranked

    async def _score_document(self, client: Any, query: str, content: str) -> float:
        """Score a single document's relevance to the query."""
        prompt = f"""Rate how relevant the following document is to the query.
Return ONLY a number from 0 to 10, where:
- 0 = completely irrelevant
- 5 = somewhat relevant
- 10 = highly relevant and directly answers the query

Query: {query}

Document:
{content[:1500]}

Relevance score (0-10):"""

        try:
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )

            score_text = response.choices[0].message.content.strip()
            # Extract numeric score
            import re

            match = re.search(r"(\d+(?:\.\d+)?)", score_text)
            if match:
                score = float(match.group(1))
                return min(max(score / 10.0, 0.0), 1.0)  # Normalize to 0-1
            return 0.5  # Default if parsing fails

        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")
            return 0.5


def create_reranker(
    reranker_type: str = "cross_encoder",
    **kwargs: Any,
) -> BaseReranker:
    """
    Factory function to create a reranker.

    Args:
        reranker_type: Type of reranker (cross_encoder, colbert, llm)
        **kwargs: Additional arguments for the reranker

    Returns:
        Reranker instance
    """
    rerankers = {
        "cross_encoder": CrossEncoderReranker,
        "colbert": ColBERTReranker,
        "llm": LLMReranker,
    }

    if reranker_type not in rerankers:
        raise ValueError(
            f"Unknown reranker type: {reranker_type}. "
            f"Available: {list(rerankers.keys())}"
        )

    return rerankers[reranker_type](**kwargs)


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
        _logger.info("RAG Rerankers Module Test")
        _logger.info("=" * 60)

        # Create mock search results for testing
        mock_results = [
            SearchResult(
                chunk_id="chunk1",
                document_id="doc1",
                content="NeuralFlow AI is a technology company specializing in "
                "machine learning and artificial intelligence solutions.",
                similarity=0.85,
                metadata={},
                document_title="Company Overview",
                document_source="overview.md",
            ),
            SearchResult(
                chunk_id="chunk2",
                document_id="doc1",
                content="The company offers 20 days of paid time off per year "
                "for full-time employees.",
                similarity=0.72,
                metadata={},
                document_title="Employee Benefits",
                document_source="benefits.md",
            ),
            SearchResult(
                chunk_id="chunk3",
                document_id="doc2",
                content="Our engineering team uses Python, TypeScript, and Go "
                "for backend development with MongoDB and PostgreSQL.",
                similarity=0.68,
                metadata={},
                document_title="Tech Stack",
                document_source="tech.md",
            ),
            SearchResult(
                chunk_id="chunk4",
                document_id="doc3",
                content="The weather forecast for tomorrow shows sunny skies "
                "with temperatures around 75 degrees.",
                similarity=0.45,
                metadata={},
                document_title="Weather Report",
                document_source="weather.md",
            ),
        ]

        test_query = "What does the company do?"
        _logger.info(f"\nTest Query: '{test_query}'")
        _logger.info(f"Mock Results: {len(mock_results)}")

        # Show original order
        _logger.info("\n--- Original Order ---")
        for i, r in enumerate(mock_results):
            _logger.info(f"  [{i + 1}] {r.document_title} (score: {r.similarity:.3f})")
            _logger.info(f"       {r.content[:60]}...")

        # Test LLM Reranker (most likely to work without extra deps)
        _logger.info("\n--- LLM Reranker Test ---")
        try:
            from rag.config.settings import load_settings

            settings = load_settings()
            llm_reranker = create_reranker(
                "llm",
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key,
            )
            _logger.info(f"  Model: {settings.llm_model}")

            reranked = await llm_reranker.rerank(test_query, mock_results, top_k=3)
            _logger.info("  Reranked Results:")
            for i, r in enumerate(reranked):
                orig_score = r.metadata.get("original_score", "N/A")
                _logger.info(
                    f"    [{i + 1}] {r.document_title} "
                    f"(LLM: {r.similarity:.3f}, orig: {orig_score})"
                )
        except Exception as e:
            _logger.error(f"  LLM Reranker failed: {e}")

        # Test CrossEncoder Reranker
        _logger.info("\n--- CrossEncoder Reranker Test ---")
        try:
            ce_reranker = create_reranker(
                "cross_encoder",
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            )
            _logger.info("  Model: cross-encoder/ms-marco-MiniLM-L-6-v2")

            reranked = await ce_reranker.rerank(test_query, mock_results, top_k=3)
            _logger.info("  Reranked Results:")
            for i, r in enumerate(reranked):
                orig_score = r.metadata.get("original_score", "N/A")
                _logger.info(
                    f"    [{i + 1}] {r.document_title} "
                    f"(CE: {r.similarity:.3f}, orig: {orig_score})"
                )
        except ImportError:
            _logger.warning(
                "  Skipped - sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            _logger.error(f"  CrossEncoder Reranker failed: {e}")

        # Test factory function
        _logger.info("\n--- Factory Function Test ---")
        _logger.info("  Available reranker types: cross_encoder, colbert, llm")
        try:
            for rtype in ["cross_encoder", "llm"]:
                reranker = create_reranker(rtype)
                _logger.info(f"  Created {rtype}: {type(reranker).__name__}")
        except Exception as e:
            _logger.error(f"  Factory test failed: {e}")

        # Test invalid reranker type
        _logger.info("\n--- Error Handling Test ---")
        try:
            create_reranker("invalid_type")
        except ValueError as e:
            _logger.info(f"  Caught expected error: {e}")

        _logger.info("\n" + "=" * 60)
        _logger.info("Rerankers test completed!")
        _logger.info("=" * 60)

    asyncio.run(main())
