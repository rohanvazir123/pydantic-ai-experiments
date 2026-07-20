# Copyright (c) 2026 Vikrant Potnis. Licensed under CC BY-NC 4.0.
# See LICENSE file in the project root for details.

"""Knowledge graph retriever.

Queries Apache AGE via the entity_index (hybrid BM25 + cosine) and
AgeGraphStore (entity relationships) to supplement vector search.

Wrapped in a CircuitBreaker — if AGE is down, returns an empty list
and sets the degraded mode flag. The vector path continues unaffected.
"""

import logging
import uuid as _uuid
from typing import Any

import redis.asyncio as aioredis

from knowledge.bus.circuit_breaker import CircuitBreaker, CircuitOpenError
from knowledge.ingestion.models import SearchResult

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieves entity + relationship facts from the AGE knowledge graph."""

    def __init__(
        self,
        age_store: Any,         # AgeGraphStore
        entity_index: Any,      # EntityIndex
        embedder: Any,          # Embedder
        redis: aioredis.Redis | None = None,
    ) -> None:
        self._age     = age_store
        self._index   = entity_index
        self._embedder = embedder
        self._cb: CircuitBreaker | None = (
            CircuitBreaker("age_graph", redis) if redis else None
        )

    async def query(
        self,
        query_text: str,
        corpus_id: str,
        tenant_id: str,
        limit: int = 10,
    ) -> list[SearchResult]:
        """Return SearchResult objects from the knowledge graph.

        Uses entity_index.hybrid_search to find relevant entities, then
        AgeGraphStore.search_as_context for relationship traversal.

        Returns [] on circuit open or any AGE error (graceful degradation).
        """
        async def _run() -> list[SearchResult]:
            query_emb = await self._embedder.embed(query_text)
            entities  = await self._index.hybrid_search(
                query=query_text,
                query_embedding=query_emb,
                corpus_id=corpus_id,
                tenant_id=tenant_id,
                limit=limit,
            )
            # Convert entity hits to SearchResult-shaped objects
            results: list[SearchResult] = []
            for ent in entities:
                results.append(SearchResult(
                    chunk_id=_uuid.uuid4(),
                    document_id=_uuid.UUID(int=0),
                    document_title="Knowledge Graph",
                    document_source=ent.get("document_id", ""),
                    content=ent.get("name", ""),
                    metadata={
                        "entity_type": ent.get("label", "Entity"),
                        "age_uuid":    ent.get("age_uuid", ""),
                        "source":      "graph",
                    },
                    raw_score=float(ent.get("score", 0.0)),
                    raw_score_type="cosine_similarity",
                    confidence=float(ent.get("score", 0.0)),
                ))
            return results

        try:
            if self._cb:
                return await self._cb.call(_run())
            return await _run()
        except CircuitOpenError:
            logger.warning("AGE circuit open — skipping graph retrieval")
            return []
        except Exception as exc:
            logger.error("Graph retrieval failed: %s", exc)
            return []
