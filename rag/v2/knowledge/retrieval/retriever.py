"""Hybrid retriever — the core of the retrieval pipeline.

retrieve() full flow:
  1. L2 Redis cache check (exact query hash)          → HIT: return cached results
  2. Embed query                                       (L1 lru_cache likely hits)
  3. L3 semantic cache check (cosine ≥ threshold)     → HIT: return cached answer
  4. asyncio.gather: semantic_search + text_search + graph_retrieval (optional)
  5. RRF fusion (k=60)
  6. CrossEncoder rerank → populate confidence
  7. Confidence filter (≥ min_confidence_score)
  8. Populate L2 Redis cache (async, non-blocking)

retrieve_with_confidence() = retrieve() + Layer 1 gate:
  aggregate confidence sum < retrieval_confidence_threshold → return []
"""

import asyncio
import logging
from typing import Any

import redis.asyncio as aioredis

from knowledge.bus.circuit_breaker import CircuitBreaker, CircuitOpenError
from knowledge.config.settings import Settings, load_settings
from knowledge.ingestion.embedder import Embedder
from knowledge.ingestion.models import SearchResult
from knowledge.retrieval.fusion import (
    apply_confidence_filter,
    fuse_to_search_results,
    rerank,
)
from knowledge.store.cache import RedisCache
from knowledge.store.vector import PostgresHybridStore

logger = logging.getLogger(__name__)

_OVERFETCH = 3   # fetch k*3 candidates before reranking


class Retriever:
    """Orchestrates hybrid retrieval, caching, reranking, and confidence gating."""

    def __init__(
        self,
        vector_store: PostgresHybridStore | None = None,
        embedder: Embedder | None = None,
        cache: RedisCache | None = None,
        semantic_cache: Any | None = None,   # SemanticCache | None
        graph_retriever: Any | None = None,  # GraphRetriever | None
        settings: Settings | None = None,
        redis: aioredis.Redis | None = None,
    ) -> None:
        self._vs      = vector_store
        self._emb     = embedder
        self._cache   = cache
        self._sc      = semantic_cache
        self._graph   = graph_retriever
        self._settings = settings or load_settings()
        # Circuit breakers — None when redis not provided (tests, tool-call retriever)
        self._embed_cb = CircuitBreaker("embedding_service", redis) if redis else None
        self._sem_cb   = CircuitBreaker("pgvector_search",   redis) if redis else None
        self._text_cb  = CircuitBreaker("pgvector_text",     redis) if redis else None

    # ── Main retrieval ────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        corpus_ids: list[str],
        tenant_id: str,
        k: int = 5,
        search_type: str = "hybrid",
        metadata_filter: Any | None = None,
        include_graph: bool = False,
    ) -> list[SearchResult]:
        """Full retrieval pipeline. Returns confidence-scored, filtered results."""

        # ── 1. L2 Redis cache (exact hash) ────────────────────────────────────
        if self._cache:
            cached = await self._cache.get_search(query, corpus_ids)
            if cached is not None:
                logger.debug("L2 cache hit for query='%s...'", query[:40])
                return self._dicts_to_results(cached)

        # ── 2. Embed query ────────────────────────────────────────────────────
        query_emb: list[float] = []
        if self._emb:
            try:
                if self._embed_cb:
                    query_emb = await self._embed_cb.call(self._emb.embed(query))
                else:
                    query_emb = await self._emb.embed(query)
            except CircuitOpenError:
                logger.warning("embedding_service circuit OPEN — semantic + L3 cache skipped")
            except Exception as exc:
                logger.warning("Embedding failed (%s) — semantic + L3 cache skipped", exc)

        # ── 3. L3 semantic cache ──────────────────────────────────────────────
        if self._sc and query_emb:
            cached_answer = await self._sc.lookup(query_emb, corpus_ids, tenant_id)
            if cached_answer is not None:
                logger.debug("L3 semantic cache hit for query='%s...'", query[:40])
                # Return the SearchResults embedded in the cached answer
                if "search_results" in cached_answer:
                    return self._dicts_to_results(cached_answer["search_results"])

        # ── 4. Parallel retrieval ─────────────────────────────────────────────
        fetch = k * _OVERFETCH
        sem_task  = self._semantic_search(query_emb, corpus_ids, tenant_id, fetch)
        text_task = self._text_search(query, corpus_ids, tenant_id, fetch)
        graph_task = (
            self._graph.query(query, corpus_ids[0] if corpus_ids else "", tenant_id, k)
            if include_graph and self._graph
            else asyncio.sleep(0, result=[])
        )

        sem_results, text_results, graph_results = await asyncio.gather(
            sem_task, text_task, graph_task
        )

        # ── 5. RRF fusion ─────────────────────────────────────────────────────
        raw_lists: list[list[dict]] = []
        if sem_results:
            raw_lists.append(sem_results)
        if text_results:
            raw_lists.append(text_results)

        if not raw_lists:
            return []

        fused = fuse_to_search_results(raw_lists, top_k=fetch)

        # Append graph results (already SearchResult objects) after fusion
        if graph_results:
            fused = fused + list(graph_results)

        # ── 6. CrossEncoder rerank ────────────────────────────────────────────
        reranked = await rerank(query, fused)

        # ── 7. Confidence filter + trim to k ─────────────────────────────────
        filtered = apply_confidence_filter(
            reranked, self._settings.min_confidence_score
        )[:k]

        # ── 8. Populate L2 Redis cache (async, non-blocking) ─────────────────
        if self._cache and filtered:
            asyncio.create_task(
                self._cache.set_search(
                    query, corpus_ids,
                    [self._result_to_dict(r) for r in filtered],
                )
            )

        return filtered

    # ── Layer 1 confidence gate ───────────────────────────────────────────────

    async def retrieve_with_confidence(
        self,
        query: str,
        corpus_ids: list[str],
        tenant_id: str,
        k: int = 5,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """retrieve() + Layer 1 gate: empty list if aggregate confidence too low."""
        results = await self.retrieve(query, corpus_ids, tenant_id, k=k, **kwargs)
        if not results:
            return []

        top_k = results[:k]
        aggregate = sum(
            r.confidence for r in top_k if r.confidence is not None
        )
        threshold = self._settings.retrieval_confidence_threshold

        if aggregate < threshold:
            logger.info(
                "Layer 1 gate: aggregate_confidence=%.2f < threshold=%.2f → abstain",
                aggregate, threshold,
            )
            return []

        return results

    # ── Context helper for agent tools ───────────────────────────────────────

    async def retrieve_as_context(
        self,
        query: str,
        corpus_ids: list[str],
        tenant_id: str,
        k: int = 5,
        **kwargs: Any,
    ) -> str:
        """Retrieve and format results as an LLM-ready context string."""
        results = await self.retrieve(query, corpus_ids, tenant_id, k=k, **kwargs)
        if not results:
            return "No relevant information found."
        lines = []
        for r in results:
            lines.append(
                f"[chunk_id: {r.chunk_id}] {r.document_title} ({r.document_source})\n"
                f"{r.content[:500]}"
            )
        return "\n\n".join(lines)

    # ── Internal search helpers ───────────────────────────────────────────────

    async def _semantic_search(
        self,
        query_emb: list[float],
        corpus_ids: list[str],
        tenant_id: str,
        k: int,
    ) -> list[dict]:
        if not self._vs or not query_emb:
            return []

        async def _run() -> list[dict]:
            rows: list[dict] = []
            for corpus_id in corpus_ids:
                rows.extend(await self._vs.semantic_search(query_emb, corpus_id, tenant_id, k))
            return rows

        try:
            if self._sem_cb:
                return await self._sem_cb.call(_run())
            return await _run()
        except CircuitOpenError:
            logger.warning("pgvector_search circuit OPEN — semantic leg skipped")
            return []
        except Exception as exc:
            logger.error("Semantic search failed (%s)", exc)
            return []

    async def _text_search(
        self,
        query: str,
        corpus_ids: list[str],
        tenant_id: str,
        k: int,
    ) -> list[dict]:
        if not self._vs:
            return []

        async def _run() -> list[dict]:
            rows: list[dict] = []
            for corpus_id in corpus_ids:
                rows.extend(await self._vs.text_search(query, corpus_id, tenant_id, k))
            return rows

        try:
            if self._text_cb:
                return await self._text_cb.call(_run())
            return await _run()
        except CircuitOpenError:
            logger.warning("pgvector_text circuit OPEN — text search leg skipped")
            return []
        except Exception as exc:
            logger.error("Text search failed (%s)", exc)
            return []

    # ── Serialisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _result_to_dict(r: SearchResult) -> dict:
        return {
            "chunk_id":       str(r.chunk_id),
            "document_id":    str(r.document_id),
            "document_title": r.document_title,
            "document_source": r.document_source,
            "content":        r.content,
            "metadata":       r.metadata,
            "raw_score":      r.raw_score,
            "raw_score_type": r.raw_score_type,
            "confidence":     r.confidence,
        }

    @staticmethod
    def _dicts_to_results(dicts: list[dict]) -> list[SearchResult]:
        import uuid as _uuid
        results: list[SearchResult] = []
        for d in dicts:
            try:
                results.append(SearchResult(
                    chunk_id=_uuid.UUID(str(d.get("chunk_id", _uuid.uuid4()))),
                    document_id=_uuid.UUID(str(d.get("document_id", _uuid.uuid4()))),
                    document_title=d.get("document_title", ""),
                    document_source=d.get("document_source", ""),
                    content=d.get("content", ""),
                    metadata=d.get("metadata", {}),
                    raw_score=float(d.get("raw_score", 0.0)),
                    raw_score_type=d.get("raw_score_type", "rrf"),
                    confidence=d.get("confidence"),
                ))
            except Exception:
                pass
        return results
