"""RRF fusion and CrossEncoder reranking.

Two-step process:
  1. RRF(k=60): merge raw results from semantic + text (+ optional graph) legs
     into a single ranked list. raw_score_type = "rrf".
  2. CrossEncoder rerank: batch score all (query, chunk) pairs; populate
     confidence = sigmoid(logit). Runs in asyncio.to_thread (CPU-bound).

confidence is the only score the agent and API expose externally.
raw_score is kept for debugging but never returned to users.

Confidence filter: after reranking, drop results where
  confidence < settings.min_confidence_score (default 0.10).
"""

import asyncio
import logging
import math
from typing import Any

from knowledge.ingestion.models import SearchResult

logger = logging.getLogger(__name__)

RRF_K: int = 60

_MODEL_NAME    = "BAAI/bge-reranker-base"
_reranker: Any = None   # loaded lazily on first use, cached for process lifetime


def _load_reranker() -> Any:
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(_MODEL_NAME)
            logger.info("CrossEncoder loaded: %s", _MODEL_NAME)
        except ImportError:
            logger.warning(
                "sentence_transformers not installed — reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
            _reranker = False   # sentinel: tried and unavailable
    return _reranker


def sigmoid(x: float) -> float:
    """Logistic sigmoid. Maps cross-encoder logit → calibrated 0-1 confidence."""
    return 1.0 / (1.0 + math.exp(-x))


# ── RRF fusion ────────────────────────────────────────────────────────────────

def rrf_fuse(
    result_lists: list[list[dict[str, Any]]],
    k: int = RRF_K,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over multiple ranked result lists.

    Each list is a flat list of dicts with at least {"id": ..., ...}.
    Returns a merged list ordered by RRF score descending, up to top_k.
    """
    scores: dict[str, float] = {}
    items:  dict[str, dict[str, Any]] = {}

    for result_list in result_lists:
        for rank, item in enumerate(result_list):
            item_id = str(item.get("id", ""))
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            if item_id not in items:
                items[item_id] = item

    merged = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)

    out: list[dict[str, Any]] = []
    for item_id, score in merged[:top_k]:
        row = dict(items[item_id])
        row["raw_score"]      = score
        row["raw_score_type"] = "rrf"
        row["confidence"]     = None   # set by CrossEncoder below
        out.append(row)

    return out


def fuse_to_search_results(
    result_lists: list[list[dict[str, Any]]],
    k: int = RRF_K,
    top_k: int = 20,
) -> list[SearchResult]:
    """RRF fuse raw DB rows into SearchResult objects (confidence still None)."""
    import uuid as _uuid

    fused = rrf_fuse(result_lists, k=k, top_k=top_k)
    results: list[SearchResult] = []
    for row in fused:
        try:
            results.append(SearchResult(
                chunk_id=_uuid.UUID(str(row.get("id", _uuid.uuid4()))),
                document_id=_uuid.UUID(str(row.get("document_id", _uuid.uuid4()))),
                document_title=str(row.get("title", row.get("metadata", {}).get("title", ""))),
                document_source=str(row.get("source", row.get("metadata", {}).get("source", ""))),
                content=str(row.get("content", "")),
                metadata=dict(row.get("metadata", {})),
                raw_score=float(row.get("raw_score", 0.0)),
                raw_score_type="rrf",
                confidence=None,
            ))
        except Exception as exc:
            logger.warning("Skipping malformed result row: %s", exc)
    return results


# ── CrossEncoder reranker ─────────────────────────────────────────────────────

def _rerank_sync(
    query: str,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Synchronous CrossEncoder reranking — runs inside asyncio.to_thread.

    Returns results unchanged (with confidence=None) when sentence_transformers
    is not installed, so the pipeline degrades gracefully to pure RRF ordering.
    """
    if not results:
        return results

    model = _load_reranker()
    if not model:   # unavailable — skip reranking
        return results

    pairs = [(query, r.content) for r in results]
    logits: list[float] = model.predict(pairs).tolist()

    for result, logit in zip(results, logits):
        result.confidence = sigmoid(logit)

    return sorted(results, key=lambda r: r.confidence or 0.0, reverse=True)


async def rerank(
    query: str,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Async CrossEncoder rerank. Offloads model inference to threadpool."""
    if not results:
        return results
    return await asyncio.to_thread(_rerank_sync, query, results)


def apply_confidence_filter(
    results: list[SearchResult],
    min_confidence: float,
) -> list[SearchResult]:
    """Drop results where confidence < min_confidence.

    Only applied after reranking (confidence is None before reranking).
    For standalone text search (confidence=None), filter is skipped.
    """
    return [
        r for r in results
        if r.confidence is None or r.confidence >= min_confidence
    ]
