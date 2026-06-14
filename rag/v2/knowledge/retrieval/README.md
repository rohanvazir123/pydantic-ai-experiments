# knowledge/retrieval/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Retrieval Path](#retrieval-path)
- [Confidence Scoring](#confidence-scoring)

---

## What This Is

The retrieval pipeline. Takes a query and returns ranked, confidence-scored chunks from the knowledge corpus, with three cache layers to avoid redundant LLM and DB calls.

---

## Files

| File | Purpose |
|------|---------|
| `retriever.py` | `Retriever`: hybrid search → RRF fusion → CrossEncoder rerank → confidence filter |
| `fusion.py` | `RRF(k=60)` + `CrossEncoder` reranker; `sigmoid(logit)` → `confidence` score |
| `semantic_cache.py` | L3 semantic cache: cosine lookup in `semantic_cache` table; JWE-encrypted answers |
| `graph_retriever.py` | NL→Cypher query against `AgeGraphStore`; wrapped in circuit breaker |
| `worker.py` | Redis consumer for async/bulk search requests |

---

## Retrieval Path

```
query
  │
  ├─ L3 semantic cache check (pgvector cosine ≥ 0.95) → HIT: return cached answer
  ├─ L2 Redis exact hash check                         → HIT: return cached chunks
  │
  ├─ asyncio.gather(
  │     semantic_search(query_embedding, k × overfetch)
  │     text_search(query_text, k × overfetch)
  │     graph_retrieval(query_text)   ← optional, circuit-broken
  │   )
  │
  ├─ RRF fusion (k=60)
  ├─ CrossEncoder rerank → confidence = sigmoid(logit)
  ├─ Confidence filter (≥ min_confidence_score)
  │
  └─ Populate L2 Redis cache (async, non-blocking)
```

---

## Confidence Scoring

`SearchResult.confidence` is set **after CrossEncoder reranking** — it is not the raw cosine similarity or RRF score.

| Search path | `raw_score` | `confidence` |
|-------------|------------|-------------|
| Hybrid (default) | RRF score (≤ 0.05) | `sigmoid(cross_encoder_logit)` — 0 to 1 |
| Semantic only | cosine similarity | cosine similarity (already 0–1) |
| Text only | `ts_rank` (unbounded) | `None` — not calibrated |

The retrieval confidence gate (`retrieve_with_confidence()`) blocks requests where `sum(confidence for top-K) < retrieval_confidence_threshold` — preventing LLM calls on empty or garbage retrieval.
