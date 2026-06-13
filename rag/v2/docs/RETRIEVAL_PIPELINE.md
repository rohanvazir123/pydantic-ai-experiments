# Retrieval Pipeline

**Key file:** `rag/retrieval/retriever.py` → `Retriever.retrieve()`

---

## Table of Contents

- [Pipeline Steps](#pipeline-steps)
- [Configuration](#configuration)
- [Caching Layers](#caching-layers)
- [Agent Integration](#agent-integration)
- [Search Types](#search-types)

---

## Pipeline Steps

```
1. Cache check
      ResultCache (LRU, TTL=5min, capacity=100)
      Key: sha256(query:search_type:match_count)[:24]
      Cache hit → return immediately

2. Query embedding
      EmbeddingGenerator.embed_query(query)
      → async LRU cache on (text, model)
      → POST /v1/embeddings → 768-dim vector

3. Search
      "semantic"  ORDER BY embedding <=> $1::vector LIMIT N
      "text"      WHERE content_tsv @@ plainto_tsquery('english', $1)
      "hybrid"    asyncio.gather(semantic, text) → RRF merge (k=60)

4. Score filter (semantic mode only)
      Drop chunks below MIN_RELEVANCE_SCORE (default 0.0)
      Not applied for hybrid or text modes — RRF and ts_rank scores are not
      calibrated to the same 0–1 scale as cosine similarity

5. Cache write + return list[SearchResult]
```

---

## Configuration

| Setting | Default | Effect |
|---------|---------|--------|
| `DEFAULT_MATCH_COUNT` | `10` | Results returned per query |
| `MIN_RELEVANCE_SCORE` | `0.0` | Drop chunks below this threshold (semantic mode only); `0.0` = disabled |

---

## Caching Layers

| Cache | Mechanism | Key | TTL | Capacity |
|-------|-----------|-----|-----|----------|
| Embeddings | `@alru_cache` | `(text, model)` | None | 1000 entries |
| Search results | `ResultCache` (OrderedDict LRU) | `SHA-256(query, type, count, metadata_filter)` | 5 min | 100 entries |

---

## Agent Integration

The retriever is invoked by the RAG agent's `search_knowledge_base` tool:

```
search_knowledge_base(query) called by agent
  → RAGState.get_retriever()          lazy-init, thread-safe via asyncio.Lock
  → retriever.retrieve_as_context()   hybrid search → formatted chunk string
  → mem0_store.get_context_string()   user memory facts (if MEM0_ENABLED)
  → combined context returned to LLM
```

`RAGState` holds all lazy-initialized resources as `PrivateAttr`:

```python
_store:       PostgresHybridStore
_retriever:   Retriever
_mem0:        Mem0Store | None       (if MEM0_ENABLED)
_initialized: bool
_init_lock:   asyncio.Lock
```

---

## Search Types

| Type | SQL | Best for |
|------|-----|---------|
| `semantic` | `ORDER BY embedding <=> $1::vector` | Conceptual / paraphrase queries |
| `text` | `WHERE content_tsv @@ plainto_tsquery(...)` | Exact keywords, legal terms |
| `hybrid` | RRF merge of both | Default — best overall recall |

RRF formula: `score = Σ 1 / (k + rank_i)` where `k=60`. Rewards chunks that rank well in both lists.
