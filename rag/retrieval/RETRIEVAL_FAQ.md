# Retrieval FAQ

**Last updated:** 2026-06-04

Answers the question: *"What is each component in `rag/retrieval/`, who actually uses it, and how is it tested?"*

---

## Table of Contents

1. [Component Map](#1-component-map)
2. [Retriever (core orchestrator)](#2-retriever-retrieverpy)
3. [Rerankers](#3-rerankers-rerankerspy)
4. [HybridKGRetriever + IntentClassifier (orphaned)](#4-hybridkgretriever--intentclassifier-orphaned)
5. [Retrieval Metrics Explained](#5-retrieval-metrics-explained)
6. [Test Files Reference](#6-test-files-reference)

---

## 1. Component Map

```
rag/retrieval/
├── retriever.py              ACTIVE — core orchestrator, used everywhere
├── rerankers.py              ACTIVE (optional) — LLMReranker / CrossEncoderReranker via settings flag
├── hybrid_kg_retriever.py    ORPHANED — no active callers; candidate for deletion
└── intent_classifier.py      ORPHANED — used only by HybridKGRetriever
```

**Call path for a normal agent query:**

```
rag_agent.py (search_knowledge_base tool)
    └── Retriever.retrieve(query)
            ├── PostgresHybridStore.hybrid_search()          — vector + BM25 + RRF
            └── [optional] LLMReranker / CrossEncoderReranker — if reranker_enabled=True
```

---

## 2. Retriever (`retriever.py`)

**What it does:** Main retrieval orchestrator. Embeds the query, hits PostgreSQL, applies optional reranking, and caches results.

### Pipeline (in order)

| Step | Code | Controlled by |
|------|------|--------------|
| 1. Cache check | `ResultCache.get()` | `use_cache=True` arg |
| 2. Query embedding | `EmbeddingGenerator.embed_query()` | always |
| 3. Over-fetch | `fetch_count = match_count × overfetch_factor` | `settings.reranker_enabled` |
| 4. Search | `hybrid_search` / `semantic_search` / `text_search` | `search_type` arg |
| 5. Rerank (optional) | `LLMReranker` or `CrossEncoderReranker` | `settings.reranker_enabled` |
| 6. Relevance guardrail | drop chunks below `min_relevance_score` | semantic mode only |
| 7. Cache write | `ResultCache.set()` | `use_cache=True` arg |

### Search types

| `search_type` | What runs | Score meaning |
|---------------|-----------|---------------|
| `"hybrid"` (default) | pgvector cosine + `ts_rank` BM25, merged with RRF | RRF rank score (~0.016 range) |
| `"semantic"` | pgvector cosine similarity only | 0–1 cosine similarity |
| `"text"` | PostgreSQL `ts_rank` full-text only | `ts_rank` score (not 0–1 calibrated) |

> The relevance guardrail (`min_relevance_score`) only applies to `semantic` mode. RRF and `ts_rank` scores are not on the same 0–1 scale so the threshold would be meaningless there.

### Feature flags (`.env` / `Settings`)

| Setting | Default | Effect |
|---------|---------|--------|
| `reranker_enabled` | `True` | Always on — rerank with CrossEncoder before returning results |
| `reranker_type` | `"cross_encoder"` | `"cross_encoder"` (local, default) or `"llm"` |
| `reranker_model` | `"BAAI/bge-reranker-base"` | CrossEncoder model loaded via `sentence-transformers` (local) |
| `reranker_overfetch_factor` | `3` | Fetch `N × factor` results before reranking |
| `min_relevance_score` | `0.0` | Drop semantic results below this cosine similarity |
| `default_match_count` | `5` | Default K when `match_count` arg omitted |

### Where it's used

- `rag/agent/rag_agent.py` — `search_knowledge_base` tool
- All retrieval quality tests (`test_retrieval_metrics.py`)
- Integration tests in `test_rag_agent.py`

---

## 3. Rerankers (`rerankers.py`)

Used by `Retriever` when `reranker_enabled=True`. The retriever first over-fetches (`match_count × overfetch_factor`) then trims to `match_count` using the reranker.

### CrossEncoderReranker — DEFAULT (local, always on)

Uses a cross-encoder model (`BAAI/bge-reranker-base`) that jointly encodes the query and document. Runs locally via `sentence-transformers` — no API call. Faster and more consistent than LLM reranking.

**Config:** `reranker_type = "cross_encoder"` (default), `reranker_model = "BAAI/bge-reranker-base"`

**Install:** `pip install sentence-transformers`

### LLMReranker — ALTERNATIVE

Sends each (query, chunk) pair to the LLM asking for a 0–10 relevance score. All pairs scored concurrently via `asyncio.gather`. Scores are normalised to 0–1.

**When to use:** If `sentence-transformers` is unavailable, or for domain-specific reranking where the chat LLM outperforms a generic cross-encoder.

**Config:** `reranker_type = "llm"`

### ColBERTReranker — NOT RECOMMENDED

**Code smell:** The full ColBERT model requires pre-indexed documents. This implementation silently falls back to standard sentence-transformer dot-product scoring, which is not ColBERT. Use `CrossEncoderReranker` instead until a proper ColBERT index is built.

---

## 4. HybridKGRetriever + IntentClassifier (orphaned)

Both files (`hybrid_kg_retriever.py`, `intent_classifier.py`) are still present but have no active callers — the `search_knowledge_graph` agent tool and all associated KG tests were removed when the legal CUAD use case was moved to `misc/kg_legal_cuad/`. **Candidates for deletion** once KG retrieval is re-implemented in the `knowledge/` module (see TODO.md Phase D).

**What they did:**
- `HybridKGRetriever` — ran semantic text retrieval (`Retriever`) and structured KG lookup (`AgeGraphStore`) in parallel, then fused both into a single context block.
- `IntentClassifier` — regex-based intent detection (no LLM); returned `HYBRID` or `STRUCTURED` to control which paths fired.

---

## 5. Retrieval Metrics Explained

Defined in `rag/tests/retrieval/test_retrieval_metrics.py`.

All metrics are computed against a **gold dataset** — a fixed set of (query → list of relevant document source stems) pairs. Relevance is determined by case-insensitive substring match of the stem against `result.document_source`.

### IR Metrics

#### Hit Rate@K
```
Hit Rate@K = fraction of queries where ≥1 relevant document appears in the top-K results
```
Binary per query: 1.0 if any relevant doc is found, 0.0 if none. The most forgiving metric — measures whether retrieval *finds anything useful*.

#### MRR@K (Mean Reciprocal Rank)
```
MRR@K = mean of (1 / rank_of_first_relevant_result) across queries
       = 1.0 if relevant at rank 1, 0.5 if rank 2, 0.33 if rank 3, ...
       = 0.0 if no relevant result in top-K
```
Measures how *highly ranked* the first relevant result is.

#### Precision@K
```
Precision@K = (number of relevant results in top-K) / K
```
Measures the *density* of relevant results in the returned set.

#### Recall@K
```
Recall@K = (relevant results in top-K) / (total relevant documents for this query)
```
Measures coverage: what fraction of all known-relevant documents does the system surface.

#### NDCG@K (Normalised Discounted Cumulative Gain)
```
DCG@K  = Σ rel_i / log2(i+2)   for i=0..K-1
IDCG@K = Σ 1    / log2(i+2)   for i=0..min(#relevant, K)-1
NDCG@K = DCG@K / IDCG@K       (1.0 = perfect ranking, 0.0 = no relevant results)
```
Rewards systems that rank relevant documents *above* irrelevant ones.

### System Metrics

#### Mean Latency
Average time (ms) per query across the full gold dataset run. Measured with `time.perf_counter()` around `retriever.retrieve()`.

#### P95 Latency
The 95th-percentile query latency. The test asserts P95 < 10 seconds.

### Gold Dataset and Thresholds

| Dataset | File | Queries | K=5 thresholds |
|---------|------|---------|----------------|
| NeuralFlow AI corpus | `test_retrieval_metrics.py` `GOLD_DATASET` | 10 | Hit Rate ≥0.60, MRR ≥0.40, Precision ≥0.15, Recall ≥0.40, NDCG ≥0.40 |

### Hybrid vs. semantic comparison test

`test_hybrid_beats_semantic_alone` verifies that adding BM25 via RRF does not hurt Hit Rate@5 by more than 10 percentage points vs. pure semantic search.

---

## 6. Test Files Reference

| File | Subfolder | What it tests | Deps |
|------|-----------|--------------|------|
| `test_retrieval_metrics.py` | `retrieval/` | `Retriever` against NeuralFlow gold dataset; IR metric unit tests | PostgreSQL + Ollama |
| `test_rag_agent.py` | `agent/` | End-to-end agent queries via `Retriever` | PostgreSQL + Ollama |
| `test_agent_flow.py` | `agent/` | Pydantic AI event stream debugging | PostgreSQL + Ollama |
| `test_api.py` | `agent/` | FastAPI REST endpoints | Mocked |
| `test_mcp_server.py` | `agent/` | MCP server tools | Mocked |
| `test_postgres_store.py` | `storage/` | `PostgresHybridStore` connection + indexes | PostgreSQL |
| `test_mem0_store.py` | `storage/` | `Mem0Store` CRUD | PostgreSQL |
| `test_config.py` | `core/` | Settings loading, credential masking | None |
| `test_ingestion.py` | `core/` | `ChunkData`, `SearchResult` models | None |
| `test_raganything.py` | `experimental/` | RAG-Anything modal processors | Mocked |
| `test_pdf_question_generator.py` | `experimental/` | `PDFQuestionStore` | PostgreSQL |

### Running subsets

```bash
# All tests
python -m pytest rag/tests/ -v

# Fast unit tests only (no external deps)
python -m pytest rag/tests/core/ -v

# Retrieval quality (requires PostgreSQL + Ollama)
python -m pytest rag/tests/retrieval/ -v --log-cli-level=INFO

# Skip all integration tests
python -m pytest rag/tests/ -m "not integration" -v
```
