# Retrieval FAQ

**Last updated:** 2026-05-14

Answers the question: *"What is each component in `rag/retrieval/`, who actually uses it, and how is it tested?"*

---

## Table of Contents

1. [Component Map](#1-component-map)
2. [Retriever (core orchestrator)](#2-retriever-retrieverpy)
3. [HybridKGRetriever](#3-hybridkgretriever-hybrid_kg_retrieverpy)
4. [Query Processors](#4-query-processors-query_processorspy)
5. [Rerankers](#5-rerankers-rerankerspy)
6. [Intent Classifier](#6-intent-classifier-intent_classifierpy)
7. [Dead Code](#7-dead-code-dead_codecontext_expanderspy)
8. [Retrieval Metrics Explained](#8-retrieval-metrics-explained)
9. [Test Files Reference](#9-test-files-reference)

---

## 1. Component Map

```
rag/retrieval/
├── retriever.py              ACTIVE — core orchestrator, used everywhere
├── hybrid_kg_retriever.py    ACTIVE — KG + text fusion (search_knowledge_graph tool)
├── query_processors.py       PARTIAL — HyDEProcessor wired in; others available but not wired
├── rerankers.py              ACTIVE (optional) — LLMReranker / CrossEncoderReranker via settings flag
├── intent_classifier.py      ACTIVE — used only by HybridKGRetriever
└── dead_code/
    └── context_expanders.py  DEAD — references MongoDB-style store methods that don't exist
                              on PostgresHybridStore; treat as a code smell until re-implemented
```

**Call path for a normal agent query:**

```
rag_agent.py (search_knowledge_base tool)
    └── Retriever.retrieve(query)
            ├── [optional] HyDEProcessor.generate_hypothetical()  — if hyde_enabled=True
            ├── PostgresHybridStore.hybrid_search()               — vector + BM25 + RRF
            └── [optional] LLMReranker / CrossEncoderReranker      — if reranker_enabled=True

rag_agent.py (search_knowledge_graph tool)
    └── HybridKGRetriever.retrieve(query)
            ├── IntentClassifier.classify(query)
            ├── [parallel] Retriever.retrieve()                    — semantic path
            └── [parallel] AgeGraphStore / PgGraphStore            — structured KG path
```

---

## 2. Retriever (`retriever.py`)

**What it does:** Main retrieval orchestrator. Embeds the query, hits PostgreSQL, applies optional post-processing, and caches results.

### Pipeline (in order)

| Step | Code | Controlled by |
|------|------|--------------|
| 1. Cache check | `ResultCache.get()` | `use_cache=True` arg |
| 2. Query embedding | `EmbeddingGenerator.embed_query()` | always |
| 2a. HyDE (optional) | `HyDEProcessor.generate_hypothetical()` | `settings.hyde_enabled` |
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
| `hyde_enabled` | `False` | Embed a hypothetical answer doc instead of the raw query |
| `reranker_enabled` | `False` | Rerank over-fetched results after initial search |
| `reranker_type` | `"llm"` | `"llm"` or `"cross_encoder"` |
| `reranker_overfetch_factor` | `3` | Fetch `N × factor` results before reranking |
| `min_relevance_score` | `0.0` | Drop semantic results below this cosine similarity |
| `default_match_count` | `5` | Default K when `match_count` arg omitted |

### Where it's used

- `rag/agent/rag_agent.py` — `search_knowledge_base` tool
- `rag/retrieval/hybrid_kg_retriever.py` — semantic path of `HybridKGRetriever`
- All retrieval quality tests (`test_retrieval_metrics.py`, `test_legal_retrieval.py`)
- Integration tests in `test_rag_agent.py`

---

## 3. HybridKGRetriever (`hybrid_kg_retriever.py`)

**What it does:** Routes queries to two paths in parallel — semantic text retrieval (via `Retriever`) and structured KG lookup (via `AgeGraphStore` or `PgGraphStore`) — then fuses both into a single LLM context block.

### When to use it

Queries that need *both* factual entity/relationship lookups AND passage-level evidence. Example: *"What are the termination obligations between Party A and Party B in distributor agreements?"* — the KG path finds entities and relationships; the text path finds the actual clause text.

### Routing (via IntentClassifier)

| Intent | Semantic path? | KG path? | Triggered by |
|--------|---------------|----------|-------------|
| `HYBRID` (default) | Yes | Yes | Most natural-language questions |
| `STRUCTURED` | No | Yes | Pure aggregation queries: "how many", "count", "distribution of" |
| `SEMANTIC` | Yes | No | (reserved — classifier currently only returns HYBRID or STRUCTURED) |

### Fused context layout

```
## Knowledge Graph Facts
  Entities (N matched):  [TYPE] EntityName (DocumentTitle)
  Relationships (N found): [SRC_TYPE] SrcName —[REL_TYPE]→ [TGT_TYPE] TgtName

## Relevant Text Passages
  [1] DocumentTitle (relevance: 0.82)
  <chunk text>
```

### Where it's used

- `rag/agent/rag_agent.py` — `search_knowledge_graph` tool

### Tests

- `rag/tests/knowledge_graph/test_hybrid_kg_retrieval.py` — unit (fully mocked) + integration

---

## 4. Query Processors (`query_processors.py`)

Four classes, only one is wired into the main `Retriever`.

### HyDEProcessor — ACTIVE (via `hyde_enabled` flag)

**What it does:** Generates a hypothetical document that *would* answer the query, then embeds that document instead of the raw query. The hypothesis is usually more similar to real documents than a short query string.

**Why:** A query like *"PTO policy"* has a very different embedding from a paragraph about time-off policies. HyDE bridges that semantic gap.

**Reference:** [Precise Zero-Shot Dense Retrieval without Relevance Labels (arXiv 2212.10496)](https://arxiv.org/abs/2212.10496)

**How it's wired:** `Retriever._get_hyde()` lazy-inits a `HyDEProcessor` from settings. When `hyde_enabled=True`, the hypothetical text replaces the raw query for embedding — but the original query string still goes to BM25 in hybrid mode.

---

### LLMQueryExpander — NOT WIRED

Generates N alternative phrasings of the query (e.g. synonyms, reformulations) using an LLM. Useful for improving recall on keyword-sensitive queries. Available via `create_query_processor("expander")` but not called by `Retriever`. **Treat as available-but-inactive until explicitly integrated.**

### MultiQueryProcessor — NOT WIRED

Combines `LLMQueryExpander` + `HyDEProcessor` into one pass. Returns multiple query variants and embeddings. Not called by `Retriever`. Would require changes to `Retriever.retrieve()` to run one search per variant and merge results.

### QueryDecomposer — NOT WIRED

Breaks a complex multi-hop question into simpler sub-questions. Not called by `Retriever`. Useful for future multi-step retrieval.

---

## 5. Rerankers (`rerankers.py`)

Used by `Retriever` when `reranker_enabled=True`. The retriever first over-fetches (`match_count × overfetch_factor`) then trims to `match_count` using the reranker.

### LLMReranker — DEFAULT, ACTIVE

Sends each (query, chunk) pair to the LLM asking for a 0–10 relevance score. All pairs scored concurrently via `asyncio.gather`. Scores are normalised to 0–1.

**When to use:** Good baseline with no extra dependencies. Slow on large candidate sets because it makes one LLM call per chunk.

**Config:** `reranker_type = "llm"` (default)

### CrossEncoderReranker — ACTIVE (requires `sentence-transformers`)

Uses a cross-encoder model (e.g. `BAAI/bge-reranker-base`) that jointly encodes the query and document. Much faster than LLM reranking for large candidate sets and typically more consistent.

**When to use:** Production workloads where latency matters.

**Config:** `reranker_type = "cross_encoder"`, `reranker_model = "BAAI/bge-reranker-base"`

**Install:** `pip install sentence-transformers`

### ColBERTReranker — NOT RECOMMENDED

**Code smell:** The full ColBERT model requires pre-indexed documents. This implementation silently falls back to standard sentence-transformer dot-product scoring, which is not ColBERT. Use `CrossEncoderReranker` instead until a proper ColBERT index is built.

---

## 6. Intent Classifier (`intent_classifier.py`)

Used exclusively by `HybridKGRetriever` to decide which retrieval path(s) to activate.

**Logic:** Regex-based, no LLM call.

| Pattern matched | Intent | Example |
|----------------|--------|---------|
| Aggregation keywords AND no text-retrieval signals | `STRUCTURED` | "how many contracts have a termination clause?" |
| Everything else | `HYBRID` | "what does the termination clause say in distributor agreements?" |

Aggregation keywords: `how many`, `count`, `total number`, `on average`, `distribution of`, `most common`, `top N`, `highest`, `lowest`, `aggregate`, `median`, etc.

Text-retrieval override: if the query also asks for *exact wording* or *full text*, it downgrades to `HYBRID` so the text path still fires.

**Tests:** `rag/tests/knowledge_graph/test_hybrid_kg_retrieval.py` includes intent classification assertions.

---

## 7. Dead Code (`dead_code/context_expanders.py`)

**Code smell:** `AdjacentChunkExpander`, `SectionExpander`, `DocumentSummaryExpander`, `CompositeExpander` all call store methods (`get_chunk_by_id`, `get_chunks_by_document`, `get_document_by_id`) using a MongoDB-style interface that `PostgresHybridStore` does not implement. None of these are called anywhere in the active codebase.

**What they were designed for:** Retrieving adjacent chunks around a matched chunk (sliding context window), or fetching the full section containing a match. Both are valid RAG enhancement patterns.

**Path forward:** Either implement the missing methods on `PostgresHybridStore` and re-integrate, or delete these files entirely. Until then, they live in `dead_code/` to signal their status.

---

## 8. Retrieval Metrics Explained

Defined in `rag/tests/retrieval/test_retrieval_metrics.py`. Reused by `test_legal_retrieval.py`.

All metrics are computed against a **gold dataset** — a fixed set of (query → list of relevant document source stems) pairs. Relevance is determined by case-insensitive substring match of the stem against `result.document_source`.

### IR Metrics

#### Hit Rate@K
```
Hit Rate@K = fraction of queries where ≥1 relevant document appears in the top-K results
```
Binary per query: 1.0 if any relevant doc is found, 0.0 if none. Averaged across all queries. The most forgiving metric — measures whether retrieval *finds anything useful*.

#### MRR@K (Mean Reciprocal Rank)
```
MRR@K = mean of (1 / rank_of_first_relevant_result) across queries
       = 1.0 if relevant at rank 1, 0.5 if rank 2, 0.33 if rank 3, ...
       = 0.0 if no relevant result in top-K
```
Measures how *highly ranked* the first relevant result is. A system that always puts the best answer at position 1 scores 1.0; one that buries it at position 5 scores 0.2.

#### Precision@K
```
Precision@K = (number of relevant results in top-K) / K
```
Measures the *density* of relevant results in the returned set. Low values are expected when K is large relative to the number of relevant documents per query.

#### Recall@K
```
Recall@K = (relevant results in top-K) / (total relevant documents for this query)
```
Measures coverage: what fraction of all known-relevant documents does the system surface. Recall@5 = 0.4 means on average 40% of relevant docs appear in top-5.

#### NDCG@K (Normalised Discounted Cumulative Gain)
```
DCG@K  = Σ rel_i / log2(i+2)   for i=0..K-1   (log2 penalises lower ranks)
IDCG@K = Σ 1    / log2(i+2)   for i=0..min(#relevant, K)-1  (ideal DCG)
NDCG@K = DCG@K / IDCG@K       (1.0 = perfect ranking, 0.0 = no relevant results)
```
Rewards systems that rank relevant documents *above* irrelevant ones. A relevant result at rank 1 contributes more than one at rank 5. Perfect NDCG = 1.0 means every relevant document is ranked before every irrelevant one.

### System Metrics

#### Mean Latency
Average time (ms) per query across the full gold dataset run. Measured with `time.perf_counter()` around `retriever.retrieve()`.

#### P95 Latency
The 95th-percentile query latency. One in twenty queries takes this long or longer. The test asserts P95 < 10 seconds. This catches slow outliers (cold cache, large result sets) that the mean would hide.

### Gold Datasets and Thresholds

| Dataset | File | Queries | K=5 thresholds |
|---------|------|---------|----------------|
| NeuralFlow AI corpus | `test_retrieval_metrics.py` `GOLD_DATASET` | 10 | Hit Rate ≥0.60, MRR ≥0.40, Precision ≥0.15, Recall ≥0.40, NDCG ≥0.40 |
| CUAD legal corpus | `test_legal_retrieval.py` `LEGAL_GOLD_DATASET` | 10 | Hit Rate ≥0.70, MRR ≥0.45, Precision ≥0.10 |

Legal thresholds are lower for Precision because 509 contracts with overlapping terminology naturally dilutes precision — many chunks look superficially relevant.

### Hybrid vs. semantic comparison test

`test_hybrid_beats_semantic_alone` (NeuralFlow) and `test_hybrid_not_worse_than_semantic` (legal) verify that adding BM25 via RRF does not hurt Hit Rate@5 by more than 10–15 percentage points vs. pure semantic search. RRF merging can sometimes demote a semantically strong match if it scores poorly on BM25, so a small tolerance is accepted.

---

## 9. Test Files Reference

| File | Subfolder | What it tests | Deps |
|------|-----------|--------------|------|
| `test_retrieval_metrics.py` | `retrieval/` | `Retriever` against NeuralFlow gold dataset; IR metric unit tests | PostgreSQL + Ollama (integration) |
| `test_legal_retrieval.py` | `retrieval/` | `Retriever` against CUAD legal gold dataset; corpus isolation | PostgreSQL + Ollama (integration) |
| `test_rag_agent.py` | `agent/` | End-to-end agent queries via `Retriever` | PostgreSQL + Ollama |
| `test_agent_flow.py` | `agent/` | Pydantic AI event stream debugging | PostgreSQL + Ollama |
| `test_hybrid_kg_retrieval.py` | `knowledge_graph/` | `HybridKGRetriever` unit + integration | Mocked / PostgreSQL + AGE + Ollama |
| `test_nl_query.py` | `knowledge_graph/` | NL→Cypher intent parsing + query builder | None / AGE (integration) |
| `test_pg_graph_store.py` | `knowledge_graph/` | `PgGraphStore` CRUD | Mocked |
| `test_age_graph_store.py` | `knowledge_graph/` | `AgeGraphStore` Cypher ops | Mocked / AGE (1 integration) |
| `test_postgres_store.py` | `storage/` | `PostgresHybridStore` connection + indexes | PostgreSQL |
| `test_mem0_store.py` | `storage/` | `Mem0Store` CRUD | PostgreSQL |
| `test_config.py` | `core/` | Settings loading, credential masking | None |
| `test_ingestion.py` | `core/` | `ChunkData`, `SearchResult` models | None |
| `test_cuad_ingestion.py` | `ingestion/` | CUAD ingestion pipeline | Mocked |
| `test_api.py` | `agent/` | FastAPI REST endpoints | Mocked |
| `test_mcp_server.py` | `agent/` | MCP server tools | Mocked |
| `test_raganything.py` | `experimental/` | RAG-Anything modal processors | Mocked |
| `test_pdf_question_generator.py` | `experimental/` | `PDFQuestionStore` | PostgreSQL |

### Running subsets

```bash
# All tests
python -m pytest rag/tests/ -v

# Only fast unit tests (no external deps)
python -m pytest rag/tests/core/ rag/tests/ingestion/ -v

# Only retrieval quality (requires PostgreSQL + Ollama)
python -m pytest rag/tests/retrieval/ -v --log-cli-level=INFO

# Skip all integration tests
python -m pytest rag/tests/ -m "not integration" -v

# Only integration tests
python -m pytest rag/tests/ -m integration -v

# Knowledge graph only
python -m pytest rag/tests/knowledge_graph/ -v
```
