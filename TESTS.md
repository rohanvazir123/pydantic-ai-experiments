# Test Suite Documentation

This document describes all available tests in the RAG system, how to run them, and what they verify.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Files Overview](#test-files-overview)
3. [Running Tests](#running-tests)
4. [Test Categories](#test-categories)
5. [Retrieval Metrics](#retrieval-metrics)
6. [Prerequisites](#prerequisites)
7. [Troubleshooting](#troubleshooting)

> **Q&A Reference**: For a complete list of every integration test question and expected answer, see [TEST_QA_REFERENCE.md](TEST_QA_REFERENCE.md).

---

## Quick Start

```bash
# Run all tests
python -m pytest rag/tests/ -v

# Run specific test file
python -m pytest rag/tests/test_postgres_store.py -v

# Run with verbose output (shows print statements)
python -m pytest rag/tests/test_agent_flow.py -v -s

# Run tests matching a pattern
python -m pytest rag/tests/ -v -k "postgres"
```

---

## Test Files Overview

| Test File | Tests | Requirements | Description |
|-----------|-------|--------------|-------------|
| `test_config.py` | 13 | None | Configuration and settings validation |
| `test_ingestion.py` | 14 | None | Data models and chunking validation |
| `test_postgres_store.py` | 18 | PostgreSQL | PostgreSQL/pgvector store operations |
| `test_rag_agent.py` | 25+ | PostgreSQL + Ollama | RAG retriever and agent queries |
| `test_retrieval_metrics.py` | 28 | PostgreSQL + Ollama | Gold dataset evaluation: Hit Rate, MRR, NDCG, Precision, Recall |
| `test_agent_flow.py` | 3 | PostgreSQL + Ollama | Agent execution flow with Pydantic AI |
| `test_pdf_question_generator.py` | 23 | PostgreSQL + Ollama | PDF question generator with pgvector |
| `test_mem0_store.py` | 15+ | PostgreSQL + Ollama | Mem0 memory store with pgvector |
| `test_raganything.py` | 10+ | Ollama + raganything | Multimodal processors (tables, equations, images) |
| `test_pg_graph_store.py` | 40 | None (all unit) | PgGraphStore SQL tables, entity/relationship CRUD, search |
| `test_age_graph_store.py` | 24 | None (unit) / AGE (integration) | AgeGraphStore Cypher ops; 1 integration test skipped unless `AGE_DATABASE_URL` set |
| `test_legal_retrieval.py` | 16 | PostgreSQL + Ollama (4 integration) | Legal contract retrieval quality; 4 integration tests require live services |
| `test_api.py` | 14 | None (all mocked) | FastAPI REST API endpoint tests (chat, stream, ingest, health) |
| `test_mcp_server.py` | 21 | None (all mocked) | MCP server tool tests (search, retrieve, ingest, health) |
| `test_cuad_ingestion.py` | 34 | None (all mocked/unit) | CUAD dataset ingestion — parsing, file extraction, eval pairs, pipeline wiring |
| `test_hybrid_kg_retrieval.py` | 19 unit + 40 integration | None (unit) / PostgreSQL + AGE + Ollama (integration) | Hybrid KG + text retrieval: intent classifier, fusion, 100 HYBRID_KG_QUESTIONS |

---

## Running Tests

### All Tests

```bash
# Run entire test suite
python -m pytest rag/tests/ -v

# Run with coverage
python -m pytest rag/tests/ -v --cov=rag

# Run tests in parallel (requires pytest-xdist)
python -m pytest rag/tests/ -v -n auto
```

### By Category

#### Configuration Tests (No External Dependencies)
```bash
python -m pytest rag/tests/test_config.py -v
```

Tests:
- `test_load_settings_returns_settings_instance` - Settings loading works
- `test_settings_has_database_config` - PostgreSQL configured
- `test_settings_has_database_url` - DATABASE_URL set
- `test_settings_has_llm_config` - LLM settings configured
- `test_settings_has_embedding_config` - Embedding settings configured
- `test_settings_has_search_config` - Search parameters set
- `test_mask_credential_*` - Credential masking functions
- `test_default_*` - Default values validation

#### Data Model Tests (No External Dependencies)
```bash
python -m pytest rag/tests/test_ingestion.py -v
```

Tests:
- `TestChunkData` - ChunkData model creation and validation
- `TestChunkingConfig` - Chunking configuration validation
- `TestIngestionConfig` - Ingestion pipeline configuration
- `TestIngestionResult` - Ingestion result model
- `TestSearchResult` - Search result model

#### PostgreSQL/pgvector Store Tests
```bash
python -m pytest rag/tests/test_postgres_store.py -v
```

**Requirements:** PostgreSQL with pgvector extension

Tests:
- `TestPostgresConnection` - Basic store initialization
- `TestPostgresConnectionLive` - Live connection tests
  - `test_postgres_connection` - Connection established
  - `test_tables_exist` - Documents and chunks tables exist
  - `test_pgvector_extension_enabled` - pgvector extension active
  - `test_vector_index_exists` - IVF vector index created
  - `test_text_search_index_exists` - GIN text search index created
- `TestPostgresStoreOperations` - CRUD operations
  - `test_save_and_get_document` - Document save/retrieve/delete
  - `test_get_all_document_sources` - List all documents
  - `test_get_document_hash` - Content hash retrieval
- `TestPostgresSearchOperations` - Search functionality
  - `test_semantic_search_empty_results` - Vector search works
  - `test_text_search_empty_results` - Full-text search works
  - `test_hybrid_search_empty_results` - RRF hybrid search works
- `TestEmbeddingDimensionValidation` - Embedding dimension checks

#### RAG Agent Tests
```bash
python -m pytest rag/tests/test_rag_agent.py -v
```

**Requirements:** PostgreSQL with data + Ollama running

Tests:
- `TestRetrieverQueries` - Search and retrieval
  - `test_company_overview_query` - Company info retrieval
  - `test_team_structure_query` - Team/employee info
  - `test_benefits_query` - PTO/benefits search
  - `test_technology_stack_query` - Technology terms
  - `test_semantic_search` - Pure vector search
  - `test_text_search` - Pure text search
  - `test_retrieve_as_context` - Context formatting
- `TestRAGAgentTool` - Tool function tests
  - `test_search_tool_basic` - Knowledge base search
  - `test_search_tool_with_different_search_types` - All search types
- `TestRAGAgentIntegration` - Full agent tests
  - `test_agent_run_simple_query` - Basic agent query
  - `test_agent_run_specific_query` - Employee count query
  - `test_agent_run_benefits_query` - Learning budget query
  - `test_agent_run_pto_query` - PTO policy query
- `TestSearchResultQuality` - Result quality checks
  - `test_results_have_required_fields` - All fields present
  - `test_results_sorted_by_relevance` - Proper ranking
  - `test_no_duplicate_chunks` - No duplicates
  - `test_relevance_scoring` - Score comparison
- `TestAudioTranscription` - Audio file tests (if Whisper installed)

#### Agent Flow Tests
```bash
python -m pytest rag/tests/test_agent_flow.py -v -s
```

**Requirements:** PostgreSQL with data + Ollama running

Tests:
- `test_agent_flow_verbose` - Agent with debug output
- `test_agent_flow_with_tool_call` - Tool invocation flow
- `test_agent_flow_no_verbose` - Normal execution mode

**Note:** Use `-s` flag to see verbose Pydantic AI event output:

#### PDF Question Generator Tests
```bash
python -m pytest rag/tests/test_pdf_question_generator.py -v
```

**Requirements:** PostgreSQL with pgvector + Ollama running

Tests:
- `TestPDFQuestionStoreBasic` - Store initialization
  - `test_store_initialization` - Basic store setup
  - `test_store_has_table_names` - Table name configuration
- `TestPDFQuestionStoreConnection` - Database connection
  - `test_connection` - Connection established
  - `test_tables_created` - Tables exist (pdf_documents, pdf_questions, pdf_chunks)
  - `test_get_statistics` - Stats retrieval works
- `TestPDFQuestionStoreCRUD` - CRUD operations
  - `test_save_pdf_result` - Save PDF with questions/chunks
  - `test_get_pdf_document` - Retrieve PDF by path
  - `test_get_questions_for_pdf` - Get questions for PDF
  - `test_delete_pdf_document` - Delete PDF (cascades)
  - `test_replace_existing_document` - Update existing PDF
- `TestPDFQuestionStoreSearch` - Search with pgvector
  - `test_semantic_search_questions` - Vector search for questions
  - `test_text_search_questions` - Full-text search for questions
  - `test_hybrid_search_questions` - RRF hybrid search for questions
  - `test_semantic_search_chunks` - Vector search for chunks
  - `test_hybrid_search_chunks` - RRF hybrid search for chunks
  - `test_search_returns_pdf_metadata` - Results include PDF info
- `TestPDFQuestionGeneratorModels` - Data models
  - `test_processing_result_dataclass` - ProcessingResult model
  - `test_chunk_context_dataclass` - ChunkContext model
  - `test_format_chunks_as_context` - Context formatting
- `TestPDFQuestionStoreIndexes` - pgvector indexes
  - `test_questions_vector_index_exists` - IVF index on questions
  - `test_chunks_vector_index_exists` - IVF index on chunks
  - `test_questions_text_index_exists` - GIN index on questions
  - `test_chunks_text_index_exists` - GIN index on chunks
```bash
python -m pytest rag/tests/test_agent_flow.py::TestAgentFlow::test_agent_flow_verbose -v -s
```

This shows:
- `NODE #1: UserPromptNode`
- `NODE #2: ModelRequestNode` (PartStartEvent, PartDeltaEvent, etc.)
- `NODE #3: CallToolsNode` (FunctionToolCallEvent, FunctionToolResultEvent)
- End node

#### Mem0 Store Tests
```bash
python -m pytest rag/tests/test_mem0_store.py -v
```

**Requirements:** PostgreSQL with pgvector + Ollama running + MEM0_ENABLED=true

Tests:
- `TestMem0StoreBasic` - Store initialization
  - `test_store_initialization` - Basic store setup
  - `test_store_has_settings` - Settings access
  - `test_create_mem0_store_factory` - Factory function
- `TestMem0StoreDatabaseParsing` - DATABASE_URL parsing
  - `test_parse_database_url_basic` - Basic PostgreSQL URL parsing
  - `test_parse_database_url_with_sslmode` - URL with query parameters
  - `test_parse_database_url_missing_raises_error` - Missing URL error
- `TestMem0StoreEnabled` - Enabled/disabled behavior
  - `test_is_enabled_returns_setting` - Reflects MEM0_ENABLED setting
  - `test_disabled_add_returns_empty` - Add returns empty when disabled
  - `test_disabled_search_returns_empty` - Search returns empty when disabled
  - `test_disabled_get_all_returns_empty` - get_all returns empty when disabled
  - `test_disabled_get_context_returns_empty` - Context returns empty when disabled
- `TestMem0StoreIntegration` - Integration tests (require MEM0_ENABLED=true)
  - `test_add_memory` - Add memory to PostgreSQL
  - `test_get_all_memories` - Retrieve all memories
  - `test_search_memories` - Search with vector similarity
  - `test_get_context_string` - Formatted context for LLM
  - `test_delete_all_memories` - Delete user memories
- `TestMem0StoreContextFormatting` - Context string formatting
  - `test_empty_memories_returns_empty_string` - Empty context handling
  - `test_history_disabled_returns_empty` - History when disabled

**Note:** Integration tests are skipped when `MEM0_ENABLED` is not `true`.

#### RAG-Anything Multimodal Tests
```bash
python -m pytest rag/tests/test_raganything.py -v
```

**Requirements:** Ollama + raganything + lightrag libraries

Tests:
- Table processing with TableModalProcessor
- Equation processing with EquationModalProcessor
- Image processing with ImageModalProcessor
- LightRAG integration tests

#### Knowledge Graph Tests — PgGraphStore
```bash
python -m pytest rag/tests/test_pg_graph_store.py -v
```

**Requirements:** None (all 40 tests are pure unit tests — no external dependencies)

Tests cover:
- `TestPgGraphStoreInit` — store initialization and settings
- `TestPgGraphStoreEntityCRUD` — `add_entity()`, `get_entity()`, `delete_entity()`, deduplication via UNIQUE INDEX
- `TestPgGraphStoreRelationshipCRUD` — `add_relationship()`, `get_relationships()`, cascade deletes
- `TestPgGraphStoreSearch` — `search_entities()` ILIKE queries, `entity_type` filtering, `limit` parameter
- `TestPgGraphStoreSubgraph` — `get_entity_subgraph()` with depth parameter
- `TestPgGraphStoreClean` — `clean()` truncates both tables
- `TestPgGraphStoreSchema` — verifies SQL DDL for `kg_entities` and `kg_relationships` tables and indexes

#### Knowledge Graph Tests — AgeGraphStore
```bash
python -m pytest rag/tests/test_age_graph_store.py -v
```

**Requirements:** None for unit tests (23 tests). 1 integration test requires `AGE_DATABASE_URL` env var pointing to a live Apache AGE instance; skipped otherwise.

Tests cover:
- `TestAgeGraphStoreInit` — store initialization, graph name config
- `TestAgeGraphStoreConnContext` — `_conn()` context manager issues `LOAD 'age'` + `SET search_path` on every acquire
- `TestAgeGraphStoreCypher` — Cypher vertex/edge merge, `COALESCE` upsert pattern, double-quote string literals
- `TestAgeGraphStoreEntityOps` — `add_entity()`, `get_entity()`, entity type filtering
- `TestAgeGraphStoreRelationshipOps` — `add_relationship()`, edge label creation, relationship queries
- `TestAgeGraphStoreSearchOps` — `search_entities()` with optional `entity_type` filter
- `TestAgeGraphStoreIntegration` (1 test, integration) — full round-trip against live AGE instance

#### Legal Retrieval Tests
```bash
python -m pytest rag/tests/test_legal_retrieval.py -v
```

**Requirements:** 12 unit tests run without external deps. 4 integration tests require PostgreSQL (with ingested CUAD data) + Ollama running.

Tests cover:
- `TestLegalRetrievalUnit` (12 tests) — entity type mapping from CUAD questions, normalization logic, relationship type resolution, builder config validation
- `TestLegalRetrievalIntegration` (4 tests, skipped without live services):
  - `test_kg_entity_search` — search `kg_entities` for known CUAD party names
  - `test_kg_relationship_query` — verify `kg_relationships` links party to contract
  - `test_agent_kg_tool` — agent calls `search_knowledge_graph` for a legal question
  - `test_hybrid_rag_kg` — combined RAG + KG retrieval on a contract clause query

#### REST API Tests
```bash
python -m pytest rag/tests/test_api.py -v
```

**Requirements:** None — all external dependencies (DB, LLM, ingestion pipeline) are mocked.

Tests cover (`TestChatEndpoint`, `TestChatStreamEndpoint`, `TestIngestEndpoint`, `TestHealthEndpoint`):
- `test_chat_returns_answer` — POST /v1/chat returns answer from `traced_agent_run`
- `test_chat_propagates_user_id` / `test_chat_propagates_session_id` — optional fields forwarded
- `test_chat_error_handling` — 500 response on agent failure
- `test_chat_stream_*` — POST /v1/chat/stream returns SSE tokens, handles errors
- `test_ingest_*` — POST /v1/ingest triggers pipeline, returns stats, handles validation
- `test_health_*` — GET /health checks DB + embedding + LLM connectivity

#### MCP Server Tests
```bash
python -m pytest rag/tests/test_mcp_server.py -v
```

**Requirements:** None — all external dependencies are mocked.

Tests cover (`TestSearch`, `TestRetrieve`, `TestIngest`, `TestHealth`):
- `TestSearch` — `search()` tool: calls `traced_agent_run`, propagates `user_id`/`session_id`, handles errors
- `TestRetrieve` — `retrieve()` tool: calls `retrieve_as_context`, passes `match_count`/`search_type`, error recovery
- `TestIngest` — `ingest()` tool: calls ingestion pipeline, optional params, handles failures
- `TestHealth` — `health()` tool: connectivity checks for DB, embedding API, LLM

#### CUAD Ingestion Tests
```bash
python -m pytest rag/tests/test_cuad_ingestion.py -v
```

**Requirements:** None — filesystem I/O uses `tmp_path`; pipeline is mocked.

Tests cover (`TestExtractQuestionType`, `TestExtractContractType`, `TestSafeFilename`, `TestLoadCuad`, `TestWriteContractFiles`, `TestSaveEvalPairs`, `TestRunIngestion`):
- `TestExtractQuestionType` — maps CUAD question strings to entity type labels
- `TestExtractContractType` — extracts contract type from CUAD title strings
- `TestSafeFilename` — path sanitization for contract filenames
- `TestLoadCuad` — parses CUAD JSON, validates `CuadContract` / `CuadQA` dataclasses
- `TestWriteContractFiles` — writes 510 contract Markdown files to `rag/documents/legal/`
- `TestSaveEvalPairs` — saves Q&A evaluation pairs to `rag/legal/cuad_eval.json`
- `TestRunIngestion` — full pipeline mock: load → write → ingest with `--limit`, `--dry-run`, `--no-clean`

#### Hybrid KG Retrieval Tests
```bash
# Unit tests (mocked — no external deps):
python -m pytest rag/tests/test_hybrid_kg_retrieval.py -v -k "not integration"

# Integration tests + record all 100 answers for review:
python -m pytest rag/tests/test_hybrid_kg_retrieval.py -m integration --record-answers -v
```

**Requirements:** Unit tests: none. Integration tests: PostgreSQL (port 5434) + Apache AGE (port 5433) + Ollama.

Tests cover:

**`TestIntentClassifier`** (9 tests) — verifies `IntentClassifier.classify()` routing:
- `test_analytical_query_returns_structured` — "how many contracts" → STRUCTURED
- `test_count_query_returns_structured` — "distribution of" → STRUCTURED
- `test_average_query_returns_structured` — "average number" → STRUCTURED
- `test_clause_text_with_count_returns_hybrid` — "how many clauses say X" → HYBRID (text needed)
- `test_default_returns_hybrid` — default for clause questions → HYBRID
- `test_needs_semantic_*` / `test_needs_structured_*` — path activation flags

**`TestFuse`** (4 tests) — verifies `_fuse()` context assembly:
- `test_fuse_with_both_paths` — KG facts section + text passages section both present
- `test_fuse_kg_only` — structured intent produces KG section only
- `test_fuse_text_only` — semantic intent produces passages section only
- `test_fuse_empty_returns_no_results_message` — graceful empty state

**`TestHybridKGRetriever`** (6 tests) — verifies `HybridKGRetriever.retrieve()`:
- `test_hybrid_intent_runs_both_paths` — asyncio.gather runs semantic + structured
- `test_structured_intent_skips_semantic_path` — retriever.retrieve NOT called
- `test_kg_failure_degrades_gracefully` — AGE connection error → semantic path still returns
- `test_fused_context_not_empty` — non-empty context when results exist
- `test_explicit_intent_override` — intent kwarg bypasses classifier
- `test_relationship_facts_included` — relationships appear in fused context

**Integration tests** (40 tests, `@pytest.mark.integration`):
- `test_party_contract_questions[1-15]` — Party & Contract Relationship queries
- `test_analytical_questions[89-100]` — NL-to-Cypher / Analytical queries
- `test_multi_hop_graph_questions[76-88]` — Cross-Domain / Multi-Hop Graph queries

**Answer recording** — pass `--record-answers` to save all results to `docs/qa_results/hybrid_kg_results.json` for manual review.

---

## Test Categories

### Unit Tests (Fast, No External Dependencies)
```bash
python -m pytest rag/tests/test_config.py rag/tests/test_ingestion.py \
    rag/tests/test_pg_graph_store.py rag/tests/test_age_graph_store.py \
    rag/tests/test_legal_retrieval.py rag/tests/test_api.py \
    rag/tests/test_mcp_server.py rag/tests/test_cuad_ingestion.py \
    rag/tests/test_hybrid_kg_retrieval.py \
    -v -k "not Integration"
```

These tests run quickly and don't require any external services. Includes the 40 PgGraphStore tests, 23 AgeGraphStore unit tests, 12 legal retrieval unit tests, 14 REST API tests, 21 MCP server tests, and 34 CUAD ingestion tests.

### Integration Tests (Require Database)
```bash
# PostgreSQL/vector store tests
python -m pytest rag/tests/test_postgres_store.py -v

# AGE graph store integration (requires AGE_DATABASE_URL)
python -m pytest rag/tests/test_age_graph_store.py -v -k "Integration"

# Legal retrieval integration (requires PostgreSQL + Ollama)
python -m pytest rag/tests/test_legal_retrieval.py -v -k "Integration"
```

### End-to-End Tests (Require Database + LLM)
```bash
python -m pytest rag/tests/test_rag_agent.py rag/tests/test_agent_flow.py -v
```

These require:
1. PostgreSQL with pgvector and ingested data
2. Ollama running with llama3.2:3b and nomic-embed-text models

### Knowledge Graph Tests (All Backends)
```bash
# PgGraphStore — no deps
python -m pytest rag/tests/test_pg_graph_store.py -v

# AgeGraphStore — unit tests only
python -m pytest rag/tests/test_age_graph_store.py -v -k "not Integration"

# AgeGraphStore — with live AGE instance
AGE_DATABASE_URL=postgresql://age_user:age_pass@localhost:5433/legal_graph \
    python -m pytest rag/tests/test_age_graph_store.py -v

# Legal retrieval
python -m pytest rag/tests/test_legal_retrieval.py -v

# Hybrid KG retrieval (unit only, no external deps):
python -m pytest rag/tests/test_hybrid_kg_retrieval.py -v -k "not integration"
```

#### Retrieval Metrics Tests
```bash
python -m pytest rag/tests/test_retrieval_metrics.py -v --log-cli-level=INFO
```

**Requirements:** PostgreSQL with ingested NeuralFlow AI documents + Ollama running

Two test classes:

**`TestMetricFunctions`** (19 unit tests, no external dependencies) — verifies the
metric math in isolation before any DB calls are made.

**`TestRetrievalMetrics`** (7 integration tests) — runs all 10 gold queries through
the live retriever and asserts minimum quality thresholds.

---

## Retrieval Metrics

### Concepts

RAG retrieval quality is measured by comparing the ranked result list for each query
against a **gold dataset** — a curated list of (query, relevant document sources) pairs.
A result is marked **relevant** if its `document_source` path contains any of the
expected document filename stems (case-insensitive match).

### Metrics at K

All ranking metrics are computed at a cutoff K (evaluated at K=1, 3, 5):

| Metric | Formula | What it measures |
|--------|---------|-----------------|
| **Hit Rate@K** | `mean( any(rel_i) for i≤K )` | Fraction of queries where at least one relevant doc appears in the top-K results. The most practical single metric — "did the system find something useful?" |
| **MRR@K** | `mean( 1/rank_first_relevant )` | Mean Reciprocal Rank — rewards finding the first relevant result as early as possible. Score of 1.0 means first result is always relevant; 0.5 means first relevant doc is at rank 2 on average. |
| **Precision@K** | `mean( relevant_in_top_K / K )` | How much of the returned list is relevant? Penalises returning irrelevant results. |
| **Recall@K** | `mean( relevant_in_top_K / total_relevant )` | What fraction of all relevant documents for a query were found? Limited by how many relevant docs exist (e.g. a query with only 1 relevant doc can achieve Recall=1.0 with a single hit). |
| **NDCG@K** | `mean( DCG@K / IDCG@K )` | Normalised Discounted Cumulative Gain — like Precision@K but position-aware: a relevant result at rank 1 contributes more than the same result at rank 5. Ideal ranking = 1.0. |

### System Metrics

| Metric | Definition |
|--------|-----------|
| **Mean latency** | Average wall-clock time per query in milliseconds, measured end-to-end including embedding and DB round-trip. |
| **P95 latency** | 95th-percentile query latency — worst-case performance for 95% of queries. Threshold: < 10 000 ms. |

### NDCG Calculation Detail

```
DCG@K  = Σ rel_i / log2(i + 2)        for i = 0 … K-1
IDCG@K = Σ 1    / log2(i + 2)        for i = 0 … min(#relevant, K)-1
NDCG@K = DCG@K / IDCG@K

rel_i ∈ {0, 1}  (binary relevance)
```

Dividing by log2(i+2) gives position 1 a weight of 1.0, position 2 a weight of 0.63,
position 5 a weight of 0.39 — relevance at the top counts much more.

### Gold Dataset

10 queries grounded in the NeuralFlow AI document corpus:

| # | Query | Expected sources |
|---|-------|-----------------|
| 1 | What does NeuralFlow AI do? | company-overview, mission-and-goals |
| 2 | What is the PTO policy? | team-handbook |
| 3 | What is the learning budget for employees? | team-handbook |
| 4 | What technologies and architecture does the platform use? | technical-architecture-guide |
| 5 | What is the company mission and vision? | mission-and-goals |
| 6 | GlobalFinance Corp loan processing success story | client-review-globalfinance, Recording4 |
| 7 | How many employees work at NeuralFlow AI? | company-overview, team-handbook |
| 8 | What is DocFlow AI and how does it process documents? | Recording2 |
| 9 | Q4 2024 business results and performance review | q4-2024-business-review |
| 10 | implementation approach and playbook | implementation-playbook |

Entries 6 and 8 (audio Recordings) are automatically scored 0 for retrieval if Whisper
is not installed and audio files were not transcribed during ingestion.

### Minimum Thresholds (K=5, hybrid search)

| Metric | Threshold |
|--------|-----------|
| Hit Rate@5 | ≥ 0.60 |
| MRR@5 | ≥ 0.40 |
| Precision@5 | ≥ 0.15 |
| Recall@5 | ≥ 0.40 |
| NDCG@5 | ≥ 0.40 |
| P95 latency | < 10 000 ms |

### Reading the Metrics Table

Running with `--log-cli-level=INFO` prints a table like:

```
=================================================================
  RETRIEVAL METRICS — hybrid search, NeuralFlow AI corpus
=================================================================
  Metric               K=1       K=3       K=5
---------------------------------------------------------
  HIT_RATE@K         0.700     0.800     0.900
  MRR@K              0.700     0.700     0.700
  PRECISION@K        0.700     0.367     0.260
  RECALL@K           0.350     0.533     0.600
  NDCG@K             0.700     0.603     0.563
---------------------------------------------------------
  Mean latency                              850ms
  P95  latency                             1420ms
=================================================================
```

Followed by a per-query breakdown showing whether each query hit (✓/✗),
its reciprocal rank, and its latency.

---

## Prerequisites

### For All Tests
```bash
pip install pytest pytest-asyncio
```

### For PostgreSQL Tests
1. PostgreSQL database with pgvector extension
2. `DATABASE_URL` in `.env`
3. Run ingestion to populate data:
   ```bash
   python -m rag.main --ingest --documents rag/documents
   ```

### For Agent Tests
1. Ollama running locally:
   ```bash
   ollama serve
   ```
2. Required models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text:latest
   ```

### For Mem0 Tests
1. PostgreSQL with pgvector extension
2. `DATABASE_URL` in `.env`
3. `MEM0_ENABLED=true` in `.env` (for integration tests)
4. Install mem0:
   ```bash
   pip install mem0ai
   ```

### For RAG-Anything Tests
```bash
pip install raganything lightrag
```

---

## Troubleshooting

### Common Issues

#### "No module named 'asyncpg'"
```bash
pip install asyncpg pgvector
```

#### "unknown type: public.vector"
The pgvector extension isn't enabled. Run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

#### "Expected at least one result"
Database is empty. Run ingestion first:
```bash
python -m rag.main --ingest --documents rag/documents
```

#### "Connection refused" (Ollama)
Start Ollama server:
```bash
ollama serve
```

#### Tests timing out
Increase timeout:
```bash
python -m pytest rag/tests/ -v --timeout=300
```

### Verbose Debug Output

For detailed output during test execution:
```bash
# Show print statements
python -m pytest rag/tests/test_agent_flow.py -v -s

# Show logging
python -m pytest rag/tests/test_rag_agent.py -v --log-cli-level=INFO

# Both
python -m pytest rag/tests/ -v -s --log-cli-level=INFO
```

---

## Test Summary Commands

```bash
# Quick validation (no external deps)
python -m pytest rag/tests/test_config.py rag/tests/test_ingestion.py -v

# PostgreSQL store validation
python -m pytest rag/tests/test_postgres_store.py -v

# Full RAG system test
python -m pytest rag/tests/test_rag_agent.py -v

# Agent flow debugging
python -m pytest rag/tests/test_agent_flow.py -v -s

# Mem0 memory store tests
python -m pytest rag/tests/test_mem0_store.py -v

# Knowledge graph — PgGraphStore (no deps, all unit)
python -m pytest rag/tests/test_pg_graph_store.py -v

# Knowledge graph — AgeGraphStore (unit only, skip integration)
python -m pytest rag/tests/test_age_graph_store.py -v -k "not Integration"

# Legal retrieval
python -m pytest rag/tests/test_legal_retrieval.py -v

# REST API (all mocked, no external deps)
python -m pytest rag/tests/test_api.py -v

# MCP server (all mocked, no external deps)
python -m pytest rag/tests/test_mcp_server.py -v

# CUAD ingestion (all mocked, no external deps)
python -m pytest rag/tests/test_cuad_ingestion.py -v

# Hybrid KG retrieval (unit tests, no external deps)
python -m pytest rag/tests/test_hybrid_kg_retrieval.py -v -k "not integration"

# Everything
python -m pytest rag/tests/ -v
```

---

## Expected Test Counts

| Test File | Expected Passed | Skipped | Notes |
|-----------|-----------------|---------|-------|
| `test_config.py` | 13 | 0 | |
| `test_ingestion.py` | 14 | 0 | |
| `test_postgres_store.py` | 18 | 0 | Requires PostgreSQL |
| `test_rag_agent.py` | 25+ | 0 | Requires PostgreSQL + Ollama |
| `test_retrieval_metrics.py` | 28 | 0 | 19 unit + 9 integration |
| `test_agent_flow.py` | 3 | 0 | Requires PostgreSQL + Ollama |
| `test_pdf_question_generator.py` | 23 | 0 | Requires PostgreSQL + Ollama |
| `test_mem0_store.py` | 15+ | varies | Integration tests require `MEM0_ENABLED=true` |
| `test_raganything.py` | 10+ | 0 | Requires raganything + lightrag installed |
| `test_pg_graph_store.py` | 40 | 0 | All unit, no external deps |
| `test_age_graph_store.py` | 23 unit + 1 integration | 1 (unless `AGE_DATABASE_URL` set) | |
| `test_legal_retrieval.py` | 16 | 4 (unless PostgreSQL + Ollama live) | 12 unit always pass |
| `test_api.py` | 14 | 0 | All unit/mocked, no external deps |
| `test_mcp_server.py` | 21 | 0 | All unit/mocked, no external deps |
| `test_cuad_ingestion.py` | 34 | 0 | All unit/mocked, no external deps |
| `test_hybrid_kg_retrieval.py` | 19 unit passed | 40 (unless PostgreSQL + AGE + Ollama live) | `--record-answers` saves docs/qa_results/hybrid_kg_results.json |

**Current totals (full suite, live services):** ~385 passed, 52 skipped, 0 failed

The 52 skipped tests are all integration tests that require live services not always available:
- Mem0 integration tests (require `MEM0_ENABLED=true`)
- AGE graph integration test (requires `AGE_DATABASE_URL`)
- Legal retrieval integration tests (require PostgreSQL + Ollama)
