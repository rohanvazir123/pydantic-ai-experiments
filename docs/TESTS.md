# Test Suite Documentation

This document describes all available tests in the RAG system, how to run them, and what they verify.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Files Overview](#test-files-overview)
3. [Running Tests](#running-tests)
4. [Test Categories](#test-categories)
5. [Prerequisites](#prerequisites)
6. [Troubleshooting](#troubleshooting)

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
| `test_postgres_store.py` | 18 | PostgreSQL/Neon | PostgreSQL/pgvector store operations |
| `test_mongo_store.py` | 12 | MongoDB Atlas | MongoDB store and index verification |
| `test_rag_agent.py` | 25+ | PostgreSQL + Ollama | RAG retriever and agent queries |
| `test_agent_flow.py` | 3 | PostgreSQL + Ollama | Agent execution flow with Pydantic AI |
| `test_pdf_question_generator.py` | 23 | PostgreSQL + Ollama | PDF question generator with pgvector |
| `test_raganything.py` | 10+ | Ollama + raganything | Multimodal processors (tables, equations, images) |

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
- `test_settings_has_database_config` - MongoDB or PostgreSQL configured
- `test_settings_has_mongodb_database` - MongoDB database name set
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

**Requirements:** PostgreSQL/Neon with pgvector extension

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

#### MongoDB Store Tests
```bash
python -m pytest rag/tests/test_mongo_store.py -v
```

**Requirements:** MongoDB Atlas with vector and text search indexes

Tests:
- `TestMongoDBConnection` - Basic store initialization
- `TestMongoDBIndexVerification` - Index verification
  - `test_mongodb_connection` - Connection established
  - `test_vector_index_exists` - Vector search index exists
  - `test_vector_index_configuration` - Index has correct settings
  - `test_text_index_exists` - Text search index exists
  - `test_collections_exist` - Required collections exist
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

**Requirements:** PostgreSQL/Neon with pgvector + Ollama running

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

---

## Test Categories

### Unit Tests (Fast, No External Dependencies)
```bash
python -m pytest rag/tests/test_config.py rag/tests/test_ingestion.py -v
```

These tests run quickly (~1 second) and don't require any external services.

### Integration Tests (Require Database)
```bash
# PostgreSQL tests
python -m pytest rag/tests/test_postgres_store.py -v

# MongoDB tests
python -m pytest rag/tests/test_mongo_store.py -v
```

### End-to-End Tests (Require Database + LLM)
```bash
python -m pytest rag/tests/test_rag_agent.py rag/tests/test_agent_flow.py -v
```

These require:
1. PostgreSQL/Neon with pgvector and ingested data
2. Ollama running with llama3.2:3b and nomic-embed-text models

---

## Prerequisites

### For All Tests
```bash
pip install pytest pytest-asyncio
```

### For PostgreSQL Tests
1. PostgreSQL/Neon database with pgvector extension
2. `DATABASE_URL` in `.env`
3. Run ingestion to populate data:
   ```bash
   python -m rag.main --ingest --documents rag/documents
   ```

### For MongoDB Tests
1. MongoDB Atlas cluster with vector/text indexes
2. `MONGODB_URI` in `.env`
3. Create indexes in Atlas UI (see CLAUDE.md)

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

#### "Vector index not found" (MongoDB)
Create the vector index in MongoDB Atlas UI. See CLAUDE.md for instructions.

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

# Everything
python -m pytest rag/tests/ -v
```

---

## Expected Test Counts

| Test File | Expected Passed |
|-----------|-----------------|
| test_config.py | 13 |
| test_ingestion.py | 14 |
| test_postgres_store.py | 18 |
| test_mongo_store.py | 12 (if MongoDB configured) |
| test_rag_agent.py | 25+ |
| test_agent_flow.py | 3 |
| test_pdf_question_generator.py | 23 |
| test_raganything.py | 10+ (if raganything installed) |

**Total with PostgreSQL setup:** ~95+ tests
