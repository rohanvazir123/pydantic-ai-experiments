# PostgreSQL RAG Agent Development Instructions

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Core Principles](#core-principles)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [CLI Usage](#cli-usage)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Architecture](#architecture)
- [Common Issues](#common-issues)
- [Development Workflow](#development-workflow)
- [Quick Reference](#quick-reference)

---

## Project Overview

Agentic RAG system combining PostgreSQL/pgvector with Pydantic AI for intelligent document retrieval. Uses Docling for multi-format ingestion (PDF, DOCX, audio via Whisper ASR), async PostgreSQL operations, and hybrid search (vector + text with RRF). Built with Python 3.13, Ollama for local LLM/embeddings, and type-safe Pydantic models.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Validate configuration
python -m rag.main --validate

# Ingest documents
python -m rag.main --ingest --documents rag/documents

# Run tests
python -m pytest rag/tests/ -v
```

## Core Principles

1. **TYPE SAFETY IS NON-NEGOTIABLE**
   - All functions, methods, and variables MUST have type annotations
   - Use Pydantic models for all data structures
   - Import `Callable` from `collections.abc`, not lowercase `callable`

2. **KISS** (Keep It Simple, Stupid)
   - Prefer simple, readable solutions over clever abstractions
   - Trust PostgreSQL RRF - no manual score combination

3. **ASYNC ALL THE WAY**
   - All I/O operations MUST be async (PostgreSQL, embeddings, LLM calls)
   - Use `asyncio` for concurrent operations

---

## Project Structure

```
rag/
├── config/
│   └── settings.py              # Pydantic Settings configuration
├── ingestion/
│   ├── pipeline.py               # Document ingestion pipeline
│   ├── cuad_ingestion.py         # CUAD dataset ingestion (510 legal contracts)
│   ├── embedder.py               # Embedding generation (OpenAI-compatible)
│   ├── models.py                 # Data models (ChunkData, SearchResult, etc.)
│   └── chunkers/
│       └── docling.py            # Docling HybridChunker integration
├── storage/
│   └── vector_store/
│       └── postgres.py           # PostgresHybridStore (vector + text search via pgvector)
├── retrieval/
│   └── retriever.py              # Search orchestrator
├── agent/
│   ├── rag_agent.py              # Pydantic AI agent (search_knowledge_base + search_knowledge_graph tools)
│   └── prompts.py                # System prompts
├── api/
│   └── app.py                    # FastAPI REST API (GET /health, POST /v1/chat, /v1/chat/stream, /v1/ingest)
├── mcp/
│   └── server.py                 # MCP server (FastMCP, stdio transport)
├── knowledge_graph/
│   ├── __init__.py               # create_kg_store() factory (reads KG_BACKEND env var)
│   ├── pg_graph_store.py         # PgGraphStore: kg_entities + kg_relationships tables (Neon)
│   ├── age_graph_store.py        # AgeGraphStore: Apache AGE / Cypher (Docker port 5433)
│   └── cuad_kg_builder.py        # CuadKgBuilder: CUAD annotations → graph (509 contracts)
├── memory/
│   └── mem0_store.py             # Mem0Store (pgvector-backed user memory)
├── legal/
│   └── __init__.py               # Legal document ingestion and evaluation utilities
├── documents/                    # Sample documents for ingestion
│   └── legal/                    # CUAD contract Markdown files (git-ignored)
├── tests/                        # Test suite
│   ├── test_config.py            # Configuration tests (13, no deps)
│   ├── test_ingestion.py         # Ingestion model tests (14, no deps)
│   ├── test_postgres_store.py    # PostgreSQL connection & index tests (18)
│   ├── test_rag_agent.py         # RAG agent integration tests (25+)
│   ├── test_api.py               # FastAPI REST API tests (14, all mocked)
│   ├── test_mcp_server.py        # MCP server tests (21, all mocked)
│   ├── test_cuad_ingestion.py    # CUAD ingestion unit tests (34, all mocked)
│   ├── test_pg_graph_store.py    # PgGraphStore unit tests (40, no external deps)
│   ├── test_age_graph_store.py   # AgeGraphStore unit + 1 integration test (24 total)
│   └── test_legal_retrieval.py   # Legal retrieval tests (16; 4 integration)
└── main.py                       # CLI entry point

docker-compose.yml                # Apache AGE container (apache/age:latest, port 5433)
```

---

## Configuration

### Environment Variables (.env)

```bash
# PostgreSQL/Neon (pgvector)
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require

# LLM (Ollama local)
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama

# Embeddings (Ollama local)
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_DIMENSION=768
```

### PostgreSQL/Neon Setup

The database tables and indexes are created automatically by `PostgresHybridStore.initialize()`. To set up manually:

1. Enable the pgvector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. Tables are created automatically: `documents` and `chunks`

3. Indexes created automatically:
   - **IVFFlat vector index** on `chunks.embedding` for cosine similarity search
   - **GIN index** on `chunks.content_tsv` for full-text search
   - **B-tree indexes** on `chunks.document_id` and `documents.source`

---

## CLI Usage

### Validate Configuration
```bash
python -m rag.main --validate
```

### Ingest Documents
```bash
# Full ingestion (cleans existing data)
python -m rag.main --ingest --documents rag/documents

# Incremental ingestion (keeps existing data)
python -m rag.main --ingest --documents rag/documents --no-clean

# With custom chunking
python -m rag.main --ingest --documents rag/documents \
    --chunk-size 1000 \
    --chunk-overlap 200 \
    --max-tokens 512

# Verbose output
python -m rag.main --ingest --documents rag/documents --verbose
```

### Supported File Formats
- Text: `.md`, `.markdown`, `.txt`
- Documents: `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.html`
- Audio: `.mp3`, `.wav`, `.m4a`, `.flac` (requires `openai-whisper`)

---

## Testing

### Run All Tests
```bash
python -m pytest rag/tests/ -v
```

### Run Specific Test Categories

```bash
# Configuration tests (fast, no external deps)
python -m pytest rag/tests/test_config.py -v

# Ingestion model tests (fast, no external deps)
python -m pytest rag/tests/test_ingestion.py -v

# PostgreSQL connection & index tests (requires PostgreSQL/Neon)
python -m pytest rag/tests/test_postgres_store.py -v

# RAG agent integration tests (requires PostgreSQL + Ollama)
python -m pytest rag/tests/test_rag_agent.py -v
python -m pytest rag/tests/test_rag_agent.py -v --log-cli-level=INFO --tb=short # log.info
```

### Test Categories

| Test File | What It Tests | Requirements |
|-----------|--------------|--------------|
| `test_config.py` | Settings loading, credential masking | None |
| `test_ingestion.py` | Data models, chunking config validation | None |
| `test_postgres_store.py` | PostgreSQL connection, vector/text indexes | PostgreSQL/Neon |
| `test_rag_agent.py` | Retriever queries, agent integration | PostgreSQL + Ollama |
| `test_api.py` | FastAPI REST endpoints (chat, stream, ingest, health) | None (mocked) |
| `test_mcp_server.py` | MCP server tools (search, retrieve, ingest, health) | None (mocked) |
| `test_cuad_ingestion.py` | CUAD parsing, file extraction, eval pairs, pipeline | None (mocked) |
| `test_pg_graph_store.py` | PgGraphStore entity/relationship CRUD, search | None (all unit) |
| `test_age_graph_store.py` | AgeGraphStore Cypher ops, AGE integration | None / AGE (1 integration) |
| `test_legal_retrieval.py` | Legal retrieval quality on CUAD corpus | Neon + Ollama (4 integration) |

### Sample Test Queries (from test_rag_agent.py)

The tests query the ingested NeuralFlow AI documents:

```python
# Company information
"What does NeuralFlow AI do?"
"How many engineers work at the company?"

# Employee benefits
"What is the PTO policy?"
"What is the learning budget for employees?"

# Technology
"What technologies and tools does the company use?"
```

### Expected Test Results

After successful ingestion of `rag/documents/`:
- `test_config.py`: 13 tests pass
- `test_ingestion.py`: 14 tests pass
- `test_postgres_store.py`: 18 tests pass
- `test_rag_agent.py`: All tests pass (requires PostgreSQL with data + Ollama running)

---

## Code Quality

### Linting & Formatting
```bash
# Check and fix
ruff check --fix rag/

# Format
ruff format rag/
```

### Makefile Commands
```bash
make ruff    # Run ruff check --fix and format
```

---

## Architecture

### Two-Table Pattern

**`documents` table**: Full document metadata
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source TEXT NOT NULL UNIQUE,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

**`chunks` table**: Embedded chunks for search
```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768),  -- 768-dim for nomic-embed-text
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
```

### Hybrid Search (RRF)

The system combines vector and text search using Reciprocal Rank Fusion:

```python
# Vector search for semantic similarity
semantic_results = await store.semantic_search(query_embedding, count)

# Text search for keyword matching
text_results = await store.text_search(query, count)

# Merge with RRF (k=60)
merged = reciprocal_rank_fusion([semantic_results, text_results])
```

---

## Common Issues

### 1. "pgvector extension not found"
Enable the pgvector extension: `CREATE EXTENSION IF NOT EXISTS vector;`

### 2. "callable is not subscriptable"
Use `Callable` from `collections.abc`:
```python
from collections.abc import Callable
def func(callback: Callable | None = None): ...
```

### 3. Audio transcription fails
Audio transcription requires both FFmpeg (in PATH) and Whisper:
```bash
# Install FFmpeg (system-level) - must be in PATH
# Windows (Chocolatey): choco install ffmpeg
#   Default path: C:\ProgramData\chocolatey\lib\ffmpeg\tools\ffmpeg\bin
# Windows (WinGet): winget install ffmpeg
# macOS: brew install ffmpeg
# Linux: sudo apt install ffmpeg

# Verify FFmpeg is in PATH
ffmpeg -version

# Install Whisper
pip install openai-whisper
```
If dependencies are missing, audio files are stored with `[Error: Could not transcribe audio file ...]` placeholder.

### 4. Ollama connection refused
Start Ollama server:
```bash
ollama serve
```

### 5. Embedding dimension mismatch
Ensure `EMBEDDING_DIMENSION` matches your model:
- `nomic-embed-text`: 768
- `text-embedding-3-small`: 1536
- `text-embedding-ada-002`: 1536

---

## Development Workflow

1. **Make changes** to source files in `rag/`
2. **Run linting**: `ruff check --fix rag/ && ruff format rag/`
3. **Run tests**: `python -m pytest rag/tests/ -v`
4. **Test manually** if needed:
   ```python
   import asyncio
   from rag.retrieval.retriever import Retriever
   from rag.storage.vector_store.postgres import PostgresHybridStore

   async def test():
       store = PostgresHybridStore()
       retriever = Retriever(store=store)
       results = await retriever.retrieve("What does the company do?")
       for r in results:
           print(f"{r.document_title}: {r.content[:100]}...")
       await store.close()

   asyncio.run(test())
   ```

---

## Quick Reference

### Run the Agent Programmatically
```python
import asyncio
from rag.agent.rag_agent import agent

async def main():
    result = await agent.run("What services does NeuralFlow AI provide?")
    print(result.output)

asyncio.run(main())
```

### Search Without Agent
```python
import asyncio
from rag.retrieval.retriever import Retriever
from rag.storage.vector_store.postgres import PostgresHybridStore

async def search(query: str):
    store = PostgresHybridStore()
    retriever = Retriever(store=store)

    # Get search results
    results = await retriever.retrieve(query, match_count=5)

    # Or get formatted context for LLM
    context = await retriever.retrieve_as_context(query)

    await store.close()
    return results, context

asyncio.run(search("employee benefits"))
```
