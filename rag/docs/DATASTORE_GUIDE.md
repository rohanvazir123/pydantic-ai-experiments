# Vector Store Implementation Guide

This guide documents the vector store architecture and PostgreSQL/pgvector implementation.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Available Stores](#2-available-stores)
3. [PostgreSQL/pgvector Implementation](#3-postgresqlpgvector-implementation)
4. [Adding a New Datastore](#4-adding-a-new-datastore)
5. [Testing](#5-testing)
6. [Configuration](#6-configuration)

---

## 1. Architecture Overview

The RAG system uses a pluggable vector store architecture. All stores implement a common interface defined in `rag/storage/vector_store/base.py`.

### Directory Structure

```
rag/storage/vector_store/
├── __init__.py          # Exports all stores
├── base.py              # VectorStore protocol (interface)
└── postgres.py          # PostgreSQL/pgvector implementation
```

### Core Interface (Protocol)

`base.py` defines a minimal `VectorStore` protocol for generic stores:

```python
class VectorStore(Protocol):
    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        ...

    def query(
        self,
        embedding: list[float],
        query_text: str,
        k: int,
    ) -> list[RetrievedChunk]:
        ...
```

> **Note:** `PostgresHybridStore` does **not** implement this protocol directly — it exposes its own richer async API (see below). The base protocol exists for simpler store implementations.

### Extended Interface (PostgresHybridStore)

| Method | Description |
|--------|-------------|
| `initialize()` | Establish connection pool, create tables/indexes |
| `close()` | Close connection pool |
| `save_document(title, source, content, metadata)` | Insert a document, return UUID |
| `add(chunks, document_id)` | Batch-insert chunks with embeddings |
| `semantic_search(query_embedding, match_count)` | pgvector cosine similarity search |
| `text_search(query, match_count)` | Full-text search via `tsvector` |
| `fuzzy_search(query, match_count)` | Trigram fuzzy search via `pg_trgm` |
| `bm25_search(query, match_count)` | BM25 search via `pg_search` (ParadeDB, optional) |
| `hybrid_search(query, query_embedding, match_count)` | RRF fusion of all four signals |
| `clean_collections()` | Delete all chunks and documents |
| `get_document_by_source(source)` | Fetch document dict by source path |
| `get_document_hash(source)` | Fetch `content_hash` from document metadata |
| `delete_document_and_chunks(source)` | Delete document + cascade-delete chunks |
| `get_all_document_sources()` | List all source paths |
| `get_chunk_count()` | Total number of chunks |
| `get_document_count()` | Total number of documents |

---

## 2. Available Stores

### PostgresHybridStore (PostgreSQL with pgvector)

**File:** `rag/storage/vector_store/postgres.py`

**Features:**
- pgvector extension for vector similarity search
- PostgreSQL tsvector for full-text search
- pg_trgm for fuzzy/trigram search (typo tolerance)
- pg_search (ParadeDB) for BM25 ranking (optional)
- RRF fusion across all four search signals
- Works with Supabase, or local PostgreSQL

**Usage:**
```python
from rag.storage.vector_store import PostgresHybridStore

store = PostgresHybridStore()
await store.initialize()
# ... use store ...
await store.close()
```

---

## 3. PostgreSQL/pgvector Implementation

### 3.1 Database Schema

#### Extensions

The following extensions are enabled automatically on `initialize()`:

| Extension | Purpose | Required |
|-----------|---------|----------|
| `vector` | pgvector — vector similarity search | Yes |
| `pg_trgm` | Trigram fuzzy matching | Yes |
| `pg_search` | ParadeDB BM25 full-text ranking | No (optional) |

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS pg_search;  -- optional (ParadeDB)
```

#### Tables

```sql
-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    source TEXT NOT NULL UNIQUE,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Chunks table with vector column
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(768),  -- dimension matches EMBEDDING_DIMENSION setting
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
```

#### Indexes

```sql
-- Vector similarity (IVFFlat, cosine distance)
CREATE INDEX chunks_embedding_idx ON chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Full-text search (GIN over generated tsvector column)
CREATE INDEX chunks_content_tsv_idx ON chunks USING GIN(content_tsv);

-- Trigram fuzzy search (GIN over raw content)
CREATE INDEX chunks_content_trgm_idx ON chunks USING GIN(content gin_trgm_ops);

-- B-tree indexes for joins / lookups
CREATE INDEX chunks_document_id_idx ON chunks(document_id);
CREATE INDEX documents_source_idx ON documents(source);

-- BM25 index (optional — requires pg_search / ParadeDB)
CREATE INDEX chunks_bm25_idx ON chunks
    USING bm25 (id, content) WITH (key_field='id');
```

> **IVFFlat auto-reindex:** After each `add()` call, the store checks if the total chunk count has grown beyond 3× the count at last index build time. If so, it issues `REINDEX INDEX CONCURRENTLY chunks_embedding_idx` automatically to maintain recall quality.

### 3.2 Search Operations

#### Semantic Search (pgvector cosine similarity)

```sql
SELECT
    c.id as chunk_id,
    c.document_id,
    c.content,
    1 - (c.embedding <=> $1::vector) as similarity,
    c.metadata,
    d.title as document_title,
    d.source as document_source
FROM chunks c
JOIN documents d ON c.document_id = d.id
ORDER BY c.embedding <=> $1::vector
LIMIT $2;
```

`ivfflat.probes` is set to `10` per connection to improve recall beyond the default of `1`.

#### Full-Text Search (tsvector / ts_rank)

```sql
SELECT
    c.id as chunk_id,
    c.document_id,
    c.content,
    ts_rank(c.content_tsv, plainto_tsquery('english', $1)) as similarity,
    c.metadata,
    d.title as document_title,
    d.source as document_source
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE c.content_tsv @@ plainto_tsquery('english', $1)
ORDER BY ts_rank(c.content_tsv, plainto_tsquery('english', $1)) DESC
LIMIT $2;
```

#### Fuzzy Search (pg_trgm word similarity)

```sql
SELECT
    c.id as chunk_id,
    c.document_id,
    c.content,
    word_similarity($1, c.content) as similarity,
    c.metadata,
    d.title as document_title,
    d.source as document_source
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE word_similarity($1, c.content) > 0.2
ORDER BY word_similarity($1, c.content) DESC
LIMIT $2;
```

Catches typos and partial-word matches that `plainto_tsquery` misses.

#### BM25 Search (pg_search / ParadeDB — optional)

```sql
SELECT
    c.id as chunk_id,
    c.document_id,
    c.content,
    paradedb.score(c.id) as similarity,
    c.metadata,
    d.title as document_title,
    d.source as document_source
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE c.id @@@ paradedb.match('content', $1)
ORDER BY paradedb.score(c.id) DESC
LIMIT $2;
```

Provides better relevance ranking than `ts_rank` via term-frequency / document-length normalization. Falls back gracefully to tsvector search when `pg_search` is not installed.

#### Hybrid Search (RRF over 4 signals)

All four searches run concurrently via `asyncio.gather`. Results are merged with Reciprocal Rank Fusion (k=60):

```
rrf_score(chunk) = Σ 1 / (k + rank_in_list)
                   across [semantic, fts, fuzzy, bm25]
```

Failed or unavailable search signals are silently excluded from the merge.

### 3.3 Settings

Added to `rag/config/settings.py`:

```python
# PostgreSQL connection
database_url: str = Field(default="", ...)

# Table names (validated: only [a-zA-Z_][a-zA-Z0-9_]* allowed)
postgres_table_documents: str = Field(default="documents", ...)
postgres_table_chunks: str = Field(default="chunks", ...)

# Connection pool
db_pool_min_size: int = Field(default=1, ...)
db_pool_max_size: int = Field(default=10, ...)
```

### 3.4 Setup Instructions

#### Local PostgreSQL (Default)

1. Install PostgreSQL 15+
2. Install pgvector:
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql-15-pgvector

   # macOS with Homebrew
   brew install pgvector

   # Windows (Chocolatey)
   choco install postgresql pgvector
   ```
3. Enable extensions (run as superuser):
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- bundled with PostgreSQL
   ```
4. Set `DATABASE_URL` in `.env`:
   ```bash
   DATABASE_URL=postgresql://postgres:password@localhost:5432/ragdb
   ```

#### Supabase

1. Create project at [supabase.com](https://supabase.com)
2. pgvector and pg_trgm are pre-enabled
3. Copy connection string from **Settings > Database** and add to `.env`:
   ```bash
   DATABASE_URL=postgresql://postgres:password@db.project.supabase.co:5432/postgres
   ```

---

## 4. Adding a New Datastore

### Step 1: Create Store File

Create `rag/storage/vector_store/<name>.py`:

```python
import asyncio
import logging
from typing import Any

from rag.config.settings import load_settings
from rag.ingestion.models import ChunkData, SearchResult

logger = logging.getLogger(__name__)


class <Name>HybridStore:
    def __init__(self):
        self.settings = load_settings()
        self._initialized = False

    async def initialize(self) -> None:
        if self._initialized:
            return
        # TODO: Connect, create tables/indexes
        self._initialized = True

    async def close(self) -> None:
        # TODO: Close connection
        self._initialized = False

    async def add(self, chunks: list[ChunkData], document_id: str) -> None:
        await self.initialize()
        # TODO: Batch-insert chunks with embeddings

    async def semantic_search(
        self, query_embedding: list[float], match_count: int | None = None
    ) -> list[SearchResult]:
        await self.initialize()
        if match_count is None:
            match_count = self.settings.default_match_count
        # TODO: Vector similarity query
        return []

    async def text_search(
        self, query: str, match_count: int | None = None
    ) -> list[SearchResult]:
        await self.initialize()
        if match_count is None:
            match_count = self.settings.default_match_count
        # TODO: Full-text query
        return []

    async def fuzzy_search(
        self, query: str, match_count: int | None = None
    ) -> list[SearchResult]:
        await self.initialize()
        if match_count is None:
            match_count = self.settings.default_match_count
        # TODO: Fuzzy/trigram query
        return []

    async def bm25_search(
        self, query: str, match_count: int | None = None
    ) -> list[SearchResult]:
        await self.initialize()
        if match_count is None:
            match_count = self.settings.default_match_count
        # TODO: BM25 query (optional — return [] if unsupported)
        return []

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        match_count: int | None = None,
    ) -> list[SearchResult]:
        await self.initialize()
        if match_count is None:
            match_count = self.settings.default_match_count

        fetch_count = match_count * 2
        semantic_results, text_results, fuzzy_results, bm25_results = await asyncio.gather(
            self.semantic_search(query_embedding, fetch_count),
            self.text_search(query, fetch_count),
            self.fuzzy_search(query, fetch_count),
            self.bm25_search(query, fetch_count),
            return_exceptions=True,
        )

        for attr in ("semantic_results", "text_results", "fuzzy_results", "bm25_results"):
            if isinstance(locals()[attr], Exception):
                locals()[attr] = []

        return self._reciprocal_rank_fusion(
            [semantic_results, text_results, fuzzy_results, bm25_results]
        )[:match_count]

    def _reciprocal_rank_fusion(
        self, search_results_list: list[list[SearchResult]], k: int = 60
    ) -> list[SearchResult]:
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        for results in search_results_list:
            for rank, result in enumerate(results):
                chunk_id = result.chunk_id
                rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
                chunk_map.setdefault(chunk_id, result)

        return [
            SearchResult(
                chunk_id=chunk_map[cid].chunk_id,
                document_id=chunk_map[cid].document_id,
                content=chunk_map[cid].content,
                similarity=score,
                metadata=chunk_map[cid].metadata,
                document_title=chunk_map[cid].document_title,
                document_source=chunk_map[cid].document_source,
            )
            for cid, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        ]

    async def save_document(
        self, title: str, source: str, content: str, metadata: dict[str, Any]
    ) -> str:
        await self.initialize()
        # TODO: Insert document, return UUID string
        return ""

    async def clean_collections(self) -> None:
        await self.initialize()
        # TODO: Truncate tables

    async def get_document_by_source(self, source: str) -> dict[str, Any] | None:
        await self.initialize()
        return None

    async def get_document_hash(self, source: str) -> str | None:
        doc = await self.get_document_by_source(source)
        if doc and "metadata" in doc:
            return doc["metadata"].get("content_hash")
        return None

    async def delete_document_and_chunks(self, source: str) -> bool:
        await self.initialize()
        return False

    async def get_all_document_sources(self) -> list[str]:
        await self.initialize()
        return []

    async def get_chunk_count(self) -> int:
        await self.initialize()
        return 0

    async def get_document_count(self) -> int:
        await self.initialize()
        return 0
```

### Step 2: Add Configuration

Update `rag/config/settings.py`:

```python
<name>_connection_string: str = Field(default="", description="<Name> connection string")
```

Update `.env`:

```bash
<NAME>_CONNECTION_STRING=...
```

### Step 3: Export Store

Update `rag/storage/vector_store/__init__.py`:

```python
from rag.storage.vector_store.<name> import <Name>HybridStore

__all__ = ["VectorStore", "PostgresHybridStore", "<Name>HybridStore"]
```

### Step 4: Add Dependencies

Update `pyproject.toml`:

```toml
dependencies = [
    "<name>-python-client>=x.x.x",
]
```

### Step 5: Write Tests

Create `rag/tests/test_<name>_store.py`:

```python
import pytest
from rag.storage.vector_store import <Name>HybridStore

@pytest.fixture
async def store():
    store = <Name>HybridStore()
    await store.initialize()
    yield store
    await store.close()

@pytest.mark.asyncio
async def test_connection(store):
    assert store._initialized

@pytest.mark.asyncio
async def test_save_and_retrieve_document(store):
    doc_id = await store.save_document(
        title="Test Doc",
        source="test.txt",
        content="Test content",
        metadata={"test": True},
    )
    assert doc_id

    doc = await store.get_document_by_source("test.txt")
    assert doc is not None
    assert doc["title"] == "Test Doc"

    await store.delete_document_and_chunks("test.txt")

@pytest.mark.asyncio
async def test_semantic_search(store):
    pass  # requires embeddings

@pytest.mark.asyncio
async def test_hybrid_search(store):
    pass  # requires embeddings
```

---

## 5. Testing

### Run Store Tests

```bash
# PostgreSQL store tests
python -m pytest rag/tests/test_postgres_store.py -v

# All tests
python -m pytest rag/tests/ -v
```

### Test PostgreSQL Store Standalone

```bash
python -m rag.storage.vector_store.postgres
```

Expected output (empty DB):
```
RAG PostgreSQL Store Module Test
============================================================
[Initializing PostgreSQL connection...]
  Connected successfully!
--- Database Stats ---
  Documents: 0
  Chunks: 0
--- Document Sources ---
[Skipping search test - no data]
============================================================
PostgreSQL store test completed successfully!
============================================================
```

### Integration Test

```python
import asyncio
from rag.storage.vector_store import PostgresHybridStore
from rag.ingestion.embedder import EmbeddingGenerator
from rag.ingestion.models import ChunkData

async def test_full_workflow():
    store = PostgresHybridStore()
    embedder = EmbeddingGenerator()
    await store.initialize()

    doc_id = await store.save_document(
        title="Test Document",
        source="test.pdf",
        content="This is a test document about machine learning.",
        metadata={"type": "test"},
    )
    print(f"Saved document: {doc_id}")

    chunk = ChunkData(
        content="Machine learning is a subset of artificial intelligence.",
        index=0,
        start_char=0,
        end_char=58,
        metadata={},
    )
    chunk.embedding = await embedder.embed_query(chunk.content)
    await store.add([chunk], doc_id)
    print("Stored chunk with embedding")

    query = "What is machine learning?"
    query_embedding = await embedder.embed_query(query)
    results = await store.hybrid_search(query, query_embedding, 5)
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  - {r.content[:50]}... (score: {r.similarity:.3f})")

    await store.delete_document_and_chunks("test.pdf")
    await store.close()

asyncio.run(test_full_workflow())
```

---

## 6. Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `POSTGRES_TABLE_DOCUMENTS` | Documents table name (default: `documents`) | No |
| `POSTGRES_TABLE_CHUNKS` | Chunks table name (default: `chunks`) | No |
| `DB_POOL_MIN_SIZE` | Min connections in pool (default: `1`) | No |
| `DB_POOL_MAX_SIZE` | Max connections in pool (default: `10`) | No |
| `EMBEDDING_DIMENSION` | Vector dimension — must match embedding model (default: `768`) | No |

Table names are validated at startup: only `[a-zA-Z_][a-zA-Z0-9_]*` is accepted to prevent SQL injection via settings.

### Quick Reference

```python
from rag.storage.vector_store import PostgresHybridStore

store = PostgresHybridStore()
await store.initialize()

doc_id = await store.save_document(title, source, content, metadata)
await store.add(chunks, doc_id)

# Individual search signals
semantic_results = await store.semantic_search(query_embedding, 10)
text_results     = await store.text_search(query, 10)
fuzzy_results    = await store.fuzzy_search(query, 10)
bm25_results     = await store.bm25_search(query, 10)  # requires pg_search

# Combined RRF over all four signals
results = await store.hybrid_search(query, query_embedding, 10)

await store.close()
```
