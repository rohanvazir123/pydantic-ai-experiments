# Vector Store Implementation Guide

This guide documents the vector store architecture, explains how PostgreSQL/pgvector support was added, and provides instructions for implementing new datastores.

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
├── mongo.py             # MongoDB Atlas implementation
└── postgres.py          # PostgreSQL/pgvector implementation
```

### Core Interface (Protocol)

```python
class VectorStore(Protocol):
    """Protocol defining the vector store interface."""

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Store document chunks with their embeddings."""
        ...

    def query(
        self,
        embedding: list[float],
        query_text: str,
        k: int,
    ) -> list[RetrievedChunk]:
        """Query the vector store for similar documents."""
        ...
```

### Extended Interface (HybridStore)

Both MongoDB and PostgreSQL stores implement an extended interface with:
- `initialize()` - Establish connection
- `close()` - Close connection
- `add(chunks, document_id)` - Store chunks
- `save_document(title, source, content, metadata)` - Store document
- `semantic_search(query_embedding, match_count)` - Vector search
- `text_search(query, match_count)` - Full-text search
- `hybrid_search(query, query_embedding, match_count)` - Combined RRF search
- `clean_collections()` - Delete all data
- `get_document_by_source(source)` - Get document by path
- `delete_document_and_chunks(source)` - Delete document
- `get_all_document_sources()` - List all documents

---

## 2. Available Stores

### MongoHybridStore (MongoDB Atlas)

**File:** `rag/storage/vector_store/mongo.py`

**Features:**
- MongoDB Atlas Vector Search for semantic search
- MongoDB Atlas Search for full-text search
- RRF fusion for hybrid search
- Requires Atlas cluster with indexes

**Usage:**
```python
from rag.storage.vector_store import MongoHybridStore

store = MongoHybridStore()
await store.initialize()
# ... use store ...
await store.close()
```

### PostgresHybridStore (PostgreSQL/Neon with pgvector)

**File:** `rag/storage/vector_store/postgres.py`

**Features:**
- pgvector extension for vector similarity search
- PostgreSQL tsvector for full-text search
- RRF fusion for hybrid search
- Works with Neon, Supabase, or local PostgreSQL

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

This section documents how PostgreSQL support was added to the RAG system.

### 3.1 Changes Made

#### 1. Environment Configuration (`.env`)

Added PostgreSQL connection settings:

```bash
# PostgreSQL/Neon Configuration (pgvector)
DATABASE_URL=postgresql://user:pass@host/db?sslmode=require
POSTGRES_TABLE_DOCUMENTS=documents
POSTGRES_TABLE_CHUNKS=chunks
```

#### 2. Settings (`rag/config/settings.py`)

Added PostgreSQL settings to the Settings class:

```python
# PostgreSQL/Neon Configuration (pgvector)
database_url: str = Field(
    default="", description="PostgreSQL connection string (Neon/Supabase/local)"
)

postgres_table_documents: str = Field(
    default="documents", description="PostgreSQL table for source documents"
)

postgres_table_chunks: str = Field(
    default="chunks", description="PostgreSQL table for document chunks with embeddings"
)
```

#### 3. PostgresHybridStore (`rag/storage/vector_store/postgres.py`)

Created new store with:

**Dependencies:**
- `asyncpg` - Async PostgreSQL driver
- `pgvector` - pgvector Python client

**Database Schema:**

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
    embedding vector(768),  -- Dimension matches embedding model
    chunk_index INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- Indexes
CREATE INDEX chunks_embedding_idx ON chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX chunks_document_id_idx ON chunks(document_id);
CREATE INDEX chunks_content_tsv_idx ON chunks USING GIN(content_tsv);
CREATE INDEX documents_source_idx ON documents(source);
```

**Search Operations:**

Semantic Search (pgvector):
```sql
SELECT c.*, d.title, d.source,
       1 - (c.embedding <=> $1::vector) as similarity
FROM chunks c
JOIN documents d ON c.document_id = d.id
ORDER BY c.embedding <=> $1::vector
LIMIT $2;
```

Text Search (tsvector):
```sql
SELECT c.*, d.title, d.source,
       ts_rank(c.content_tsv, plainto_tsquery('english', $1)) as similarity
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE c.content_tsv @@ plainto_tsquery('english', $1)
ORDER BY ts_rank(...) DESC
LIMIT $2;
```

#### 4. Package Exports (`rag/storage/vector_store/__init__.py`)

Added export for PostgresHybridStore:

```python
from rag.storage.vector_store.postgres import PostgresHybridStore

__all__ = ["VectorStore", "MongoHybridStore", "PostgresHybridStore"]
```

#### 5. Dependencies (`pyproject.toml`)

Added required packages:

```toml
dependencies = [
    # ... existing deps ...
    "asyncpg>=0.30.0",
    "pgvector>=0.3.0",
]
```

### 3.2 Setup Instructions

#### Neon (Recommended for Serverless)

1. Create account at [neon.tech](https://neon.tech)
2. Create new project
3. Enable pgvector extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
4. Copy connection string to `.env`

#### Local PostgreSQL

1. Install PostgreSQL 15+
2. Install pgvector extension:
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql-15-pgvector

   # macOS with Homebrew
   brew install pgvector
   ```
3. Enable extension:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

#### Supabase

1. Create project at [supabase.com](https://supabase.com)
2. pgvector is pre-enabled
3. Copy connection string from Settings > Database

---

## 4. Adding a New Datastore

Follow these steps to add support for a new vector database.

### Step 1: Create Store File

Create `rag/storage/vector_store/<name>.py`:

```python
"""
<Name> vector store implementation.
"""

import asyncio
import logging
from typing import Any

from rag.config.settings import load_settings
from rag.ingestion.models import ChunkData, SearchResult

logger = logging.getLogger(__name__)


class <Name>HybridStore:
    """<Name> implementation with hybrid vector + text search."""

    def __init__(self):
        """Initialize connection."""
        self.settings = load_settings()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize connection and create tables/indexes."""
        if self._initialized:
            return
        # TODO: Connect to database
        # TODO: Create tables/indexes if needed
        self._initialized = True

    async def close(self) -> None:
        """Close connection."""
        # TODO: Close connection
        self._initialized = False

    async def add(self, chunks: list[ChunkData], document_id: str) -> None:
        """Store document chunks with embeddings."""
        await self.initialize()
        # TODO: Insert chunks with embeddings

    async def semantic_search(
        self, query_embedding: list[float], match_count: int | None = None
    ) -> list[SearchResult]:
        """Vector similarity search."""
        await self.initialize()
        if match_count is None:
            match_count = self.settings.default_match_count
        # TODO: Query by vector similarity
        return []

    async def text_search(
        self, query: str, match_count: int | None = None
    ) -> list[SearchResult]:
        """Full-text search."""
        await self.initialize()
        if match_count is None:
            match_count = self.settings.default_match_count
        # TODO: Query by text
        return []

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        match_count: int | None = None,
    ) -> list[SearchResult]:
        """Combined search using RRF."""
        await self.initialize()
        if match_count is None:
            match_count = self.settings.default_match_count

        # Run both searches concurrently
        semantic_results, text_results = await asyncio.gather(
            self.semantic_search(query_embedding, match_count * 2),
            self.text_search(query, match_count * 2),
            return_exceptions=True,
        )

        # Handle errors
        if isinstance(semantic_results, Exception):
            semantic_results = []
        if isinstance(text_results, Exception):
            text_results = []

        # Merge with RRF
        return self._reciprocal_rank_fusion(
            [semantic_results, text_results]
        )[:match_count]

    def _reciprocal_rank_fusion(
        self, search_results_list: list[list[SearchResult]], k: int = 60
    ) -> list[SearchResult]:
        """Merge ranked lists using RRF algorithm."""
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, SearchResult] = {}

        for results in search_results_list:
            for rank, result in enumerate(results):
                chunk_id = result.chunk_id
                rrf_score = 1.0 / (k + rank)

                if chunk_id in rrf_scores:
                    rrf_scores[chunk_id] += rrf_score
                else:
                    rrf_scores[chunk_id] = rrf_score
                    chunk_map[chunk_id] = result

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

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
            for cid, score in sorted_chunks
        ]

    async def save_document(
        self, title: str, source: str, content: str, metadata: dict[str, Any]
    ) -> str:
        """Save a document and return its ID."""
        await self.initialize()
        # TODO: Insert document and return ID
        return ""

    async def clean_collections(self) -> None:
        """Delete all data."""
        await self.initialize()
        # TODO: Truncate tables

    async def get_document_by_source(self, source: str) -> dict[str, Any] | None:
        """Get document by source path."""
        await self.initialize()
        # TODO: Query by source
        return None

    async def get_document_hash(self, source: str) -> str | None:
        """Get content hash for document."""
        doc = await self.get_document_by_source(source)
        if doc and "metadata" in doc:
            return doc["metadata"].get("content_hash")
        return None

    async def delete_document_and_chunks(self, source: str) -> bool:
        """Delete document and its chunks."""
        await self.initialize()
        # TODO: Delete by source
        return False

    async def get_all_document_sources(self) -> list[str]:
        """Get all document source paths."""
        await self.initialize()
        # TODO: Query all sources
        return []

    async def get_chunk_count(self) -> int:
        """Get total number of chunks."""
        await self.initialize()
        # TODO: Count chunks
        return 0

    async def get_document_count(self) -> int:
        """Get total number of documents."""
        await self.initialize()
        # TODO: Count documents
        return 0
```

### Step 2: Add Configuration

Update `rag/config/settings.py`:

```python
# <Name> Configuration
<name>_connection_string: str = Field(
    default="", description="<Name> connection string"
)
# Add other settings as needed
```

Update `.env`:

```bash
# <Name> Configuration
<NAME>_CONNECTION_STRING=...
```

### Step 3: Export Store

Update `rag/storage/vector_store/__init__.py`:

```python
from rag.storage.vector_store.<name> import <Name>HybridStore

__all__ = ["VectorStore", "MongoHybridStore", "PostgresHybridStore", "<Name>HybridStore"]
```

### Step 4: Add Dependencies

Update `pyproject.toml`:

```toml
dependencies = [
    # ... existing deps ...
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
    """Test database connection."""
    assert store._initialized

@pytest.mark.asyncio
async def test_save_and_retrieve_document(store):
    """Test document save and retrieval."""
    doc_id = await store.save_document(
        title="Test Doc",
        source="test.txt",
        content="Test content",
        metadata={"test": True}
    )
    assert doc_id

    doc = await store.get_document_by_source("test.txt")
    assert doc is not None
    assert doc["title"] == "Test Doc"

    # Cleanup
    await store.delete_document_and_chunks("test.txt")

@pytest.mark.asyncio
async def test_semantic_search(store):
    """Test vector similarity search."""
    # Requires embeddings to be generated
    pass

@pytest.mark.asyncio
async def test_hybrid_search(store):
    """Test combined search."""
    # Requires embeddings to be generated
    pass
```

---

## 5. Testing

### Run All Store Tests

```bash
# MongoDB store tests
python -m pytest rag/tests/test_mongo_store.py -v

# PostgreSQL store tests
python -m pytest rag/tests/test_postgres_store.py -v

# All tests
python -m pytest rag/tests/ -v
```

### Test PostgreSQL Store Standalone

```bash
python -m rag.storage.vector_store.postgres
```

Expected output:
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

    # Save document
    doc_id = await store.save_document(
        title="Test Document",
        source="test.pdf",
        content="This is a test document about machine learning.",
        metadata={"type": "test"}
    )
    print(f"Saved document: {doc_id}")

    # Create and store chunk
    chunk = ChunkData(
        content="Machine learning is a subset of artificial intelligence.",
        index=0,
        start_char=0,
        end_char=58,
        metadata={}
    )
    chunk.embedding = await embedder.embed_query(chunk.content)
    await store.add([chunk], doc_id)
    print("Stored chunk with embedding")

    # Search
    query = "What is machine learning?"
    query_embedding = await embedder.embed_query(query)

    results = await store.hybrid_search(query, query_embedding, 5)
    print(f"Found {len(results)} results")
    for r in results:
        print(f"  - {r.content[:50]}... (score: {r.similarity:.3f})")

    # Cleanup
    await store.delete_document_and_chunks("test.pdf")
    await store.close()

asyncio.run(test_full_workflow())
```

---

## 6. Configuration

### Environment Variables

| Variable | Description | Required For |
|----------|-------------|--------------|
| `MONGODB_URI` | MongoDB Atlas connection string | MongoDB |
| `MONGODB_DATABASE` | Database name | MongoDB |
| `DATABASE_URL` | PostgreSQL connection string | PostgreSQL |
| `POSTGRES_TABLE_DOCUMENTS` | Documents table name | PostgreSQL |
| `POSTGRES_TABLE_CHUNKS` | Chunks table name | PostgreSQL |
| `EMBEDDING_DIMENSION` | Vector dimension (768 for nomic-embed-text) | All |

### Switching Between Stores

The stores are independent - you can use either one:

```python
# MongoDB
from rag.storage.vector_store import MongoHybridStore
store = MongoHybridStore()

# PostgreSQL
from rag.storage.vector_store import PostgresHybridStore
store = PostgresHybridStore()
```

For the retriever and agent, update the store initialization:

```python
# In rag/retrieval/retriever.py or your code
from rag.storage.vector_store import PostgresHybridStore  # or MongoHybridStore

store = PostgresHybridStore()  # Change store here
retriever = Retriever(store=store)
```

---

## Quick Reference

### PostgreSQL Store

```python
from rag.storage.vector_store import PostgresHybridStore

# Initialize
store = PostgresHybridStore()
await store.initialize()

# Save document
doc_id = await store.save_document(title, source, content, metadata)

# Add chunks
await store.add(chunks, doc_id)

# Search
results = await store.hybrid_search(query, query_embedding, 10)

# Cleanup
await store.close()
```

### MongoDB Store

```python
from rag.storage.vector_store import MongoHybridStore

# Initialize
store = MongoHybridStore()
await store.initialize()

# Save document
doc_id = await store.save_document(title, source, content, metadata)

# Add chunks
await store.add(chunks, doc_id)

# Search
results = await store.hybrid_search(query, query_embedding, 10)

# Cleanup
await store.close()
```
