# Vector Store Implementation Guide

This guide documents the vector store architecture and PostgreSQL/pgvector implementation.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Available Stores](#2-available-stores)
3. [PostgreSQL/pgvector Implementation](#3-postgresqlpgvector-implementation)
   - [3.1 Database Schema](#31-database-schema)
   - [3.2 Search Operations](#32-search-operations)
     - [3.2.1 Metadata Filtering](#metadata-filtering)
   - [3.3 Settings](#33-settings)
   - [3.4 Setup Instructions](#34-setup-instructions)
4. [Database Schema Reference](#4-database-schema-reference)
   - [4.1 `documents` table](#41-documents-table)
   - [4.2 `chunks` table](#42-chunks-table)
   - [4.3 Indexes](#43-indexes)
   - [4.4 JSONB metadata conventions](#44-jsonb-metadata-conventions)
5. [Adding a New Datastore](#5-adding-a-new-datastore)
6. [Testing](#6-testing)
7. [Configuration](#7-configuration)

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
| `hybrid_search(query, query_embedding, match_count)` | RRF fusion of all three signals |
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
- RRF fusion across all three search signals

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

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
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

#### Hybrid Search (RRF over 3 signals)

All three searches run concurrently via `asyncio.gather`. Results are merged with Reciprocal Rank Fusion (k=60):

```
rrf_score(chunk) = Σ 1 / (k + rank_in_list)
                   across [semantic, fts, fuzzy]
```

Any leg that raises an exception is silently excluded from the merge.

---

#### Parameter Binding (`$1`, `$2`, ...)

asyncpg uses **positional parameters** — `$1`, `$2`, `$3`, ... — instead of named placeholders like `:name` or `%s`. Values are passed as a tuple after the query string and are bound server-side, preventing SQL injection.

```python
# Python call
rows = await conn.fetch(query, arg1, arg2, arg3)

# Inside the query
# $1 → arg1, $2 → arg2, $3 → arg3
```

Mapping for semantic search:

```python
query = """
    SELECT ..., 1 - (c.embedding <=> $1::vector) as similarity
    FROM chunks c JOIN documents d ON c.document_id = d.id
    ORDER BY c.embedding <=> $1::vector
    LIMIT $2
"""
await conn.fetch(query, query_embedding, match_count)
#                        ↑ $1              ↑ $2
```

Mapping for full-text search:

```python
query = """
    SELECT ..., ts_rank(c.content_tsv, plainto_tsquery('english', $1)) as similarity
    FROM chunks c JOIN documents d ON c.document_id = d.id
    WHERE c.content_tsv @@ plainto_tsquery('english', $1)
    ORDER BY ts_rank(c.content_tsv, plainto_tsquery('english', $1)) DESC
    LIMIT $2
"""
await conn.fetch(query, search_text, match_count)
#                        ↑ $1          ↑ $2
```

When metadata filters are applied, filter parameters are appended **after** the base parameters. For semantic search the filter params start at `$3`; for text/fuzzy search they start at `$3` as well (the base query already uses `$1` and `$2`).

---

#### Metadata Filtering

Metadata filters are built by `_build_filter_clause(metadata_filter, param_offset)` and appended to the base WHERE clause. Keys are **parameterized** (not inlined) to prevent injection — even the JSONB key name is passed as a `$N` argument.

**`MetadataFilter` fields:**

| Field | SQL fragment generated | Use case |
|-------|----------------------|----------|
| `metadata_eq` | `c.metadata->>$N = $N+1` | Exact match on a single JSONB key |
| `metadata_in` | `c.metadata->>$N = ANY($N+1::text[])` | One key, multiple allowed values |
| `metadata_gte` | `c.metadata->>$N >= $N+1` | Range lower bound — ISO 8601 dates or comparable text |
| `metadata_lte` | `c.metadata->>$N <= $N+1` | Range upper bound — ISO 8601 dates or comparable text |
| `document_source` | `d.source = $N` | Restrict to a single document by source path |
| `document_sources` | `d.source = ANY($N::text[])` | Restrict to a set of documents |
| `document_title` | `d.title = $N` | Restrict to a document by title |

> **Date comparison note:** `metadata` is JSONB; `->>'key'` extracts values as text. ISO 8601 strings (`"YYYY-MM-DD"`) sort lexicographically in chronological order, so text `>=` / `<=` gives correct date range semantics without casting.

> **Filtering is always post-JOIN.** All filter predicates are appended to a single `WHERE` clause that sits after `JOIN documents d ON c.document_id = d.id`. This is a structural requirement: document-level filters (`d.source`, `d.title`) reference columns that only exist on the `documents` table, so the join must happen before those predicates can be evaluated. In practice this is not a performance problem:
> - **Chunk-level predicates** (`c.metadata->>...`) are pushed down by the PostgreSQL planner to a scan on `chunks` before the join rows are assembled — so unmatched chunks are discarded early.
> - **Document-level predicates** (`d.source = $N`, `d.title = $N`) are cheap equality lookups on B-tree indexed columns and are evaluated after the join.
>
> If you need to filter to a small set of documents before doing a vector scan (e.g., tenant isolation at scale), the right approach is to add a `document_id` or tenant column directly to `chunks` and index it there — keeping the predicate entirely within the `chunks` scan and avoiding the join cost altogether.

**Example — join order with mixed chunk + document filter:**

```sql
-- MetadataFilter(metadata_eq={"doc_type": "policy"}, document_title="Employee Handbook")
--
-- Logical execution order (as the planner sees it):
--   1. Scan chunks      → apply c.metadata->>$3 = $4   (chunk-level, pushed down)
--   2. JOIN documents   → match c.document_id = d.id
--   3. Filter joined    → apply d.title = $5            (document-level, post-join)
--   4. ORDER BY / LIMIT → vector distance sort, return top $2

SELECT
    c.id            AS chunk_id,
    c.document_id,
    c.content,
    1 - (c.embedding <=> $1::vector) AS similarity,
    c.metadata,
    d.title         AS document_title,
    d.source        AS document_source
FROM chunks c
JOIN documents d ON c.document_id = d.id   -- join happens first in SQL syntax
WHERE c.metadata->>$3 = $4                 -- chunk-level: planner pushes to chunks scan
  AND d.title = $5                         -- document-level: evaluated after join
ORDER BY c.embedding <=> $1::vector
LIMIT $2;
-- $1 = query_embedding, $2 = match_count
-- $3 = "doc_type",      $4 = "policy"
-- $5 = "Employee Handbook"
```

The `WHERE` clause is always a single block after the `JOIN` line — there is no pre-join subquery. The planner decides independently which predicates to push down to each table's scan.

**Example — exact metadata match (`metadata_eq`):**

```python
# MetadataFilter(metadata_eq={"doc_type": "policy", "category": "hr"})
# _build_filter_clause generates:
clauses = [
    "c.metadata->>$3 = $4",   # key="doc_type", value="policy"
    "c.metadata->>$5 = $6",   # key="category", value="hr"
]
params = ["doc_type", "policy", "category", "hr"]

# Full semantic search query becomes:
query = """
    SELECT c.id, c.content, 1 - (c.embedding <=> $1::vector) as similarity, ...
    FROM chunks c JOIN documents d ON c.document_id = d.id
    WHERE c.metadata->>$3 = $4
      AND c.metadata->>$5 = $6
    ORDER BY c.embedding <=> $1::vector
    LIMIT $2
"""
await conn.fetch(query, query_embedding, match_count, "doc_type", "policy", "category", "hr")
#                        ↑ $1              ↑ $2         ↑ $3         ↑ $4      ↑ $5         ↑ $6
```

**Example — multi-value match (`metadata_in`):**

```python
# MetadataFilter(metadata_in={"status": ["active", "pending"]})
clauses = ["c.metadata->>$3 = ANY($4::text[])"]
params   = ["status", ["active", "pending"]]

query = """
    ...
    WHERE c.metadata->>$3 = ANY($4::text[])
    ORDER BY c.embedding <=> $1::vector
    LIMIT $2
"""
await conn.fetch(query, query_embedding, match_count, "status", ["active", "pending"])
```

**Example — restrict to a single document (`document_source`):**

```python
# MetadataFilter(document_source="rag/documents/legal/Amazon_2021.md")
clauses = ["d.source = $3"]
params  = ["rag/documents/legal/Amazon_2021.md"]

query = """
    ...
    FROM chunks c JOIN documents d ON c.document_id = d.id
    WHERE d.source = $3
    ORDER BY c.embedding <=> $1::vector
    LIMIT $2
"""
await conn.fetch(query, query_embedding, match_count, "rag/documents/legal/Amazon_2021.md")
```

**Example — restrict to a set of documents (`document_sources`):**

```python
# MetadataFilter(document_sources=["rag/documents/legal/Amazon_2021.md",
#                                   "rag/documents/legal/Google_2022.md"])
clauses = ["d.source = ANY($3::text[])"]
params  = [["rag/documents/legal/Amazon_2021.md", "rag/documents/legal/Google_2022.md"]]

await conn.fetch(query, query_embedding, match_count,
                 ["rag/documents/legal/Amazon_2021.md", "rag/documents/legal/Google_2022.md"])
#                  ↑ $3 (list passed as PostgreSQL array)
```

**Example — combined filters (metadata + document source):**

```python
# MetadataFilter(
#     metadata_eq={"clause_type": "termination"},
#     document_source="rag/documents/legal/Amazon_2021.md"
# )
# Generates params: ["clause_type", "termination", "rag/documents/legal/Amazon_2021.md"]
# Generates clauses:
#   c.metadata->>$3 = $4       -- key/value pair
#   d.source = $5              -- document filter

await conn.fetch(query, query_embedding, match_count,
                 "clause_type", "termination", "rag/documents/legal/Amazon_2021.md")
#                  ↑ $3            ↑ $4            ↑ $5
```

**Example — Q4 2024 by date range (`metadata_gte` + `metadata_lte`):**

```python
# MetadataFilter(
#     metadata_gte={"date": "2024-10-01"},
#     metadata_lte={"date": "2024-12-31"},
# )
# Generates two range clauses starting at $3:
clauses = [
    "c.metadata->>$3 >= $4",   # lower bound: date >= 2024-10-01
    "c.metadata->>$5 <= $6",   # upper bound: date <= 2024-12-31
]
params = ["date", "2024-10-01", "date", "2024-12-31"]

await conn.fetch(query, query_embedding, match_count,
                 "date", "2024-10-01", "date", "2024-12-31")
#                  ↑ $3    ↑ $4          ↑ $5   ↑ $6
```

> **Why keys are parameterized:** `c.metadata->>$3` passes the JSONB key name as a bound parameter rather than interpolating it into the SQL string. This prevents injection even if a caller supplies a key name like `'; DROP TABLE chunks; --`.

---

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

---

## 4. Database Schema Reference

This section is the authoritative column-level reference. The DDL is generated automatically by `PostgresHybridStore.initialize()` — do not run it manually unless rebuilding from scratch.

### 4.1 `documents` table

Stores one row per source document.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Primary key |
| `title` | `TEXT` | NOT NULL | — | Human-readable document name (used in citation `[Source: title]`) |
| `source` | `TEXT` | NOT NULL | — | Unique file path or URL (e.g. `rag/documents/legal/Amazon_2021.md`) |
| `content` | `TEXT` | — | — | Full document text (optional — large docs may omit to save space) |
| `metadata` | `JSONB` | — | `'{}'` | Arbitrary key-value pairs set at ingest time (see §4.4) |
| `created_at` | `TIMESTAMPTZ` | — | `NOW()` | Ingest timestamp |

**Unique constraint:** `source` — prevents duplicate ingestion of the same file.

**Cascade:** deleting a `documents` row automatically deletes all `chunks` rows with that `document_id` (ON DELETE CASCADE).

```sql
SELECT id, title, source, metadata, created_at
FROM documents
WHERE source = 'rag/documents/legal/Amazon_2021.md';
```

---

### 4.2 `chunks` table

Stores one row per text chunk with its embedding vector.

| Column | Type | Nullable | Default | Description |
|--------|------|----------|---------|-------------|
| `id` | `UUID` | NOT NULL | `gen_random_uuid()` | Primary key (used as `chunk_id` in `SearchResult`) |
| `document_id` | `UUID` | NOT NULL | — | FK → `documents.id` (cascades on delete) |
| `content` | `TEXT` | NOT NULL | — | Raw chunk text sent to embedder and returned to agent |
| `embedding` | `vector(768)` | — | — | Dense embedding vector; dimension set by `EMBEDDING_DIMENSION` (768 for `nomic-embed-text`) |
| `chunk_index` | `INTEGER` | NOT NULL | — | 0-based position of this chunk within its document |
| `metadata` | `JSONB` | — | `'{}'` | Per-chunk metadata (e.g. `{"page": 3, "clause_type": "termination"}`) |
| `token_count` | `INTEGER` | — | — | Approximate token count from the chunker |
| `created_at` | `TIMESTAMPTZ` | — | `NOW()` | Insert timestamp |
| `content_tsv` | `tsvector` | GENERATED | — | English tsvector of `content`, maintained automatically by PostgreSQL |

**`content_tsv`** is a generated stored column — it is always in sync with `content` and never written directly.

**Embedding dimension:** the `vector(768)` type is set once at table creation. Changing `EMBEDDING_DIMENSION` after the fact requires dropping and recreating the table and all embeddings.

```sql
-- Inspect a chunk and its parent document
SELECT
    c.id,
    c.chunk_index,
    c.token_count,
    c.metadata,
    left(c.content, 120) AS content_preview,
    d.title,
    d.source
FROM chunks c
JOIN documents d ON c.document_id = d.id
WHERE d.source = 'rag/documents/legal/Amazon_2021.md'
ORDER BY c.chunk_index;
```

---

### 4.3 Indexes

| Index name | Table | Type | Columns | Purpose |
|------------|-------|------|---------|---------|
| `chunks_embedding_idx` | `chunks` | IVFFlat | `embedding vector_cosine_ops` | ANN vector search (cosine distance) |
| `chunks_content_tsv_idx` | `chunks` | GIN | `content_tsv` | Full-text search (`@@` operator) |
| `chunks_content_trgm_idx` | `chunks` | GIN | `content gin_trgm_ops` | Trigram fuzzy search (`word_similarity`) |
| `chunks_document_id_idx` | `chunks` | B-tree | `document_id` | Efficient JOIN to `documents` |
| `documents_source_idx` | `documents` | B-tree | `source` | Lookup / dedup by file path |

**IVFFlat `lists` parameter:** defaults to `100` (good up to ~1 M rows). Tune with:
```
lists ≈ sqrt(total_rows)   -- rule of thumb
```

**Probes at query time:** `SET ivfflat.probes = 10` per connection. Higher probes = better recall, slower query. The store sets this in the pool `init` callback so every connection has it.

**Auto-reindex trigger:** after each `add()` the store checks if `total_chunks > 3 × chunks_at_last_build`. If true it issues `REINDEX INDEX CONCURRENTLY chunks_embedding_idx` to keep recall quality as the dataset grows.

```sql
-- Check index sizes
SELECT
    indexname,
    pg_size_pretty(pg_relation_size(indexname::regclass)) AS index_size
FROM pg_indexes
WHERE tablename IN ('documents', 'chunks')
ORDER BY pg_relation_size(indexname::regclass) DESC;
```

---

### 4.4 JSONB metadata conventions

Both tables have a `metadata JSONB` column. Keys are set by the ingestion pipeline and can be filtered at query time via `MetadataFilter`.

**`documents.metadata` — common keys:**

| Key | Type | Example | Set by |
|-----|------|---------|--------|
| `content_hash` | `string` | `"a3f2..."` | Pipeline (MD5 via `hashlib.md5()`, used for incremental re-ingest detection) |
| `file_type` | `string` | `"pdf"`, `"md"` | Pipeline |
| `page_count` | `number` | `42` | Docling converter |
| `word_count` | `number` | `8320` | Pipeline |

**`chunks.metadata` — common keys:**

| Key | Type | Example | Set by |
|-----|------|---------|--------|
| `page` | `string` | `"3"` | Docling chunker |
| `heading` | `string` | `"Section 4 — Termination"` | Docling chunker |
| `doc_type` | `string` | `"policy"`, `"contract"` | Caller at ingest |
| `category` | `string` | `"hr"`, `"legal"` | Caller at ingest |
| `clause_type` | `string` | `"termination"`, `"payment"` | KG extraction pipeline |
| `quarter` | `string` | `"Q4"` | Caller at ingest — fiscal/calendar quarter |
| `year` | `string` | `"2024"` | Caller at ingest — fiscal/calendar year |
| `date` | `string (ISO 8601)` | `"2024-12-31"` | Caller at ingest — publication or report date |
| `report_type` | `string` | `"earnings"`, `"guidance"` | Caller at ingest |

> **Why `date` is a string, not a `timestamp`:** JSONB stores arbitrary values; extracting with `->>'date'` returns text. ISO 8601 strings (`"YYYY-MM-DD"`) sort lexicographically in the correct chronological order, so text `>=` / `<=` comparisons give correct date range semantics without any casting.

**Temporal filtering examples:**

```sql
-- Q4 2024 by explicit quarter/year tags
SELECT c.id, left(c.content, 100)
FROM chunks c
WHERE c.metadata->>'quarter' = 'Q4'
  AND c.metadata->>'year' = '2024';

-- Q4 2024 by date range (ISO 8601 text comparison)
SELECT c.id, left(c.content, 100)
FROM chunks c
WHERE c.metadata->>'date' >= '2024-10-01'
  AND c.metadata->>'date' <= '2024-12-31';

-- All 2024 earnings reports (both quarter tag and date range approaches)
SELECT c.id, c.metadata->>'quarter' AS quarter, left(c.content, 80)
FROM chunks c
WHERE c.metadata->>'year' = '2024'
  AND c.metadata->>'report_type' = 'earnings'
ORDER BY c.metadata->>'date';

-- Q4 2024 via MetadataFilter (Python)
-- MetadataFilter(metadata_eq={"quarter": "Q4", "year": "2024"})

-- Q4 2024 via date range MetadataFilter (Python)
-- MetadataFilter(
--     metadata_gte={"date": "2024-10-01"},
--     metadata_lte={"date": "2024-12-31"},
-- )
```

**Ingesting with temporal metadata (Python):**

```python
from rag.ingestion.models import ChunkData

# Set quarter/year/date on each chunk at ingest time
chunk = ChunkData(
    content="Q4 2024 net income was $X billion...",
    index=0,
    start_char=0,
    end_char=200,
    metadata={
        "quarter": "Q4",
        "year": "2024",
        "date": "2024-12-31",    # ISO 8601 — enables date range filtering
        "report_type": "earnings",
    },
    embedding=[...],
)
```

**Querying JSONB directly in SQL:**

```sql
-- Find all chunks with clause_type = 'termination'
SELECT c.id, left(c.content, 100)
FROM chunks c
WHERE c.metadata->>'clause_type' = 'termination';

-- Find documents ingested as contracts
SELECT title, source
FROM documents
WHERE metadata->>'file_type' = 'pdf';

-- Count chunks per heading
SELECT c.metadata->>'heading' AS heading, count(*)
FROM chunks c
WHERE c.metadata->>'heading' IS NOT NULL
GROUP BY heading
ORDER BY count(*) DESC;
```

---

## 5. Adding a New Datastore

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
        semantic_results, text_results, fuzzy_results = await asyncio.gather(
            self.semantic_search(query_embedding, fetch_count),
            self.text_search(query, fetch_count),
            self.fuzzy_search(query, fetch_count),
            return_exceptions=True,
        )

        for attr in ("semantic_results", "text_results", "fuzzy_results"):
            if isinstance(locals()[attr], Exception):
                locals()[attr] = []

        return self._reciprocal_rank_fusion(
            [semantic_results, text_results, fuzzy_results]
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

## 6. Testing

### Run Store Tests

```bash
# PostgreSQL store tests
python -m pytest rag/tests/storage/test_postgres_store.py -v

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

## 7. Configuration

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

# Combined RRF over all three signals
results = await store.hybrid_search(query, query_embedding, 10)

await store.close()
```
