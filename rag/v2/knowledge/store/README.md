# knowledge/store/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Two Databases](#two-databases)

---

## What This Is

The storage layer. Three distinct stores, each wrapping a different backend. All store classes are async and manage their own connection pools.

---

## Files

| File | Backend | Purpose |
|------|---------|---------|
| `vector.py` | PostgreSQL + pgvector | `PostgresHybridStore`: chunk upsert, semantic search (HNSW), text search (GIN), RRF hybrid, corpus-scoped queries with RLS |
| `graph.py` | Apache AGE (port 5433) | `AgeGraphStore`: per-corpus AGE graphs, `import_docling_graph()` from NetworkX DiGraph, MATCH queries, delete by document/corpus |
| `entity_index.py` | PostgreSQL (main DB) | `EntityIndex`: `kg_entity_index` shadow table — tsvector GIN + pgvector HNSW hybrid search for AGE entities (AGE has no index support) |
| `cache.py` | Redis | `RedisCache`: L2 embedding cache, search result cache, document fingerprint cache; `msgpack` serialisation |

---

## Two Databases

The store layer talks to **two separate PostgreSQL instances**:

| Instance | Container | Port | Used by |
|----------|-----------|------|---------|
| Main DB | `pgvector/pgvector:pg16` | 5432 | `vector.py`, `entity_index.py`, all other tables |
| AGE DB | `apache/age:latest` | 5433 | `graph.py` only |

They are separate because Apache AGE and pgvector are different PostgreSQL extensions that conflict when installed on the same instance.

Every connection to the AGE DB must run two setup statements before any Cypher:
```sql
LOAD 'age';
SET search_path = ag_catalog, "$user", public;
```
`graph.py` registers this as an asyncpg pool `init` callback and re-applies it on every `pool.acquire()` (AGE state is reset by `RESET ALL` on connection return).
