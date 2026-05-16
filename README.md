# pydantic-ai-experiments

Agentic RAG system combining PostgreSQL/pgvector with a legal knowledge graph built on Apache AGE.
Contracts are ingested, chunked, embedded, and stored for hybrid (vector + BM25) retrieval.
A parallel knowledge graph pipeline extracts entities, relationships, and risk signals from the same contracts.

---

## Stack

- **Python 3.13**, Pydantic AI, asyncpg
- **pgvector** (local PostgreSQL) — vector + full-text search
- **Apache AGE** (PostgreSQL 16) — openCypher knowledge graph
- **Ollama** (local LLM/embeddings) — llama3.1:8b, nomic-embed-text
- **Streamlit** UI, **Langfuse** observability, **Mem0** user memory

---

## Quick start

```powershell
# Install
pip install -e .

# Start AGE + viewer
docker compose up -d age age-viewer

# Populate graph from CUAD annotations (fast, no LLM)
python -m kg.cuad_kg_ingest

# Ingest documents into vector store
python -m rag.main --ingest --documents rag/documents

# Run tests
python -m pytest rag/tests/ -v
```

---

## Knowledge Graph

All four logical graphs share a single Apache AGE graph named `legal_graph`.
They are distinguished by vertex labels and edge types, not separate graph objects.

### 1. Legal Semantic Graph
Parties, jurisdictions, clause types, indemnification chains, payment terms.

```cypher
MATCH (c:Contract)-[r]->(n)
RETURN c, r, n
LIMIT 60
```

### 2. Document Hierarchy Graph
Contract → Section → Clause structure.

```cypher
MATCH (c:Contract)-[:HAS_SECTION]->(s:Section)-[:HAS_CLAUSE]->(cl:Clause)
RETURN c, s, cl
LIMIT 80
```

### 3. Cross-Contract Lineage Graph
Amendments, supersessions, incorporated-by-reference documents.

```cypher
MATCH (c1:Contract)-[r:AMENDS|SUPERCEDES|REPLACES|REFERENCES|INCORPORATES_BY_REFERENCE|ATTACHES]->(c2)
RETURN c1, r, c2
LIMIT 60
```

### 4. Risk Dependency Graph
Compliance gaps (missing clauses) and risk cascade chains — rule-based, no LLM.

```cypher
MATCH (r:Risk)-[rel]->(n)
RETURN r, rel, n
LIMIT 60
```

### Graph viewer

Open **http://localhost:3001** in Chrome after `docker compose up -d age age-viewer`.

Connect with: Host `age` · Port `5432` · Database `legal_graph` · User `age_user` · Password `age_pass` · Graph Path `legal_graph`

See [`kg/docs/GRAPH_VIEWER.md`](kg/docs/GRAPH_VIEWER.md) for extended queries and filter examples.

### Populating the graph

```powershell
# Option 1: CUAD annotations — 510 contracts, minutes, no LLM (recommended first)
python -m kg.cuad_kg_ingest

# Option 2: LLM extraction — Bronze → Silver → Gold → Risk, all 5 passes
python -m kg.extraction_pipeline --all          # all contracts (~5 days local)
python -m kg.extraction_pipeline --limit 20     # small batch to test
python -m kg.extraction_pipeline --project --all  # replay Silver+Gold from existing Bronze
```

---

## Architecture

```
rag/docs/ARCHITECTURE_SUMMARY.md      — full system overview (start here)
kg/docs/KG_INGESTION_PIPELINE.md      — extraction pipeline design
kg/docs/KG_FAQ.md                     — decisions, findings, run log
kg/docs/KG_RETRIEVAL_PIPELINE.md      — NL→Cypher retrieval pipeline
kg/docs/GRAPH_VIEWER.md               — AGE Viewer setup and Cypher queries
rag/docs/RAG.md                       — RAG techniques deep dive
rag/docs/CALL_GRAPH.md                — method-level call graphs
nl2sql/docs/ARCHITECTURE.md           — NL→SQL pipeline design
CLAUDE.md                             — dev conventions
TESTS.md                              — test suite docs
```

---

## Tests

```powershell
# All unit + Cypher shape tests (no external deps)
pytest rag/tests/test_nl_query.py -v

# Integration tests (requires live AGE + pgvector)
pytest rag/tests/test_nl_query.py -m integration -v

# Full suite
pytest rag/tests/ -v
```
