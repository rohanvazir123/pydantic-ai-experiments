# RAG v2 — Architecture Proposal

## Table of Contents

- [Goals](#goals)
- [Knowledge Layer — Multi-Corpus Design](#knowledge-layer--multi-corpus-design)

---

## Goals

1. Single `knowledge/` module replacing `rag/` + domain-specific KG code.
2. Multi-corpus ingestion: any folder on disk (or remote source) becomes a corpus namespace.
3. Docling-graph integration: chunking and KG extraction run as parallel async tasks per document.
4. Redis Streams + async workers for all heavyweight I/O (ingestion, retrieval, LLM calls).
5. Multi-level caching: in-process LRU → Redis → semantic similarity cache.
6. Enterprise security baseline: JWT auth, JWE payload encryption, TLS 1.3, RBAC, audit log.
7. Docker Compose for local development; cloud-native deployment (K8s + managed services) for production.

---

### Knowledge Layer — Multi-Corpus Design

Each corpus is an independent namespace sharing the same PostgreSQL cluster. Corpus isolation is enforced at the storage layer via a `corpus_id` column on `documents` and `chunks`.

**Corpus config** (`knowledge/corpus/registry.py`):
```python
class CorpusConfig(BaseModel):
    id: str                          # slug, e.g. "hr-policies"
    display_name: str
    source_folders: list[Path]       # local paths scanned on ingest
    allowed_roles: list[str]         # RBAC: which JWT roles can read/write
    metadata_tags: dict[str, str]    # extra metadata attached to every chunk

    # Knowledge graph extraction (docling-graph)
    enable_graph_extraction: bool = False
    # Path to the Pydantic ontology template, relative to knowledge/corpus/ontologies/
    # If None, uses the generic default template (extracts entities/relations without domain specifics)
    graph_ontology_path: str | None = None
    # LLM backend provider — any LiteLLM-compatible provider; "ollama" for local
    graph_extraction_provider: str = "ollama"
    # Model for graph extraction (can differ from chat model; smaller is fine for entity extraction)
    graph_extraction_model: str = "llama3.2:3b"
    # Extraction contract:
    #   "direct"  — single LLM call per chunk; fastest; good for large models (≥ 70B)
    #   "staged"  — multi-pass ID → fill → quality gate; recommended for small models (≤ 8B)
    #   "delta"   — chunk-by-chunk with merge + dedup resolvers; best for long documents
    graph_extraction_contract: Literal["direct", "staged", "delta"] = "staged"
    # Processing mode:
    #   "many-to-one" — all chunks merged into one graph; best for most docs
    #   "one-to-one"  — page-by-page; best for forms and complex layouts
    graph_processing_mode: Literal["many-to-one", "one-to-one"] = "many-to-one"
    # VLM extraction for scanned/image-heavy PDFs (requires GPU)
    graph_extraction_backend: Literal["llm", "vlm"] = "llm"
```

**Schema change** (additive migration):
- `documents.corpus_id TEXT NOT NULL` + B-tree index
- `chunks.corpus_id TEXT NOT NULL` + B-tree index (for fast corpus-scoped search)
- All queries gain a `WHERE corpus_id = $1` predicate automatically

**Cross-corpus search**: allowed with explicit `corpus_ids: list[str]` in the search request, subject to JWT role check across all listed corpora.

---


