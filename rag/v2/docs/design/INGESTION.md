# RAG v2 — Ingestion Pipeline

## Table of Contents

- [Ingestion Pipeline — Docling-Graph Parallel Paths](#ingestion-pipeline--docling-graph-parallel-paths)
- [Knowledge Graph Extraction — Ontology and docling-graph API](#knowledge-graph-extraction--ontology-and-docling-graph-api)
  - [The ontology is a Pydantic template](#the-ontology-is-a-pydantic-template)
  - [Entities vs Components — decision rule](#entities-vs-components--decision-rule)
  - [Extraction contracts — which to use](#extraction-contracts--which-to-use)
  - [Actual API call](#actual-api-call)
- [Apache AGE — Graph Store Design (`knowledge/store/graph.py`)](#apache-age--graph-store-design-knowledgestoregraphpy)
  - [How AGE works with asyncpg](#how-age-works-with-asyncpg)
  - [Graph name per corpus](#graph-name-per-corpus)
  - [Key method: `import_docling_graph()`](#key-method-import_docling_graph)
  - [Label and relationship-type sanitization (v2 — no hardcoded allowlist)](#label-and-relationship-type-sanitization-v2--no-hardcoded-allowlist)
  - [Vertex upsert (MERGE pattern)](#vertex-upsert-merge-pattern)
  - [Read-only query (for the graph retriever)](#read-only-query-for-the-graph-retriever)
  - [Corpus-scoped delete (tenant offboarding)](#corpus-scoped-delete-tenant-offboarding)
  - [Entity index (shadow table in main PostgreSQL)](#entity-index-shadow-table-in-main-postgresql)
  - [Docker Compose — AGE runs separately from the main PostgreSQL](#docker-compose--age-runs-separately-from-the-main-postgresql)
  - [Ontology storage and loading (`knowledge/corpus/ontologies/`)](#ontology-storage-and-loading-knowledgecorpusontologies)
  - [Ontology management API](#ontology-management-api)

---

### Ingestion Pipeline — Docling-Graph Parallel Paths

Per document, after Docling conversion, two async tasks run concurrently:

```
DocumentConverter.convert(path)
        │
        ▼
 DoclingDocument (in memory)
        │
   asyncio.gather(
     ├── chunker_task:
     │      HybridChunker → List[ChunkData]
     │      → embedder.embed_batch()
     │      → vector_store.upsert_chunks()
     │
     └── graph_task (if corpus.enable_graph_extraction):
            load ontology class from corpus.graph_ontology_path
            run_pipeline(PipelineConfig(template=OntologyClass, ...))
            → PipelineContext.knowledge_graph (NetworkX DiGraph)
            → age_graph_store.import_docling_graph(context, corpus_id, doc_id)
               ├── iterate graph.nodes(data=True) → upsert_vertex() per node
               └── iterate graph.edges(data=True) → add_edge() per edge
            → entity_index.upsert_batch(vertices)
   )
        │
        ▼
  publish IngestCompleteEvent to Redis
```

---

### Knowledge Graph Extraction — Ontology and docling-graph API

This section documents exactly how docling-graph is used. Read this before implementing `knowledge/ingestion/graph_extractor.py`.

#### The ontology is a Pydantic template

docling-graph extracts entities and relationships whose shape is defined entirely by a **Pydantic `BaseModel` subclass** (called a "template" in docling-graph terminology). The template IS the ontology — there is no separate schema format.

**Minimal template** (required structure every ontology file must follow):

```python
# knowledge/corpus/ontologies/my_domain.py
"""
HR policy ontology.
Extracts policies, benefits, people, and departments from HR documents.
"""
from typing import Any, List
from pydantic import BaseModel, ConfigDict, Field

def edge(label: str, **kwargs: Any) -> Any:
    """Required helper — marks a field as a directed graph edge."""
    return Field(..., json_schema_extra={"edge_label": label}, **kwargs)

# --- Components (value objects — no stable graph identity) ---
class ContactInfo(BaseModel):
    model_config = ConfigDict(is_entity=False)
    email: str | None = Field(None, description="Email address. LOOK FOR: @ symbol. EXAMPLES: 'hr@company.com'")
    phone: str | None = Field(None, description="Phone number")

# --- Entities (unique, identifiable — get stable node IDs) ---
class Person(BaseModel):
    model_config = ConfigDict(graph_id_fields=["full_name"])   # stable ID from these fields
    full_name: str = Field(description="Full name. LOOK FOR: Names near job titles. EXAMPLES: 'Jane Smith'")
    title: str | None = Field(None, description="Job title")
    contact: ContactInfo | None = Field(None, description="Contact details")

class Department(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(description="Department name. EXAMPLES: 'Engineering', 'HR'")
    head: Person | None = edge(label="LED_BY", default=None, description="Department head")
    members: List[Person] = edge(label="HAS_MEMBER", default_factory=list, description="Staff in dept")

class Policy(BaseModel):
    model_config = ConfigDict(graph_id_fields=["policy_id"])
    policy_id: str = Field(description="Policy identifier. EXAMPLES: 'PTO-001', 'REMOTE-002'")
    title: str = Field(description="Policy title")
    description: str | None = Field(None, description="Policy text summary")
    applies_to: List[Department] = edge(label="APPLIES_TO", default_factory=list,
                                         description="Departments this policy covers")

# --- Root document model (last in file, captures the whole document) ---
class HRPolicyDocument(BaseModel):
    model_config = ConfigDict(graph_id_fields=["document_title"])
    document_title: str = Field(description="Document title. LOOK FOR: Title page heading.")
    policies: List[Policy] = edge(label="CONTAINS_POLICY", default_factory=list,
                                   description="Policies described in this document")
    departments: List[Department] = edge(label="REFERENCES_DEPT", default_factory=list,
                                          description="Departments mentioned")

HRPolicyDocument.model_rebuild()
```

**Key rules for every ontology template:**
1. The `edge()` helper MUST be defined identically in every template file — `Field(..., json_schema_extra={"edge_label": label}, **kwargs)`
2. **Entities** have `graph_id_fields` in `ConfigDict` — these fields create stable node IDs and enable cross-chunk deduplication
3. **Components** have `is_entity=False` — they are value objects embedded in entities, deduplicated by content
4. **List edges** MUST have `default_factory=list`
5. Field `description` follows `LOOK FOR / EXTRACT / EXAMPLES` pattern — this is the prompt the LLM sees; poor descriptions = poor extraction
6. The root model (last class in file) is what `PipelineConfig.template` points to
7. Call `Model.model_rebuild()` at file end when using forward references

#### Entities vs Components — decision rule

| Question | Entity | Component |
|----------|--------|-----------|
| Does it need a stable, reusable node ID? | Yes | No |
| Can two instances be "the same thing"? | Yes (dedup by `graph_id_fields`) | Yes (dedup by content) |
| Can it appear as a standalone node? | Yes | No — only embedded in entities |
| Example | Person, Department, Policy | Address, ContactInfo, MonetaryAmount |

#### Extraction contracts — which to use

| Contract | When to use | How it works |
|----------|-------------|-------------|
| `"direct"` | Large models (≥ 70B), simple schemas | One LLM call per chunk; fastest |
| `"staged"` | Small models (≤ 8B like llama3.2:3b), complex nested schemas | Multi-pass: ID discovery → fill pass → quality gate; recommended for Ollama |
| `"delta"` | Long documents (>50 pages), many entities of the same type | Chunk-by-chunk with incremental merge and semantic deduplication resolvers |

**Default for our system:** `"staged"` — we use `llama3.2:3b` via Ollama for graph extraction. Staged contract breaks complex templates into simpler multi-pass operations that smaller models handle reliably.

#### Actual API call

```python
# knowledge/ingestion/graph_extractor.py
from pathlib import Path
from docling_graph import PipelineConfig, run_pipeline
from docling_graph.pipeline.context import PipelineContext

async def extract_graph(
    doc_path: Path,
    ontology_class: type,          # loaded from corpus.graph_ontology_path
    corpus_config: CorpusConfig,
    settings: Settings,
) -> PipelineContext | None:
    """Run docling-graph extraction. Returns PipelineContext or None on failure.

    NOTE: Do NOT use CypherExporter. AGE uses ag_catalog.cypher() SQL wrapper
    syntax — not Neo4j-compatible raw Cypher. Feed the NetworkX DiGraph directly
    to AgeGraphStore.import_docling_graph() instead.
    """

    def _run_sync() -> PipelineContext:
        config = PipelineConfig(
            source=str(doc_path),
            template=ontology_class,
            backend=corpus_config.graph_extraction_backend,           # "llm" | "vlm"
            inference="local",
            provider_override=corpus_config.graph_extraction_provider, # "ollama"
            model_override=corpus_config.graph_extraction_model,       # "llama3.2:3b"
            processing_mode=corpus_config.graph_processing_mode,       # "many-to-one"
            extraction_contract=corpus_config.graph_extraction_contract, # "staged"
            use_chunking=True,
            chunk_max_tokens=settings.chunk_max_tokens,
            structured_output=True,
            dump_to_disk=False,    # API mode — no files on disk
        )
        return run_pipeline(config)   # returns PipelineContext, not a string

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_sync),
            timeout=settings.graph_extraction_timeout_s,
        )
    except TimeoutError:
        logger.warning("Graph extraction timed out for %s", doc_path.name)
        return None
    except Exception as exc:
        logger.error("Graph extraction failed for %s: %s", doc_path.name, exc)
        return None
```

Then in the pipeline orchestrator:
```python
context = await extract_graph(doc_path, ontology_class, corpus_config, settings)
if context:
    node_count, edge_count = await age_store.import_docling_graph(
        context, corpus_id=corpus_config.id, document_id=document_id
    )
    await entity_index.upsert_batch_from_graph(context.knowledge_graph, document_id)
else:
    chunk_metadata["graph_extraction_failed"] = True
```

---

### Apache AGE — Graph Store Design (`knowledge/store/graph.py`)

The v2 `AgeGraphStore` is a rewrite of `kg/age_graph_store.py` adapted for multi-corpus, multi-tenant use. The v1 implementation is hardwired to the CUAD legal ontology (label allowlist from `cuad_ontology.py`); v2 accepts any labels from the user's docling-graph template.

#### How AGE works with asyncpg

Apache AGE adds openCypher graph queries to PostgreSQL via a SQL function wrapper. Every Cypher statement must be wrapped:

```sql
SELECT * FROM ag_catalog.cypher('graph_name', $$
    MATCH (n:Person) RETURN n.name, n.uuid
$$) AS (name agtype, uuid agtype)
```

`agtype` columns are returned as strings by asyncpg (they look like `"Acme Corp"` with surrounding quotes). Strip with `s[1:-1]` if starts/ends with `"`.

Every connection must run two setup statements before any Cypher:
```python
await conn.execute("LOAD 'age'")
await conn.execute("SET search_path = ag_catalog, \"$user\", public")
```

Register this as an asyncpg pool `init` callback — AGE state is connection-local and gets reset by `RESET ALL` when connections return to the pool.

#### Graph name per corpus

Each corpus gets its own AGE graph: `f"{tenant_id}_{corpus_id}"` (e.g. `"acme_corp_hr_policies"`). This gives hard isolation — queries against one corpus never touch another's graph. The graph is created on first ingest:

```python
await conn.execute(f"SELECT create_graph('{graph_name}')")
```

Use `try/except` around creation — AGE raises `InvalidSchemaNameError` if the graph already exists.

#### Key method: `import_docling_graph()`

This is the primary write path from docling-graph. It iterates the NetworkX DiGraph from `PipelineContext` directly — **not** `CypherExporter`. AGE uses a SQL wrapper syntax that is incompatible with the raw Cypher `CREATE` statements that `CypherExporter` generates for Neo4j.

```python
async def import_docling_graph(
    self,
    context: "PipelineContext",   # from docling_graph.pipeline.context
    corpus_id: str,
    document_id: str,
) -> tuple[int, int]:
    """Import a docling-graph PipelineContext into Apache AGE.

    Iterates context.knowledge_graph (NetworkX DiGraph) directly.
    Do NOT use CypherExporter — its output is Neo4j syntax, incompatible with AGE.

    Returns (node_count, edge_count).
    """
    graph = context.knowledge_graph     # networkx.DiGraph
    graph_name = self._graph_name(corpus_id)

    node_id_map: dict[str, str] = {}    # NetworkX node_id → AGE vertex uuid

    # 1. Upsert all vertices
    for nx_id, attrs in graph.nodes(data=True):
        label = _sanitize_label(attrs.get("label", "Entity"))
        name  = str(attrs.get("name") or attrs.get("id") or nx_id)
        props = {k: str(v) for k, v in attrs.items()
                 if k not in ("label",) and v is not None}
        props["corpus_id"]   = corpus_id
        props["document_id"] = document_id

        uuid = await self._upsert_vertex(graph_name, nx_id, label, name, props)
        node_id_map[str(nx_id)] = uuid

    # 2. Upsert all edges
    edge_count = 0
    for src_nx, tgt_nx, edge_attrs in graph.edges(data=True):
        rel_type = _sanitize_rel_type(edge_attrs.get("label", "RELATED_TO"))
        src_uuid = node_id_map.get(str(src_nx))
        tgt_uuid = node_id_map.get(str(tgt_nx))
        if src_uuid and tgt_uuid:
            await self._add_edge(graph_name, src_uuid, tgt_uuid, rel_type,
                                  {"corpus_id": corpus_id, "document_id": document_id})
            edge_count += 1

    return len(graph.nodes), edge_count
```

#### Label and relationship-type sanitization (v2 — no hardcoded allowlist)

v1 validated labels against a hardcoded CUAD list. v2 accepts any label from the user's ontology template, only sanitizing characters:

```python
import re

def _sanitize_label(label: str) -> str:
    """Strip non-alphanumeric characters; ensure starts with uppercase letter."""
    cleaned = re.sub(r"[^A-Za-z0-9]", "", label)
    if not cleaned:
        return "Entity"
    return cleaned[0].upper() + cleaned[1:]

def _sanitize_rel_type(rel_type: str) -> str:
    """Uppercase + strip non-alphanumeric except underscore."""
    cleaned = re.sub(r"[^A-Z0-9_]", "", rel_type.upper())
    return cleaned or "RELATED_TO"
```

#### Vertex upsert (MERGE pattern)

```python
async def _upsert_vertex(
    self, graph_name: str, nx_id: str, label: str, name: str, props: dict
) -> str:
    """MERGE vertex by (nx_id, corpus_id); return AGE uuid."""
    vertex_uuid = str(uuid.uuid4())
    name_esc = name.replace('"', '\\"')
    nx_id_esc = str(nx_id).replace('"', '\\"')
    corpus_esc = props.get("corpus_id", "").replace('"', '\\"')

    # MERGE on stable identity: the docling-graph node ID + corpus
    cypher = (
        f'MERGE (v:{label} {{nx_id: "{nx_id_esc}", corpus_id: "{corpus_esc}"}}) '
        f'SET v.uuid = COALESCE(v.uuid, "{vertex_uuid}"), '
        f'v.name = "{name_esc}", '
        f'v.label = "{label}" '
        f'RETURN v.uuid'
    )
    async with self._conn() as conn:
        rows = await conn.fetch(
            f"SELECT * FROM ag_catalog.cypher('{graph_name}', $${cypher}$$) AS (uuid agtype)"
        )
    return _unquote_agtype(rows[0]["uuid"]) if rows else vertex_uuid
```

#### Read-only query (for the graph retriever)

```python
async def run_cypher_query(self, cypher: str, corpus_id: str) -> str:
    """Execute a read-only MATCH query scoped to corpus_id's graph."""
    if re.search(r"\b(CREATE|MERGE|SET|DELETE|DROP|DETACH|FOREACH)\b", cypher, re.I):
        return "Error: only MATCH queries are permitted."

    graph_name = self._graph_name(corpus_id)
    aliases = _parse_return_aliases(cypher)   # from v1; infer column names from RETURN clause
    as_clause = ", ".join(f"c{i} agtype" for i in range(len(aliases)))

    async with self._conn() as conn:
        try:
            rows = await conn.fetch(
                f"SELECT * FROM ag_catalog.cypher('{graph_name}', $${cypher}$$) AS ({as_clause})"
            )
        except Exception as exc:
            return f"Cypher error: {exc}"

    if not rows:
        return "No results."
    header = " | ".join(aliases)
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(" | ".join(_unquote_agtype(row[f"c{i}"]) for i in range(len(aliases))))
    lines.append(f"\n({len(rows)} row{'s' if len(rows) != 1 else ''})")
    return "\n".join(lines)
```

#### Corpus-scoped delete (tenant offboarding)

```python
async def delete_corpus_graph(self, corpus_id: str) -> None:
    """Drop the entire AGE graph for a corpus — all vertices and edges."""
    graph_name = self._graph_name(corpus_id)
    async with self._conn() as conn:
        await conn.execute(f"SELECT drop_graph('{graph_name}', true)")

async def delete_document_vertices(self, corpus_id: str, document_id: str) -> None:
    """Remove all vertices (and their edges) for one document from a corpus graph."""
    graph_name = self._graph_name(corpus_id)
    cypher = f'MATCH (v {{document_id: "{document_id}"}}) DETACH DELETE v'
    async with self._conn() as conn:
        await conn.execute(
            f"SELECT * FROM ag_catalog.cypher('{graph_name}', $${cypher}$$) AS (r agtype)"
        )
```

#### Entity index (shadow table in main PostgreSQL)

AGE does not support `tsvector` GIN indexes or `pgvector` — all CONTAINS scans in AGE are O(n). The `knowledge/store/entity_index.py` (ported from `kg/entity_index.py`) maintains a `kg_entity_index` shadow table in the main PostgreSQL database with:
- `age_uuid TEXT PRIMARY KEY` — maps back to the AGE vertex
- `name TEXT` + `name_tsv tsvector GENERATED` — GIN-indexed for BM25 search
- `label TEXT` — B-tree indexed for label filtering
- `corpus_id TEXT` + `document_id TEXT` — for scoped deletes
- `embedding vector(768)` — HNSW-indexed for semantic entity search

After each `import_docling_graph()`, call `entity_index.upsert_batch_from_graph(graph, document_id, corpus_id)` to sync vertex names into the shadow table.

#### Docker Compose — AGE runs separately from the main PostgreSQL

AGE cannot run in the same container as the main pgvector database (different extension sets, potential version conflicts). Use two separate PostgreSQL instances:

```yaml
postgres:
  image: pgvector/pgvector:pg16    # main DB: pgvector for vector search
  ports: ["5432:5432"]

age:
  image: apache/age:latest         # graph DB: Apache AGE for Cypher queries
  ports: ["5433:5432"]             # mapped to 5433 on host to avoid conflict
  environment:
    POSTGRES_DB: age_graph
    POSTGRES_USER: age
    POSTGRES_PASSWORD: ${AGE_DB_PASSWORD}
```

Settings:
```python
database_url: str         # main PostgreSQL (pgvector) — port 5432
age_database_url: str     # AGE PostgreSQL — port 5433
age_graph_prefix: str = "kg"  # graph names: f"{prefix}_{tenant_id}_{corpus_id}"
```

#### Ontology storage and loading (`knowledge/corpus/ontologies/`)

```
knowledge/corpus/ontologies/
├── __init__.py
├── loader.py          # load_ontology(path: str) → type[BaseModel]; LRU-cached
├── generic.py         # default ontology when no corpus-specific template provided
├── hr_policy.py       # example domain ontology
├── legal_contract.py  # example domain ontology
└── <user-defined>.py  # uploaded by admin via POST /v1/corpus/{id}/ontology
```

**Generic default ontology** (`generic.py`) — used when `corpus_config.graph_ontology_path is None`. Extracts named entities, organizations, dates, and generic relationships without domain specifics:

```python
class GenericEntity(BaseModel):
    model_config = ConfigDict(graph_id_fields=["name"])
    name: str = Field(description="Entity name. EXTRACT: The most specific identifier. EXAMPLES: 'Apple Inc', 'John Smith', 'ISO 27001'")
    entity_type: str = Field(description="Type of entity. EXAMPLES: 'Organization', 'Person', 'Location', 'Concept', 'Date', 'Product'")
    description: str | None = Field(None, description="Brief description from document context")
    related_to: List["GenericEntity"] = edge(label="RELATED_TO", default_factory=list,
        description="Entities this one is related to per document context")

class GenericDocument(BaseModel):
    model_config = ConfigDict(graph_id_fields=["title"])
    title: str = Field(description="Document title or best identifying label")
    entities: List[GenericEntity] = edge(label="MENTIONS", default_factory=list,
        description="All named entities mentioned in the document")

GenericEntity.model_rebuild()
GenericDocument.model_rebuild()
```

**Ontology loader** (`loader.py`) — loads a Python file from the ontologies directory and returns the root Pydantic class:

```python
import importlib.util, functools
from pathlib import Path
from pydantic import BaseModel

ONTOLOGIES_DIR = Path(__file__).parent

@functools.lru_cache(maxsize=32)
def load_ontology(ontology_path: str | None) -> type[BaseModel]:
    """Load ontology class from path relative to ontologies/. LRU-cached per worker."""
    if ontology_path is None:
        from knowledge.corpus.ontologies.generic import GenericDocument
        return GenericDocument

    full_path = ONTOLOGIES_DIR / ontology_path
    if not full_path.exists():
        raise FileNotFoundError(f"Ontology not found: {full_path}")

    spec = importlib.util.spec_from_file_location("_ontology", full_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # executes the Python file

    # Root class = last BaseModel subclass defined in the file (by convention)
    root_class = None
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
            root_class = obj  # last one wins
    if root_class is None:
        raise ValueError(f"No BaseModel subclass found in {full_path}")
    return root_class
```

#### Ontology management API

Admins can upload new ontologies per corpus via the API:

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET`  | `/v1/corpus/{id}/ontology` | `admin` | Get current ontology file for corpus |
| `POST` | `/v1/corpus/{id}/ontology` | `admin` | Upload Python ontology file; validates it is a valid Pydantic template |
| `DELETE` | `/v1/corpus/{id}/ontology` | `admin` | Remove custom ontology (reverts to generic default) |

On upload, the API:
1. Parses the Python file and verifies it contains a root `BaseModel` subclass
2. Checks the `edge()` helper is defined correctly
3. Saves to `knowledge/corpus/ontologies/{corpus_id}.py`
4. Updates `CorpusConfig.graph_ontology_path` in the corpus registry
5. Clears the `load_ontology` LRU cache so next extraction uses the new template

---

