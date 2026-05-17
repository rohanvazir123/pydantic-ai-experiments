# FAQ — LightRAG

## LightRAG Architecture and Internals

### What is LightRAG?

LightRAG is a graph-augmented RAG framework. Unlike plain vector RAG (which retrieves chunks by embedding similarity), LightRAG first builds a **knowledge graph** from the document corpus using an LLM, then at query time retrieves both graph entities/relationships and raw text chunks, combining them into a richer context for the answer LLM.

The key idea: entities and their relationships are stored as first-class objects alongside the text, enabling queries that require multi-hop reasoning ("what is the relationship between X and Y?") that pure vector search cannot answer.

### Internal Pipeline

```
Documents (text / markdown / chunks from Docling)
        │
        ▼
┌──────────────────────────────────────┐
│   Chunking                           │  Token-based splitting (default 1200
│   (chunking_by_token_size)           │  tokens, 100-token overlap). Each chunk
│                                      │  gets a hash ID. Stored in
│                                      │  LIGHTRAG_DOC_CHUNKS.
└──────────────────────────────────────┘
        │
        ▼  (for each chunk, in parallel up to max_async=4)
┌──────────────────────────────────────┐
│   Entity & Relationship Extraction   │  LLM called with entity_extraction
│   (operate.py)                       │  system prompt. Outputs structured
│                                      │  tuples: entity and relation lines
│                                      │  delimited by <|#|>.
│                                      │  On partial output → gleaning pass
│                                      │  (up to max_gleaning=1 by default).
└──────────────────────────────────────┘
        │
        ├──► Entities → deduplicated, descriptions merged via LLM summariser
        │             → embedded → LIGHTRAG_VDB_ENTITY (pgvector)
        │             → stored as graph nodes in AGE
        │
        └──► Relations → deduplicated, keywords + description merged
                       → embedded → LIGHTRAG_VDB_RELATION (pgvector)
                       → stored as graph edges in AGE
        │
        ▼
┌──────────────────────────────────────┐
│   Query                              │  Keywords extracted from query (high-
│                                      │  level + low-level). Vector search on
│                                      │  entities + relations. Graph traversal
│                                      │  from matched nodes. Chunk retrieval.
│                                      │  Combined context → answer LLM.
└──────────────────────────────────────┘
```

### Graph Ontology

LightRAG uses a **flat, open-ended ontology** — it does not enforce a fixed schema. The entity types and relationship types are whatever the LLM extracts from the text.

**Default entity types** (configurable via `ENTITY_TYPES` env var):

```
Person, Creature, Organization, Location, Event,
Concept, Method, Content, Data, Artifact, NaturalObject
```

Any entity that doesn't fit is classified as `Other`.

**Relationship structure:** All relationships are binary (two entities) and treated as **undirected** unless the text explicitly states direction. Each relationship has:
- `source_entity` and `target_entity` (entity names, title-cased)
- `relationship_keywords` — comma-separated high-level themes (e.g. `power dynamics, observation`)
- `relationship_description` — a sentence explaining the connection

**N-ary decomposition:** If a statement involves 3+ entities (e.g. "Alice, Bob, and Carol collaborated on Project X"), the LLM decomposes it into binary pairs automatically.

This is a **property graph**, not an RDF triple store or fixed ontology. Entity and relationship descriptions accumulate as the same entity appears in multiple chunks, then get LLM-summarised when a merge threshold is hit (default: 8 descriptions trigger a summary).

### LLM Prompts

#### Entity extraction (the core prompt)

Called once per chunk. The system prompt instructs the LLM to output one line per entity and one line per relationship, delimited by `<|#|>`, ending with `<|COMPLETE|>`.

**System prompt structure:**
```
---Role---
You are a Knowledge Graph Specialist...

---Instructions---
1. Entity Extraction: output lines like:
   entity<|#|>entity_name<|#|>entity_type<|#|>entity_description

2. Relationship Extraction: output lines like:
   relation<|#|>source_entity<|#|>target_entity<|#|>keywords<|#|>description

3. Delimiter usage: <|#|> is a field separator, never filled with content.
4. N-ary decomposition into binary pairs.
5. Undirected relationships (no duplicates for A→B and B→A).
6. Output all entities first, then all relationships.
7. End with <|COMPLETE|>.
```

**User prompt:**
```
Extract entities and relationships from:

<Entity_types>
[Person, Organization, Location, ...]

<Input Text>
```{chunk_text}```
```

#### Gleaning pass (catch misses)

If the LLM output is truncated or misses entities, a follow-up user prompt asks it to re-output only the **missed or incorrectly formatted** ones — not the already-correct ones. Run once by default (`max_gleaning=1`).

#### Entity description summarisation

When an entity accumulates ≥8 descriptions from different chunks, an LLM call merges them:
```
Synthesize a list of descriptions of a given entity into a single
comprehensive summary. Max {summary_length} tokens. Third-person, objective.
```

#### Keyword extraction (at query time)

Before searching, the query is analysed to extract two keyword types:
- `high_level_keywords` — overarching themes/concepts
- `low_level_keywords` — specific entities, proper nouns, technical terms

Output is a JSON object used to drive both vector search (low-level) and graph traversal (high-level).

### Local LLM and Configurables

**Recommended local models:**

| Model | VRAM | Suitability |
|---|---|---|
| `qwen2.5:14b` | ~8GB | Best — follows structured JSON/delimiter output reliably |
| `qwen2.5:7b` | ~5GB | Good — reasonable JSON adherence |
| `llama3.1:8b` | ~5.5GB | Acceptable — occasional format drift, needs gleaning |
| `mistral:7b` | ~4.5GB | Borderline — frequent format failures on complex chunks |

**Key configurables (set in `.env` or passed to `LightRAG()`):**

| Parameter | Default | Effect |
|---|---|---|
| `chunk_token_size` | 1200 | Tokens per chunk sent to LLM for extraction |
| `chunk_overlap_token_size` | 100 | Overlap between adjacent chunks |
| `max_gleaning` | 1 | Extra extraction passes to catch missed entities |
| `entity_extract_max_gleaning` | 1 | Same, specifically for entity extraction |
| `max_async` | 4 | Parallel LLM calls during ingestion |
| `ENTITY_TYPES` | (11 types) | Override entity type list via env var |
| `summary_language` | `English` | Language for entity/relation descriptions |
| `llm_model_max_token_size` | model-dependent | Hard cap on tokens sent to LLM |

### Context Window Management

Each chunk sent for extraction consumes:
- `chunk_token_size` tokens of input (default 1200)
- System prompt: ~600 tokens
- Examples in prompt: ~800 tokens
- **Total input per chunk: ~2600 tokens minimum**

The extraction output (entities + relations) can be another 500–1500 tokens depending on document density.

**Minimum safe context window: 4096 tokens.** Recommended: 8192+.

For `ollama`, set `num_ctx` to avoid silent truncation:
```python
# In lightrag_utils.py — already wired for this project
extra_body={"num_ctx": 131072}
```

**What happens when the context window is too small:**
- The LLM truncates its output mid-entity or mid-relation line
- The `<|COMPLETE|>` delimiter is never emitted
- LightRAG detects the incomplete output and fires the gleaning pass
- If gleaning also truncates, entities from that chunk are silently lost — no error is raised

**Rule of thumb:** Set `chunk_token_size` ≤ 20% of your model's context window. For `llama3.1:8b` at 8K context: max `chunk_token_size` = 1600. For `qwen2.5:14b` at 128K context: no practical limit.

### PostgreSQL Storage Schema

LightRAG creates 11 tables (all prefixed `LIGHTRAG_`). All tables have a `workspace` column for multi-tenancy.

| Table | Purpose |
|---|---|
| `LIGHTRAG_DOC_FULL` | Raw full document content + metadata |
| `LIGHTRAG_DOC_CHUNKS` | Text chunks with order index and token count |
| `LIGHTRAG_VDB_CHUNKS` | Chunks + embedding vector (pgvector) for chunk retrieval |
| `LIGHTRAG_VDB_ENTITY` | Entity name + description embedding (pgvector) |
| `LIGHTRAG_VDB_RELATION` | Relation description embedding (pgvector) |
| `LIGHTRAG_LLM_CACHE` | Cache of LLM prompt → response pairs (avoids re-extraction) |
| `LIGHTRAG_DOC_STATUS` | Ingestion status per document (pending/processing/done/failed) |
| `LIGHTRAG_FULL_ENTITIES` | All entity names grouped by document |
| `LIGHTRAG_FULL_RELATIONS` | All relation pairs grouped by document |
| `LIGHTRAG_ENTITY_CHUNKS` | Entity → chunk_ids mapping |
| `LIGHTRAG_RELATION_CHUNKS` | Relation → chunk_ids mapping |

**Vector indexes:** HNSW (default), IVFFLAT, HNSW_HALFVEC, or VChordrq — set via `POSTGRES_VECTOR_INDEX_TYPE`.

**Key DDL examples:**

```sql
-- Chunk text + vector
CREATE TABLE LIGHTRAG_VDB_CHUNKS (
    id VARCHAR(255),
    workspace VARCHAR(255),
    full_doc_id VARCHAR(256),
    tokens INTEGER,
    content TEXT,
    content_vector VECTOR(768),   -- dimension = EMBEDDING_DIMENSION
    file_path TEXT,
    CONSTRAINT LIGHTRAG_VDB_CHUNKS_PK PRIMARY KEY (workspace, id)
);

-- Entity name + description vector
CREATE TABLE LIGHTRAG_VDB_ENTITY (
    id VARCHAR(255),
    workspace VARCHAR(255),
    entity_name VARCHAR(512),
    content TEXT,                  -- merged description text
    content_vector VECTOR(768),
    chunk_ids VARCHAR(255)[],      -- source chunk IDs
    file_path TEXT,
    CONSTRAINT LIGHTRAG_VDB_ENTITY_PK PRIMARY KEY (workspace, id)
);

-- Relation source→target + description vector
CREATE TABLE LIGHTRAG_VDB_RELATION (
    id VARCHAR(255),
    workspace VARCHAR(255),
    source_id VARCHAR(512),        -- source entity name
    target_id VARCHAR(512),        -- target entity name
    content TEXT,                  -- keywords + description
    content_vector VECTOR(768),
    chunk_ids VARCHAR(255)[],
    file_path TEXT,
    CONSTRAINT LIGHTRAG_VDB_RELATION_PK PRIMARY KEY (workspace, id)
);
```

### How Apache AGE is Used

LightRAG uses Apache AGE for the **graph traversal** part of retrieval — finding connected entities and multi-hop paths between nodes.

**Setup:** LightRAG calls `CREATE EXTENSION IF NOT EXISTS AGE CASCADE` and `create_graph('{graph_name}')` at initialisation. The `search_path` is set to include `ag_catalog` on each connection that uses AGE.

**What's stored in AGE:**
- **Nodes** = entities (entity_name as the node label/property)
- **Edges** = relationships (source_entity → target_entity, with keywords and description as edge properties)

**What's stored in pgvector (not AGE):**
- Entity and relation embeddings (`LIGHTRAG_VDB_ENTITY`, `LIGHTRAG_VDB_RELATION`)
- Chunk embeddings (`LIGHTRAG_VDB_CHUNKS`)

**Query flow:**
1. Vector search on `LIGHTRAG_VDB_ENTITY` → matched entity names
2. AGE Cypher traversal from those entity nodes → neighbouring nodes and edges
3. Vector search on `LIGHTRAG_VDB_RELATION` → matched relationships
4. `LIGHTRAG_ENTITY_CHUNKS` + `LIGHTRAG_RELATION_CHUNKS` → chunk IDs
5. Chunks retrieved from `LIGHTRAG_VDB_CHUNKS`
6. All combined into context for the answer LLM

**In short:** pgvector finds the entry points into the graph; AGE traverses the graph from those entry points.

### Query Modes

LightRAG supports 4 query modes, selectable per query:

| Mode | What it searches | Best for |
|---|---|---|
| `naive` | Raw chunk vector search only (no graph) | Simple factual lookups |
| `local` | Entity + relation vector search → linked chunks | Specific entity questions |
| `global` | High-level keyword search across the full graph | Thematic / summary questions |
| `hybrid` | `local` + `global` combined | General use — recommended default |

### Flat Ontology in LightRAG vs Well-Defined Schema in docling-graph

#### The core difference

| | LightRAG (flat ontology) | docling-graph / project KG (well-defined schema) |
|---|---|---|
| Entity types | Whatever the LLM decides | Fixed set, defined upfront |
| Relationship types | Free-text keywords | Fixed set, typed and directional |
| Schema enforcement | None | Validated at extraction time |
| Cypher queries | Impractical | Precise and predictable |
| Setup cost | Zero — works on any domain | Requires domain expertise upfront |
| Cross-domain use | Yes | No — schema is domain-specific |

---

#### LightRAG: flat, open-ended ontology

LightRAG extracts whatever entity types and relationship keywords the LLM produces from the text. There is no schema — just 11 suggested type names passed as a hint in the prompt.

**Entity types the LLM assigned in our actual demo run:**

```
Person        — Ashish Vaswani
Organization  — Google Brain, Google AI Language
Method        — Transformer Architecture, BERT Model, Multi-Head Self-Attention
Content       — Paper "Attention Is All You Need"
category      — 3.4% Decline, Market Selloff, Quarterly Earnings Report
product       — Gold Futures, Crude Oil
equipment     — The Device
```

**Relationship keywords from our demo:**

```
"academic collaboration, model creation"
"model introduction, research development"
"model enhancement, technological advancement"
"financial performance, market reaction"
"knowledge graph traversal, system component"
```

Notice: these are free-text descriptions, not typed relationship labels. There is no `INVENTED_BY`, no `EXTENDS`, no `GOVERNED_BY`. You cannot write a precise Cypher query like:

```cypher
-- This CANNOT be done reliably in LightRAG's graph
MATCH (p:Person)-[:INVENTED]->(m:Method)
RETURN p.name, m.name
```

Because `Person` might be typed as `person` or `researcher` in a different chunk, and the relationship might be described as `"model creation"` in one chunk and `"research development"` in another. The graph has no consistent type system.

**What you can do instead** is vector search — find entities semantically similar to "person who invented the Transformer" and let the LLM answer from the retrieved descriptions. This works, but it is retrieval, not graph traversal.

---

#### docling-graph / project KG: well-defined schema

This project's `kg/` module uses a fixed ontology for legal contract analysis, defined in `kg/legal/common/cuad_ontology.py`. Every extracted entity and relationship must conform to the schema.

**18 fixed vertex labels:**

```
Contract, Section, Clause, Party, Jurisdiction,
EffectiveDate, ExpirationDate, RenewalTerm,
LiabilityClause, IndemnityClause, PaymentTerm,
ConfidentialityClause, TerminationClause, GoverningLawClause,
Obligation, Risk, Amendment, ReferenceDocument
```

**35 typed, directional relationship types:**

```
PARTY_TO          — Party → Contract
GOVERNED_BY       — Contract → Jurisdiction
INDEMNIFIES       — Party → Party
HAS_TERMINATION   — Contract → TerminationClause
HAS_OBLIGATION    — Contract → Obligation
OBLIGATES         — Contract → Obligation
CAN_TERMINATE     — Party → Contract
AMENDS            — Contract → Contract
INCREASES_RISK_FOR — Risk → Party
... (35 total)
```

**What this enables — precise Cypher queries:**

```cypher
-- Find all parties to contracts governed by California law
MATCH (p:Party)-[:PARTY_TO]->(c:Contract)-[:GOVERNED_BY]->(j:Jurisdiction)
WHERE j.name = 'California'
RETURN p.name, c.name

-- Find contracts where Party A can terminate
MATCH (p:Party)-[:CAN_TERMINATE]->(c:Contract)
RETURN p.name, c.name

-- Find all obligations between two specific parties
MATCH (p1:Party)-[:OWES_OBLIGATION_TO]->(p2:Party)
WHERE p1.name = 'Acme Corp'
RETURN p2.name, p1.name
```

These queries are only possible because the schema is enforced — every `CAN_TERMINATE` edge is guaranteed to go from a `Party` to a `Contract`, every `GOVERNED_BY` goes from a `Contract` to a `Jurisdiction`.

---

#### Where each approach breaks down

**LightRAG flat ontology fails when:**

- You need **precise graph traversal** — "find all contracts where Party A can terminate" requires consistent typed relationships.
- You need **schema validation** — junk entities (`3.4% Decline`, `The Device`) get inserted alongside meaningful ones with no way to distinguish them at query time.
- You have **multiple document types** in one workspace — entities from legal docs, financial reports, and research papers share the same untyped graph, causing cross-domain pollution.
- You need **deduplication by type** — `Google Brain` (Organization) and `Google Brain Research` (unknown) are separate nodes because name normalisation is the only deduplication mechanism.

**Well-defined schema fails when:**

- You are processing **new or unknown domains** — extracting a medical record through a legal contract schema produces empty or nonsense results.
- The schema is **too narrow** — a relationship type not in the fixed set gets dropped or forced into the wrong type.
- Schema design is **wrong upfront** — a bad schema is worse than no schema because it silently misclassifies entities.

---

#### When to use each

| Use case | Recommendation |
|---|---|
| General-purpose RAG over mixed documents | LightRAG flat — schema-free, works immediately |
| Domain-specific Q&A requiring graph traversal | Well-defined schema — precise queries, consistent types |
| Legal contract analysis | Well-defined schema (as in this project's `kg/` module) |
| Research paper exploration | LightRAG flat — entity descriptions good enough for semantic search |
| Regulatory compliance checks ("does this contract contain X?") | Well-defined schema only |
| Exploratory knowledge discovery on unknown corpus | LightRAG flat — let the LLM decide what matters |

---

#### The hybrid approach

This project uses both:
- **LightRAG** (or RAG-Anything) for general document Q&A — flat ontology, any domain
- **Project `kg/` module** for legal contract analysis — fixed schema, Apache AGE, Cypher queries

The two graphs are separate. LightRAG's graph lives in the `lightrag_demo` AGE graph. The legal KG lives in its own AGE graph with typed nodes and relationships. They are queried by different retrieval paths depending on the question type.

---

### What does the entity-relationship output look like?

The following is real output from our demo run (`lightrag_demo.py`) using `qwen2.5:14b` on this input chunk:

> *"The Transformer architecture was introduced by Ashish Vaswani and colleagues at Google Brain in 2017. It uses multi-head self-attention mechanisms instead of recurrence. BERT, developed by Google AI Language in 2018, extended the Transformer by using bidirectional pre-training on large text corpora."*

#### Extracted entities

```
[Ashish Vaswani]                    type: Person
  "Ashish Vaswani, along with colleagues at Google Brain, introduced the
   Transformer model in 2017."

[Transformer Architecture]          type: Method
  "The Transformer is a deep learning model architecture that uses multi-head
   self-attention mechanisms for parallel computation across sequence positions."

[Google Brain]                      type: Organization
  "Google Brain is a research group within Google that develops and introduces
   advanced machine learning models."

[BERT Model]                        type: Method
  "BERT is a model developed by Google AI Language that enhances the Transformer
   with bidirectional pre-training on large text corpora."

[Google AI Language]                type: Organization
  "Google AI Language, part of Google's AI research division, developed and
   introduced BERT in 2018."

[Paper "Attention Is All You Need"] type: Content
  "The paper describes the introduction of the Transformer architecture by
   Ashish Vaswani et al."

[Multi-Head Self-Attention Mechanisms]  type: Method
  "A key feature of the Transformer allowing parallel computation across
   sequence positions."
```

#### Extracted relationships

```
Ashish Vaswani → Transformer Architecture
  keywords: academic collaboration, model creation
  "The Transformer was created by Ashish Vaswani at Google Brain."

Google Brain → Transformer Architecture
  keywords: model introduction, research development
  "Google Brain developed and introduced the Transformer Architecture."

Paper "Attention Is All You Need" → Transformer Architecture
  keywords: academic contribution, publication
  "The paper describes the Transformer architecture."

BERT Model → Transformer Architecture
  keywords: model enhancement, technological advancement
  "BERT builds upon the foundation laid by the Transformer."

BERT Model → Google AI Language
  keywords: model development, performance improvement
  "Google AI Language developed BERT to enhance the Transformer."

Multi-Head Self-Attention Mechanisms → Transformer Architecture
  keywords: innovative technology, parallel processing
  "Multi-head self-attention is the core mechanism of the Transformer."
```

#### What to notice

- **Entity descriptions accumulate across chunks.** If `Ashish Vaswani` appears in two chunks, his description in the database is `desc1<SEP>desc2`. When description count hits 8, an LLM summarisation pass merges them into one.
- **Entity types are inferred by the LLM**, not enforced by a schema. The same entity can be typed differently across ingestion runs if the surrounding context changes.
- **Workspace isolation.** Entities from different ingestion sessions in the same workspace are merged — so running the demo twice accumulates descriptions rather than replacing them.
- **Cross-case contamination.** In our demo, entities from the stock market example chunk (`3.4% Decline`, `Quarterly Earnings Report`) and the figure placeholder chunk (`Multi-Head Attention Mechanism`, `Queries (Q)`) all landed in the same workspace and appear alongside the clean Transformer entities.

---

### How is data stored in LIGHTRAG_VDB_ENTITY and LIGHTRAG_VDB_RELATION?

**Not JSON.** Both tables use plain `TEXT` columns with internal delimiter conventions. Here is the actual raw row data from our PostgreSQL instance:

#### LIGHTRAG_VDB_ENTITY — raw row

```sql
id:           'ent-35c1b18b9d0148c261e3a78ceb21f974'
workspace:    'lightrag_demo'
entity_name:  'Transformer Architecture'          -- VARCHAR(512), plain text
content:      'Transformer Architecture\n         -- TEXT, plain text
               The Transformer is a deep learning model architecture that uses
               multi-head self-attention mechanisms...'
content_vector: [0.021, -0.043, ...]              -- VECTOR(768), pgvector type
chunk_ids:    ['chunk-9c66fbe704ad6a189b64128bb955dd6e']  -- native PG ARRAY
file_path:    'unknown_source'                    -- TEXT
```

**`content` field format (entity):**
```
{entity_name}\n{description}
```
When the same entity appears in multiple chunks and descriptions are merged:
```
{entity_name}\n{desc_from_chunk_1}<SEP>{desc_from_chunk_2}
```
`<SEP>` is LightRAG's internal separator constant (`GRAPH_FIELD_SEP = "<SEP>"`). It is never JSON.

#### LIGHTRAG_VDB_RELATION — raw row

```sql
id:        'rel-034e8ef5f6fe9806ce3bc62882e7745a'
workspace: 'lightrag_demo'
source_id: 'Market Selloff'                       -- VARCHAR(512), plain entity name
target_id: 'Quarterly Earnings Report'            -- VARCHAR(512), plain entity name
content:   'financial performance,market reaction\tMarket Selloff\n  -- TEXT
            Quarterly Earnings Report\n
            Negative quarterly earnings from tech companies contributed...'
content_vector: [0.031, -0.012, ...]              -- VECTOR(768)
chunk_ids: ['chunk-d21a3132f9a0732fb3c60e39e6a06993']  -- native PG ARRAY
```

**`content` field format (relation):**
```
{keywords}\t{source_name}\n{target_name}\n{description}
```
- Keywords are comma-separated, followed by a **tab** (`\t`)
- Then source entity name, newline, target entity name, newline, description
- Again, never JSON — a tab + newline delimited string

#### Column types summary

| Column | Table | Type | Format |
|---|---|---|---|
| `entity_name` | VDB_ENTITY | `VARCHAR(512)` | Plain text |
| `source_id` / `target_id` | VDB_RELATION | `VARCHAR(512)` | Plain text (entity name) |
| `content` | Both | `TEXT` | Entity: `name\ndesc` / Relation: `keywords\tsrc\ntgt\ndesc` |
| `content_vector` | Both | `VECTOR(768)` | pgvector binary |
| `chunk_ids` | Both | `VARCHAR(255)[]` | Native PostgreSQL array, not JSON |
| `file_path` | Both | `TEXT` | Plain text |

#### Why this matters for querying

If you query these tables directly (e.g. for debugging or building a custom retrieval layer), you cannot `json_parse` the `content` field. You need to split on `\n` and `\t` and `<SEP>` to extract the structured parts:

```python
# Parse entity content
name, *desc_parts = content.split("\n", 1)
descriptions = desc_parts[0].split("<SEP>") if desc_parts else []

# Parse relation content
keywords_part, rest = content.split("\t", 1)
keywords = keywords_part.split(",")
source, target, description = rest.split("\n", 2)
```

---

### Does LightRAG process images?

**LightRAG does not process images directly.** Its `ainsert()` method takes text strings only — it never receives or inspects image data.

However, the Docling → LightRAG pipeline CAN handle images, but only by passing image content through a text layer first:

```
Image file / PDF with figures
        │
        ▼  Docling
        │
        ├── OCR path (scanned docs, screenshots of text)
        │       PDFMiner / RapidOCR extracts text from image pixels
        │       → text string passed to HybridChunker
        │
        └── VLM path (diagrams, charts, figures)
                VLM (llava, granite3.2-vision, etc.) describes the image
                → text description passed to HybridChunker
        │
        ▼  LightRAG ainsert(text_chunks)
        │
        Entity + relationship extraction (from the TEXT, not the image)
```

#### What entities get extracted from image descriptions?

A VLM description like:
> "This diagram shows the attention mechanism with three input vectors Q, K, V being multiplied and passed through a softmax operation to produce weighted output vectors."

Produces entities: `Attention Mechanism` (Method), `Q` (Data), `K` (Data), `V` (Data), `Softmax` (Method)
And relationships: `Q → input to → Attention Mechanism`, `Attention Mechanism → uses → Softmax`

These are reasonable but **limited** — the entities reflect what the VLM chose to describe, not the full visual structure of the diagram.

#### Quality is bounded by the VLM

| Docling image handling | What LightRAG receives | Entity quality |
|---|---|---|
| No VLM configured | `[Figure 3]` caption text only | Poor — entities are figure labels |
| OCR (scanned text) | Raw OCR output | Good for text content, poor for diagrams |
| VLM (llava:13b) | Prose description of image | Moderate — misses fine detail |
| VLM (granite3.2-vision) | Structured description | Better for tables and charts |

#### Demonstrated in the demo

The `fails_figure_placeholder` test case in `lightrag_demo.py` showed this exactly: LightRAG received only the Docling caption (`[Image content not available] Caption: Multi-head attention...`) and when asked "how does multi-head attention work mechanically?", the LLM answered from its own parametric knowledge rather than from the document — because the document contained no actual mechanism description, only a caption label. The answer was factually correct but **not grounded** in the ingested content.

#### Bottom line

- Without a VLM in Docling: image content is lost before LightRAG sees it
- With a VLM in Docling: LightRAG can extract entities from the description, but quality depends on VLM detail
- LightRAG itself needs no changes — the image handling is entirely Docling's responsibility

### Will it work with digital PDFs + Docling VLM?

Yes — this is the recommended production setup. Digital PDFs (born-digital, not scanned) combined with a Docling VLM is meaningfully better than any other combination.

#### What improves

- **Text content** — PDFMiner extracts text directly from the PDF byte stream. No OCR, no noise, no garbled words. Clean text produces clean entities in LightRAG. This alone eliminates the largest class of extraction failures.
- **Figures** — VLM describes diagram content as text. Figure entities that were completely lost without VLM are now at least partially represented in the graph.
- **Single-column body text** — works reliably end-to-end with no additional configuration.

#### What still doesn't fully work

- **Two-column layouts** — column mixing still occurs in `FAST` mode, reduced in `ACCURATE` mode. The underlying PDF coordinate ambiguity is a layout model problem, not a text extraction problem. Digital vs scanned makes no difference here.
- **Comparative table queries** — even with Docling extracting a table correctly, LightRAG stores each row as a separate entity with no inter-row relationships. A query like "which model has the best BLEU score?" requires reasoning across all rows simultaneously — the graph cannot do this. Confirmed in the demo: LightRAG gave the wrong answer on the markdown table test case.
- **Multi-page tables** — still split at page boundaries before LightRAG sees them.
- **VLM description quality** — figure entities are bounded by what the VLM chose to describe. Fine-grained quantitative data in charts (e.g. exact bar heights, axis values) is often missed or approximated.

#### Comparison across setups

| Setup | Body text | Figures | Tables | Comparative queries | Column mixing |
|---|---|---|---|---|---|
| Scanned PDF, no VLM | Poor (OCR noise) | Lost | Poor | Fails | Bad |
| Digital PDF, no VLM | Good | Lost | Moderate | Fails | Moderate |
| Digital PDF + VLM | Good | Partial | Moderate | Fails | Moderate |

#### Remaining failures are architectural

The failures that persist with digital PDF + VLM are not fixable by better document parsing. They are structural limitations of how LightRAG stores knowledge:

- Comparative queries fail because entities are stored independently — no ranked-above or better-than relationships are extracted from tables
- Multi-column text mixing is a layout model classification problem, not a text quality problem
- Multi-page table splitting is a page-by-page processing limitation in Docling itself

**Digital PDF + Docling VLM is the ceiling for this pipeline.** To go beyond it, you need either a different graph model that can represent tabular comparisons natively, or a post-processing step that converts tables to key-value triplets before ingestion (as described in [Scenario B — Multi-Level Hierarchy Header Collapse](#scenario-b--multi-level-hierarchy-header-collapse)).

### Limitations and Where It Will Fail

**Structural failures:**

- **Images and figures** — LightRAG ingests text only. If you feed it Docling's markdown output, figure content is lost (same as Docling without VLM). There is no built-in VLM modal processor — that's RAG-Anything's addition.
- **Tables** — if Docling exports tables as markdown grid text, LightRAG ingests the raw markdown. Cell relationships are treated as prose and often extracted as vague entities. Complex financial or multi-level tables produce poor graph nodes.
- **Column-mixed chunks** — if Docling produces bad chunks (column mixing), LightRAG ingests the garbled text and extracts garbled entities. Garbage in, garbage out.

**LLM extraction failures:**

- **Format drift** — smaller models (`mistral:7b`, `llama3.1:8b`) frequently deviate from the `entity<|#|>...` format, especially on long or complex chunks. The result is silently dropped tuples.
- **Overly generic entities** — on dense technical text, the LLM extracts vague entities (`The System`, `This Method`, `The Model`) that have low retrieval value.
- **Hallucinated relationships** — the LLM sometimes invents relationships not stated in the text, especially on ambiguous pronouns. The third-person/no-pronoun instruction in the prompt reduces but doesn't eliminate this.
- **Context overflow** — chunks that exceed the model's context window produce truncated extraction with no error. See [Context Window Management](#context-window-management).

**Scalability limitations:**

- **Ingestion is slow** — every chunk requires at least one LLM call. At 1200 tokens/chunk and a local 8B model doing ~20 tokens/s, a 100-page document (~300 chunks) takes ~30–60 minutes.
- **Entity merging is O(n) per new document** — as the graph grows, deduplication and description merging become increasingly expensive.
- **AGE graph traversal does not scale past ~100K nodes** on a single Postgres instance without query optimisation. For large corpora, graph traversal becomes the bottleneck.
- **`max_async=4`** limits parallel LLM calls. Increasing this helps throughput but requires more VRAM if models are loaded concurrently.

### Will It Scale?

For a **single-domain corpus up to ~50K chunks**: yes, with a properly indexed Postgres instance.

For **100K+ chunks or multi-domain corpora**: graph traversal and entity merging become bottlenecks. The practical ceiling depends on the quality of entity deduplication — if the LLM produces many near-duplicate entity names, the graph grows faster than the content warrants.

**Mitigation strategies:**
- Increase `chunk_token_size` to reduce chunk count (trade: less granular retrieval)
- Use `qwen2.5:14b` for cleaner entity naming (reduces near-duplicates)
- Partition by domain using the `workspace` field (separate graphs per domain)
- Add an entity normalisation post-processor to canonicalise names before insertion
