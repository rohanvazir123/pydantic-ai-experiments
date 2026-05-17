# FAQ — docling-graph

## Table of Contents

- [Does docling-graph require a schema?](#does-docling-graph-require-a-schema)
- [Ontology and schema comparison with LightRAG](#ontology-and-schema-comparison-with-lightrag)
- [Why docling-graph needs a well-defined schema](#why-docling-graph-needs-a-well-defined-schema)
- [Why LightRAG uses a flat ontology](#why-lightrag-uses-a-flat-ontology)
- [Which approach fits your use case?](#which-approach-fits-your-use-case)

---

## Does docling-graph require a schema?

Yes — a well-defined schema is mandatory. This is the sharpest architectural difference between docling-graph and LightRAG, which operates on a flat, schema-less ontology. The two frameworks target entirely different paradigms of graph-based extraction and retrieval.

---

## Ontology and schema comparison with LightRAG

| Feature | docling-graph | LightRAG |
|---|---|---|
| Ontology type | Strict, domain-specific, hierarchical | Flat, open-domain, dynamic |
| Schema requirement | Mandatory — enforced via Pydantic models | None — prompt-driven open extraction |
| Extraction engine | Deterministic layout parsing + schema-constrained LLM/VLM | Open-ended LLM entity/relationship extraction |
| Primary goal | Data integrity: precise, validated facts (e.g. matching a chemical formula to a specific test result) | Contextual discovery: scalable semantic search across large document pools via local/global/hybrid retrieval |
| Cypher queries | Precise and predictable — fixed labels and relationship types | Impractical — entity types and relationship keywords vary across extraction runs |
| Cross-domain use | No — the schema is domain-specific by design | Yes — works on any corpus without upfront configuration |
| Setup cost | High — requires domain expertise to design the schema | Near zero — ingest raw text and query immediately |

---

## Why docling-graph needs a well-defined schema

docling-graph is engineered to convert complex documents into exact, validated structured data. It binds Docling's layout extraction to Pydantic templates, which enforces structure at every step of the pipeline.

**Type safety and constraints.** Your Pydantic models define exactly which entity types can exist, what attributes they carry, and which relationship edges are permitted between them. The extraction engine cannot produce a node or edge that the schema doesn't allow.

**Schema enforcement at extraction time.** The underlying LLM or VLM is constrained to emit output that conforms to the Pydantic schema during extraction — not as a post-processing filter, but as a hard generation constraint. This eliminates a whole class of hallucination: the model cannot invent entity types or relationship labels that aren't in the schema.

**Preventing synonym proliferation.** Without a schema, the same real-world concept accumulates multiple inconsistent node labels across documents: `AI_Company`, `Tech_Firm`, `Organization`, `technology company`. An explicit schema collapses these to a single canonical type at extraction time, before they ever enter the graph. This is categorically more reliable than post-hoc entity resolution, which tries to undo the damage after the fact.

---

## Why LightRAG uses a flat ontology

LightRAG is optimised for high-throughput, low-friction GraphRAG over large unstructured text corpora. It bypasses traditional database schemas entirely.

**Zero-configuration ingestion.** You can feed LightRAG a raw text file with no schema configuration at all. The LLM scans each chunk and extracts whatever entities and relationships it finds — people, places, events, concepts — without any upfront ontology design.

**Flat key-value entity profiles.** Extracted entities are stored as generalised key-value records: the key is the entity name, the value is a merged text description accumulated from every chunk where the entity appears. There are no typed attributes, no mandatory fields, no foreign key constraints.

**Post-hoc deduplication.** Instead of enforcing type constraints upfront, LightRAG runs a deduplication and description-merging step after extraction. When the same entity appears under slightly different names across chunks, descriptions are merged and the graph is compacted. This works reasonably well for simple cases but cannot resolve deep synonym ambiguity the way a schema-enforced extraction can.

The trade-off is intentional: LightRAG sacrifices the precision of a typed graph in exchange for the ability to work on any domain, any document, without spending days designing an ontology.

---

## Which approach fits your use case?

**Choose docling-graph** if you are processing complex, high-dimensional documents — financial audits, legal contracts, scientific papers, clinical trial reports — where a missing edge or a hallucinated entity label will break downstream application logic. The schema cost is paid once and the extraction quality is deterministic and auditable.

**Choose LightRAG** if your goal is plug-and-play semantic search over a large unstructured corpus and you want to ask broad, exploratory questions without investing in ontology design. Acceptable for general knowledge retrieval; not acceptable where graph traversal correctness is a hard requirement.
