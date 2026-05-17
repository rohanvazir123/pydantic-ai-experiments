# FAQ — docling-graph

## Table of Contents

- [Does docling-graph require a schema?](#does-docling-graph-require-a-schema)
- [Ontology and schema comparison with LightRAG](#ontology-and-schema-comparison-with-lightrag)
- [Why docling-graph needs a well-defined schema](#why-docling-graph-needs-a-well-defined-schema)
- [Why LightRAG uses a flat ontology](#why-lightrag-uses-a-flat-ontology)
- [What ontology does RAGAnything use?](#what-ontology-does-raganything-use)
- [How does the LLM detect entities and relationships with no ontology?](#how-does-the-llm-detect-entities-and-relationships-with-no-ontology)
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

## What ontology does RAGAnything use?

RAGAnything is a multimodal extension of LightRAG. It inherits LightRAG's **flat, schema-less ontology** and adds modal-specific processors (image captioners, table parsers, audio transcribers) that convert non-text content into text before feeding it into the same open-ended graph extraction pipeline.

| Feature | RAGAnything |
|---------|-------------|
| Ontology type | Flat, open-domain — inherited from LightRAG |
| Schema requirement | None — no entity types or relationship labels defined upfront |
| Multimodal input | Yes — images, tables, audio, video processed by modal-specific models |
| Extraction engine | LLM entity/relationship extraction on modal-normalised text |
| Graph backend | LightRAG graph store (same deduplication and merging logic) |

The modal processors (e.g. a VLM for images, a table parser for spreadsheets) produce a textual description of the content. That description is then processed exactly like any other text chunk — the LLM extracts entities and relationships from it with no domain-specific guidance. This means image content becomes searchable in the graph, but the extracted entities are still untyped ("ResNet-50", "accuracy metric", "benchmark dataset") rather than schema-enforced typed nodes.

**Implication:** RAGAnything is well suited for broad exploratory search over heterogeneous documents where the question is "find everything related to X." It is not suited for extracting precise, typed, validated facts where downstream logic depends on graph correctness.

---

## How does the LLM detect entities and relationships with no ontology?

This is one of the most important things to understand about open-domain graph extraction. The short answer: **yes, it relies entirely on the LLM's natural language pretraining.** There is no rule engine, no regex, no external NER model — just the LLM deciding what counts as an entity based on patterns learned from billions of tokens of text.

### What the extraction prompt actually does

LightRAG's extraction prompt looks roughly like this (simplified):

```
Given the following text, extract all entities and relationships.
For each entity, provide: name, type, description.
For each relationship, provide: source entity, target entity, relationship description, keywords.

Text:
{chunk_text}
```

No entity type vocabulary is provided. No allowed relationship labels are listed. The LLM fills in whatever it decides is an entity and whatever it decides is a relationship.

### How the LLM decides what is an entity

The model applies patterns learned during pretraining:

**Named entity recognition heuristics.** Pretraining on annotated corpora (Wikipedia, news, books) teaches the model that proper nouns (people, organisations, places, products) are canonical entity candidates. "OpenAI", "Paris", "GPT-4" get extracted reliably because they appear as named entities repeatedly in training data.

**Noun phrase salience.** Multi-word noun phrases that appear as the subject or object of sentences get elevated. "attention mechanism", "transformer architecture", "retrieval-augmented generation" — the model learned that these are semantically coherent concepts worth naming as nodes.

**Contextual repetition.** If a term appears multiple times in a chunk in different grammatical roles (subject, object, prepositional phrase), the model treats it as an anchor concept. A term mentioned once in passing is less likely to be extracted than one structuring an argument.

**No grounding to a type vocabulary.** Without a schema, the model picks entity type labels freely. On the same document in two different extraction runs you might see `Company`, `Organization`, `Tech_Firm`, and `AI_Company` all referring to the same thing. The model is not wrong — it is pattern-matching on the local context of each chunk, not enforcing global consistency.

### How the LLM decides what is a relationship

Relationships are inferred from grammatical dependency patterns the model learned during pretraining:

- **Subject-verb-object**: "OpenAI *developed* GPT-4" → `(OpenAI) -[DEVELOPED]→ (GPT-4)`
- **Prepositional attachment**: "GPT-4 *is used in* ChatGPT" → `(GPT-4) -[USED_IN]→ (ChatGPT)`
- **Appositive and copula**: "Sam Altman, *CEO of* OpenAI" → `(Sam Altman) -[CEO_OF]→ (OpenAI)`
- **Causal connectives**: "*because of* the attention mechanism, transformers scale" → `(attention mechanism) -[ENABLES]→ (transformers)`

The relationship label itself is generated by the model based on the verb or preposition in the local sentence — it is not drawn from a fixed vocabulary. Two chunks saying "X causes Y" and "X leads to Y" will likely produce different relationship labels for the same semantic fact.

### What this means in practice

| Consequence | Detail |
|-------------|--------|
| **Non-deterministic extraction** | Same chunk, different run → different entity names, different relationship labels. Temperature > 0 amplifies this. |
| **Synonym proliferation** | No schema means the same real-world entity accumulates multiple node identities over a corpus. LightRAG's post-hoc merging step reduces but does not eliminate this. |
| **Domain blindness** | The LLM cannot distinguish between a "company" and a "product" unless the local sentence makes it explicit. Schema-less extraction on a legal document treats a contractual clause and a contracting party with equal ambiguity. |
| **Works well for breadth** | Because it is unconstrained, it captures latent relationships a schema designer would never have thought to include. Great for exploratory "what is connected to X?" queries. |
| **Fails for precision** | If your downstream logic traverses `(Party)-[GOVERNS]→(Contract)` edges, you need a schema. Free extraction will produce `(party)`, `(contracting party)`, `(Party A)`, `(signatory)` as separate nodes for the same concept. |

### The core insight

The LLM is doing **implicit named entity recognition and relation extraction** — the same tasks that traditionally required hand-labelled training data and dedicated NER/RE models. The difference is that a large pretrained LLM has absorbed enough linguistic structure from its training corpus that it can perform these tasks zero-shot, without domain-specific fine-tuning. The cost of that flexibility is consistency: a dedicated NER model trained on your domain with a fixed label set will be more consistent and auditable than a general LLM prompted to extract freely.

---

## Which approach fits your use case?

**Choose docling-graph** if you are processing complex, high-dimensional documents — financial audits, legal contracts, scientific papers, clinical trial reports — where a missing edge or a hallucinated entity label will break downstream application logic. The schema cost is paid once and the extraction quality is deterministic and auditable.

**Choose LightRAG** if your goal is plug-and-play semantic search over a large unstructured corpus and you want to ask broad, exploratory questions without investing in ontology design. Acceptable for general knowledge retrieval; not acceptable where graph traversal correctness is a hard requirement.
