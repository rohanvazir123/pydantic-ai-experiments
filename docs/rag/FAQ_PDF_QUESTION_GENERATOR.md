# FAQ — PDF Question Generator, LightRAG & RAGAnything

Code references: line numbers point to files under `rag/` in this repo.

---
## Table of Contents

- [Q1. What are the `pdf_processing_l1/l2/l3` folders and how were they generated?](#q1)
- [Q2. What is LightRAG and how does it differ from this project's RAG approach?](#q2)
- [Q3. What is RAGAnything and what does it add on top of LightRAG?](#q3)
- [Q4. What is MinerU and why is it used instead of Docling for this pipeline?](#q4)
- [Q5. How does the question generation prompt work and what does the output JSON look like?](#q5)
- [Q6. How would you use these generated questions as a gold dataset for the main RAG system?](#q6)

---


## PDF Question Generator & LightRAG Experiment

<a id="q1"></a>
**Q1. What are the `pdf_processing_l1/l2/l3` folders and how were they generated?**

These are output folders from the **PDF Question Generator** pipeline (`rag/ingestion/processors/pdf_question_generator.py`). Each folder corresponds to one CS168 lecture PDF that was run through the pipeline — `l1`, `l2`, `l3` are lecture numbers, not RAG processing levels.

**The PDFs processed:**

| Folder | PDF | Topic | Pages | Chunks |
|---|---|---|---|---|
| `pdf_processing_l1` | l1.pdf | Consistent Hashing | 12 | 121 |
| `pdf_processing_l2` | l2.pdf | Count-Min Sketch / Heavy Hitters | 14 | 14 |
| `pdf_processing_l3` | l3.pdf | kd-Trees / Curse of Dimensionality | 6 | 6 |

Source: CS168 — The Modern Algorithmic Toolbox (Stanford, Tim Roughgarden & Gregory Valiant, 2024).

**What each folder contains:**

```
pdf_processing_l1/                          ← most complete run (full RAGAnything mode)
├── kv_store_text_chunks.json               ← LightRAG's KV store: chunk_id → content
├── kv_store_llm_response_cache.json        ← cached LLM calls (avoid re-running)
├── graph_chunk_entity_relation.graphml     ← knowledge graph: entities + relationships
├── vdb_chunks.json                         ← vector DB entries for text chunks
├── vdb_entities.json                       ← vector DB entries for extracted entities
├── vdb_relationships.json                  ← vector DB entries for relationships
├── l1_questions.json                       ← generated questions (JSON)
└── l1_complete_output.txt                  ← full run log: context sent to LLM + raw response + questions

pdf_processing_l2/ and pdf_processing_l3/   ← simpler runs (fewer artefacts)
├── l{n}_questions.json
└── l{n}_complete_output.txt
```

**How the pipeline works (`pdf_question_generator.py`):**

```
PDF file
   │
   ▼
MinerU VLM parser  (or PyPDF2 fallback)
   │   ├── text blocks  → content_list items type="text"
   │   ├── tables       → content_list items type="table"
   │   ├── equations    → content_list items type="equation"
   │   └── images       → content_list items type="image"  (with figure captions)
   │
   ▼
LightRAG initialised in working_dir
   │   └── stores chunks in kv_store_text_chunks.json + builds knowledge graph
   │
   ▼
Multimodal processors (RAGAnything)
   │   ├── TableModalProcessor   → LLM describes table content
   │   ├── EquationModalProcessor → LLM explains equations
   │   └── ImageModalProcessor   → vision LLM captions figures
   │
   ▼
format_chunks_as_context()
   │   └── builds "[chunk_id=c1][page=0]\n<text>\n..." string (max 10,000 chars)
   │
   ▼
LLM called with QUESTION_GENERATION_PROMPT
   │   └── returns JSON: [{ question, supported_by: [c1,c3], difficulty: easy|medium|hard }]
   │
   ▼
Saved to {working_dir}/{name}_questions.json
         {working_dir}/{name}_complete_output.txt
```

**Two processing modes:**

- **`process_pdf_with_raganything`** (default) — uses the full `RAGAnything` wrapper which orchestrates LightRAG + MinerU + modal processors end to end. Produced the l1 folder with the GraphML knowledge graph.
- **`process_pdf_simple`** (flag `--simple`) — calls MinerU directly, then manually applies the modal processors, then feeds chunks to LightRAG. More transparent, easier to debug. Used for l2 and l3.

**LightRAG's role:**

LightRAG is a graph-based RAG library. Unlike this project's vector-only approach, LightRAG also:
- Extracts named entities and relationships from chunks via LLM
- Builds a knowledge graph (`graph_chunk_entity_relation.graphml`) where nodes are entities and edges are relationships
- Supports query modes: `local` (chunk-level), `global` (community summaries), `hybrid`

The `kv_store_*` and `vdb_*` JSON files are LightRAG's persistent storage format (file-based, no separate DB needed — useful for experiments).

**How to run it:**

```bash
# Process a PDF with Ollama (default)
python -m rag.ingestion.processors.pdf_question_generator path/to/lecture.pdf

# With OpenAI
python -m rag.ingestion.processors.pdf_question_generator --api-key YOUR_KEY path/to/lecture.pdf

# Simple mode (MinerU + modal processors directly)
python -m rag.ingestion.processors.pdf_question_generator --simple path/to/lecture.pdf

# List PDFs in a directory
python -m rag.ingestion.processors.pdf_question_generator --list-dir path/to/dir
```

**Relationship to the main RAG system:**

This pipeline is a standalone experiment — it does not share the PostgreSQL store or the Pydantic AI agent. It is used to generate question datasets from academic PDFs, which can then be used as gold datasets for evaluating the main RAG system (similar to the `GOLD_DATASET` in `test_retrieval_metrics.py`). The `test_rag_anything_pdfs_for_question_generation/` folder holds the source PDFs fed into this pipeline.

---

<a id="q2"></a>
**Q2. What is LightRAG and how does it differ from this project's RAG approach?**

LightRAG is a graph-based RAG framework that combines vector search with a knowledge graph. The key architectural difference from this project:

| Aspect | This project (pgvector RAG) | LightRAG |
|---|---|---|
| Storage | PostgreSQL — `documents` + `chunks` tables | File-based KV stores (`kv_store_*.json`) + GraphML |
| Search | Hybrid: vector (IVFFlat) + text (GIN/tsvector) | Graph traversal + vector similarity |
| Knowledge representation | Flat chunks with heading context | Entities + relationships + community summaries |
| Query modes | Single hybrid mode | `local` (chunk), `global` (community), `hybrid`, `naive` |
| Setup complexity | PostgreSQL + pgvector | No DB required — pure file-based for experiments |
| Scale | Millions of chunks via PostgreSQL indexes | Designed for smaller corpora in file-based mode |

LightRAG's graph extraction pipeline: for each chunk, an LLM call extracts entities (node: `ConsistentHashing`, type: `Algorithm`) and relationships (edge: `ConsistentHashing —USED_FOR→ WebCaching`). These are stored in the `.graphml` file. At query time, the graph is traversed to find related entities, then their associated chunks are retrieved — this can surface information that pure vector similarity misses (e.g. indirect relationships two hops away).

---

<a id="q3"></a>
**Q3. What is RAGAnything and what does it add on top of LightRAG?**

RAGAnything (`raganything` package) is a multimodal extension of LightRAG. It adds:

- **MinerU integration** — uses MinerU VLM (Vision Language Model) to parse PDFs with layout awareness: identifies text blocks, tables, equations, and figures with their captions, preserving reading order across columns.
- **Modal processors** — `TableModalProcessor`, `EquationModalProcessor`, `ImageModalProcessor` each take their respective content block and call an LLM/vision-LLM to generate a natural language description. This description is then treated as a text chunk and indexed into LightRAG.
- **`process_document_complete()`** — single call that orchestrates the full pipeline: MinerU parse → modal processing → LightRAG insert → knowledge graph build.

Without RAGAnything, a table in a PDF would either be ignored or stored as raw cell text. With RAGAnything, the table is described by the LLM ("This table compares the latency of consistent hashing variants across cluster sizes of 10, 100, and 1000 nodes...") and that description becomes a searchable, embeddable chunk.

---

<a id="q4"></a>
**Q4. What is MinerU and why is it used instead of Docling for this pipeline?**

Both MinerU and Docling are layout-aware PDF parsers that use ML models. They serve the same role — structured PDF extraction — but have different strengths:

| Aspect | MinerU (magic-pdf) | Docling |
|---|---|---|
| Figure handling | Vision LLM describes figures | Detects figures, limited description |
| Equation handling | LaTeX extraction | Basic equation detection |
| Table handling | Row/column structure + LLM description | Row/column structure (markdown) |
| GPU requirement | Recommended for VLM mode | CPU viable |
| Integration | RAGAnything-native | This project's main pipeline |
| Output format | `content_list` JSON (typed blocks) | `DoclingDocument` object |

MinerU is used in this pipeline because RAGAnything was built around it, and its vision LLM mode provides richer figure descriptions. Docling is used in the main ingestion pipeline because it integrates with `HybridChunker` for structure-aware chunking and `contextualize()`.

---

<a id="q5"></a>
**Q5. How does the question generation prompt work and what does the output JSON look like?**

The `QUESTION_GENERATION_PROMPT` sends up to 10,000 characters of chunk context to the LLM, formatted with chunk IDs:

```
[chunk_id=c1][page=0]
CS168: The Modern Algorithmic Toolbox Lecture #1: Introduction and Consistent Hashing

[chunk_id=c2][page=0]
Tim Roughgarden & Gregory Valiant

[chunk_id=c4][page=0]
1 Consistent Hashing
...
```

The LLM is instructed to return valid JSON only:

```json
{
  "questions": [
    {
      "question": "What are the key characteristics of consistent hashing?",
      "supported_by": ["c4", "c6"],
      "difficulty": "medium"
    },
    {
      "question": "How does consistent hashing solve the hot-spot problem in distributed caches?",
      "supported_by": ["c9"],
      "difficulty": "hard"
    }
  ]
}
```

`supported_by` lists the chunk IDs that contain the evidence for the answer — this is what makes the generated questions useful as a gold dataset: you know exactly which source chunks should be retrieved to answer each question.

The pipeline handles JSON parse failures by: (1) stripping markdown code fences, (2) falling back to line-by-line extraction of any line containing `?`.

---

<a id="q6"></a>
**Q6. How would you use these generated questions as a gold dataset for the main RAG system?**

The `l{n}_questions.json` files contain questions grounded in specific chunk IDs. To use them as evaluation data for the main RAG system:

1. **Map chunk IDs to document sources** — the chunk IDs (`c1`, `c2`...) correspond to content blocks in the lecture PDF. The PDF's `source` path in the `documents` table becomes the `relevant_sources` entry.

2. **Add to `GOLD_DATASET`** in `test_retrieval_metrics.py`:
```python
GOLD_DATASET = [
    # existing NeuralFlow entries...
    {
        "query": "What are the key characteristics of consistent hashing?",
        "relevant_sources": ["l1"],  # matches document_source containing "l1"
    },
    {
        "query": "How does Count-Min Sketch relate to the heavy hitters problem?",
        "relevant_sources": ["l2"],
    },
]
```

3. **Re-run the evaluation harness** — `test_retrieval_metrics.py` computes Hit Rate, MRR, NDCG across all queries including the new ones.

The `difficulty` field (`easy`/`medium`/`hard`) can be used to segment metrics — e.g. report NDCG@5 separately for hard questions to understand where the retriever struggles most.
