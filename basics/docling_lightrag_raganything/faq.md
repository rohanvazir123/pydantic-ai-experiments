# FAQ — Docling + LightRAG / RAG-Anything Evaluation

## Table of Contents

- [Setup](#setup)
  - [Context Window — 8GB VRAM](#context-window--8gb-vram)
  - [Models Pulled](#models-pulled)
- [Docling Architecture and Internals](#docling-architecture-and-internals)
  - [Internal Workflow](#internal-workflow)
  - [Customization and Tuning](#customization-and-tuning)
  - [Image Handling and Local VLMs](#image-handling-and-local-vlms)
- [Docling Chunking Failures](#docling-chunking-failures)
  - [Which documents demonstrate bad chunks?](#which-documents-demonstrate-that-docling-does-not-produce-correct-chunks)
  - [Failure Scenarios and Fixes](#failure-scenarios-and-fixes)
  - [Scripts and Results](#scripts-and-results)
- [Docling Output Format](#docling-output-format)
  - [Chunk JSON structure](#chunk-json-structure)
  - [How to spot column mixing in the output](#how-to-spot-column-mixing-in-the-output)
- [Does Docling work with research papers?](#does-docling-work-with-research-papers)
  - [Will a VLM fix the remaining issues?](#will-a-vlm-fix-the-remaining-issues)
  - [Do we need a specialist model like Nougat for equations?](#do-we-need-a-specialist-model-like-nougat-for-equations)
- [Will RAG-Anything do better than Docling on research papers?](#will-rag-anything-do-better-than-docling-on-research-papers)
- [LightRAG Architecture and Internals](#lightrag-architecture-and-internals)
  - [What is LightRAG?](#what-is-lightrag)
  - [Internal Pipeline](#internal-pipeline)
  - [Graph Ontology](#graph-ontology)
  - [LLM Prompts](#llm-prompts)
  - [Local LLM and Configurables](#local-llm-and-configurables)
  - [Context Window Management](#context-window-management)
  - [PostgreSQL Storage Schema](#postgresql-storage-schema)
  - [How Apache AGE is Used](#how-apache-age-is-used)
  - [Query Modes](#query-modes)
  - [Limitations and Where It Will Fail](#limitations-and-where-it-will-fail)
  - [Will It Scale?](#will-it-scale)

---

## Setup

### Context window — 8GB VRAM

Ollama defaults to a 2048-token context window, which is too small for processing full document chunks. We bumped it to 128K globally via a new `LLM_NUM_CTX` setting (default `131072`).

**How it works:** passed as `extra_body={"num_ctx": 131072}` in `ModelSettings` on every pydantic-ai `Agent` call. Only applied when `LLM_PROVIDER=ollama` — cloud providers (OpenAI, Anthropic) are unaffected.

**Where it's wired:**
- `rag/config/settings.py` — `llm_num_ctx` field (default 131072)
- `kg/extraction_pipeline.py` — all 5 extraction agents (`_make_agent()`)
- `rag/agent/rag_agent.py` — main RAG agent
- `rag/agent/kg_agent.py` — all KG agents (cypher, transformer, QA, fallback)

To override: set `LLM_NUM_CTX=<value>` in `.env`.

### Models pulled

**Embedding**

| Model | Command |
|---|---|
| `nomic-embed-text:latest` | `ollama pull nomic-embed-text` |

Configured via `EMBEDDING_MODEL=nomic-embed-text:latest` in `.env`.

**Inferencing**

| Model | VRAM | Use |
|---|---|---|
| `qwen2.5:14b` | ~8GB | KG extraction (`KG_LLM_MODEL`) — best structured JSON output at this size |
| `llama3.1:8b` | ~5.5GB | Default RAG chat model (`LLM_MODEL`) |
| `qwen2.5:7b` | ~5GB | Lighter alternative to qwen2.5:14b |
| `mistral:7b` | ~4.5GB | Fast alternative, 32K context |

`qwen2.5:14b` is set as the KG extraction model (`KG_LLM_MODEL=qwen2.5:14b` in `.env`) because it follows strict JSON schemas more reliably than `llama3.1:8b`.

---

## Docling Architecture and Internals

### Internal Workflow

Docling converts raw documents (PDF, DOCX, HTML, images, etc.) into a rich, structured intermediate representation called a `DoclingDocument`, then optionally chunks it for downstream use.

```
Raw file (PDF / DOCX / HTML / image)
        │
        ▼
┌─────────────────────────────┐
│   DocumentConverter         │  Entry point. Detects format,
│   (pipeline selection)      │  picks the right pipeline.
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Layout Analysis           │  PDF: PDFMiner extracts raw text
│   (page-by-page)            │  + coordinates. Layout model
│                             │  (TableFormer or DocLayNet-based)
│                             │  classifies regions: text, table,
│                             │  figure, header, footer, list.
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   Structure Recovery        │  Tables → TableItem with row/col
│                             │  metadata. Lists → ListItem.
│                             │  Sections → SectionHeaderItem.
│                             │  Figures → FigureItem (+ caption).
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│   DoclingDocument           │  Unified in-memory tree. Each
│   (intermediate repr)       │  element knows its type, text,
│                             │  bounding box, page number,
│                             │  and parent/child relationships.
└─────────────────────────────┘
        │
        ├──► Export: Markdown / JSON / HTML / DataFrame
        │
        └──► Chunker
                │
        ┌───────┴──────────────┐
        │                      │
  HybridChunker          HierarchicalChunker
  (token-aware,          (structure-aware,
   merges small           splits only at
   siblings)              section boundaries)
        │
        ▼
  List[Chunk]  →  embed → vector store
```

**Key facts:**
- Processing is **page-by-page** — this is the root cause of cross-page table splitting failures.
- The `DoclingDocument` tree is the authoritative representation. Markdown export is lossy — always work with the document object directly when fixing structural issues.
- Chunkers operate on the document tree, not raw text. They respect element boundaries by default but can still produce bad chunks when the tree itself is wrong.

### Customization and Tuning

Docling exposes two main customization surfaces: **pipeline options** (controls how the document is parsed) and **chunker options** (controls how the parsed document is split).

#### Pipeline Options

```python
from docling.document_converter import DocumentConverter, PdfPipelineOptions
from docling.datamodel.pipeline_options import TableFormerMode

options = PdfPipelineOptions()

# Table structure detection
options.table_structure_options.mode = TableFormerMode.ACCURATE   # slower, higher accuracy
options.table_structure_options.mode = TableFormerMode.FAST       # default, faster

# Cell matching — set False for tables with merged/multi-line cells
options.table_structure_options.do_cell_matching = False

# Fix overlapping cell bounding boxes (common in dense financial tables)
options.table_structure_options.correct_overlapping_cells = True

# OCR — enable for scanned documents
options.do_ocr = True
options.ocr_options.lang = ["en"]

# Image export — needed if you want to pass figures to a VLM
options.images_scale = 2.0
options.generate_page_images = True
options.generate_picture_images = True

converter = DocumentConverter(pipeline_options=options)
result = converter.convert("document.pdf")
```

#### Chunker Options

```python
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer import OpenAITokenizer

# HybridChunker: merges small siblings, splits large ones on token budget
chunker = HybridChunker(
    tokenizer=OpenAITokenizer(model_name="text-embedding-3-small"),
    max_tokens=512,        # hard token ceiling per chunk
    merge_peers=True,      # merge adjacent same-level elements
)

chunks = list(chunker.chunk(dl_doc=result.document))
```

#### When to use which chunker

| Scenario | Recommended chunker | Why |
|---|---|---|
| General mixed documents | `HybridChunker` | Balances size and structure |
| Long reports, books | `HierarchicalChunker` | Keeps section coherence |
| Tables only | Export to DataFrame directly | Chunkers lose row/cell context |
| Dense legal contracts | `HierarchicalChunker` + post-process | Preserves clause hierarchy |

#### Adapting to document complexity

- **Multi-column layouts**: Docling's layout model usually handles two-column academic PDFs correctly in `ACCURATE` mode. In `FAST` mode, column boundaries are often missed.
- **Dense tables**: Combine `TableFormerMode.ACCURATE` + `do_cell_matching=False` + `correct_overlapping_cells=True`.
- **Scanned documents**: Enable `do_ocr=True`. For better accuracy, use the EasyOCR or Tesseract backend.
- **Legal contracts**: Use `HierarchicalChunker` — it respects numbered section headers and keeps sub-clauses with their parent.
- **Short chunks losing context**: Increase `max_tokens` or set `merge_peers=True` to merge adjacent small elements.

### Image Handling and Local VLMs

#### Does Docling handle images?

Yes, with caveats:

| Content type | Default behaviour | With VLM |
|---|---|---|
| Figures / diagrams | Detected as `FigureItem`, caption extracted, image **skipped** by text pipeline | VLM generates description |
| Tables as images | Parsed by TableFormer (layout model), not a VLM | VLM improves accuracy |
| Scanned pages | OCR extracts text, images ignored | VLM can replace OCR entirely |
| Charts / infographics | Caption extracted, data **lost** | VLM describes or extracts data |

By default, Docling extracts figure captions and bounding boxes but does **not** describe image content. To get descriptions, you must wire in a VLM.

#### Local VLMs you can use

| Model | Pull command | Strength |
|---|---|---|
| `llava:13b` | `ollama pull llava:13b` | General vision-language, good at figures |
| `llava:7b` | `ollama pull llava:7b` | Lighter, ~4.5GB VRAM |
| `moondream2` | `ollama pull moondream2` | Very small (~2GB), fast, weaker reasoning |
| `granite3.2-vision:2b` | `ollama pull granite3.2-vision:2b` | IBM's Docling-aware VLM, best for tables |
| `SmolDocling-VLM` | HuggingFace | Purpose-built for document understanding |

#### Wiring a VLM for figure descriptions

```python
from docling.document_converter import DocumentConverter, PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling.models.picture_description_api_model import PictureDescriptionApiOptions

# Enable image export so figures are available as PIL images
pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True
pipeline_options.images_scale = 2.0

# Point to a local Ollama VLM endpoint
picture_options = PictureDescriptionApiOptions(
    url="http://localhost:11434/v1/chat/completions",
    params={"model": "llava:13b"},
    prompt="Describe this figure concisely for a retrieval system.",
)
pipeline_options.picture_description_options = picture_options

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfPipelineOptions(**pipeline_options.__dict__)}
)
result = converter.convert("document_with_figures.pdf")

# Figure descriptions are now in FigureItem.annotations
for item, _ in result.document.iterate_items():
    if hasattr(item, "annotations") and item.annotations:
        print(item.annotations[0].text)
```

**Practical note:** VLM processing is slow — expect ~5–30s per figure depending on model size and GPU. For bulk ingestion, filter to only `FigureItem` objects and batch them.

---

## Docling Chunking Failures

### Which documents demonstrate that Docling does not produce correct chunks?

Docling's `HybridChunker` has several known failure modes. The following document types and sources reliably expose them.

#### 1. Two-Column Academic PDFs — Column Mixing

**Failure:** Text from the left and right columns gets interleaved mid-sentence, producing chunks that mix unrelated paragraphs.

**Source:** arxiv.org — any NeurIPS/ICML/ICLR paper in the standard two-column format. Download the PDF from the paper's abstract page.

**How to verify:** Run the PDF through `DoclingChunker`, then `print(chunk.content)` for chunks near column boundaries. Sentences from unrelated paragraphs will be merged.

#### 2. Financial Documents with Complex Tables — Table Splitting

**Failure:** Table rows and cells are split across chunk boundaries, breaking tabular data context and making the chunks uninterpretable without surrounding rows.

**Source:** SEC EDGAR (sec.gov/edgar) — any company's 10-K annual report. These have multi-page tables with merged cells.

#### 3. Legal Contracts with Nested Indentation — Hierarchy Loss

**Failure:** Sub-clauses lose their parent clause context after chunking. A chunk may contain clause 3.2(a) with no reference to clause 3.2 or section 3.

**Source:** CUAD dataset — the contracts at `rag/documents/legal/` already demonstrate this. Visible in `test_legal_retrieval.py` retrieval failures.

#### 4. Scanned / Image-Heavy PDFs — OCR Artifacts

**Failure:** The chunker splits on OCR noise (garbled characters, line-break artifacts), producing junk chunks with broken words or symbols.

**Source:** IRS forms from irs.gov — older publications that are scan-based rather than digitally typeset.

#### 5. Documents with Running Headers and Footers — Content Contamination

**Failure:** Page numbers, chapter headers, and footers get injected into content chunks, polluting retrieval with meaningless text.

**Source:** NIST publications from csrc.nist.gov — long technical documents with repeated running headers.

#### Quick test

The fastest failure to reproduce is the arxiv two-column mixing:

```python
from rag.ingestion.chunkers.docling import DoclingChunker

chunker = DoclingChunker()
chunks = chunker.chunk("path/to/arxiv_paper.pdf")
for chunk in chunks:
    print(chunk.content[:300])
    print("---")
```

Look for chunks where a sentence abruptly switches topic mid-paragraph — that is the column boundary being crossed.

### Failure Scenarios and Fixes

See `docling_failures.md` for detailed code-level fixes. Summary:

#### Scenario A — Multi-Page Table Splitting

**Root cause:** Docling processes page-by-page. A table crossing a page boundary becomes two separate `TableItem` objects.

**Fix 1 — Heuristic header-matching post-processor:** After conversion, iterate `DoclingDocument` elements and merge adjacent `TableItem` objects that share the same column count. See `docling_failures.md` → Scenario A, Fix 1.

**Fix 2 — VLM pipeline override:** Replace the layout model with `Granite-Docling` or `SmolDocling-VLM` and set `TableFormerMode.ACCURATE`. These models detect spatial continuity across page bounds.

```python
options.table_structure_options.mode = TableFormerMode.ACCURATE
options.table_structure_options.do_cell_matching = False
```

**Fix 3 — `repeat_table_header=True` in HybridChunker:** Injects the table header row into every chunk that contains a table continuation, preserving column context even when rows are split.

```python
chunker = HybridChunker(repeat_table_header=True)
```

#### Scenario B — Multi-Level Hierarchy Header Collapse

**Root cause:** When Docling exports to Markdown, hierarchical table headers (nested row keys) flatten to a single header row, losing the relationship between parent and child rows.

**Fix 1 — Export directly to DataFrame:** Use `table.export_to_dataframe(doc=result.document)` and `ffill()` to propagate hierarchy down blank cells.

**Fix 2 — Key-value triplet serialisation:** Serialize tables as explicit assertion strings (`Revenue (2024, Q1): 130`) rather than Markdown grids so LLMs preserve full context per value.

#### Scenario C — Inconsistent and Merged Cell Layouts

**Root cause:** Tables with varying cell heights, inline footnotes, or asymmetric columns produce structural errors during coordinate-based cell matching.

**Fix 1 — Use `HybridChunker` (not `RecursiveCharacterTextSplitter`):** The native chunker inspects the document tree and prevents splits inside a row or between a footnote and its table.

**Fix 2 — HTML serialisation:** Export tables as HTML (`table.export_to_html()`) which preserves `colspan`/`rowspan` attributes that Markdown cannot represent.

**Fix 3 — Structural heuristic parameters:**
- `do_cell_matching=False` — prevents multi-line cell content from leaking into neighbouring blocks
- `correct_overlapping_cells=True` — resolves overlap bounding errors in dense financial rows

---

### Scripts and Results

#### Scripts

| Script | Purpose |
|---|---|
| `download_demo_documents.py` | Downloads 3 PDFs that demonstrate failure modes |
| `produce_bad_chunks.py` | Runs default Docling settings, flags anomalous chunks |
| `produce_good_chunks.py` | Applies fixes per failure type, saves improved chunks |

#### Documents Downloaded

| File | Failure demonstrated |
|---|---|
| `attention_is_all_you_need.pdf` | Two-column column mixing (NeurIPS 2017) |
| `bert_paper.pdf` | Two-column column mixing (NAACL 2019) |
| `nist_sp800_53.pdf` | Header/footer contamination (NIST SP 800-53 Rev 5) |
| `q4_financial_report.pdf` | Table splitting (internal Q4 business review) |

#### Results (bad chunks run)

| Document | Total chunks | Flagged anomalies | Failure type |
|---|---|---|---|
| `attention_is_all_you_need.pdf` | 67 | 2 | two_column_mixing |
| `bert_paper.pdf` | 97 | 1 | two_column_mixing |
| `nist_sp800_53.pdf` | 1998 | **135** | header_footer_contamination |
| `q4_financial_report.pdf` | 25 | 0 | table_splitting |

#### What worked / what didn't

| Fix | Status | Notes |
|---|---|---|
| `TableFormerMode.ACCURATE` | **Works** | Better layout analysis; BERT paper produced 3 extra chunks showing finer boundary detection |
| `repeat_table_header=True` | **Works** | Table header rows are injected into every continuation chunk |
| `strip_headers_footers()` via doc tree mutation | **Did not work** | `doc.body.children` manipulation does not remove items from the document in Docling 2.x — the internal ref structure is more complex |
| Post-hoc chunk filtering for headers/footers | **TODO — revisit** | Simpler and more reliable: filter chunks by short length + all-caps or page-number patterns after chunking, without touching the document tree |

#### Output locations

```
output/docling/bad_chunks/   — default chunking, anomaly-flagged JSON per document + summary.json
output/docling/good_chunks/  — fixed chunking JSON per document + summary.json
```

---

## Docling Output Format

### Chunk JSON structure

Each chunk produced by `HybridChunker` is serialised to JSON with the following structure:

```json
{
  "index": 0,
  "text": "The plain text content of the chunk.",
  "char_count": 173,
  "meta": {
    "schema_name": "docling_core.transforms.chunker.DocMeta",
    "version": "1.0.0",
    "doc_items": [...],
    "headings": ["Section heading this chunk falls under"],
    "origin": {
      "mimetype": "application/pdf",
      "binary_hash": 1234567890,
      "filename": "document.pdf"
    }
  }
}
```

#### Key fields

| Field | Description |
|---|---|
| `text` | Plain text content of the chunk — this is what gets embedded and retrieved |
| `char_count` | Character length |
| `meta.headings` | Section heading(s) this chunk belongs to — useful for prepending context to the chunk before embedding |
| `meta.origin` | Source filename, mime type, and binary hash of the original document |
| `meta.doc_items` | List of `DoclingDocument` tree elements that make up this chunk (see below) |

#### `doc_items` — the structural backbone

Each entry in `doc_items` corresponds to one element from the `DoclingDocument` tree:

```json
{
  "self_ref": "#/texts/1",
  "label": "text",
  "prov": [
    {
      "page_no": 1,
      "bbox": {
        "l": 124.3, "t": 717.8, "r": 487.9, "b": 679.7,
        "coord_origin": "BOTTOMLEFT"
      },
      "charspan": [0, 173]
    }
  ]
}
```

| Field | Description |
|---|---|
| `self_ref` | Pointer into the `DoclingDocument` tree (e.g. `#/texts/1`, `#/tables/0`) |
| `label` | Element type: `text`, `table`, `section_header`, `list_item`, `picture`, `page_header`, `page_footer`, etc. |
| `prov[].page_no` | Page the element appears on |
| `prov[].bbox` | Bounding box in PDF coordinates (bottom-left origin). `l`=left, `t`=top, `r`=right, `b`=bottom |
| `prov[].charspan` | Character range within `text` that this bounding box covers |

### How to spot column mixing in the output

Column mixing is visible in `doc_items[].prov`. A clean single-column chunk has one `prov` entry per `doc_item`. A mixed chunk has **multiple `prov` entries with non-overlapping `bbox` x-ranges** on the same page — text fragments from physically separate columns stitched together.

Example from `attention_is_all_you_need.pdf` (flagged chunk):

```
doc_item has 5 prov entries on page 1:
  bbox l=116  → r=216   (left column, row 1)
  bbox l=230  → r=309   (right column, row 1)
  bbox l=126  → r=210   (left column, row 2)
  bbox l=323  → r=407   (right column, row 2)
  bbox l=422  → r=497   (right column, row 2 continued)
```

The alternating `l` (left) values — 116, 230, 126, 323, 422 — jumping between ~120 and ~320 confirm text from two separate columns is merged into one chunk.

**Practical use:** When debugging retrieval quality, inspect `meta.doc_items[].prov` to diagnose whether a bad chunk is caused by column mixing (non-contiguous bounding boxes), table splitting (label=`table` with very few rows), or header contamination (label=`page_header` or `page_footer`).

---

## Does Docling work with research papers?

Yes — but with caveats depending on paper type.

### What Docling gets right

- Section headers, abstract, conclusion, references — cleanly extracted
- Single-column sections (most of the body in standard ML/NLP/CS papers) work correctly
- Figure captions are detected and attached to their `FigureItem`
- Tables contained within a single page are handled well

### What it struggles with

- **Author blocks** — the author name grid on page 1 is typically two-column and gets interleaved. This is what was flagged in the "Attention Is All You Need" run. It is noise, not content, so it rarely affects retrieval quality.
- **Figures** — image content is not described unless you wire a VLM. The caption is captured but the chart/diagram data is lost.
- **Multi-column tables** — dense tables that span columns (more common in medical and clinical papers than ML papers) get mixed.
- **Equations across columns** — mathematical notation that spans column boundaries can produce garbled text.

### Practical reality from our test run

"Attention Is All You Need" produced **67 chunks with only 2 flagged** — both were the author block and a fragment of the reference list, not the core content. Introduction, method, experiments, and results all chunked correctly.

### When it genuinely fails

| Paper type | Risk |
|---|---|
| Standard ML/NLP/CS papers | Low — mostly fine for RAG |
| Biomedical / clinical trial papers | Medium — complex multi-column result tables |
| Papers with heavy mathematical notation spanning columns | Medium — equations can get garbled |
| Papers with many figures carrying quantitative data | High — figure data is lost without a VLM |

### Recommendation

For standard ML/NLP research paper RAG: Docling is good enough. Use `TableFormerMode.ACCURATE` and expect minor noise in author blocks and reference lists. For biomedical or equation-heavy papers, add a post-hoc filter to drop very short chunks and inspect `prov` bounding boxes to catch column mixing.

### Will a VLM fix the remaining issues?

Partially — it depends on the failure type.

**Author block / column mixing** — VLM does not help. This is a layout analysis problem, not a vision understanding problem. The text is already extracted by the text pipeline, just mis-ordered. VLM only processes image content, so it never sees this.

**Figures and charts** — Yes, fully fixed. VLM describes the figure and injects the description as text into the chunk. Quantitative data from charts and diagrams becomes queryable.

**Multi-column tables** — Depends on how Docling classified the table:
- If Docling fell back to treating the table as an image (`FigureItem`) — VLM can describe it.
- If Docling correctly identified it as a `TableItem` but got cell ordering wrong — VLM is not in the loop. The table goes through TableFormer, not the VLM.

**Mathematical equations** — Depends on the model. `llava:13b` produces vague descriptions ("a mathematical expression"). `granite3.2-vision` is better but still not LaTeX-accurate. For proper equation extraction, use `Nougat` (Meta) — a specialist model purpose-built for scientific PDF math.

#### Summary

| Problem | VLM fixes it? |
|---|---|
| Column mixing (author blocks, body text) | No |
| Figure content (charts, diagrams) | Yes |
| Multi-column tables classified as images | Yes |
| Multi-column tables classified as TableItems | No — goes through TableFormer, not VLM |
| Mathematical equations | Partially — description only, not LaTeX |

#### Bottom line

For a research paper RAG system, wiring a VLM gets you figure descriptions but does not solve the fundamental layout analysis failures. The remaining gaps (column mixing, equation accuracy) require either a specialist model (`Nougat` for equations) or accepting the noise and relying on the surrounding chunks for retrieval quality.

### Do we need a specialist model like Nougat for equations?

Not necessarily — it depends on what you need from the equations.

**If you just need equations to be retrievable:** a VLM description like "scaled dot-product attention formula with queries, keys and values" is usually enough. The chunk will match semantic queries even without LaTeX. Most RAG use cases fall here.

**If you need LaTeX-accurate equations** (math/physics papers where the actual formula matters for downstream computation or rendering), then yes — reach for a specialist model:

| Model | Type | Strength | Notes |
|---|---|---|---|
| **Nougat** (Meta) | Local | Best accuracy, full PDF end-to-end, outputs clean Markdown + LaTeX | Slow; can hallucinate on non-equation content |
| **GOT-OCR** | Local | Newer, faster, handles equations + tables + charts in one model | Competitive with Nougat |
| **Pix2Tex** | Local | Lightweight LaTeX OCR, equation-only | Fast; good if Docling already isolated equations as `FigureItem` images |
| **MathPix** | Commercial | Most accurate | Not local |

**Practical recommendation:** Use VLM descriptions for general research paper RAG. Only add Nougat or GOT-OCR if your queries are themselves mathematical or if you need to render or compute with the extracted equations.

---

## Will RAG-Anything do better than Docling on research papers?

Partially — it depends on the failure type.

**Where RAG-Anything does better:**

- **Figures** — dedicated modal processors per content type. Figures go through a VLM pipeline by design, not as an optional add-on wired manually.
- **Tables** — dedicated table processor that exports to structured formats before chunking, rather than relying on TableFormer's coordinate-based cell matching. Better at complex multi-column table layouts.
- **Multi-modal coherence** — text, tables, figures, and audio are processed in separate pipelines then recombined. A bad table parse does not corrupt the surrounding text chunks.

**Where it won't help:**

- **Column mixing** — RAG-Anything still depends on an underlying PDF text extractor (Docling, PyMuPDF, or PDFMiner). If the extractor mis-orders columns, RAG-Anything inherits that problem. It does not fix layout analysis.
- **Equations** — same situation as Docling + VLM unless a specialist math processor (Nougat, GOT-OCR) is explicitly configured.

**Summary:**

| Problem | Docling alone | Docling + VLM | RAG-Anything |
|---|---|---|---|
| Column mixing | Partial (ACCURATE mode) | No improvement | No improvement |
| Figures / charts | Lost | Fixed | Fixed (built-in) |
| Complex tables | Partial | No improvement | Better |
| Equations | Garbled | Description only | Description only |
| Multi-modal coherence | Poor | Moderate | Good |

**The honest caveat:** Task C is specifically about evaluating RAG-Anything empirically. The real question is which PDF backend it uses and whether its table processor actually outperforms TableFormer on the same documents we already tested. We will find out rather than rely on claims.

---

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
