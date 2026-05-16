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
