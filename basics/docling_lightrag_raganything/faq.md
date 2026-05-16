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
