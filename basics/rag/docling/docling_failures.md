# Docling Chunking Failures

## Table of Contents

- [Scenario A — Multi-Page Table Splitting](#scenario-a--multi-page-table-splitting)
- [Scenario B — Multi-Level Hierarchy Header Collapse](#scenario-b--multi-level-hierarchy-header-collapse)
- [Scenario C — Inconsistent and Merged Cell Layouts](#scenario-c--inconsistent-and-merged-cell-layouts)
- [Scenario D — Two-Column Academic PDF Column Mixing](#scenario-d--two-column-academic-pdf-column-mixing)
- [References](#references)

---

## Scenario A — Multi-Page Table Splitting

Docling processes layout page-by-page. Tables that cross a page boundary are fractured into separate `TableItem` objects, one per page. [1, 2]

### Fix 1 — Heuristic header-matching post-processor

Do not rely on Markdown text output. Intercept the native `DoclingDocument` structure — it retains structural cell metadata (`row_span`, `col_span`). Iterate elements and merge adjacent `TableItem` objects that share the same column count before chunking. [1, 2, 3, 4]

```python
from docling.document_converter import DocumentConverter
from docling.datamodel.document import TableItem

def merge_multipage_tables(doc):
    refined_tables = []
    current_table = None

    for item in doc.element_iterable():
        if isinstance(item, TableItem):
            if current_table is None:
                current_table = item
            else:
                # Same column count → treat as continuation of the same table
                if len(item.data.columns) == len(current_table.data.columns):
                    current_table.data = pd.concat(
                        [current_table.data, item.data], ignore_index=True
                    )
                else:
                    refined_tables.append(current_table)
                    current_table = item
        else:
            if current_table:
                refined_tables.append(current_table)
                current_table = None

    return refined_tables
```

### Fix 2 — VLM pipeline override

Replace the default layout model with `TableFormerMode.ACCURATE`. This activates the high-fidelity visual cell recovery path and detects spatial continuity across page bounds better than the default coordinate-only approach. [2, 4, 5]

```python
from docling.document_converter import DocumentConverter, PdfPipelineOptions
from docling.datamodel.pipeline_options import TableFormerMode

options = PdfPipelineOptions()
options.table_structure_options.mode = TableFormerMode.ACCURATE
options.table_structure_options.do_cell_matching = False  # avoids strict per-page box clipping

converter = DocumentConverter(pipeline_options=options)
```

### Fix 3 — `repeat_table_header=True` in HybridChunker

Injects the table header row into every chunk that contains a table continuation, preserving column context even when rows are split across chunks.

```python
from docling.chunking import HybridChunker

chunker = HybridChunker(repeat_table_header=True)
chunks = list(chunker.chunk(dl_doc=result.document))
```

---

## Scenario B — Multi-Level Hierarchy Header Collapse

When Docling exports to Markdown, hierarchical table headers (nested row keys) flatten to a single header row, stripping the relationship between parent and child rows. [2, 4]

### Fix 1 — Export directly to DataFrame

Convert `TableItem` objects to Pandas DataFrames instead of parsing raw Markdown strings. Use `ffill()` to propagate hierarchy values down across blank cells caused by merged rows. [2, 4, 6, 7]

```python
# Export the parsed table into a structured dataframe
df = table.export_to_dataframe(doc=result.document)

# Propagate hierarchy values down across rows that were merged in the original
df.iloc[:, 0] = df.iloc[:, 0].ffill()
df.iloc[:, 1] = df.iloc[:, 1].ffill()
```

### Fix 2 — Key-value triplet serialisation

For graph extractors and LLM pipelines, serialise tables as explicit assertion strings rather than Markdown grids. Each value then carries its full contextual path. [8, 9]

```
Revenue (2024, Q1): 130
Revenue (2024, Q2): 155
Revenue (2025, Q1): 140
Revenue (2025, Q2): 135
```

---

## Scenario C — Inconsistent and Merged Cell Layouts

Tables with varying cell heights, inline footnotes, or asymmetric columns cause structural errors during coordinate-based cell matching. [4, 10]

### Fix 1 — Use HybridChunker instead of RecursiveCharacterTextSplitter

The native chunker inspects the document tree and prevents splits inside a row or between a footnote and its table. [8, 9, 11, 12, 13]

```python
from docling.chunking import HybridChunker

# Preserves unified logical element blocks, table captions, and footnotes
chunker = HybridChunker()
chunks = list(chunker.chunk(dl_doc=result.document))
```

### Fix 2 — HTML serialisation

Markdown cannot represent cells with internal line breaks, multiple paragraphs, or `colspan`/`rowspan`. Export tables as HTML to preserve these attributes for LLM interpretation. [2, 4, 14, 15]

```python
# Preserves colspan/rowspan properties that Markdown cannot represent
html_table_string = table.export_to_html()
```

### Fix 3 — Structural heuristic parameters

For dense tables where cell content leaks into neighbouring blocks or bounding boxes overlap: [16]

- **`do_cell_matching=False`** — prevents Docling from clipping multi-line cell content back into rigid bounding boxes, stopping content from spilling into adjacent cells. [4, 16, 17]
- **`correct_overlapping_cells=True`** — resolves overlap bounding errors common in dense financial tables.

```python
options = PdfPipelineOptions()
options.table_structure_options.do_cell_matching = False
options.table_structure_options.correct_overlapping_cells = True
```

---

## Scenario D — Two-Column Academic PDF Column Mixing

Two-column PDFs (standard NeurIPS/ICML/ICLR format) produce chunks where text from the left and right columns is interleaved mid-sentence. Sentences from unrelated paragraphs are merged into a single chunk.

**Root cause:** `FAST` mode (the default) uses a lighter layout model that does not reliably detect column boundaries. It reads text left-to-right across the full page width, treating both columns as a single block. Reading order is assigned before column regions are separated.

### Fix 1 — Switch to `TableFormerMode.ACCURATE`

`ACCURATE` activates the full `DocLayNet`-based classifier, which identifies column regions as separate layout blocks first, then assigns reading order within each column independently.

```python
from docling.document_converter import DocumentConverter, PdfPipelineOptions
from docling.datamodel.pipeline_options import TableFormerMode

options = PdfPipelineOptions()
options.table_structure_options.mode = TableFormerMode.ACCURATE

converter = DocumentConverter(pipeline_options=options)
```

**Confirmed results (from test run):**

| Document | Flagged chunks (FAST) | Flagged chunks (ACCURATE) | What remained |
|---|---|---|---|
| `attention_is_all_you_need.pdf` | 2+ | 2 | Author name grid (page 1) + reference list fragment |
| `bert_paper.pdf` | 1+ | 1 | Author name grid (page 1) |

Content sections (introduction, methods, experiments, results) chunked correctly after applying `ACCURATE`. Residual anomalies are the author block on page 1 — a dense name/affiliation grid that the layout model still mis-reads. This is structural noise, not content, and does not affect retrieval quality for RAG.

### What does not work

**VLMs do not fix column mixing.** This is a layout analysis problem — text is already extracted by the text pipeline, just in the wrong reading order. VLMs only process image content and are never in the loop for this failure.

**`do_cell_matching=False` and `correct_overlapping_cells=True`** are for table cell bounding-box errors (Scenario C). They have no effect on column reading order.

**Post-processing to reorder mixed text** is not practical — by the time chunks are produced, source sentence boundaries are lost and column provenance cannot be reconstructed from text alone. To detect mixing programmatically, inspect `prov[].bbox` x-ranges in `doc_items`: left column has `l` ≈ 50–250, right column has `l` ≈ 300–500 (exact values depend on page width).

---

## References

[1] https://github.com/docling-project/docling/issues/2976  
[2] https://github.com/docling-project/docling/issues/2862  
[3] https://github.com/docling-project/docling/issues/2130  
[4] https://github.com/docling-project/docling/issues/2756  
[5] https://aihorizonforecast.substack.com/p/docling-extracting-structured-data  
[6] https://github.com/docling-project/docling/issues/2790  
[7] https://codecut.ai/docling-vs-marker-vs-llamaparse/  
[8] https://www.reddit.com/r/LangChain/comments/1qvio23/rag_with_docling_on_a_policy_document/  
[9] https://github.com/docling-project/docling-graph/blob/main/docs/fundamentals/extraction-process/chunking-strategies.md  
[10] https://www.reddit.com/r/LangChain/comments/1puphjs/free_pdftomarkdown_demo_that_finally_extracts/  
[11] https://docling-project.github.io/docling/concepts/chunking/  
[12] https://www.youtube.com/watch?v=tMwdl9hFPns  
[13] https://docling-project.github.io/docling/examples/hybrid_chunking/  
[14] https://github.com/docling-project/docling/issues/1927  
[15] https://github.com/docling-project/docling/issues/2330  
[16] https://github.com/docling-project/docling/issues/1922  
[17] https://github.com/docling-project/docling/issues/1611  
