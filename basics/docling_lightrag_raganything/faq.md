# FAQ — Docling + LightRAG / RAG-Anything Evaluation

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
