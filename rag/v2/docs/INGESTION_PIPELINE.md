# Ingestion Pipeline

**Entry point:** `python -m rag.main --ingest --documents rag/documents`

**Key file:** `rag/ingestion/pipeline.py` → `DocumentIngestionPipeline`

---

## Table of Contents

- [Pipeline Steps](#pipeline-steps)
- [Planned Improvements](#planned-improvements)
- [Modes](#modes)
- [CUAD Legal Contract Ingestion](#cuad-legal-contract-ingestion)
- [Supported Formats](#supported-formats)
- [Configuration](#configuration)

---

## Pipeline Steps

For each document file:

```
1. _compute_file_hash()
      MD5 of file bytes

2. Incremental check (--no-clean mode only)
      Hash unchanged → skip
      Hash changed   → delete existing chunks + re-ingest

3. _read_document()
      .pdf → _get_pdf_converter() [cached]
            VLM_ENABLED=false (default)
              → DocumentConverter()  StandardPdfPipeline (layout + OCR)
            VLM_ENABLED=true
              → DocumentConverter(PdfPipelineOptions(do_picture_description=True)
                                  + PictureDescriptionApiOptions)
                  cropped figure/table images → POST Ollama /v1/chat/completions
                  model=qwen2.5vl:7b → figure captions inserted into DoclingDocument
                  DoclingDocument flows into HybridChunker unchanged
      .docx / .pptx / .xlsx / .html / .htm / .md / .markdown
            → _get_standard_converter() [cached]
              → DocumentConverter()  standard pipeline
              (text layer embedded — no VLM, no OCR ever run)
      .mp3 / .wav / .m4a / .flac
            → Docling ASR pipeline + Whisper (requires ffmpeg in PATH)
      .txt and others
            → direct file read

4. _extract_title()
      First "# " heading, falling back to filename stem

5. chunker.chunk_document()
      Docling HybridChunker — token-aware, preserves heading context
      Fallback: sliding-window _simple_fallback_chunk()

6. embedder.embed_chunks()
      Batched POST to OpenAI-compatible /v1/embeddings
      Async LRU cache on (text, model) — embeddings reused across runs

7. store.save_document()
      INSERT INTO documents (id, title, source, content, metadata)

8. store.add(chunks)
      executemany INSERT INTO chunks (single batch per document)
```

---

## Planned Improvements

| Item | Description |
|------|-------------|
| PDF preprocessing | Detect text layer presence before routing to VLM or OCR — skip OCR for digital PDFs, skip VLM for text-only PDFs |
| DOCX/PPTX image extraction | Run VLM on embedded images in Word/PowerPoint files |

---

## Modes

| Flag | Behaviour |
|------|-----------|
| `--ingest` | `DELETE FROM` both tables, ingest everything from scratch |
| `--ingest --no-clean` | Hash-based skip / update / delete — only changed files re-ingested |

---

## CUAD Legal Contract Ingestion

> **Moved:** The CUAD ingestion script and associated data were moved to `misc/kg_legal_cuad/` and are no longer importable as `rag.ingestion.*` modules. See `misc/kg_legal_cuad/cuad_ingestion.py` for the archived implementation.

Download the dataset:
```python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="theatticusproject/cuad",
    filename="CUAD_v1/CUAD_v1.json",
    repo_type="dataset",
    local_dir="C:/hf/cuad",
)
```

---

## Supported Formats

| Category | Extensions |
|----------|-----------|
| Text | `.md`, `.markdown`, `.txt` |
| Documents | `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.html` |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac` (requires `ffmpeg` + `openai-whisper`) |

---

## Configuration

| Setting | Default | Effect |
|---------|---------|--------|
| `EMBEDDING_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `EMBEDDING_DIMENSION` | `768` | Must match model output |
| `EMBEDDING_BASE_URL` | `http://localhost:11434/v1` | Ollama or OpenAI endpoint |
| `VLM_ENABLED` | `false` | Enable figure-description VLM for PDF ingestion (`PictureDescriptionApiOptions`) |
| `VLM_MODEL` | `qwen2.5vl:7b` | Ollama model tag for VLM (must be pulled first) |
| `VLM_BASE_URL` | `http://localhost:11434/v1/chat/completions` | VLM API endpoint |
| `VLM_TIMEOUT` | `120.0` | Per-page timeout in seconds |
| `VLM_CONCURRENCY` | `1` | Concurrent page requests to VLM |
