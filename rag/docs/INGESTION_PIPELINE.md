# Ingestion Pipeline

**Entry point:** `python -m rag.main --ingest --documents rag/documents`

**Key file:** `rag/ingestion/pipeline.py` → `DocumentIngestionPipeline`

---

## Table of Contents

- [Pipeline Steps](#pipeline-steps)
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
      PDF / DOCX / PPTX / XLSX / HTML
            → Docling DocumentConverter (ML model, cached per pipeline instance)
      Audio (.mp3 / .wav / .m4a / .flac)
            → Docling ASR pipeline + Whisper (requires ffmpeg in PATH)
      Markdown / TXT
            → direct read

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

## Modes

| Flag | Behaviour |
|------|-----------|
| `--ingest` | TRUNCATE both tables, ingest everything from scratch |
| `--ingest --no-clean` | Hash-based skip / update / delete — only changed files re-ingested |

---

## CUAD Legal Contract Ingestion

Standalone script for the [CUAD dataset](https://huggingface.co/datasets/theatticusproject/cuad) (510 commercial contracts, 41 annotated question types).

**Key file:** `rag/ingestion/cuad_ingestion.py`

- Converts contracts to Markdown → `rag/documents/legal/`
- Saves evaluation Q&A pairs → `rag/legal/cuad_eval.json`
- Feeds contracts through `DocumentIngestionPipeline`

```bash
# Dry run — extract files only, no DB write
python -m rag.ingestion.cuad_ingestion --dry-run

# Test — first 10 contracts
python -m rag.ingestion.cuad_ingestion --limit 10

# Full — all 510 contracts
python -m rag.ingestion.cuad_ingestion

# Incremental
python -m rag.ingestion.cuad_ingestion --no-clean
```

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
