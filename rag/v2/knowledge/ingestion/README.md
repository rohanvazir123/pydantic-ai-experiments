# knowledge/ingestion/

## Table of Contents

- [What This Is](#what-this-is)
- [Files](#files)
- [Pipeline Flow](#pipeline-flow)
- [Sync vs Async](#sync-vs-async)

---

## What This Is

The document ingestion pipeline. Converts documents (PDF, DOCX, audio, Markdown) into embedded chunks stored in PostgreSQL, and optionally extracts a knowledge graph via docling-graph and Apache AGE. Runs as a Redis Streams consumer worker process.

---

## Files

| File | Purpose |
|------|---------|
| `worker.py` | Entrypoint: connects stores, starts `consume_loop`; `python -m knowledge.ingestion.worker` |
| `pipeline.py` | Per-document orchestrator: Docling → `asyncio.gather(chunk_task, graph_task)` |
| `docling_processor.py` | Docling wrapper: two cached converters (PDF vs standard), format routing, audio ASR |
| `chunker.py` | `DoclingHybridChunker`: `HybridChunker` with `contextualize()`, fallback sliding window |
| `embedder.py` | `AsyncOpenAI`-compatible embedder; L1 `lru_cache`, timeout, exponential backoff |
| `graph_extractor.py` | docling-graph `run_pipeline()` in `asyncio.to_thread`; returns `PipelineContext | None` |
| `models.py` | `ChunkData`, `SearchResult`, `Citation`, `ConversionResult`, `IngestResult` |

---

## Pipeline Flow

```
IngestJob (from Redis stream)
  │
  ├─ DoclingProcessor.process(path)          # sync in to_thread; PDF or standard converter
  │     └─ returns (markdown, DoclingDocument)
  │
  ├─ asyncio.gather(
  │     chunker_task:
  │       DoclingHybridChunker.chunk_document()
  │       → embedder.embed_batch()
  │       → vector_store.upsert_chunks()
  │
  │     graph_task (if enable_graph_extraction):
  │       graph_extractor.extract_graph()    # run_pipeline() in to_thread
  │       → age_store.import_docling_graph() # iterates NetworkX DiGraph directly
  │       → entity_index.upsert_batch()
  │   )
  │
  └─ Publish IngestCompleteEvent to knowledge:events
```

---

## Sync vs Async

Docling is a synchronous library. Heavy CPU-bound operations are offloaded to the thread pool via `asyncio.to_thread()` so the event loop stays unblocked:

| Call | Pattern | Reason |
|------|---------|--------|
| `DocumentConverter.convert(path)` | `await asyncio.to_thread(converter.convert, path)` | CPU-bound; no async API |
| `HybridChunker.chunk()` | chained in same `to_thread` block after conversion | CPU-bound; shares thread |
| `AutoTokenizer.encode(text)` | inline sync | < 1ms; `to_thread` overhead exceeds benefit |
| `run_pipeline()` (docling-graph) | `await asyncio.to_thread(_run_sync)` | sync LLM I/O |
| `asyncpg`, `redis.asyncio`, `AsyncOpenAI` | native `await` | async I/O; never block |
