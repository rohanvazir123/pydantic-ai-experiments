# tests/ingestion/

## Table of Contents

- [What This Is](#what-this-is)
- [Requirements](#requirements)

---

## What This Is

Ingestion pipeline correctness tests. Covers format routing (PDF vs standard converter), incremental hash-based skip, chunk quality (token counts, contextualization), graph extraction, and corpus scoping.

---

## Requirements

- PostgreSQL + Redis running
- Ollama with `nomic-embed-text:latest` for embedding
- For VLM test: `VLM_ENABLED=true` and `qwen2.5vl:7b` pulled
- For audio test: FFmpeg in PATH + Whisper installed (test gracefully fails otherwise)
