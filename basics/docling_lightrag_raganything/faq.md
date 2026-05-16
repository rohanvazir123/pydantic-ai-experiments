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
