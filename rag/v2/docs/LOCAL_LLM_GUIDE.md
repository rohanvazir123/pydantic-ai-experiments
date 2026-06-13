# RAG System — Local LLM Guide

> **Hardware baseline**
> - Dev: 8 GB VRAM (single consumer GPU — RTX 3070 / RX 6800 XT class)
> - Production: RunPod cloud GPU (RTX 4090 24 GB · A40 48 GB · A100 80 GB)
> - Hard constraint: **all inference runs locally via Ollama — no cloud LLM API calls**

---

## Table of Contents

1. [Where LLMs Are Called in the RAG Pipeline](#1-where-llms-are-called-in-the-rag-pipeline)
2. [Embedding Model](#2-embedding-model)
3. [Generation / Chat Model](#3-generation--chat-model)
4. [Vision Language Models (VLMs)](#4-vision-language-models-vlms)
5. [Token Limits Reference](#5-token-limits-reference)
6. [VRAM Requirements](#6-vram-requirements)
7. [Ollama Configuration](#7-ollama-configuration)
8. [RunPod GPU Recommendations](#8-runpod-gpu-recommendations)
9. [What Breaks on 8 GB VRAM](#9-what-breaks-on-8-gb-vram)
10. [Quantisation Tiers](#10-quantisation-tiers)
11. [Model Recommendation Matrix](#11-model-recommendation-matrix)

---

## 1. Where LLMs Are Called in the RAG Pipeline

| Step | File | Model type | Call frequency |
|------|------|-----------|----------------|
| Chunk embedding (ingestion) | `rag/ingestion/embedder.py` | Embedding | Once per chunk (~100–500 tokens each) |
| Query embedding (retrieval) | `rag/retrieval/retriever.py` | Embedding | Every query |
| Answer synthesis | `rag/agent/rag_agent.py` | Chat / instruction | Every query |
| Mem0 memory extraction | `rag/memory/mem0_store.py` | Chat / instruction | Every interaction (if memory enabled) |
| HyDE hypothetical doc (disabled) | `rag/retrieval/retriever.py` | Chat / instruction | — (currently off) |
| Reranker scoring (disabled) | `rag/retrieval/retriever.py` | Chat / instruction | — (currently off) |

The **critical path per query** is: embed query → hybrid search → synthesise answer.  
Two model calls back-to-back through Ollama on the same GPU.

---

## 2. Embedding Model

### Current: `nomic-embed-text:latest` (768-dim)

| Property | Value |
|----------|-------|
| Dimensions | 768 |
| Context window | 8 192 tokens |
| VRAM (FP16) | ~0.3 GB |
| Inference speed | ~5 ms / chunk on RTX 4090 |

### Limitations

- Trained on general web text — not legal or financial domain
- Chunks > 8 192 tokens are **silently truncated** to 8 192 before embedding
- After any model swap all existing vectors are invalid and the vector store must be re-ingested

### Recommended alternatives by domain

| Domain | Model | Dims | Notes |
|--------|-------|------|-------|
| General (default) | `nomic-embed-text:latest` | 768 | Good baseline, fast |
| Code | `nomic-embed-text:latest` | 768 | Acceptable |
| Legal / financial | `mxbai-embed-large:latest` | 1024 | Better legal recall |
| Multilingual | `multilingual-e5-large` | 1024 | Non-English docs |
| High-accuracy (prod) | `bge-m3` | 1024 | Stronger retrieval |

**Config change required** when swapping: update `EMBEDDING_MODEL` and `EMBEDDING_DIMENSION` in `.env`, then re-ingest all documents.

---

## 3. Generation / Chat Model

### What the agent needs from the LLM

1. Follow the RAG system prompt precisely (answer from context only)
2. Acknowledge "I don't know" when context is insufficient — no confabulation
3. Handle multi-turn conversation history in the prompt
4. Respect context length: retrieved chunks + history + question must fit in one call
5. Produce structured JSON if `search_knowledge_graph` tool result formatting is required

### Minimum viable model

A **7B instruction-tuned model at Q4_K_M** can answer factual questions from provided context reasonably well. The RAG pattern is forgiving — the LLM only needs to summarise retrieved text, not generate knowledge from scratch.

### Recommended models — Ollama tags

| Model | Params | Quant | VRAM | Quality | Use case |
|-------|--------|-------|------|---------|----------|
| `llama3.1:8b-instruct-q4_K_M` | 8B | Q4_K_M | ~5 GB | Good | Dev / 8 GB GPU |
| `llama3.1:8b-instruct-q8_0` | 8B | Q8_0 | ~9 GB | Better | RunPod RTX 4090 |
| `qwen2.5:14b-instruct-q4_K_M` | 14B | Q4_K_M | ~9 GB | Better | RunPod RTX 4090 |
| `qwen2.5:14b-instruct-q8_0` | 14B | Q8_0 | ~16 GB | Best 14B | RunPod A40 |
| `llama3.3:70b-instruct-q4_K_M` | 70B | Q4_K_M | ~40 GB | Excellent | RunPod A100 |
| `mistral-nemo:12b-instruct-q4_K_M` | 12B | Q4_K_M | ~8 GB | Good | 8 GB GPU (tight) |

### Why Qwen 2.5 over Llama for RAG

- Stronger instruction following at 14B than Llama 3.1 8B
- Better at citing only what is in the provided context
- 128K context window (Llama 3.1 8B: 128K but effective ~16K on local hardware)

---

## 4. Vision Language Models (VLMs)

VLMs are needed in the RAG pipeline wherever **documents contain images, tables as images, or scanned PDFs**.

### Where VLMs are used

| Component | State | Notes |
|-----------|-------|-------|
| Docling PDF pipeline | **Opt-in** — `VLM_ENABLED=false` by default | Cropped figure/table images → Qwen2.5-VL via Ollama → captions inserted into DoclingDocument |
| Docling figure extraction | **Opt-in** via `do_picture_description=True` | Figures described in-line; uses `PictureDescriptionApiOptions` |
| Scanned PDF OCR | Not handled by VLM path | Standard Docling OCR; VLM only handles extracted figures |
| Audio via Whisper | No VLM | Whisper is a standalone ASR model |

### Recommended VLMs

| Use case | Model | Params | VRAM | Ollama tag |
|----------|-------|--------|------|------------|
| **Default (implemented)** | `qwen2.5vl:7b` | 7B | ~5 GB | `qwen2.5vl:7b` |
| Higher accuracy | `qwen2.5vl:72b` | 72B | ~44 GB | `qwen2.5vl:72b` |

The VLM runs via Ollama — no VRAM is consumed by the Python process directly.

### How VLM is wired into Docling (implemented)

`_get_converter()` in `rag/ingestion/pipeline.py` builds the converter based on `VLM_ENABLED`:

```python
# VLM_ENABLED=false (default) — standard layout + OCR pipeline
converter = DocumentConverter()

# VLM_ENABLED=true — cropped figure images sent to Qwen2.5-VL via Ollama
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.pipeline_options import PictureDescriptionApiOptions

vlm_options = PictureDescriptionApiOptions(
    url="http://localhost:11434/v1/chat/completions",
    params={"model": "qwen2.5vl:7b", "max_tokens": 1024},
    timeout=120.0,
    prompt=(
        "Describe this figure or table in detail. "
        "Focus on what data, relationships, or visual content it conveys."
    ),
)
pipeline_options = PdfPipelineOptions(
    do_picture_description=True,
    picture_description_options=vlm_options,
)
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)
```

The resulting `DoclingDocument` flows unchanged into `HybridChunker` — figure descriptions land in the chunks and are embedded and indexed alongside text.

Enable via `.env`:

```bash
VLM_ENABLED=true
VLM_MODEL=qwen2.5vl:7b        # must be pulled in Ollama first
VLM_BASE_URL=http://localhost:11434/v1/chat/completions
VLM_TIMEOUT=120.0              # seconds per page
VLM_CONCURRENCY=1              # parallel page requests
```

On 8 GB VRAM, **do not run VLM + chat LLM simultaneously** — pull model before serving and set `OLLAMA_NUM_PARALLEL=1`.

---

## 5. Token Limits Reference

| Model / component | Context window | Effective limit | Notes |
|-------------------|---------------|-----------------|-------|
| `nomic-embed-text` | 8 192 tokens | 8 192 tokens | Truncates silently beyond limit |
| `mxbai-embed-large` | 512 tokens | 512 tokens | Short embeddings — chunk must fit |
| `bge-m3` | 8 192 tokens | 8 192 tokens | Pooling handles length well |
| Llama 3.1 8B | 128K tokens | ~8K–16K effective | Local hardware limits KV cache |
| Qwen 2.5 14B | 128K tokens | ~16K–32K effective | Better long-context on A40/A100 |
| Qwen 2.5 72B | 128K tokens | ~32K–64K effective | Full context on A100 80GB |
| Mistral Nemo 12B | 128K tokens | ~16K effective | |
| Whisper (ASR) | ~30 s audio chunks | N/A (not tokens) | Chunked by librosa |

### Budget breakdown per RAG query

```
System prompt          ~300 tokens
Retrieved chunks       ~1 500–4 000 tokens   (5–10 chunks × 300–500 tokens each)
Conversation history   ~500–2 000 tokens      (last N turns)
User question          ~20–100 tokens
LLM response           ~200–800 tokens
─────────────────────────────────────────────
Total per query        ~2 500–7 000 tokens
```

A 7B model at 8K effective context fits comfortably. At 10+ chunks or long history, context overflow is possible — guard with a `max_tokens` count before calling Ollama.

---

## 6. VRAM Requirements

### 8 GB VRAM (dev) — what fits

| Scenario | VRAM used | Fits? |
|----------|-----------|-------|
| `nomic-embed-text` alone | ~0.3 GB | Yes |
| Llama 3.1 8B Q4_K_M alone | ~5.0 GB | Yes |
| Embed + generate simultaneously | ~5.3 GB | Yes (tight) |
| Qwen 2.5 14B Q4_K_M alone | ~9.0 GB | **No** — OOM |
| Mistral Nemo 12B Q4_K_M alone | ~7.8 GB | Borderline |
| Any 13B+ model at Q8_0 | 10 GB+ | **No** |
| VLM (llava:13b) + embed | 10 GB+ | **No** |

### RunPod — recommended GPU by workload

| GPU | VRAM | Recommended for |
|-----|------|----------------|
| RTX 4090 | 24 GB | Qwen 2.5 14B Q8_0 + embed simultaneously; dev-scale production |
| A40 | 48 GB | Qwen 2.5 14B Q8_0 + VLM for ingestion; mid-scale production |
| A100 80GB | 80 GB | Llama 3.3 70B Q4_K_M + embed; high-throughput production |
| 2× A100 80GB | 160 GB | Llama 3.3 70B Q8_0 or full precision; enterprise production |

---

## 7. Ollama Configuration

### Key environment variables

```bash
# Keep model loaded between requests (prevents cold-start latency)
OLLAMA_KEEP_ALIVE=30m           # 0 = unload immediately; -1 = never unload

# Number of GPU layers to offload (higher = faster; must fit in VRAM)
OLLAMA_NUM_GPU=99               # 99 = offload all layers

# CPU threads for layers that don't fit in VRAM
OLLAMA_NUM_THREAD=8             # match physical CPU cores

# Max parallel requests (set to 1 on 8GB GPU to avoid OOM)
OLLAMA_NUM_PARALLEL=1           # increase to 2–4 on A40/A100

# Flash attention (reduces VRAM by ~20% with no quality loss)
OLLAMA_FLASH_ATTENTION=1
```

### Modelfile overrides (per model)

```modelfile
FROM llama3.1:8b-instruct-q4_K_M

PARAMETER num_ctx 8192          # effective context window
PARAMETER num_gpu 99            # GPU layers
PARAMETER num_thread 8          # CPU threads
PARAMETER temperature 0.1       # low temp for factual RAG answers
PARAMETER repeat_penalty 1.1    # reduce repetition
```

Create with: `ollama create rag-llm -f Modelfile`

### Temperature guidance

| Task | Recommended temperature |
|------|------------------------|
| Factual RAG answer synthesis | 0.0–0.1 |
| Query understanding / chat | 0.1–0.3 |
| Creative summarisation | 0.3–0.7 |

Always use **low temperature (≤ 0.1)** for RAG — the LLM should quote/summarise context, not generate novel text.

---

## 8. RunPod GPU Recommendations

### Per-component GPU sizing

| Component | Min GPU | Recommended GPU | Notes |
|-----------|---------|----------------|-------|
| Embedding only | RTX 3080 10GB | RTX 4090 24GB | Fast throughput needed for bulk ingestion |
| RAG answer synthesis | RTX 4090 24GB | A40 48GB | Run 14B model for better answer quality |
| RAG + KG combined | A40 48GB | A100 80GB | Multiple model calls per query |
| Docling + VLM ingestion | RTX 4090 24GB | A40 48GB | VLM + embed can share 24GB |
| Full stack (RAG + KG + NL2SQL) | A100 80GB | 2× A100 | Avoid Ollama model eviction under load |

### RunPod pod configuration

```yaml
# Recommended RunPod template for production RAG
GPU: NVIDIA A40 (48GB)
vCPU: 8
RAM: 32GB
Storage: 50GB (for models + vector store)
Image: runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Pre-pull models on pod start
startup: |
  ollama pull nomic-embed-text:latest
  ollama pull qwen2.5:14b-instruct-q4_K_M
```

---

## 9. What Breaks on 8 GB VRAM

| Scenario | What happens | Workaround |
|----------|-------------|------------|
| Running 14B model | OOM — Ollama kills process | Use 8B Q4_K_M or offload CPU layers |
| Running embed + generate simultaneously | ~5.3 GB — works at 8B, fails at 14B | Serialise calls; `OLLAMA_NUM_PARALLEL=1` |
| Running VLM + chat model | ~10 GB+ | Run VLM at ingestion only; unload before serving |
| Large batch embedding (1000s of chunks) | VRAM spikes, may OOM | Batch size ≤ 32 chunks; `OLLAMA_KEEP_ALIVE=0` between batches |
| 70B model any quant | 40 GB min | RunPod A100 only |
| Full context window (128K) | KV cache exhausts VRAM | Cap `num_ctx` at 8192 for 8B on 8GB |

### CPU offloading (last resort)

```bash
# Offload some layers to CPU when VRAM is full
OLLAMA_NUM_GPU=20    # keep only 20 layers on GPU; rest on CPU
```

This works but increases latency by 3–10× per token. Acceptable for batch ingestion, not for interactive queries.

---

## 10. Quantisation Tiers

| Tier | Size reduction | Quality loss | Recommended when |
|------|---------------|-------------|-----------------|
| Q8_0 | ~50% vs FP16 | Minimal (<0.5%) | Preferred for production on A40/A100 |
| Q6_K | ~62% vs FP16 | Very low (~1%) | Good quality/size trade-off |
| Q5_K_M | ~69% vs FP16 | Low (~2%) | A good middle ground |
| Q4_K_M | ~75% vs FP16 | Moderate (~3–5%) | **Recommended for 8GB dev** |
| Q4_0 | ~75% vs FP16 | Higher (~5–7%) | Avoid — K-quants are always better |
| Q3_K_M | ~81% vs FP16 | High (~8–12%) | Last resort for memory-constrained |
| Q2_K | ~87% vs FP16 | Very high (>15%) | Not recommended for production |

**Rule of thumb**: Use Q4_K_M on 8 GB VRAM dev hardware; use Q8_0 on RunPod A40/A100 for best quality.

---

## 11. Model Recommendation Matrix

### Complete recommendation by hardware tier

| Hardware | Embedding | Chat LLM | VLM (if needed) |
|----------|-----------|----------|-----------------|
| 8 GB VRAM (dev) | `nomic-embed-text` | `llama3.1:8b-instruct-q4_K_M` | Not feasible — skip VLM |
| RTX 4090 24 GB | `nomic-embed-text` or `mxbai-embed-large` | `qwen2.5:14b-instruct-q4_K_M` | `qwen2.5vl:7b-instruct-q4_K_M` |
| A40 48 GB | `bge-m3` | `qwen2.5:14b-instruct-q8_0` | `qwen2.5vl:7b-instruct-q8_0` |
| A100 80 GB | `bge-m3` | `llama3.3:70b-instruct-q4_K_M` | `qwen2.5vl:72b-instruct-q4_K_M` |

### Priority upgrade path

1. **First**: Upgrade chat model from 8B → 14B (biggest quality improvement for RAG answer synthesis)
2. **Second**: Upgrade embedding model from `nomic-embed-text` → `bge-m3` (better recall especially for legal domain)
3. **Third**: Add VLM for document ingestion (only if source docs contain image-heavy tables/figures)
4. **Fourth**: Upgrade to 70B model (diminishing returns for RAG specifically — 14B is usually enough)
