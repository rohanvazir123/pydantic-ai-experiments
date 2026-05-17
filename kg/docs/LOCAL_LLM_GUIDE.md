# Knowledge Graph — Local LLM Guide

> **Hardware baseline**
> - Dev: 8 GB VRAM (single consumer GPU)
> - Production: RunPod cloud GPU (RTX 4090 24 GB · A40 48 GB · A100 80 GB)
> - Hard constraint: **all inference runs locally via Ollama — no cloud LLM API calls**

---

## Table of Contents

1. [Where LLMs Are Called in the KG Pipeline](#1-where-llms-are-called-in-the-kg-pipeline)
2. [Extraction LLM — Bronze/Silver/Gold Pipeline](#2-extraction-llm--bronzesilverGold-pipeline)
3. [Embedding Model — EntityIndex](#3-embedding-model--entityindex)
4. [Vision Language Models (VLMs)](#4-vision-language-models-vlms)
5. [NL→Cypher — No LLM Required (Rule-Based)](#5-nlcypher--no-llm-required-rule-based)
6. [Token Limits Reference](#6-token-limits-reference)
7. [VRAM Requirements](#7-vram-requirements)
8. [Ollama Configuration](#8-ollama-configuration)
9. [RunPod GPU Recommendations](#9-runpod-gpu-recommendations)
10. [What Breaks on 8 GB VRAM](#10-what-breaks-on-8-gb-vram)
11. [Quantisation Tiers](#11-quantisation-tiers)
12. [Model Recommendation Matrix](#12-model-recommendation-matrix)

---

## 1. Where LLMs Are Called in the KG Pipeline

| Step | File | Model type | Call frequency |
|------|------|-----------|----------------|
| Entity + relation extraction (Bronze) | `kg/legal/ingestion/extraction_pipeline.py` | Chat / instruction | Once per contract page (~10–50 pages/contract) |
| Entity canonicalisation (Silver) | `kg/legal/ingestion/extraction_pipeline.py` | Chat / instruction | Once per entity cluster |
| NL→Cypher conversion | `kg/legal/retrieval/nl2cypher.py` | **None** — rule-based | N/A |
| EntityIndex embedding (ingestion) | `kg/entity_index.py` | Embedding | Once per entity |
| EntityIndex query embedding | `kg/legal/retrieval/` | Embedding | Every query |

The KG extraction pipeline is **LLM-heavy**: every page of every contract goes through the chat model for entity/relation extraction. This is the most expensive operation in the entire repo, running on local hardware.

---

## 2. Extraction LLM — Bronze/Silver/Gold Pipeline

### What the extraction LLM must do

The model receives a structured prompt containing:
- A page of contract text (1–3 KB)
- A list of valid entity labels (`VALID_LABELS` from `kg/legal/common/cuad_ontology.py`)
- A list of valid relation types (`VALID_REL_TYPES`)
- Instructions to output JSON with `entities` and `relationships` arrays

It must:
1. **Follow the output schema exactly** — no extra keys, no markdown fences around JSON
2. **Use only valid label/relation types** — hallucinated labels break the Silver canonicalisation step
3. **Extract relationships as triples** — `(subject_label, subject_name, relation, object_label, object_name)`
4. **Handle long dense text** — legal contracts are verbose and entity-rich
5. **Not confabulate entities** — only extract what is explicitly stated in the page

This is a **demanding structured extraction task**. Small models (7B) frequently output malformed JSON, hallucinate entity types, or omit relationships.

### Recommended models for extraction

| Model | Params | Quant | VRAM | JSON accuracy | Notes |
|-------|--------|-------|------|--------------|-------|
| `qwen2.5:14b-instruct-q4_K_M` | 14B | Q4_K_M | ~9 GB | High | **Best choice for RunPod RTX 4090** |
| `qwen2.5:14b-instruct-q8_0` | 14B | Q8_0 | ~16 GB | Very high | Preferred for A40/A100 |
| `qwen2.5:7b-instruct-q8_0` | 7B | Q8_0 | ~7 GB | Moderate | Fallback for 8GB dev (expect ~20% JSON failures) |
| `llama3.1:8b-instruct-q4_K_M` | 8B | Q4_K_M | ~5 GB | Low | Not recommended — frequent JSON schema violations |
| `mistral-nemo:12b-instruct-q4_K_M` | 12B | Q4_K_M | ~8 GB | Moderate-high | Acceptable on 8GB (borderline) |
| `llama3.3:70b-instruct-q4_K_M` | 70B | Q4_K_M | ~40 GB | Excellent | RunPod A100 only |

### Why Qwen 2.5 outperforms Llama for extraction

- Qwen 2.5 was trained with **stronger structured output / JSON-mode instruction following**
- Legal entity extraction requires precise adherence to a closed vocabulary — Qwen 2.5 14B respects this significantly better than Llama 3.1 8B
- Qwen 2.5 7B at Q8_0 outperforms Llama 3.1 8B at Q4_K_M for this specific task

### Prompt hardening for local models

Even Qwen 2.5 14B will occasionally produce malformed JSON. Apply these mitigations:

```python
# 1. Use Ollama's structured output / format=json
response = ollama.chat(
    model="qwen2.5:14b-instruct-q4_K_M",
    messages=[...],
    format="json",          # forces JSON-only output
    options={"temperature": 0.0},   # deterministic extraction
)

# 2. Retry up to 3 times on JSON parse failure
# 3. Validate against Pydantic schema before accepting
# 4. Log rejection reasons to Bronze audit file
```

### Expected failure rates by model

| Model | JSON parse failure | Wrong label used | Missing relation |
|-------|-------------------|-----------------|-----------------|
| Qwen 2.5 14B Q8_0 | ~2% | ~5% | ~15% |
| Qwen 2.5 14B Q4_K_M | ~3% | ~8% | ~18% |
| Qwen 2.5 7B Q8_0 | ~8% | ~15% | ~25% |
| Llama 3.1 8B Q4_K_M | ~20% | ~30% | ~40% |
| Llama 3.3 70B Q4_K_M | ~0.5% | ~2% | ~8% |

---

## 3. Embedding Model — EntityIndex

The `EntityIndex` in `kg/entity_index.py` embeds entity names + descriptions for semantic entity search. This uses the **same embedding model** as the RAG system.

| Property | Current value |
|----------|--------------|
| Model | `nomic-embed-text:latest` |
| Dimensions | 768 |
| Context window | 8 192 tokens |
| Entity string format | `"{label}: {name} — {description}"` |
| Typical entity string length | 20–80 tokens |

### Recommended embedding models for entity index

| Model | Dims | Entity matching quality | Notes |
|-------|------|------------------------|-------|
| `nomic-embed-text:latest` | 768 | Baseline | Fast, adequate for general entities |
| `mxbai-embed-large:latest` | 1024 | Better | Better synonym matching for legal terms |
| `bge-m3` | 1024 | Best | Strongest semantic similarity for entity names |

**Important**: if you change the embedding model, re-index all entities in `EntityIndex`. Mixing embedding spaces silently corrupts search results.

---

## 4. Vision Language Models (VLMs)

### Where VLMs apply in KG

| Step | VLM needed | Use case |
|------|-----------|---------|
| Contract PDF ingestion via Docling | Optional | Improve table/signature block extraction |
| Org chart / relationship diagram parsing | Yes | Extract org relationships from figures |
| Scanned contract page OCR | Yes | Dense scans degrade Tesseract; VLM improves |
| Entity extraction from text | No | Text-only — chat LLM handles this |

### Legal document VLM considerations

Legal contracts are mostly dense text — VLMs add significant value primarily for:
- **Signature blocks** (scanned PDFs): Tesseract often fails on signatures and stamps
- **Exhibit tables**: complex multi-column tables are mis-parsed by heuristic layout detectors
- **Redacted documents**: VLMs cannot un-redact but can identify redaction patterns

### Recommended VLMs for legal document ingestion

| Model | Params | VRAM | Notes |
|-------|--------|------|-------|
| `qwen2.5vl:7b-instruct-q4_K_M` | 7B | ~5 GB | Good general document understanding |
| `qwen2.5vl:7b-instruct-q8_0` | 7B | ~8 GB | Better accuracy; fits 8GB (tight) |
| `qwen2.5vl:72b-instruct-q4_K_M` | 72B | ~44 GB | Excellent; RunPod A100 only |
| `minicpm-v:8b` | 8B | ~6 GB | Strong OCR specifically |

**Strategy**: Run VLM **only during ingestion**, not during query time. Unload VLM before loading the extraction/chat model to avoid OOM on 8 GB dev hardware.

---

## 5. NL→Cypher — No LLM Required (Rule-Based)

The current `kg/legal/retrieval/nl2cypher.py` uses a **rule-based pipeline** — no LLM call is made:

```
User question → IntentParser (regex) → QUERY_CAPABILITIES[intent] → builder function → Cypher string
```

This means:
- **Zero VRAM cost** for query-time NL→Cypher
- **Zero latency** from LLM inference at query time
- **Deterministic** — same question always produces the same Cypher
- **Limited**: can only handle 24 defined intents; `list_contracts` catch-all fires for everything else

### If adding LLM-based NL→Cypher as fallback

An LLM fallback for queries that don't match any of the 24 intents would require:

| Property | Requirement |
|----------|------------|
| Model capability | Must know Apache AGE Cypher syntax specifically |
| Training data | AGE Cypher is rare in training corpora — expect high failure rate |
| Minimum model size | 14B+ for acceptable Cypher quality |
| Recommended model | `qwen2.5:14b-instruct-q4_K_M` with AGE Cypher examples in the system prompt |
| Validation required | Every LLM-generated Cypher must be validated before execution |
| Expected failure rate | ~40–60% malformed or semantically wrong Cypher on 14B local models |

See `kg/docs/KG_RETRIEVAL_PIPELINE.md §12` for the full rule-based pipeline internals.  
See `PRODUCTION_RISKS.md §7` for LLM Cypher failure rate details.

---

## 6. Token Limits Reference

| Model / component | Context window | Effective limit on local HW | Notes |
|-------------------|---------------|----------------------------|-------|
| `nomic-embed-text` | 8 192 tokens | 8 192 tokens | Entity strings well within limit |
| `bge-m3` | 8 192 tokens | 8 192 tokens | Better pooling at long lengths |
| Qwen 2.5 14B | 128K tokens | ~16K–32K effective | Use `num_ctx=8192` on RTX 4090 to save VRAM |
| Qwen 2.5 7B | 128K tokens | ~8K effective | |
| Llama 3.1 8B | 128K tokens | ~8K effective | |
| Llama 3.3 70B | 128K tokens | ~32K–64K on A100 | |

### Token budget for extraction per contract page

```
System prompt (labels + instructions)   ~800 tokens
Contract page text                      ~600–1 200 tokens
JSON response (entities + relations)    ~400–800 tokens
─────────────────────────────────────────────────────
Total per page                          ~1 800–2 800 tokens
```

A 50-page contract → 50 LLM calls × ~2 500 tokens each = ~125 000 tokens per contract.  
At 30 tokens/sec on RTX 4090 with Qwen 2.5 14B Q4_K_M → ~70 minutes per contract.  
At 60 tokens/sec on A100 with 70B Q4_K_M → ~35 minutes per contract.

**Implication**: batch ingestion of 500+ contracts is a multi-day job on local hardware. Plan accordingly.

### Token budget for NL→Cypher (rule-based — no LLM)

Zero tokens consumed. The rule-based pipeline does not call the LLM at query time.

---

## 7. VRAM Requirements

### KG extraction (ingestion)

| Scenario | VRAM used | Fits 8 GB? |
|----------|-----------|-----------|
| `nomic-embed-text` alone | ~0.3 GB | Yes |
| Qwen 2.5 7B Q8_0 alone | ~7.5 GB | Yes (tight) |
| Qwen 2.5 14B Q4_K_M alone | ~9.0 GB | **No** |
| Qwen 2.5 14B Q8_0 alone | ~16.0 GB | **No** |
| Llama 3.3 70B Q4_K_M alone | ~40.0 GB | **No** |
| Embed + Qwen 2.5 7B Q8_0 | ~7.8 GB | **No** (OOM) |

**On 8 GB dev**: use Qwen 2.5 7B Q4_K_M (~5 GB). Expect ~20% JSON failures. Fix with retry logic.

### KG query (retrieval — rule-based pipeline)

| Scenario | VRAM used | Notes |
|----------|-----------|-------|
| EntityIndex embed query | ~0.3 GB | Embed only — no chat model |
| Rule-based NL→Cypher | 0 GB | No LLM call |
| Rule-based + entity embed | ~0.3 GB | Very cheap |

The rule-based pipeline is extremely VRAM-efficient at query time.

---

## 8. Ollama Configuration

### For extraction (ingestion) — throughput priority

```bash
OLLAMA_KEEP_ALIVE=60m        # keep model loaded across contract pages
OLLAMA_NUM_GPU=99            # all layers on GPU
OLLAMA_NUM_PARALLEL=1        # single worker; extraction is sequential
OLLAMA_FLASH_ATTENTION=1     # saves VRAM
```

### Modelfile for extraction

```modelfile
FROM qwen2.5:14b-instruct-q4_K_M

PARAMETER num_ctx 4096       # extraction prompt fits in 4K; saves VRAM vs 8K
PARAMETER temperature 0.0    # deterministic JSON extraction
PARAMETER num_gpu 99
PARAMETER num_thread 8
```

Temperature **must be 0.0** for extraction — any randomness causes inconsistent entity/relation output across retries.

---

## 9. RunPod GPU Recommendations

| GPU | VRAM | Recommended for |
|-----|------|----------------|
| RTX 4090 | 24 GB | Qwen 2.5 14B Q4_K_M extraction; dev-scale ingestion |
| A40 | 48 GB | Qwen 2.5 14B Q8_0; batch ingestion of 100–500 contracts |
| A100 80GB | 80 GB | Llama 3.3 70B Q4_K_M; best extraction quality; 500+ contracts |

### Cost estimate for full CUAD ingestion (510 contracts)

| GPU | Model | Throughput | Est. time | RunPod $/hr | Est. cost |
|-----|-------|-----------|-----------|------------|----------|
| RTX 4090 | Qwen 2.5 14B Q4_K_M | ~30 tok/s | ~60 hrs | ~$0.74 | ~$44 |
| A40 | Qwen 2.5 14B Q8_0 | ~45 tok/s | ~40 hrs | ~$1.20 | ~$48 |
| A100 80GB | Llama 3.3 70B Q4_K_M | ~25 tok/s | ~72 hrs | ~$2.50 | ~$180 |

Qwen 2.5 14B Q4_K_M on RTX 4090 is the best cost-quality trade-off for CUAD-scale ingestion.

---

## 10. What Breaks on 8 GB VRAM

| Scenario | What happens | Workaround |
|----------|-------------|------------|
| Running Qwen 2.5 14B any quant | OOM | Use Qwen 2.5 7B Q4_K_M (accept higher error rate) |
| Running embed + 7B extraction simultaneously | ~7.8 GB → OOM | Serialise: embed entities after extraction run |
| Batch ingestion of 500 contracts | 12+ hours continuous — thermal throttling | Use RunPod; don't run on dev 8GB GPU |
| Loading VLM + extraction model | ~12+ GB | Use VLM ingestion pipeline separately; swap models manually |
| 70B extraction model | 40+ GB | RunPod A100 only |

---

## 11. Quantisation Tiers

| Tier | Size vs FP16 | Quality loss | Use when |
|------|-------------|-------------|---------|
| Q8_0 | ~50% | Minimal | A40/A100 — use for extraction if VRAM allows |
| Q5_K_M | ~69% | Low | Good middle ground |
| Q4_K_M | ~75% | Moderate (~3–5%) | **Default for dev 8 GB** |
| Q3_K_M | ~81% | High | Avoid for extraction — JSON failures spike |
| Q2_K | ~87% | Very high | Not suitable for structured extraction |

For **entity extraction specifically**, never go below Q4_K_M. JSON schema violation rates increase non-linearly below Q4.

---

## 12. Model Recommendation Matrix

| Hardware | Extraction LLM | Embedding | VLM (optional) |
|----------|---------------|-----------|----------------|
| 8 GB VRAM (dev) | `qwen2.5:7b-instruct-q4_K_M` | `nomic-embed-text` | Skip — run separately |
| RTX 4090 24 GB | `qwen2.5:14b-instruct-q4_K_M` | `nomic-embed-text` | `qwen2.5vl:7b-instruct-q4_K_M` |
| A40 48 GB | `qwen2.5:14b-instruct-q8_0` | `bge-m3` | `qwen2.5vl:7b-instruct-q8_0` |
| A100 80 GB | `llama3.3:70b-instruct-q4_K_M` | `bge-m3` | `qwen2.5vl:72b-instruct-q4_K_M` |

### NL→Cypher (query time) — no LLM needed

The rule-based pipeline requires **zero GPU resources** at query time. If adding an LLM fallback for unsupported intents, add Qwen 2.5 14B to the query stack — same model already loaded for extraction, so no additional VRAM cost if `OLLAMA_KEEP_ALIVE` is set.
