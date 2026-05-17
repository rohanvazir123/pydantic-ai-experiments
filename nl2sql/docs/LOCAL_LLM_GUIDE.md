# NL-to-SQL / NL-to-Cypher — Local LLM Guide

> **Hardware baseline**
> - Dev: 8 GB VRAM (single consumer GPU)
> - Production: RunPod cloud GPU (RTX 4090 24 GB · A40 48 GB · A100 80 GB)
> - Hard constraint: **all inference runs locally via Ollama — no cloud LLM API calls**

---

## Table of Contents

1. [Where LLMs Are Called in the NL-to-SQL Pipeline](#1-where-llms-are-called-in-the-nl-to-sql-pipeline)
2. [SQL Generation LLM](#2-sql-generation-llm)
3. [NL→Cypher Generation — No LLM (Rule-Based)](#3-nlcypher-generation--no-llm-rule-based)
4. [If Adding LLM-Based NL→Cypher Fallback](#4-if-adding-llm-based-nlcypher-fallback)
5. [Token Limits Reference](#5-token-limits-reference)
6. [Context Window Budget](#6-context-window-budget)
7. [VRAM Requirements](#7-vram-requirements)
8. [Ollama Configuration](#8-ollama-configuration)
9. [RunPod GPU Recommendations](#9-runpod-gpu-recommendations)
10. [What Breaks on 8 GB VRAM](#10-what-breaks-on-8-gb-vram)
11. [Quantisation Tiers](#11-quantisation-tiers)
12. [Model Recommendation Matrix](#12-model-recommendation-matrix)

---

## 1. Where LLMs Are Called in the NL-to-SQL Pipeline

| Step | File | Model | Call frequency |
|------|------|-------|----------------|
| SQL generation (v1) | `nl2sql/nlp_sql_postgres_v2.py` | Chat / code | Every query |
| Self-correction retry (v1) | `nl2sql/nlp_sql_postgres_v2.py` | Chat / code | On validation failure (up to 3×) |
| Table/column discovery (v2) | `nl2sql/sql_discovery.py` | Chat + tool calls | Every query (multi-turn) |
| NL→Cypher (current) | `kg/legal/retrieval/nl2cypher.py` | **None** — rule-based | N/A |

The v1 pipeline makes **1–4 LLM calls per query** (1 initial + up to 3 self-correction retries).  
The v2 tool-calling pipeline can make **3–8 LLM calls per query** (discovery turns + generation).

---

## 2. SQL Generation LLM

### What the LLM must do

The NL-to-SQL model receives:
- A system prompt describing DuckDB dialect and table naming conventions
- The full schema context (table names, column names, types, sample values)
- The natural language question
- (On retry) the failed SQL and the error message

It must produce:
1. **Plain SQL** — no markdown fences, no explanation text, no comments
2. **DuckDB-compatible syntax** — not standard PostgreSQL, not MySQL
3. **Correct table name prefix** — GCS tables bare, rag_db tables via `rag.main.<table>`, local_pg via `local_pg.main.<table>`
4. **SELECT only** — no INSERT, UPDATE, DELETE, DROP
5. **Schema-faithful** — only reference tables and columns that exist in the provided schema

This is a **harder task than general SQL generation** because:
- DuckDB SQL dialect differs from mainstream SQL in subtle ways
- The three-prefix table naming convention is non-standard and easy to get wrong
- Local 7B models frequently hallucinate column names even when the schema is in context

### What SQL tasks local models handle well

| Task | Local 7B-14B reliability |
|------|------------------------|
| Simple SELECT with WHERE | High (~85–95%) |
| JOINs between two tables | Moderate (~70–80%) |
| GROUP BY + aggregation | High (~80–90%) |
| Subqueries | Moderate (~65–75%) |
| CTEs (WITH clause) | Moderate (~60–70%) |
| Window functions | Low (~40–60%) |
| DuckDB-specific functions (e.g., `list_aggregate`, `struct_extract`) | Low (~30–50%) |
| Table prefix convention (GCS / rag.main / local_pg.main) | Low on 7B (~40%); improves to ~70% on 14B |
| Self-correction on error | High (~75–90%) — error message gives strong signal |

### Recommended models for SQL generation

| Model | Params | Quant | VRAM | SQL quality | Notes |
|-------|--------|-------|------|------------|-------|
| `qwen2.5-coder:7b-instruct-q4_K_M` | 7B | Q4_K_M | ~5 GB | Good | **Best 7B for SQL**; code-tuned |
| `qwen2.5-coder:7b-instruct-q8_0` | 7B | Q8_0 | ~7.5 GB | Better | Use on 8GB (tight) |
| `qwen2.5-coder:14b-instruct-q4_K_M` | 14B | Q4_K_M | ~9 GB | High | **Best choice for RunPod RTX 4090** |
| `qwen2.5-coder:14b-instruct-q8_0` | 14B | Q8_0 | ~16 GB | Very high | A40/A100 |
| `sqlcoder:15b-q4_K_M` | 15B | Q4_K_M | ~10 GB | High | Specialised for SQL; strong table fidelity |
| `codellama:13b-instruct-q4_K_M` | 13B | Q4_K_M | ~8.5 GB | Moderate | General code model |
| `llama3.1:8b-instruct-q4_K_M` | 8B | Q4_K_M | ~5 GB | Low-moderate | Not code-tuned; higher hallucination |
| `llama3.3:70b-instruct-q4_K_M` | 70B | Q4_K_M | ~40 GB | Excellent | RunPod A100; overkill for simple SQL |

### Why Qwen 2.5 Coder beats general instruction models for SQL

- Trained on large code corpus including SQL across multiple dialects
- Better schema adherence — correctly references provided column names
- Produces cleaner output (bare SQL without markdown fences) more consistently
- Self-correction loop convergence is faster — model understands error messages better

### Why NOT `llama3.1:8b` for SQL

- Not a code-specialised model — SQL generation is a secondary capability
- Frequently wraps output in markdown (```sql ... ```) which breaks direct execution
- High hallucination rate for column names even when schema is in the prompt
- DuckDB-specific syntax failures are common (~50–60% incorrect on non-trivial queries)

---

## 3. NL→Cypher Generation — No LLM (Rule-Based)

The current `kg/legal/retrieval/nl2cypher.py` uses a **rule-based pipeline**:

```
Natural language question
    → IntentParser (24 regex patterns)
    → QUERY_CAPABILITIES[intent](params)
    → Cypher string
```

**Zero LLM calls at query time.** No VRAM consumed. No latency from inference.  
The trade-off: only 24 intents are supported. Everything else hits the `list_contracts` catch-all.

For detailed internals see `kg/docs/KG_RETRIEVAL_PIPELINE.md §12`.

---

## 4. If Adding LLM-Based NL→Cypher Fallback

The rule-based pipeline cannot handle:
- Date range queries ("contracts expiring before 2026")
- Numeric threshold queries ("payment amounts over $1M")
- Multi-hop graph traversals ("parties connected to contracts that have an exclusivity clause")
- Free-form questions outside the 24 intents

Adding an LLM fallback requires a model that understands **Apache AGE Cypher specifically** — which almost no local model does well.

### Apache AGE Cypher vs standard Cypher

| Feature | AGE Cypher | Neo4j Cypher |
|---------|-----------|-------------|
| Training data availability | Very rare | Common |
| Execution syntax | `SELECT * FROM ag_catalog.cypher('graph', $$ ... $$) AS ...` | Direct Cypher shell |
| `agtype` column handling | Requires `::text` cast | Native types |
| RETURN format | Must match declared columns | Flexible |

Local 7B–14B models have essentially **no AGE-specific training data**. They will generate Neo4j Cypher, which requires post-processing to wrap in the AGE execution syntax.

### Expected failure rates for LLM-generated AGE Cypher

| Model | Cypher valid (Neo4j) | Wraps AGE correctly | Produces correct results |
|-------|---------------------|--------------------|-----------------------|
| Qwen 2.5 7B Q4_K_M | ~50% | ~15% | ~20% |
| Qwen 2.5 14B Q4_K_M | ~70% | ~30% | ~40% |
| Llama 3.3 70B Q4_K_M | ~85% | ~50% | ~60% |

**Recommendation**: if adding LLM fallback, provide the AGE execution wrapper in the system prompt and validate every generated query against the graph schema before execution. Never execute LLM-generated Cypher without validation.

### System prompt addition for AGE Cypher

```
You generate Apache AGE Cypher queries.
AGE is a PostgreSQL extension. Cypher must be wrapped as:
  SELECT * FROM ag_catalog.cypher('legal_graph', $$ <cypher here> $$)
  AS (col1 agtype, col2 agtype, ...);

Rules:
- Use only node labels: Contract, Party, Clause, RiskFlag, DateEvent, Obligation
- Use only relationship types: SIGNED_BY, HAS_CLAUSE, HAS_RISK, HAS_DATE, HAS_OBLIGATION, RELATED_TO
- Always LIMIT results to 20 unless the user asks for all
- Output format: <cypher>...</cypher><columns>col1,col2</columns>
```

---

## 5. Token Limits Reference

| Model / component | Context window | Effective limit on local HW | Notes |
|-------------------|---------------|----------------------------|-------|
| Qwen 2.5 Coder 7B | 32K tokens | ~8K effective on 8GB GPU | KV cache exhausts VRAM beyond 8K |
| Qwen 2.5 Coder 14B | 128K tokens | ~16K–32K on RTX 4090 | Use `num_ctx=8192` to save VRAM |
| SQLCoder 15B | 32K tokens | ~16K on RTX 4090 | Trained on 4K schema context |
| Llama 3.1 8B | 128K tokens | ~8K effective | |
| Llama 3.3 70B | 128K tokens | ~32K–64K on A100 | |

### Schema size is the critical variable

DuckDB may expose tables with **hundreds of columns**. The full schema context can easily exceed 4K tokens. Mitigation strategies:

1. **Schema truncation**: include only tables relevant to the question (use schema vector search or keyword match)
2. **Column filtering**: include only the top-N most relevant columns per table
3. **Tool-calling v2 discovery**: let the model request only the tables it needs (`list_tables` → `describe_table`)

---

## 6. Context Window Budget

### v1 — Single prompt SQL generation

```
System prompt (DuckDB rules + table naming)    ~400 tokens
Schema context (tables + columns + types)      ~500–4 000 tokens   ← biggest variable
Conversation history (last N turns)            ~200–1 500 tokens
User question                                  ~20–100 tokens
SQL response                                   ~50–400 tokens
─────────────────────────────────────────────────────────────
Total                                          ~1 200–6 400 tokens
```

At wide schemas (50+ tables, 500+ columns), the schema alone exceeds the effective context window of a 7B model running on 8 GB VRAM. Implement schema filtering before the prompt is assembled.

### v2 — Tool-calling multi-turn

Each tool call adds a round trip. For a 5-turn discovery session:
```
Turn 1: list_tables request + response         ~300 tokens
Turn 2: describe_table (table A)               ~400 tokens
Turn 3: describe_table (table B)               ~400 tokens
Turn 4: describe_table (table C)               ~400 tokens
Turn 5: SQL generation                         ~500 tokens
─────────────────────────────────────────────
Total                                          ~2 000 tokens
```

Tool-calling is more token-efficient than dumping the full schema — but requires a model that reliably generates valid JSON tool calls.

### Self-correction additional budget

Each self-correction retry adds:
```
Failed SQL                                     ~50–400 tokens
Error message                                  ~50–200 tokens
Correction response                            ~50–400 tokens
```

With 3 retries the total budget grows by ~750–3 000 tokens.

---

## 7. VRAM Requirements

### SQL generation (v1 / v2)

| Scenario | VRAM | Fits 8 GB? |
|----------|------|-----------|
| Qwen 2.5 Coder 7B Q4_K_M | ~5 GB | Yes |
| Qwen 2.5 Coder 7B Q8_0 | ~7.5 GB | Yes (tight) |
| Qwen 2.5 Coder 14B Q4_K_M | ~9 GB | **No** |
| SQLCoder 15B Q4_K_M | ~10 GB | **No** |
| Llama 3.3 70B Q4_K_M | ~40 GB | **No** |

### Combined NL2SQL + RAG on same GPU

If the NL2SQL pipeline runs on the same GPU as the RAG pipeline:

| Scenario | VRAM | Fits 8 GB? |
|----------|------|-----------|
| Shared model (one 8B model for both) | ~5 GB | Yes |
| Separate models simultaneously | ~10 GB+ | **No** |
| Coder 7B + nomic-embed | ~5.3 GB | Yes |

Use a **single shared Ollama model** for both NL2SQL and RAG synthesis on 8 GB dev hardware. Specialise to a coder model (it handles both tasks acceptably).

---

## 8. Ollama Configuration

### For NL2SQL — latency priority (interactive queries)

```bash
OLLAMA_KEEP_ALIVE=30m        # keep model warm between queries
OLLAMA_NUM_GPU=99            # all layers on GPU
OLLAMA_NUM_PARALLEL=1        # serialise queries
OLLAMA_FLASH_ATTENTION=1     # reduces VRAM ~20%
```

### Modelfile for SQL generation

```modelfile
FROM qwen2.5-coder:14b-instruct-q4_K_M

PARAMETER num_ctx 8192        # plenty for typical schema + query
PARAMETER temperature 0.0     # fully deterministic SQL
PARAMETER num_gpu 99
PARAMETER num_thread 8
PARAMETER stop "<|im_end|>"   # Qwen stop token
```

Temperature **must be 0.0** for SQL generation — any randomness causes non-reproducible SQL for the same question.

### Handling the self-correction retry loop

The v1 pipeline retries up to 3 times on validation failure. Each retry is a full LLM call.  
On 8 GB VRAM with a 7B model at ~30 tokens/sec, 3 retries × ~200 tokens each ≈ 20 seconds total.  
On RunPod RTX 4090 with 14B at ~30 tokens/sec: same duration but better quality per call.

Set `OLLAMA_KEEP_ALIVE` to at least 5 minutes to avoid cold-start latency between the first attempt and retries.

---

## 9. RunPod GPU Recommendations

| GPU | VRAM | Recommended for |
|-----|------|----------------|
| RTX 4090 | 24 GB | Qwen 2.5 Coder 14B Q4_K_M; interactive NL2SQL |
| A40 | 48 GB | Qwen 2.5 Coder 14B Q8_0; NL2SQL + RAG on same GPU |
| A100 80 GB | 80 GB | Full stack (NL2SQL + RAG + KG) on same GPU; 70B SQL model |

### Why NL2SQL benefits most from a code model on a larger GPU

NL2SQL is the component most sensitive to model size and quality. A 14B coder model makes ~40–50% fewer schema-hallucination errors than an 8B general model. The improvement from 14B → 70B is smaller for SQL than for extraction or reasoning tasks.

**Best cost-quality for NL2SQL production**: RTX 4090 + Qwen 2.5 Coder 14B Q4_K_M.

---

## 10. What Breaks on 8 GB VRAM

| Scenario | What happens | Workaround |
|----------|-------------|------------|
| Running Coder 14B | OOM | Use Coder 7B Q4_K_M |
| Wide schema (100+ columns) in prompt | Context overflow; LLM truncates silently | Implement schema filtering pre-prompt |
| 3 self-correction retries with long SQL | Cumulative context grows; possible truncation | Limit history to last 1 retry in correction prompt |
| Running NL2SQL + RAG embed simultaneously | ~5.3 GB → OK at 7B; OOM at 14B | Serialise or use one shared model |
| Tool-calling v2 with many tool turns | Multi-turn context grows quickly | Cap discovery at 5 tool turns |
| DuckDB reading large GCS Parquet | DuckDB memory (not VRAM) issue | Set `SET memory_limit='4GB'` in DuckDB |

---

## 11. Quantisation Tiers

| Tier | Size vs FP16 | SQL quality impact | Use when |
|------|-------------|-------------------|---------|
| Q8_0 | ~50% | Minimal | A40/A100 preferred for production |
| Q5_K_M | ~69% | Low | Good balance |
| Q4_K_M | ~75% | Moderate — acceptable | **Dev default on 8 GB** |
| Q3_K_M | ~81% | High — SQL failure rate spikes | Avoid for SQL generation |
| Q2_K | ~87% | Severe | Not usable |

SQL generation is particularly sensitive to quantisation because DuckDB syntax precision is required. Never go below Q4_K_M for NL2SQL.

---

## 12. Model Recommendation Matrix

### SQL generation

| Hardware | SQL LLM | Notes |
|----------|---------|-------|
| 8 GB VRAM (dev) | `qwen2.5-coder:7b-instruct-q4_K_M` | Expect ~30% non-trivial query failures |
| RTX 4090 24 GB | `qwen2.5-coder:14b-instruct-q4_K_M` | **Recommended production config** |
| A40 48 GB | `qwen2.5-coder:14b-instruct-q8_0` | Higher quality; same speed as Q4_K_M on RTX 4090 |
| A100 80 GB | `llama3.3:70b-instruct-q4_K_M` | Best quality; marginal improvement over 14B for simple SQL |

### NL→Cypher (current — rule-based; no LLM)

No GPU needed at query time. Rule-based pipeline runs on CPU in microseconds.

### NL→Cypher (if adding LLM fallback)

| Hardware | LLM | Expected accuracy |
|----------|-----|------------------|
| 8 GB VRAM (dev) | `qwen2.5:7b-instruct-q4_K_M` | ~20% correct AGE Cypher |
| RTX 4090 24 GB | `qwen2.5:14b-instruct-q4_K_M` | ~40% correct AGE Cypher |
| A100 80 GB | `llama3.3:70b-instruct-q4_K_M` | ~60% correct AGE Cypher |

Even at 70B the Cypher failure rate is substantial. Always validate LLM-generated Cypher before execution. See `PRODUCTION_RISKS.md §7` for the full risk breakdown.

### Shared model strategy (cost-optimised)

On a single RTX 4090, use **one model for everything**:

```
qwen2.5:14b-instruct-q4_K_M
```

This model handles SQL generation, RAG answer synthesis, and (if needed) NL→Cypher fallback. VRAM: ~9 GB — fits comfortably on RTX 4090 with room for the embedding model.

Keep `OLLAMA_KEEP_ALIVE=30m` — model stays hot between requests and serves all components without cold-start latency.
