# Prompts Reference

Every prompt in the RAG system: where it lives, what it does, and how to tune it.

---

## Table of Contents

1. [Main System Prompt](#1-main-system-prompt)
2. [Corpus-Specific Prompts](#2-corpus-specific-prompts)
3. [Tool Docstrings as Prompts](#3-tool-docstrings-as-prompts)
4. [LLM Reranker Prompt](#4-llm-reranker-prompt)
5. [Tuning Guidelines](#5-tuning-guidelines)
6. [Prompt Injection Defence](#6-prompt-injection-defence)

---

## 1. Main System Prompt

**File:** `rag/agent/prompts.py` → `MAIN_SYSTEM_PROMPT`  
**Used by:** `rag/agent/rag_agent.py` — passed to `PydanticAgent(system_prompt=...)`

### What it does

General-purpose knowledge base assistant prompt. Defines the agent's behaviour, when to call `search_knowledge_base`, mandatory citation rules, and uncertainty handling. Domain-agnostic — works with any ingested corpus.

### Key sections

| Section | Purpose |
|---------|---------|
| Tool routing | When to call `search_knowledge_base` (always before answering facts) |
| Citation rules | `[Source: document_title, chunk_id]` — enforced in prompt, not in code |
| Uncertainty rules | Instructs the model to say "I don't have that information" rather than hallucinate |
| When NOT to search | Greetings and follow-ups on already-retrieved context |

### Tuning notes

- **Model ignores citation rule:** move the citation section to the *end* of the prompt — recency bias helps small models.
- **Model answers without searching:** strengthen "Always search first" with a negative example: "Do not answer from memory — call the tool."
- **Verbose answers:** add "Be concise. Aim for 3–5 sentences unless the user asks for detail."
- **Small models:** replace `##` markdown headers with plain text separators — sub-8B models often ignore markdown structure.

---

## 2. Corpus-Specific Prompts

When the agent is configured for a specific corpus, swap `MAIN_SYSTEM_PROMPT` for a corpus-specific prompt.

### LEGAL_CONTRACT_SYSTEM_PROMPT

**File:** `rag/agent/prompts.py` → `LEGAL_CONTRACT_SYSTEM_PROMPT`  
**Corpus:** CUAD legal contracts (moved to `misc/kg_legal_cuad/`)

Used when the agent is configured with the CUAD corpus and the Apache AGE knowledge graph. Extends the base prompt with:
- Descriptions of all five tools including the four KG tools
- KG entity/relationship schema (must stay in sync with `misc/kg_legal_cuad/kg_legal/common/cuad_ontology.py`)
- Tool combination strategy (graph lookup → text retrieval)
- Cypher writing rules (MATCH/RETURN only, always LIMIT, toLower for names)

To activate:
```python
# rag/agent/rag_agent.py
from rag.agent.prompts import LEGAL_CONTRACT_SYSTEM_PROMPT
agent = PydanticAgent(get_llm_model(), system_prompt=LEGAL_CONTRACT_SYSTEM_PROMPT, ...)
```

---

## 3. Tool Docstrings as Prompts

Pydantic AI passes the Python docstring of each `@agent.tool` function to the model as the tool description. These are **prompts**, not just documentation — change them carefully.

### `search_knowledge_base`

**File:** `rag/agent/rag_agent.py:247`

The docstring describes the parameters the model can pass, including `document_source` and `metadata_filters`. The descriptions of these parameters directly control when the model uses metadata filtering — if the description is vague, the model will rarely use them.

**Tuning:** if the model never uses `document_source`, make the example more concrete: `"e.g. 'rag/documents/benefits.md' to restrict to that file"`.

### KG tools (`search_knowledge_graph`, `search_hybrid_kg`, `run_graph_query`, `nl_graph_query`)

These four tools are corpus-specific (CUAD legal contracts + Apache AGE graph). Their docstrings describe the KG schema and query patterns. When using a general corpus without a knowledge graph, these tools are registered but will return empty results — the model will learn not to call them if the system prompt doesn't mention them.

---

## 4. LLM Reranker Prompt

**File:** `rag/retrieval/rerankers.py` → `LLMReranker`  
**Activated by:** `reranker_enabled = True`, `reranker_type = "llm"` in settings (both off by default)

### What it does

After hybrid search returns N candidates, the reranker calls the LLM once per chunk in parallel asking for a 0–10 relevance score, then re-sorts and trims to `match_count`.

### Prompt shape

```
Score the relevance of the following passage to the query on a scale of 0-10.
Return only the integer score, nothing else.

Query: {query}

Passage: {chunk_content}

Score:
```

### Tuning notes

- **Slow reranking:** lower `reranker_overfetch_factor` (default 3) — fewer candidates to score.
- **Model assigns 10/10 to everything:** add calibration: "A score of 10 means the passage directly and completely answers the query. A score of 0 means it is completely unrelated."
- **Speed over accuracy:** switch to `reranker_type = "cross_encoder"` — local `BAAI/bge-reranker-base` via `sentence-transformers`, no LLM call, much faster.

---

## 5. Tuning Guidelines

### General principles

1. **One change at a time.** `MAIN_SYSTEM_PROMPT`, tool docstrings, and the reranker prompt all affect answer quality independently. Change one, evaluate, then the next.

2. **Measure before tuning.** Run `python -m pytest rag/tests/retrieval/ -v` to get a Hit Rate / MRR baseline before any prompt change.

3. **Small models need more structure.** Llama 3.1 8B needs explicit "Return ONLY JSON" and clear section separators. Larger models (70B+) tolerate more natural prose.

4. **Citation rules degrade over long context.** If the model drops `[Source: ...]` on long answers, move the citation rule to the *end* of the system prompt.

### Common failure modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Model answers without searching | "Always search" rule not prominent enough | Move it to the first line of the prompt |
| Wrong tool called | Ambiguous tool descriptions | Sharpen "Example triggers" and add "NOT for:" |
| No citations | Citation rule buried in prompt | Move citation section to end of system prompt |
| Reranker assigns 10/10 to everything | Calibration missing | Add explicit 0/10 and 10/10 examples |
| KG extraction returns empty `[]` | Model ignores JSON schema | Add `"If no entities found, return {\"entities\": [], \"relationships\": []}"` |

---

## 6. Prompt Injection Defence

User queries reach the LLM via `search_knowledge_base` — the query is embedded in the tool result, not injected into the system prompt directly. However, malicious content in retrieved chunks could attempt to override instructions.

### Current mitigations

- The system prompt uses a dedicated persona with explicit output constraints (citations, uncertainty rules).
- The `run_graph_query` tool enforces read-only Cypher at the code level regardless of what the prompt says.
- Pydantic AI's structured output mode (when used) prevents free-form instruction override.

### Planned mitigations (see RAGV2_DESIGN.md — Query Validation & Hook System)

- V4 prompt injection detector: regex + embedding similarity against known attack patterns, fires before any LLM call.
- V5 content policy classifier (nano model): rejects off-topic and inappropriate queries.
- Query length cap (`MAX_QUERY_CHARS = 4096`) to prevent context flooding.

### What to watch for

- Queries containing `"Ignore previous instructions"`, `"You are now"`, `"Forget your system prompt"`.
- Queries that look like Cypher or SQL (`MATCH`, `SELECT`, `DROP TABLE`).
- Unusually long queries (> 500 chars) — potential context stuffing.
