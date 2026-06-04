# Prompts Reference

This document covers every prompt in the RAG system: where it lives, what it does, how to tune it, and what to watch for.

---

## Table of Contents

1. [Main System Prompt](#1-main-system-prompt)
2. [Tool Docstrings as Prompts](#2-tool-docstrings-as-prompts)
3. [LLM Reranker Prompt](#3-llm-reranker-prompt)
4. [KG Extraction Prompts](#4-kg-extraction-prompts)
5. [Tuning Guidelines](#5-tuning-guidelines)
6. [Prompt Injection Defence](#6-prompt-injection-defence)

---

## 1. Main System Prompt

**File:** `rag/agent/prompts.py` → `MAIN_SYSTEM_PROMPT`  
**Used by:** `rag/agent/rag_agent.py` — passed to `PydanticAgent(system_prompt=...)`

### What it does

Defines the agent's persona, the three available tools, when to use each, how to combine them, citation rules, and Cypher writing constraints.

### Key sections

| Section | Purpose |
|---------|---------|
| Persona | "Legal Contract Assistant with access to 509 CUAD contracts" |
| Tool routing | Tells the model when to use `search_knowledge_base` vs. `search_knowledge_graph` vs. `run_graph_query` |
| KG schema | Entity types + relationship types — must stay in sync with `kg/legal/common/cuad_ontology.py` |
| Tool combination strategy | Instructs the model to combine graph + text for most questions |
| Citation rules | `[Source: contract_title]` and `[KG: entity_type]` — enforced in the prompt, not in code |
| Cypher rules | MATCH/RETURN only, always include LIMIT, use `toLower()` |

### Full text

```
You are a Legal Contract Assistant with access to 509 CUAD legal contracts.
You have three tools and must choose the right one — or combine them — for each question.
...
```

See `rag/agent/prompts.py` for the full current text.

### Tuning notes

- **Routing failures** (model uses wrong tool): add a negative example to the "Example triggers" list of the correct tool, and add "NOT for: ..." to the wrong tool's description.
- **Hallucinations**: tighten the citation rules section. Add: "If you are not certain a fact came from a retrieved result, say so."
- **Verbose answers**: add "Be concise. Aim for 3-5 sentences unless the user asks for detail."
- **Model swap**: if you switch from Llama to a non-instruction-tuned model, you may need to replace the `##` markdown headers with plain text separators — smaller models ignore markdown.

---

## 2. Tool Docstrings as Prompts

Pydantic AI passes the Python docstring of each `@agent.tool` function to the model as the tool description. These are **prompts**, not just documentation.

### `search_knowledge_base`

**File:** `rag/agent/rag_agent.py:247`

```python
async def search_knowledge_base(
    ctx,
    query: str,
    match_count: int | None = 5,
    search_type: str | None = "hybrid",
    document_source: str | None = None,
    metadata_filters: dict[str, str] | None = None,
) -> str:
    """
    Search the knowledge base for relevant information.
    ...
    document_source: Restrict search to a specific document by its source path
        (e.g. "rag/documents/benefits.md"). Leave None to search all documents.
    metadata_filters: Key-value pairs to filter chunks by metadata fields
        (e.g. {"doc_type": "policy", "category": "hr"}). Leave None for no filter.
    """
```

**What the model sees:** the full docstring including parameter descriptions. The descriptions of `document_source` and `metadata_filters` directly control when the model uses metadata filtering.

**Tuning:** if the model rarely uses `document_source`, make the example more concrete: `"e.g. 'rag/documents/legal/Amazon_contract.md' to search only that file"`.

### `search_knowledge_graph`

Queries the Apache AGE graph for entity/relationship lookups.

**Tuning:** if the model uses graph search for questions that should be text-only, add to the system prompt: "Do NOT use `search_knowledge_graph` for questions about clause language or specific contract text."

### `run_graph_query`

Executes raw openCypher against Apache AGE.

**Safety note:** the tool itself enforces read-only (MATCH/RETURN only) — see `kg/legal/retrieval/cli.py`. The system prompt reinforces this but the code is the real guard.

---

## 3. LLM Reranker Prompt

**File:** `rag/retrieval/rerankers.py` → `LLMReranker`  
**Activated by:** `reranker_enabled: True`, `reranker_type: "llm"` in settings

### What it does

After the hybrid search returns N candidates, the reranker calls the LLM once per chunk (in parallel) asking it to score relevance 0–10, then re-sorts.

### Typical prompt shape

```
Score the relevance of the following passage to the query on a scale of 0-10.
Return only the integer score, nothing else.

Query: {query}

Passage: {chunk_content}

Score:
```

### Tuning notes

- **Slow reranking:** reduce `reranker_overfetch_factor` (default 3) — fewer candidates to score.
- **Bad scores:** add "A score of 10 means the passage directly and completely answers the query. A score of 0 means it is completely unrelated." to the prompt.
- **Cross-encoder alternative:** set `reranker_type: "cross_encoder"` — no LLM call, uses a local bi-encoder model (`BAAI/bge-reranker-base`), much faster.

---

## 4. KG Extraction Prompts

**File:** `kg/legal/ingestion/extraction_pipeline.py`  
**Used during:** one-time KG ingestion (not at query time)

### Bronze pass — raw extraction

Prompt instructs the model to extract entities and relationships from a contract chunk as JSON. Strict JSON-only output format.

```
Extract all entities and relationships from the following legal contract text.
Return a JSON object with keys "entities" and "relationships".

Entity format: {"name": str, "type": str, "confidence": float}
Relationship format: {"source": str, "relation": str, "target": str, "confidence": float}

Valid entity types: {VALID_LABELS}
Valid relationship types: {VALID_REL_TYPES}

Text:
{chunk_text}

JSON:
```

### Silver pass — validation and deduplication

A second prompt validates Bronze entities, normalises names, and merges duplicates. See `extraction_pipeline.py` for the full prompt.

### Tuning notes

- **JSON parse failures:** add `"Return ONLY the JSON object. No explanation, no markdown code fences."` to the prompt end.
- **Wrong entity types:** add more few-shot examples showing the correct type for edge cases.
- **Low confidence scores:** the model often assigns 0.9+ to everything; add calibration guidance: "Only assign 0.9+ when the entity or relationship is unambiguous."

---

## 5. Tuning Guidelines

### General principles

1. **One change at a time.** The system prompt, tool docstrings, HyDE prompt, and reranker prompt all affect retrieval quality independently. Change one, evaluate, then change the next.

2. **Measure before tuning.** Run `python -m pytest rag/tests/retrieval/ -v` (or `kg/tests/`) to get a Hit Rate / MRR baseline before any prompt change.

3. **Small models need more structure.** Llama 3.1 8B needs explicit "Return ONLY JSON" and clear section separators. Larger models (70B+) tolerate more natural prose.

4. **Citation rules degrade over long context.** If the model starts dropping `[Source: ...]` citations on long answers, move the citation rule to the *end* of the system prompt (recency bias helps).

### Common failure modes

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Model calls wrong tool | Ambiguous tool descriptions | Sharpen "Example triggers" and "NOT for:" |
| No citations | Citation rule buried in prompt | Move citation section to end of system prompt |
| HyDE hallucinations passed to embedder | Query not factual | Add "Base the passage strictly on legal contract conventions" |
| Reranker assigns 10/10 to everything | Calibration missing | Add explicit 0/10 and 10/10 examples |
| KG extraction returns empty `[]` | Model ignores JSON schema | Add `"If no entities found, return {\"entities\": [], \"relationships\": []}"` |

---

## 6. Prompt Injection Defence

User queries reach the LLM via `search_knowledge_base` — the query is embedded in the tool result, not injected into the system prompt directly. However, malicious content in retrieved chunks could attempt to override instructions.

### Current mitigations

- The system prompt uses a dedicated persona with explicit output constraints (citations, Cypher-only).
- The `run_graph_query` tool enforces read-only Cypher at the code level regardless of what the prompt says.
- Pydantic AI's structured output mode (when used) prevents free-form instruction override.

### Planned mitigations (see TODO.md)

- Input validation layer on `/v1/chat` to detect and reject prompt injection patterns before the query reaches the agent.
- Query length cap (`max_query_length` setting) to prevent context flooding.

### What to watch for

- Queries containing `"Ignore previous instructions"`, `"You are now"`, `"Forget your system prompt"`.
- Queries that look like Cypher or SQL (`MATCH`, `SELECT`, `DROP TABLE`).
- Unusually long queries (>500 chars) — potential context stuffing.
