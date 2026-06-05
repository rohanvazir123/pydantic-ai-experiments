# Prompts Reference

Every prompt in the RAG system: where it lives, what it does, and how to tune it.

> **Architecture note:** The current `rag/agent/rag_agent.py` carries significant legacy from the CUAD legal contract corpus — four KG tools hardwired to Apache AGE, mixed concerns between RAG retrieval and graph traversal, and CUAD-specific docstrings still in tool definitions. This is a known mess. The `knowledge/` module planned in `RAGV2_DESIGN.md` (Phase C) redesigns this cleanly: corpus-configurable tool sets, separated agent + graph modules, and a proper hook system for prompt injection guards.

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
**Used by:** `rag/agent/rag_agent.py:243` — `PydanticAgent(system_prompt=MAIN_SYSTEM_PROMPT)`

General-purpose knowledge base assistant. Search-first, cite everything, acknowledge uncertainty. Domain-agnostic — works with any ingested corpus.

| Section | What it controls |
|---------|-----------------|
| Tool routing | "Always search before answering facts" — forces retrieval; without this the model answers from training weights |
| Citation rules | `[Source: document_title, chunk_id]` — enforced in prompt, not in code |
| Uncertainty | "Say 'I don't have that information'" — prevents hallucination on empty retrieval |
| When NOT to search | Greetings, follow-ups on already-retrieved results |

**Tuning:**
- Model answers without searching → move "Always search first" to line 1
- No citations → move citation rule to end of prompt (recency bias)
- Small models (< 8B) → replace `##` markdown headers with plain text separators

---

## 2. Corpus-Specific Prompts

Swap `MAIN_SYSTEM_PROMPT` for a corpus-specific prompt when the agent is configured for a domain corpus.

**`LEGAL_CONTRACT_SYSTEM_PROMPT`** — `rag/agent/prompts.py`

For the CUAD corpus + Apache AGE KG. Describes all five tools, the KG entity/relationship schema, tool combination strategy, and Cypher writing rules. Schema must stay in sync with `misc/kg_legal_cuad/kg_legal/common/cuad_ontology.py`.

To activate:
```python
from rag.agent.prompts import LEGAL_CONTRACT_SYSTEM_PROMPT
agent = PydanticAgent(get_llm_model(), system_prompt=LEGAL_CONTRACT_SYSTEM_PROMPT, ...)
```

---

## 3. Tool Docstrings as Prompts

Pydantic AI sends each `@agent.tool` docstring to the model as the tool description. **These are prompts.** The model decides whether to call a tool — and with what arguments — based almost entirely on these docstrings.

### `search_knowledge_base` — the only general-purpose tool

```python
async def search_knowledge_base(
    ctx: PydanticRunContext,
    query: str,
    match_count: int | None = 5,
    search_type: str | None = "hybrid",
    document_source: str | None = None,
    metadata_filters: dict[str, str] | None = None,
) -> str:
    """
    Search the knowledge base for relevant information.

    Combines RAG retrieval with Mem0 user context (if enabled and user_id provided).

    Args:
        ctx: Agent runtime context
        query: Search query text
        match_count: Number of results to return (default: 5)
        search_type: Type of search - "semantic", "text", or "hybrid" (default)
        document_source: Restrict search to a specific document by its source path
            (e.g. "rag/documents/benefits.md"). Leave None to search all documents.
        metadata_filters: Key-value pairs to filter chunks by metadata fields
            (e.g. {"doc_type": "policy", "category": "hr"}). Leave None for no filter.

    Returns:
        String containing the retrieved information formatted for the LLM,
        optionally prefixed with user context from Mem0
    """
```

The `document_source` and `metadata_filters` parameter descriptions directly control whether the model uses scoped search. If the description is vague, the model will rarely pass these — make the examples concrete.

### `search_knowledge_graph` — CUAD entity lookup (orphaned for general use)

```python
async def search_knowledge_graph(
    ctx: PydanticRunContext,
    query: str,
    entity_type: str | None = None,
    limit: int | None = 15,
) -> str:
    """
    Search the legal knowledge graph for entities and relationships.

    Use this tool when the question asks about:
    - Parties to contracts ("who are the parties?")
    - Governing law / jurisdiction ("which contracts are governed by X law?")
    - Specific clause types (termination, license, non-compete, liability)
    - Relationships between companies and contracts
    ...
    """
```

### `search_hybrid_kg` — parallel semantic + graph fusion

```python
async def search_hybrid_kg(
    ctx: PydanticRunContext,
    query: str,
    match_count: int = 5,
) -> str:
    """
    Run both semantic retrieval (vector + BM25 + RRF) and KG structured reasoning
    in parallel, then fuse the results into a single context block.

    Use this tool for questions that need both:
    - Clause text / supporting passages (semantic path)
    - Graph facts: parties, jurisdictions, relationships (KG path)

    For example:
    - "Which parties in contracts governed by Delaware law indemnify each other,
       and what do those indemnification clauses say?"
    ...
    """
```

### `run_graph_query` — raw Cypher

```python
async def run_graph_query(
    ctx: PydanticRunContext,
    cypher: str,
) -> str:
    """
    Execute a read-only openCypher MATCH query against the Apache AGE knowledge graph.

    Use this tool when you already know the exact Cypher. For natural-language
    questions, prefer ``nl_graph_query`` — it writes the Cypher for you.

    Use this tool for:
    - Multi-hop traversal (e.g. Party → Contract → Jurisdiction)
    - Aggregation / analytics: counts, distributions, co-occurrence
    - Complex pattern matching that search_knowledge_graph cannot express

    Only MATCH/RETURN queries are permitted. CREATE/MERGE/SET/DELETE are blocked.
    Always include a LIMIT clause.
    ...
    """
```

The `prefer nl_graph_query` instruction here is a routing hint — the model reads it and delegates to the other tool when it doesn't know the Cypher.

### `nl_graph_query` — NL→Cypher

```python
async def nl_graph_query(
    ctx: PydanticRunContext,
    question: str,
) -> str:
    """
    Answer a natural-language question by routing to the right graph schema,
    generating Cypher, and executing it against the Apache AGE knowledge graph.

    Use this tool instead of ``run_graph_query`` when you do not already know
    the exact Cypher — this tool writes the query for you.

    Pipeline:
      1. GraphRouter classifies *question* → relevant graph type(s)
      2. get_schema() returns the compact, token-bounded schema for those types.
      3. NL2CypherConverter sends (question, schema) to LLM → MATCH…RETURN query.
      4. AgeGraphStore.run_cypher_query() executes it and returns a table string.
    ...
    """
```

The pipeline description in the docstring is visible to the model — it teaches the model what happens internally so it can set expectations when answering.

---

## 4. LLM Reranker Prompt

**File:** `rag/retrieval/rerankers.py` → `LLMReranker`  
**Off by default** (`reranker_enabled = False`, `reranker_type = "llm"`)

```
Score the relevance of the following passage to the query on a scale of 0-10.
Return only the integer score, nothing else.

Query: {query}

Passage: {chunk_content}

Score:
```

**Tuning:**
- All 10s → add: "A score of 10 means the passage directly and completely answers the query."
- Too slow → lower `reranker_overfetch_factor` (default 3), or switch to `reranker_type = "cross_encoder"` (local, no LLM call)

---

## 5. Tuning Guidelines

One change at a time. Run `python -m pytest rag/tests/retrieval/ -v` for a Hit Rate / MRR baseline before any prompt change.

| Symptom | Fix |
|---------|-----|
| Model answers without searching | Move "Always search first" to line 1 of system prompt |
| Wrong tool called | Sharpen "Example triggers" and add "NOT for: ..." |
| No citations | Move citation rule to end of system prompt (recency bias) |
| Reranker scores everything 10/10 | Add calibration example to reranker prompt |

---

## 6. Prompt Injection Defence

User queries reach the LLM via `search_knowledge_base` — embedded in the tool result, not the system prompt. Retrieved chunks could attempt instruction override.

**Current:** System prompt persona + explicit constraints; `run_graph_query` enforces read-only Cypher in code regardless of prompt.

**Planned** (see `RAGV2_DESIGN.md` — Query Validation & Hook System):
- V4 regex + embedding injection detector, fires before any LLM call
- V5 nano-model content policy classifier
- `MAX_QUERY_CHARS = 4096` cap
