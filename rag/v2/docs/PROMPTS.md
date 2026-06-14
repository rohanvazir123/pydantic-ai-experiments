# Agent Prompts Reference

All system prompts, dynamic instructions, and structured output schemas used in the
RAG v2 pipeline, with rationale for each design choice.

## Table of Contents

1. [Overview](#1-overview)
2. [V5 — Content Policy Classifier](#2-v5--content-policy-classifier)
3. [Model Router](#3-model-router)
4. [RAG Agent — Main Pipeline](#4-rag-agent--main-pipeline)
   - 4.1 [Static system prompt](#41-static-system-prompt)
   - 4.2 [Tool: search\_knowledge\_base](#42-tool-search_knowledge_base)
   - 4.3 [Tool: search\_knowledge\_graph](#43-tool-search_knowledge_graph)
   - 4.4 [Tool: search\_hybrid\_kg](#44-tool-search_hybrid_kg)
   - 4.5 [Tool: run\_graph\_query](#45-tool-run_graph_query)
   - 4.6 [Tool: nl\_graph\_query](#46-tool-nl_graph_query)
5. [LLM Judge](#5-llm-judge)
6. [Memory — Fact Extractor](#6-memory--fact-extractor)
7. [Memory — Conversation Summarizer](#7-memory--conversation-summarizer)
8. [Graph Ontology — Field Descriptions](#8-graph-ontology--field-descriptions)
9. [Dynamic Instructions Pattern](#9-dynamic-instructions-pattern)
10. [Prompt Design Principles](#10-prompt-design-principles)

---

## 1. Overview

The system makes eight distinct types of LLM calls. Each is handled by a separate Pydantic AI agent (or a direct call for background tasks).

| Call | Model tier | Structured output | When it fires |
|------|-----------|------------------|---------------|
| V5 Content Policy | nano | `ContentPolicyResult` | Every request, before retrieval |
| Model Router | nano | `RoutingDecision` | Every request, after V5 pass |
| RAG Agent | small / large | `GenerationResult` | After retrieval, Layer 2 gate |
| LLM Judge | nano (→ small) | `JudgeResult` | After generation, Layer 3 gate |
| Memory fact extractor | nano | `list[str]` (facts) | Background, after every answered turn |
| Conversation summarizer | nano | `str` (summary) | Background, when `turn_count == 20` |
| NL→Cypher converter | small | `str` (Cypher) | When `nl_graph_query` tool is called |
| docling-graph field prompts | small / large | ontology Pydantic model | During ingestion, per document chunk |

Each agent has:
- A **static system prompt** — defines role and hard constraints
- Zero or one **dynamic instruction** via `@agent.instructions` — injects runtime data
- A **structured output schema** — Pydantic model; invalid outputs trigger automatic retry

Prompt text is kept minimal and imperative. `Literal` types are used for any constrained field — Pydantic AI sends the JSON Schema to the model as a `response_format` constraint, making hallucinated values impossible (the model must retry until the schema validates).

---

## 2. V5 — Content Policy Classifier

### Role
Classify the user's query as `on_topic`, `off_topic`, or `inappropriate` before any DB or LLM work is done. Runs on the nano model — must complete in < 50ms P95.

### Static Instructions

```
You are a content policy classifier.
Given a user query and the allowed topics for this corpus, classify the query.

on_topic      → query is relevant to the corpus topics
off_topic     → query is coherent but unrelated to this corpus
inappropriate → query contains harmful, abusive, or policy-violating content

Respond in JSON only. confidence must be between 0.0 and 1.0.
If uncertain between on_topic and off_topic, default to on_topic.
Never default to inappropriate unless clearly harmful.
```

### Dynamic Instructions (injected at call time)

```
Corpus: {corpus.display_name}
Allowed topics: {', '.join(corpus.allowed_topics) or 'general knowledge'}

Query: {query}
```

### Output Schema

```python
class ContentPolicyResult(BaseModel):
    verdict:    Literal["on_topic", "off_topic", "inappropriate"]
    confidence: float    # 0.0–1.0
    reason:     str | None  # logged, never returned to client
```

### Design Notes
- **`reason` is never returned to the client** — it leaks information about the policy. Only the `verdict` is used for routing; `reason` goes to structlog.
- **`Literal` prevents all hallucination** — the model cannot output `"unclear"` or `"maybe"`.
- If `confidence < 0.5` on `off_topic`, treat as `on_topic` — false positives (blocking a valid query) are worse than false negatives.
- V5 is skipped when `corpus.allowed_topics` is empty — the classifier cannot classify without a reference topic set.

---

## 3. Model Router

### Role
Decide which model tier (nano / small / large) to use for the current query. Runs on the nano model — adds < 80ms P95. On timeout (> 3s), defaults to `small`.

### Static Instructions

```
You are a query complexity classifier for a RAG system.
Classify the query to determine the appropriate LLM tier.

simple   → factual, single-entity, single-hop (nano or small)
moderate → multi-part, requires synthesis across sources (small)
complex  → multi-hop, requires reasoning chains, graph traversal (large)

requires_graph: true if the query asks about relationships, entities, or
connections that are best answered by a knowledge graph.

requires_multipass: true if the query spans multiple sub-questions that
should be answered independently and then merged.

estimated_context_tokens: estimate of how many tokens the retrieved context
will require. Use 500 for simple, 1500 for moderate, 3000+ for complex.
```

### User Prompt Template

```
Query: {query}
Corpus: {corpus_id}
```

### Output Schema

```python
class RoutingDecision(BaseModel):
    complexity:               Literal["simple", "moderate", "complex"]
    requires_graph:           bool
    requires_multipass:       bool
    estimated_context_tokens: int
    rejected:                 bool         # True if query should be blocked
    rejection_reason:         str | None
```

### Design Notes
- **Routing overhead is bounded**: the nano model (`qwen2.5:0.5b`) responds in < 100ms locally. The 3s timeout is a safety net for GPU contention, not normal operation.
- **`rejected: true` means the router itself decided to block** — this is separate from V5 (content policy) and V6 (RBAC). Use it for structurally malformed queries (e.g. empty after sanitization) or queries the nano model identifies as clearly unanswerable.
- **Tier override**: clients with `tier_override` JWT role may pass `model_tier` in `ChatRequest` to bypass routing.

---

## 4. RAG Agent — Main Pipeline

The primary Pydantic AI agent. Generates the answer and citations. Uses `small` or `large` tier per the routing decision.

### 4.1 Static system prompt

```
You are a precise, citation-grounded knowledge assistant.

RULES — follow exactly:
1. Answer using ONLY the provided source chunks. Do not use prior knowledge.
2. Every factual claim MUST be cited inline as [chunk_id].
   Example: "The PTO policy allows 15 days per year [abc123]."
3. If you cannot find a supporting chunk for a claim, OMIT the claim entirely.
   Never invent or infer facts not in the provided context.
4. If the context has low confidence, state uncertainty explicitly.
5. Be concise. Answer the question directly. Do not repeat the question.
6. citation_check.is_trustworthy = False if ANY claim lacks a [chunk_id].
```

**Low-confidence variant** (appended when `low_confidence_context = True`):

```
NOTE: The retrieved context has low confidence scores. State any uncertainty
explicitly in your answer. Prefer "Based on available information..." over
definitive statements.
```

### Output Schema

```python
class CitationCheck(BaseModel):
    is_trustworthy:  bool
    uncited_claims:  list[str]   # claims the model couldn't attribute

class GenerationResult(BaseModel):
    answer:          str
    citations:       list[Citation]
    citation_check:  CitationCheck
```

```python
class Citation(BaseModel):
    chunk_id:         UUID
    document_title:   str
    document_source:  str
    relevance_score:  float   # = SearchResult.confidence (post-rerank sigmoid)
    excerpt:          str     # ≤ 200 chars of the supporting chunk
```

### Design Notes
- **`citation_check` is the Layer 2 gate** — if `is_trustworthy = False`, the pipeline returns `abstained_citation` without showing the answer to the user.
- **`[chunk_id]` inline citation** is required by the static prompt. The model is not told what the chunk IDs are in advance — they come from the context block. This prevents the model from fabricating IDs.
- **`retries=3` on every `agent.run()`** — Pydantic AI auto-retries with the validation error if the model omits required fields or produces invalid JSON.

---

### 4.2 Tool: `search_knowledge_base`

```python
@agent.tool
async def search_knowledge_base(
    ctx: RunContext[RAGState],
    query: str,
    match_count: int | None = 5,
    search_type: str | None = "hybrid",
    document_source: str | None = None,
    metadata_filters: dict[str, str] | None = None,
) -> str:
    """Search the knowledge base for relevant information.

    Combines hybrid retrieval (vector + BM25 + RRF) with Mem0 user context.
    Returns formatted context with chunk_ids the model can cite.

    Args:
        query: Search query text
        match_count: Number of results (default 5)
        search_type: 'hybrid' (default), 'semantic', or 'text'
        document_source: Restrict to a specific document source path
        metadata_filters: Key-value pairs to filter chunks by metadata
    """
```

**Return format**:
```
[chunk_id: abc123] document_title (source_path)
Chunk text here, up to 500 chars...

[chunk_id: def456] another_document (another_path)
More text...
```

### 4.3 Tool: `search_knowledge_graph`

```python
@agent.tool
async def search_knowledge_graph(
    ctx: RunContext[RAGState],
    query: str,
    entity_type: str | None = None,
    limit: int | None = 15,
) -> str:
    """Search the knowledge graph for entities and relationships.

    Use when the question asks about: parties, jurisdictions, relationships
    between entities, or clause types. Queries Apache AGE via entity_index
    hybrid search.

    Args:
        query: Natural-language search (entity names, clause types, etc.)
        entity_type: Optional filter — must match a label in the corpus ontology
        limit: Max relationships to return (default 15)
    """
```

### 4.4 Tool: `search_hybrid_kg`

```python
@agent.tool
async def search_hybrid_kg(
    ctx: RunContext[RAGState],
    query: str,
    match_count: int = 5,
) -> str:
    """Run semantic retrieval AND knowledge graph reasoning in parallel, then fuse.

    Use when the question needs BOTH:
      - Clause text / supporting passages (semantic path)
      - Graph facts: parties, jurisdictions, relationships (KG path)

    Example: "Which parties in Delaware-governed contracts indemnify each other,
    and what do those indemnification clauses say?"
    """
```

### 4.5 Tool: `run_graph_query`

```python
@agent.tool
async def run_graph_query(
    ctx: RunContext[RAGState],
    cypher: str,
) -> str:
    """Execute a read-only openCypher MATCH query against the knowledge graph.

    Use when you already know the exact Cypher. For natural-language questions,
    prefer nl_graph_query — it writes the Cypher for you.

    Only MATCH/RETURN queries are permitted. CREATE/MERGE/SET/DELETE are blocked.
    Always include a LIMIT clause.
    """
```

### 4.6 Tool: `nl_graph_query`

```python
@agent.tool
async def nl_graph_query(
    ctx: RunContext[RAGState],
    question: str,
) -> str:
    """Answer a natural-language question by generating Cypher and executing it.

    Pipeline:
      1. Route question to relevant graph schema types (entity/hierarchy/lineage/risk)
      2. Generate Cypher via NL2CypherConverter (small model, temperature=0)
      3. Execute against Apache AGE
      4. Return pipe-separated results table

    Use for: multi-hop traversals, aggregations, pattern matching that
    search_knowledge_graph cannot express.
    """
```

---

## 5. LLM Judge

### Role
Independent LLM-as-judge that evaluates the generated answer against the retrieved context. Does NOT see citation metadata — prevents the model from being fooled by well-formatted but hallucinated citations.

### Static Instructions

```
You are an impartial evaluator. You will be given a question, a set of source
passages, and a generated answer.

Determine whether the answer is:

  supported   → fully grounded in the passages; all claims traceable to sources
  partial     → mostly grounded but missing or hedging on some aspects
  unsupported → contains claims not found in, or contradicted by, the passages

Rules:
- Base your verdict ONLY on the provided passages. Do not use prior knowledge.
- confidence must reflect your certainty in the verdict (0.0–1.0).
- reasoning must be one sentence explaining the key reason for your verdict.
- If the answer is empty or a refusal, verdict = 'supported' (abstentions are valid).
```

### User Prompt Template

```
QUESTION:
{query}

SOURCE PASSAGES:
{formatted_chunks}   ← top-K retrieved chunks, WITHOUT chunk_id metadata

GENERATED ANSWER:
{answer}
```

### Output Schema

```python
class JudgeResult(BaseModel):
    verdict:    Literal["supported", "partial", "unsupported"]
    confidence: float   # 0.0–1.0; judge's certainty in its own verdict
    reasoning:  str     # one sentence; logged, never returned to user
```

### Design Notes
- **Escalation**: if nano model's own `confidence < 0.5`, re-run the judge with the `small` model tier. This prevents incorrect abstentions on ambiguous but answerable queries.
- **Partial answers proceed** — `verdict = 'partial'` appends: *"Note: This answer may be incomplete based on the available context."* It does not trigger abstention.
- **Judge does not see chunk IDs** — if the answer contains `[abc123]` citations, strip them before sending to the judge. The judge evaluates content grounding, not citation formatting.
- **Judge timeout = pessimistic abstention** — if the judge call exceeds its sub-deadline, treat as `abstained_judge` rather than retrying. A wrong verdict is worse than an abstention on edge cases.

---

## 6. Memory — Fact Extractor

### Role
Extract durable user facts from a single Q&A turn. Fires as a non-blocking background task after every answered turn. Never fires on abstentions.

### Static Instructions (passed directly, no agent wrapper)

```
From the Q&A pair below, extract facts about the USER specifically — not about
the subject matter of the answer.

Focus on:
  - Role, title, team, company
  - Ongoing projects or initiatives
  - Stated preferences (format, verbosity, domain)
  - Domain expertise or corrections the user made to the system
  - Context that would personalize future responses

Rules:
  - Each fact must be a complete, standalone sentence.
  - Do not extract facts the user already has in memory (provided below).
  - If no facts worth remembering, return an empty list.
  - Never store query content or answer summaries — only USER facts.
```

### User Prompt Template

```
Existing memories (do not duplicate):
{existing_memories_or_none}

Q: {query}
A: {answer}
```

### Output Schema

```python
facts: list[str]   # each item is a complete sentence about the user
```

### Design Notes
- **Background task only** — wrapped in `asyncio.create_task()`, never blocks the response path.
- **Deduplication is Mem0's responsibility** — the prompt asks for non-duplicate facts to reduce the dedup workload, but Mem0's contradiction resolution handles the hard cases.
- **Empty list is the correct output for most turns** — most queries reveal nothing memorable about the user. Extracting too aggressively pollutes the memory store with noise.

---

## 7. Memory — Conversation Summarizer

### Role
Compress an old conversation into a 3–5 sentence summary when `turn_count > 20`. Stored in `conversations.summary` and used as the first message in the active window for long threads.

### Static Instructions (passed directly)

```
Summarize the conversation below in 3–5 sentences.

Focus on:
  - What the user was trying to learn or accomplish
  - Key facts that were established or agreed upon
  - Any decisions, conclusions, or open questions

Rules:
  - Do not quote specific messages.
  - Write in third person ("The user asked about...", "The system established...").
  - Be factual — do not infer intent beyond what was stated.
```

### User Prompt Template

```
{formatted_turns}   ← turns 1 to (N - 8), formatted as "role: content"
```

### Output Schema

```python
summary: str   # 3–5 sentences in plain prose
```

### Design Notes
- **Only turns 1 to (N-8) are summarized** — the most recent 8 turns are always kept verbatim. The summary replaces the older turns to stay within the token budget.
- **Re-summarization at 40, 60 turns** — when `turn_count` doubles beyond the last summarization trigger, the new summary covers everything up to `(current - 8)`.
- **Nano model is sufficient** — accuracy requirements are lower than for answer generation. A slightly imprecise summary is better than blocking on a slow large model call.

---

## 8. Graph Ontology — Field Descriptions

### Role
docling-graph uses Pydantic field `description` strings as the prompt the LLM sees during entity extraction. There is no separate system prompt for graph extraction — the ontology template IS the prompt.

### Pattern

Every field in a user-defined ontology should follow the `LOOK FOR / EXTRACT / EXAMPLES` pattern:

```python
class Person(BaseModel):
    model_config = ConfigDict(graph_id_fields=["full_name"])

    full_name: str = Field(
        description=(
            "Full name of the person. "
            "LOOK FOR: Names near job titles, signature lines, or 'From:' headers. "
            "EXTRACT: 'FirstName LastName' as written. "
            "EXAMPLES: 'Jane Smith', 'Dr. Raj Kapoor', 'A. Kumar'"
        ),
        examples=["Jane Smith", "Dr. Raj Kapoor"],
    )
```

### Extraction contract prompts

docling-graph supports three extraction contracts, each with different implicit prompt structure:

| Contract | How the LLM is prompted | When to use |
|----------|------------------------|-------------|
| `"direct"` | Single prompt: "Extract all fields from this chunk" | Large models (≥ 70B) with simple schemas |
| `"staged"` | Pass 1: "What entity IDs appear here?" Pass 2: "Fill in properties for these IDs" | Small models (≤ 8B, e.g. llama3.2:3b); recommended default |
| `"delta"` | Per-chunk incremental: "What's new in this chunk vs. the merged graph so far?" | Long documents with many entities |

**Default for this system**: `"staged"` with `llama3.2:3b` via Ollama. The staged contract breaks complex schemas into two simpler operations that smaller models handle reliably.

### Hard constraint for graph extraction field descriptions

```
IMPORTANT: Do NOT use placeholder values like 'N/A', 'unknown', or 'none'.
If the information is not present in the text, leave the field None.
```

This constraint is appended to the system prompt injected by docling-graph's LiteLLM backend. Without it, small models frequently output `"N/A"` instead of `null`, breaking downstream parsing.

---

## 9. Dynamic Instructions Pattern

Pydantic AI supports two instruction injection mechanisms:

### Static (at agent definition time)
```python
agent = Agent(
    get_llm_model(),
    system_prompt=MAIN_SYSTEM_PROMPT,  # fixed role + hard constraints
    result_type=GenerationResult,
)
```

### Dynamic (injected at run time via `RAGState`)

Runtime data — retrieved chunks, user context, low-confidence flag — is injected through `RAGState.deps`, not through the system prompt:

```python
@agent.tool
async def search_knowledge_base(ctx: RunContext[RAGState], query: str) -> str:
    retriever = await ctx.deps.get_retriever()
    results = await retriever.retrieve(query, ctx.deps.corpus_ids)
    return format_chunks_for_llm(results)   # chunk_ids included in output
```

The agent calls its tools before generating the answer. Tool results — including chunk IDs — become part of the conversation context the model reasons over. This keeps the system prompt clean and reusable across all queries.

**Rule**: Never put query-specific, corpus-specific, or user-specific data in the static system prompt. Use tools or `@agent.instructions` for that.

**Low-confidence notice injection**:
```python
# Appended to system_prompt at pipeline assembly time — not baked into the static prompt
if aggregate_confidence < confidence_warn_threshold:
    system_prompt += LOW_CONFIDENCE_NOTICE
```

---

## 10. Prompt Design Principles

| Principle | Why |
|-----------|-----|
| **Imperative verbs** | "Extract", "Classify", "Validate" — not "You might want to..." |
| **`Literal` types for constrained fields** | JSON Schema enum constraint makes hallucinated values impossible; model retries automatically |
| **CAPS for hard constraints** | `verdict MUST be exactly...` gets the model's attention more reliably than lowercase |
| **`| None` for optional Literal** | Better to omit than force a wrong value — model chooses `null` when uncertain |
| **Few-shot for complex extraction** | Content policy and graph ontology fields need examples; abstract rules alone fail on local models |
| **Minimal static prompt** | Dynamic data (chunks, transcripts, user context) goes through tools or dynamic instructions |
| **Never cite specific chunk IDs in static prompts** | Chunk IDs are runtime data — they come through tool results, not from the static prompt |
| **`retries=3` on all agent calls** | Pydantic AI auto-retries with validation error; never rely on one-shot completion |
| **Token budget before call** | `trim_to_budget()` before every `agent.run()` — never let silent model truncation corrupt the output |
| **Background tasks for non-blocking work** | Fact extraction and summarization use `asyncio.create_task()` — never block the response path for memory operations |
