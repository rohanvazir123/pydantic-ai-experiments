# Prompts Reference

System prompts, tool docstrings, and structured output schemas for every LLM
call in the RAG v2 pipeline, with rationale for each design choice.

## Table of Contents

1. [Overview — All LLM Calls](#1-overview--all-llm-calls)
2. [How Context Reaches the LLM](#2-how-context-reaches-the-llm)
3. [V5 — Content Policy Classifier](#3-v5--content-policy-classifier)
4. [Model Router](#4-model-router)
5. [RAG Agent — Blocking Path](#5-rag-agent--blocking-path)
   - 5.1 [System Prompt](#51-system-prompt)
   - 5.2 [Tool: search_knowledge_base](#52-tool-search_knowledge_base)
   - 5.3 [Tool: search_knowledge_graph](#53-tool-search_knowledge_graph)
   - 5.4 [Output Schema](#54-output-schema)
6. [RAG Agent — Streaming Path](#6-rag-agent--streaming-path)
7. [LLM Judge](#7-llm-judge)
8. [Memory — Fact Extractor](#8-memory--fact-extractor)
9. [Memory — Conversation Summarizer](#9-memory--conversation-summarizer)
10. [Graph Extraction — Ontology Field Descriptions](#10-graph-extraction--ontology-field-descriptions)
11. [Prompt Design Principles](#11-prompt-design-principles)

---

## 1. Overview — All LLM Calls

Seven distinct LLM calls happen per request (some are skipped on cache hits or
abstentions). Each uses a different model tier and fires at a different point in
the pipeline.

| # | Call | Model tier | Output type | When it fires |
|---|------|-----------|-------------|---------------|
| 1 | V5 Content Policy | nano | `ContentPolicyResult` | Every request, before retrieval |
| 2 | Model Router | nano | `RoutingDecision` | Every request, after V5 passes |
| 3 | RAG Agent | small / large | `GenerationResult` | After retrieval, if Layer 1 gate passes |
| 4 | LLM Judge | nano (→ small) | `JudgeResult` | After generation, Layer 3 gate (blocking path only) |
| 5 | Fact Extractor | nano | `list[str]` | Background, after every answered turn |
| 6 | Conversation Summarizer | nano | `str` | Background, when `turn_count == 20` |
| 7 | KG field extraction | small / large | ontology Pydantic model | During ingestion, per document chunk |

**Streaming path difference:** the RAG Agent uses a simpler `output_type=str`
variant for streaming. Tool calls and citation extraction are not available in
the streaming path. The Judge (call 4) is skipped entirely.

---

## 2. How Context Reaches the LLM

### Pre-retrieval: what the LLM already has

**Pre-retrieved chunks are NOT injected through tools.** They are assembled into
the message history before `agent.run()` is called, by `working_memory.py`.

```
working_memory.assemble(
    system_prompt,          # Tier 5: static role + hard rules
    user_memory_context,    # Tier 3: top-3 Mem0 facts for this user
    conversation_history,   # Tier 2: last 8 turns (or summary + last 8)
    retrieved_chunks,       # Tier 4: top-K chunks with [chunk_id] anchors
    query,                  # current user question
) → AssembledContext

agent.run(query, message_history=assembled_context.as_message_history(), deps=state)
```

The LLM receives ALL of this before it produces a single token. For most
queries, this is everything it needs — it reads the chunks, writes an answer
with `[chunk_id]` citations, and returns a `GenerationResult`. No tools fire.

### Why tools exist at all

The initial retrieval is a single-pass hybrid search: one query embedding,
scoped to the top-K chunks by confidence. It is fast and accurate for
single-hop questions, but it has a structural limit:

> **It can only retrieve what the original query embedding is close to.**

That fails in three specific situations:

| Situation | What goes wrong | Example |
|-----------|----------------|---------|
| **Multi-hop** | The answer requires a fact that lives in a different part of the corpus from where the initial query lands | "Who approves the budget for the team Alice manages?" — retrieval finds Alice's profile but not the budget policy |
| **Cross-reference** | A retrieved chunk points elsewhere ("see Benefits Addendum §3") but the addendum itself was not in top-K | The chunk is retrieved; the addendum content is not |
| **Relationship traversal** | The question is about how entities connect, not about a passage of text | "Which teams report to the CTO?" — this is a graph edge, not a chunk |

Tools are the answer to all three. They let the LLM make a second (or third)
targeted retrieval call with a narrower, more specific query — one the LLM
derives from what it has already read.

### When the LLM actually calls a tool

The LLM calls a tool when it reaches a point in drafting the answer where it
needs to make a factual claim but has no `[chunk_id]` in the assembled context
to cite. The system prompt (§5.1) makes this explicit:

```
3. If the provided context does not contain a chunk that supports a claim:
   a. Call search_knowledge_base with a more targeted or decomposed query FIRST.
   b. For questions about entities, relationships, or connections between things,
      call search_knowledge_graph instead.
   c. Only omit the claim if additional retrieval also returns nothing relevant.
```

Without rule 3, the LLM would silently omit the claim, and the user would get
an incomplete answer with no indication that relevant information exists
elsewhere in the corpus.

**The critical instruction: use a decomposed query, not the original one.**
The original query already ran — repeating it verbatim returns exactly the same
chunks. The tool call is only useful if the LLM narrows it to the specific
sub-topic it's missing. For example:

```
Original query:  "What is the parental leave policy for contractors?"
Initial chunks:  [General PTO Policy], [Contractor Classification Guide]
                 ← neither mentions parental leave for contractors specifically

LLM calls:  search_knowledge_base("parental leave contractor entitlement")
                                   ↑ decomposed to the specific gap

Tool returns: [Benefits Addendum §4 — Contractor Parental Leave]
LLM now has a chunk_id → cites it → returns GenerationResult
```

### Tool call mechanics (Pydantic AI)

Each tool call is a full round-trip:

```
Turn 1: LLM reads assembled context → outputs tool call JSON
          { "tool": "search_knowledge_base", "args": { "query": "..." } }

Pydantic AI executes the tool (async Python function)
  → retriever.retrieve() or age_store.query()
  → returns result string to Pydantic AI

Turn 2: Pydantic AI appends tool result as a new message
  → re-invokes the LLM with original context + tool result
  → LLM decides: call another tool, or produce GenerationResult

Loop terminates when:
  • LLM outputs GenerationResult (structured output)
  • Max iterations reached (default: 5)
  • Timeout
```

Most queries: **1 turn** (no tool call).
Multi-hop or cross-reference queries: **2–3 turns**.
Complex graph traversal: **up to 4 turns**.

---

## 3. V5 — Content Policy Classifier

**File:** `knowledge/validation/pipeline.py` (inline prompt, not in `prompts.py`)

### Role

Classify the user's query as `on_topic`, `off_topic`, or `inappropriate` before
any database or LLM work is done. Runs on the nano model. Must complete in
< 50ms P95. Skipped entirely when the corpus has no `allowed_topics` configured.

### Prompt (assembled inline)

```
Corpus topics: {topics_str}

Query: {query}

Classify as:
  on_topic      — relevant to the corpus topics
  off_topic     — coherent but unrelated
  inappropriate — harmful, abusive, or policy-violating

If uncertain between on_topic and off_topic, choose on_topic.
Respond with JSON: {"verdict": "...", "confidence": 0.0, "reason": "..."}
```

### Output Schema

```python
@dataclass
class ContentPolicyResult:
    verdict:    Literal["on_topic", "off_topic", "inappropriate"]
    confidence: float       # 0.0–1.0
    reason:     str | None  # logged only — never returned to the client
```

### Design Notes

- **`reason` is never returned to the client** — it leaks policy internals. Only
  `verdict` drives routing; `reason` goes to structlog.
- **`Literal` prevents hallucination** — the model cannot output `"unclear"`.
- **False-positive bias**: if `confidence < 0.5` on `off_topic`, treat as
  `on_topic`. Blocking a valid query is worse than passing a borderline one.
- **Direct LLM call, not a Pydantic AI agent** — `json.loads()` + manual
  `ContentPolicyResult` construction. No retry on malformed JSON (fast path).

---

## 4. Model Router

**File:** `knowledge/agent/prompts.py` → `ROUTER_SYSTEM_PROMPT`

### Role

Decide which model tier to use for the current query. Runs on the nano model.
On timeout (> 3s), defaults to `small`.

### System Prompt

```
You are a query complexity classifier for a RAG system.

Classify the query to select the appropriate LLM tier:
  simple   — factual, single-entity, single-hop
  moderate — multi-part, requires synthesis across sources
  complex  — multi-hop, reasoning chains, graph traversal required

requires_graph: true if the query asks about relationships or entity connections.
requires_multipass: true if the query spans multiple sub-questions.
estimated_context_tokens: rough token estimate (simple=500, moderate=1500, complex=3000+).
rejected: true only for structurally malformed queries.
```

### User Prompt

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
    rejected:                 bool        # True for structurally malformed queries only
    rejection_reason:         str | None
```

### Design Notes

- **`requires_graph: true`** causes the parallel `graph_retrieval()` leg to run
  during hybrid search. If `False`, the AGE query is skipped entirely.
- **`rejected: true` is not the same as V5 `inappropriate`** — use it only for
  queries that are structurally unanswerable (empty after sanitization, binary
  garbage). Content policy belongs in V5.
- **Tier override**: clients with the `tier_override` JWT role may pass
  `model_tier` in `ChatRequest` to bypass routing.

---

## 5. RAG Agent — Blocking Path

**File:** `knowledge/agent/prompts.py` → `MAIN_SYSTEM_PROMPT`  
**Implementation:** `knowledge/agent/agent.py` → `_build_agent()`

The primary Pydantic AI agent. Generates the grounded answer and citations.
Uses `small` or `large` tier per the routing decision.

### 5.1 System Prompt

```
You are a precise, citation-grounded knowledge assistant with access to search tools.

RULES — follow exactly:
1. Answer using ONLY chunks from the knowledge base — either the context already
   provided or results you retrieve via tools. Do not use prior knowledge.
2. Every factual claim MUST be cited inline as [chunk_id].
   Example: "The PTO policy allows 15 days per year [abc123]."
3. If the provided context does not contain a chunk that supports a claim:
   a. Call search_knowledge_base with a more targeted or decomposed query FIRST.
   b. For questions about entities, relationships, or connections between things,
      call search_knowledge_graph instead.
   c. Only omit the claim if additional retrieval also returns nothing relevant.
4. Be concise. Answer the question directly. Do not repeat the question.
5. citation_check.is_trustworthy = False if ANY claim lacks a [chunk_id].
```

**Low-confidence variant** — appended to the above when the aggregate retrieval
confidence falls below the warning threshold:

```
NOTE: The retrieved context has low confidence scores. State any uncertainty
explicitly in your answer. Prefer "Based on available information..." over
definitive statements.
```

### 5.2 Tool: `search_knowledge_base`

```python
@agent.tool
async def search_knowledge_base(
    ctx: RunContext[RAGState],
    query: str,
    match_count: int | None = 5,
    search_type: str | None = "hybrid",
) -> str:
    """Search the knowledge base for additional chunks.

    Call this when the provided context is missing information needed to
    support a claim. Use a more targeted or decomposed query than the
    original question — search for a specific policy name, section title,
    or sub-topic rather than repeating the full query verbatim.
    Returns formatted context with [chunk_id] anchors for citation.
    """
```

**Return format:**
```
[chunk_id: abc123] Document Title (source/path)
Chunk text, up to 500 chars...

[chunk_id: def456] Another Document (source/path)
More text...
```

**When to use:** the LLM has a claim to make but no `[chunk_id]` in the
pre-retrieved chunks supports it. The tool runs a full hybrid search
(pgvector + tsvector + RRF) scoped to the current `corpus_ids` and `tenant_id`.

**Key instruction:** use a decomposed sub-query, not the full original query.
The original query already ran — repeating it returns the same chunks.

### 5.3 Tool: `search_knowledge_graph`

```python
@agent.tool
async def search_knowledge_graph(
    ctx: RunContext[RAGState],
    query: str,
    entity_type: str | None = None,
    limit: int | None = 15,
) -> str:
    """Search the knowledge graph for entities and relationships.

    Call this when the question involves connections between things — who
    works where, what applies to whom, how entities relate. Prefer this
    over search_knowledge_base when the answer requires traversing a
    relationship rather than finding a passage of text.
    """
```

**Return format:**
```
## Knowledge Graph — Entities
- [Person] Alice Smith — works at Engineering
- [Team] Engineering — reports to CTO
```

**When to use:** questions about who/what/where relationships between entities,
not questions about the content of a document. The tool queries `kg_entity_index`
(HNSW + GIN in PostgreSQL) — not the vector chunks table.

### 5.4 Output Schema

```python
class CitationCheck(BaseModel):
    is_trustworthy: bool
    uncited_claims: list[str]   # claims the model couldn't attribute

class GenerationResult(BaseModel):
    answer:         str
    citations:      list[Citation]
    citation_check: CitationCheck

class Citation(BaseModel):
    chunk_id:        UUID
    document_title:  str
    document_source: str
    relevance_score: float   # post-rerank sigmoid confidence
    excerpt:         str     # ≤ 200 chars of the supporting chunk
```

### Design Notes

- **`citation_check` drives Layer 2 gate** — if `is_trustworthy = False`, the
  pipeline returns `abstained_citation` without showing the answer to the user.
- **`[chunk_id]` comes from tool results or the pre-assembled context** — chunk
  IDs are runtime data. The model cannot fabricate them because it only sees IDs
  that appear in the context it was given.
- **`retries=3` on every `agent.run()`** — Pydantic AI auto-retries with the
  validation error if the model omits required fields or outputs invalid JSON.

---

## 6. RAG Agent — Streaming Path

**File:** `knowledge/agent/prompts.py` → `STREAM_SYSTEM_PROMPT`  
**Implementation:** `knowledge/agent/agent.py` → `_build_stream_agent()`

A separate agent with `output_type=str` instead of `GenerationResult`.
This allows Pydantic AI's `run_stream()` to yield tokens as they are generated.

### System Prompt

```
You are a knowledge assistant for a company's internal knowledge base.

RULES:
1. ONLY answer questions about topics covered in the provided source passages.
2. If the question is personal, off-topic, or not answerable from the sources, respond:
   "I can only answer questions about the knowledge base. Please ask about
   company policies, teams, documents, or business topics."
3. Answer using ONLY the provided source passages. Do not use prior knowledge.
4. Cite every source document you draw from, inline, using its title in brackets,
   e.g. [Team Handbook].
5. ALWAYS write a comprehensive, multi-paragraph answer. A single sentence or a
   single fact is NEVER an acceptable answer. Cover every relevant aspect found
   across ALL source passages.
6. Synthesise across sources: combine information into one unified answer rather
   than listing each source separately.
7. Use bullet points or numbered lists whenever the answer contains multiple
   distinct items or steps.
8. Do not repeat the question. Start the answer directly with substance.
```

### Trade-offs vs the blocking path

| | Blocking (`/chat`) | Streaming (`/chat/stream`) |
|---|---|---|
| Output type | `GenerationResult` (structured) | `str` (plain text) |
| Citations | Structured `Citation` objects with `chunk_id` | Document title in `[brackets]` only |
| Tool calls | Yes — `search_knowledge_base`, `search_knowledge_graph` | No |
| Layer 2 gate | Yes (citation check) | No |
| Layer 3 gate | Yes (judge) | No |
| Latency | Higher (structured output + judge) | Lower (first token faster) |

Use the streaming path for interactive chat where token-by-token rendering
matters. Use the blocking path where citation fidelity and judge verification
are required.

---

## 7. LLM Judge

**File:** `knowledge/agent/prompts.py` → `JUDGE_SYSTEM_PROMPT`  
**Implementation:** `knowledge/agent/judge.py`

### Role

An independent LLM-as-judge that verifies the generated answer against the
retrieved passages. Runs **after** generation, as Layer 3 gate. Does not see
citation metadata (`[chunk_id]` markers are stripped before sending) — it
evaluates content grounding, not citation formatting.

### System Prompt

```
You are an impartial evaluator.

Given a question, source passages, and a generated answer, determine:
  supported   — fully grounded in the passages; all claims traceable to sources
  partial     — mostly grounded but missing or hedging on some aspects
  unsupported — contains claims not found in or contradicted by the passages

RULES:
- Base your verdict ONLY on the provided passages. Do not use prior knowledge.
- confidence must reflect your certainty in the verdict (0.0-1.0).
- reasoning must be one sentence explaining the key reason.
- If the answer is a refusal or abstention, verdict = 'supported'.
```

### User Prompt

```
QUESTION:
{query}

SOURCE PASSAGES:
{formatted_chunks}   ← top-K retrieved chunks, [chunk_id] markers stripped

GENERATED ANSWER:
{answer}
```

### Output Schema

```python
class JudgeResult(BaseModel):
    verdict:    Literal["supported", "partial", "unsupported"]
    confidence: float   # judge's certainty in its own verdict
    reasoning:  str     # one sentence; logged, never returned to user
```

### Design Notes

- **Verdict → pipeline action:**
  - `supported` → status = `"answered"`
  - `partial` → status = `"answered"` + uncertainty note appended to answer
  - `unsupported` → status = `"abstained_judge"`, answer withheld
- **Escalation:** if nano judge returns `confidence < 0.5`, re-run with the
  `small` model (one retry). A confident wrong verdict is worse than a slow
  correct one.
- **Timeout = pessimistic abstention:** if the judge call exceeds its deadline,
  treat as `abstained_judge` rather than retrying. Better to abstain than to
  pass a potentially unsupported answer.
- **Chunk IDs are stripped** before sending to the judge — prevents the judge
  from being influenced by citation formatting rather than content grounding.

---

## 8. Memory — Fact Extractor

**File:** `knowledge/agent/prompts.py` → `FACT_EXTRACTOR_PROMPT`

### Role

Extract durable facts about the user from a single Q&A turn. Fires as a
non-blocking background task after every answered turn (never on abstentions).

### System Prompt

```
From the Q&A pair below, extract facts about the USER specifically.

Focus on: role, title, company, ongoing projects, stated preferences, domain expertise,
corrections the user made to the system.

RULES:
- Each fact must be a complete, standalone sentence.
- Do not extract facts about the subject matter — only USER facts.
- Never store query content or answer summaries.
- If no memorable user facts, return an empty list.
```

### User Prompt

```
Q: {query}
A: {answer}
```

### Output Schema

```python
facts: list[str]   # each item is a complete sentence about the user
                   # empty list is the correct output for most turns
```

### Design Notes

- **Background only** — wrapped in `asyncio.create_task()`, never in the
  response path. A slow or failed extraction has zero impact on latency.
- **Empty list is the right answer for most turns** — most queries reveal nothing
  memorable about the user. Over-extraction pollutes the memory store with noise.
- **Deduplication is Mem0's job** — the prompt instructs the model not to
  duplicate existing memories, but Mem0's contradiction resolution handles edge
  cases.

---

## 9. Memory — Conversation Summarizer

**File:** `knowledge/agent/prompts.py` → `SUMMARIZER_PROMPT`

### Role

Compress old turns into a 3–5 sentence summary when `turn_count > 20`. Stored
in `conversations.summary`. Prepended as the first message in the active window
for long threads so the token budget is never exceeded.

### System Prompt

```
Summarize the conversation below in 3-5 sentences.

Cover:
- What the user was trying to learn or accomplish
- Key facts that were established or agreed upon
- Any decisions, conclusions, or open questions

RULES:
- Do not quote specific messages.
- Write in third person ("The user asked about...").
- Be factual — do not infer intent beyond what was stated.
```

### User Prompt

```
{formatted_turns}   ← turns 1 to (N - 8), formatted as "role: content"
```

### Output Schema

```python
summary: str   # 3–5 sentences in plain prose
```

### Design Notes

- **Only turns 1 to (N-8) are summarized** — the most recent 8 turns are always
  kept verbatim. The summary replaces the older turns only.
- **Re-summarization triggers** at turn 40, 60, etc. — each time the unsummarized
  tail grows beyond 20 turns, the summarizer re-runs covering everything up to
  `(current - 8)`.
- **Nano model is sufficient** — slightly imprecise summaries are acceptable.
  Accuracy requirements here are much lower than for answer generation.

---

## 10. Graph Extraction — Ontology Field Descriptions

**File:** corpus ontology `.py` files (user-defined)  
**Mechanism:** docling-graph uses Pydantic `Field(description=...)` strings as
the per-field extraction prompt. There is no separate system prompt file — the
ontology schema IS the prompt.

### Field Description Pattern

Every extractable field should follow the `LOOK FOR / EXTRACT / EXAMPLES` pattern:

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

**Why:** Abstract rules ("extract names") fail on small models. Concrete location
hints and examples raise extraction accuracy significantly.

### Extraction Contracts

docling-graph supports three strategies, each implying a different prompt structure:

| Contract | How the LLM is prompted | When to use |
|----------|------------------------|-------------|
| `"direct"` | Single prompt: extract all fields from this chunk | Large models (≥ 70B) with simple schemas |
| `"staged"` | Pass 1: what entity IDs appear? Pass 2: fill in their properties | Small models (≤ 8B); **recommended default** |
| `"delta"` | Per-chunk: what's new vs. the merged graph so far? | Long documents with many entities |

**Default:** `"staged"` with `llama3.2:3b` via Ollama. The staged contract splits
one complex extraction into two simpler calls that smaller models handle reliably.

### Hard Constraint (appended by docling-graph automatically)

```
IMPORTANT: Do NOT use placeholder values like 'N/A', 'unknown', or 'none'.
If the information is not present in the text, leave the field None.
```

Without this, small models frequently output `"N/A"` instead of `null`,
breaking downstream graph import.

---

## 11. Prompt Design Principles

| Principle | Rationale |
|-----------|-----------|
| **Imperative verbs** | "Extract", "Classify", "Validate" — not "You might want to..." |
| **`Literal` types for constrained fields** | Pydantic AI sends the JSON Schema enum to the model as a `response_format` constraint; hallucinated values are structurally impossible |
| **CAPS for hard constraints** | `MUST`, `NEVER`, `ONLY` — gets the model's attention more reliably than lowercase prose, especially on smaller models |
| **`\| None` rather than a forced default** | Better to return `null` when uncertain than to force a wrong `Literal` value |
| **Few-shot examples for extraction** | Content policy and graph ontology fields need `EXAMPLES:` — abstract rules alone fail on local 3B models |
| **Static prompt for role + rules; tools/context for data** | Never put query text, chunk content, or user data in the static system prompt; keep it reusable and cacheable |
| **Chunk IDs come from context, never from the static prompt** | Chunk IDs are runtime data; putting them in the static prompt would require a different prompt per request and allows fabrication |
| **`retries=3` on every `agent.run()`** | Pydantic AI auto-retries with the validation error message; never rely on one-shot completion for structured output |
| **`trim_to_budget()` before every agent call** | Silent model truncation corrupts structured output; always trim explicitly and set `context_truncated: True` |
| **Background tasks for memory operations** | Fact extraction and summarization use `asyncio.create_task()` — a slow or failed memory operation must never touch response latency |
